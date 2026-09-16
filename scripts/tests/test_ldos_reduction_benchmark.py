from __future__ import annotations

import importlib.util
import contextlib
import copy
import ctypes
import errno
import gc
import json
import os
import pathlib
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = SCRIPTS / "run-ldos-reduction-benchmark.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_ldos_runner_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ldos = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ldos
SPEC.loader.exec_module(ldos)
ldos.EVIDENCE = ldos._load_source_module(
    "gpmeep_ldos_test_evidence", SCRIPTS / "gpmeep_benchmark_evidence.py"
)
ldos.PROVENANCE = ldos._load_source_module(
    "gpmeep_ldos_test_provenance", SCRIPTS / "gpmeep_provenance.py"
)
ldos.PROFILE_SHA256 = ldos.PROVENANCE.canonical_sha256(dict(ldos.PROFILE))


class PythonTestFdAllocationShim:
    """Unit-test double for the receipt-bound caller-owned native ABI."""

    @staticmethod
    def _single_begin(result) -> None:
        result.abi_version = ldos.FD_ALLOCATION_SHIM_ABI
        result.completed = 0
        result.fd = -1
        result.error_number = 0

    @staticmethod
    def _single_finish(result, descriptor: int, error_number: int) -> None:
        result.fd = descriptor
        result.error_number = error_number
        result.completed = 1

    def duplicate_into(self, source_fd, minimum_fd, result) -> None:
        self._single_begin(result)
        try:
            descriptor = ldos.fcntl.fcntl(
                source_fd, ldos.fcntl.F_DUPFD_CLOEXEC, minimum_fd
            )
        except OSError as error:
            self._single_finish(result, -1, error.errno)
        else:
            self._single_finish(result, descriptor, 0)

    def pidfd_open_into(self, pid, flags, result) -> None:
        self._single_begin(result)
        try:
            function = getattr(ldos.os, "pidfd_open", None)
            if callable(function):
                descriptor = function(pid, flags)
            else:
                libc = ctypes.CDLL(None, use_errno=True)
                libc.syscall.restype = ctypes.c_long
                descriptor = int(libc.syscall(ldos.SYS_PIDFD_OPEN, pid, flags))
                if descriptor < 0:
                    number = ctypes.get_errno()
                    raise OSError(number, os.strerror(number))
        except OSError as error:
            self._single_finish(result, -1, error.errno)
        else:
            self._single_finish(result, descriptor, 0)

    def pipe2_into(self, flags, result) -> None:
        result.abi_version = ldos.FD_ALLOCATION_SHIM_ABI
        result.completed = 0
        result.read_fd = -1
        result.write_fd = -1
        result.error_number = 0
        try:
            read_descriptor, write_descriptor = ldos.os.pipe()
            result.read_fd = read_descriptor
            result.write_fd = write_descriptor
            if flags & getattr(os, "O_CLOEXEC", 0):
                os.set_inheritable(read_descriptor, False)
                os.set_inheritable(write_descriptor, False)
        except OSError as error:
            result.error_number = error.errno
        result.completed = 1


ldos._FD_ALLOCATION_SHIM = PythonTestFdAllocationShim()


SPEEDUPS = [1.3, 1.4, 1.5, 1.6, 1.7]
SAMPLES = [[float(index + 1), float(-(index + 1))] for index in range(9)]
SAMPLE_COMPLEX = [complex(real, imaginary) for real, imaginary in SAMPLES]
SAMPLE_DIGEST = ldos._sample_digest(SAMPLE_COMPLEX)
CUDA_UUIDS = (
    "aaaaaaaa_bbbb_cccc_dddd_eeeeeeeeeeee".replace("_", ""),
    "11111111_2222_3333_4444_555555555555".replace("_", ""),
)


def compact(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=False)


def system_tool_elf_bytes(
    *, elf_type: int = 2, version: int = 1, header_size: int | None = None,
    program_offset: int | None = None, program_count: int = 1,
    executable_load: bool = True, segment_file_size: int | None = None,
    entry_point: int | None = None,
) -> bytes:
    """Create a minimal native ELF executable fixture with one PT_LOAD."""

    elf_class = 2 if struct.calcsize("P") == 8 else 1
    encoding = 1 if sys.byteorder == "little" else 2
    endian = "<" if encoding == 1 else ">"
    machine = {"x86_64": 62, "aarch64": 183}[ldos._normalized_machine()]
    if elf_class == 2:
        expected_header_size = 64
        program_entry_size = 56
        header_format = endian + "HHIQQQIHHHHHH"
        program_format = endian + "IIQQQQQQ"
    else:
        expected_header_size = 52
        program_entry_size = 32
        header_format = endian + "HHIIIIIHHHHHH"
        program_format = endian + "IIIIIIII"
    header_size = expected_header_size if header_size is None else header_size
    program_offset = expected_header_size if program_offset is None else program_offset
    entry_point = expected_header_size if entry_point is None else entry_point
    size = expected_header_size + program_entry_size + 32
    data = bytearray(size)
    data[:16] = (
        b"\x7fELF" + bytes((elf_class, encoding, 1, 0, 0)) + b"\0" * 7
    )
    struct.pack_into(
        header_format, data, 16,
        elf_type, machine, version, entry_point,
        program_offset, 0, 0, header_size, program_entry_size,
        program_count, 0, 0, 0,
    )
    if 0 <= program_offset <= size - program_entry_size:
        file_size = size if segment_file_size is None else segment_file_size
        if elf_class == 2:
            values = (1, 5 if executable_load else 4, 0, 0, 0, file_size, file_size, 4096)
        else:
            values = (1, 0, 0, 0, file_size, file_size, 5 if executable_load else 4, 4096)
        struct.pack_into(program_format, data, program_offset, *values)
    return bytes(data)


def timed_updates(base: int, cpu: int, duration_ns: int) -> list[dict]:
    return [
        {
            "start_monotonic_ns": base + index * (duration_ns + 10) + 1,
            "stop_monotonic_ns": base + index * (duration_ns + 10) + 1 + duration_ns,
            "cpu_before": cpu,
            "cpu_after": cpu,
            "affinity_before": [cpu],
            "affinity_after": [cpu],
            "elapsed_ns": duration_ns,
            "voluntary_context_switches": 0,
            "involuntary_context_switches": 0,
            "minor_faults": 0,
            "major_faults": 0,
        }
        for index in range(16)
    ]


def benchmark_line(ranks: int = 1) -> str:
    return (
        f"ldos-transfer-benchmark: ranks={ranks} pixels=64 updates=16 "
        "repetitions=5 minimum-speedup=1.3 median-speedup=1.5 "
        "paired-speedups=1.3,1.4,1.5,1.6,1.7"
    )


def valid_stdout(ranks: int = 1) -> str:
    records: list[str] = []
    for repetition, speedup in enumerate(SPEEDUPS):
        repetition_base = repetition * 100_000_000_000
        if repetition % 2 == 0:
            host_base = repetition_base
            resident_base = repetition_base + 2_000_000_000
        else:
            resident_base = repetition_base
            host_base = repetition_base + 2_000_000_000
        for rank in range(ranks):
            records.append(
                ldos.RANK_PREFIX
                + compact(
                    {
                        "rank": rank,
                        "repetition": repetition,
                        "device_ordinal": rank,
                        "device_uuid": CUDA_UUIDS[rank],
                        "host_d2h": 4194304 // ranks,
                        "resident_d2h": 512,
                        "host_cpu_reduction_calls": 16,
                        "host_cuda_reduction_calls": 0,
                        "resident_cpu_reduction_calls": 0,
                        "resident_cuda_reduction_calls": 16,
                        "resident_cuda_kernel_launches": 32,
                        "resident_result_d2h": 512,
                        "cpu_affinity": [rank + 1],
                        "host_voluntary_context_switches": 0,
                        "host_involuntary_context_switches": 0,
                        "resident_voluntary_context_switches": 0,
                        "resident_involuntary_context_switches": 0,
                        "host_major_faults": 0,
                        "resident_major_faults": 0,
                        "host_minor_faults": 0,
                        "resident_minor_faults": 0,
                        "host_timed_updates": timed_updates(
                            host_base,
                            rank + 1,
                            int(speedup * 1_000_000_000) // 16,
                        ),
                        "resident_timed_updates": timed_updates(
                            resident_base,
                            rank + 1,
                            1_000_000_000 // 16,
                        ),
                    }
                )
            )
        records.append(
            ldos.PAIR_PREFIX
            + compact(
                {
                    "repetition": repetition,
                    "order": "host-resident" if repetition % 2 == 0 else "resident-host",
                    "host_seconds": speedup,
                    "resident_seconds": 1,
                    "speedup": speedup,
                    "max_absolute_error": 0,
                    "max_relative_error": 0,
                    "host_digest": SAMPLE_DIGEST,
                    "resident_digest": SAMPLE_DIGEST,
                    "host_samples": SAMPLES,
                    "resident_samples": SAMPLES,
                }
            )
        )
    for rank in range(ranks):
        libmeep = f"/evidence/libmeep-rank-{rank}.so"
        records.append(
            ldos.RUNTIME_PREFIX
            + compact(
                {
                    "rank": rank,
                    "device_ordinal": rank,
                    "device_uuid": CUDA_UUIDS[rank],
                    "libmeep": libmeep,
                    "mappings": [
                        {"path": libmeep, "device": 1, "inode": rank + 10, "size": 1, "mtime_ns": 1, "ctime_ns": 1},
                        {"path": f"/lib/loader-rank-{rank}", "device": 1, "inode": rank + 20, "size": 1, "mtime_ns": 1, "ctime_ns": 1},
                    ],
                    "special_mappings": [],
                }
            )
        )
    return "\n".join(
        (
            "Using MPI version 3.1",
            "PASS: unrelated bounded topology regression",
            *records,
            benchmark_line(ranks),
            ldos.PASS_LINE,
            "",
        )
    )


def process_result(ranks: int) -> dict:
    return {
        "command": ["mpiexec", "-n", str(ranks), "gpu-step-db"],
        "environment": {"HOME": "/isolated", "MEEP_GPU_BACKEND": "cuda"},
        "cwd": "/isolated",
        "started_at_utc": "2026-08-11T00:00:00Z",
        "finished_at_utc": "2026-08-11T00:00:01Z",
        "process_wall_seconds": 1.0,
        "process_started_monotonic_ns": 1,
        "process_finished_monotonic_ns": 1_000_000_000_000,
        "root_pid": 12345,
        "returncode": 0,
        "timed_out": False,
        "timeout_seconds": 600,
        "stdout": valid_stdout(ranks),
        "stderr": "",
    }


def rejected_stdout(ranks: int = 1, *, minor_faults: int = 0) -> str:
    lines = valid_stdout(ranks).splitlines()
    pair = json.loads(
        next(
            line for line in lines
            if line.startswith(ldos.PAIR_PREFIX)
            and json.loads(line.removeprefix(ldos.PAIR_PREFIX))["repetition"] == 0
        ).removeprefix(ldos.PAIR_PREFIX)
    )
    rejected_lines = []
    for rank in range(ranks):
        accepted = json.loads(
            next(
                line for line in lines
                if line.startswith(ldos.RANK_PREFIX)
                and json.loads(line.removeprefix(ldos.RANK_PREFIX))["rank"] == rank
                and json.loads(line.removeprefix(ldos.RANK_PREFIX))["repetition"] == 0
            ).removeprefix(ldos.RANK_PREFIX)
        )
        accepted["host_timed_updates"][0]["minor_faults"] = minor_faults
        accepted["host_minor_faults"] = minor_faults
        if rank == 0:
            accepted["host_timed_updates"][0]["involuntary_context_switches"] = 1
            accepted["host_involuntary_context_switches"] = 1
        record = {
            "schema_version": 1,
            "rank": rank,
            "repetition": 0,
            "order": "host-resident",
            "global_gate_accepted": False,
            "device_ordinal": accepted["device_ordinal"],
            "device_uuid": accepted["device_uuid"],
            "host_seconds": pair["host_seconds"],
            "resident_seconds": pair["resident_seconds"],
            "host_d2h": accepted["host_d2h"],
            "resident_d2h": accepted["resident_d2h"],
            "host_cpu_reduction_calls": accepted["host_cpu_reduction_calls"],
            "host_cuda_reduction_calls": accepted["host_cuda_reduction_calls"],
            "resident_cpu_reduction_calls": accepted["resident_cpu_reduction_calls"],
            "resident_cuda_reduction_calls": accepted["resident_cuda_reduction_calls"],
            "resident_cuda_kernel_launches": accepted["resident_cuda_kernel_launches"],
            "resident_result_d2h": accepted["resident_result_d2h"],
            "max_absolute_error": pair["max_absolute_error"],
            "max_relative_error": pair["max_relative_error"],
            "host_digest": pair["host_digest"],
            "resident_digest": pair["resident_digest"],
            "host_samples": pair["host_samples"],
            "resident_samples": pair["resident_samples"],
            "cpu_affinity": accepted["cpu_affinity"],
            "host_voluntary_context_switches": accepted["host_voluntary_context_switches"],
            "host_involuntary_context_switches": accepted["host_involuntary_context_switches"],
            "resident_voluntary_context_switches": accepted["resident_voluntary_context_switches"],
            "resident_involuntary_context_switches": accepted["resident_involuntary_context_switches"],
            "host_major_faults": accepted["host_major_faults"],
            "resident_major_faults": accepted["resident_major_faults"],
            "host_minor_faults": accepted["host_minor_faults"],
            "resident_minor_faults": accepted["resident_minor_faults"],
            "reason_predicates": {
                "host_involuntary_context_switches_nonzero": rank == 0,
                "resident_involuntary_context_switches_nonzero": False,
                "host_major_faults_nonzero": False,
                "resident_major_faults_nonzero": False,
            },
            "host_timed_updates": accepted["host_timed_updates"],
            "resident_timed_updates": accepted["resident_timed_updates"],
        }
        rejected_lines.append(ldos.REJECTED_PREFIX + compact(record))
    return "\n".join((*rejected_lines, "FAIL: focused LDOS timing segment gate", ""))


def rejected_stdout_at_repetition_one(ranks: int = 1) -> str:
    accepted = [
        line for line in valid_stdout(ranks).splitlines()
        if (
            line.startswith(ldos.RANK_PREFIX)
            and json.loads(line.removeprefix(ldos.RANK_PREFIX))["repetition"] == 0
        )
        or (
            line.startswith(ldos.PAIR_PREFIX)
            and json.loads(line.removeprefix(ldos.PAIR_PREFIX))["repetition"] == 0
        )
    ]
    accepted_stop = max(
        update["stop_monotonic_ns"]
        for line in accepted
        if line.startswith(ldos.RANK_PREFIX)
        for segment in ("host", "resident")
        for update in json.loads(
            line.removeprefix(ldos.RANK_PREFIX)
        )[f"{segment}_timed_updates"]
    )
    rejected = []
    for line in rejected_stdout(ranks).splitlines():
        if not line.startswith(ldos.REJECTED_PREFIX):
            continue
        record = json.loads(line.removeprefix(ldos.REJECTED_PREFIX))
        record["repetition"] = 1
        record["order"] = "resident-host"
        for segment, base in (
            ("resident", accepted_stop + 1_000),
            ("host", accepted_stop + 10_000_000_000),
        ):
            cursor = base
            for update in record[f"{segment}_timed_updates"]:
                duration = update["elapsed_ns"]
                update["start_monotonic_ns"] = cursor + 1
                update["stop_monotonic_ns"] = cursor + 1 + duration
                cursor = update["stop_monotonic_ns"] + 10
        rejected.append(ldos.REJECTED_PREFIX + compact(record))
    return "\n".join(
        (*accepted, *rejected, "FAIL: focused LDOS timing segment gate", "")
    )


def device_csv() -> str:
    return (
        "0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, RTX A, "
        "00000000:01:00.0, 8.6\n"
        "1, GPU-11111111-2222-3333-4444-555555555555, RTX B, "
        "00000000:02:00.0, 8.6\n"
    )


class BenchmarkParserTests(unittest.TestCase):
    def parse(self, stdout: str | None = None, stderr: str = "", code: int = 0):
        return ldos.parse_benchmark_output(
            valid_stdout() if stdout is None else stdout,
            stderr,
            returncode=code,
            expected_ranks=1,
        )

    def test_valid_record(self) -> None:
        result = self.parse()
        self.assertEqual(result["per_rank_device_to_host_bytes"][0]["resident"], 512)
        self.assertEqual(result["paired_speedups"], SPEEDUPS)

    def test_valid_two_rank_record(self) -> None:
        result = ldos.parse_benchmark_output(
            valid_stdout(2), "", returncode=0, expected_ranks=2
        )
        self.assertEqual(result["ranks"], 2)

    def test_structured_runtime_paths_do_not_trigger_failure_word_scan(self) -> None:
        for component in ("failed-runs", "failure", "skip", "skipped"):
            with self.subTest(component=component):
                stdout = valid_stdout().replace(
                    "/evidence/", f"/scratch/{component}/"
                )
                self.assertEqual(self.parse(stdout)["ranks"], 1)
                result = process_result(1)
                result["stdout"] = stdout
                primary, evidence = ldos._classify_lane_process_failure(
                    result, ranks=1
                )
                self.assertIsNone(primary)
                self.assertIsNone(evidence)
        for line in ("worker FAILED", "worker requested skip"):
            with self.subTest(line=line), self.assertRaisesRegex(
                ldos.EvidenceError, "forbidden"
            ):
                self.parse(line + "\n" + valid_stdout())

    def test_malformed_structured_failure_path_cannot_become_success(self) -> None:
        syntactically_broken = (
            ldos.RUNTIME_PREFIX + '{"path":"/scratch/failed-runs"'
        )
        result = process_result(1)
        result["stdout"] = syntactically_broken + "\n" + result["stdout"]
        primary, _evidence = ldos._classify_lane_process_failure(result, ranks=1)
        self.assertIsNotNone(primary)

        wrong_schema = ldos.RUNTIME_PREFIX + compact(
            {"path": "/scratch/failed-runs"}
        )
        result = process_result(1)
        result["stdout"] = wrong_schema + "\n" + result["stdout"]
        primary, _evidence = ldos._classify_lane_process_failure(result, ranks=1)
        self.assertIsNone(primary)
        with self.assertRaises(ldos.EvidenceError):
            ldos.parse_benchmark_output(
                result["stdout"], result["stderr"], returncode=0,
                expected_ranks=1,
            )

    def test_nonzero_exit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ldos.EvidenceError, "status 9"):
            self.parse(code=9)

    def test_rejected_repetition_is_full_rank_ordered_evidence(self) -> None:
        for ranks in (1, 2):
            with self.subTest(ranks=ranks):
                parsed = ldos._parse_rejected_repetition_evidence(
                    rejected_stdout(ranks, minor_faults=7), "",
                    expected_ranks=ranks,
                )
                self.assertIsNotNone(parsed)
                self.assertEqual(
                    [record["rank"] for record in parsed["records"]],
                    list(range(ranks)),
                )
                self.assertEqual(parsed["records"][0]["host_minor_faults"], 7)

    def test_rejected_repetition_one_requires_exact_accepted_prefix(self) -> None:
        stdout = rejected_stdout_at_repetition_one(2)
        parsed = ldos._parse_rejected_repetition_evidence(
            stdout, "", expected_ranks=2
        )
        self.assertEqual(parsed["repetition"], 1)
        self.assertEqual(parsed["order"], "resident-host")

        lines = stdout.splitlines()
        with self.assertRaisesRegex(ldos.EvidenceError, "accepted repetition prefix"):
            ldos._parse_rejected_repetition_evidence(
                "\n".join(lines[1:]), "", expected_ranks=2
            )

        rank_indexes = [
            index for index, line in enumerate(lines)
            if line.startswith(ldos.RANK_PREFIX)
        ]
        reordered = list(lines)
        reordered[rank_indexes[0]], reordered[rank_indexes[1]] = (
            reordered[rank_indexes[1]], reordered[rank_indexes[0]]
        )
        with self.assertRaisesRegex(ldos.EvidenceError, "coordinate/order"):
            ldos._parse_rejected_repetition_evidence(
                "\n".join(reordered), "", expected_ranks=2
            )

        boolean_coordinate = list(lines)
        record = json.loads(
            boolean_coordinate[rank_indexes[0]].removeprefix(ldos.RANK_PREFIX)
        )
        record["repetition"] = False
        boolean_coordinate[rank_indexes[0]] = ldos.RANK_PREFIX + compact(record)
        with self.assertRaisesRegex(ldos.EvidenceError, "coordinate/order"):
            ldos._parse_rejected_repetition_evidence(
                "\n".join(boolean_coordinate), "", expected_ranks=2
            )

    def test_rejected_accepted_prefix_reuses_full_rank_pair_validation(self) -> None:
        base_lines = rejected_stdout_at_repetition_one().splitlines()
        rank_index = next(
            index for index, line in enumerate(base_lines)
            if line.startswith(ldos.RANK_PREFIX)
        )
        pair_index = next(
            index for index, line in enumerate(base_lines)
            if line.startswith(ldos.PAIR_PREFIX)
        )
        for marker_index, prefix in (
            (rank_index, ldos.RANK_PREFIX),
            (pair_index, ldos.PAIR_PREFIX),
        ):
            original = json.loads(base_lines[marker_index].removeprefix(prefix))
            for key in original:
                with self.subTest(prefix=prefix, removed_key=key):
                    lines = list(base_lines)
                    mutated = dict(original)
                    mutated.pop(key)
                    lines[marker_index] = prefix + compact(mutated)
                    with self.assertRaises(ldos.EvidenceError):
                        ldos._parse_rejected_repetition_evidence(
                            "\n".join(lines), "", expected_ranks=1
                        )

        lines = list(base_lines)
        minimal = {"rank": 0, "repetition": 0}
        lines[rank_index] = ldos.RANK_PREFIX + compact(minimal)
        with self.assertRaisesRegex(ldos.EvidenceError, "keys|full validation"):
            ldos._parse_rejected_repetition_evidence(
                "\n".join(lines), "", expected_ranks=1
            )

    def test_rejected_accepted_prefix_binds_planned_uuid_and_affinity(self) -> None:
        for mutation, message in (("uuid", "UUID"), ("affinity", "affinity")):
            lines = rejected_stdout_at_repetition_one().splitlines()
            rank_index = next(
                index for index, line in enumerate(lines)
                if line.startswith(ldos.RANK_PREFIX)
            )
            record = json.loads(lines[rank_index].removeprefix(ldos.RANK_PREFIX))
            if mutation == "uuid":
                record["device_uuid"] = "f" * 32
            else:
                record["cpu_affinity"] = [9]
                for segment in ("host", "resident"):
                    for update in record[f"{segment}_timed_updates"]:
                        update["cpu_before"] = 9
                        update["cpu_after"] = 9
                        update["affinity_before"] = [9]
                        update["affinity_after"] = [9]
            lines[rank_index] = ldos.RANK_PREFIX + compact(record)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ldos.EvidenceError, message
            ):
                ldos._parse_rejected_repetition_evidence(
                    "\n".join(lines),
                    "",
                    expected_ranks=1,
                    expected_device_uuids=[
                        "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                    ],
                    expected_worker_cpu_masks=[{1}],
                )

    def test_rejected_stream_rejects_focused_success_but_allows_unrelated_pass(self) -> None:
        realistic = (
            "PASS: unrelated bounded topology regression\n"
            "COMPLETELY unrelated diagnostic\n"
            "/scratch/failed-runs/skip is an ordinary diagnostic path\n"
            + rejected_stdout()
        )
        self.assertIsNotNone(
            ldos._parse_rejected_repetition_evidence(
                realistic, "diagnostic warning\n", expected_ranks=1
            )
        )
        forbidden = (
            ldos.PASS_LINE,
            benchmark_line(),
            ldos.RUNTIME_PREFIX + "{}",
            "COMPLETE",
            "prefix COMPLETE authority",
            "[rank0] COMPLETE authority",
            "gpmeep-unknown-success-v1:{}",
        )
        for line in forbidden:
            for stream in ("stdout", "stderr"):
                with self.subTest(line=line, stream=stream), self.assertRaises(
                    ldos.EvidenceError
                ):
                    stdout = rejected_stdout()
                    stderr = ""
                    if stream == "stdout":
                        stdout = line + "\n" + stdout
                    else:
                        stderr = line + "\n"
                    ldos._parse_rejected_repetition_evidence(
                        stdout, stderr, expected_ranks=1
                    )

    def test_rejected_cross_segment_order_and_later_success_are_rejected(self) -> None:
        lines = rejected_stdout_at_repetition_one().splitlines()
        rejected_index = next(
            index for index, line in enumerate(lines)
            if line.startswith(ldos.REJECTED_PREFIX)
        )
        record = json.loads(lines[rejected_index].removeprefix(ldos.REJECTED_PREFIX))
        record["host_timed_updates"][0]["start_monotonic_ns"] = 5_000
        record["host_timed_updates"][0]["stop_monotonic_ns"] = (
            5_000 + record["host_timed_updates"][0]["elapsed_ns"]
        )
        lines[rejected_index] = ldos.REJECTED_PREFIX + compact(record)
        with self.assertRaisesRegex(ldos.EvidenceError, "scheduled order"):
            ldos._parse_rejected_repetition_evidence(
                "\n".join(lines), "", expected_ranks=1
            )

        later_success = rejected_stdout() + valid_stdout().splitlines()[2] + "\n"
        with self.assertRaisesRegex(ldos.EvidenceError, "success evidence"):
            ldos._parse_rejected_repetition_evidence(
                later_success, "", expected_ranks=1
            )

    def test_rank2_collective_abba_overlap_rejects_success_and_rejected(self) -> None:
        def shift_rank_one(lines, prefix):
            for index, line in enumerate(lines):
                if not line.startswith(prefix):
                    continue
                record = json.loads(line.removeprefix(prefix))
                if record.get("rank") != 1 or record.get("repetition") != 0:
                    continue
                for segment in ("host", "resident"):
                    for update in record[f"{segment}_timed_updates"]:
                        update["start_monotonic_ns"] += 1_000_000_000
                        update["stop_monotonic_ns"] += 1_000_000_000
                lines[index] = prefix + compact(record)
                return
            raise AssertionError("rank-1 repetition-0 fixture not found")

        success_lines = valid_stdout(2).splitlines()
        shift_rank_one(success_lines, ldos.RANK_PREFIX)
        with self.assertRaisesRegex(ldos.EvidenceError, "collective AB/BA"):
            ldos.parse_benchmark_output(
                "\n".join(success_lines), "", returncode=0, expected_ranks=2
            )

        rejected_lines = rejected_stdout(2).splitlines()
        shift_rank_one(rejected_lines, ldos.REJECTED_PREFIX)
        with self.assertRaisesRegex(ldos.EvidenceError, "collective AB/BA"):
            ldos._parse_rejected_repetition_evidence(
                "\n".join(rejected_lines), "", expected_ranks=2
            )

    def test_rank1_success_per_rank_abba_overlap_is_rejected(self) -> None:
        lines = valid_stdout().splitlines()
        for index, line in enumerate(lines):
            if not line.startswith(ldos.RANK_PREFIX):
                continue
            record = json.loads(line.removeprefix(ldos.RANK_PREFIX))
            if record["rank"] != 0 or record["repetition"] != 1:
                continue
            for update in record["host_timed_updates"]:
                update["start_monotonic_ns"] -= 1_500_000_000
                update["stop_monotonic_ns"] -= 1_500_000_000
            lines[index] = ldos.RANK_PREFIX + compact(record)
            break
        with self.assertRaisesRegex(ldos.EvidenceError, "per-rank AB/BA"):
            ldos.parse_benchmark_output(
                "\n".join(lines), "", returncode=0, expected_ranks=1
            )

    def test_rejected_reason_and_rank_matrix_are_recomputed(self) -> None:
        lines = rejected_stdout(2).splitlines()
        record = json.loads(lines[0].removeprefix(ldos.REJECTED_PREFIX))
        record["reason_predicates"][
            "host_involuntary_context_switches_nonzero"
        ] = False
        lines[0] = ldos.REJECTED_PREFIX + compact(record)
        with self.assertRaisesRegex(ldos.EvidenceError, "predicate"):
            ldos._parse_rejected_repetition_evidence(
                "\n".join(lines), "", expected_ranks=2
            )
        with self.assertRaisesRegex(ldos.EvidenceError, "matrix"):
            ldos._parse_rejected_repetition_evidence(
                rejected_stdout(2).split(ldos.REJECTED_PREFIX, 1)[0]
                + rejected_stdout(2).splitlines()[0] + "\n",
                "", expected_ranks=2,
            )

    def test_rejected_uuid_and_affinity_bindings_are_exact(self) -> None:
        for mutation, message in (("duplicate-uuid", "distinct"), ("overlap", "overlap")):
            lines = rejected_stdout(2).splitlines()
            second = json.loads(lines[1].removeprefix(ldos.REJECTED_PREFIX))
            if mutation == "duplicate-uuid":
                first = json.loads(lines[0].removeprefix(ldos.REJECTED_PREFIX))
                second["device_uuid"] = first["device_uuid"]
            else:
                second["cpu_affinity"] = [1]
                for segment in ("host", "resident"):
                    for update in second[f"{segment}_timed_updates"]:
                        update["cpu_before"] = 1
                        update["cpu_after"] = 1
                        update["affinity_before"] = [1]
                        update["affinity_after"] = [1]
            lines[1] = ldos.REJECTED_PREFIX + compact(second)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ldos.EvidenceError, message
            ):
                ldos._parse_rejected_repetition_evidence(
                    "\n".join(lines), "", expected_ranks=2
                )

    def test_stderr_only_rejected_marker_is_not_silently_ignored(self) -> None:
        with self.assertRaisesRegex(ldos.EvidenceError, "stdout-only"):
            ldos._parse_rejected_repetition_evidence(
                "", rejected_stdout(), expected_ranks=1
            )

    def test_rejected_marker_is_incompatible_with_success(self) -> None:
        rejected = "\n".join(
            line for line in rejected_stdout().splitlines()
            if not line.startswith("FAIL:")
        )
        with self.assertRaisesRegex(ldos.EvidenceError, "rejected repetition"):
            self.parse(valid_stdout() + rejected)

    def test_worker_fail_remains_primary_over_callback_teardown(self) -> None:
        result = process_result(1)
        result.update(
            {
                "returncode": 7,
                "stdout": rejected_stdout(),
                "stderr": "FAIL: focused LDOS scheduling/fault gate\n",
                "callback_error": "EvidenceError: missing pidfd teardown",
            }
        )
        primary, evidence = ldos._classify_lane_process_failure(
            result, ranks=1, go_monotonic_ns=1
        )
        self.assertIsNotNone(primary)
        self.assertIn("worker primary failure", str(primary))
        self.assertNotIn("callback failed", str(primary))
        self.assertEqual(evidence["rejected_repetition"]["repetition"], 0)

    def test_process_execution_error_recovers_worker_primary_and_evidence(self) -> None:
        result = process_result(1)
        result.update(
            {
                "returncode": 7,
                "stdout": rejected_stdout(),
                "stderr": "FAIL: focused LDOS scheduling/fault gate\n",
                "callback_error": "EvidenceError: teardown observation failed",
                "monitoring_error_caused_termination": False,
            }
        )
        error = ldos.ProcessExecutionError(
            "process-control failed", result, OSError("stream drain failed")
        )
        primary, evidence = ldos._classify_lane_process_execution_error(
            error, ranks=1, go_monotonic_ns=1
        )
        self.assertIn("worker primary failure", str(primary))
        self.assertEqual(evidence["rejected_repetition"]["repetition"], 0)
        self.assertTrue(
            any("stream drain failed" in note for note in primary.__notes__)
        )

    def test_lane_process_execution_handler_persists_rejected_primary(self) -> None:
        result = process_result(1)
        result.update(
            {
                "returncode": 7,
                "stdout": rejected_stdout(),
                "stderr": "FAIL: focused LDOS scheduling/fault gate\n",
                "callback_error": "EvidenceError: teardown observation failed",
                "monitoring_error_caused_termination": False,
            }
        )
        process_error = ldos.ProcessExecutionError(
            "process-control failed", result, OSError("stream drain failed")
        )
        monitor = types.SimpleNamespace(
            go_monotonic_ns=1,
            diagnose_failure_teardown=mock.Mock(
                return_value=["pidfd cleanup diagnostic"]
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            with self.assertRaisesRegex(
                ldos.EvidenceError, "worker primary failure"
            ) as caught:
                ldos._raise_lane_process_execution_error(
                    output=output,
                    label="ldos-rank-1",
                    error=process_error,
                    lane_monitor=monitor,
                    ranks=1,
                    expected_device_uuids=[
                        "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                    ],
                    expected_worker_cpu_masks=[{1}],
                )
            self.assertIs(caught.exception.__cause__, process_error)
            self.assertTrue(
                any("stream drain failed" in note for note in caught.exception.__notes__)
            )
            self.assertTrue(
                any("pidfd cleanup diagnostic" in note for note in caught.exception.__notes__)
            )
            record = json.loads((output / "ldos-rank-1.json").read_text())
            self.assertEqual(
                record["benchmark"]["rejected_repetition"]["repetition"], 0
            )
            self.assertIn("worker primary failure", record["validation_error"])

    def test_teardown_diagnostic_exception_never_replaces_either_primary_path(self) -> None:
        result = process_result(1)
        result.update(
            {
                "returncode": 7,
                "stdout": rejected_stdout(),
                "stderr": "FAIL: focused LDOS scheduling/fault gate\n",
            }
        )
        monitor = types.SimpleNamespace(
            go_monotonic_ns=1,
            diagnose_failure_teardown=mock.Mock(
                side_effect=RuntimeError("diagnose exploded")
            )
        )
        normal_primary, failure_evidence = ldos._classify_lane_process_failure(
            result, ranks=1, go_monotonic_ns=1
        )
        with tempfile.TemporaryDirectory() as normal_directory:
            normal_output = pathlib.Path(normal_directory)
            with self.assertRaises(ldos.EvidenceError) as normal_caught:
                ldos._raise_lane_result_primary(
                    normal_primary,
                    result=result,
                    lane_monitor=monitor,
                    output=normal_output,
                    label="ldos-rank-1",
                    failure_evidence=failure_evidence,
                )
            self.assertIs(normal_caught.exception, normal_primary)
            self.assertTrue(
                any("diagnose exploded" in note for note in normal_primary.__notes__)
            )
            normal_record = json.loads(
                (normal_output / "ldos-rank-1.json").read_text()
            )
            self.assertIsNotNone(
                normal_record["benchmark"]["rejected_repetition"]
            )

        process_error = ldos.ProcessExecutionError(
            "control failed", result, OSError("drain failed")
        )
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            with self.assertRaisesRegex(
                ldos.EvidenceError, "worker primary failure"
            ) as process_caught:
                ldos._raise_lane_process_execution_error(
                    output=output,
                    label="ldos-rank-1",
                    error=process_error,
                    lane_monitor=monitor,
                    ranks=1,
                    expected_device_uuids=[
                        "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                    ],
                    expected_worker_cpu_masks=[{1}],
                )
            self.assertTrue(
                any(
                    "diagnose exploded" in note
                    for note in process_caught.exception.__notes__
                )
            )
            persisted = json.loads((output / "ldos-rank-1.json").read_text())
            self.assertIsNotNone(persisted["benchmark"]["rejected_repetition"])

    def test_process_control_induced_exit_primary_and_independent_worker_evidence(self) -> None:
        bare = process_result(1)
        bare.update(
            {
                "returncode": -15,
                "stdout": "partial output\n",
                "stderr": "control path failed\n",
                "process_control_error_caused_termination": True,
            }
        )
        bare_error = ldos.ProcessExecutionError(
            "control failed", bare, OSError("communicate failed")
        )
        primary, evidence = ldos._classify_lane_process_execution_error(
            bare_error, ranks=1
        )
        self.assertIn("process-control failure terminated", str(primary))
        self.assertIsNone(evidence)

        rejected = process_result(1)
        rejected.update(
            {
                "returncode": 7,
                "stdout": rejected_stdout(),
                "stderr": "FAIL: focused LDOS scheduling/fault gate\n",
                "process_control_error_caused_termination": True,
            }
        )
        rejected_error = ldos.ProcessExecutionError(
            "control failed", rejected, OSError("unrelated drain failure")
        )
        primary, evidence = ldos._classify_lane_process_execution_error(
            rejected_error, ranks=1, go_monotonic_ns=1
        )
        self.assertIn("worker primary failure", str(primary))
        self.assertEqual(evidence["rejected_repetition"]["repetition"], 0)
        self.assertTrue(
            any("unrelated drain failure" in note for note in primary.__notes__)
        )

    def test_monitor_termination_is_observer_primary_not_worker_primary(self) -> None:
        result = process_result(1)
        result.update(
            {
                "returncode": -15,
                "stdout": "partial worker output\n",
                "callback_error": "EvidenceError: NVML binding changed",
                "monitoring_error_caused_termination": True,
            }
        )
        primary, evidence = ldos._classify_lane_process_failure(result, ranks=1)
        self.assertIn("telemetry callback failed", str(primary))
        self.assertNotIn("worker primary failure", str(primary))
        self.assertIsNone(evidence)

    def test_bare_nonzero_remains_primary_over_callback_teardown(self) -> None:
        result = process_result(1)
        result.update(
            {
                "returncode": 7,
                "stdout": "",
                "stderr": "",
                "callback_error": "EvidenceError: teardown callback",
            }
        )
        primary, evidence = ldos._classify_lane_process_failure(result, ranks=1)
        self.assertIsNone(evidence)
        self.assertIn("worker primary failure", str(primary))
        self.assertIn("exit status 7", str(primary))
        self.assertNotIn("callback failed", str(primary))

    def test_fail_and_skip_tokens_are_rejected_in_either_stream(self) -> None:
        for stdout, stderr in (
            (valid_stdout() + "FAIL: injected\n", ""),
            (valid_stdout(), "SKIP: no GPU\n"),
            (valid_stdout() + "failed\n", ""),
        ):
            with self.subTest(stdout=stdout[-20:], stderr=stderr):
                with self.assertRaisesRegex(ldos.EvidenceError, "forbidden"):
                    self.parse(stdout, stderr)

    def test_unknown_gpmeep_markers_are_rejected_in_either_stream(self) -> None:
        for stdout, stderr in (
            (valid_stdout() + 'gpmeep-unknown-v1:{"x":1}\n', ""),
            (valid_stdout(), 'rank 1 gpmeep-device-v1:{"x":1}\n'),
            (valid_stdout() + '[rank-0]gpmeep-hidden-v1:{"x":1}\n', ""),
        ):
            with self.subTest(stderr=stderr):
                with self.assertRaisesRegex(ldos.EvidenceError, "unknown gpmeep"):
                    self.parse(stdout, stderr)

    def test_record_must_be_unique_and_stdout_only(self) -> None:
        for stdout, stderr in (
            (valid_stdout() + benchmark_line() + "\n", ""),
            (valid_stdout().replace(benchmark_line(), ""), ""),
            (valid_stdout().replace(benchmark_line(), ""), benchmark_line() + "\n"),
        ):
            with self.subTest(stderr=stderr):
                with self.assertRaisesRegex(ldos.EvidenceError, "exactly one LDOS benchmark"):
                    self.parse(stdout, stderr)

    def test_focused_pass_must_be_unique_exact_and_stdout_only(self) -> None:
        for stdout, stderr in (
            (valid_stdout() + ldos.PASS_LINE + "\n", ""),
            (valid_stdout().replace(ldos.PASS_LINE, "PASS: almost"), ""),
            (valid_stdout().replace(ldos.PASS_LINE, ""), ldos.PASS_LINE + "\n"),
        ):
            with self.subTest(stderr=stderr):
                with self.assertRaisesRegex(ldos.EvidenceError, "focused LDOS PASS"):
                    self.parse(stdout, stderr)

    def test_fixed_integer_contract_is_enforced(self) -> None:
        replacements = {
            "ranks=1": "ranks=2",
            "pixels=64": "pixels=63",
            "updates=16": "updates=15",
            "repetitions=5": "repetitions=4",
        }
        for old, new in replacements.items():
            with self.subTest(field=old):
                with self.assertRaisesRegex(ldos.EvidenceError, "differs from fixed"):
                    self.parse(valid_stdout().replace(old, new))

    def test_host_transfer_must_exceed_resident_result(self) -> None:
        with self.assertRaisesRegex(ldos.EvidenceError, "does not exceed"):
            self.parse(valid_stdout().replace('"host_d2h":4194304', '"host_d2h":512'))

    def test_major_fault_and_timed_update_gates_are_enforced(self) -> None:
        lines = valid_stdout().splitlines()
        rank_index = next(
            index for index, line in enumerate(lines)
            if line.startswith(ldos.RANK_PREFIX)
        )
        original = json.loads(lines[rank_index].removeprefix(ldos.RANK_PREFIX))
        mutations = []
        fault = copy.deepcopy(original)
        fault["host_major_faults"] = 1
        mutations.append((fault, "usage sum|scheduling/fault"))
        missing = copy.deepcopy(original)
        missing["host_timed_updates"].pop()
        mutations.append((missing, "cardinality"))
        migrated = copy.deepcopy(original)
        migrated["resident_timed_updates"][0]["cpu_after"] = 99
        mutations.append((migrated, "CPU/interval"))
        broad = copy.deepcopy(original)
        broad["host_timed_updates"][0]["affinity_after"] = [1, 2]
        mutations.append((broad, "CPU/interval"))
        overflow = copy.deepcopy(original)
        overflow["host_timed_updates"][0]["stop_monotonic_ns"] = 1 << 64
        mutations.append((overflow, "integer"))
        for record, message in mutations:
            candidate = list(lines)
            candidate[rank_index] = ldos.RANK_PREFIX + compact(record)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ldos.EvidenceError, message):
                    self.parse("\n".join(candidate))

    def test_each_raw_update_and_aggregate_timing_gate_is_independently_enforced(self) -> None:
        lines = valid_stdout().splitlines()
        rank_index = next(
            index for index, line in enumerate(lines)
            if line.startswith(ldos.RANK_PREFIX)
        )
        original = json.loads(lines[rank_index].removeprefix(ldos.RANK_PREFIX))
        mutations = []
        involuntary = copy.deepcopy(original)
        involuntary["host_timed_updates"][0]["involuntary_context_switches"] = 1
        mutations.append(involuntary)
        major = copy.deepcopy(original)
        major["resident_timed_updates"][0]["major_faults"] = 1
        mutations.append(major)
        aggregate = copy.deepcopy(original)
        aggregate["host_minor_faults"] = 1
        mutations.append(aggregate)
        elapsed = copy.deepcopy(original)
        elapsed["host_timed_updates"][0]["elapsed_ns"] += 1
        mutations.append(elapsed)
        for record in mutations:
            candidate = list(lines)
            candidate[rank_index] = ldos.RANK_PREFIX + compact(record)
            with self.assertRaises(ldos.EvidenceError):
                self.parse("\n".join(candidate))

        pair_index = next(
            index for index, line in enumerate(lines)
            if line.startswith(ldos.PAIR_PREFIX)
        )
        pair = json.loads(lines[pair_index].removeprefix(ldos.PAIR_PREFIX))
        pair["host_seconds"] += 0.000001
        pair["speedup"] = pair["host_seconds"] / pair["resident_seconds"]
        candidate = list(lines)
        candidate[pair_index] = ldos.PAIR_PREFIX + compact(pair)
        with self.assertRaisesRegex(ldos.EvidenceError, "raw per-update"):
            self.parse("\n".join(candidate))

    def test_speedups_must_be_finite_and_positive(self) -> None:
        candidates = (
            ('"host_seconds":1.3', '"host_seconds":NaN'),
            ('"resident_seconds":1', '"resident_seconds":0'),
            ('"speedup":1.3', '"speedup":-1'),
            ('"speedup":1.3', '"speedup":NaN'),
        )
        for old, new in candidates:
            with self.subTest(new=new):
                with self.assertRaisesRegex(ldos.EvidenceError, "JSON|positive|noncanonical"):
                    self.parse(valid_stdout().replace(old, new))

    def test_speedup_list_length_is_exact(self) -> None:
        for replacement in (
            "paired-speedups=1.3,1.4,1.5,1.6",
            "paired-speedups=1.3,1.4,1.5,1.6,1.7,1.8",
            "paired-speedups=1.3,1.4,,1.6,1.7",
        ):
            candidate = valid_stdout().replace(
                "paired-speedups=1.3,1.4,1.5,1.6,1.7", replacement
            )
            with self.subTest(replacement=replacement):
                with self.assertRaisesRegex(ldos.EvidenceError, "schema"):
                    self.parse(candidate)

    def test_minimum_and_median_are_recomputed(self) -> None:
        for old, new, message in (
            ("minimum-speedup=1.3", "minimum-speedup=1.31", "minimum"),
            ("median-speedup=1.5", "median-speedup=1.51", "median"),
        ):
            with self.subTest(field=message):
                with self.assertRaisesRegex(ldos.EvidenceError, message):
                    self.parse(valid_stdout().replace(old, new))

    def test_minimum_must_strictly_exceed_gate(self) -> None:
        candidate = (
            valid_stdout()
            .replace('"host_seconds":1.3', '"host_seconds":1.2', 1)
            .replace('"speedup":1.3', '"speedup":1.2', 1)
            .replace("minimum-speedup=1.3", "minimum-speedup=1.2")
            .replace("paired-speedups=1.3", "paired-speedups=1.2")
        )
        with self.assertRaisesRegex(
            ldos.EvidenceError, "does not exceed|raw per-update"
        ):
            self.parse(candidate)

    def test_extra_or_malformed_record_fields_are_rejected(self) -> None:
        for candidate in (
            valid_stdout().replace(" pixels=64", " extra=1 pixels=64"),
            valid_stdout().replace("ranks=1", "ranks=01"),
            valid_stdout().replace('"host_d2h":4194304', '"host_d2h":01'),
        ):
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(ldos.EvidenceError, "schema|canonical"):
                    self.parse(candidate)

    def test_raw_pair_speedup_digest_error_and_order_are_recomputed(self) -> None:
        candidates = (
            valid_stdout().replace('"speedup":1.3', '"speedup":1.31', 1),
            valid_stdout().replace(SAMPLE_DIGEST, "0" * 16, 1),
            valid_stdout().replace('"max_absolute_error":0', '"max_absolute_error":0.1', 1),
            valid_stdout().replace('"order":"host-resident"', '"order":"resident-host"', 1),
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate[:100]):
                with self.assertRaises(ldos.EvidenceError):
                    self.parse(candidate)

    def test_noncanonical_json_decimals_are_rejected(self) -> None:
        for old, new in (
            ('"host_seconds":1.3', '"host_seconds":01.3'),
            ('"host_seconds":1.3', '"host_seconds":1_3'),
            ('"max_absolute_error":0', '"max_absolute_error":-0'),
            ('"host_seconds":1.3', '"host_seconds":1E0'),
        ):
            with self.subTest(new=new):
                with self.assertRaisesRegex(ldos.EvidenceError, "JSON|canonical"):
                    self.parse(valid_stdout().replace(old, new, 1))

    def test_cross_repetition_samples_must_be_bitwise_deterministic(self) -> None:
        lines = valid_stdout().splitlines()
        for index, line in enumerate(lines):
            if not line.startswith(ldos.PAIR_PREFIX):
                continue
            record = json.loads(line.removeprefix(ldos.PAIR_PREFIX))
            if record["repetition"] != 1:
                continue
            record["resident_samples"][0][0] = 100000000
            record["host_samples"][0][0] = 100000000
            resident = [complex(*sample) for sample in record["resident_samples"]]
            host = [complex(*sample) for sample in record["host_samples"]]
            record["resident_digest"] = ldos._sample_digest(resident)
            record["host_digest"] = ldos._sample_digest(host)
            absolute, relative = ldos._sample_errors(host, resident)
            record["max_absolute_error"] = absolute
            record["max_relative_error"] = relative
            lines[index] = ldos.PAIR_PREFIX + compact(record)
            break
        with self.assertRaisesRegex(ldos.EvidenceError, "bitwise deterministic"):
            self.parse("\n".join(lines))

    def test_special_runtime_mapping_schema_is_exact(self) -> None:
        lines = valid_stdout().splitlines()
        runtime_index = next(
            index for index, line in enumerate(lines)
            if line.startswith(ldos.RUNTIME_PREFIX)
        )
        record = json.loads(lines[runtime_index].removeprefix(ldos.RUNTIME_PREFIX))
        record["special_mappings"] = [
            {
                "path": "/dev/nvidia0",
                "kind": "nvidia-character-device",
                "device": 1,
                "inode": 2,
                "mode": 0o20600,
                "rdev": 3,
                "deleted": False,
                "executable": False,
            },
            {
                "path": "/dev/shm/pmix-data",
                "kind": "ephemeral-data",
                "device": 4,
                "inode": 5,
                "mode": 0,
                "rdev": 0,
                "deleted": True,
                "executable": False,
            },
        ]
        lines[runtime_index] = ldos.RUNTIME_PREFIX + compact(record)
        self.parse("\n".join(lines))

        record["special_mappings"][1]["path"] = "/tmp/unbound-deleted"
        lines[runtime_index] = ldos.RUNTIME_PREFIX + compact(record)
        with self.assertRaisesRegex(ldos.EvidenceError, "whitelisted"):
            self.parse("\n".join(lines))

    def test_rank_uuid_counters_and_runtime_matrix_are_exact(self) -> None:
        candidates = (
            valid_stdout(2).replace(f'"device_uuid":"{CUDA_UUIDS[1]}"', f'"device_uuid":"{CUDA_UUIDS[0]}"', 1),
            valid_stdout().replace('"resident_cuda_kernel_launches":32', '"resident_cuda_kernel_launches":31', 1),
            valid_stdout().replace('"resident_d2h":512', '"resident_d2h":513', 1),
            valid_stdout().replace(ldos.RUNTIME_PREFIX, "gpmeep-removed-runtime-v1:", 1),
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate[:100]):
                with self.assertRaises(ldos.EvidenceError):
                    ldos.parse_benchmark_output(
                        candidate,
                        "",
                        returncode=0,
                        expected_ranks=2 if candidate.startswith(valid_stdout(2)[:20]) else 1,
                    )

    def test_focused_marker_topology_and_isolation_are_exact(self) -> None:
        lines = valid_stdout(2).splitlines()
        rank_indexes = [
            index for index, line in enumerate(lines)
            if line.startswith(ldos.RANK_PREFIX)
        ]
        reordered = list(lines)
        reordered[rank_indexes[0]], reordered[rank_indexes[1]] = (
            reordered[rank_indexes[1]], reordered[rank_indexes[0]]
        )
        with self.assertRaisesRegex(ldos.EvidenceError, "topology"):
            ldos.parse_benchmark_output(
                "\n".join(reordered), "", returncode=0, expected_ranks=2
            )
        for injected in (
            f"diagnostic {ldos.RANK_PREFIX}{{}}\n" + valid_stdout(),
            valid_stdout().replace(
                ldos.RANK_PREFIX, ldos.RANK_PREFIX + ldos.PAIR_PREFIX, 1
            ),
        ):
            with self.subTest(injected=injected[:80]):
                with self.assertRaisesRegex(
                    ldos.EvidenceError, "embedded|duplicated|noncanonical"
                ):
                    self.parse(injected)

    def test_adjacent_repetition_collective_order_is_exact(self) -> None:
        for ranks in (1, 2):
            source = valid_stdout(ranks).splitlines()
            previous_stop = max(
                json.loads(line.removeprefix(ldos.RANK_PREFIX))[
                    "resident_timed_updates"
                ][-1]["stop_monotonic_ns"]
                for line in source
                if line.startswith(ldos.RANK_PREFIX)
                and json.loads(line.removeprefix(ldos.RANK_PREFIX))[
                    "repetition"
                ] == 0
            )

            def shifted(target_start: int) -> str:
                lines = list(source)
                for index, line in enumerate(lines):
                    if not line.startswith(ldos.RANK_PREFIX):
                        continue
                    record = json.loads(line.removeprefix(ldos.RANK_PREFIX))
                    if record["repetition"] != 1:
                        continue
                    delta = (
                        target_start
                        - record["resident_timed_updates"][0][
                            "start_monotonic_ns"
                        ]
                    )
                    for segment in ("resident", "host"):
                        for update in record[f"{segment}_timed_updates"]:
                            update["start_monotonic_ns"] += delta
                            update["stop_monotonic_ns"] += delta
                    lines[index] = ldos.RANK_PREFIX + compact(record)
                return "\n".join(lines)

            with self.subTest(ranks=ranks, boundary="equal"):
                ldos.parse_benchmark_output(
                    shifted(previous_stop), "", returncode=0,
                    expected_ranks=ranks,
                )
            with self.subTest(ranks=ranks, boundary="overlap"):
                with self.assertRaisesRegex(
                    ldos.EvidenceError, "adjacent collective"
                ):
                    ldos.parse_benchmark_output(
                        shifted(previous_stop - 1), "", returncode=0,
                        expected_ranks=ranks,
                    )

    def test_rejected_prefix_boundary_is_exact_for_rank_one_and_two(self) -> None:
        for ranks in (1, 2):
            source = rejected_stdout_at_repetition_one(ranks).splitlines()
            accepted_stop = max(
                update["stop_monotonic_ns"]
                for line in source if line.startswith(ldos.RANK_PREFIX)
                for segment in ("host", "resident")
                for update in json.loads(
                    line.removeprefix(ldos.RANK_PREFIX)
                )[f"{segment}_timed_updates"]
            )

            def shifted(target_start: int) -> str:
                lines = list(source)
                for index, line in enumerate(lines):
                    if not line.startswith(ldos.REJECTED_PREFIX):
                        continue
                    record = json.loads(
                        line.removeprefix(ldos.REJECTED_PREFIX)
                    )
                    delta = (
                        target_start
                        - record["resident_timed_updates"][0][
                            "start_monotonic_ns"
                        ]
                    )
                    for segment in ("resident", "host"):
                        for update in record[f"{segment}_timed_updates"]:
                            update["start_monotonic_ns"] += delta
                            update["stop_monotonic_ns"] += delta
                    lines[index] = ldos.REJECTED_PREFIX + compact(record)
                return "\n".join(lines)

            with self.subTest(ranks=ranks, boundary="equal"):
                self.assertIsNotNone(
                    ldos._parse_rejected_repetition_evidence(
                        shifted(accepted_stop), "", expected_ranks=ranks
                    )
                )
            with self.subTest(ranks=ranks, boundary="overlap"):
                with self.assertRaisesRegex(ldos.EvidenceError, "overlaps"):
                    ldos._parse_rejected_repetition_evidence(
                        shifted(accepted_stop - 1), "", expected_ranks=ranks
                    )

    def test_rejected_rank_physics_is_identical_after_global_reduction(self) -> None:
        lines = rejected_stdout(2).splitlines()
        indexes = [
            index for index, line in enumerate(lines)
            if line.startswith(ldos.REJECTED_PREFIX)
        ]
        record = json.loads(lines[indexes[1]].removeprefix(ldos.REJECTED_PREFIX))
        record["host_samples"][0][0] = 999
        record["resident_samples"][0][0] = 999
        host = [complex(*sample) for sample in record["host_samples"]]
        resident = [complex(*sample) for sample in record["resident_samples"]]
        record["host_digest"] = ldos._sample_digest(host)
        record["resident_digest"] = ldos._sample_digest(resident)
        absolute, relative = ldos._sample_errors(host, resident)
        record["max_absolute_error"] = absolute
        record["max_relative_error"] = relative
        lines[indexes[1]] = ldos.REJECTED_PREFIX + compact(record)
        with self.assertRaisesRegex(ldos.EvidenceError, "physics differs"):
            ldos._parse_rejected_repetition_evidence(
                "\n".join(lines), "", expected_ranks=2
            )

    def test_uint64_runtime_identity_upper_bound_is_exact(self) -> None:
        lines = valid_stdout().splitlines()
        runtime_index = next(
            index for index, line in enumerate(lines)
            if line.startswith(ldos.RUNTIME_PREFIX)
        )
        record = json.loads(lines[runtime_index].removeprefix(ldos.RUNTIME_PREFIX))
        record["mappings"][0]["mtime_ns"] = (1 << 64) - 1
        lines[runtime_index] = ldos.RUNTIME_PREFIX + compact(record)
        self.parse("\n".join(lines))
        record["mappings"][0]["mtime_ns"] = 1 << 64
        lines[runtime_index] = ldos.RUNTIME_PREFIX + compact(record)
        with self.assertRaisesRegex(ldos.EvidenceError, "identity"):
            self.parse("\n".join(lines))

    def test_rejected_prior_prefix_does_not_apply_final_speed_gate(self) -> None:
        lines = rejected_stdout_at_repetition_one().splitlines()
        rank_index = next(
            index for index, line in enumerate(lines)
            if line.startswith(ldos.RANK_PREFIX)
        )
        pair_index = next(
            index for index, line in enumerate(lines)
            if line.startswith(ldos.PAIR_PREFIX)
        )
        rank_record = json.loads(lines[rank_index].removeprefix(ldos.RANK_PREFIX))
        cursor = rank_record["host_timed_updates"][0]["start_monotonic_ns"] - 1
        for update in rank_record["host_timed_updates"]:
            update["elapsed_ns"] = 68_750_000
            update["start_monotonic_ns"] = cursor + 1
            update["stop_monotonic_ns"] = cursor + 1 + update["elapsed_ns"]
            cursor = update["stop_monotonic_ns"] + 10
        lines[rank_index] = ldos.RANK_PREFIX + compact(rank_record)
        pair_record = json.loads(lines[pair_index].removeprefix(ldos.PAIR_PREFIX))
        pair_record["host_seconds"] = 1.1
        pair_record["speedup"] = 1.1
        lines[pair_index] = ldos.PAIR_PREFIX + compact(pair_record)
        parsed = ldos._parse_rejected_repetition_evidence(
            "\n".join(lines), "", expected_ranks=1
        )
        self.assertEqual(parsed["accepted_prefix"]["pairs"][0]["speedup"], 1.1)

    def test_accepted_prefix_returns_actual_bounds_for_every_length(self) -> None:
        for ranks in (1, 2):
            source = valid_stdout(ranks).splitlines()
            for repetitions in range(5):
                with self.subTest(ranks=ranks, repetitions=repetitions):
                    markers = [
                        line for line in source
                        if (
                            line.startswith(ldos.RANK_PREFIX)
                            and json.loads(
                                line.removeprefix(ldos.RANK_PREFIX)
                            )["repetition"] < repetitions
                        ) or (
                            line.startswith(ldos.PAIR_PREFIX)
                            and json.loads(
                                line.removeprefix(ldos.PAIR_PREFIX)
                            )["repetition"] < repetitions
                        )
                    ]
                    prefix = ldos._validate_accepted_prefix_with_success_parser(
                        markers,
                        repetitions=repetitions,
                        expected_ranks=ranks,
                        expected_device_uuids=None,
                        expected_worker_cpu_masks=None,
                    )
                    self.assertEqual(len(prefix["pairs"]), repetitions)
                    self.assertEqual(
                        len(prefix["rank_accounting"]), repetitions * ranks
                    )
                    self.assertEqual(
                        len(prefix["timeline_bounds"]), repetitions
                    )

    def test_accepted_prefix_synthetic_tail_uses_checked_high_clock(self) -> None:
        source = [
            line for line in valid_stdout().splitlines()
            if (
                line.startswith(ldos.RANK_PREFIX)
                and json.loads(line.removeprefix(ldos.RANK_PREFIX))[
                    "repetition"
                ] < 4
            ) or (
                line.startswith(ldos.PAIR_PREFIX)
                and json.loads(line.removeprefix(ldos.PAIR_PREFIX))[
                    "repetition"
                ] < 4
            )
        ]

        def shifted(offset: int) -> list[str]:
            result = []
            for line in source:
                if not line.startswith(ldos.RANK_PREFIX):
                    result.append(line)
                    continue
                record = json.loads(line.removeprefix(ldos.RANK_PREFIX))
                for segment in ("host", "resident"):
                    for update in record[f"{segment}_timed_updates"]:
                        update["start_monotonic_ns"] += offset
                        update["stop_monotonic_ns"] += offset
                result.append(ldos.RANK_PREFIX + compact(record))
            return result

        high = shifted(1_000_000_000_000_000)
        parsed = ldos._validate_accepted_prefix_with_success_parser(
            high, repetitions=4, expected_ranks=1,
            expected_device_uuids=None, expected_worker_cpu_masks=None,
        )
        self.assertEqual(len(parsed["timeline_bounds"]), 4)

        original_max = max(
            update["stop_monotonic_ns"]
            for line in source if line.startswith(ldos.RANK_PREFIX)
            for segment in ("host", "resident")
            for update in json.loads(
                line.removeprefix(ldos.RANK_PREFIX)
            )[f"{segment}_timed_updates"]
        )
        near_limit = shifted((1 << 64) - 1 - original_max - 100)
        with self.assertRaisesRegex(ldos.EvidenceError, "uint64"):
            ldos._validate_accepted_prefix_with_success_parser(
                near_limit, repetitions=4, expected_ranks=1,
                expected_device_uuids=None, expected_worker_cpu_masks=None,
            )

    def test_success_and_rejected_timeline_endpoint_binders(self) -> None:
        parsed = self.parse()
        first_start = parsed["timeline_bounds"][0][
            "first_start_monotonic_ns_by_rank"
        ][0]
        last_stop = parsed["timeline_bounds"][-1][
            "second_stop_monotonic_ns_by_rank"
        ][0]
        host = {
            "go_monotonic_ns": first_start,
            "process_finished_monotonic_ns": last_stop + 2,
            "results_ready_records": [
                {"rank": 0, "monotonic_ns": last_stop + 1}
            ],
        }
        ldos._validate_success_host_timeline(parsed, host, last_stop + 2)
        with self.assertRaisesRegex(ldos.EvidenceError, "before controller GO"):
            ldos._validate_success_host_timeline(
                parsed, {**host, "go_monotonic_ns": first_start + 1},
                last_stop + 2,
            )
        with self.assertRaisesRegex(ldos.EvidenceError, "RESULTS_READY"):
            ldos._validate_success_host_timeline(
                parsed,
                {
                    **host,
                    "results_ready_records": [
                        {"rank": 0, "monotonic_ns": last_stop - 1}
                    ],
                },
                last_stop + 2,
            )

        parsed_rank_two = ldos.parse_benchmark_output(
            valid_stdout(2), "", returncode=0, expected_ranks=2
        )
        first_rank_two = parsed_rank_two["timeline_bounds"][0][
            "first_start_monotonic_ns_by_rank"
        ]
        final_rank_two = parsed_rank_two["timeline_bounds"][-1][
            "second_stop_monotonic_ns_by_rank"
        ]
        rank_two_finished = max(final_rank_two) + 2
        rank_two_host = {
            "go_monotonic_ns": min(first_rank_two),
            "process_finished_monotonic_ns": rank_two_finished,
            "results_ready_records": [
                {"rank": rank, "monotonic_ns": final_rank_two[rank]}
                for rank in range(2)
            ],
        }
        ldos._validate_success_host_timeline(
            parsed_rank_two, rank_two_host, rank_two_finished
        )
        rank_zero_early = copy.deepcopy(rank_two_host)
        rank_zero_early["results_ready_records"][0]["monotonic_ns"] = (
            final_rank_two[0] - 1
        )
        rank_zero_early["results_ready_records"][1]["monotonic_ns"] = (
            rank_two_finished
        )
        with self.assertRaisesRegex(ldos.EvidenceError, "RESULTS_READY"):
            ldos._validate_success_host_timeline(
                parsed_rank_two, rank_zero_early, rank_two_finished
            )

        rejected = ldos._parse_rejected_repetition_evidence(
            rejected_stdout_at_repetition_one(), "", expected_ranks=1
        )
        all_records = [
            *rejected["accepted_prefix"]["rank_accounting"],
            *rejected["records"],
        ]
        starts = [
            update["start_monotonic_ns"]
            for record in all_records
            for segment in ("host", "resident")
            for update in record[f"{segment}_timed_updates"]
        ]
        stops = [
            update["stop_monotonic_ns"]
            for record in all_records
            for segment in ("host", "resident")
            for update in record[f"{segment}_timed_updates"]
        ]
        binding = ldos._cross_bind_rejected_timing(
            rejected,
            go_monotonic_ns=min(starts),
            process_finished_monotonic_ns=max(stops),
        )
        self.assertEqual(binding["last_update_stop_monotonic_ns"], max(stops))
        with self.assertRaisesRegex(ldos.EvidenceError, "before controller GO"):
            ldos._cross_bind_rejected_timing(
                rejected,
                go_monotonic_ns=min(starts) + 1,
                process_finished_monotonic_ns=max(stops),
            )

        for zero_binding in (
            lambda: ldos._cross_bind_rejected_timing(
                rejected,
                go_monotonic_ns=0,
                process_finished_monotonic_ns=max(stops),
            ),
            lambda: ldos._validate_success_host_timeline(
                parsed, {**host, "go_monotonic_ns": 0}, last_stop + 2
            ),
        ):
            with self.subTest(binding=zero_binding):
                with self.assertRaisesRegex(ldos.EvidenceError, "endpoints"):
                    zero_binding()

    def test_rejected_failure_evidence_requires_controller_timeline(self) -> None:
        result = process_result(1)
        result.update(
            {
                "returncode": 7,
                "stdout": rejected_stdout(),
                "stderr": "FAIL: focused LDOS scheduling/fault gate\n",
            }
        )
        primary, evidence = ldos._classify_lane_process_failure(result, ranks=1)
        self.assertIn("worker primary failure", str(primary))
        self.assertIsNone(evidence)
        self.assertTrue(
            any(
                "timeline endpoints are incomplete" in note
                for note in getattr(primary, "__notes__", ())
            )
        )


class DeviceInventoryTests(unittest.TestCase):
    def test_valid_inventory_and_explicit_selection(self) -> None:
        inventory = ldos.parse_device_inventory(device_csv())
        selected = ldos.select_devices(
            inventory,
            "GPU-11111111-2222-3333-4444-555555555555,"
            "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )
        self.assertEqual(selected[0]["index"], 1)
        self.assertEqual(selected[1]["index"], 0)

    def test_implicit_device_selection_is_forbidden(self) -> None:
        inventory = ldos.parse_device_inventory("\n".join(reversed(device_csv().splitlines())))
        with self.assertRaisesRegex(ldos.EvidenceError, "fallback is forbidden"):
            ldos.select_devices(inventory, None)


class NativeFdAllocationShimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_shim = ldos._FD_ALLOCATION_SHIM
        ldos._FD_ALLOCATION_SHIM = None

    def tearDown(self) -> None:
        ldos._FD_ALLOCATION_SHIM = self.previous_shim

    def _compile_native_library(self, directory: str) -> pathlib.Path:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler is unavailable")
        library = pathlib.Path(directory) / "libgpmeep-fd-allocation-shim.so.1"
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-O2",
                "-fPIC",
                "-fvisibility=hidden",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                "-Wl,-z,relro",
                "-Wl,-z,now",
                "-Wl,-soname,libgpmeep-fd-allocation-shim.so.1",
                str(SCRIPTS / "gpmeep-fd-allocation-shim.c"),
                "-o",
                str(library),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        library.chmod(0o400)
        return library

    def test_handled_outer_exception_does_not_close_new_allocations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = self._compile_native_library(directory)
            artifact = ldos.StableFile.open("outer-exception native shim", library)
            source_descriptor: int | None = None
            allocated: list[int] = []
            try:
                ldos._initialize_fd_allocation_shim(artifact)
                source_descriptor = os.open(
                    "/dev/null", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                )
                baseline = ldos._stable_open_descriptor_snapshot()
                try:
                    raise RuntimeError("outer exception remains handled")
                except RuntimeError:
                    guard = ldos._allocate_generation_guard(source_descriptor)
                    pidfd = ldos._pidfd_open(os.getpid())
                    read_descriptor, write_descriptor = ldos._allocate_probe_pipe()
                    allocated.extend(
                        (guard, pidfd, read_descriptor, write_descriptor)
                    )
                    for descriptor in allocated:
                        os.fstat(descriptor)
                    ldos._pidfd_send_signal(pidfd, 0)
                for descriptor in reversed(allocated):
                    ldos._close_native_owned_descriptor_once(
                        "handled-outer-exception test descriptor", descriptor
                    )
                allocated.clear()
                self.assertEqual(
                    ldos._stable_open_descriptor_snapshot(), baseline
                )
                self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
                self.assertFalse(ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)
            finally:
                for descriptor in reversed(allocated):
                    try:
                        ldos._close_native_owned_descriptor_once(
                            "handled-outer-exception cleanup descriptor",
                            descriptor,
                        )
                    except ldos.EvidenceError:
                        pass
                if source_descriptor is not None:
                    os.close(source_descriptor)
                ldos._FD_ALLOCATION_SHIM = None
                artifact.close()

    def test_finalizer_exception_never_closes_reused_descriptor_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = self._compile_native_library(directory)
            artifact = ldos.StableFile.open("finalizer ABA native shim", library)
            source_descriptor: int | None = None
            replacement: int | None = None
            previous_trace = sys.gettrace()
            try:
                ldos._initialize_fd_allocation_shim(artifact)
                source_descriptor = os.open(
                    "/dev/null", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                )
                baseline = ldos._stable_open_descriptor_snapshot()
                owner = ldos._allocate_generation_guard(source_descriptor)
                original_number = int(owner)
                state = {"injected": False}

                def tracer(frame, event, _argument):
                    nonlocal replacement
                    if (
                        frame.f_code is ldos.NativeOwnedFd.__del__.__code__
                        and event == "line"
                        and "closed" in frame.f_locals
                        and not state["injected"]
                    ):
                        self.assertFalse(
                            getattr(frame.f_locals["self"], "_native_owned", True)
                        )
                        replacement = os.open(
                            "/dev/zero",
                            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
                        )
                        self.assertEqual(replacement, original_number)
                        state["injected"] = True
                        raise KeyboardInterrupt("finalizer return-boundary injection")
                    return tracer

                sys.settrace(tracer)
                del owner
                gc.collect()
                sys.settrace(previous_trace)
                self.assertTrue(state["injected"])
                self.assertIsNotNone(replacement)
                replacement_state = os.fstat(replacement)
                self.assertTrue(stat.S_ISCHR(replacement_state.st_mode))
                os.close(replacement)
                replacement = None
                self.assertEqual(
                    ldos._stable_open_descriptor_snapshot(), baseline
                )
                self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
                self.assertFalse(ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)
            finally:
                sys.settrace(previous_trace)
                if replacement is not None:
                    os.close(replacement)
                if source_descriptor is not None:
                    os.close(source_descriptor)
                ldos._FD_ALLOCATION_SHIM = None
                artifact.close()

    def test_close_helper_retry_never_closes_reused_descriptor_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = self._compile_native_library(directory)
            artifact = ldos.StableFile.open("close-retry ABA native shim", library)
            source_descriptor: int | None = None
            replacement: int | None = None
            previous_trace = sys.gettrace()
            try:
                ldos._initialize_fd_allocation_shim(artifact)
                source_descriptor = os.open(
                    "/dev/null", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                )
                baseline = ldos._stable_open_descriptor_snapshot()
                owner = ldos._allocate_generation_guard(source_descriptor)
                original_number = int(owner)
                state = {"injected": False}

                def tracer(frame, event, _argument):
                    nonlocal replacement
                    if (
                        frame.f_code
                        is ldos._close_native_owned_descriptor_once.__code__
                        and event == "line"
                        and "closed" in frame.f_locals
                        and not state["injected"]
                    ):
                        self.assertFalse(
                            getattr(frame.f_locals["descriptor"], "_native_owned", True)
                        )
                        replacement = os.open(
                            "/dev/zero",
                            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
                        )
                        self.assertEqual(replacement, original_number)
                        state["injected"] = True
                        raise KeyboardInterrupt("close-helper return injection")
                    return tracer

                sys.settrace(tracer)
                with self.assertRaisesRegex(
                    KeyboardInterrupt, "close-helper return injection"
                ):
                    ldos._close_native_owned_descriptor_once(
                        "trace close-helper descriptor", owner
                    )
                sys.settrace(previous_trace)
                self.assertTrue(state["injected"])
                self.assertIsNotNone(replacement)
                ldos._close_native_owned_descriptor_once(
                    "repeated disarmed close-helper descriptor", owner
                )
                replacement_state = os.fstat(replacement)
                self.assertTrue(stat.S_ISCHR(replacement_state.st_mode))
                os.close(replacement)
                replacement = None
                self.assertEqual(
                    ldos._stable_open_descriptor_snapshot(), baseline
                )
                self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
                self.assertFalse(ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)
            finally:
                sys.settrace(previous_trace)
                if replacement is not None:
                    os.close(replacement)
                if source_descriptor is not None:
                    os.close(source_descriptor)
                ldos._FD_ALLOCATION_SHIM = None
                artifact.close()

    def test_strict_compiled_native_library_loads_and_self_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = self._compile_native_library(directory)
            artifact = ldos.StableFile.open("compiled native shim", library)
            try:
                ldos._initialize_fd_allocation_shim(artifact)
                self.assertIsInstance(
                    ldos._FD_ALLOCATION_SHIM, ldos.NativeFdAllocationShim
                )
                self.assertEqual(
                    ldos._FD_ALLOCATION_SHIM.artifact_identity,
                    artifact.record(),
                )
            finally:
                ldos._FD_ALLOCATION_SHIM = None
                artifact.close()

    def test_trace_exceptions_preserve_post_call_and_return_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = self._compile_native_library(directory)
            artifact = ldos.StableFile.open("trace-test native shim", library)
            source_descriptor: int | None = None
            previous_trace = sys.gettrace()
            try:
                ldos._initialize_fd_allocation_shim(artifact)
                source_descriptor = os.open(
                    "/dev/null", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                )
                baseline = ldos._stable_open_descriptor_snapshot()

                def expect_trace_exception(call, tracer, state) -> None:
                    sys.settrace(tracer)
                    try:
                        with self.assertRaisesRegex(
                            KeyboardInterrupt, "trace ownership boundary"
                        ):
                            call()
                    finally:
                        sys.settrace(previous_trace)
                    gc.collect()
                    self.assertTrue(state["injected"])
                    self.assertEqual(
                        ldos._stable_open_descriptor_snapshot(), baseline
                    )
                    self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
                    self.assertFalse(ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)

                def post_call_tracer(code):
                    state = {"injected": False}

                    def tracer(frame, event, _argument):
                        if (
                            frame.f_code is code
                            and event == "line"
                            and not state["injected"]
                        ):
                            result = frame.f_locals.get("result")
                            if result is not None and result.completed == 1:
                                state["injected"] = True
                                raise KeyboardInterrupt(
                                    "trace ownership boundary after native call"
                                )
                        return tracer

                    return tracer, state

                tracer, state = post_call_tracer(
                    ldos._native_single_fd_allocation.__code__
                )
                expect_trace_exception(
                    lambda: ldos._allocate_generation_guard(source_descriptor),
                    tracer,
                    state,
                )
                tracer, state = post_call_tracer(
                    ldos._native_pipe_allocation.__code__
                )
                expect_trace_exception(ldos._allocate_probe_pipe, tracer, state)

                for function, call in (
                    (
                        ldos._allocate_generation_guard,
                        lambda: ldos._allocate_generation_guard(source_descriptor),
                    ),
                    (ldos._pidfd_open, lambda: ldos._pidfd_open(os.getpid())),
                    (ldos._allocate_probe_pipe, ldos._allocate_probe_pipe),
                ):
                    with self.subTest(return_boundary=function.__name__):
                        state = {"injected": False}

                        def return_tracer(frame, event, _argument, *, target=function):
                            if (
                                frame.f_code is target.__code__
                                and event == "return"
                                and not state["injected"]
                            ):
                                state["injected"] = True
                                raise KeyboardInterrupt(
                                    "trace ownership boundary at helper return"
                                )
                            return return_tracer

                        expect_trace_exception(call, return_tracer, state)
            finally:
                sys.settrace(previous_trace)
                if source_descriptor is not None:
                    os.close(source_descriptor)
                ldos._FD_ALLOCATION_SHIM = None
                artifact.close()

    def test_native_layout_fingerprint_mismatch_is_rejected(self) -> None:
        class Function:
            def __init__(self, result=None):
                self.result = result
                self.argtypes = None
                self.restype = None

            def __call__(self, *_arguments):
                return self.result

        library = types.SimpleNamespace(
            gpmeep_fd_allocation_shim_abi=Function(
                ldos.FD_ALLOCATION_SHIM_ABI
            ),
            gpmeep_fd_result_layout=Function(
                ldos.FD_ALLOCATION_SHIM_FD_LAYOUT + 1
            ),
            gpmeep_pipe_result_layout=Function(
                ldos.FD_ALLOCATION_SHIM_PIPE_LAYOUT
            ),
            gpmeep_dupfd_cloexec_into=Function(),
            gpmeep_pidfd_open_into=Function(),
            gpmeep_pipe2_into=Function(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "shim.so"
            path.write_bytes(b"synthetic shim")
            artifact = ldos.StableFile.open("synthetic native shim", path)
            try:
                with (
                    mock.patch.object(ldos.ctypes, "CDLL", return_value=library),
                    self.assertRaisesRegex(
                        ldos.EvidenceError, "cannot load exact FD allocation shim"
                    ) as caught,
                ):
                    ldos.NativeFdAllocationShim(artifact)
                self.assertIn("native layout", str(caught.exception.__cause__))
            finally:
                artifact.close()

    def test_loaded_shim_must_match_later_archive_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source_path = root / "source.so"
            archive_path = root / "archive.so"
            source_path.write_bytes(b"loaded")
            archive_path.write_bytes(b"altered")
            source = ldos.StableFile.open("loaded shim", source_path)
            archive = ldos.StableFile.open("archived shim", archive_path)
            shim = PythonTestFdAllocationShim()
            shim.artifact_identity = source.record()
            ldos._FD_ALLOCATION_SHIM = shim
            try:
                with self.assertRaisesRegex(
                    ldos.EvidenceError, "archived FD allocation shim differs"
                ):
                    ldos._bind_loaded_fd_allocation_shim_archive(source, archive)
            finally:
                source.close()
                archive.close()


class NvmlSamplerTests(unittest.TestCase):
    class FakeFunction:
        def __init__(self, implementation):
            self.implementation = implementation
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.implementation(*args)

    def fake_nvml_library(self, *, current_source=None, utilization=False):
        shutdowns = []

        def current_query(source):
            def query(_handle, count_pointer, records):
                if source != current_source:
                    count_pointer._obj.value = 0
                    return ldos.NvmlSampler.SUCCESS
                count_pointer._obj.value = 1
                if records is None:
                    return ldos.NvmlSampler.INSUFFICIENT_SIZE
                records[0].pid = 77
                return ldos.NvmlSampler.SUCCESS
            return self.FakeFunction(query)

        def utilization_query(_handle, records, count_pointer, _cursor):
            if not utilization:
                count_pointer._obj.value = 0
                return ldos.NvmlSampler.NOT_FOUND
            count_pointer._obj.value = 1
            if records is None:
                return ldos.NvmlSampler.INSUFFICIENT_SIZE
            records[0].pid = 77
            records[0].timestamp_us = 1
            return ldos.NvmlSampler.SUCCESS

        functions = {
            "nvmlInit_v2": self.FakeFunction(lambda: 0),
            "nvmlShutdown": self.FakeFunction(lambda: shutdowns.append(True) or 0),
            "nvmlDeviceGetHandleByUUID": self.FakeFunction(
                lambda _uuid, pointer: setattr(pointer._obj, "value", 1) or 0
            ),
            "nvmlDeviceGetComputeRunningProcesses_v3": current_query("compute"),
            "nvmlDeviceGetGraphicsRunningProcesses_v3": current_query("graphics"),
            "nvmlDeviceGetMPSComputeRunningProcesses_v3": current_query("mps"),
            "nvmlDeviceGetProcessUtilization": self.FakeFunction(utilization_query),
        }
        for name in (
            "nvmlDeviceGetUtilizationRates", "nvmlDeviceGetClockInfo",
            "nvmlDeviceGetPowerUsage", "nvmlDeviceGetTemperature",
            "nvmlDeviceGetPerformanceState", "nvmlDeviceGetMemoryInfo",
        ):
            functions[name] = self.FakeFunction(lambda *_args: 0)
        return types.SimpleNamespace(**functions), shutdowns

    def test_lp64_abi_layouts_are_exact(self) -> None:
        ldos._verify_fd_allocation_shim_abi_layout()
        self.assertEqual(ctypes.sizeof(ldos.NativeFdResult), 16)
        self.assertEqual(ldos.NativeFdResult.abi_version.offset, 0)
        self.assertEqual(ldos.NativeFdResult.completed.offset, 4)
        self.assertEqual(ldos.NativeFdResult.fd.offset, 8)
        self.assertEqual(ldos.NativeFdResult.error_number.offset, 12)
        self.assertEqual(ctypes.sizeof(ldos.NativePipeResult), 20)
        self.assertEqual(ldos.NativePipeResult.abi_version.offset, 0)
        self.assertEqual(ldos.NativePipeResult.completed.offset, 4)
        self.assertEqual(ldos.NativePipeResult.read_fd.offset, 8)
        self.assertEqual(ldos.NativePipeResult.write_fd.offset, 12)
        self.assertEqual(ldos.NativePipeResult.error_number.offset, 16)
        self.assertEqual(ldos.FD_ALLOCATION_SHIM_FD_LAYOUT, 0x100004080C)
        self.assertEqual(ldos.FD_ALLOCATION_SHIM_PIPE_LAYOUT, 0x140004080C10)
        self.assertEqual(ctypes.sizeof(ldos.NvmlProcessInfoV3), 24)
        self.assertEqual(ldos.NvmlProcessInfoV3.used_gpu_memory.offset, 8)
        self.assertEqual(ldos.NvmlProcessInfoV3.gpu_instance_id.offset, 16)
        self.assertEqual(ldos.NvmlProcessInfoV3.compute_instance_id.offset, 20)
        self.assertEqual(ctypes.sizeof(ldos.NvmlProcessUtilizationSample), 32)
        self.assertEqual(
            ldos.NvmlProcessUtilizationSample.timestamp_us.offset, 8
        )

    def test_process_retrieval_retries_bounded_growth(self) -> None:
        sampler = object.__new__(ldos.NvmlSampler)
        calls = 0

        def query(_handle, count_pointer, records):
            nonlocal calls
            calls += 1
            count = count_pointer._obj
            if records is None:
                count.value = 1
                return ldos.NvmlSampler.INSUFFICIENT_SIZE
            if calls == 2:
                count.value = 2
                return ldos.NvmlSampler.INSUFFICIENT_SIZE
            count.value = 2
            records[0].pid = 11
            records[1].pid = 12
            return ldos.NvmlSampler.SUCCESS

        rows = sampler._process_rows(query, ctypes.c_void_p(1))
        self.assertEqual([row["pid"] for row in rows], [11, 12])
        self.assertEqual(calls, 3)

    def test_process_utilization_not_found_after_size_probe_is_empty(self) -> None:
        sampler = object.__new__(ldos.NvmlSampler)
        calls = 0

        def query(_handle, records, count_pointer, _cursor):
            nonlocal calls
            calls += 1
            count_pointer._obj.value = 72
            return (
                ldos.NvmlSampler.INSUFFICIENT_SIZE
                if records is None
                else ldos.NvmlSampler.NOT_FOUND
            )

        sampler.process_utilization = query
        self.assertEqual(
            sampler._process_utilization_rows(ctypes.c_void_p(1), 0), []
        )
        self.assertEqual(calls, 2)

    def test_uuid_crossing_and_graphics_process_are_outsiders(self) -> None:
        sampler = object.__new__(ldos.NvmlSampler)
        sampler.handles = {"GPU-a": "a", "GPU-b": "b"}
        sampler.process_functions = {
            "compute": "compute", "graphics": "graphics", "mps": "mps"
        }
        sampler.utilization_cursors = {"GPU-a": 0, "GPU-b": 0}

        def rows(source, handle):
            if handle == "a" and source in {"compute", "graphics"}:
                return [{"pid": 77}]
            return []

        sampler._process_rows = rows
        sampler._process_utilization_rows = lambda _handle, _cursor: []

        def utilization(_handle, pointer):
            pointer._obj.gpu = 0
            pointer._obj.memory = 0
            return 0

        def memory(_handle, pointer):
            pointer._obj.total = 10
            pointer._obj.free = 9
            pointer._obj.used = 1
            return 0

        sampler.utilization = utilization
        sampler.memory = memory
        sampler._require_success = lambda result, _label: self.assertEqual(result, 0)
        sampler._scalar = lambda *_arguments: 0
        sampler.clock = sampler.power = sampler.temperature = sampler.pstate = object()
        with mock.patch.object(ldos, "_process_start_time_ticks", return_value=9):
            sample = sampler.sample(
                allowed_process_epochs_by_uuid={"GPU-a": {}, "GPU-b": {77: 9}},
                scheduled_monotonic_ns=time.monotonic_ns(),
            )
        self.assertEqual(
            {(item["uuid"], item["source"]) for item in sample["outsiders"]},
            {("GPU-a", "compute"), ("GPU-a", "graphics")},
        )

    def test_constructor_rejects_every_nonempty_nvml_baseline(self) -> None:
        library_file = types.SimpleNamespace(
            proc_path="/proc/self/fd/9", record=lambda: {"sha256": "a" * 64}
        )
        for source, utilization in (
            ("compute", False), ("graphics", False), ("mps", False),
            (None, True),
        ):
            with self.subTest(source=source, utilization=utilization):
                library, shutdowns = self.fake_nvml_library(
                    current_source=source, utilization=utilization
                )
                with self.assertRaisesRegex(ldos.EvidenceError, "baseline"):
                    ldos.NvmlSampler(
                        library_file, ["GPU-a"], cdll_factory=lambda *_args, **_kwargs: library
                    )
                self.assertEqual(shutdowns, [True])

    def test_close_is_retryable_after_nvml_shutdown_failure(self) -> None:
        sampler = object.__new__(ldos.NvmlSampler)
        sampler.closed = False
        statuses = iter((17, ldos.NvmlSampler.SUCCESS))
        calls = []
        sampler.shutdown = lambda: calls.append(True) or next(statuses)
        with self.assertRaisesRegex(ldos.EvidenceError, "nvmlShutdown.*17"):
            sampler.close()
        self.assertFalse(sampler.closed)
        sampler.close()
        self.assertTrue(sampler.closed)
        self.assertEqual(calls, [True, True])

    def test_constructor_preserves_primary_when_cleanup_also_fails(self) -> None:
        library_file = types.SimpleNamespace(
            proc_path="/proc/self/fd/9", record=lambda: {"sha256": "a" * 64}
        )
        library, _shutdowns = self.fake_nvml_library(current_source="compute")
        library.nvmlShutdown = self.FakeFunction(lambda: 19)
        with self.assertRaisesRegex(ldos.EvidenceError, "baseline") as caught:
            ldos.NvmlSampler(
                library_file, ["GPU-a"],
                cdll_factory=lambda *_args, **_kwargs: library,
            )
        self.assertTrue(
            any("cleanup" in note and "19" in note for note in caught.exception.__notes__)
        )

    def test_constructor_shutdown_covers_first_post_init_allocation(self) -> None:
        library_file = types.SimpleNamespace(
            proc_path="/proc/self/fd/9", record=lambda: {"sha256": "a" * 64}
        )
        library, shutdowns = self.fake_nvml_library()

        def fail_first_post_init_assignment(instance, name, value):
            if name == "handles":
                raise MemoryError("injected post-init allocation")
            object.__setattr__(instance, name, value)

        with mock.patch.object(
            ldos.NvmlSampler, "__setattr__", new=fail_first_post_init_assignment
        ):
            with self.assertRaisesRegex(MemoryError, "post-init allocation"):
                ldos.NvmlSampler(
                    library_file, ["GPU-a"],
                    cdll_factory=lambda *_args, **_kwargs: library,
                )
        self.assertEqual(shutdowns, [True])

    def test_inventory_rejects_missing_duplicate_or_incompatible_devices(self) -> None:
        one = device_csv().splitlines()[0] + "\n"
        duplicate_uuid = device_csv().replace(
            "GPU-11111111-2222-3333-4444-555555555555",
            "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )
        duplicate_index = device_csv().replace("\n1,", "\n0,")
        old_compute = device_csv().replace("8.6", "5.2", 1)
        malformed = device_csv().replace(", 8.6", "", 1)
        for candidate, message in (
            (one, "two compatible"),
            (duplicate_uuid, "duplicate UUID"),
            (duplicate_index, "duplicate indices"),
            (old_compute, "support floor"),
            (malformed, "five columns"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ldos.EvidenceError, message):
                    ldos.parse_device_inventory(candidate)


class NvmlElfIdentityTests(unittest.TestCase):
    def parse_bytes(self, data: bytes):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "libnvidia-ml.so.1"
            path.write_bytes(data)
            descriptor = os.open(path, os.O_RDONLY)
            try:
                return ldos._nvml_elf_identity(descriptor, len(data))
            finally:
                os.close(descriptor)

    @staticmethod
    def elf64_sections(data: bytes) -> tuple[int, list[tuple[int, ...]]]:
        if data[:6] != b"\x7fELF\x02\x01":
            raise AssertionError("real-NVML mutation fixture must be ELF64 little-endian")
        header = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
        section_offset = header[5]
        section_size = header[10]
        section_count = header[11]
        if section_size != 64:
            raise AssertionError("unexpected real-NVML section size")
        return section_offset, [
            struct.unpack_from("<IIQQQQIIQQ", data, section_offset + index * 64)
            for index in range(section_count)
        ]

    @staticmethod
    def elf64_programs(data: bytes) -> tuple[int, list[tuple[int, ...]]]:
        if data[:6] != b"\x7fELF\x02\x01":
            raise AssertionError("mutation fixture must be ELF64 little-endian")
        header = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
        program_offset = header[4]
        program_size = header[8]
        program_count = header[9]
        if program_size != 56:
            raise AssertionError("unexpected program-header size")
        return program_offset, [
            struct.unpack_from("<IIQQQQQQ", data, program_offset + index * 56)
            for index in range(program_count)
        ]

    @classmethod
    def real_nvml_bytes(cls) -> bytearray:
        handle = ldos._resolve_nvml()
        try:
            return bytearray(os.pread(handle.descriptor, handle.size_bytes, 0))
        finally:
            handle.close()

    def test_close_registry_never_closes_reused_descriptor_number(self) -> None:
        original = os.open("/dev/null", os.O_RDONLY)
        guard = os.dup(original)
        replacement: int | None = None
        try:
            ldos._RETAINED_PROBE_DESCRIPTORS[guard] = original
            os.close(original)
            replacement = os.open("/dev/null", os.O_RDONLY)
            self.assertEqual(replacement, original)
            diagnostics = ldos._drain_retained_probe_descriptors()
            self.assertTrue(
                any("closed or reused" in item for item in diagnostics)
            )
            os.fstat(replacement)
            self.assertNotIn(guard, ldos._RETAINED_PROBE_DESCRIPTORS)
        finally:
            ldos._RETAINED_PROBE_DESCRIPTORS.pop(guard, None)
            try:
                os.close(guard)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
            if replacement is not None:
                try:
                    os.close(replacement)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise

    def test_close_fallback_never_closes_immediate_same_resource_reuse(
        self,
    ) -> None:
        descriptor = os.open("/dev/null", os.O_RDONLY)
        replacement: int | None = None
        real_close = os.close

        def close_reopen_then_raise(target):
            nonlocal replacement
            if target == descriptor and replacement is None:
                real_close(target)
                replacement = os.open("/dev/null", os.O_RDONLY)
                self.assertEqual(replacement, descriptor)
                raise OSError(errno.EBADF, "injected post-close EBADF")
            return real_close(target)

        try:
            with mock.patch.object(
                ldos.os, "close", side_effect=close_reopen_then_raise
            ):
                closed, diagnostics = ldos._close_descriptor_verified(descriptor)
            self.assertTrue(closed)
            self.assertTrue(replacement is not None)
            os.fstat(replacement)
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
        finally:
            if replacement is not None:
                try:
                    os.close(replacement)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise

    def test_generation_guard_uses_atomic_cloexec_duplication(self) -> None:
        descriptor = os.open("/dev/null", os.O_RDONLY)
        with mock.patch.object(
            ldos.os,
            "set_inheritable",
            side_effect=RuntimeError("non-atomic guard setup forbidden"),
        ) as setter:
            closed, diagnostics = ldos._close_descriptor_verified(descriptor)
        self.assertTrue(closed)
        self.assertEqual(diagnostics, [])
        setter.assert_not_called()
        self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)

    def test_fresh_generation_guard_rebinds_freed_poisoned_number(self) -> None:
        poisoned = os.open("/dev/null", os.O_RDONLY)
        descriptor = os.open("/dev/null", os.O_RDONLY)
        os.close(poisoned)
        ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS.add(poisoned)
        try:
            closed, diagnostics = ldos._close_descriptor_verified(descriptor)
            self.assertTrue(closed)
            self.assertEqual(diagnostics, [])
            self.assertNotIn(
                poisoned, ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS
            )
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
            for target in (poisoned, descriptor):
                with self.assertRaises(OSError) as caught:
                    os.fstat(target)
                self.assertEqual(caught.exception.errno, errno.EBADF)
        finally:
            ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS.discard(poisoned)
            for target in (poisoned, descriptor):
                try:
                    os.close(target)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise

    def test_generation_guard_post_call_baseexception_recovers_new_fd(
        self,
    ) -> None:
        baseline_fds = ldos._stable_open_descriptor_snapshot()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        allocated: int | None = None
        shim = ldos._require_fd_allocation_shim()
        real_duplicate = shim.duplicate_into

        def allocate_then_raise(source_fd, minimum_fd, result):
            nonlocal allocated
            real_duplicate(source_fd, minimum_fd, result)
            allocated = result.fd
            raise KeyboardInterrupt("injected post-dup exception")

        try:
            with mock.patch.object(
                shim, "duplicate_into", side_effect=allocate_then_raise
            ):
                closed, diagnostics = ldos._close_descriptor_verified(
                    descriptor
                )
            self.assertTrue(closed)
            self.assertTrue(
                any("KeyboardInterrupt" in item for item in diagnostics)
            )
            self.assertIsNotNone(allocated)
            for target in (descriptor, allocated):
                with self.assertRaises(OSError) as caught:
                    os.fstat(target)
                self.assertEqual(caught.exception.errno, errno.EBADF)
            self.assertEqual(
                ldos._stable_open_descriptor_snapshot(), baseline_fds
            )
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
            self.assertFalse(ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)
        finally:
            for target in (descriptor, allocated):
                if target is not None:
                    try:
                        os.close(target)
                    except OSError as error:
                        if error.errno != errno.EBADF:
                            raise

    def test_partial_native_out_parameter_recovers_new_guard(
        self,
    ) -> None:
        baseline_fds = ldos._stable_open_descriptor_snapshot()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        allocated: int | None = None
        shim = ldos._require_fd_allocation_shim()

        def allocate_write_fd_then_raise(source_fd, minimum_fd, result):
            nonlocal allocated
            result.abi_version = ldos.FD_ALLOCATION_SHIM_ABI
            result.completed = 0
            allocated = ldos.fcntl.fcntl(
                source_fd, ldos.fcntl.F_DUPFD_CLOEXEC, minimum_fd
            )
            result.fd = allocated
            result.error_number = 0
            raise KeyboardInterrupt("injected partial native result")

        try:
            with mock.patch.object(
                shim,
                "duplicate_into",
                side_effect=allocate_write_fd_then_raise,
            ):
                closed, diagnostics = ldos._close_descriptor_verified(
                    descriptor
                )
            self.assertTrue(closed)
            self.assertTrue(
                any("KeyboardInterrupt" in item for item in diagnostics)
            )
            self.assertIsNotNone(allocated)
            for target in (descriptor, allocated):
                with self.assertRaises(OSError) as caught:
                    os.fstat(target)
                self.assertEqual(caught.exception.errno, errno.EBADF)
            self.assertEqual(
                ldos._stable_open_descriptor_snapshot(), baseline_fds
            )
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
            self.assertFalse(ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)
        finally:
            for target in (descriptor, allocated):
                if target is not None:
                    try:
                        os.close(target)
                    except OSError as error:
                        if error.errno != errno.EBADF:
                            raise

    def test_generation_guard_rejects_fresh_wrong_open_description(self) -> None:
        baseline_fds = ldos._stable_open_descriptor_snapshot()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        wrong_guard: int | None = None
        shim = ldos._require_fd_allocation_shim()

        def return_wrong_guard(_source_fd, _minimum_fd, result):
            nonlocal wrong_guard
            wrong_guard = os.open(
                "/dev/zero", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            )
            result.abi_version = ldos.FD_ALLOCATION_SHIM_ABI
            result.fd = wrong_guard
            result.error_number = 0
            result.completed = 1

        try:
            with mock.patch.object(
                shim, "duplicate_into", side_effect=return_wrong_guard
            ):
                closed, diagnostics = ldos._close_descriptor_verified(
                    descriptor
                )
            self.assertTrue(closed)
            self.assertTrue(any("not bound" in item for item in diagnostics))
            for target in (descriptor, wrong_guard):
                self.assertIsNotNone(target)
                with self.assertRaises(OSError) as caught:
                    os.fstat(target)
                self.assertEqual(caught.exception.errno, errno.EBADF)
            self.assertEqual(
                ldos._stable_open_descriptor_snapshot(), baseline_fds
            )
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
            self.assertFalse(ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)
        finally:
            for target in (descriptor, wrong_guard):
                if target is not None:
                    try:
                        os.close(target)
                    except OSError as error:
                        if error.errno != errno.EBADF:
                            raise

    def test_pidfd_rejects_descriptor_for_different_live_pid(self) -> None:
        baseline_fds = ldos._stable_open_descriptor_snapshot()
        children: list[int] = []
        for _index in range(2):
            child = os.fork()
            if child == 0:
                time.sleep(30.0)
                os._exit(0)
            children.append(child)
        wrong_pidfd: int | None = None
        shim = ldos._require_fd_allocation_shim()
        real_pidfd_open = shim.pidfd_open_into

        def return_other_pidfd(_pid, flags, result):
            nonlocal wrong_pidfd
            real_pidfd_open(children[1], flags, result)
            wrong_pidfd = result.fd

        try:
            with (
                mock.patch.object(
                    shim, "pidfd_open_into", side_effect=return_other_pidfd
                ),
                self.assertRaisesRegex(ldos.EvidenceError, "wrong PID"),
            ):
                ldos._pidfd_open(children[0])
            self.assertIsNotNone(wrong_pidfd)
            with self.assertRaises(OSError) as caught:
                os.fstat(wrong_pidfd)
            self.assertEqual(caught.exception.errno, errno.EBADF)
            self.assertEqual(
                ldos._stable_open_descriptor_snapshot(), baseline_fds
            )
        finally:
            for child in children:
                try:
                    os.kill(child, ldos.signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for child in children:
                try:
                    os.waitpid(child, 0)
                except ChildProcessError:
                    pass
            if wrong_pidfd is not None:
                try:
                    os.close(wrong_pidfd)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise

    def test_pidfd_rejects_epoch_change_during_allocation(self) -> None:
        baseline_fds = ldos._stable_open_descriptor_snapshot()
        child = os.fork()
        if child == 0:
            time.sleep(30.0)
            os._exit(0)
        real_start = ldos._process_start_time_ticks
        expected_start = real_start(child)
        calls = 0

        def change_after_open(pid):
            nonlocal calls
            observed = real_start(pid)
            if pid == child:
                calls += 1
                if calls >= 2 and observed is not None:
                    return observed + 1
            return observed

        try:
            with (
                mock.patch.object(
                    ldos, "_process_start_time_ticks", side_effect=change_after_open
                ),
                self.assertRaisesRegex(ldos.EvidenceError, "epoch changed"),
            ):
                ldos._pidfd_open(
                    child, expected_start_time_ticks=expected_start
                )
            self.assertEqual(
                ldos._stable_open_descriptor_snapshot(), baseline_fds
            )
            self.assertFalse(ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)
        finally:
            try:
                os.kill(child, ldos.signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass

    def test_pidfd_allocation_allows_persistent_native_task(self) -> None:
        ready = ldos.threading.Event()
        release = ldos.threading.Event()

        def native_task():
            ready.set()
            release.wait(5.0)

        thread = ldos.threading.Thread(target=native_task)
        thread.start()
        descriptor: int | None = None
        try:
            self.assertTrue(ready.wait(1.0))
            descriptor = ldos._pidfd_open(os.getpid())
            ldos._pidfd_send_signal(descriptor, 0)
        finally:
            if descriptor is not None:
                ldos._close_native_owned_descriptor_once(
                    "persistent-task test pidfd", descriptor
                )
            release.set()
            thread.join(5.0)
        self.assertFalse(thread.is_alive())

    def test_pidfd_post_call_baseexception_recovers_and_contains_child(
        self,
    ) -> None:
        baseline_children = ldos._process_descendants(os.getpid())
        baseline_fds = ldos._stable_open_descriptor_snapshot()
        child = os.fork()
        if child == 0:
            time.sleep(30.0)
            os._exit(0)
        allocated: int | None = None
        shim = ldos._require_fd_allocation_shim()
        real_pidfd_open = shim.pidfd_open_into

        def allocate_then_raise(pid, flags, result):
            nonlocal allocated
            real_pidfd_open(pid, flags, result)
            allocated = result.fd
            raise KeyboardInterrupt("injected post-pidfd exception")

        try:
            with mock.patch.object(
                shim, "pidfd_open_into", side_effect=allocate_then_raise
            ):
                observed, diagnostics = ldos._contain_subreaper_descendants(
                    baseline_children, 1.0
                )
            self.assertTrue(observed)
            self.assertTrue(
                any("KeyboardInterrupt" in item for item in diagnostics)
            )
            self.assertIsNotNone(allocated)
            with self.assertRaises(OSError) as caught:
                os.fstat(allocated)
            self.assertEqual(caught.exception.errno, errno.EBADF)
            self.assertEqual(
                ldos._process_descendants(os.getpid()), baseline_children
            )
            self.assertEqual(
                ldos._stable_open_descriptor_snapshot(), baseline_fds
            )
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
            self.assertFalse(ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)
        finally:
            try:
                os.kill(child, ldos.signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass
            if allocated is not None:
                try:
                    os.close(allocated)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise

    def test_pipe_post_call_baseexception_recovers_and_restores_probe(
        self,
    ) -> None:
        handle = ldos._resolve_nvml()
        allocated: tuple[int, int] | None = None
        shim = ldos._require_fd_allocation_shim()
        real_pipe = shim.pipe2_into

        def allocate_then_raise(flags, result):
            nonlocal allocated
            real_pipe(flags, result)
            allocated = (result.read_fd, result.write_fd)
            raise KeyboardInterrupt("injected post-pipe exception")

        try:
            baseline_children = ldos._process_descendants(os.getpid())
            baseline_subreaper = ldos._child_subreaper_state()
            baseline_fds = ldos._stable_open_descriptor_snapshot()
            with (
                mock.patch.object(
                    shim, "pipe2_into", side_effect=allocate_then_raise
                ),
                self.assertRaisesRegex(
                    KeyboardInterrupt, "injected post-pipe exception"
                ),
            ):
                ldos._probe_nvml_loadability(handle)
            self.assertIsNotNone(allocated)
            for descriptor in allocated:
                with self.assertRaises(OSError) as caught:
                    os.fstat(descriptor)
                self.assertEqual(caught.exception.errno, errno.EBADF)
            self.assertEqual(
                ldos._process_descendants(os.getpid()), baseline_children
            )
            self.assertEqual(
                ldos._child_subreaper_state(), baseline_subreaper
            )
            self.assertEqual(
                ldos._stable_open_descriptor_snapshot(), baseline_fds
            )
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
            self.assertFalse(ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)
        finally:
            if allocated is not None:
                for descriptor in allocated:
                    try:
                        os.close(descriptor)
                    except OSError as error:
                        if error.errno != errno.EBADF:
                            raise
            handle.close()

    def test_fresh_containment_pidfd_rebinds_freed_poisoned_number(self) -> None:
        poisoned = os.open("/dev/null", os.O_RDONLY)
        os.close(poisoned)
        ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS.add(poisoned)
        baseline = ldos._process_descendants(os.getpid())
        before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
        child = os.fork()
        if child == 0:
            time.sleep(30.0)
            os._exit(0)
        opened: list[int] = []
        real_pidfd_open = ldos._pidfd_open

        def track_pidfd(pid, **kwargs):
            descriptor = real_pidfd_open(pid, **kwargs)
            opened.append(descriptor)
            return descriptor

        try:
            with mock.patch.object(
                ldos, "_pidfd_open", side_effect=track_pidfd
            ):
                observed, diagnostics = ldos._contain_subreaper_descendants(
                    baseline, 1.0
                )
            self.assertTrue(observed)
            self.assertEqual(diagnostics, [])
            self.assertIn(poisoned, opened)
            self.assertNotIn(
                poisoned, ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS
            )
            self.assertEqual(
                ldos._process_descendants(os.getpid()), baseline
            )
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
        finally:
            ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS.discard(poisoned)
            try:
                os.kill(child, ldos.signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass

    def test_ambiguous_guard_close_is_poisoned_and_never_retried(self) -> None:
        descriptor = os.open("/dev/null", os.O_RDONLY)
        filler: int | None = None
        replacement: int | None = None
        real_raw_close = ldos._raw_close_owned_descriptor
        ambiguous_guard: int | None = None

        def close_guard_then_report_ambiguous(target):
            nonlocal ambiguous_guard
            if ambiguous_guard is None:
                ambiguous_guard = target
                closed, diagnostic = real_raw_close(target)
                self.assertTrue(closed)
                self.assertIsNone(diagnostic)
                return False, "injected post-close BaseException"
            return real_raw_close(target)

        try:
            with mock.patch.object(
                ldos,
                "_raw_close_owned_descriptor",
                side_effect=close_guard_then_report_ambiguous,
            ):
                closed, diagnostics = ldos._close_descriptor_verified(descriptor)
            self.assertTrue(closed)
            self.assertIsNotNone(ambiguous_guard)
            self.assertIn(
                ambiguous_guard, ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS
            )
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
            filler = os.open("/dev/zero", os.O_RDONLY)
            self.assertEqual(filler, descriptor)
            replacement = os.open("/dev/null", os.O_RDONLY)
            self.assertEqual(replacement, ambiguous_guard)
            ldos._drain_retained_probe_descriptors()
            os.fstat(replacement)
            invalidated, repeated_diagnostics = (
                ldos._close_descriptor_verified(replacement)
            )
            self.assertTrue(invalidated)
            self.assertTrue(
                any("poisoned" in item for item in repeated_diagnostics)
            )
            os.fstat(replacement)
        finally:
            if ambiguous_guard is not None:
                ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS.discard(ambiguous_guard)
            if filler is not None:
                try:
                    os.close(filler)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise
            if replacement is not None:
                try:
                    os.close(replacement)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise

    def test_post_close_errno_baseexception_poisons_guard_without_retry(
        self,
    ) -> None:
        descriptor = os.open("/dev/null", os.O_RDONLY)
        filler: int | None = None
        replacement: int | None = None
        ambiguous_guard: int | None = None
        real_close = os.close

        class FakeClose:
            argtypes = None
            restype = None

            def __call__(self, target):
                nonlocal ambiguous_guard
                ambiguous_guard = target
                real_close(target)
                return -1

        class FakeLibc:
            close = FakeClose()

        try:
            with (
                mock.patch.object(ldos.ctypes, "CDLL", return_value=FakeLibc()),
                mock.patch.object(
                    ldos,
                    "_same_open_file_description",
                    return_value=(True, None),
                ),
                mock.patch.object(
                    ldos.ctypes,
                    "get_errno",
                    side_effect=KeyboardInterrupt("post-close errno failure"),
                ),
            ):
                closed, diagnostics = ldos._close_descriptor_verified(descriptor)
            self.assertTrue(closed)
            self.assertIsNotNone(ambiguous_guard)
            self.assertIn(
                ambiguous_guard, ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS
            )
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
            filler = os.open("/dev/zero", os.O_RDONLY)
            self.assertEqual(filler, descriptor)
            replacement = os.open("/dev/null", os.O_RDONLY)
            self.assertEqual(replacement, ambiguous_guard)
            ldos._drain_retained_probe_descriptors()
            os.fstat(replacement)
        finally:
            if ambiguous_guard is not None:
                ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS.discard(ambiguous_guard)
            for target in (filler, replacement):
                if target is not None:
                    try:
                        os.close(target)
                    except OSError as error:
                        if error.errno != errno.EBADF:
                            raise

    def test_poisoned_candidate_reference_is_invalidated_without_retry(
        self,
    ) -> None:
        descriptor = os.open("/dev/null", os.O_RDONLY)
        real_close = os.close
        real_raw_close = ldos._raw_close_owned_descriptor
        raw_candidate_failed = False

        def fail_primary_close(target):
            if target == descriptor:
                raise RuntimeError("injected primary close failure")
            return real_close(target)

        def fail_candidate_raw_once(target):
            nonlocal raw_candidate_failed
            if target == descriptor and not raw_candidate_failed:
                raw_candidate_failed = True
                return False, "injected ambiguous candidate raw close"
            return real_raw_close(target)

        try:
            with (
                mock.patch.object(ldos.os, "close", side_effect=fail_primary_close),
                mock.patch.object(
                    ldos,
                    "_same_open_file_description",
                    side_effect=(
                        (True, None),
                        (True, None),
                        (None, "injected unknown state"),
                    ),
                ),
                mock.patch.object(
                    ldos,
                    "_raw_close_owned_descriptor",
                    side_effect=fail_candidate_raw_once,
                ),
            ):
                invalidated, diagnostics = ldos._close_descriptor_verified(
                    descriptor
                )
            self.assertTrue(invalidated)
            self.assertTrue(raw_candidate_failed)
            self.assertIn(descriptor, ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS)
            os.fstat(descriptor)
            repeated, repeated_diagnostics = ldos._close_descriptor_verified(
                descriptor
            )
            self.assertTrue(repeated)
            self.assertTrue(
                any("poisoned" in item for item in repeated_diagnostics)
            )
            os.fstat(descriptor)
        finally:
            ldos._POISONED_PROBE_DESCRIPTOR_NUMBERS.discard(descriptor)
            try:
                os.close(descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise

    def test_successful_close_return_is_authoritative_without_aba_probe(
        self,
    ) -> None:
        descriptor = os.open("/dev/null", os.O_RDONLY)
        with mock.patch.object(
            ldos, "_same_open_file_description", return_value=(True, None)
        ) as comparison:
            closed, diagnostics = ldos._close_descriptor_verified(descriptor)
        self.assertTrue(closed)
        self.assertEqual(diagnostics, [])
        self.assertEqual(comparison.call_count, 1)
        self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)

    def test_nvml_probe_close_fallback_construction_failure_preserves_primary(
        self,
    ) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            real_pidfd_open = ldos._pidfd_open
            real_close = os.close
            real_cdll = ctypes.CDLL
            exact_pidfd: int | None = None
            close_failed = False
            fallback_failed = False

            def track_exact_pidfd(pid, **kwargs):
                nonlocal exact_pidfd
                descriptor = real_pidfd_open(pid, **kwargs)
                if exact_pidfd is None:
                    exact_pidfd = descriptor
                return descriptor

            def fail_exact_close(descriptor):
                nonlocal close_failed
                if exact_pidfd is not None and descriptor == exact_pidfd:
                    close_failed = True
                    raise RuntimeError("persistent exact close failure")
                return real_close(descriptor)

            def fail_first_fallback_cdll(name, *args, **kwargs):
                nonlocal fallback_failed
                if close_failed and not fallback_failed and name is None:
                    fallback_failed = True
                    raise RuntimeError("injected libc close construction failure")
                return real_cdll(name, *args, **kwargs)

            def slow_loader(_path, **_kwargs):
                time.sleep(1.0)
                return RequiredSymbols()

            before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            before_children = ldos._process_descendants(os.getpid())
            before_subreaper = ldos._child_subreaper_state()
            with (
                mock.patch.object(
                    ldos, "SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS", 0.1
                ),
                mock.patch.object(
                    ldos, "_pidfd_open", side_effect=track_exact_pidfd
                ),
                mock.patch.object(ldos.os, "close", side_effect=fail_exact_close),
                mock.patch.object(
                    ldos.ctypes, "CDLL", side_effect=fail_first_fallback_cdll
                ),
                self.assertRaisesRegex(ldos.EvidenceError, "timed out"),
            ):
                ldos._probe_nvml_loadability(
                    handle, cdll_factory=slow_loader
                )
            self.assertTrue(close_failed)
            self.assertTrue(fallback_failed)
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
            self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
            self.assertEqual(ldos._process_descendants(os.getpid()), before_children)
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
        finally:
            handle.close()

    def test_real_nvml_has_exact_soname_build_id_and_required_symbols(self) -> None:
        handle = ldos._resolve_nvml()
        try:
            identity = ldos._nvml_elf_identity(
                handle.descriptor, handle.size_bytes
            )
        finally:
            handle.close()
        self.assertEqual(identity["soname"], "libnvidia-ml.so.1")
        self.assertRegex(identity["gnu_build_id"], r"^[0-9a-f]{16,128}$")
        self.assertEqual(identity["required_symbols"], list(ldos.NVML_REQUIRED_SYMBOLS))

    def test_nvml_resolver_has_bounded_multiarch_and_wsl_allowlist(self) -> None:
        self.assertEqual(ldos._native_machine("x86_64", 64), "x86_64")
        self.assertEqual(ldos._native_machine("aarch64", 64), "aarch64")
        for machine, bits in (("x86_64", 32), ("aarch64", 32), ("x86_64", 16)):
            with self.subTest(machine=machine, bits=bits), self.assertRaisesRegex(
                ldos.EvidenceError, "64-bit LP64"
            ):
                ldos._native_machine(machine, bits)
        self.assertEqual(
            tuple(
                str(path)
                for path in ldos._nvml_library_candidates("x86_64", False)
            ),
            (
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                "/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                "/usr/lib64/libnvidia-ml.so.1",
                "/lib64/libnvidia-ml.so.1",
                "/usr/lib/libnvidia-ml.so.1",
                "/lib/libnvidia-ml.so.1",
                "/usr/lib/wsl/lib/libnvidia-ml.so.1",
                "/usr/lib/wsl/drivers/libnvidia-ml.so.1",
            ),
        )
        self.assertEqual(
            tuple(
                str(path)
                for path in ldos._nvml_library_candidates("x86_64", True)
            ),
            (
                "/usr/lib/wsl/lib/libnvidia-ml.so.1",
                "/usr/lib/wsl/drivers/libnvidia-ml.so.1",
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                "/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                "/usr/lib64/libnvidia-ml.so.1",
                "/lib64/libnvidia-ml.so.1",
                "/usr/lib/libnvidia-ml.so.1",
                "/lib/libnvidia-ml.so.1",
            ),
        )
        self.assertEqual(
            tuple(
                str(path)
                for path in ldos._nvml_library_candidates("aarch64", False)
            ),
            (
                "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
                "/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
                "/usr/lib64/libnvidia-ml.so.1",
                "/lib64/libnvidia-ml.so.1",
                "/usr/lib/libnvidia-ml.so.1",
                "/lib/libnvidia-ml.so.1",
                "/usr/lib/wsl/lib/libnvidia-ml.so.1",
                "/usr/lib/wsl/drivers/libnvidia-ml.so.1",
            ),
        )
        expected_aarch64_wsl = (
            "/usr/lib/wsl/lib/libnvidia-ml.so.1",
            "/usr/lib/wsl/drivers/libnvidia-ml.so.1",
            "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
            "/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib64/libnvidia-ml.so.1",
            "/lib64/libnvidia-ml.so.1",
            "/usr/lib/libnvidia-ml.so.1",
            "/lib/libnvidia-ml.so.1",
        )
        for alias in ("aarch64", "arm64"):
            with self.subTest(alias=alias):
                self.assertEqual(
                    tuple(
                        str(path)
                        for path in ldos._nvml_library_candidates(alias, True)
                    ),
                    expected_aarch64_wsl,
                )
        for machine in ("armv7l", "riscv64"):
            with self.subTest(machine=machine), self.assertRaisesRegex(
                ldos.EvidenceError, "does not support controller machine"
            ):
                ldos._nvml_library_candidates(machine, False)

    def test_resolvers_fail_before_candidate_probe_outside_lp64_floor(self) -> None:
        for resolver in (ldos._resolve_nvml, ldos._resolve_nvidia_smi):
            with (
                self.subTest(resolver=resolver.__name__),
                mock.patch.object(
                    ldos, "_native_machine",
                    side_effect=ldos.EvidenceError("requires 64-bit LP64"),
                ),
                mock.patch.object(pathlib.Path, "is_file") as is_file,
                self.assertRaisesRegex(ldos.EvidenceError, "64-bit LP64"),
            ):
                resolver()
            is_file.assert_not_called()

    def test_system_tool_elf_requires_complete_executable_program_image(self) -> None:
        valid = system_tool_elf_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "tool"

            def parse(data: bytes) -> dict:
                path.write_bytes(data)
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    return ldos._elf_identity(descriptor, len(data))
                finally:
                    os.close(descriptor)

            identity = parse(valid)
            self.assertIn(identity["elf_type"], {"ET_EXEC", "ET_DYN"})
            self.assertEqual(identity["entry_point"], 64)
            self.assertEqual(identity["program_header_count"], 1)
            invalid = (
                valid[:20],
                system_tool_elf_bytes(elf_type=0),
                system_tool_elf_bytes(elf_type=1),
                system_tool_elf_bytes(version=0),
                system_tool_elf_bytes(header_size=1),
                system_tool_elf_bytes(program_offset=63),
                system_tool_elf_bytes(program_offset=len(valid) + 1),
                system_tool_elf_bytes(program_count=0),
                system_tool_elf_bytes(executable_load=False),
                system_tool_elf_bytes(segment_file_size=len(valid) + 1),
                system_tool_elf_bytes(entry_point=0),
                system_tool_elf_bytes(entry_point=len(valid)),
            )
            for index, data in enumerate(invalid):
                with self.subTest(index=index), self.assertRaises(ldos.EvidenceError):
                    parse(data)

    def test_nvidia_smi_resolver_skips_invalid_elf_for_valid_later_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            invalid = root / "invalid-nvidia-smi"
            valid = root / "valid-nvidia-smi"
            invalid.write_bytes(system_tool_elf_bytes(entry_point=0))
            valid.write_bytes(system_tool_elf_bytes())
            invalid.chmod(0o755)
            valid.chmod(0o755)
            with (
                mock.patch.object(
                    ldos, "_nvidia_smi_candidates", return_value=(invalid, valid)
                ),
                mock.patch.object(
                    ldos, "_trusted_system_tool_tcb_handle",
                    return_value={"trusted": True},
                ),
                mock.patch.object(ldos, "_probe_nvidia_smi_startup"),
            ):
                selected = ldos._resolve_nvidia_smi()
            try:
                self.assertEqual(selected.path, valid.resolve())
                os.fstat(selected.descriptor)
            finally:
                selected.close()

    def test_real_system_tool_rejects_entry_program_and_interp_mutations(self) -> None:
        handle = ldos._resolve_nvidia_smi()
        try:
            source = bytearray(os.pread(handle.descriptor, handle.size_bytes, 0))
        finally:
            handle.close()
        program_offset, programs = self.elf64_programs(source)
        executable_loads = [
            program for program in programs if program[0] == 1 and program[1] & 1
        ]
        interpreter_indices = [
            index for index, program in enumerate(programs) if program[0] == 3
        ]
        self.assertTrue(executable_loads)
        self.assertEqual(len(interpreter_indices), 1)

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "nvidia-smi"

            def parse(data: bytes) -> dict:
                path.write_bytes(data)
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    return ldos._elf_identity(descriptor, len(data))
                finally:
                    os.close(descriptor)

            identity = parse(source)
            self.assertGreater(identity["entry_point"], 0)
            mutations: list[bytearray] = []

            entry_zero = bytearray(source)
            struct.pack_into("<Q", entry_zero, 24, 0)
            mutations.append(entry_zero)

            entry_outside = bytearray(source)
            executable_stop = max(
                program[3] + program[5] for program in executable_loads
            )
            struct.pack_into("<Q", entry_outside, 24, executable_stop)
            mutations.append(entry_outside)

            early_program_table = bytearray(source)
            struct.pack_into("<Q", early_program_table, 32, 63)
            mutations.append(early_program_table)

            first_program = program_offset
            file_larger_than_memory = bytearray(source)
            struct.pack_into(
                "<Q", file_larger_than_memory, first_program + 32,
                programs[0][6] + 1,
            )
            mutations.append(file_larger_than_memory)

            invalid_alignment = bytearray(source)
            struct.pack_into("<Q", invalid_alignment, first_program + 48, 3)
            mutations.append(invalid_alignment)

            interpreter_index = interpreter_indices[0]
            interpreter_program = programs[interpreter_index]
            interpreter_header = program_offset + interpreter_index * 56
            tiny_interpreter = bytearray(source)
            struct.pack_into("<Q", tiny_interpreter, interpreter_header + 32, 1)
            struct.pack_into("<Q", tiny_interpreter, interpreter_header + 40, 1)
            mutations.append(tiny_interpreter)

            unterminated_interpreter = bytearray(source)
            interpreter_end = interpreter_program[2] + interpreter_program[5]
            self.assertEqual(unterminated_interpreter[interpreter_end - 1], 0)
            unterminated_interpreter[interpreter_end - 1] = ord("X")
            mutations.append(unterminated_interpreter)

            duplicate_interpreter = bytearray(source)
            duplicate_index = next(
                index for index in range(len(programs))
                if index != interpreter_index
            )
            struct.pack_into(
                "<I", duplicate_interpreter,
                program_offset + duplicate_index * 56, 3,
            )
            mutations.append(duplicate_interpreter)

            for index, mutation in enumerate(mutations):
                with self.subTest(index=index), self.assertRaises(ldos.EvidenceError):
                    parse(bytes(mutation))

    def test_nvml_program_image_mutations_fail_and_resolver_falls_back(self) -> None:
        source = self.real_nvml_bytes()
        program_offset, programs = self.elf64_programs(source)
        _section_offset, sections = self.elf64_sections(source)
        load_indices = [
            index for index, program in enumerate(programs) if program[0] == 1
        ]
        dynamic_index = next(
            index for index, program in enumerate(programs) if program[0] == 2
        )
        self.assertTrue(load_indices)

        mutations: list[bytearray] = []
        for absolute_offset, pack_format, value in (
            (32, "<Q", 63),
            (54, "<H", 0),
            (56, "<H", 0),
        ):
            mutation = bytearray(source)
            struct.pack_into(pack_format, mutation, absolute_offset, value)
            mutations.append(mutation)

        first_load_index = load_indices[0]
        first_load = programs[first_load_index]
        first_load_header = program_offset + first_load_index * 56
        file_larger_than_memory = bytearray(source)
        struct.pack_into(
            "<Q", file_larger_than_memory, first_load_header + 32,
            first_load[6] + 1,
        )
        mutations.append(file_larger_than_memory)

        invalid_alignment = bytearray(source)
        struct.pack_into("<Q", invalid_alignment, first_load_header + 48, 3)
        mutations.append(invalid_alignment)

        incongruent_address = bytearray(source)
        struct.pack_into(
            "<Q", incongruent_address, first_load_header + 16,
            first_load[3] + 1,
        )
        mutations.append(incongruent_address)

        missing_dynamic = bytearray(source)
        struct.pack_into(
            "<I", missing_dynamic, program_offset + dynamic_index * 56, 0
        )
        mutations.append(missing_dynamic)

        no_executable_load = bytearray(source)
        for load_index in load_indices:
            flag_offset = program_offset + load_index * 56 + 4
            flags = struct.unpack_from("<I", no_executable_load, flag_offset)[0]
            struct.pack_into("<I", no_executable_load, flag_offset, flags & ~1)
        mutations.append(no_executable_load)

        dynamic_program = programs[dynamic_index]
        self.assertGreater(dynamic_program[5], 16)
        truncated_dynamic = bytearray(source)
        dynamic_header = program_offset + dynamic_index * 56
        struct.pack_into(
            "<Q", truncated_dynamic, dynamic_header + 32,
            dynamic_program[5] - 16,
        )
        struct.pack_into(
            "<Q", truncated_dynamic, dynamic_header + 40,
            dynamic_program[6] - 16,
        )
        mutations.append(truncated_dynamic)

        allocated_index = next(
            index for index, section in enumerate(sections)
            if section[2] & 2 and section[5] > 0 and section[8] > 0
        )
        allocated_section = sections[allocated_index]
        misplaced_allocated = bytearray(source)
        allocated_header = struct.unpack_from(
            "<Q", source, 40
        )[0] + allocated_index * 64
        struct.pack_into(
            "<Q", misplaced_allocated, allocated_header + 16,
            allocated_section[3] + allocated_section[8],
        )
        mutations.append(misplaced_allocated)

        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ldos.EvidenceError):
                self.parse_bytes(bytes(mutation))

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            invalid = root / "invalid-libnvidia-ml.so.1"
            valid = root / "valid-libnvidia-ml.so.1"
            invalid.write_bytes(mutations[1])
            valid.write_bytes(source)
            before = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            with (
                mock.patch.object(
                    ldos, "_nvml_library_candidates", return_value=(invalid, valid)
                ),
                mock.patch.object(
                    ldos, "_system_runtime_tcb_handle", return_value={"trusted": True}
                ),
            ):
                selected = ldos._resolve_nvml()
            try:
                self.assertEqual(selected.path, valid.resolve())
                os.fstat(selected.descriptor)
            finally:
                selected.close()
            after = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            self.assertEqual(after, before)

    def test_nvml_unloadable_dynamic_pointer_is_rejected_before_selection(self) -> None:
        source = self.real_nvml_bytes()
        _section_offset, sections = self.elf64_sections(source)
        dynamic = next(section for section in sections if section[1] == 6)
        string_table_entry = next(
            offset
            for offset in range(dynamic[4], dynamic[4] + dynamic[5], 16)
            if struct.unpack_from("<q", source, offset)[0] == 5
        )
        unloadable = bytearray(source)
        struct.pack_into("<Q", unloadable, string_table_entry + 8, (1 << 64) - 4096)

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            invalid = root / "invalid-libnvidia-ml.so.1"
            valid = root / "valid-libnvidia-ml.so.1"
            invalid.write_bytes(unloadable)
            valid.write_bytes(source)

            held_invalid = ldos.StableFile.open("unloadable NVML", invalid)
            try:
                # The offline identity remains useful but cannot establish
                # everything the native dynamic loader will consume.
                identity = ldos._nvml_elf_identity(
                    held_invalid.descriptor, held_invalid.size_bytes
                )
                self.assertEqual(identity["soname"], "libnvidia-ml.so.1")
                with self.assertRaisesRegex(ldos.EvidenceError, "load"):
                    ldos._probe_nvml_loadability(held_invalid)
            finally:
                held_invalid.close()

            before = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            with (
                mock.patch.object(
                    ldos, "_nvml_library_candidates", return_value=(invalid, valid)
                ),
                mock.patch.object(
                    ldos, "_system_runtime_tcb_handle", return_value={"trusted": True}
                ),
            ):
                selected = ldos._resolve_nvml()
            try:
                self.assertEqual(selected.path, valid.resolve())
                os.fstat(selected.descriptor)
            finally:
                selected.close()
            after = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            self.assertEqual(after, before)

    def test_nvidia_smi_missing_interp_is_rejected_before_selection(self) -> None:
        handle = ldos._resolve_nvidia_smi()
        try:
            source = bytearray(os.pread(handle.descriptor, handle.size_bytes, 0))
        finally:
            handle.close()
        program_offset, programs = self.elf64_programs(source)
        interpreter_index = next(
            index for index, program in enumerate(programs) if program[0] == 3
        )
        interpreter = programs[interpreter_index]
        self.assertGreater(interpreter[5], 2)
        missing_path = ("/" + "x" * (interpreter[5] - 2)).encode("ascii") + b"\0"
        self.assertEqual(len(missing_path), interpreter[5])
        invalid_bytes = bytearray(source)
        invalid_bytes[
            interpreter[2]:interpreter[2] + interpreter[5]
        ] = missing_path

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            invalid = root / "invalid-nvidia-smi"
            valid = root / "valid-nvidia-smi"
            invalid.write_bytes(invalid_bytes)
            valid.write_bytes(source)
            invalid.chmod(0o755)
            valid.chmod(0o755)
            descriptor = os.open(invalid, os.O_RDONLY)
            try:
                identity = ldos._elf_identity(descriptor, len(invalid_bytes))
            finally:
                os.close(descriptor)
            self.assertEqual(identity["interpreter_path"], missing_path[:-1].decode())

            before = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            with (
                mock.patch.object(
                    ldos, "_nvidia_smi_candidates", return_value=(invalid, valid)
                ),
                mock.patch.object(
                    ldos, "_trusted_system_tool_tcb_handle",
                    return_value={"trusted": True},
                ),
            ):
                selected = ldos._resolve_nvidia_smi()
            try:
                self.assertEqual(selected.path, valid.resolve())
                self.assertEqual(len(selected.companions), 1)
                os.fstat(selected.descriptor)
                os.fstat(selected.companions[0].descriptor)
            finally:
                selected.close()
            after = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            self.assertEqual(after, before)

    def test_nvidia_smi_exit_zero_nonloader_interp_cannot_shadow_valid_tool(self) -> None:
        handle = ldos._resolve_nvidia_smi()
        try:
            source = bytearray(os.pread(handle.descriptor, handle.size_bytes, 0))
        finally:
            handle.close()
        program_offset, programs = self.elf64_programs(source)
        interpreter_index = next(
            index for index, program in enumerate(programs) if program[0] == 3
        )
        interpreter = programs[interpreter_index]
        true_path = pathlib.Path("/usr/bin/true").resolve(strict=True)
        true_payload = str(true_path).encode("ascii") + b"\0"
        self.assertLessEqual(len(true_payload), interpreter[5])
        invalid_bytes = bytearray(source)
        invalid_bytes[
            interpreter[2]:interpreter[2] + len(true_payload)
        ] = true_payload
        interpreter_header = program_offset + interpreter_index * 56
        struct.pack_into(
            "<Q", invalid_bytes, interpreter_header + 32, len(true_payload)
        )
        struct.pack_into(
            "<Q", invalid_bytes, interpreter_header + 40, len(true_payload)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            invalid = root / "invalid-nvidia-smi"
            valid = root / "valid-nvidia-smi"
            invalid.write_bytes(invalid_bytes)
            valid.write_bytes(source)
            invalid.chmod(0o755)
            valid.chmod(0o755)

            invalid_handle = ldos.StableFile.open("invalid tool", invalid)
            true_handle = ldos.StableFile.open("exit-zero nonloader", true_path)
            invalid_handle.companions = (true_handle,)
            try:
                with self.assertRaisesRegex(ldos.EvidenceError, "startup"):
                    ldos._probe_nvidia_smi_startup(invalid_handle)
            finally:
                invalid_handle.close()

            before = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            with (
                mock.patch.object(
                    ldos, "_nvidia_smi_candidates", return_value=(invalid, valid)
                ),
                mock.patch.object(
                    ldos, "_trusted_system_tool_tcb_handle",
                    return_value={"trusted": True},
                ),
            ):
                selected = ldos._resolve_nvidia_smi()
            try:
                self.assertEqual(selected.path, valid.resolve())
                self.assertEqual(len(selected.companions), 1)
                self.assertNotEqual(selected.companions[0].path, true_path)
            finally:
                selected.close()
            after = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            self.assertEqual(after, before)

    def test_nvml_probe_timeout_kills_child_before_setsid_handshake(self) -> None:
        handle = ldos._resolve_nvml()
        try:
            before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            before_children = ldos._process_descendants(os.getpid())
            started = time.monotonic()
            real_waitpid = os.waitpid
            interrupted = False

            def interrupt_first_waitpid(pid, options):
                nonlocal interrupted
                if not interrupted:
                    interrupted = True
                    raise InterruptedError()
                return real_waitpid(pid, options)

            with (
                mock.patch.object(
                    ldos, "SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS", 0.1
                ),
                mock.patch.object(
                    ldos.os, "setsid", side_effect=lambda: time.sleep(1.0)
                ),
                mock.patch.object(
                    ldos.os, "waitpid", side_effect=interrupt_first_waitpid
                ),
                self.assertRaisesRegex(ldos.EvidenceError, "timed out"),
            ):
                ldos._probe_nvml_loadability(handle)
            elapsed = time.monotonic() - started
            self.assertTrue(interrupted)
            self.assertLess(elapsed, 0.6)
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
            self.assertEqual(ldos._process_descendants(os.getpid()), before_children)
        finally:
            handle.close()

    def test_nvml_probe_persistent_waitpid_interrupt_is_bounded(self) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            def slow_loader(_path, **_kwargs):
                time.sleep(1.0)
                return RequiredSymbols()

            before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            before_children = ldos._process_descendants(os.getpid())
            before_subreaper = ldos._child_subreaper_state()
            started = time.monotonic()
            with (
                mock.patch.object(
                    ldos, "SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS", 0.1
                ),
                mock.patch.object(
                    ldos.os,
                    "waitpid",
                    side_effect=InterruptedError("persistent injected EINTR"),
                ),
                self.assertRaisesRegex(ldos.EvidenceError, "timed out"),
            ):
                ldos._probe_nvml_loadability(
                    handle, cdll_factory=slow_loader
                )
            self.assertLess(time.monotonic() - started, 0.6)
            self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
            self.assertEqual(ldos._process_descendants(os.getpid()), before_children)
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
        finally:
            handle.close()

    def test_nvml_probe_cleanup_clock_baseexception_cannot_skip_finalizer(
        self,
    ) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            real_monotonic = time.monotonic
            cleanup_clock_failed = False

            def fail_cleanup_clock_once():
                nonlocal cleanup_clock_failed
                caller = sys._getframe(1)
                while (
                    caller is not None
                    and caller.f_code.co_name != "_probe_nvml_loadability"
                ):
                    caller = caller.f_back
                if (
                    caller is not None
                    and caller.f_locals.get("primary_error") is not None
                    and not cleanup_clock_failed
                ):
                    cleanup_clock_failed = True
                    raise RuntimeError("injected cleanup clock failure")
                return real_monotonic()

            def slow_loader(_path, **_kwargs):
                time.sleep(1.0)
                return RequiredSymbols()

            before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            before_children = ldos._process_descendants(os.getpid())
            before_subreaper = ldos._child_subreaper_state()
            with (
                mock.patch.object(
                    ldos, "SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS", 0.1
                ),
                mock.patch.object(
                    ldos.time, "monotonic", side_effect=fail_cleanup_clock_once
                ),
                self.assertRaisesRegex(ldos.EvidenceError, "timed out"),
            ):
                ldos._probe_nvml_loadability(
                    handle, cdll_factory=slow_loader
                )
            self.assertTrue(cleanup_clock_failed)
            self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
            self.assertIsNone(ldos._UNRESOLVED_SUBREAPER_TARGET)
            self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
            self.assertEqual(ldos._process_descendants(os.getpid()), before_children)
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
        finally:
            handle.close()

    def test_nvml_probe_persistent_exact_pidfd_close_uses_independent_close(
        self,
    ) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            real_pidfd_open = ldos._pidfd_open
            real_close = os.close
            exact_pidfd: int | None = None
            targeted_close_failures = 0

            def track_exact_pidfd(pid, **kwargs):
                nonlocal exact_pidfd
                descriptor = real_pidfd_open(pid, **kwargs)
                if exact_pidfd is None:
                    exact_pidfd = descriptor
                return descriptor

            def fail_exact_pidfd_close(descriptor):
                nonlocal targeted_close_failures
                if exact_pidfd is not None and descriptor == exact_pidfd:
                    targeted_close_failures += 1
                    raise RuntimeError("persistent exact pidfd close failure")
                return real_close(descriptor)

            before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            before_children = ldos._process_descendants(os.getpid())
            before_subreaper = ldos._child_subreaper_state()
            with (
                mock.patch.object(
                    ldos, "_pidfd_open", side_effect=track_exact_pidfd
                ),
                mock.patch.object(
                    ldos.os, "close", side_effect=fail_exact_pidfd_close
                ),
                self.assertRaisesRegex(
                    ldos.EvidenceError, "child pidfd close|cleanup"
                ),
            ):
                ldos._probe_nvml_loadability(
                    handle, cdll_factory=lambda *_args, **_kwargs: RequiredSymbols()
                )
            self.assertIsNotNone(exact_pidfd)
            self.assertGreater(targeted_close_failures, 0)
            with self.assertRaises(OSError) as closed:
                os.fstat(exact_pidfd)
            self.assertEqual(closed.exception.errno, errno.EBADF)
            self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
            self.assertEqual(ldos._process_descendants(os.getpid()), before_children)
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
        finally:
            handle.close()

    def test_nvml_probe_retries_transient_exact_child_termination_failures(
        self,
    ) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            real_pidfd_send_signal = ldos._pidfd_send_signal
            real_kill = os.kill
            real_killpg = os.killpg
            pidfd_failed = False
            direct_failed = False
            group_failed = False

            def fail_first_pidfd_signal(descriptor, sig):
                nonlocal pidfd_failed
                if not pidfd_failed:
                    pidfd_failed = True
                    raise RuntimeError("injected exact pidfd kill failure")
                return real_pidfd_send_signal(descriptor, sig)

            def fail_first_direct_kill(pid, sig):
                nonlocal direct_failed
                if not direct_failed:
                    direct_failed = True
                    raise RuntimeError("injected exact direct kill failure")
                return real_kill(pid, sig)

            def fail_first_group_kill(pid, sig):
                nonlocal group_failed
                if not group_failed:
                    group_failed = True
                    raise RuntimeError("injected group kill failure")
                return real_killpg(pid, sig)

            def slow_loader(_path, **_kwargs):
                time.sleep(1.0)
                return RequiredSymbols()

            before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            before_children = ldos._process_descendants(os.getpid())
            before_subreaper = ldos._child_subreaper_state()
            started = time.monotonic()
            with (
                mock.patch.object(
                    ldos, "SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS", 0.1
                ),
                mock.patch.object(
                    ldos,
                    "_pidfd_send_signal",
                    side_effect=fail_first_pidfd_signal,
                ),
                mock.patch.object(
                    ldos.os, "kill", side_effect=fail_first_direct_kill
                ),
                mock.patch.object(
                    ldos.os, "killpg", side_effect=fail_first_group_kill
                ),
                self.assertRaisesRegex(ldos.EvidenceError, "timed out"),
            ):
                ldos._probe_nvml_loadability(
                    handle, cdll_factory=slow_loader
                )
            elapsed = time.monotonic() - started
            self.assertTrue(pidfd_failed)
            self.assertTrue(direct_failed)
            self.assertTrue(group_failed)
            self.assertLess(elapsed, 0.6)
            self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
            self.assertEqual(ldos._process_descendants(os.getpid()), before_children)
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
        finally:
            handle.close()

    def test_nvml_probe_subreaper_contains_double_fork_new_session(self) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            with tempfile.TemporaryDirectory() as directory:
                escaped_pid_path = pathlib.Path(directory) / "escaped-pid"

                def double_fork_loader(_path, **_kwargs):
                    intermediate = os.fork()
                    if intermediate == 0:
                        daemon = os.fork()
                        if daemon == 0:
                            os.setsid()
                            escaped_pid_path.write_text(
                                str(os.getpid()), encoding="ascii"
                            )
                            time.sleep(60)
                            os._exit(0)
                        os._exit(0)
                    while True:
                        try:
                            os.waitpid(intermediate, 0)
                            break
                        except InterruptedError:
                            continue
                    return RequiredSymbols()

                before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
                before_children = ldos._process_descendants(os.getpid())
                with self.assertRaisesRegex(
                    ldos.EvidenceError, "contained descendant"
                ):
                    ldos._probe_nvml_loadability(
                        handle, cdll_factory=double_fork_loader
                    )
                self.assertTrue(escaped_pid_path.is_file())
                escaped_pid = int(escaped_pid_path.read_text(encoding="ascii"))
                self.assertFalse(pathlib.Path(f"/proc/{escaped_pid}").exists())
                self.assertEqual(
                    ldos._process_descendants(os.getpid()), before_children
                )
                self.assertEqual(
                    len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
                )
        finally:
            handle.close()

    def test_nvml_probe_containment_sleep_baseexception_runs_finalizer(
        self,
    ) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            with tempfile.TemporaryDirectory() as directory:
                escaped_pid_path = pathlib.Path(directory) / "escaped-pid"

                def double_fork_loader(_path, **_kwargs):
                    intermediate = os.fork()
                    if intermediate == 0:
                        daemon = os.fork()
                        if daemon == 0:
                            os.setsid()
                            escaped_pid_path.write_text(
                                str(os.getpid()), encoding="ascii"
                            )
                            time.sleep(60)
                            os._exit(0)
                        os._exit(0)
                    while True:
                        try:
                            os.waitpid(intermediate, 0)
                            break
                        except InterruptedError:
                            continue
                    return RequiredSymbols()

                real_containment = ldos._contain_subreaper_descendants
                real_sleep = time.sleep
                controller_pid = os.getpid()
                inside_containment = False
                sleep_failed = False

                def mark_containment(*args, **kwargs):
                    nonlocal inside_containment
                    inside_containment = True
                    try:
                        return real_containment(*args, **kwargs)
                    finally:
                        inside_containment = False

                def fail_containment_sleep_once(seconds):
                    nonlocal sleep_failed
                    if (
                        os.getpid() == controller_pid
                        and inside_containment
                        and seconds == 0.005
                        and not sleep_failed
                    ):
                        sleep_failed = True
                        raise RuntimeError("injected containment sleep failure")
                    return real_sleep(seconds)

                before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
                before_children = ldos._process_descendants(os.getpid())
                before_subreaper = ldos._child_subreaper_state()
                with (
                    mock.patch.object(
                        ldos,
                        "_contain_subreaper_descendants",
                        side_effect=mark_containment,
                    ),
                    mock.patch.object(
                        ldos.time, "sleep", side_effect=fail_containment_sleep_once
                    ),
                    self.assertRaisesRegex(
                        ldos.EvidenceError, "contained descendant|sleep"
                    ),
                ):
                    ldos._probe_nvml_loadability(
                        handle, cdll_factory=double_fork_loader
                    )
                self.assertTrue(sleep_failed)
                escaped_pid = int(escaped_pid_path.read_text(encoding="ascii"))
                self.assertFalse(pathlib.Path(f"/proc/{escaped_pid}").exists())
                self.assertFalse(ldos._RETAINED_PROBE_DESCRIPTORS)
                self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
                self.assertEqual(
                    ldos._process_descendants(os.getpid()), before_children
                )
                self.assertEqual(
                    len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
                )
        finally:
            handle.close()

    def test_nvml_probe_retries_and_verifies_subreaper_restore(self) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            real_set_subreaper = ldos._set_child_subreaper
            restore_failed = False

            def fail_first_restore(enabled):
                nonlocal restore_failed
                if not enabled and not restore_failed:
                    restore_failed = True
                    raise RuntimeError("injected subreaper restore failure")
                return real_set_subreaper(enabled)

            before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            before_children = ldos._process_descendants(os.getpid())
            before_subreaper = ldos._child_subreaper_state()
            with (
                mock.patch.object(
                    ldos, "_set_child_subreaper", side_effect=fail_first_restore
                ),
                self.assertRaisesRegex(
                    ldos.EvidenceError, "subreaper restore|cleanup"
                ),
            ):
                ldos._probe_nvml_loadability(
                    handle,
                    cdll_factory=lambda *_args, **_kwargs: RequiredSymbols(),
                )
            self.assertTrue(restore_failed)
            self.assertIsNone(ldos._UNRESOLVED_SUBREAPER_TARGET)
            self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
            self.assertEqual(ldos._process_descendants(os.getpid()), before_children)
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
        finally:
            handle.close()

    def test_nvml_probe_restores_after_enable_changes_state_then_raises(
        self,
    ) -> None:
        handle = ldos._resolve_nvml()
        try:
            real_set_subreaper = ldos._set_child_subreaper
            enable_failed = False

            def set_then_fail(enabled):
                nonlocal enable_failed
                if enabled and not enable_failed:
                    enable_failed = True
                    real_set_subreaper(True)
                    raise RuntimeError("injected post-enable failure")
                return real_set_subreaper(enabled)

            before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            before_children = ldos._process_descendants(os.getpid())
            before_subreaper = ldos._child_subreaper_state()
            with (
                mock.patch.object(
                    ldos, "_set_child_subreaper", side_effect=set_then_fail
                ),
                self.assertRaisesRegex(RuntimeError, "post-enable failure"),
            ):
                ldos._probe_nvml_loadability(handle)
            self.assertTrue(enable_failed)
            self.assertIsNone(ldos._UNRESOLVED_SUBREAPER_TARGET)
            self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
            self.assertEqual(ldos._process_descendants(os.getpid()), before_children)
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
        finally:
            handle.close()

    def test_nvml_probe_retries_interrupted_readiness_read_without_leak(self) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            real_read = os.read
            interrupted = False

            def interrupt_first_read(descriptor, size):
                nonlocal interrupted
                if not interrupted:
                    interrupted = True
                    raise InterruptedError()
                return real_read(descriptor, size)

            def slow_loader(_path, **_kwargs):
                time.sleep(0.05)
                return RequiredSymbols()

            before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            before_children = ldos._process_descendants(os.getpid())
            with mock.patch.object(
                ldos.os, "read", side_effect=interrupt_first_read
            ):
                ldos._probe_nvml_loadability(
                    handle, cdll_factory=slow_loader
                )
            self.assertTrue(interrupted)
            self.assertEqual(ldos._process_descendants(os.getpid()), before_children)
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
        finally:
            handle.close()

    def test_nvml_probe_reaps_adopted_child_when_pidfd_open_fails(self) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            with tempfile.TemporaryDirectory() as directory:
                escaped_pid_path = pathlib.Path(directory) / "escaped-pid"

                def double_fork_loader(_path, **_kwargs):
                    intermediate = os.fork()
                    if intermediate == 0:
                        daemon = os.fork()
                        if daemon == 0:
                            os.setsid()
                            escaped_pid_path.write_text(
                                str(os.getpid()), encoding="ascii"
                            )
                            time.sleep(60)
                            os._exit(0)
                        deadline = time.monotonic() + 2.0
                        while (
                            not escaped_pid_path.is_file()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.001)
                        os._exit(0)
                    while True:
                        try:
                            os.waitpid(intermediate, 0)
                            break
                        except InterruptedError:
                            continue
                    return RequiredSymbols()

                real_pidfd_open = ldos._pidfd_open
                pidfd_calls = 0

                def fail_descendant_pidfd(pid, **kwargs):
                    nonlocal pidfd_calls
                    pidfd_calls += 1
                    if pidfd_calls > 1:
                        raise OSError(errno.EMFILE, "injected descendant pidfd")
                    return real_pidfd_open(pid, **kwargs)

                before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
                before_children = ldos._process_descendants(os.getpid())
                before_subreaper = ldos._child_subreaper_state()
                with (
                    mock.patch.object(
                        ldos, "_pidfd_open", side_effect=fail_descendant_pidfd
                    ),
                    self.assertRaisesRegex(ldos.EvidenceError, "pidfd_open|cleanup"),
                ):
                    ldos._probe_nvml_loadability(
                        handle, cdll_factory=double_fork_loader
                    )
                self.assertGreater(pidfd_calls, 1)
                self.assertTrue(escaped_pid_path.is_file())
                escaped_pid = int(escaped_pid_path.read_text(encoding="ascii"))
                self.assertFalse(pathlib.Path(f"/proc/{escaped_pid}").exists())
                self.assertEqual(
                    ldos._child_subreaper_state(), before_subreaper
                )
                self.assertEqual(
                    ldos._process_descendants(os.getpid()), before_children
                )
                self.assertEqual(
                    len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
                )
        finally:
            handle.close()

    def test_nvml_probe_fallback_enumeration_contains_detached_child(self) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            with tempfile.TemporaryDirectory() as directory:
                escaped_pid_path = pathlib.Path(directory) / "escaped-pid"

                def double_fork_loader(_path, **_kwargs):
                    intermediate = os.fork()
                    if intermediate == 0:
                        daemon = os.fork()
                        if daemon == 0:
                            os.setsid()
                            escaped_pid_path.write_text(
                                str(os.getpid()), encoding="ascii"
                            )
                            time.sleep(60)
                            os._exit(0)
                        deadline = time.monotonic() + 2.0
                        while (
                            not escaped_pid_path.is_file()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.001)
                        os._exit(0)
                    while True:
                        try:
                            os.waitpid(intermediate, 0)
                            break
                        except InterruptedError:
                            continue
                    return RequiredSymbols()

                real_process_children = ldos._process_children
                enumeration_calls = 0

                def fail_after_baseline(pid):
                    nonlocal enumeration_calls
                    enumeration_calls += 1
                    if enumeration_calls > 1:
                        raise RuntimeError("persistent primary enumeration failure")
                    return real_process_children(pid)

                before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
                before_children = ldos._process_descendants(os.getpid())
                before_subreaper = ldos._child_subreaper_state()
                with (
                    mock.patch.object(
                        ldos,
                        "_process_children",
                        side_effect=fail_after_baseline,
                    ),
                    self.assertRaisesRegex(
                        ldos.EvidenceError, "enumeration|cleanup"
                    ),
                ):
                    ldos._probe_nvml_loadability(
                        handle, cdll_factory=double_fork_loader
                    )
                self.assertGreater(enumeration_calls, 1)
                self.assertTrue(escaped_pid_path.is_file())
                escaped_pid = int(escaped_pid_path.read_text(encoding="ascii"))
                self.assertFalse(pathlib.Path(f"/proc/{escaped_pid}").exists())
                self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
                self.assertEqual(
                    ldos._process_descendants(os.getpid()), before_children
                )
                self.assertEqual(
                    len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
                )
        finally:
            handle.close()

    def test_nvml_probe_persistent_descendant_pidfd_close_uses_independent_close(
        self,
    ) -> None:
        class RequiredSymbols:
            def __getattr__(self, _name):
                return object()

        handle = ldos._resolve_nvml()
        try:
            with tempfile.TemporaryDirectory() as directory:
                escaped_pid_path = pathlib.Path(directory) / "escaped-pid"

                def double_fork_loader(_path, **_kwargs):
                    intermediate = os.fork()
                    if intermediate == 0:
                        daemon = os.fork()
                        if daemon == 0:
                            os.setsid()
                            escaped_pid_path.write_text(
                                str(os.getpid()), encoding="ascii"
                            )
                            time.sleep(60)
                            os._exit(0)
                        deadline = time.monotonic() + 2.0
                        while (
                            not escaped_pid_path.is_file()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.001)
                        os._exit(0)
                    while True:
                        try:
                            os.waitpid(intermediate, 0)
                            break
                        except InterruptedError:
                            continue
                    return RequiredSymbols()

                real_pidfd_open = ldos._pidfd_open
                real_close = os.close
                opened_pidfds: list[int] = []
                descendant_close_failures = 0

                def track_pidfd(pid, **kwargs):
                    descriptor = real_pidfd_open(pid, **kwargs)
                    opened_pidfds.append(descriptor)
                    return descriptor

                def fail_descendant_close(descriptor):
                    nonlocal descendant_close_failures
                    if (
                        len(opened_pidfds) > 1
                        and descriptor in opened_pidfds[1:]
                    ):
                        descendant_close_failures += 1
                        raise RuntimeError("injected descendant pidfd close failure")
                    return real_close(descriptor)

                before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
                before_children = ldos._process_descendants(os.getpid())
                before_subreaper = ldos._child_subreaper_state()
                with (
                    mock.patch.object(
                        ldos, "_pidfd_open", side_effect=track_pidfd
                    ),
                    mock.patch.object(
                        ldos.os, "close", side_effect=fail_descendant_close
                    ),
                    self.assertRaisesRegex(
                        ldos.EvidenceError, "pidfd close|cleanup"
                    ),
                ):
                    ldos._probe_nvml_loadability(
                        handle, cdll_factory=double_fork_loader
                    )
                self.assertGreater(descendant_close_failures, 0)
                self.assertTrue(escaped_pid_path.is_file())
                escaped_pid = int(escaped_pid_path.read_text(encoding="ascii"))
                self.assertFalse(pathlib.Path(f"/proc/{escaped_pid}").exists())
                self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
                self.assertEqual(
                    ldos._process_descendants(os.getpid()), before_children
                )
                self.assertEqual(
                    len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
                )
        finally:
            handle.close()

    def test_nvml_probe_persistent_read_baseexception_cannot_skip_cleanup(self) -> None:
        handle = ldos._resolve_nvml()
        try:
            before_fds = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            before_children = ldos._process_descendants(os.getpid())
            before_subreaper = ldos._child_subreaper_state()
            with (
                mock.patch.object(
                    ldos.os,
                    "read",
                    side_effect=RuntimeError("persistent readiness failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "persistent readiness failure"),
            ):
                ldos._probe_nvml_loadability(handle)
            self.assertEqual(ldos._child_subreaper_state(), before_subreaper)
            self.assertEqual(ldos._process_descendants(os.getpid()), before_children)
            self.assertEqual(
                len(tuple(pathlib.Path("/proc/self/fd").iterdir())), before_fds
            )
        finally:
            handle.close()

    def test_nvml_resolver_skips_stale_untrusted_and_wrong_elf_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            stale = root / "stale.so.1"
            untrusted = root / "untrusted.so.1"
            wrong_elf = root / "wrong-elf.so.1"
            wrong_class = root / "wrong-class.so.1"
            wrong_encoding = root / "wrong-encoding.so.1"
            valid = root / "valid.so.1"
            for path in (
                untrusted, wrong_elf, wrong_class, wrong_encoding, valid
            ):
                path.write_bytes(b"x" * 64)

            def trust(handle):
                return (
                    None
                    if handle.path.name == "untrusted.so.1"
                    else {"trusted": True}
                )

            with (
                mock.patch.object(
                    ldos,
                    "_nvml_library_candidates",
                    return_value=(
                        stale, untrusted, wrong_elf, wrong_class,
                        wrong_encoding, valid,
                    ),
                ),
                mock.patch.object(
                    ldos, "_system_runtime_tcb_handle", side_effect=trust
                ),
                mock.patch.object(
                    ldos,
                    "_nvml_elf_identity",
                    side_effect=(
                        ldos.EvidenceError("wrong ELF"),
                        {
                            "machine": ldos._normalized_machine(),
                            "elf_class_bits": 32
                            if struct.calcsize("P") == 8 else 64,
                            "endianness": sys.byteorder,
                        },
                        {
                            "machine": ldos._normalized_machine(),
                            "elf_class_bits": struct.calcsize("P") * 8,
                            "endianness": "big"
                            if sys.byteorder == "little" else "little",
                        },
                        {
                            "machine": ldos._normalized_machine(),
                            "elf_class_bits": struct.calcsize("P") * 8,
                            "endianness": sys.byteorder,
                        },
                    ),
                ),
                mock.patch.object(ldos, "_probe_nvml_loadability"),
            ):
                selected = ldos._resolve_nvml()
                try:
                    self.assertEqual(selected.path, valid.resolve())
                    os.fstat(selected.descriptor)
                finally:
                    selected.close()

    def test_nvidia_smi_resolver_is_bounded_and_skips_untrusted_candidate(self) -> None:
        self.assertEqual(
            tuple(str(path) for path in ldos.NVIDIA_SMI_CANDIDATES),
            (
                "/usr/bin/nvidia-smi",
                "/usr/local/bin/nvidia-smi",
                "/usr/lib/wsl/lib/nvidia-smi",
            ),
        )
        self.assertEqual(
            tuple(str(path) for path in ldos._nvidia_smi_candidates(True)),
            (
                "/usr/lib/wsl/lib/nvidia-smi",
                "/usr/bin/nvidia-smi",
                "/usr/local/bin/nvidia-smi",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            absent = root / "absent-nvidia-smi"
            untrusted = root / "untrusted-nvidia-smi"
            wrong_arch = root / "wrong-arch-nvidia-smi"
            wrong_class = root / "wrong-class-nvidia-smi"
            wrong_encoding = root / "wrong-encoding-nvidia-smi"
            valid = root / "valid-nvidia-smi"
            for path in (
                untrusted, wrong_arch, wrong_class, wrong_encoding, valid
            ):
                path.write_bytes(b"x" * 64)
                path.chmod(0o755)
            with (
                mock.patch.object(
                    ldos,
                    "_nvidia_smi_candidates",
                    return_value=(
                        absent, untrusted, wrong_arch, wrong_class,
                        wrong_encoding, valid,
                    ),
                ),
                mock.patch.object(
                    ldos,
                    "_trusted_system_tool_tcb_handle",
                    side_effect=lambda handle: (
                        None
                        if handle.path.name.startswith("untrusted")
                        else {"trusted": True}
                    ),
                ),
                mock.patch.object(
                    ldos,
                    "_elf_identity",
                    side_effect=(
                        {
                            "machine": "aarch64",
                            "elf_class_bits": struct.calcsize("P") * 8,
                            "endianness": sys.byteorder,
                        },
                        {
                            "machine": ldos._normalized_machine(),
                            "elf_class_bits": 32
                            if struct.calcsize("P") == 8 else 64,
                            "endianness": sys.byteorder,
                        },
                        {
                            "machine": ldos._normalized_machine(),
                            "elf_class_bits": struct.calcsize("P") * 8,
                            "endianness": "big"
                            if sys.byteorder == "little" else "little",
                        },
                        {
                            "machine": ldos._normalized_machine(),
                            "elf_class_bits": struct.calcsize("P") * 8,
                            "endianness": sys.byteorder,
                        },
                    ),
                ),
                mock.patch.object(ldos, "_probe_nvidia_smi_startup"),
            ):
                selected = ldos._resolve_nvidia_smi()
                try:
                    self.assertEqual(selected.path, valid.resolve())
                    os.fstat(selected.descriptor)
                finally:
                    selected.close()

    def test_resolvers_close_every_rejected_candidate_handle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            candidates = tuple(root / f"candidate-{index}" for index in range(3))
            for candidate in candidates:
                candidate.write_bytes(b"x" * 64)
            before = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            with (
                mock.patch.object(
                    ldos, "_nvml_library_candidates", return_value=candidates
                ),
                mock.patch.object(
                    ldos, "_system_runtime_tcb_handle", return_value={"trusted": True}
                ),
                mock.patch.object(
                    ldos,
                    "_nvml_elf_identity",
                    side_effect=ldos.EvidenceError("invalid candidate"),
                ),
            ):
                with self.assertRaisesRegex(ldos.EvidenceError, "no bounded"):
                    ldos._resolve_nvml()
            after = len(tuple(pathlib.Path("/proc/self/fd").iterdir()))
            self.assertEqual(after, before)

    def test_truncated_and_oversized_elf_offsets_are_rejected(self) -> None:
        with self.assertRaisesRegex(ldos.EvidenceError, "size|truncated|ELF"):
            self.parse_bytes(b"\x7fELF" + b"\0" * 20)
        source = self.real_nvml_bytes()[:64]
        oversized_offset = bytearray(source)
        struct.pack_into("<Q", oversized_offset, 40, 1 << 63)
        with self.assertRaisesRegex(ldos.EvidenceError, "range"):
            self.parse_bytes(bytes(oversized_offset))
        oversized_count = bytearray(source)
        struct.pack_into("<H", oversized_count, 60, 8193)
        with self.assertRaisesRegex(ldos.EvidenceError, "header contract"):
            self.parse_bytes(bytes(oversized_count))

    def test_real_nvml_dynamic_semantics_stop_at_first_dt_null(self) -> None:
        source = self.real_nvml_bytes()
        _section_offset, sections = self.elf64_sections(source)
        dynamic = next(section for section in sections if section[1] == 6)
        entries = [
            (offset, *struct.unpack_from("<qQ", source, offset))
            for offset in range(dynamic[4], dynamic[4] + dynamic[5], 16)
        ]
        soname_entry = next(item for item in entries if item[1] == 14)
        first_null = next(item for item in entries if item[1] == 0)

        missing_pre_null = bytearray(source)
        struct.pack_into("<q", missing_pre_null, soname_entry[0], 0)
        with self.assertRaisesRegex(ldos.EvidenceError, "SONAME cardinality"):
            self.parse_bytes(bytes(missing_pre_null))

        unterminated = bytearray(source)
        for offset, tag, _value in entries:
            if tag == 0:
                struct.pack_into("<q", unterminated, offset, 1)
        with self.assertRaisesRegex(ldos.EvidenceError, "SONAME cardinality"):
            self.parse_bytes(bytes(unterminated))

        ignored_post_null = bytearray(source)
        self.assertLess(first_null[0] + 16, dynamic[4] + dynamic[5])
        struct.pack_into("<qQ", ignored_post_null, first_null[0] + 16, 14, 1 << 63)
        self.assertEqual(
            self.parse_bytes(bytes(ignored_post_null)),
            self.parse_bytes(bytes(source)),
        )

    def test_real_nvml_required_symbol_invalid_indices_are_rejected(self) -> None:
        original = self.real_nvml_bytes()
        _section_offset, sections = self.elf64_sections(original)
        symbols = next(section for section in sections if section[1] == 11)
        strings = sections[symbols[6]]
        table = original[strings[4]:strings[4] + strings[5]]
        target_offset = None
        for offset in range(symbols[4], symbols[4] + symbols[5], 24):
            name_offset = struct.unpack_from("<I", original, offset)[0]
            end = table.find(b"\0", name_offset)
            if table[name_offset:end] == b"nvmlInit_v2":
                target_offset = offset
                break
        self.assertIsNotNone(target_offset)
        for invalid_index in (0, 0xFF00, 0xFFFF):
            with self.subTest(section_index=hex(invalid_index)):
                source = bytearray(original)
                struct.pack_into("<H", source, target_offset + 6, invalid_index)
                with self.assertRaisesRegex(ldos.EvidenceError, "invalid section index"):
                    self.parse_bytes(bytes(source))

    def test_real_nvml_build_id_requires_canonical_gnu_name_encoding(self) -> None:
        original = self.real_nvml_bytes()
        _section_offset, sections = self.elf64_sections(original)
        note = next(section for section in sections if section[1] == 7)
        name_size, _descriptor_size, note_type = struct.unpack_from(
            "<III", original, note[4]
        )
        self.assertEqual((name_size, note_type), (4, 3))
        for mutation in ("namesz", "name"):
            with self.subTest(mutation=mutation):
                source = bytearray(original)
                if mutation == "namesz":
                    struct.pack_into("<I", source, note[4], 3)
                else:
                    source[note[4] + 12:note[4] + 16] = b"GNUx"
                with self.assertRaisesRegex(ldos.EvidenceError, "build-id"):
                    self.parse_bytes(bytes(source))

    def test_real_nvml_relevant_section_overlap_is_rejected(self) -> None:
        source = self.real_nvml_bytes()
        section_offset, sections = self.elf64_sections(source)
        note_index = next(index for index, section in enumerate(sections) if section[1] == 7)
        dynamic = next(section for section in sections if section[1] == 6)
        dynamic_strings = sections[dynamic[6]]
        struct.pack_into(
            "<Q", source, section_offset + note_index * 64 + 24,
            dynamic_strings[4],
        )
        struct.pack_into(
            "<Q", source, section_offset + note_index * 64 + 16,
            dynamic_strings[3],
        )
        with self.assertRaisesRegex(ldos.EvidenceError, "overlap"):
            self.parse_bytes(bytes(source))

    def test_real_nvml_aggregate_parse_work_budget_is_enforced(self) -> None:
        source = bytes(self.real_nvml_bytes())
        with mock.patch.object(ldos, "NVML_ELF_MAX_PARSE_WORK_BYTES", 1024):
            with self.assertRaisesRegex(ldos.EvidenceError, "parse-work budget"):
                self.parse_bytes(source)


class SystemRuntimeTcbTests(unittest.TestCase):
    def test_runtime_tcb_roots_include_distinct_lib64_layouts(self) -> None:
        self.assertEqual(
            tuple(str(path) for path in ldos.SYSTEM_RUNTIME_TCB_ROOTS),
            ("/usr/lib", "/lib", "/usr/lib64", "/lib64"),
        )

    def test_root_owned_system_runtime_is_bound_with_digest(self) -> None:
        candidates = (
            pathlib.Path("/lib/x86_64-linux-gnu/libc.so.6"),
            pathlib.Path("/usr/lib/x86_64-linux-gnu/libc.so.6"),
        )
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            self.skipTest("no system libc fixture")
        record = ldos._system_runtime_tcb(path)
        self.assertIsNotNone(record)
        self.assertEqual(record["source"], "immutable_system_runtime_tcb")
        self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")

    def test_writable_or_non_system_runtime_is_rejected(self) -> None:
        for mode in (stat.S_IFREG | 0o644, stat.S_IFREG | 0o664, stat.S_IFREG | 0o646):
            info = types.SimpleNamespace(st_mode=mode, st_uid=0)
            with self.subTest(mode=oct(mode)):
                self.assertEqual(
                    ldos._root_owned_immutable_regular(info),
                    not bool(mode & 0o022),
                )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "libhostile.so"
            path.write_bytes(b"hostile")
            self.assertIsNone(ldos._system_runtime_tcb(path))

    def test_selection_rejects_bad_missing_or_duplicate_uuid(self) -> None:
        inventory = ldos.parse_device_inventory(device_csv())
        good = inventory[0]["uuid"]
        for requested, message in (
            (good, "exactly two"),
            (f"{good},{good}", "distinct"),
            (f"{good},GPU-ffffffff-ffff-ffff-ffff-ffffffffffff", "not visible"),
            (f"{good},MIG-GPU-invalid", "exactly two"),
        ):
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(ldos.EvidenceError, message):
                    ldos.select_devices(inventory, requested)


class StableFileTests(unittest.TestCase):
    def test_content_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "artifact"
            path.write_bytes(b"before")
            handle = ldos.StableFile.open("artifact", path)
            try:
                path.write_bytes(b"after!")
                with self.assertRaisesRegex(ldos.EvidenceError, "changed"):
                    handle.verify()
            finally:
                handle.close()

    def test_path_replacement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = root / "artifact"
            old = root / "old"
            path.write_bytes(b"same")
            handle = ldos.StableFile.open("artifact", path)
            try:
                path.rename(old)
                path.write_bytes(b"same")
                with self.assertRaisesRegex(ldos.EvidenceError, "changed"):
                    handle.verify()
            finally:
                handle.close()

    def test_archived_copy_is_exact_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source_path = root / "source"
            destination = root / "copy"
            source_path.write_bytes(b"payload")
            source = ldos.StableFile.open("source", source_path)
            archived = None
            try:
                archived = ldos._copy_stable_file(source, destination, 0o400)
                self.assertEqual(archived.sha256, source.sha256)
                with self.assertRaises(FileExistsError):
                    ldos._copy_stable_file(source, destination, 0o400)
            finally:
                source.close()
                if archived is not None:
                    archived.close()

    def test_gate_publication_never_exposes_partial_final_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            final = pathlib.Path(directory) / "RELEASE"
            real_write = os.write
            observations = []

            def checked_write(descriptor, data):
                observations.append(final.exists())
                return real_write(descriptor, data[: max(1, len(data) // 2)])

            with mock.patch.object(ldos.os, "write", side_effect=checked_write):
                ldos._write_text_exclusive(final, "complete-release\n", 0o400)
            self.assertTrue(observations)
            self.assertFalse(any(observations))
            self.assertEqual(final.read_text(), "complete-release\n")
            self.assertEqual(stat.S_IMODE(final.stat().st_mode), 0o400)
            with self.assertRaises(ldos.EvidenceError):
                ldos._write_text_exclusive(final, "replacement\n", 0o400)

    def test_named_no_follow_open_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = root / "target"
            target.write_text("payload")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(ldos.EvidenceError, "exact regular file"):
                ldos.StableFile.open_named_no_follow(
                    "named", link, maximum_size_bytes=4096
                )

    def test_stable_timeout_executes_held_inode_with_semantic_argv0(self) -> None:
        system_timeout = pathlib.Path(shutil.which("timeout") or "/usr/bin/timeout")
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = root / "timeout"
            path.write_bytes(system_timeout.read_bytes())
            path.chmod(0o700)
            handle = ldos.StableFile.open("test timeout", path)
            try:
                path.rename(root / "replaced-timeout")
                path.write_text("not the timeout binary\n", encoding="utf-8")
                result = ldos._run_process(
                    command=["timeout", "--version"],
                    executable=handle.proc_path,
                    pass_fds=(handle.descriptor,),
                    environment={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                    cwd=root,
                    timeout_seconds=5,
                )
                self.assertEqual(result["returncode"], 0)
                self.assertEqual(result["command"][0], "timeout")
                self.assertEqual(result["executable"], handle.proc_path)
                with self.assertRaisesRegex(ldos.EvidenceError, "changed"):
                    handle.verify()
            finally:
                handle.close()

    def test_runtime_closure_handles_open_and_verify_without_controller_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            runtime = root / "libmpi.so"
            runtime.write_bytes(b"runtime")
            closure_path = root / "closure.json"
            closure_path.write_text(
                json.dumps(
                    {
                        "groups": {
                            "mpi": [
                                {
                                    "path": str(runtime),
                                    "sha256": ldos.EVIDENCE.sha256_file(runtime),
                                    "size_bytes": runtime.stat().st_size,
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            closure = ldos.StableFile.open("closure", closure_path)
            handles = {}
            try:
                handles = ldos.open_runtime_closure_handles(closure)
                self.assertEqual(set(handles), {str(runtime)})
                handles[str(runtime)].verify()
            finally:
                closure.close()
                for handle in handles.values():
                    handle.close()


class ProcessContainmentTests(unittest.TestCase):
    @staticmethod
    def orphan_program(pid_file: pathlib.Path) -> str:
        child = (
            "import os,time,pathlib;os.setsid();"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
            "time.sleep(30)"
        )
        return (
            "import subprocess,sys,time,pathlib;"
            f"p=subprocess.Popen([sys.executable,'-c',{child!r}],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
            f"path=pathlib.Path({str(pid_file)!r});"
            "deadline=time.time()+2;"
            "\nwhile not path.exists() and time.time()<deadline: time.sleep(0.01)"
        )

    def test_runtime_observer_failure_returns_logs_after_containment(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            ldos, "_observe_runtime_processes", side_effect=RuntimeError("observer boom")
        ):
            result = ldos._run_process(
                command=[
                    sys.executable, "-c",
                    "import sys,time;print('started',flush=True);time.sleep(30)",
                ],
                environment={"PATH": "/usr/bin:/bin"},
                cwd=pathlib.Path(directory),
                timeout_seconds=5,
                observe_runtime=True,
            )
        self.assertIn("observer boom", result["runtime_observer_error"])
        self.assertIsInstance(result["stdout"], str)
        self.assertIsInstance(result["stderr"], str)
        self.assertFalse(result["timed_out"])
        self.assertIsNone(result["containment_error"])

    def test_poll_callback_failure_is_returned_after_containment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = ldos._run_process(
                command=[sys.executable, "-c", "import time;time.sleep(30)"],
                environment={"PATH": "/usr/bin:/bin"},
                cwd=pathlib.Path(directory),
                timeout_seconds=5,
                poll_callback=lambda _pid: (_ for _ in ()).throw(
                    RuntimeError("injected telemetry failure")
                ),
            )
        self.assertFalse(result["timed_out"])
        self.assertIn("injected telemetry failure", result["callback_error"])
        self.assertTrue(result["monitoring_error_caused_termination"])
        self.assertIsNone(result["containment_error"])
        primary, _ = ldos._classify_lane_process_failure(result, ranks=1)
        self.assertIn("telemetry callback failed", str(primary))

    def test_dead_root_live_grandchild_is_contained_after_observer_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            pid_file = root / "orphan.pid"

            def delayed_failure(*_args):
                time.sleep(0.2)
                raise RuntimeError("observer after root exit")

            with mock.patch.object(
                ldos, "_observe_runtime_processes", side_effect=delayed_failure
            ):
                result = ldos._run_process(
                    command=[sys.executable, "-c", self.orphan_program(pid_file)],
                    environment={"PATH": "/usr/bin:/bin"},
                    cwd=root,
                    timeout_seconds=3,
                    observe_runtime=True,
                )
            self.assertIn("observer after root exit", result["runtime_observer_error"])
            self.assertTrue(pid_file.exists())
            orphan_pid = int(pid_file.read_text())
            self.assertFalse(pathlib.Path(f"/proc/{orphan_pid}").exists())

    def test_finally_contains_dead_root_descendant_when_leak_snapshot_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            pid_file = root / "orphan.pid"
            real_snapshot = ldos._snapshot_pidfds
            calls = 0

            def fail_first_snapshot(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("snapshot interrupted")
                return real_snapshot(*args, **kwargs)

            with mock.patch.object(
                ldos, "_snapshot_pidfds", side_effect=fail_first_snapshot
            ):
                with self.assertRaisesRegex(RuntimeError, "snapshot interrupted"):
                    ldos._run_process(
                        command=[sys.executable, "-c", self.orphan_program(pid_file)],
                        environment={"PATH": "/usr/bin:/bin"},
                        cwd=root,
                        timeout_seconds=3,
                    )
            self.assertGreaterEqual(calls, 2)
            self.assertTrue(pid_file.exists())
            orphan_pid = int(pid_file.read_text())
            self.assertFalse(pathlib.Path(f"/proc/{orphan_pid}").exists())

    def test_subreaper_is_restored_when_spawn_raises(self) -> None:
        transitions = []
        state = False

        def set_state(value):
            nonlocal state
            transitions.append(value)
            state = value

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                ldos, "_raw_child_subreaper_state", side_effect=lambda: state
            ),
            mock.patch.object(
                ldos, "_set_child_subreaper", side_effect=set_state
            ),
            mock.patch.object(ldos.subprocess, "Popen", side_effect=OSError("spawn")),
        ):
            with self.assertRaisesRegex(OSError, "spawn"):
                ldos._run_process(
                    command=["missing"], environment={}, cwd=pathlib.Path(directory),
                    timeout_seconds=1,
                )
        self.assertEqual(transitions, [True, False])

    def test_communicate_error_carries_and_persists_recovered_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "evidence"
            ldos._claim_output(output, "run")
            home = output / "home"
            home.mkdir()
            original_communicate = ldos.subprocess.Popen.communicate
            calls = 0

            def fail_first_communicate(process, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    time.sleep(0.1)
                    raise OSError("communicate injected")
                return original_communicate(process, *args, **kwargs)

            with mock.patch.object(
                ldos.subprocess.Popen,
                "communicate",
                autospec=True,
                side_effect=fail_first_communicate,
            ):
                with self.assertRaises(ldos.ProcessExecutionError) as caught:
                    ldos._run_process(
                        command=[
                            sys.executable, "-c",
                            "import sys;print('raw-before-failure',flush=True)",
                        ],
                        environment={"PATH": "/usr/bin:/bin"},
                        cwd=home,
                        timeout_seconds=2,
                    )
            error = caught.exception
            self.assertIn("raw-before-failure", error.result["stdout"])
            ldos._persist_failed_process_execution(output, "injected-process", error)
            ldos._publish_failure(output, "run", error)
            self.assertIn(
                "raw-before-failure",
                (output / "injected-process.stdout.log").read_text(),
            )
            failure = json.loads((output / "FAILED").read_text())
            paths = {record["path"] for record in failure["partial_artifacts"]}
            self.assertIn("injected-process.stdout.log", paths)

    def test_control_exception_flag_survives_containment_raise_after_termination(self) -> None:
        real_communicate = ldos.subprocess.Popen.communicate
        communicate_calls = 0
        containment_calls = 0

        def fail_first_communicate(process, *args, **kwargs):
            nonlocal communicate_calls
            communicate_calls += 1
            if communicate_calls == 1:
                raise OSError("injected process-control failure")
            return real_communicate(process, *args, **kwargs)

        def terminate_then_raise(process, _baseline, *, terminate):
            nonlocal containment_calls
            containment_calls += 1
            if containment_calls == 1:
                self.assertTrue(terminate)
                process.terminate()
                process.wait(timeout=2)
                raise RuntimeError("containment raised after termination")
            return set(), []

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                ldos.subprocess.Popen,
                "communicate",
                autospec=True,
                side_effect=fail_first_communicate,
            ),
            mock.patch.object(
                ldos, "_contain_process_tree", side_effect=terminate_then_raise
            ),
            self.assertRaises(ldos.ProcessExecutionError) as caught,
        ):
            ldos._run_process(
                command=[sys.executable, "-c", "import time;time.sleep(30)"],
                environment={"PATH": "/usr/bin:/bin"},
                cwd=pathlib.Path(directory),
                timeout_seconds=3,
            )
        result = caught.exception.result
        self.assertTrue(result["process_control_error_caused_termination"])
        self.assertTrue(
            any(
                "containment raised after termination" in note
                for note in caught.exception.original.__notes__
            )
        )

    def test_final_containment_failure_carries_completed_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            ldos,
            "_contain_process_tree",
            return_value=({999}, ["injected cleanup error"]),
        ):
            with self.assertRaises(ldos.ProcessExecutionError) as caught:
                ldos._run_process(
                    command=[
                        sys.executable, "-c",
                        "print('payload-log',flush=True)",
                    ],
                    environment={"PATH": "/usr/bin:/bin"},
                    cwd=pathlib.Path(directory),
                    timeout_seconds=2,
                )
            self.assertIn("payload-log", caught.exception.result["stdout"])
            self.assertIn("999", caught.exception.result["containment_error"])
    def test_timeout_reaps_setsid_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            pid_file = root / "descendant.pid"
            child = (
                "import os,time,pathlib;os.setsid();"
                f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
                "time.sleep(30)"
            )
            parent = (
                "import subprocess,sys,time;"
                f"subprocess.Popen([sys.executable,'-c',{child!r}],"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
                "time.sleep(30)"
            )
            result = ldos._run_process(
                command=[sys.executable, "-c", parent],
                environment={"PATH": "/usr/bin:/bin"},
                cwd=root,
                timeout_seconds=0.3,
            )
            self.assertTrue(result["timed_out"])
            self.assertIsNone(result["containment_error"])
            self.assertIn(result["returncode"], (-9, -15))
            self.assertTrue(pid_file.exists())
            pid = int(pid_file.read_text())
            self.assertFalse(pathlib.Path(f"/proc/{pid}").exists())

    def test_ctypes_pidfd_fallback_without_stdlib_attributes(self) -> None:
        with (
            mock.patch.object(ldos.os, "pidfd_open", None, create=True),
            mock.patch.object(ldos.signal, "pidfd_send_signal", None, create=True),
        ):
            descriptor = ldos._pidfd_open(os.getpid())
            try:
                ldos._pidfd_send_signal(descriptor, 0)
            finally:
                ldos._close_native_owned_descriptor_once(
                    "ctypes-fallback test pidfd", descriptor
                )

    def test_runtime_observer_captures_exec_and_late_dlopen_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source_library = pathlib.Path("/lib/x86_64-linux-gnu/libm.so.6")
            if not source_library.exists():
                self.skipTest("test requires the standard libm shared library")
            late_library = root / "libpmix.so.99"
            late_library.write_bytes(source_library.read_bytes())
            late_library.chmod(0o500)
            exec_payload = (
                "import ctypes,time;time.sleep(0.15);"
                f"ctypes.CDLL({str(late_library)!r});time.sleep(0.25)"
            )
            program = (
                "import ctypes,os,sys,time;"
                "pid=os.fork();"
                "\nif pid==0:"
                "\n time.sleep(0.15);"
                "\n os.execv(sys.executable,[sys.executable,'-c',"
                f"{exec_payload!r}]);"
                "\ntime.sleep(0.75)"
            )
            result = ldos._run_process(
                command=[sys.executable, "-c", program],
                environment={"PATH": "/usr/bin:/bin"},
                cwd=root,
                timeout_seconds=3,
                observe_runtime=True,
            )
            self.assertEqual(result["returncode"], 0, result["stderr"])
            by_pid: dict[int, list[dict]] = {}
            for observation in result["runtime_observations"]:
                by_pid.setdefault(observation["pid"], []).append(observation)
            self.assertTrue(any(len(items) >= 2 for items in by_pid.values()))
            self.assertTrue(
                any(
                    mapping["path"] == str(late_library)
                    for observation in result["runtime_observations"]
                    for mapping in observation["mappings"]
                )
            )
            self.assertGreaterEqual(len(result["runtime_observation_attempts"]), 2)

    def test_runtime_observer_hot_path_never_hashes_mapped_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            ldos, "_sha256_fd", side_effect=AssertionError("hot hash forbidden")
        ):
            result = ldos._run_process(
                command=[sys.executable, "-c", "import time;time.sleep(0.15)"],
                environment={"PATH": "/usr/bin:/bin"},
                cwd=pathlib.Path(directory),
                timeout_seconds=2,
                observe_runtime=True,
            )
        self.assertEqual(result["returncode"], 0)


class ControllerAffinityTests(unittest.TestCase):
    def test_restore_reaches_fixed_point_with_arriving_native_tid(self) -> None:
        one = {
            "start_time_ticks": 1, "ticks": 1,
            "task_affinities": {"10": [2]},
        }
        two = {
            "start_time_ticks": 1, "ticks": 2,
            "task_affinities": {"10": [0, 1], "11": [0, 1]},
        }
        states = [one, two, two, two, two, two]
        with (
            mock.patch.object(ldos, "_process_task_cpu_state", side_effect=states),
            mock.patch.object(ldos.os, "sched_setaffinity") as set_affinity,
        ):
            ldos._restore_process_task_affinities(
                10, {"10": [0, 1]}, {0, 1}
            )
        calls = [(call.args[0], set(call.args[1])) for call in set_affinity.call_args_list]
        self.assertIn((11, {0, 1}), calls)
        self.assertGreaterEqual(sum(tid == 11 for tid, _mask in calls), 2)

    def test_restore_fails_closed_when_native_tid_set_never_stabilizes(self) -> None:
        counter = 10

        def changing_state(_pid):
            nonlocal counter
            counter += 1
            return {
                "start_time_ticks": 1, "ticks": counter,
                "task_affinities": {str(counter): [0]},
            }

        with (
            mock.patch.object(ldos, "_process_task_cpu_state", side_effect=changing_state),
            mock.patch.object(ldos.os, "sched_setaffinity"),
        ):
            with self.assertRaisesRegex(ldos.EvidenceError, "fixed point"):
                ldos._restore_process_task_affinities(10, {}, {0, 1})


class RawTelemetryJournalTests(unittest.TestCase):
    @staticmethod
    def rewrite_rows(path: pathlib.Path, rows: list[dict]) -> None:
        for sequence, row in enumerate(rows):
            row["sequence"] = sequence
        path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    @staticmethod
    def refresh_ready_file(path: pathlib.Path, rows: list[dict]) -> None:
        ready = next(row["payload"] for row in rows if row["record_type"] == "ready")
        ordered_ready = {
            "schema_version": ready["schema_version"],
            "rank": ready["rank"],
            "pid": ready["pid"],
            "start_time_ticks": ready["start_time_ticks"],
            "device_ordinal": ready["device_ordinal"],
            "device_uuid": ready["device_uuid"],
            "nonce": ready["nonce"],
            "task_affinities": {
                tid: ready["task_affinities"][tid]
                for tid in sorted(ready["task_affinities"], key=int)
            },
        }
        data = json.dumps(
            ordered_ready, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        ready_path = path.parent / f"rank-{ready['rank']}.ready"
        ready_path.write_bytes(data)
        ready_path.chmod(0o600)
        info = ready_path.stat()
        ready["file"] = {
            "label": f"LDOS rank {ready['rank']} ready record",
            "path": str(ready_path),
            "size_bytes": len(data),
            "sha256": ldos.hashlib.sha256(data).hexdigest(),
            "fingerprint": {
                "device": info.st_dev,
                "inode": info.st_ino,
                "mode": info.st_mode,
                "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns,
            },
        }

    def make_valid_journal(self, root: pathlib.Path) -> tuple[pathlib.Path, dict]:
        gpu0, gpu1 = (
            "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "GPU-11111111-2222-3333-4444-555555555555",
        )
        source_nvml = ldos._resolve_nvml()
        archive_nvml = ldos._copy_stable_file(
            source_nvml,
            root / "archive" / "system" / "lib" / "libnvidia-ml.so.1",
            0o400,
        )
        try:
            nvml_archive_identity = ldos._nvml_archive_content_identity(
                archive_nvml
            )
        finally:
            source_nvml.close()
            archive_nvml.close()
        header = {
            "nonce": "a" * 64,
            "ranks": 1,
            "controller_cpu_mask": [0],
            "worker_cpu_masks": [[1]],
            "expected_device_uuids": [
                gpu0.removeprefix("GPU-").replace("-", ""),
                gpu1.removeprefix("GPU-").replace("-", ""),
            ],
            "online_cpu_list": "0-1",
            "sample_period_ns": 50_000_000,
            "departed_nvml_lag_limit_ns": ldos.NVML_DEPARTED_CONTEXT_LAG_NS,
            "nvml_absence_observation_to_pidfd_observation_limit_ns": (
                ldos.NVML_ABSENCE_OBSERVATION_TO_PIDFD_OBSERVATION_NS
            ),
            "allowed_cpus": [0, 1],
            "physical_topology_plan": {
                "allowed_cpus": [0, 1],
                "available_full_cores": [
                    {"package_id": 0, "core_id": 0, "logical_cpus": [0]},
                    {"package_id": 0, "core_id": 1, "logical_cpus": [1]},
                ],
                "controller_core": {
                    "package_id": 0, "core_id": 0, "logical_cpus": [0],
                },
                "worker_cores": [
                    {"package_id": 0, "core_id": 1, "logical_cpus": [1]},
                ],
            },
            "controller_pid": 40,
            "controller_start_time_ticks": 4,
            "nvml_archive_identity": nvml_archive_identity,
            "nvml_baseline_process_utilization": {gpu0: [], gpu1: []},
            "nvml_baseline_current_processes": {
                gpu0: {"compute": [], "graphics": [], "mps": []},
                gpu1: {"compute": [], "graphics": [], "mps": []},
            },
            "nvml_device_uuids": [gpu0, gpu1],
        }
        control = root / "ldos-rank-1-control"
        control.mkdir(mode=0o700)
        path = control / "telemetry.raw.jsonl"
        journal = ldos.RawTelemetryJournal(path, header)
        ready_payload = {
            "schema_version": 1,
            "rank": 0,
            "pid": 42,
            "start_time_ticks": 10,
            "device_ordinal": 0,
            "device_uuid": header["expected_device_uuids"][0],
            "nonce": header["nonce"],
            "task_affinities": {"42": [1]},
        }
        ready_bytes = json.dumps(
            ready_payload, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        ready_path = control / "rank-0.ready"
        ready_path.write_bytes(ready_bytes)
        ready_path.chmod(0o600)
        ready_info = ready_path.stat()
        ready_payload["file"] = {
            "label": "LDOS rank 0 ready record",
            "path": str(ready_path),
            "size_bytes": len(ready_bytes),
            "sha256": ldos.hashlib.sha256(ready_bytes).hexdigest(),
            "fingerprint": {
                "device": ready_info.st_dev,
                "inode": ready_info.st_ino,
                "mode": ready_info.st_mode,
                "mtime_ns": ready_info.st_mtime_ns,
                "ctime_ns": ready_info.st_ctime_ns,
            },
        }
        journal.append("ready", ready_payload)

        def nvml_sample(index: int, current: bool) -> dict:
            scheduled = 100_000_000 + index * 50_000_000
            process = (
                [{
                    "pid": 42,
                    "used_gpu_memory_bytes": 1,
                    "gpu_instance_id": (1 << 32) - 1,
                    "compute_instance_id": (1 << 32) - 1,
                    "observed_start_time_ticks": 10,
                }]
                if current else []
            )
            devices = []
            for uuid_value, compute in ((gpu0, process), (gpu1, [])):
                devices.append(
                    {
                        "uuid": uuid_value,
                        "process_lists": {
                            "compute": compute, "graphics": [], "mps": [],
                        },
                        "process_utilization_since_cursor": [],
                        "utilization": {"gpu_percent": 0, "memory_percent": 0},
                        "graphics_clock_mhz": 1,
                        "memory_clock_mhz": 1,
                        "power_mw": 1,
                        "temperature_c": 1,
                        "pstate": 8,
                        "memory_bytes": {"total": 10, "free": 9, "used": 1},
                    }
                )
            return {
                "scheduled_monotonic_ns": scheduled,
                "started_monotonic_ns": scheduled + 1,
                "finished_monotonic_ns": scheduled + 2,
                "lateness_ns": 1,
                "duration_ns": 1,
                "devices": devices,
                "outsiders": [],
            }

        def host_sample(
            index: int, go: bool, after_exit: bool, departed: bool = False,
            nvml_current: bool | None = None,
        ) -> dict:
            scheduled = 100_000_000 + index * 50_000_000
            state = None if after_exit or departed else {
                "start_time_ticks": 10,
                "ticks": 10 + index * 10,
                "task_affinities": {"42": [1]},
            }
            process_states = {} if after_exit else {
                "41": {
                    "start_time_ticks": 5,
                    "ticks": 5 + index,
                    "task_affinities": {"41": [0]},
                },
            }
            if state is not None:
                process_states["42"] = state
            pressure = {
                "cpu": {"some": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 1}},
                "memory": {
                    "some": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 1},
                    "full": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 1},
                },
                "io": {
                    "some": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 1},
                    "full": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 1},
                },
            }
            return {
                "monotonic_ns": scheduled,
                "scheduled_monotonic_ns": scheduled,
                "sample_started_monotonic_ns": scheduled,
                "sample_finished_monotonic_ns": scheduled + 3,
                "go_released": go,
                "descendant_pids": (
                    [] if after_exit else ([41] if departed else [41, 42])
                ),
                "worker_process_state": {"0": state},
                "process_task_state": process_states,
                "controller_process_state": {
                    "start_time_ticks": 4,
                    "ticks": 1 + index,
                    "task_affinities": {"40": [0]},
                },
                "cpu_counters": {
                    "0": {"total_ticks": 100 + index * 10, "idle_ticks": 50 + index * 5},
                    "1": {"total_ticks": 100 + index * 10, "idle_ticks": 50 + index * 5},
                },
                "current_frequency_khz": {"0": 1, "1": 1},
                "governors": {"0": "performance", "1": "performance"},
                "loadavg": "0 0 0 1/1 1",
                "pressure": pressure,
                "nvml": nvml_sample(
                    index,
                    not after_exit and not departed
                    if nvml_current is None else nvml_current,
                ),
                "after_process_exit": after_exit,
                "departed_worker_ranks": [0] if after_exit or departed else [],
            }

        journal.append("sample", host_sample(0, False, False))
        journal.append("sample", host_sample(1, False, False))
        journal.append(
            "go",
            {"monotonic_ns": 190_000_000, "clean_affinity_polls": 2, "clean_gpu_polls": 2},
        )
        journal.append("sample", host_sample(2, True, False))
        results_payload = {
            "schema_version": 1,
            "state": "RESULTS_READY",
            "rank": 0,
            "pid": 42,
            "start_time_ticks": 10,
            "device_ordinal": 0,
            "device_uuid": header["expected_device_uuids"][0],
            "nonce": header["nonce"],
            "monotonic_ns": 225_000_000,
        }
        results_bytes = json.dumps(
            results_payload, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        results_path = control / "rank-0.results-ready"
        results_path.write_bytes(results_bytes)
        results_path.chmod(0o600)
        results_info = results_path.stat()
        results_payload.update(
            {
                "observed_monotonic_ns": 226_000_000,
                "file": {
                    "label": "LDOS rank 0 results-ready record",
                    "path": str(results_path),
                    "size_bytes": len(results_bytes),
                    "sha256": ldos.hashlib.sha256(results_bytes).hexdigest(),
                    "fingerprint": {
                        "device": results_info.st_dev,
                        "inode": results_info.st_ino,
                        "mode": results_info.st_mode,
                        "mtime_ns": results_info.st_mtime_ns,
                        "ctime_ns": results_info.st_ctime_ns,
                    },
                },
            }
        )
        journal.append("results-ready", results_payload)
        journal.append("sample", host_sample(3, True, False))
        release_bytes = (
            json.dumps(
                {
                    "schema_version": 1,
                    "nonce": header["nonce"],
                    "ranks": 1,
                    "state": "RELEASE",
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        release_path = control / "RELEASE"
        release_path.write_bytes(release_bytes)
        release_path.chmod(0o400)
        release_info = release_path.stat()
        journal.append(
            "release",
            {
                "monotonic_ns": 255_000_000,
                "results_ready_ranks": [0],
                "file": {
                    "label": "LDOS RELEASE gate",
                    "path": str(release_path),
                    "size_bytes": len(release_bytes),
                    "sha256": ldos.hashlib.sha256(release_bytes).hexdigest(),
                    "fingerprint": {
                        "device": release_info.st_dev,
                        "inode": release_info.st_ino,
                        "mode": release_info.st_mode,
                        "mtime_ns": release_info.st_mtime_ns,
                        "ctime_ns": release_info.st_ctime_ns,
                    },
                },
            },
        )
        # Prove the allowed bounded post-RELEASE context disappearance before
        # the authoritative pidfd transition.
        journal.append("sample", host_sample(4, True, False, False, False))
        journal.append(
            "worker-exit",
            {"rank": 0, "pid": 42, "start_time_ticks": 10, "monotonic_ns": 345_000_000},
        )
        journal.append("sample", host_sample(5, True, False, True))
        journal.append("process-exit", {"monotonic_ns": 390_000_000})
        journal.append("sample", host_sample(6, True, True, True))
        journal.append("sample", host_sample(7, True, True, True))
        journal.close("sealed")
        return path, header

    def test_independent_validator_recomputes_complete_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, header = self.make_valid_journal(pathlib.Path(directory))
            result = ldos.validate_lane_journal(path, expected_header=header)
            self.assertNotIn("nvml_library_identity", header)
            self.assertNotIn("nvml_library_archive_identity", header)
            self.assertNotIn("nvml_library_archive_relative_path", header)
            self.assertEqual(
                set(header["nvml_archive_identity"]),
                {
                    "schema_version", "relative_path", "size_bytes", "sha256",
                    "elf_nvml_identity",
                },
            )
            self.assertEqual(result["sample_count"], 8)
            self.assertEqual(
                result[
                    "nvml_absence_observation_to_pidfd_observation_ns_by_rank"
                ],
                [44_999_998],
            )
            self.assertEqual(
                result["nvml_absence_pidfd_observation_order_by_rank"],
                ["absence-observed-first"],
            )
            self.assertEqual(
                result["first_nvml_absence_observation_finished_ns_by_rank"],
                [300_000_002],
            )
            self.assertEqual(
                result["pidfd_readiness_observed_ns_by_rank"],
                [345_000_000],
            )
            self.assertEqual(
                result[
                    "pidfd_observation_to_first_nvml_absence_observation_ns_by_rank"
                ],
                [None],
            )
            self.assertEqual(result["valid_worker_cpu_intervals"], 2)
            linger = next(
                sample for sample in result["timeline"]
                if sample["departed_worker_ranks"] and not sample["after_process_exit"]
            )
            self.assertEqual(linger["descendant_pids"], [41])
            self.assertIsNone(linger["worker_process_state"]["0"])

    def test_ready_records_must_precede_the_first_sample(self) -> None:
        for samples_before_ready in (1, 2):
            with self.subTest(samples=samples_before_ready), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                ready = next(row for row in rows if row["record_type"] == "ready")
                rows.remove(ready)
                sample_indices = [
                    index for index, row in enumerate(rows)
                    if row["record_type"] == "sample"
                ]
                rows.insert(sample_indices[samples_before_ready - 1] + 1, ready)
                self.rewrite_rows(path, rows)
                with self.assertRaisesRegex(ldos.EvidenceError, "ordering|coverage"):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_journal_interval_sequence_allows_equality_and_rejects_one_ns_overlap(self) -> None:
        def sample(start: int, finish: int) -> dict:
            return {
                "record_type": "sample",
                "payload": {
                    "monotonic_ns": start,
                    "sample_started_monotonic_ns": start,
                    "sample_finished_monotonic_ns": finish,
                },
            }

        for ranks in (1, 2):
            events = [
                {
                    "record_type": "worker-exit",
                    "payload": {"rank": rank, "monotonic_ns": 200},
                }
                for rank in range(ranks)
            ]
            exact = [sample(100, 200), *events, sample(200, 300)]
            ldos._validate_journal_interval_sequence(exact)
            before_event = copy.deepcopy(exact)
            before_event[0]["payload"]["sample_finished_monotonic_ns"] = 201
            after_event = copy.deepcopy(exact)
            after_event[-1]["payload"]["monotonic_ns"] = 199
            after_event[-1]["payload"]["sample_started_monotonic_ns"] = 199
            for mutation in (before_event, after_event):
                with self.subTest(ranks=ranks), self.assertRaisesRegex(
                    ldos.EvidenceError, "overlap|backwards"
                ):
                    ldos._validate_journal_interval_sequence(mutation)

    def test_complete_journal_rejects_adjacent_sample_and_event_crossings(self) -> None:
        for mode, delta in (("sample", 0), ("sample", 1), ("event", 0), ("event", 1)):
            with self.subTest(mode=mode, delta=delta), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                samples = [row for row in rows if row["record_type"] == "sample"]
                if mode == "sample":
                    samples[0]["payload"]["sample_finished_monotonic_ns"] = (
                        samples[1]["payload"]["sample_started_monotonic_ns"] + delta
                    )
                else:
                    before_results = next(
                        row for row in samples
                        if row["payload"]["monotonic_ns"] == 200_000_000
                    )
                    observed = next(
                        row["payload"]["observed_monotonic_ns"]
                        for row in rows if row["record_type"] == "results-ready"
                    )
                    before_results["payload"]["sample_finished_monotonic_ns"] = (
                        observed + delta
                    )
                self.rewrite_rows(path, rows)
                if delta:
                    with self.assertRaisesRegex(
                        ldos.EvidenceError, "overlap|backwards|crossed"
                    ):
                        ldos.validate_lane_journal(path, expected_header=header)
                else:
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_sample_and_nvml_ns_endpoints_are_plain_uint64(self) -> None:
        mutations = (
            ("sample_started_monotonic_ns", 100_000_000.0, False),
            ("sample_finished_monotonic_ns", True, False),
            ("scheduled_monotonic_ns", -1, False),
            ("sample_finished_monotonic_ns", 1 << 64, False),
            ("scheduled_monotonic_ns", 100_000_000.0, True),
            ("started_monotonic_ns", True, True),
            ("finished_monotonic_ns", 1 << 64, True),
            ("lateness_ns", -1, True),
            ("duration_ns", 1.0, True),
        )
        for name, value, inside_nvml in mutations:
            with self.subTest(name=name, value=value), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                sample = next(
                    row["payload"] for row in rows if row["record_type"] == "sample"
                )
                target = sample["nvml"] if inside_nvml else sample
                target[name] = value
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_abort_closes_every_pidfd_when_journal_close_fails(self) -> None:
        monitor = object.__new__(ldos.LaneControlMonitor)
        monitor.journal_record = None
        monitor.journal = types.SimpleNamespace(
            close=lambda *_args: (_ for _ in ()).throw(OSError("journal close"))
        )
        read_descriptor, write_descriptor = os.pipe()
        monitor.ready_pidfds = {0: read_descriptor, 1: write_descriptor}
        with self.assertRaisesRegex(OSError, "journal close"):
            monitor.abort("primary lane failure")
        self.assertEqual(monitor.ready_pidfds, {})
        for descriptor in (read_descriptor, write_descriptor):
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_finish_verifies_live_original_and_archived_nvml_handles(self) -> None:
        monitor = object.__new__(ldos.LaneControlMonitor)
        monitor.go_monotonic_ns = 1
        monitor.ready = {0: {}}
        monitor.results_ready = {0: {}}
        monitor.release_monotonic_ns = 2
        monitor.ranks = 1
        monitor.online_cpu_list = "0"
        monitor.physical_topology_plan = {"fixture": True}
        source_identity = {"sha256": "a" * 64, "size_bytes": 10}
        archive_identity = {"sha256": "b" * 64, "size_bytes": 10}
        source_verify = mock.Mock(return_value=source_identity)
        archive_verify = mock.Mock(return_value=archive_identity)
        monitor.nvml_sampler = types.SimpleNamespace(
            library_file=types.SimpleNamespace(verify=source_verify),
            library_identity=source_identity,
        )
        monitor.nvml_library_archive = types.SimpleNamespace(
            verify=archive_verify, record=lambda: archive_identity,
        )
        monitor.journal_header = {
            "allowed_cpus": [0],
            "nvml_archive_identity": {},
        }
        with (
            mock.patch.object(pathlib.Path, "read_text", return_value="0\n"),
            mock.patch.object(
                ldos, "_physical_cpu_core_plan",
                return_value=monitor.physical_topology_plan,
            ),
            self.assertRaisesRegex(ldos.EvidenceError, "held/archived NVML"),
        ):
            monitor.finish()
        source_verify.assert_called_once_with()
        archive_verify.assert_called_once_with()

    def test_ready_and_worker_identity_fields_require_plain_integers(self) -> None:
        mutations = (
            "ready-worker-start-float", "ready-worker-pid-float",
            "worker-pid-float", "worker-start-float", "device-ordinal-float",
            "ready-schema-bool",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                ready = next(row["payload"] for row in rows if row["record_type"] == "ready")
                worker_exit = next(
                    row["payload"] for row in rows if row["record_type"] == "worker-exit"
                )
                if mutation == "ready-worker-start-float":
                    ready["start_time_ticks"] = 10.0
                    worker_exit["start_time_ticks"] = 10.0
                elif mutation == "ready-worker-pid-float":
                    ready["pid"] = 42.0
                    worker_exit["pid"] = 42.0
                elif mutation == "worker-pid-float":
                    worker_exit["pid"] = 42.0
                elif mutation == "worker-start-float":
                    worker_exit["start_time_ticks"] = 10.0
                elif mutation == "device-ordinal-float":
                    ready["device_ordinal"] = 0.0
                else:
                    ready["schema_version"] = True
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_outer_schema_and_sequence_require_plain_integers(self) -> None:
        for field, value in (
            ("schema_version", True), ("schema_version", 1.0),
            ("sequence", 0.0),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0][field] = value
                self.rewrite_rows(path, rows)
                if field == "sequence":
                    rows[0][field] = value
                    path.write_text(
                        "".join(
                            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                            for row in rows
                        ), encoding="utf-8",
                    )
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_pidfd_ready_zombie_is_committed_as_exact_worker_exit(self) -> None:
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        descriptor = ldos._pidfd_open(pid)
        try:
            poller = ldos.select.poll()
            poller.register(descriptor, ldos.select.POLLIN)
            self.assertTrue(poller.poll(1000), "child did not become a zombie")
            start_ticks = ldos._process_start_time_ticks(pid)
            self.assertIsNotNone(start_ticks)
            self.assertIsNotNone(
                ldos._process_task_cpu_state(pid),
                "fixture must still expose the unreaped zombie in /proc",
            )
            events = []
            monitor = object.__new__(ldos.LaneControlMonitor)
            monitor.go_monotonic_ns = 1
            monitor.ready = {
                0: {"pid": pid, "start_time_ticks": start_ticks}
            }
            monitor.ready_pidfds = {0: descriptor}
            monitor.worker_exits = {}
            monitor.first_nvml_absence_ns = {}
            monitor.journal = types.SimpleNamespace(
                append=lambda record_type, payload: events.append(
                    (record_type, payload)
                )
            )
            monitor._record_worker_exits()
            self.assertEqual(set(monitor.worker_exits), {0})
            self.assertEqual(events[0][0], "worker-exit")
            self.assertEqual(events[0][1]["pid"], pid)
        finally:
            ldos._close_native_owned_descriptor_once(
                "zombie-ready test pidfd", descriptor
            )
            os.waitpid(pid, 0)

    def test_nvml_absence_observation_to_pidfd_observation_boundary_is_exact(self) -> None:
        for delta, accepted in (
            (ldos.NVML_ABSENCE_OBSERVATION_TO_PIDFD_OBSERVATION_NS, True),
            (ldos.NVML_ABSENCE_OBSERVATION_TO_PIDFD_OBSERVATION_NS + 1, False),
        ):
            with self.subTest(delta=delta):
                read_descriptor, write_descriptor = os.pipe()
                try:
                    os.write(write_descriptor, b"x")
                    monitor = object.__new__(ldos.LaneControlMonitor)
                    monitor.go_monotonic_ns = 1
                    monitor.ready = {0: {"pid": 42, "start_time_ticks": 10}}
                    monitor.ready_pidfds = {0: read_descriptor}
                    monitor.worker_exits = {}
                    monitor.first_nvml_absence_ns = {0: 100}
                    monitor.journal = types.SimpleNamespace(append=lambda *_args: None)
                    with (
                        mock.patch.object(
                            ldos, "_process_start_time_ticks", return_value=10
                        ),
                        mock.patch.object(
                            ldos.time, "monotonic_ns", return_value=100 + delta
                        ),
                    ):
                        if accepted:
                            monitor._record_worker_exits()
                            self.assertEqual(set(monitor.worker_exits), {0})
                        else:
                            with self.assertRaisesRegex(
                                ldos.EvidenceError, "bounded pidfd"
                            ):
                                monitor._record_worker_exits()
                            self.assertEqual(
                                monitor.worker_exits[0]["monotonic_ns"],
                                100 + delta,
                            )
                finally:
                    os.close(read_descriptor)
                    os.close(write_descriptor)

    def test_two_rank_pidfds_are_harvested_independently(self) -> None:
        rank0_read, rank0_write = os.pipe()
        rank1_read, rank1_write = os.pipe()
        try:
            os.write(rank0_write, b"x")
            monitor = object.__new__(ldos.LaneControlMonitor)
            monitor.go_monotonic_ns = 1
            monitor.ready = {
                0: {"pid": 42, "start_time_ticks": 10},
                1: {"pid": 43, "start_time_ticks": 11},
            }
            monitor.ready_pidfds = {0: rank0_read, 1: rank1_read}
            monitor.worker_exits = {}
            monitor.first_nvml_absence_ns = {}
            monitor.journal = types.SimpleNamespace(append=lambda *_args: None)

            def epoch(pid):
                return {42: 10, 43: 11}[pid]

            with mock.patch.object(ldos, "_process_start_time_ticks", side_effect=epoch):
                monitor._record_worker_exits()
                self.assertEqual(set(monitor.worker_exits), {0})
                os.write(rank1_write, b"x")
                monitor._record_worker_exits()
                self.assertEqual(set(monitor.worker_exits), {0, 1})
        finally:
            for descriptor in (rank0_read, rank0_write, rank1_read, rank1_write):
                os.close(descriptor)

    def test_two_rank_delayed_pidfd_poll_uses_fresh_per_rank_endpoints(self) -> None:
        rank0_read, rank0_write = os.pipe()
        rank1_read, rank1_write = os.pipe()
        try:
            os.write(rank0_write, b"x")
            os.write(rank1_write, b"x")
            events = []
            monitor = object.__new__(ldos.LaneControlMonitor)
            monitor.go_monotonic_ns = 1
            monitor.ready = {
                0: {"pid": 42, "start_time_ticks": 10},
                1: {"pid": 43, "start_time_ticks": 11},
            }
            monitor.ready_pidfds = {0: rank0_read, 1: rank1_read}
            monitor.worker_exits = {}
            monitor.first_nvml_absence_ns = {0: 100, 1: 100}
            monitor.journal = types.SimpleNamespace(
                append=lambda kind, payload: events.append((kind, dict(payload)))
            )
            exact = 100 + ldos.NVML_ABSENCE_OBSERVATION_TO_PIDFD_OBSERVATION_NS
            clock = {"now": exact}
            poll_count = 0

            class DelayedPoll:
                def __init__(self, index):
                    self.index = index

                def register(self, _descriptor, _events):
                    return None

                def poll(self, _timeout):
                    if self.index == 1:
                        clock["now"] += 1
                    return [(1, ldos.select.POLLIN)]

            def poll_factory():
                nonlocal poll_count
                result = DelayedPoll(poll_count)
                poll_count += 1
                return result

            with (
                mock.patch.object(
                    ldos,
                    "_process_start_time_ticks",
                    side_effect=lambda pid: {42: 10, 43: 11}[pid],
                ),
                mock.patch.object(
                    ldos.time,
                    "monotonic_ns",
                    side_effect=lambda: clock["now"],
                ),
                mock.patch.object(ldos.select, "poll", side_effect=poll_factory),
                self.assertRaisesRegex(ldos.EvidenceError, "bounded pidfd"),
            ):
                monitor._record_worker_exits()
            self.assertEqual(
                [monitor.worker_exits[rank]["monotonic_ns"] for rank in (0, 1)],
                [exact, exact + 1],
            )
            self.assertEqual([event[1]["rank"] for event in events], [0, 1])
        finally:
            for descriptor in (rank0_read, rank0_write, rank1_read, rank1_write):
                os.close(descriptor)

    def test_rank2_partial_exit_poll_starts_sample_after_fresh_exit_event(self) -> None:
        monitor = object.__new__(ldos.LaneControlMonitor)
        monitor.ranks = 2
        monitor.go_monotonic_ns = 1
        monitor.release_monotonic_ns = 2
        monitor.process_finished_monotonic_ns = None
        monitor.sample_origin_ns = None
        monitor.next_sample_index = 0
        monitor.controller_pid = 40
        monitor.controller_start_time_ticks = 100
        monitor.controller_cpu_mask = {0}
        monitor.worker_cpu_masks = [{1}, {2}]
        monitor.nvml_device_uuids = ["gpu-0", "gpu-1"]
        monitor.ready = {
            0: {"pid": 42, "start_time_ticks": 10},
            1: {"pid": 43, "start_time_ticks": 11},
        }
        monitor.results_ready = {0: {}, 1: {}}
        monitor.worker_exits = {}
        monitor.first_nvml_absence_ns = {}
        monitor.lifecycle_issues = []
        monitor.timeline = []
        monitor.clean_gpu_polls = 0
        monitor.clean_affinity_polls = 0
        events = []
        monitor.journal = types.SimpleNamespace(
            append=lambda kind, payload: events.append((kind, copy.deepcopy(payload)))
        )
        clock = {"now": 100}

        def now() -> int:
            clock["now"] += 1
            return clock["now"]

        def record_partial_exit() -> None:
            record = {
                "rank": 0, "pid": 42, "start_time_ticks": 10,
                "monotonic_ns": now(),
            }
            monitor.worker_exits[0] = record
            monitor.journal.append("worker-exit", record)

        def process_state(pid: int):
            if pid == 40:
                return {
                    "start_time_ticks": 100, "ticks": 1,
                    "task_affinities": {"40": [0]},
                }
            if pid == 43:
                return {
                    "start_time_ticks": 11, "ticks": 1,
                    "task_affinities": {"43": [2]},
                }
            return None

        def read_text(path: pathlib.Path, *_args, **_kwargs) -> str:
            value = str(path)
            if value.endswith("scaling_cur_freq"):
                return "1000\n"
            if value.endswith("scaling_governor"):
                return "performance\n"
            if value == "/proc/loadavg":
                return "0 0 0 1/1 1\n"
            raise AssertionError(value)

        def nvml_sample(**kwargs):
            started = now()
            return {
                "scheduled_monotonic_ns": kwargs["scheduled_monotonic_ns"],
                "started_monotonic_ns": started,
                "finished_monotonic_ns": started,
                "lateness_ns": started - kwargs["scheduled_monotonic_ns"],
                "duration_ns": 0,
                "devices": [], "outsiders": [],
            }

        monitor.nvml_sampler = types.SimpleNamespace(sample=nvml_sample)
        monitor._record_worker_exits = record_partial_exit
        monitor._require_absence_deadlines = lambda _now: None
        monitor._observe_nvml_lifecycle_and_maybe_release = lambda *_args, **_kwargs: None
        with (
            mock.patch.object(ldos.time, "monotonic_ns", side_effect=now),
            mock.patch.object(ldos, "_process_start_time_ticks", return_value=99),
            mock.patch.object(ldos, "_process_descendants", return_value={43}),
            mock.patch.object(ldos, "_process_task_cpu_state", side_effect=process_state),
            mock.patch.object(ldos.pathlib.Path, "read_text", new=read_text),
            mock.patch.object(
                ldos, "_proc_stat_cpu_counters",
                return_value={
                    0: {"total_ticks": 1, "idle_ticks": 0},
                    1: {"total_ticks": 1, "idle_ticks": 0},
                    2: {"total_ticks": 1, "idle_ticks": 0},
                },
            ),
            mock.patch.object(ldos, "_parse_pressure_file", return_value={}),
            mock.patch.object(ldos, "_derive_nvml_outsiders", return_value=[]),
            mock.patch.object(
                ldos, "_nvml_current_presence_with_departed_lag", return_value=True
            ),
        ):
            monitor.poll(40)
        self.assertEqual([kind for kind, _payload in events], ["worker-exit", "sample"])
        exit_ns = events[0][1]["monotonic_ns"]
        sample = events[1][1]
        self.assertGreaterEqual(sample["sample_started_monotonic_ns"], exit_ns)
        self.assertEqual(sample["departed_worker_ranks"], [0])

    def test_pre_go_missing_ready_never_starts_or_advances_sampling(self) -> None:
        for ranks, available in ((1, set()), (2, {0})):
            with self.subTest(ranks=ranks):
                monitor = object.__new__(ldos.LaneControlMonitor)
                monitor.ranks = ranks
                monitor.go_monotonic_ns = None
                monitor.ready = {}
                monitor.ready_pidfds = {}
                monitor.worker_exits = {}
                monitor.sample_origin_ns = None
                monitor.next_sample_index = 0
                monitor.timeline = []
                events = []
                monitor.journal = types.SimpleNamespace(
                    append=lambda kind, payload: events.append((kind, payload))
                )
                monitor._require_absence_deadlines = lambda _now: None
                monitor._load_ready = lambda rank: (
                    {"rank": rank, "pid": 42 + rank, "start_time_ticks": 10 + rank}
                    if rank in available else None
                )
                with mock.patch.object(
                    ldos, "_pidfd_open", side_effect=lambda pid, **_kwargs: pid + 100
                ):
                    monitor.poll(40)
                self.assertEqual(set(monitor.ready), available)
                self.assertIsNone(monitor.sample_origin_ns)
                self.assertEqual(monitor.next_sample_index, 0)
                self.assertEqual(monitor.timeline, [])
                self.assertEqual([kind for kind, _payload in events], ["ready"] * len(available))

    def test_rank2_last_ready_precedes_first_complete_sample(self) -> None:
        monitor = object.__new__(ldos.LaneControlMonitor)
        monitor.ranks = 2
        monitor.go_monotonic_ns = None
        monitor.release_monotonic_ns = None
        monitor.process_finished_monotonic_ns = None
        monitor.ready = {}
        monitor.ready_pidfds = {}
        monitor.results_ready = {}
        monitor.worker_exits = {}
        monitor.first_nvml_absence_ns = {}
        monitor.sample_origin_ns = None
        monitor.next_sample_index = 0
        monitor.timeline = []
        monitor.clean_affinity_polls = 0
        monitor.clean_gpu_polls = 0
        monitor.lifecycle_issues = []
        monitor.controller_pid = 40
        monitor.controller_cpu_mask = {0}
        monitor.worker_cpu_masks = [{1}, {2}]
        monitor.nvml_device_uuids = ["gpu-0", "gpu-1"]
        events = []
        monitor.journal = types.SimpleNamespace(
            append=lambda kind, payload: events.append((kind, copy.deepcopy(payload)))
        )
        rank1_available = {"value": False}

        def load_ready(rank: int):
            if rank == 1 and not rank1_available["value"]:
                return None
            return {
                "rank": rank, "pid": 42 + rank,
                "start_time_ticks": 10 + rank,
            }

        monitor._load_ready = load_ready
        monitor._require_absence_deadlines = lambda _now: None
        monitor._observe_nvml_lifecycle_and_maybe_release = lambda *_args, **_kwargs: None
        clock = {"now": 100}

        def now() -> int:
            clock["now"] += 1
            return clock["now"]

        def process_state(pid: int):
            masks = {40: [0], 42: [1], 43: [2]}
            starts = {40: 100, 42: 10, 43: 11}
            if pid not in masks:
                return None
            return {
                "start_time_ticks": starts[pid], "ticks": 1,
                "task_affinities": {str(pid): masks[pid]},
            }

        def read_text(path: pathlib.Path, *_args, **_kwargs) -> str:
            value = str(path)
            if value.endswith("scaling_cur_freq"):
                return "1000\n"
            if value.endswith("scaling_governor"):
                return "performance\n"
            if value == "/proc/loadavg":
                return "0 0 0 1/1 1\n"
            raise AssertionError(value)

        def nvml_sample(**kwargs):
            started = now()
            return {
                "scheduled_monotonic_ns": kwargs["scheduled_monotonic_ns"],
                "started_monotonic_ns": started,
                "finished_monotonic_ns": started,
                "lateness_ns": started - kwargs["scheduled_monotonic_ns"],
                "duration_ns": 0, "devices": [], "outsiders": [],
            }

        monitor.nvml_sampler = types.SimpleNamespace(sample=nvml_sample)
        with (
            mock.patch.object(ldos.time, "monotonic_ns", side_effect=now),
            mock.patch.object(
                ldos, "_pidfd_open", side_effect=lambda pid, **_kwargs: pid + 100
            ),
            mock.patch.object(ldos, "_process_descendants", return_value={42, 43}),
            mock.patch.object(ldos, "_process_task_cpu_state", side_effect=process_state),
            mock.patch.object(ldos.pathlib.Path, "read_text", new=read_text),
            mock.patch.object(
                ldos, "_proc_stat_cpu_counters",
                return_value={
                    0: {"total_ticks": 1, "idle_ticks": 0},
                    1: {"total_ticks": 1, "idle_ticks": 0},
                    2: {"total_ticks": 1, "idle_ticks": 0},
                },
            ),
            mock.patch.object(ldos, "_parse_pressure_file", return_value={}),
            mock.patch.object(ldos, "_derive_nvml_outsiders", return_value=[]),
            mock.patch.object(
                ldos, "_nvml_current_presence_with_departed_lag", return_value=True
            ),
        ):
            monitor.poll(40)
            self.assertEqual([kind for kind, _payload in events], ["ready"])
            self.assertIsNone(monitor.sample_origin_ns)
            self.assertEqual(monitor.next_sample_index, 0)
            rank1_available["value"] = True
            monitor.poll(40)
        self.assertEqual([kind for kind, _payload in events], ["ready", "ready", "sample"])
        sample = events[-1][1]
        self.assertEqual(set(sample["worker_process_state"]), {"0", "1"})
        self.assertTrue(all(sample["worker_process_state"].values()))
        self.assertEqual(monitor.next_sample_index, 1)

    def test_results_ready_observation_binds_to_following_nvml_start(self) -> None:
        nvml = {"started_monotonic_ns": 200}
        self.assertTrue(
            ldos._results_ready_precede_nvml_sample(
                [{"observed_monotonic_ns": 200}], nvml
            )
        )
        self.assertFalse(
            ldos._results_ready_precede_nvml_sample(
                [{"observed_monotonic_ns": 201}], nvml
            )
        )

    @staticmethod
    def handshake_state(ranks: int = 2):
        monitor = object.__new__(ldos.LaneControlMonitor)
        monitor.ranks = ranks
        monitor.ready = {
            rank: {"pid": 42 + rank, "start_time_ticks": 10 + rank}
            for rank in range(ranks)
        }
        monitor.results_ready = {
            rank: {"observed_monotonic_ns": 100 + rank}
            for rank in range(ranks)
        }
        monitor.worker_exits = {}
        monitor.lifecycle_issues = []
        monitor.release_monotonic_ns = None
        monitor.first_nvml_absence_ns = {}
        monitor.nvml_device_uuids = [f"gpu-{rank}" for rank in range(ranks)]
        return monitor

    def test_all_rank_results_ready_and_exact_nvml_publish_release(self) -> None:
        monitor = self.handshake_state(2)
        monitor._active_nvml_rank_presence = mock.Mock(
            return_value={0: True, 1: True}
        )
        published = []
        monitor._publish_release = lambda: published.append(True)
        monitor._observe_nvml_lifecycle_and_maybe_release(
            {"started_monotonic_ns": 102, "finished_monotonic_ns": 103},
            presence_exact=True,
        )
        self.assertEqual(published, [True])

    def test_rank2_partial_mismatched_results_or_missing_nvml_never_releases(self) -> None:
        for mode, presence in (
            ("partial", True), ("mismatched", True), ("missing-nvml", False)
        ):
            with self.subTest(mode=mode, presence=presence):
                monitor = self.handshake_state(2)
                if mode == "partial":
                    monitor.results_ready.pop(1)
                elif mode == "mismatched":
                    monitor.results_ready[2] = monitor.results_ready.pop(1)
                monitor._active_nvml_rank_presence = mock.Mock(
                    return_value={0: True, 1: presence}
                )
                publish = mock.Mock()
                monitor._publish_release = publish
                monitor._observe_nvml_lifecycle_and_maybe_release(
                    {"started_monotonic_ns": 102, "finished_monotonic_ns": 103},
                    presence_exact=presence,
                )
                publish.assert_not_called()
                if not presence:
                    self.assertEqual(monitor.first_nvml_absence_ns, {1: 103})
                    self.assertTrue(monitor.lifecycle_issues)

    def test_pre_release_nvml_disappearance_then_reappearance_rejects(self) -> None:
        monitor = self.handshake_state(1)
        monitor._active_nvml_rank_presence = mock.Mock(
            side_effect=({0: False}, {0: True})
        )
        monitor._publish_release = mock.Mock()
        monitor._observe_nvml_lifecycle_and_maybe_release(
            {"started_monotonic_ns": 110, "finished_monotonic_ns": 111},
            presence_exact=False,
        )
        with self.assertRaisesRegex(ldos.EvidenceError, "reappeared"):
            monitor._observe_nvml_lifecycle_and_maybe_release(
                {"started_monotonic_ns": 120, "finished_monotonic_ns": 121},
                presence_exact=True,
            )

    def test_post_absence_reappearance_after_pidfd_exit_rejects(self) -> None:
        monitor = self.handshake_state(1)
        monitor.worker_exits = {
            0: {"rank": 0, "pid": 42, "start_time_ticks": 10, "monotonic_ns": 150}
        }
        monitor._active_nvml_rank_presence = mock.Mock(return_value={})
        monitor._observe_nvml_lifecycle_and_maybe_release(
            {
                "started_monotonic_ns": 151,
                "finished_monotonic_ns": 152,
                "devices": [
                    {
                        "uuid": "gpu-0",
                        "process_lists": {
                            "compute": [
                                {"pid": 42, "observed_start_time_ticks": 10}
                            ],
                            "graphics": [],
                            "mps": [],
                        },
                    }
                ],
            },
            presence_exact=True,
        )
        self.assertEqual(monitor.first_nvml_absence_ns, {})
        monitor._observe_nvml_lifecycle_and_maybe_release(
            {
                "started_monotonic_ns": 155,
                "finished_monotonic_ns": 156,
                "devices": [
                    {
                        "uuid": "gpu-0",
                        "process_lists": {
                            "compute": [], "graphics": [], "mps": [],
                        },
                    }
                ],
            },
            presence_exact=True,
        )
        self.assertEqual(monitor.first_nvml_absence_ns, {0: 156})
        with self.assertRaisesRegex(ldos.EvidenceError, "reappeared"):
            monitor._observe_nvml_lifecycle_and_maybe_release(
                {
                    "started_monotonic_ns": 160,
                    "finished_monotonic_ns": 161,
                    "devices": [
                        {
                            "uuid": "gpu-0",
                            "process_lists": {
                                "compute": [
                                    {"pid": 42, "observed_start_time_ticks": 10}
                                ],
                                "graphics": [],
                                "mps": [],
                            },
                        }
                    ],
                },
                presence_exact=True,
            )

    def test_departed_live_nvml_binding_change_rejects_before_absence(self) -> None:
        monitor = self.handshake_state(1)
        monitor.worker_exits = {
            0: {"rank": 0, "pid": 42, "start_time_ticks": 10, "monotonic_ns": 150}
        }
        monitor._active_nvml_rank_presence = mock.Mock(return_value={})
        with self.assertRaisesRegex(ldos.EvidenceError, "binding changed"):
            monitor._observe_nvml_lifecycle_and_maybe_release(
                {
                    "started_monotonic_ns": 155,
                    "finished_monotonic_ns": 156,
                    "devices": [
                        {
                            "uuid": "gpu-0",
                            "process_lists": {
                                "compute": [
                                    {"pid": 43, "observed_start_time_ticks": 11}
                                ],
                                "graphics": [],
                                "mps": [],
                            },
                        }
                    ],
                },
                presence_exact=True,
            )

    def test_absence_deadline_and_exit0_without_handshake_fail_closed(self) -> None:
        monitor = self.handshake_state(1)
        monitor.first_nvml_absence_ns = {0: 100}
        with self.assertRaisesRegex(ldos.EvidenceError, "exceeded"):
            monitor._require_absence_deadlines(
                100
                + ldos.NVML_ABSENCE_OBSERVATION_TO_PIDFD_OBSERVATION_NS
                + 1
            )
        monitor.worker_exits = {
            0: {"rank": 0, "pid": 42, "start_time_ticks": 10, "monotonic_ns": 200}
        }
        monitor.results_ready = {}
        monitor.release_monotonic_ns = None
        with self.assertRaisesRegex(ldos.EvidenceError, "lifecycle"):
            monitor.drain(99, 300)

    def test_results_ready_wrong_identity_fields_are_rejected(self) -> None:
        base = {
            "schema_version": 1,
            "state": "RESULTS_READY",
            "rank": 0,
            "pid": 42,
            "start_time_ticks": 10,
            "device_ordinal": 0,
            "device_uuid": "a" * 32,
            "nonce": "b" * 64,
            "monotonic_ns": 100,
        }
        mutations = {
            "nonce": "c" * 64,
            "pid": 43,
            "start_time_ticks": 11,
            "device_ordinal": 1,
            "device_uuid": "d" * 32,
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                record = dict(base)
                record[field] = replacement
                path = root / "rank-0.results-ready"
                path.write_text(compact(record), encoding="utf-8")
                path.chmod(0o600)
                monitor = object.__new__(ldos.LaneControlMonitor)
                monitor.directory = root
                monitor.ready = {
                    0: {
                        "pid": 42, "start_time_ticks": 10,
                        "device_ordinal": 0, "device_uuid": "a" * 32,
                    }
                }
                monitor.worker_cpu_masks = [{1}]
                monitor.nonce = "b" * 64
                monitor.go_monotonic_ns = 50
                with (
                    mock.patch.object(ldos.time, "monotonic_ns", return_value=200),
                    mock.patch.object(
                        ldos, "_process_task_cpu_state",
                        return_value={
                            "start_time_ticks": 10,
                            "ticks": 1,
                            "task_affinities": {"42": [1]},
                        },
                    ),
                    self.assertRaisesRegex(ldos.EvidenceError, "lane binding"),
                ):
                    monitor._load_results_ready(0)
    def test_results_ready_and_release_are_mandatory_and_ordered(self) -> None:
        for mutation in ("missing-results", "missing-release", "results-after-release"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if mutation == "missing-results":
                    rows = [row for row in rows if row["record_type"] != "results-ready"]
                elif mutation == "missing-release":
                    rows = [row for row in rows if row["record_type"] != "release"]
                else:
                    result_index = next(
                        index for index, row in enumerate(rows)
                        if row["record_type"] == "results-ready"
                    )
                    release_index = next(
                        index for index, row in enumerate(rows)
                        if row["record_type"] == "release"
                    )
                    rows[result_index], rows[release_index] = (
                        rows[release_index], rows[result_index]
                    )
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_handshake_integer_fields_reject_boolean_aliases(self) -> None:
        for mutation in ("results-schema", "release-rank"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if mutation == "release-rank":
                    release = next(
                        row["payload"] for row in rows
                        if row["record_type"] == "release"
                    )
                    release["results_ready_ranks"] = [False]
                else:
                    result = next(
                        row["payload"] for row in rows
                        if row["record_type"] == "results-ready"
                    )
                    result["schema_version"] = True
                    worker = {
                        key: result[key]
                        for key in (
                            "schema_version", "state", "rank", "pid",
                            "start_time_ticks", "device_ordinal", "device_uuid",
                            "nonce", "monotonic_ns",
                        )
                    }
                    data = compact(worker).encode("utf-8")
                    named = path.parent / "rank-0.results-ready"
                    named.write_bytes(data)
                    named.chmod(0o600)
                    info = named.stat()
                    result["file"] = {
                        "label": "LDOS rank 0 results-ready record",
                        "path": str(named),
                        "size_bytes": len(data),
                        "sha256": ldos.hashlib.sha256(data).hexdigest(),
                        "fingerprint": {
                            "device": info.st_dev,
                            "inode": info.st_ino,
                            "mode": info.st_mode,
                            "mtime_ns": info.st_mtime_ns,
                            "ctime_ns": info.st_ctime_ns,
                        },
                    }
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_handshake_named_files_reject_links_and_hardlinks(self) -> None:
        for target in ("results-symlink", "release-hardlink"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                path, header = self.make_valid_journal(root)
                if target == "results-symlink":
                    named = path.parent / "rank-0.results-ready"
                    replacement = path.parent / "replacement-results"
                    replacement.write_bytes(named.read_bytes())
                    replacement.chmod(0o600)
                    named.unlink()
                    named.symlink_to(replacement)
                else:
                    os.link(path.parent / "RELEASE", path.parent / "release-hardlink")
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_offline_validator_rejects_reappearance_after_pidfd_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, header = self.make_valid_journal(pathlib.Path(directory))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            pre_pidfd = next(
                row["payload"] for row in rows
                if row["record_type"] == "sample"
                and row["payload"]["monotonic_ns"] == 300_000_000
            )
            compute_row = {
                "pid": 42,
                "used_gpu_memory_bytes": 1,
                "gpu_instance_id": (1 << 32) - 1,
                "compute_instance_id": (1 << 32) - 1,
                "observed_start_time_ticks": 10,
            }
            pre_pidfd["nvml"]["devices"][0]["process_lists"]["compute"] = [
                copy.deepcopy(compute_row)
            ]
            departed_template = next(
                row["payload"] for row in rows
                if row["record_type"] == "sample"
                and row["payload"]["monotonic_ns"] == 350_000_000
            )
            departed_template["nvml"]["devices"][0]["process_lists"][
                "compute"
            ] = [copy.deepcopy(compute_row)]
            absence_row = next(
                row for row in rows
                if row["record_type"] == "sample"
                and row["payload"]["monotonic_ns"] == 400_000_000
            )
            reappearance_row = next(
                row for row in rows
                if row["record_type"] == "sample"
                and row["payload"]["monotonic_ns"] == 450_000_000
            )
            for sample_row in (absence_row, reappearance_row):
                sample = sample_row["payload"]
                sample["after_process_exit"] = False
                sample["departed_worker_ranks"] = [0]
                sample["descendant_pids"] = [41]
                sample["process_task_state"] = copy.deepcopy(
                    departed_template["process_task_state"]
                )
            reappearance = reappearance_row["payload"]
            reappearance["nvml"]["devices"][0]["process_lists"]["compute"] = [
                copy.deepcopy(compute_row)
            ]
            process_exit_row = next(
                row for row in rows if row["record_type"] == "process-exit"
            )
            process_exit_row["payload"]["monotonic_ns"] = 490_000_000
            rows.remove(process_exit_row)
            rows.insert(rows.index(reappearance_row) + 1, process_exit_row)
            self.rewrite_rows(path, rows)
            with self.assertRaisesRegex(ldos.EvidenceError, "reappeared"):
                ldos.validate_lane_journal(path, expected_header=header)

    def test_offline_departed_nvml_binding_change_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, header = self.make_valid_journal(pathlib.Path(directory))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for monotonic_ns, pid, epoch in (
                (300_000_000, 42, 10),
                (350_000_000, 43, 11),
            ):
                sample = next(
                    row["payload"] for row in rows
                    if row["record_type"] == "sample"
                    and row["payload"]["monotonic_ns"] == monotonic_ns
                )
                sample["nvml"]["devices"][0]["process_lists"]["compute"] = [{
                    "pid": pid,
                    "used_gpu_memory_bytes": 1,
                    "gpu_instance_id": (1 << 32) - 1,
                    "compute_instance_id": (1 << 32) - 1,
                    "observed_start_time_ticks": epoch,
                }]
            self.rewrite_rows(path, rows)
            with self.assertRaisesRegex(ldos.EvidenceError, "binding changed"):
                ldos.validate_lane_journal(path, expected_header=header)

    def test_results_ready_loader_rejects_dangling_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "rank-0.results-ready").symlink_to(root / "missing")
            monitor = object.__new__(ldos.LaneControlMonitor)
            monitor.directory = root
            with self.assertRaisesRegex(ldos.EvidenceError, "exact regular file"):
                monitor._load_results_ready(0)

    def test_results_ready_loader_rejects_oversize_before_hash_or_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = root / "rank-0.results-ready"
            with path.open("wb") as stream:
                stream.truncate(ldos.LDOS_HANDSHAKE_MAX_BYTES + 1)
            path.chmod(0o600)
            monitor = object.__new__(ldos.LaneControlMonitor)
            monitor.directory = root
            with (
                mock.patch.object(
                    ldos, "_sha256_fd",
                    side_effect=AssertionError("oversize file must not hash"),
                ),
                self.assertRaises(ldos.EvidenceError),
            ):
                monitor._load_results_ready(0)

    def test_independent_validator_rejects_truncation_and_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path, header = self.make_valid_journal(root)
            original = path.read_bytes()
            path.write_bytes(original[:-1])
            with self.assertRaisesRegex(ldos.EvidenceError, "truncated"):
                ldos.validate_lane_journal(path, expected_header=header)
            path.write_bytes(original.replace(b'"sequence":1', b'"sequence":9', 1))
            with self.assertRaisesRegex(ldos.EvidenceError, "sequence"):
                ldos.validate_lane_journal(path, expected_header=header)

    def test_event_order_and_departure_transition_are_exact(self) -> None:
        mutations = ("worker-before-go", "departed-before-event", "missing-transition")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                worker_index = next(
                    index for index, row in enumerate(rows)
                    if row["record_type"] == "worker-exit"
                )
                go_index = next(
                    index for index, row in enumerate(rows)
                    if row["record_type"] == "go"
                )
                if mutation == "worker-before-go":
                    worker = rows.pop(worker_index)
                    rows.insert(go_index, worker)
                else:
                    sample_rows = [
                        row for row in rows if row["record_type"] == "sample"
                    ]
                    target = sample_rows[3] if mutation == "departed-before-event" else sample_rows[5]
                    target["payload"]["departed_worker_ranks"] = (
                        [0] if mutation == "departed-before-event" else []
                    )
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_record_topology_and_boolean_fields_are_exact(self) -> None:
        mutations = ("duplicate-header", "ready-after-go", "string-go")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if mutation == "duplicate-header":
                    rows.insert(1, copy.deepcopy(rows[0]))
                elif mutation == "ready-after-go":
                    ready = rows.pop(1)
                    go_index = next(
                        index for index, row in enumerate(rows)
                        if row["record_type"] == "go"
                    )
                    rows.insert(go_index + 1, ready)
                else:
                    next(
                        row for row in rows
                        if row["record_type"] == "sample"
                        and row["payload"]["go_released"]
                    )["payload"]["go_released"] = "yes"
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_journal_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path, header = self.make_valid_journal(root)
            link = root / "journal-link"
            link.symlink_to(path)
            with self.assertRaisesRegex(ldos.EvidenceError, "symbolic link"):
                ldos.validate_lane_journal(link, expected_header=header)

    def test_journal_check_then_open_symlink_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path, header = self.make_valid_journal(root)
            saved = root / "saved-journal"
            real_open = ldos.os.open
            swapped = False

            def swap_before_open(target, flags, *args):
                nonlocal swapped
                if pathlib.Path(target) == path and not swapped:
                    swapped = True
                    path.rename(saved)
                    path.symlink_to(saved)
                return real_open(target, flags, *args)

            with mock.patch.object(ldos.os, "open", side_effect=swap_before_open):
                with self.assertRaisesRegex(ldos.EvidenceError, "symbolic link"):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_controller_tid_and_process_membership_are_exact(self) -> None:
        mutations = ("controller-mask", "extra-child", "missing-worker")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                samples = [row["payload"] for row in rows if row["record_type"] == "sample"]
                if mutation == "controller-mask":
                    samples[0]["controller_process_state"]["task_affinities"]["40"] = [1]
                elif mutation == "extra-child":
                    samples[2]["descendant_pids"].append(99)
                    samples[2]["process_task_state"]["99"] = {
                        "start_time_ticks": 99,
                        "ticks": 1,
                        "task_affinities": {"99": [0]},
                    }
                else:
                    samples[2]["descendant_pids"].remove(42)
                    samples[2]["process_task_state"].pop("42")
                    samples[2]["worker_process_state"]["0"] = None
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_host_and_ready_schema_mutations_are_all_rejected(self) -> None:
        mutations = (
            "controller-leader", "prte-leader", "controller-ticks",
            "process-ticks", "ready-file", "governor", "negative-frequency",
            "frequency-key", "loadavg", "terminal-time",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                samples = [row["payload"] for row in rows if row["record_type"] == "sample"]
                sample = samples[0]
                if mutation == "controller-leader":
                    sample["controller_process_state"]["task_affinities"] = {"999": [0]}
                elif mutation == "prte-leader":
                    sample["process_task_state"]["41"]["task_affinities"] = {"999": [0]}
                elif mutation == "controller-ticks":
                    sample["controller_process_state"]["ticks"] = "1"
                elif mutation == "process-ticks":
                    sample["process_task_state"]["41"]["ticks"] = "1"
                elif mutation == "ready-file":
                    next(
                        row for row in rows if row["record_type"] == "ready"
                    )["payload"]["file"] = {"sha256": "c" * 64}
                elif mutation == "governor":
                    sample["governors"]["0"] = "arbitrary"
                elif mutation == "negative-frequency":
                    sample["current_frequency_khz"]["0"] = -1
                elif mutation == "frequency-key":
                    sample["current_frequency_khz"] = {"999": 1, "1": 1}
                elif mutation == "loadavg":
                    sample["loadavg"] = True
                else:
                    next(
                        row for row in rows if row["record_type"] == "terminal"
                    )["payload"]["closed_at_utc"] = 7
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_ready_file_digest_and_existence_are_revalidated(self) -> None:
        for mutation in ("forged-digest", "deleted-file"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                path, header = self.make_valid_journal(root)
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                ready = next(row["payload"] for row in rows if row["record_type"] == "ready")
                if mutation == "forged-digest":
                    ready["file"]["sha256"] = "d" * 64
                    self.rewrite_rows(path, rows)
                else:
                    (path.parent / "rank-0.ready").unlink()
                with self.assertRaisesRegex(ldos.EvidenceError, "ready task/file"):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_ready_file_name_swap_after_read_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path, header = self.make_valid_journal(root)
            ready_path = path.parent / "rank-0.ready"
            saved_path = root / "saved-ready"
            ready_inode = ready_path.stat().st_ino
            real_fstat = ldos.os.fstat
            ready_fstats = 0

            def swap_after_second_ready_fstat(descriptor):
                nonlocal ready_fstats
                info = real_fstat(descriptor)
                if info.st_ino == ready_inode:
                    ready_fstats += 1
                    if ready_fstats == 2:
                        ready_path.rename(saved_path)
                        ready_path.symlink_to(saved_path)
                return info

            with mock.patch.object(ldos.os, "fstat", side_effect=swap_after_second_ready_fstat):
                with self.assertRaisesRegex(ldos.EvidenceError, "ready task/file"):
                    ldos.validate_lane_journal(path, expected_header=header)
            self.assertEqual(ready_fstats, 2)

    def test_archived_nvml_missing_hash_and_relative_path_are_rejected(self) -> None:
        for mutation in ("missing", "content", "record-hash", "relative-path"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                journal_header = rows[0]["payload"]
                archive_path = (
                    path.parent.parent / "archive" / "system" / "lib"
                    / "libnvidia-ml.so.1"
                )
                if mutation == "missing":
                    archive_path.unlink()
                elif mutation == "content":
                    archive_path.chmod(0o600)
                    archive_path.write_bytes(b"tampered archived NVML")
                elif mutation == "record-hash":
                    journal_header["nvml_archive_identity"]["sha256"] = (
                        "0" * 64
                    )
                    self.rewrite_rows(path, rows)
                    header = copy.deepcopy(journal_header)
                else:
                    journal_header["nvml_archive_identity"]["relative_path"] = (
                        "../archive/system/lib/../forged"
                    )
                    self.rewrite_rows(path, rows)
                    header = copy.deepcopy(journal_header)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_coherently_hashed_non_elf_nvml_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _header = self.make_valid_journal(pathlib.Path(directory))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            header = rows[0]["payload"]
            archive_path = (
                path.parent.parent / "archive" / "system" / "lib"
                / "libnvidia-ml.so.1"
            )
            forged = b"not an ELF NVML library" * 8
            archive_path.chmod(0o600)
            archive_path.write_bytes(forged)
            archive_path.chmod(0o400)
            header["nvml_archive_identity"]["size_bytes"] = len(forged)
            header["nvml_archive_identity"]["sha256"] = (
                ldos.hashlib.sha256(forged).hexdigest()
            )
            self.rewrite_rows(path, rows)
            with self.assertRaises(ldos.EvidenceError):
                ldos.validate_lane_journal(
                    path, expected_header=copy.deepcopy(header)
                )

    def test_archived_nvml_name_swap_after_read_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path, header = self.make_valid_journal(root)
            archive_path = (
                path.parent.parent / "archive" / "system" / "lib"
                / "libnvidia-ml.so.1"
            )
            saved_path = archive_path.with_name(archive_path.name + ".saved")
            archive_inode = archive_path.stat().st_ino
            real_fstat = ldos.os.fstat
            archive_fstats = 0

            def swap_after_second_archive_fstat(descriptor):
                nonlocal archive_fstats
                info = real_fstat(descriptor)
                if info.st_ino == archive_inode:
                    archive_fstats += 1
                    if archive_fstats == 2:
                        archive_path.rename(saved_path)
                        archive_path.symlink_to(saved_path)
                return info

            with mock.patch.object(
                ldos.os, "fstat", side_effect=swap_after_second_archive_fstat
            ):
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)
            self.assertEqual(archive_fstats, 2)

    def test_archived_nvml_bytes_survive_evidence_tree_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            original = base / "original"
            relocated = base / "relocated"
            original.mkdir(mode=0o700)
            relocated.mkdir(mode=0o700)
            _path, header = self.make_valid_journal(original)
            shutil.copytree(original / "archive", relocated / "archive")
            relocated_control = relocated / "ldos-rank-1-control"
            relocated_control.mkdir(mode=0o700)
            shutil.rmtree(original / "archive")
            self.assertTrue(
                ldos._validate_archived_nvml_identity(
                    relocated_control / "telemetry.raw.jsonl",
                    header["nvml_archive_identity"],
                )
            )

    def test_two_distinct_gpu_contract_cannot_be_coherently_downgraded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, header = self.make_valid_journal(pathlib.Path(directory))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            journal_header = rows[0]["payload"]
            removed_uuid = journal_header["nvml_device_uuids"].pop()
            journal_header["expected_device_uuids"].pop()
            journal_header["nvml_baseline_process_utilization"].pop(removed_uuid)
            journal_header["nvml_baseline_current_processes"].pop(removed_uuid)
            for row in rows:
                if row["record_type"] == "sample":
                    row["payload"]["nvml"]["devices"] = [
                        device for device in row["payload"]["nvml"]["devices"]
                        if device["uuid"] != removed_uuid
                    ]
            self.rewrite_rows(path, rows)
            with self.assertRaisesRegex(ldos.EvidenceError, "two-GPU"):
                ldos.validate_lane_journal(
                    path, expected_header=copy.deepcopy(journal_header)
                )

    def test_coherent_header_mutations_cannot_weaken_standalone_validation(self) -> None:
        mutations = (
            "sample-period-float", "departed-lag-float", "online-not-string",
            "online-noncanonical", "online-omits-allowed", "nvml-forged-schema",
            "nvml-forged-hash", "nvml-forged-path", "nvml-forged-size",
            "nvml-forged-soname", "nvml-missing-symbol",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, _header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                header = rows[0]["payload"]
                if mutation == "sample-period-float":
                    header["sample_period_ns"] = 50_000_000.0
                elif mutation == "departed-lag-float":
                    header["departed_nvml_lag_limit_ns"] = float(
                        ldos.NVML_DEPARTED_CONTEXT_LAG_NS
                    )
                elif mutation == "online-not-string":
                    header["online_cpu_list"] = 17
                elif mutation == "online-noncanonical":
                    header["online_cpu_list"] = "0,1"
                elif mutation == "online-omits-allowed":
                    header["online_cpu_list"] = "0"
                elif mutation == "nvml-forged-schema":
                    header["nvml_archive_identity"] = {"forged": True}
                elif mutation == "nvml-forged-hash":
                    header["nvml_archive_identity"]["sha256"] = "0" * 64
                elif mutation == "nvml-forged-path":
                    header["nvml_archive_identity"]["relative_path"] = (
                        "../archive/system/lib/not-nvml.so"
                    )
                elif mutation == "nvml-forged-size":
                    header["nvml_archive_identity"]["size_bytes"] += 1
                elif mutation == "nvml-forged-soname":
                    header["nvml_archive_identity"]["elf_nvml_identity"][
                        "soname"
                    ] = "libforged.so.1"
                else:
                    header["nvml_archive_identity"]["elf_nvml_identity"][
                        "required_symbols"
                    ].pop()
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(
                        path, expected_header=copy.deepcopy(header)
                    )

    def test_controller_worker_overlap_is_rejected_after_coherent_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _header = self.make_valid_journal(pathlib.Path(directory))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            header = rows[0]["payload"]
            core = {"package_id": 0, "core_id": 0, "logical_cpus": [0]}
            header.update({
                "controller_cpu_mask": [0],
                "worker_cpu_masks": [[0]],
                "online_cpu_list": "0",
                "allowed_cpus": [0],
                "physical_topology_plan": {
                    "allowed_cpus": [0],
                    "available_full_cores": [copy.deepcopy(core)],
                    "controller_core": copy.deepcopy(core),
                    "worker_cores": [copy.deepcopy(core)],
                },
            })
            ready = next(row["payload"] for row in rows if row["record_type"] == "ready")
            ready["task_affinities"] = {"42": [0]}
            for row in rows:
                if row["record_type"] != "sample":
                    continue
                sample = row["payload"]
                sample["cpu_counters"] = {"0": sample["cpu_counters"]["0"]}
                sample["current_frequency_khz"] = {
                    "0": sample["current_frequency_khz"]["0"]
                }
                sample["governors"] = {"0": sample["governors"]["0"]}
                state = sample["worker_process_state"]["0"]
                if state is not None:
                    state["task_affinities"] = {"42": [0]}
                if "42" in sample["process_task_state"]:
                    sample["process_task_state"]["42"]["task_affinities"] = {
                        "42": [0]
                    }
            self.refresh_ready_file(path, rows)
            self.rewrite_rows(path, rows)
            with self.assertRaisesRegex(ldos.EvidenceError, "masks overlap"):
                ldos.validate_lane_journal(path, expected_header=copy.deepcopy(header))

    def test_nvml_timing_scalar_cursor_and_outsider_schema_are_exact(self) -> None:
        mutations = (
            "duplicate-uuid", "boolean-scalar", "utilization-overflow",
            "stale-cursor", "late", "slow", "outsider-mismatch",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                samples = [row["payload"] for row in rows if row["record_type"] == "sample"]
                nvml = samples[0]["nvml"]
                device = nvml["devices"][0]
                if mutation == "duplicate-uuid":
                    nvml["devices"][1]["uuid"] = device["uuid"]
                elif mutation == "boolean-scalar":
                    device["power_mw"] = True
                elif mutation == "utilization-overflow":
                    device["utilization"]["gpu_percent"] = 101
                elif mutation == "stale-cursor":
                    utilization = {
                        "pid": 42,
                        "timestamp_us": 7,
                        "sm_utilization_percent": 1,
                        "memory_utilization_percent": 1,
                        "encoder_utilization_percent": 0,
                        "decoder_utilization_percent": 0,
                        "observed_start_time_ticks": 10,
                    }
                    samples[0]["nvml"]["devices"][0][
                        "process_utilization_since_cursor"
                    ] = [copy.deepcopy(utilization)]
                    samples[1]["nvml"]["devices"][0][
                        "process_utilization_since_cursor"
                    ] = [copy.deepcopy(utilization)]
                elif mutation == "late":
                    nvml["started_monotonic_ns"] = nvml["scheduled_monotonic_ns"] + 50_000_000
                    nvml["finished_monotonic_ns"] = nvml["started_monotonic_ns"] + 1
                    nvml["lateness_ns"] = 50_000_000
                    samples[0]["sample_finished_monotonic_ns"] = nvml["finished_monotonic_ns"]
                elif mutation == "slow":
                    nvml["finished_monotonic_ns"] = nvml["started_monotonic_ns"] + 50_000_000
                    nvml["duration_ns"] = 50_000_000
                    samples[0]["sample_finished_monotonic_ns"] = nvml["finished_monotonic_ns"]
                else:
                    nvml["outsiders"] = [{"forged": True}]
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_psi_and_nvml_abi_ranges_are_exact(self) -> None:
        mutations = (
            "psi-string", "psi-infinite", "psi-total-overflow",
            "gpu-instance-overflow", "graphics-clock-overflow",
            "memory-overflow", "utilization-pid-overflow",
            "utilization-timestamp-overflow",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                sample = next(row["payload"] for row in rows if row["record_type"] == "sample")
                device = sample["nvml"]["devices"][0]
                if mutation == "psi-string":
                    sample["pressure"]["cpu"]["some"]["avg10"] = "0.0"
                elif mutation == "psi-infinite":
                    sample["pressure"]["cpu"]["some"]["avg10"] = 1e400
                elif mutation == "psi-total-overflow":
                    sample["pressure"]["cpu"]["some"]["total"] = 1 << 64
                elif mutation == "gpu-instance-overflow":
                    device["process_lists"]["compute"][0]["gpu_instance_id"] = 1 << 100
                elif mutation == "graphics-clock-overflow":
                    device["graphics_clock_mhz"] = 1 << 100
                elif mutation == "memory-overflow":
                    device["memory_bytes"] = {
                        "total": 1 << 64, "free": 1 << 64, "used": 0,
                    }
                else:
                    utilization = {
                        "pid": 42,
                        "timestamp_us": 7,
                        "sm_utilization_percent": 0,
                        "memory_utilization_percent": 0,
                        "encoder_utilization_percent": 0,
                        "decoder_utilization_percent": 0,
                        "observed_start_time_ticks": 10,
                    }
                    if mutation == "utilization-pid-overflow":
                        utilization["pid"] = 1 << 32
                    else:
                        utilization["timestamp_us"] = 1 << 64
                    device["process_utilization_since_cursor"] = [utilization]
                self.rewrite_rows(path, rows)
                with self.assertRaises(ldos.EvidenceError):
                    ldos.validate_lane_journal(path, expected_header=header)

    def test_departed_nvml_context_lag_is_bounded_and_drain_must_clear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, header = self.make_valid_journal(pathlib.Path(directory))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            pre_departure = next(
                row["payload"] for row in rows
                if row["record_type"] == "sample"
                and row["payload"]["monotonic_ns"] == 300_000_000
            )
            pre_departure["nvml"]["devices"][0]["process_lists"]["compute"] = [{
                "pid": 42,
                "used_gpu_memory_bytes": 1,
                "gpu_instance_id": (1 << 32) - 1,
                "compute_instance_id": (1 << 32) - 1,
                "observed_start_time_ticks": 10,
            }]
            departed_sample = next(
                row["payload"] for row in rows
                if row["record_type"] == "sample"
                and row["payload"]["departed_worker_ranks"]
                and not row["payload"]["after_process_exit"]
            )
            departed_sample["nvml"]["devices"][0]["process_lists"]["compute"] = [{
                "pid": 42,
                "used_gpu_memory_bytes": 1,
                "gpu_instance_id": (1 << 32) - 1,
                "compute_instance_id": (1 << 32) - 1,
                "observed_start_time_ticks": 10,
            }]
            self.rewrite_rows(path, rows)
            result = ldos.validate_lane_journal(path, expected_header=header)
            self.assertEqual(
                result["nvml_absence_pidfd_observation_order_by_rank"],
                ["pidfd-observed-first"],
            )
            self.assertEqual(
                result[
                    "nvml_absence_observation_to_pidfd_observation_ns_by_rank"
                ],
                [None],
            )
            self.assertEqual(
                result[
                    "pidfd_observation_to_first_nvml_absence_observation_ns_by_rank"
                ],
                [55_000_002],
            )

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            next(
                row for row in rows if row["record_type"] == "worker-exit"
            )["payload"]["monotonic_ns"] = 190_000_000
            self.rewrite_rows(path, rows)
            with self.assertRaises(ldos.EvidenceError):
                ldos.validate_lane_journal(path, expected_header=header)

    def test_departed_lag_uses_nvml_finish_at_exact_boundary_and_plus_one(self) -> None:
        limit = ldos.NVML_DEPARTED_CONTEXT_LAG_NS
        self.assertTrue(ldos._within_departed_nvml_lag(100 + limit, 100))
        self.assertFalse(ldos._within_departed_nvml_lag(101 + limit, 100))
        for over_limit in (False, True):
            with self.subTest(over_limit=over_limit), tempfile.TemporaryDirectory() as directory:
                path, header = self.make_valid_journal(pathlib.Path(directory))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                compute_row = {
                    "pid": 42,
                    "used_gpu_memory_bytes": 1,
                    "gpu_instance_id": (1 << 32) - 1,
                    "compute_instance_id": (1 << 32) - 1,
                    "observed_start_time_ticks": 10,
                }
                pre_departure = next(
                    row["payload"] for row in rows
                    if row["record_type"] == "sample"
                    and row["payload"]["monotonic_ns"] == 300_000_000
                )
                pre_departure["nvml"]["devices"][0]["process_lists"][
                    "compute"
                ] = [copy.deepcopy(compute_row)]
                worker_exit = next(
                    row["payload"] for row in rows
                    if row["record_type"] == "worker-exit"
                )
                worker_exit["monotonic_ns"] = (
                    349_999_998 - int(over_limit)
                )
                departed = next(
                    row["payload"] for row in rows
                    if row["record_type"] == "sample"
                    and row["payload"]["monotonic_ns"] == 350_000_000
                )
                departed["nvml"]["devices"][0]["process_lists"]["compute"] = [
                    copy.deepcopy(compute_row)
                ]
                departed["nvml"].update(
                    {
                        "started_monotonic_ns": 399_999_999,
                        "finished_monotonic_ns": 449_999_998,
                        "lateness_ns": 49_999_999,
                        "duration_ns": 49_999_999,
                    }
                )
                departed["sample_finished_monotonic_ns"] = 449_999_999
                process_exit = next(
                    row["payload"] for row in rows
                    if row["record_type"] == "process-exit"
                )
                process_exit["monotonic_ns"] = 449_999_999
                drains = [
                    row["payload"] for row in rows
                    if row["record_type"] == "sample"
                    and row["payload"]["after_process_exit"]
                ]
                for sample, start_ns in zip(
                    drains, (449_999_999, 450_000_003)
                ):
                    sample["monotonic_ns"] = start_ns
                    sample["sample_started_monotonic_ns"] = start_ns
                    sample["sample_finished_monotonic_ns"] = start_ns + 3
                    nvml = sample["nvml"]
                    nvml["started_monotonic_ns"] = start_ns
                    nvml["finished_monotonic_ns"] = start_ns + 2
                    nvml["lateness_ns"] = (
                        start_ns - nvml["scheduled_monotonic_ns"]
                    )
                    nvml["duration_ns"] = 2
                self.rewrite_rows(path, rows)
                if over_limit:
                    with self.assertRaisesRegex(
                        ldos.EvidenceError, "NVML lifecycle"
                    ):
                        ldos.validate_lane_journal(path, expected_header=header)
                else:
                    ldos.validate_lane_journal(path, expected_header=header)

        with tempfile.TemporaryDirectory() as directory:
            path, header = self.make_valid_journal(pathlib.Path(directory))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            drain = next(
                row["payload"] for row in rows
                if row["record_type"] == "sample" and row["payload"]["after_process_exit"]
            )
            drain["nvml"]["devices"][0]["process_lists"]["compute"] = [{
                "pid": 42,
                "used_gpu_memory_bytes": 1,
                "gpu_instance_id": (1 << 32) - 1,
                "compute_instance_id": (1 << 32) - 1,
                "observed_start_time_ticks": None,
            }]
            self.rewrite_rows(path, rows)
            with self.assertRaises(ldos.EvidenceError):
                ldos.validate_lane_journal(path, expected_header=header)

    def test_constructor_closes_descriptor_when_initial_sync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "telemetry.raw.jsonl"
            before = set(pathlib.Path("/proc/self/fd").iterdir())
            with mock.patch.object(ldos.os, "fsync", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    ldos.RawTelemetryJournal(path, {})
            after = set(pathlib.Path("/proc/self/fd").iterdir())
            self.assertEqual(after, before)

    def test_partial_append_is_rolled_back_before_error_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "telemetry.raw.jsonl"
            journal = ldos.RawTelemetryJournal(path, {"nonce": "a" * 64})
            real_write = ldos.os.write
            calls = 0

            def partial_then_error(descriptor, data):
                nonlocal calls
                if descriptor == journal.descriptor:
                    calls += 1
                    if calls == 1:
                        return real_write(descriptor, data[: max(1, len(data) // 2)])
                    raise OSError("partial write")
                return real_write(descriptor, data)

            with mock.patch.object(ldos.os, "write", side_effect=partial_then_error):
                with self.assertRaisesRegex(OSError, "partial write"):
                    journal.append("sample", {"monotonic_ns": 7})
            journal.close("error", "append failed")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["sequence"] for row in rows], [0, 1])
            self.assertEqual([row["record_type"] for row in rows], ["header", "terminal"])
            self.assertEqual(rows[-1]["payload"]["state"], "error")

    def test_partial_terminal_sync_failure_is_bound_by_global_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "evidence"
            ldos._claim_output(output, "run")
            control = output / "control"
            control.mkdir()
            path = control / "telemetry.raw.jsonl"
            journal = ldos.RawTelemetryJournal(path, {"nonce": "a" * 64})
            with mock.patch.object(ldos.os, "fdatasync", side_effect=OSError("sync")):
                with self.assertRaisesRegex(OSError, "sync"):
                    journal.close("sealed")
            self.assertTrue(journal.closed)
            ldos._publish_failure(output, "run", RuntimeError("journal close failed"))
            failure = json.loads((output / "FAILED").read_text())
            paths = {record["path"] for record in failure["partial_artifacts"]}
            self.assertIn("control/telemetry.raw.jsonl", paths)
            self.assertEqual(json.loads((output / "TERMINAL").read_text())["state"], "FAILED")

    def test_error_journal_is_exclusive_durable_and_sequenced(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            ldos.os, "fdatasync", wraps=ldos.os.fdatasync
        ) as durable:
            path = pathlib.Path(directory) / "telemetry.raw.jsonl"
            journal = ldos.RawTelemetryJournal(path, {"nonce": "a" * 64})
            journal.append("sample", {"monotonic_ns": 7})
            record = journal.close("error", "injected")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(record["sha256"], ldos.EVIDENCE.sha256_file(path))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["sequence"] for row in rows], [0, 1, 2])
            self.assertEqual(rows[-1]["payload"]["state"], "error")
            self.assertTrue(durable.called)

    def test_journal_refuses_reuse_and_post_close_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "telemetry.raw.jsonl"
            journal = ldos.RawTelemetryJournal(path, {})
            with self.assertRaises(FileExistsError):
                ldos.RawTelemetryJournal(path, {})
            journal.close("complete")
            with self.assertRaisesRegex(ldos.EvidenceError, "closed"):
                journal.append("sample", {})


class MutationSentinelTests(unittest.TestCase):
    def test_swap_and_restore_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = root / "component.so"
            target.write_bytes(b"trusted")
            sentinel = ldos.MutationSentinel(root)
            try:
                original = root / "component.original"
                target.rename(original)
                target.write_bytes(b"hostile")
                target.unlink()
                original.rename(target)
                events = sentinel.finish()
            finally:
                sentinel.close()
            self.assertTrue(events)
            self.assertTrue(any("component.so" in event["path"] for event in events))


class MpiRuntimeProofTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.prefix = self.repo / "prefix"
        (self.prefix / "bin").mkdir(parents=True)
        (self.prefix / "lib").mkdir()
        self.runtime_files = {
            "prte": self.prefix / "bin" / "prte",
            "mpi": self.prefix / "lib" / "libmpi.so.1",
            "pmix": self.prefix / "lib" / "libpmix.so.2",
        }
        for label, path in self.runtime_files.items():
            path.write_bytes((label + " runtime bytes").encode())
        self.archive_root = self.root / "archive"
        self.archive_root.mkdir()
        archive_paths = {
            "timeout": self.archive_root / "timeout",
            "mpiexec": self.archive_root / "mpiexec",
            "loader": self.archive_root / "ld-linux.so",
            "gpu_step_db": self.archive_root / "gpu-step-db",
            "libmeep": self.archive_root / "libmeep.so.38.0.0",
            "runtime_closure": self.archive_root / "runtime-closure.json",
        }
        for label, path in archive_paths.items():
            path.write_bytes(
                b'{"groups":{}}' if label == "runtime_closure" else (label + " bytes").encode()
            )
        libmeep_soname = self.archive_root / "libmeep.so.38"
        os.link(archive_paths["libmeep"], libmeep_soname)
        archive_paths["libmeep_soname"] = libmeep_soname
        self.archived = {
            label: ldos.StableFile.open(label, path)
            for label, path in archive_paths.items()
        }
        self.addCleanup(
            lambda: [handle.close() for handle in self.archived.values()]
        )
        manifest_files = []
        for path in self.runtime_files.values():
            manifest_files.append(
                {
                    "path": path.relative_to(self.prefix).as_posix(),
                    "sha256": ldos.EVIDENCE.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        self.receipt = {
            "manifests": {
                "installed_environment": {
                    "root": str(self.prefix),
                    "files": manifest_files,
                }
            }
        }
        paths = [
            self.archived["timeout"].path,
            self.archived["mpiexec"].path,
            self.archived["loader"].path,
            self.archived["gpu_step_db"].path,
            *self.runtime_files.values(),
        ]
        mappings = []
        for path in paths:
            info = path.stat()
            mappings.append(
                {
                    "path": str(path),
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "size_bytes": info.st_size,
                    "mtime_ns": info.st_mtime_ns,
                    "ctime_ns": info.st_ctime_ns,
                    "sha256": ldos.EVIDENCE.sha256_file(path),
                    "segments": [{"permissions": "r--p", "offset": 0}],
                }
            )
        mapping_by_path = {item["path"]: item for item in mappings}

        def mappings_for(*selected_paths: pathlib.Path) -> list[dict]:
            return [
                copy.deepcopy(mapping_by_path[str(path)]) for path in selected_paths
            ]
        def executable_record(path: pathlib.Path) -> dict:
            handle = ldos.StableFile.open("executable fixture", path)
            try:
                return handle.record()
            finally:
                handle.close()
        self.library_path = f"{self.archive_root}:{self.prefix / 'lib'}"
        full_command = ldos._mpi_command(
            self.archived["timeout"],
            self.archived["mpiexec"],
            self.archived["loader"],
            self.archived["gpu_step_db"],
            1,
            600,
            self.library_path,
            [{1}],
        )
        worker_command = [
            self.archived["loader"].proc_path,
            "--inhibit-rpath",
            "",
            "--library-path",
            self.library_path,
            self.archived["gpu_step_db"].proc_path,
        ]
        self.result = {
            "command": full_command,
            "root_pid": 100,
            "process_started_monotonic_ns": 0,
            "process_finished_monotonic_ns": 152_000_001,
            "runtime_observation_attempts": [
                {
                    "start_monotonic_ns": index * 50_000_000 + 1,
                    "duration_ns": 1_000_000,
                    "error": None,
                    "process_count": 4,
                }
                for index in range(4)
            ],
            "runtime_observations": [
                {
                    "pid": 100,
                    "parent_pid": 1,
                    "start_time_ticks": 10,
                    "mapping_epoch_sha256": "e" * 64,
                    "executable": executable_record(self.archived["timeout"].path),
                    "command_line": full_command,
                    "mappings": mappings_for(self.archived["timeout"].path),
                },
                {
                    "pid": 101,
                    "parent_pid": 100,
                    "start_time_ticks": 11,
                    "mapping_epoch_sha256": "f" * 64,
                    "executable": executable_record(self.archived["loader"].path),
                    "command_line": full_command[4:],
                    "mappings": mappings_for(
                        self.archived["loader"].path,
                        self.archived["mpiexec"].path,
                        self.runtime_files["mpi"],
                    ),
                },
                {
                    "pid": 102,
                    "parent_pid": 101,
                    "start_time_ticks": 12,
                    "mapping_epoch_sha256": "a" * 64,
                    "executable": executable_record(self.runtime_files["prte"]),
                    "command_line": [str(self.runtime_files["prte"])],
                    "mappings": mappings_for(
                        self.runtime_files["prte"],
                        self.runtime_files["pmix"],
                    ),
                },
                {
                    "pid": 103,
                    "parent_pid": 102,
                    "start_time_ticks": 13,
                    "mapping_epoch_sha256": "b" * 64,
                    "executable": executable_record(self.archived["loader"].path),
                    "command_line": worker_command,
                    "mappings": mappings_for(
                        self.archived["loader"].path,
                        self.archived["gpu_step_db"].path,
                        self.runtime_files["mpi"],
                    ),
                },
            ]
        }

    def verify(self, result=None):
        return ldos.verify_mpi_runtime_observations(
            self.result if result is None else result,
            repo=self.repo,
            receipt=self.receipt,
            archived=self.archived,
            expected_ranks=1,
            expected_library_path=self.library_path,
            expected_timeout_seconds=600,
            expected_worker_cpu_masks=[{1}],
        )

    def test_valid_actual_execution_proof(self) -> None:
        proof = self.verify()
        self.assertTrue(all(proof["actual_execution_proofs"].values()))
        self.assertTrue(proof["verified_executable_identities"])
        self.assertTrue(proof["verified_mapping_identities"])
        self.assertTrue(
            all(
                re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                and item["trust_source"]
                for key in (
                    "verified_executable_identities",
                    "verified_mapping_identities",
                )
                for item in proof[key]
            )
        )

    def test_missing_prte_pmix_or_mpi_is_rejected(self) -> None:
        for token, message in (("/prte", "PRTE"), ("libpmix", "PMIx"), ("libmpi", "MPI")):
            result = copy.deepcopy(self.result)
            for observation in result["runtime_observations"]:
                observation["mappings"] = [
                    item for item in observation["mappings"] if token not in item["path"]
                ]
            with self.subTest(token=token):
                with self.assertRaisesRegex(ldos.EvidenceError, message):
                    self.verify(result)

    def test_libmpi_and_libpmix_sonames_cannot_substitute_for_each_other(self) -> None:
        result = copy.deepcopy(self.result)
        worker = result["runtime_observations"][3]
        worker["mappings"] = [
            item for item in worker["mappings"]
            if pathlib.Path(item["path"]).name != "libmpi.so.1"
        ]
        pmix_mapping = next(
            item for item in result["runtime_observations"][2]["mappings"]
            if pathlib.Path(item["path"]).name == "libpmix.so.2"
        )
        worker["mappings"].append(copy.deepcopy(pmix_mapping))
        with self.assertRaisesRegex(ldos.EvidenceError, "MPI"):
            self.verify(result)

        result = copy.deepcopy(self.result)
        prte = result["runtime_observations"][2]
        prte["mappings"] = [
            item for item in prte["mappings"]
            if pathlib.Path(item["path"]).name != "libpmix.so.2"
        ]
        mpi_mapping = next(
            item for item in result["runtime_observations"][1]["mappings"]
            if pathlib.Path(item["path"]).name == "libmpi.so.1"
        )
        prte["mappings"].append(copy.deepcopy(mpi_mapping))
        with self.assertRaisesRegex(ldos.EvidenceError, "PMIx"):
            self.verify(result)

    def test_missing_archived_execution_mapping_is_rejected(self) -> None:
        for label in ("timeout", "loader", "gpu_step_db"):
            result = copy.deepcopy(self.result)
            path = str(self.archived[label].path)
            for observation in result["runtime_observations"]:
                observation["mappings"] = [
                    item for item in observation["mappings"] if item["path"] != path
                ]
            with self.subTest(label=label):
                with self.assertRaises(ldos.EvidenceError):
                    self.verify(result)

        result = copy.deepcopy(self.result)
        mpiexec_path = str(self.archived["mpiexec"].path)
        for observation in result["runtime_observations"]:
            observation["mappings"] = [
                item for item in observation["mappings"]
                if item["path"] != mpiexec_path
            ]
        proof = self.verify(result)
        self.assertFalse(
            proof["actual_execution_proofs"]["archived_mpiexec_same_epoch_observed"]
        )

    def test_missing_held_fd_command_line_is_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        result["runtime_observations"][1]["command_line"] = ["mpiexec"]
        result["runtime_observations"][3]["command_line"] = ["gpu-step-db"]
        with self.assertRaises(ldos.EvidenceError):
            self.verify(result)

    def test_mpi_command_preserves_noncontiguous_linux_cpu_masks(self) -> None:
        command = ldos._mpi_command(
            self.archived["timeout"], self.archived["mpiexec"],
            self.archived["loader"], self.archived["gpu_step_db"],
            2, 600, self.library_path, [{4, 20}, {7, 31}],
        )
        self.assertIn("slot", command)
        self.assertIn("none", command)
        self.assertIn("MEEP_GPU_LDOS_CPU_LIST=4,20", command)
        self.assertIn("MEEP_GPU_LDOS_CPU_LIST=7,31", command)
        self.assertFalse(any("pe-list" in argument for argument in command))

    def test_persisted_timeout_command_must_match_fixed_grammar(self) -> None:
        result = copy.deepcopy(self.result)
        result["command"][3] = "601s"
        result["runtime_observations"][0]["command_line"] = result["command"]
        with self.assertRaisesRegex(ldos.EvidenceError, "fixed held-FD grammar"):
            self.verify(result)

    def test_split_argv_and_mapping_epochs_are_rejected(self) -> None:
        for observation_index, missing_label in ((3, "gpu_step_db"),):
            result = copy.deepcopy(self.result)
            original = result["runtime_observations"][observation_index]
            incomplete_argv_epoch = copy.deepcopy(original)
            missing_path = str(self.archived[missing_label].path)
            incomplete_argv_epoch["mappings"] = [
                item for item in incomplete_argv_epoch["mappings"]
                if item["path"] != missing_path
            ]
            incomplete_map_epoch = copy.deepcopy(original)
            incomplete_map_epoch["mapping_epoch_sha256"] = "c" * 64
            incomplete_map_epoch["command_line"] = ["nested-argument-spoof"]
            result["runtime_observations"][observation_index:observation_index + 1] = [
                incomplete_argv_epoch,
                incomplete_map_epoch,
            ]
            with self.subTest(role=missing_label):
                with self.assertRaisesRegex(ldos.EvidenceError, "same-epoch|split"):
                    self.verify(result)

    def test_transient_launcher_may_transition_to_prte_same_identity(self) -> None:
        result = copy.deepcopy(self.result)
        prte = result["runtime_observations"][2]
        prte.update({"pid": 101, "parent_pid": 100, "start_time_ticks": 11})
        result["runtime_observations"][3]["parent_pid"] = 101
        proof = self.verify(result)
        self.assertTrue(
            proof["actual_execution_proofs"]["archived_mpiexec_same_epoch_observed"]
        )

        result = copy.deepcopy(self.result)
        result["runtime_observations"].pop(1)
        result["runtime_observations"][1]["parent_pid"] = 100
        result["runtime_observations"][2]["parent_pid"] = 102
        proof = self.verify(result)
        self.assertFalse(
            proof["actual_execution_proofs"]["archived_mpiexec_same_epoch_observed"]
        )

    def test_executable_and_parent_role_mismatch_are_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        result["runtime_observations"][3]["executable"] = copy.deepcopy(
            result["runtime_observations"][0]["executable"]
        )
        with self.assertRaises(ldos.EvidenceError):
            self.verify(result)

        result = copy.deepcopy(self.result)
        result["runtime_observations"][2]["parent_pid"] = 103
        with self.assertRaisesRegex(ldos.EvidenceError, "cycle|parent chain"):
            self.verify(result)

    def test_worker_cardinality_must_equal_lane_ranks(self) -> None:
        result = copy.deepcopy(self.result)
        result["runtime_observations"].pop()
        with self.assertRaisesRegex(ldos.EvidenceError, "cardinality"):
            self.verify(result)

    def test_runtime_observer_gap_is_fail_closed(self) -> None:
        result = copy.deepcopy(self.result)
        result["runtime_observation_attempts"][2]["start_monotonic_ns"] = (
            result["runtime_observation_attempts"][1]["start_monotonic_ns"]
            + 300_000_000
        )
        result["runtime_observation_attempts"][3]["start_monotonic_ns"] = (
            result["runtime_observation_attempts"][2]["start_monotonic_ns"]
            + 50_000_000
        )
        result["process_finished_monotonic_ns"] = (
            result["runtime_observation_attempts"][3]["start_monotonic_ns"]
            + result["runtime_observation_attempts"][3]["duration_ns"]
        )
        with self.assertRaisesRegex(ldos.EvidenceError, "sampling gap"):
            self.verify(result)

        result = copy.deepcopy(self.result)
        result["process_finished_monotonic_ns"] += 300_000_000
        with self.assertRaisesRegex(ldos.EvidenceError, "sampling gap"):
            self.verify(result)

    def test_unbound_mapping_is_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        path = self.root / "unbound-pmix.so"
        path.write_bytes(b"unbound")
        info = path.stat()
        result["runtime_observations"][0]["mappings"].append(
            {
                "path": str(path), "device": info.st_dev, "inode": info.st_ino,
                "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns, "sha256": ldos.EVIDENCE.sha256_file(path),
            }
        )
        with self.assertRaisesRegex(ldos.EvidenceError, "unbound"):
            self.verify(result)

    def test_hash_or_identity_mismatch_is_rejected(self) -> None:
        for field, value, message in (
            ("size_bytes", 0, "changed after execution"),
            ("inode", 1, "changed after execution"),
        ):
            result = copy.deepcopy(self.result)
            result["runtime_observations"][0]["mappings"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ldos.EvidenceError, message):
                    self.verify(result)


class ControllerBootstrapTests(unittest.TestCase):
    def test_authoritative_entrypoint_requires_isolated_python(self) -> None:
        direct = subprocess.run(
            [str(MODULE_PATH), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(direct.returncode, 2)
        self.assertIn("requires Python -I -S", direct.stderr)

        isolated = subprocess.run(
            [sys.executable, "-I", "-S", str(MODULE_PATH), "--help"],
            env={"HOME": "/nonexistent", "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertNotEqual(isolated.returncode, 2)
        self.assertNotIn("requires Python -I -S", isolated.stderr)

    def test_isolated_entrypoint_does_not_execute_sitecustomize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            marker = root / "sitecustomize-ran"
            (root / "sitecustomize.py").write_text(
                f"from pathlib import Path;Path({str(marker)!r}).write_text('bad')\n",
                encoding="utf-8",
            )
            environment = {
                "HOME": "/nonexistent",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(root),
            }
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(MODULE_PATH), "--help"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("rejects ambient", completed.stderr)
            self.assertFalse(marker.exists())

    def test_fresh_import_does_not_execute_helpers(self) -> None:
        name = "gpmeep_ldos_prebootstrap_probe"
        spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            self.assertIsNone(module.EVIDENCE)
            self.assertIsNone(module.PROVENANCE)
            with self.assertRaisesRegex(module.EvidenceError, "before controller bootstrap"):
                module._initialize_verified_helpers()
        finally:
            sys.modules.pop(name, None)

    def test_spoofed_self_reported_fd_is_rejected_against_git_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            scripts = repo / "scripts"
            scripts.mkdir()
            descriptors: list[int] = []
            sources: dict[str, dict] = {}
            environment: dict[str, str] = {}
            try:
                for source_name in ldos.SOURCE_HELPERS:
                    path = scripts / source_name
                    path.write_bytes(b"hostile self-reported bytes")
                    descriptor = os.open(path, os.O_RDONLY)
                    descriptors.append(descriptor)
                    info = os.fstat(descriptor)
                    sources[source_name] = {
                        "path": str(path),
                        "size_bytes": info.st_size,
                        "sha256": ldos._sha256_fd(descriptor),
                        "fingerprint": list(ldos._fingerprint(info)),
                    }
                    environment[ldos._controller_source_environment_key(source_name)] = str(descriptor)
                metadata = {
                    "repo": str(repo),
                    "git_commit": "a" * 40,
                    "sources": sources,
                    "python_interpreter": {},
                }
                environment["GPMEEP_CONTROLLER_BOOTSTRAP_METADATA"] = compact(metadata)

                def git_result(_repo, *arguments):
                    if arguments[:2] == ("rev-parse", "--show-toplevel"):
                        return (str(repo) + "\n").encode()
                    if arguments[:2] == ("rev-parse", "HEAD"):
                        return b"a" * 40 + b"\n"
                    if arguments and arguments[0] == "status":
                        return b""
                    if arguments and arguments[0] == "show":
                        return b"clean committed bytes"
                    raise AssertionError(arguments)

                with (
                    mock.patch.dict(os.environ, environment, clear=False),
                    mock.patch.object(ldos, "_git_command", side_effect=git_result),
                ):
                    with self.assertRaisesRegex(ldos.EvidenceError, "differ from Git HEAD"):
                        ldos._controller_bootstrap(["--authoritative-repo", str(repo)])
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

    def test_authoritative_repo_redirect_and_duplicate_flag_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            with self.assertRaisesRegex(ldos.EvidenceError, "exactly one"):
                ldos._bootstrap_authoritative_repo(
                    ["--authoritative-repo", first, "--authoritative-repo", second]
                )

    def test_alternate_committed_source_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            alternate = repo / "alternate"
            scripts = repo / "scripts"
            alternate.mkdir()
            scripts.mkdir()
            for source_name in ldos.SOURCE_HELPERS:
                (alternate / source_name).write_bytes(b"committed alternate")
                (scripts / source_name).write_bytes(b"canonical")

            def git_result(_repo, *arguments):
                if arguments[:2] == ("rev-parse", "--show-toplevel"):
                    return (str(repo) + "\n").encode()
                if arguments[:2] == ("rev-parse", "HEAD"):
                    return b"a" * 40 + b"\n"
                if arguments and arguments[0] == "status":
                    return b""
                raise AssertionError(arguments)

            with (
                mock.patch.object(ldos, "SCRIPT_DIRECTORY", alternate),
                mock.patch.object(ldos, "_git_command", side_effect=git_result),
            ):
                with self.assertRaisesRegex(ldos.EvidenceError, "not exact"):
                    ldos._controller_bootstrap(
                        ["--authoritative-repo", str(repo)]
                    )

    def test_claimed_python_fd_must_be_current_interpreter(self) -> None:
        repo = SCRIPTS.parent.resolve()
        descriptors: list[int] = []
        sources: dict[str, dict] = {}
        environment: dict[str, str] = {}
        for source_name in ldos.SOURCE_HELPERS:
            path = SCRIPTS / source_name
            descriptor = os.open(path, os.O_RDONLY)
            descriptors.append(descriptor)
            info = os.fstat(descriptor)
            sources[source_name] = {
                "path": str(path),
                "size_bytes": info.st_size,
                "sha256": ldos._sha256_fd(descriptor),
                "fingerprint": list(ldos._fingerprint(info)),
            }
            environment[ldos._controller_source_environment_key(source_name)] = str(descriptor)
        other_python = pathlib.Path("/usr/bin/true").resolve(strict=True)
        python_descriptor = os.open(other_python, os.O_RDONLY)
        descriptors.append(python_descriptor)
        python_info = os.fstat(python_descriptor)
        metadata = {
            "repo": str(repo),
            "git_commit": "a" * 40,
            "sources": sources,
            "python_interpreter": {
                "path": str(other_python),
                "size_bytes": python_info.st_size,
                "sha256": ldos._sha256_fd(python_descriptor),
                "fingerprint": list(ldos._fingerprint(python_info)),
            },
        }
        environment["GPMEEP_CONTROLLER_BOOTSTRAP_METADATA"] = compact(metadata)
        environment["GPMEEP_CONTROLLER_PYTHON_FD"] = str(python_descriptor)

        def git_result(_repo, *arguments):
            if arguments[:2] == ("rev-parse", "--show-toplevel"):
                return (str(repo) + "\n").encode()
            if arguments[:2] == ("rev-parse", "HEAD"):
                return b"a" * 40 + b"\n"
            if arguments and arguments[0] == "status":
                return b""
            if arguments and arguments[0] == "show":
                relative = arguments[1].split("HEAD:", 1)[1]
                return (repo / relative).read_bytes()
            raise AssertionError(arguments)

        try:
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(ldos, "_git_command", side_effect=git_result),
            ):
                with self.assertRaisesRegex(ldos.EvidenceError, "current Python interpreter"):
                    ldos._controller_bootstrap(["--authoritative-repo", str(repo)])
        finally:
            for descriptor in descriptors:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


class ReceiptReverificationTests(unittest.TestCase):
    def test_initial_receipt_contract_binds_id_kind_paths_and_mpiexec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            repo = root / "repo"
            repo.mkdir()
            receipt_path = root / "receipt.json"
            receipt_path.write_text("{}", encoding="utf-8")
            paths = {
                "worker": root / "gpu-step-db",
                "libmeep": root / "libmeep.so.38.0.0",
                "fd_allocation_shim": root / "libgpmeep-fd-allocation-shim.so.1",
                "mca": root / "mca.conf",
                "mpiexec": root / "mpiexec",
                "timeout": root / "timeout",
                "runtime_closure": root / "runtime-closure.json",
            }
            for name, path in paths.items():
                path.write_text(name, encoding="utf-8")

            def record(path: pathlib.Path) -> dict:
                return {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": ldos.EVIDENCE.sha256_file(path),
                }

            receipt = {
                "build_kind": "cuda-mpi-python-fp32",
                "receipt_id": "a" * 64,
                "source_end": {"sha256": "source"},
                "artifacts": {
                    "gpu_step_db_test": record(paths["worker"]),
                    "libmeep": record(paths["libmeep"]),
                    "fd_allocation_shim": record(paths["fd_allocation_shim"]),
                },
                "configuration_files": {
                    "openmpi_qualification_params": record(paths["mca"]),
                    "runtime_dependency_closure": record(paths["runtime_closure"]),
                },
                "toolchain": {
                    "mpiexec": record(paths["mpiexec"]),
                    "timeout": record(paths["timeout"]),
                },
            }
            with (
                mock.patch.object(ldos.PROVENANCE, "verify_build_receipt", return_value=receipt),
                mock.patch.object(ldos, "elf_interpreter", return_value=paths["timeout"]),
            ):
                checked = ldos.validate_receipt_inputs(
                    repo=repo,
                    receipt_path=receipt_path,
                    expected_receipt_id="a" * 64,
                    mpiexec_path=paths["mpiexec"],
                )
            self.assertEqual(checked["paths"]["gpu_step_db"], paths["worker"])
            self.assertEqual(
                checked["paths"]["fd_allocation_shim"],
                paths["fd_allocation_shim"],
            )

            wrong_kind = dict(receipt, build_kind="cuda-mpi-fp32")
            with mock.patch.object(
                ldos.PROVENANCE,
                "verify_build_receipt",
                return_value=wrong_kind,
            ):
                with self.assertRaisesRegex(ldos.EvidenceError, "CUDA.*Python FP32"):
                    ldos.validate_receipt_inputs(
                        repo=repo,
                        receipt_path=receipt_path,
                        expected_receipt_id="a" * 64,
                        mpiexec_path=paths["mpiexec"],
                    )

            copied_mpiexec = root / "copied-mpiexec"
            copied_mpiexec.write_bytes(paths["mpiexec"].read_bytes())
            with (
                mock.patch.object(ldos.PROVENANCE, "verify_build_receipt", return_value=receipt),
                mock.patch.object(ldos, "elf_interpreter", return_value=paths["timeout"]),
            ):
                with self.assertRaisesRegex(ldos.EvidenceError, "exact.*pathname"):
                    ldos.validate_receipt_inputs(
                        repo=repo,
                        receipt_path=receipt_path,
                        expected_receipt_id="a" * 64,
                        mpiexec_path=copied_mpiexec,
                    )

            wrong_id = dict(receipt, receipt_id="b" * 64)
            with mock.patch.object(
                ldos.PROVENANCE, "verify_build_receipt", return_value=wrong_id
            ):
                with self.assertRaisesRegex(ldos.EvidenceError, "differs"):
                    ldos.validate_receipt_inputs(
                        repo=repo,
                        receipt_path=receipt_path,
                        expected_receipt_id="a" * 64,
                        mpiexec_path=paths["mpiexec"],
                    )

    def test_source_mutation_is_rejected_after_run(self) -> None:
        expected_source = {"sha256": "source-a"}
        checked = {
            "receipt": {"receipt_id": "a" * 64, "source_end": {"sha256": "source-b"}},
            "records": {"gpu_step_db": {"sha256": "1" * 64}},
        }
        with mock.patch.object(ldos, "validate_receipt_inputs", return_value=checked):
            with self.assertRaisesRegex(ldos.EvidenceError, "source identity changed"):
                ldos.reverify_receipt_inputs(
                    repo=pathlib.Path("/repo"),
                    receipt_path=pathlib.Path("/receipt"),
                    expected_receipt_id="a" * 64,
                    mpiexec_path=pathlib.Path("/mpiexec"),
                    expected_source=expected_source,
                    expected_records=checked["records"],
                )

    def test_artifact_record_mutation_is_rejected_after_run(self) -> None:
        source = {"sha256": "source"}
        checked = {
            "receipt": {"receipt_id": "a" * 64, "source_end": source},
            "records": {"gpu_step_db": {"sha256": "2" * 64}},
        }
        with mock.patch.object(ldos, "validate_receipt_inputs", return_value=checked):
            with self.assertRaisesRegex(ldos.EvidenceError, "records changed"):
                ldos.reverify_receipt_inputs(
                    repo=pathlib.Path("/repo"),
                    receipt_path=pathlib.Path("/receipt"),
                    expected_receipt_id="a" * 64,
                    mpiexec_path=pathlib.Path("/mpiexec"),
                    expected_source=source,
                    expected_records={"gpu_step_db": {"sha256": "1" * 64}},
                )

    def test_expected_receipt_id_must_be_exact(self) -> None:
        with self.assertRaisesRegex(ldos.EvidenceError, "lowercase SHA-256"):
            ldos.validate_receipt_inputs(
                repo=pathlib.Path("/repo"),
                receipt_path=pathlib.Path("/receipt"),
                expected_receipt_id="A" * 64,
                mpiexec_path=pathlib.Path("/mpiexec"),
            )


class OutputPublicationTests(unittest.TestCase):
    @staticmethod
    def make_nvml_bound_publication(
        output: pathlib.Path,
    ) -> tuple[dict, dict, pathlib.Path]:
        journal_fixture = RawTelemetryJournalTests(
            methodName="test_independent_validator_recomputes_complete_journal"
        )
        journal_path, journal_header = journal_fixture.make_valid_journal(output)
        archive_path = (
            output / "archive" / "system" / "lib" / "libnvidia-ml.so.1"
        )
        archive = ldos.StableFile.open("archived trusted system NVML", archive_path)
        try:
            trusted_archive = archive.record()
        finally:
            archive.close()
        ldos.EVIDENCE.atomic_write_json(
            output / "archive" / "archive-manifest.json",
            {
                "schema_version": 1,
                "snapshots": {"nvml_library": trusted_archive},
            },
        )
        host_control = ldos.validate_lane_journal(
            journal_path, expected_header=journal_header
        )
        host_control["raw_journal"] = ldos._relative_file_record(
            journal_path, output
        )
        lanes = [
            ldos.parse_benchmark_output(
                valid_stdout(), "", returncode=0, expected_ranks=1
            ),
            ldos.parse_benchmark_output(
                valid_stdout(2), "", returncode=0, expected_ranks=2
            ),
        ]
        claim_scope = {
            "valid_for_end_to_end_fdtd_speedup": False,
            "valid_for_cpu_vs_gpu_speedup": False,
        }
        summary = {"lanes": lanes}
        report = {
            "build_receipt": {"receipt_id": "a" * 64},
            "claim_scope": claim_scope,
            "device_binding": {"trusted_nvml_archive": trusted_archive},
            "samples": [
                {"benchmark": {"host_runtime_control": host_control}}
            ],
        }
        return summary, report, archive_path

    def test_nofollow_directory_tree_fsyncs_exact_nested_targets_leaf_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "evidence"
            leaf = root / "archive" / "system" / "lib"
            leaf.mkdir(parents=True)
            (leaf / "partial").write_bytes(b"partial")
            outside = pathlib.Path(directory) / "outside"
            outside.mkdir()
            (root / "hostile-link").symlink_to(outside, target_is_directory=True)
            (root / "not-a-directory").write_bytes(b"regular")
            fsynced = ldos._fsync_directory_tree_nofollow(root)
            self.assertEqual(
                fsynced,
                [leaf, leaf.parent, leaf.parent.parent, root],
            )
            self.assertNotIn(outside, fsynced)

    def test_failure_fsyncs_partial_file_and_nested_dirs_before_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "evidence"
            ldos._claim_output(output, "run")
            leaf = output / "archive" / "system" / "lib"
            leaf.mkdir(parents=True)
            partial = leaf / "libnvidia-ml.so.1"
            partial.write_bytes(b"partial archive")
            events = []
            real_fsync = ldos.os.fsync
            real_atomic = ldos.EVIDENCE.atomic_write_json
            real_tree = ldos._fsync_directory_tree_nofollow

            def tracked_fsync(descriptor):
                try:
                    target = pathlib.Path(os.readlink(f"/proc/self/fd/{descriptor}"))
                except OSError:
                    target = pathlib.Path("unavailable")
                events.append(("fsync", target))
                return real_fsync(descriptor)

            def tracked_atomic(path, value):
                events.append(("atomic", pathlib.Path(path).name))
                return real_atomic(path, value)

            def tracked_tree(root):
                result = real_tree(root)
                events.append(("tree", tuple(result)))
                return result

            with (
                mock.patch.object(ldos.os, "fsync", side_effect=tracked_fsync),
                mock.patch.object(
                    ldos.EVIDENCE, "atomic_write_json", side_effect=tracked_atomic
                ),
                mock.patch.object(
                    ldos, "_fsync_directory_tree_nofollow", side_effect=tracked_tree
                ),
            ):
                ldos._publish_failure(output, "run", RuntimeError("injected"))
            tree_index = next(
                index for index, event in enumerate(events) if event[0] == "tree"
            )
            failed_index = events.index(("atomic", "FAILED"))
            terminal_index = events.index(("atomic", "TERMINAL"))
            partial_fsync_index = next(
                index for index, event in enumerate(events)
                if event == ("fsync", partial)
            )
            self.assertLess(partial_fsync_index, tree_index)
            self.assertLess(tree_index, failed_index)
            self.assertLess(failed_index, terminal_index)
            tree_targets = events[tree_index][1]
            self.assertTrue(
                all(target in tree_targets for target in (
                    leaf, leaf.parent, leaf.parent.parent, output,
                ))
            )

    def test_failure_durability_walk_skips_hostile_archive_directory_entry(self) -> None:
        for mutation in ("symlink", "regular"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                output = root / "evidence"
                ldos._claim_output(output, "run")
                archive = output / "archive"
                archive.mkdir()
                system = archive / "system"
                if mutation == "symlink":
                    outside = root / "outside"
                    outside.mkdir()
                    (outside / "must-not-be-traversed").write_bytes(b"outside")
                    system.symlink_to(outside, target_is_directory=True)
                else:
                    system.write_bytes(b"not a directory")
                ldos._publish_failure(
                    output, "run", RuntimeError("hostile archive directory")
                )
                failure = json.loads((output / "FAILED").read_text())
                records = {
                    record["path"]: record for record in failure["partial_artifacts"]
                }
                self.assertIn("archive/system", records)
                self.assertNotIn(
                    "archive/system/must-not-be-traversed", records
                )
                self.assertEqual(
                    json.loads((output / "TERMINAL").read_text())["state"],
                    "FAILED",
                )

    def test_output_claim_requires_absent_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "evidence"
            ldos._claim_output(output, "run-one")
            before = sorted(path.name for path in output.iterdir())
            with self.assertRaisesRegex(ldos.EvidenceError, "fresh, absent"):
                ldos._claim_output(output, "run-two")
            self.assertEqual(sorted(path.name for path in output.iterdir()), before)

    def test_success_publication_writes_terminal_last_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "evidence"
            ldos._claim_output(output, "run")
            summary = {
                "lanes": [
                    ldos.parse_benchmark_output(
                        valid_stdout(), "", returncode=0, expected_ranks=1
                    ),
                    ldos.parse_benchmark_output(
                        valid_stdout(2), "", returncode=0, expected_ranks=2
                    ),
                ]
            }
            report = {
                "build_receipt": {"receipt_id": "a" * 64},
                "claim_scope": {
                    "valid_for_end_to_end_fdtd_speedup": False,
                    "valid_for_cpu_vs_gpu_speedup": False,
                },
            }
            with mock.patch.object(ldos, "_verify_published_nvml_cross_binding"):
                ldos._publish_success(
                    output=output, run_id="run", summary=summary, report=report
                )
            terminal = json.loads((output / "TERMINAL").read_text())
            self.assertEqual(terminal["state"], "COMPLETE")
            self.assertTrue((output / "COMPLETE").is_file())
            self.assertFalse((output / "RUNNING.json").exists())
            self.assertEqual(
                terminal["authority"]["sha256"],
                ldos.EVIDENCE.sha256_file(output / "COMPLETE"),
            )

    def test_nvml_bound_success_reverifies_current_manifest_and_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "evidence"
            ldos._claim_output(output, "run")
            summary, report, _archive_path = self.make_nvml_bound_publication(output)
            ldos._publish_success(
                output=output, run_id="run", summary=summary, report=report
            )
            self.assertEqual(
                json.loads((output / "TERMINAL").read_text())["state"],
                "COMPLETE",
            )

    def test_nvml_swaps_at_publication_boundaries_end_in_failed_terminal(self) -> None:
        for swap_after in (
            "report.json",
            "artifacts.sha256.json",
            "COMPLETE",
            "TERMINAL",
        ):
            with self.subTest(swap_after=swap_after), tempfile.TemporaryDirectory() as directory:
                output = pathlib.Path(directory) / "evidence"
                ldos._claim_output(output, "run")
                self.assertEqual(
                    json.loads((output / "RUNNING.json").read_text())["state"],
                    "RUNNING",
                )
                summary, report, archive_path = self.make_nvml_bound_publication(output)
                original_archive = archive_path.read_bytes()
                real_atomic = ldos.EVIDENCE.atomic_write_json
                swapped = False

                def swap_after_atomic(path, value):
                    nonlocal swapped
                    result = real_atomic(path, value)
                    if pathlib.Path(path).name == swap_after and not swapped:
                        replacement = output / ".nvml-publication-swap"
                        replacement.write_bytes(original_archive + b"forged")
                        replacement.chmod(0o400)
                        os.replace(replacement, archive_path)
                        swapped = True
                    return result

                try:
                    with mock.patch.object(
                        ldos.EVIDENCE,
                        "atomic_write_json",
                        side_effect=swap_after_atomic,
                    ):
                        ldos._publish_success(
                            output=output,
                            run_id="run",
                            summary=summary,
                            report=report,
                        )
                except ldos.EvidenceError as error:
                    ldos._publish_failure(output, "run", error)
                else:
                    self.fail(f"NVML swap after {swap_after} was accepted")
                self.assertTrue(swapped)
                self.assertEqual(
                    json.loads((output / "TERMINAL").read_text())["state"],
                    "FAILED",
                )
                self.assertFalse((output / "COMPLETE").exists())
                self.assertFalse((output / "RUNNING.json").exists())

    def test_success_crash_boundaries_preserve_unambiguous_authority(self) -> None:
        class InjectedCrash(BaseException):
            pass

        for crash_stage in (
            "complete",
            "pre-terminal",
            "post-terminal",
            "pre-retirement",
            "retirement-fsync",
        ):
            with self.subTest(crash_stage=crash_stage), tempfile.TemporaryDirectory() as directory:
                output = pathlib.Path(directory) / "evidence"
                ldos._claim_output(output, "run")
                summary, report, _archive_path = self.make_nvml_bound_publication(output)
                real_atomic = ldos.EVIDENCE.atomic_write_json
                real_tree_sync = ldos._fsync_directory_tree
                real_directory_sync = ldos._fsync_directory
                injected = False

                def crash_at_atomic(path, value):
                    nonlocal injected
                    name = pathlib.Path(path).name
                    if crash_stage == "pre-terminal" and name == "TERMINAL" and not injected:
                        injected = True
                        raise InjectedCrash("pre-terminal")
                    result = real_atomic(path, value)
                    if crash_stage == "complete" and name == "COMPLETE" and not injected:
                        injected = True
                        raise InjectedCrash("complete")
                    return result

                def crash_after_terminal_sync(path):
                    nonlocal injected
                    result = real_tree_sync(path)
                    if (
                        crash_stage == "post-terminal"
                        and (output / "TERMINAL").exists()
                        and not injected
                    ):
                        injected = True
                        raise InjectedCrash("post-terminal")
                    return result

                def crash_before_retirement(*_args, **_kwargs):
                    nonlocal injected
                    injected = True
                    raise InjectedCrash("pre-retirement")

                def crash_during_retirement_sync(path):
                    nonlocal injected
                    result = real_directory_sync(path)
                    if (
                        crash_stage == "retirement-fsync"
                        and pathlib.Path(path) == output
                        and (output / "TERMINAL").exists()
                        and not (output / "RUNNING.json").exists()
                        and not injected
                    ):
                        injected = True
                        raise InjectedCrash("retirement-fsync")
                    return result

                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        ldos.EVIDENCE,
                        "atomic_write_json",
                        side_effect=crash_at_atomic,
                    ))
                    stack.enter_context(mock.patch.object(
                        ldos,
                        "_fsync_directory_tree",
                        side_effect=crash_after_terminal_sync,
                    ))
                    stack.enter_context(mock.patch.object(
                        ldos,
                        "_fsync_directory",
                        side_effect=crash_during_retirement_sync,
                    ))
                    if crash_stage == "pre-retirement":
                        stack.enter_context(mock.patch.object(
                            ldos,
                            "_retire_success_running_marker",
                            side_effect=crash_before_retirement,
                        ))
                    with self.assertRaises(InjectedCrash):
                        ldos._publish_success(
                            output=output,
                            run_id="run",
                            summary=summary,
                            report=report,
                        )
                self.assertTrue(injected)
                terminal_path = output / "TERMINAL"
                running_path = output / "RUNNING.json"
                if crash_stage in ("complete", "pre-terminal"):
                    self.assertFalse(terminal_path.exists())
                    self.assertEqual(
                        json.loads(running_path.read_text())["state"], "RUNNING"
                    )
                elif crash_stage in ("post-terminal", "pre-retirement"):
                    self.assertEqual(
                        json.loads(terminal_path.read_text())["state"], "COMPLETE"
                    )
                    self.assertEqual(
                        json.loads(running_path.read_text())["state"], "RUNNING"
                    )
                else:
                    self.assertEqual(
                        json.loads(terminal_path.read_text())["state"], "COMPLETE"
                    )
                    self.assertFalse(running_path.exists())

    def test_failure_publication_binds_partial_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "evidence"
            ldos._claim_output(output, "run")
            (output / "raw.log").write_text("partial\n", encoding="utf-8")
            ldos._publish_failure(output, "run", RuntimeError("boom"))
            terminal = json.loads((output / "TERMINAL").read_text())
            failure = json.loads((output / "FAILED").read_text())
            self.assertEqual(terminal["state"], "FAILED")
            self.assertEqual(failure["error"], "boom")
            self.assertEqual(failure["partial_artifacts"][0]["path"], "raw.log")
            self.assertFalse((output / "COMPLETE").exists())

    def test_failure_publication_survives_hostile_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "evidence"
            ldos._claim_output(output, "run")
            (output / "outside").write_text("outside", encoding="utf-8")
            (output / "hostile-link").symlink_to(output / "outside")
            ldos._publish_failure(output, "run", RuntimeError("bad entry"))
            failure = json.loads((output / "FAILED").read_text())
            kinds = {
                record["path"]: record["kind"]
                for record in failure["partial_artifacts"]
            }
            self.assertEqual(kinds["hostile-link"], "symlink")
            self.assertEqual(
                json.loads((output / "TERMINAL").read_text())["state"], "FAILED"
            )

    def test_failure_publication_quarantines_nonempty_terminal_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "evidence"
            ldos._claim_output(output, "run")
            outside = root / "outside"
            outside.write_text("untouched", encoding="utf-8")
            hostile = output / "TERMINAL"
            hostile.mkdir()
            (hostile / "nested").mkdir()
            (hostile / "nested" / "outside-link").symlink_to(outside)
            ldos._publish_failure(output, "run", RuntimeError("hostile terminal"))
            self.assertEqual(json.loads((output / "TERMINAL").read_text())["state"], "FAILED")
            quarantined = list(output.glob("superseded-TERMINAL-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertTrue(quarantined[0].is_dir())
            self.assertTrue((quarantined[0] / "nested" / "outside-link").is_symlink())
            self.assertEqual(outside.read_text(), "untouched")
            failure = json.loads((output / "FAILED").read_text())
            partial = {record["path"]: record for record in failure["partial_artifacts"]}
            nested = next(
                record for name, record in partial.items()
                if name.endswith("nested/outside-link")
            )
            self.assertEqual(nested["kind"], "symlink")

    def test_failure_terminal_same_json_symlink_swap_is_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "evidence"
            outside = root / "outside-terminal.json"
            ldos._claim_output(output, "run")
            real_atomic = ldos.EVIDENCE.atomic_write_json
            swapped = False

            def replace_terminal_with_same_json_symlink(path, value):
                nonlocal swapped
                result = real_atomic(path, value)
                if pathlib.Path(path).name == "TERMINAL" and not swapped:
                    real_atomic(outside, value)
                    pathlib.Path(path).unlink()
                    pathlib.Path(path).symlink_to(outside)
                    swapped = True
                return result

            with mock.patch.object(
                ldos.EVIDENCE,
                "atomic_write_json",
                side_effect=replace_terminal_with_same_json_symlink,
            ):
                with self.assertRaisesRegex(
                    ldos.EvidenceError, "single-link regular file"
                ):
                    ldos._publish_failure(output, "run", RuntimeError("boom"))
            outside_before = outside.read_bytes()
            self.assertTrue(swapped)
            self.assertEqual(
                json.loads((output / "RUNNING.json").read_text())["state"],
                "RUNNING",
            )
            self.assertFalse((output / "TERMINAL").exists())
            quarantined = list(output.glob("superseded-TERMINAL-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertTrue(quarantined[0].is_symlink())
            self.assertEqual(quarantined[0].resolve(strict=True), outside)
            self.assertEqual(outside.read_bytes(), outside_before)

            ldos._publish_failure(output, "run", RuntimeError("retry"))
            self.assertEqual(outside.read_bytes(), outside_before)
            self.assertEqual(
                json.loads((output / "TERMINAL").read_text())["state"],
                "FAILED",
            )
            self.assertFalse((output / "RUNNING.json").exists())

    def test_failure_publication_recovers_running_after_final_sync_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "evidence"
            ldos._claim_output(output, "run")
            real_sync = ldos._fsync_directory
            injected = False

            def fail_after_retirement(path):
                nonlocal injected
                if (
                    path == output
                    and not (output / "RUNNING.json").exists()
                    and (output / "TERMINAL").exists()
                    and not injected
                ):
                    injected = True
                    raise OSError("final directory sync")
                return real_sync(path)

            with mock.patch.object(ldos, "_fsync_directory", side_effect=fail_after_retirement):
                with self.assertRaisesRegex(OSError, "final directory sync"):
                    ldos._publish_failure(output, "run", RuntimeError("boom"))
            running = json.loads((output / "RUNNING.json").read_text())
            self.assertEqual(running["run_id"], "run")
            self.assertEqual(running["state"], "RUNNING")
            self.assertFalse((output / "TERMINAL").exists())
            self.assertTrue(list(output.glob("superseded-TERMINAL-*")))

    def test_failure_publication_replaces_hostile_running_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output = root / "evidence"
            ldos._claim_output(output, "run")
            outside = root / "outside"
            outside.write_text("outside", encoding="utf-8")
            (output / "RUNNING.json").unlink()
            (output / "RUNNING.json").symlink_to(outside)
            ldos._publish_failure(output, "run", RuntimeError("bad running"))
            self.assertEqual(outside.read_text(), "outside")
            self.assertEqual(json.loads((output / "TERMINAL").read_text())["state"], "FAILED")
            self.assertTrue(list(output.glob("superseded-RUNNING.json-*")))

    def test_partial_manifest_rejects_regular_to_symlink_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            target = output / "partial"
            outside = output / "outside"
            target.write_text("trusted", encoding="utf-8")
            outside.write_text("outside", encoding="utf-8")
            real_open = ldos.os.open
            swapped = False

            def swap_before_open(path, flags, *args):
                nonlocal swapped
                if pathlib.Path(path) == target and not swapped:
                    swapped = True
                    target.unlink()
                    target.symlink_to(outside)
                return real_open(path, flags, *args)

            with mock.patch.object(ldos.os, "open", side_effect=swap_before_open):
                with self.assertRaisesRegex(ldos.EvidenceError, "changed while opening"):
                    ldos._failure_artifact_records(output, set())

    def test_record_reverification_rejects_post_hash_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            path = output / "artifact"
            path.write_text("first", encoding="utf-8")
            record = ldos._relative_file_record(path, output)
            path.write_text("other", encoding="utf-8")
            with self.assertRaisesRegex(ldos.EvidenceError, "changed"):
                ldos._verify_relative_records(output, [record])

    def test_manifest_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            target = output / "target"
            target.write_text("x", encoding="utf-8")
            (output / "link").symlink_to(target)
            with self.assertRaisesRegex(ldos.EvidenceError, "Symbolic|symbolic"):
                ldos._artifact_records(output, set())


class ControllerWithoutGpuTests(unittest.TestCase):
    def setUp(self) -> None:
        # execute() owns and clears the production singleton. Keep this
        # synthetic controller class from leaking that lifecycle into later
        # tests in the same interpreter.
        ldos._FD_ALLOCATION_SHIM = PythonTestFdAllocationShim()

    def tearDown(self) -> None:
        ldos._FD_ALLOCATION_SHIM = PythonTestFdAllocationShim()

    def test_full_controller_with_synthetic_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            repo = root / "authoritative"
            repo.mkdir()
            (repo / ".git").mkdir()
            prefix = root / "prefix"
            (prefix / "bin").mkdir(parents=True)
            (prefix / "lib" / "openmpi").mkdir(parents=True)
            (prefix / "lib" / "pmix").mkdir(parents=True)
            files = {
                "receipt": root / "receipt.json",
                "gpu_step_db": root / "gpu-step-db",
                "libmeep": root / "libmeep.so.38.0.0",
                "fd_allocation_shim": root / "libgpmeep-fd-allocation-shim.so.1",
                "mpiexec": prefix / "bin" / "mpiexec",
                "timeout": root / "timeout",
                "mca": root / "mca.conf",
                "loader": root / "ld-linux.so",
                "runtime_closure": root / "runtime-closure.json",
            }
            for name, path in files.items():
                path.write_bytes((name + "\n").encode())
            nvidia_smi = root / "nvidia-smi"
            nvidia_smi.write_bytes(b"nvidia-smi\n")
            source = {"sha256": "source"}
            records = {
                name: {
                    "path": str(path),
                    "sha256": ldos.EVIDENCE.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for name, path in files.items()
            }
            validated = {
                "receipt": {
                    "receipt_id": "a" * 64,
                    "build_input_id": "b" * 64,
                    "artifact_set_id": "c" * 64,
                    "source_end": source,
                },
                "paths": files,
                "records": records,
            }

            telemetry_csv = (
                "0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, RTX A, 00000000:01:00.0, 8.6, 595.71.05, P8, 210, 405, 15.0, 350.0, 30, 0, 0, 24564\n"
                "1, GPU-11111111-2222-3333-4444-555555555555, RTX B, 00000000:02:00.0, 8.6, 595.71.05, P8, 210, 405, 15.0, 350.0, 30, 0, 0, 24564\n"
            )

            def fake_run(*, command, environment, cwd, timeout_seconds, **kwargs):
                if any(
                    argument.startswith("--query-gpu=") for argument in command
                ):
                    result = process_result(1)
                    result.update(
                        {
                            "command": command,
                            "environment": environment,
                            "cwd": str(cwd),
                            "stdout": telemetry_csv,
                            "containment_error": None,
                            "executable": kwargs.get("executable"),
                        }
                    )
                    return result
                if any(argument.startswith("--query-compute-apps=") for argument in command):
                    result = process_result(1)
                    result.update({"command": command, "environment": environment, "cwd": str(cwd), "stdout": "", "containment_error": None, "executable": kwargs.get("executable")})
                    return result
                if "--list" in command:
                    result = process_result(1)
                    result.update({"command": command, "environment": environment, "cwd": str(cwd), "stdout": next(argument for argument in command if "archive/runtime/lib" in argument), "containment_error": None, "executable": kwargs.get("executable")})
                    return result
                ranks = sum(
                    argument.startswith("MEEP_GPU_DEVICE=")
                    for argument in command
                )
                result = process_result(ranks)
                result.update(
                    {
                        "command": command,
                        "environment": environment,
                        "cwd": str(cwd),
                        "containment_error": None,
                        "executable": kwargs.get("executable"),
                    }
                )
                return result

            class FakeLaneMonitor:
                def __init__(self, *, directory, ranks, worker_cpu_masks, **_kwargs):
                    self.ranks = ranks
                    self.worker_cpu_masks = worker_cpu_masks
                    self.nonce = "0" * 64
                    self.go_monotonic_ns = 0
                    directory.mkdir()
                    journal_path = directory / "telemetry.raw.jsonl"
                    journal_path.write_text("synthetic\n", encoding="utf-8")
                    self.journal = types.SimpleNamespace(path=journal_path)

                def poll(self, _root_pid):
                    return None

                def drain(self, _root_pid, _finished_ns):
                    return None

                def finish(self):
                    return {
                        "schema_version": 1,
                        "worker_cpu_masks": [
                            sorted(mask) for mask in self.worker_cpu_masks
                        ],
                        "synthetic": True,
                    }

                def abort(self, _error):
                    return {"synthetic": True}

            class FakeNvmlSampler:
                def __init__(self, _library, uuids):
                    self.handles = {value: object() for value in uuids}
                    self.library_identity = {"synthetic": True}
                    self.baseline_process_utilization = {
                        value: [] for value in uuids
                    }

                def close(self):
                    return None

            output = root / "evidence"
            controller_handles = {
                name: ldos.StableFile.open(
                    f"synthetic controller source {name}", SCRIPTS / name
                )
                for name in ldos.SOURCE_HELPERS
            }
            self.addCleanup(
                lambda: [handle.close() for handle in controller_handles.values()]
            )
            arguments = [
                "--authoritative-repo",
                str(repo),
                "--build-receipt",
                str(files["receipt"]),
                "--expected-receipt-id",
                "a" * 64,
                "--mpiexec",
                str(files["mpiexec"]),
                "--device-uuids",
                (
                    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,"
                    "GPU-11111111-2222-3333-4444-555555555555"
                ),
                "--output",
                str(output),
            ]
            bootstrap_events: list[str] = []

            def fake_initialize_shim(artifact) -> None:
                self.assertEqual(bootstrap_events, [])
                shim = PythonTestFdAllocationShim()
                shim.artifact_identity = artifact.record()
                ldos._FD_ALLOCATION_SHIM = shim
                bootstrap_events.append("shim")

            def fake_resolve_nvidia_smi():
                self.assertEqual(bootstrap_events, ["shim"])
                bootstrap_events.append("nvidia-smi")
                return ldos.StableFile.open(
                    "trusted system nvidia-smi", nvidia_smi
                )

            def fake_resolve_nvml():
                self.assertEqual(bootstrap_events, ["shim", "nvidia-smi"])
                bootstrap_events.append("nvml")
                return ldos.StableFile.open("trusted system NVML", nvidia_smi)

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.multiple(
                        ldos,
                        validate_receipt_inputs=mock.Mock(return_value=validated),
                        reverify_receipt_inputs=mock.Mock(return_value=validated),
                        _run_process=mock.Mock(side_effect=fake_run),
                        _controller_bootstrap=mock.Mock(),
                        _initialize_verified_helpers=mock.Mock(),
                        _initialize_fd_allocation_shim=mock.Mock(
                            side_effect=fake_initialize_shim
                        ),
                        _verify_controller_identity=mock.Mock(
                            return_value={"git_commit": "d" * 40}
                        ),
                        _resolve_nvidia_smi=mock.Mock(
                            side_effect=fake_resolve_nvidia_smi
                        ),
                        _resolve_nvml=mock.Mock(side_effect=fake_resolve_nvml),
                        _system_runtime_tcb_handle=mock.Mock(
                            return_value={"synthetic": True}
                        ),
                        _probe_nvml_loadability=mock.Mock(),
                        _probe_nvidia_smi_startup=mock.Mock(),
                        NvmlSampler=FakeNvmlSampler,
                        open_runtime_closure_handles=mock.Mock(return_value={}),
                        verify_runtime_attestation=mock.Mock(return_value=[]),
                        verify_mpi_runtime_observations=mock.Mock(
                            return_value={
                                "actual_execution_proofs": {"synthetic": True}
                            }
                        ),
                        _validate_success_host_timeline=mock.Mock(return_value=[]),
                        LaneControlMonitor=FakeLaneMonitor,
                        _verify_published_nvml_cross_binding=mock.Mock(),
                        _revalidate_sample=mock.Mock(
                            side_effect=lambda _output, sample: {
                                "record_file": sample["record_file"],
                                "benchmark": sample["record"]["benchmark"],
                            }
                        ),
                        _physical_cpu_core_plan=mock.Mock(
                            return_value={
                                "allowed_cpus": [0, 1, 2],
                                "available_full_cores": [],
                                "controller_core": {"logical_cpus": [0]},
                                "worker_cores": [
                                    {"logical_cpus": [1]},
                                    {"logical_cpus": [2]},
                                ],
                            }
                        ),
                        _set_all_process_task_affinity=mock.Mock(
                            return_value={
                                "start_time_ticks": 1,
                                "ticks": 0,
                                "task_affinities": {str(os.getpid()): [0]},
                            }
                        ),
                        _restore_process_task_affinities=mock.Mock(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        ldos.os, "sched_getaffinity", return_value={0, 1, 2}
                    )
                )
                stack.enter_context(mock.patch.object(ldos.os, "sched_setaffinity"))
                stack.enter_context(
                    mock.patch.dict(
                        ldos.CONTROLLER_SOURCE_HANDLES,
                        controller_handles,
                        clear=True,
                    )
                )
                self.assertEqual(ldos.main(arguments), 0)
                self.assertEqual(
                    bootstrap_events, ["shim", "nvidia-smi", "nvml"]
                )
            terminal = json.loads((output / "TERMINAL").read_text())
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(terminal["state"], "COMPLETE")
            self.assertEqual(
                [lane["ranks"] for lane in report["summary"]["lanes"]], [1, 2]
            )
            self.assertFalse(
                report["claim_scope"]["valid_for_end_to_end_fdtd_speedup"]
            )
            self.assertFalse(report["claim_scope"]["valid_for_cpu_vs_gpu_speedup"])
            commands = [
                json.loads((output / f"ldos-rank-{ranks}.json").read_text())["command"]
                for ranks in (1, 2)
            ]
            self.assertTrue(
                all(command[0] == "timeout" for command in commands)
            )
            self.assertIn("MEEP_GPU_DEVICE=0", commands[1])
            self.assertIn("MEEP_GPU_DEVICE=1", commands[1])
            self.assertEqual(
                report["device_binding"]["selected"][0]["uuid"],
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            )
            self.assertEqual(
                report["device_binding"]["trusted_nvml_library"]["path"],
                str(nvidia_smi.resolve()),
            )
            self.assertEqual(
                report["device_binding"]["trusted_nvml_library"]["sha256"],
                report["device_binding"]["trusted_nvml_archive"]["sha256"],
            )
            archive_manifest = json.loads(
                (output / "archive" / "archive-manifest.json").read_text()
            )
            self.assertEqual(
                archive_manifest["originals"]["nvml_library"],
                report["device_binding"]["trusted_nvml_library"],
            )
            self.assertEqual(
                archive_manifest["snapshots"]["nvml_library"],
                report["device_binding"]["trusted_nvml_archive"],
            )
            artifact_paths = {
                record["path"] for record in json.loads(
                    (output / "artifacts.sha256.json").read_text()
                )["records"]
            }
            self.assertIn("archive/archive-manifest.json", artifact_paths)
            self.assertIn("report.json", artifact_paths)

    def test_failure_after_claim_publishes_failed_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            repo = root / "repo"
            repo.mkdir()
            (repo / ".git").mkdir()
            receipt = root / "receipt"
            receipt.write_text("{}", encoding="utf-8")
            mpiexec = root / "mpiexec"
            mpiexec.write_text("x", encoding="utf-8")
            output = root / "evidence"
            arguments = [
                "--authoritative-repo",
                str(repo),
                "--build-receipt",
                str(receipt),
                "--expected-receipt-id",
                "a" * 64,
                "--mpiexec",
                str(mpiexec),
                "--device-uuids",
                (
                    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,"
                    "GPU-11111111-2222-3333-4444-555555555555"
                ),
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(ldos, "_controller_bootstrap"),
                mock.patch.object(ldos, "_initialize_verified_helpers"),
                mock.patch.object(
                    ldos,
                    "validate_receipt_inputs",
                    side_effect=ldos.EvidenceError("mutated receipt"),
                ),
                mock.patch.object(ldos, "_verify_controller_identity", return_value={"git_commit": "d" * 40}),
            ):
                self.assertEqual(ldos.main(arguments), 1)
            self.assertEqual(
                json.loads((output / "TERMINAL").read_text())["state"], "FAILED"
            )
            self.assertIn("mutated receipt", (output / "FAILED").read_text())

    def test_existing_output_is_not_modified_by_failed_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            repo = root / "repo"
            repo.mkdir()
            (repo / ".git").mkdir()
            output = root / "existing"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("owned", encoding="utf-8")
            arguments = [
                "--authoritative-repo", str(repo),
                "--build-receipt", str(root / "missing"),
                "--expected-receipt-id", "a" * 64,
                "--mpiexec", str(root / "missing-mpi"),
                "--device-uuids",
                (
                    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,"
                    "GPU-11111111-2222-3333-4444-555555555555"
                ),
                "--output", str(output),
            ]
            with (
                mock.patch.object(ldos, "_controller_bootstrap"),
                mock.patch.object(ldos, "_initialize_verified_helpers"),
            ):
                self.assertEqual(ldos.main(arguments), 1)
            self.assertEqual(sentinel.read_text(), "owned")
            self.assertEqual([path.name for path in output.iterdir()], ["sentinel"])


if __name__ == "__main__":
    unittest.main()
