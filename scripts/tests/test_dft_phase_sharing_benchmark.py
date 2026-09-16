from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import pathlib
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_dft_phase_benchmark",
    SCRIPTS / "benchmark-dft-phase-sharing.py",
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "gpmeep_initialization_marker_validator_test",
    SCRIPTS / "validate-multi-gpu-initialization-markers.py",
)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)
MGB = VALIDATOR.MGB


def prefixed(prefix: str, value: dict) -> str:
    return prefix + json.dumps(value, separators=(",", ":")) + "\n"


def sample_output(*, workload: str, condition: str) -> str:
    pixels, warmup, steps = (
        (16, 50, 5000) if workload == "small" else (192, 12, 80)
    )
    benchmark = {
        "schema_version": 4,
        "mpi_ranks": 2,
        "pixels": pixels,
        "warmup_steps": warmup,
        "steps": steps,
        "initial_condition": "trigonometric-v1",
        "requested_transport": "cuda-aware",
        "selected_transport": "cuda-aware",
        "cpu_calls": 0,
        "cuda_calls": 100,
        "pinned_bytes": 0,
        "cuda_aware_bytes": 128,
    }
    if workload == "small" and condition == "on":
        preparations, reuses = 10, 90
    else:
        preparations, reuses = 100, 0
    dft = {
        "batch_calls": 5,
        "submitted_updates": 100,
        "phase_preparation_launches": preparations,
        "phase_reuses": reuses,
        "update_kernel_launches": 100,
        "maximum_batch_size": 20,
    }
    dft_multi = {
        "automatic_checks": 0,
        "automatic_selected": 0,
        "automatic_rejected": 0,
        "forced_batches": 0,
        "batched_updates": 0,
        "unbatched_updates": 100,
        "plan_uploads": 0,
        "plan_reuses": 0,
        "metadata_host_to_device_bytes": 0,
    }
    eager = {
        "receive_start_calls": 2,
        "send_start_calls": 2,
        "receive_requests": 2,
        "send_requests": 2,
    }
    pingpong = {
        "warmup_secondary_allocations": 2,
        "selections": 4,
        "secondary_selections": 2,
    }
    graph = {"creations": 0, "launches": 4}
    devices = [
        {"rank": rank, "ordinal": rank, "uuid": f"gpu-{rank}"}
        for rank in (0, 1)
    ]
    return "".join(
        [
            *(prefixed(BENCHMARK.DEVICE_PREFIX, value) for value in devices),
            prefixed(
                BENCHMARK.SOURCE_PROFILE_PREFIX,
                {"profile": "single-ez-v1"},
            ),
            prefixed(
                BENCHMARK.INITIAL_CONDITION_PREFIX,
                {"profile": "trigonometric-v1"},
            ),
            prefixed(
                BENCHMARK.INITIALIZATION_TIMING_PREFIX,
                {"applications_per_rank": 1, "max_seconds": 0.25},
            ),
            prefixed(BENCHMARK.GRAPH_PREFIX, graph),
            prefixed(BENCHMARK.EAGER_PREFIX, eager),
            prefixed(BENCHMARK.PINGPONG_PREFIX, pingpong),
            prefixed(BENCHMARK.DFT_PREFIX, dft),
            prefixed(BENCHMARK.DFT_MULTI_PREFIX, dft_multi),
            prefixed(BENCHMARK.BENCHMARK_PREFIX, benchmark),
        ]
    )


def lifecycle_record() -> dict:
    pixels = 16
    steps = 2
    seconds = 0.5
    probes = [[0.10 + 0.01 * index, 0.0] for index in range(MGB.PROBE_COUNT)]
    phase_calls = {
        phase: {
            "cpu_calls": 0,
            "cuda_calls": 4,
            "expectation": "cuda_required",
        }
        for phase in MGB.REQUIRED_CUDA_PHASES
    }
    phase_calls["polarization"] = {
        "cpu_calls": 0,
        "cuda_calls": 0,
        "expectation": "not_applicable",
    }
    cells = pixels**3
    mpi_scalars = 48
    return {
        "schema_version": MGB.SCHEMA_VERSION,
        "mpi_ranks": 2,
        "pixels": pixels,
        "cells": cells,
        "warmup_steps": 2,
        "steps": steps,
        "loop_tile_base_db": 128,
        "bfast": False,
        "source_enabled": True,
        "overlap_material": False,
        "seconds": seconds,
        "mcells_per_second": cells * steps / seconds / 1e6,
        "requested_transport": "cuda-aware",
        "selected_transport": "cuda-aware",
        "completion_policy": "waitsome",
        "initial_condition": "trigonometric-v1",
        "cut_probe_values": probes,
        "cut_probe_l2": math.sqrt(
            sum(real * real + imag * imag for real, imag in probes)
        ),
        "ez_l2": 1.2,
        "ez_weighted_checksum": 1.7,
        "energy": 2.1,
        "dft_norm": 0.35,
        "remote_cut_scalars": 24,
        "remote_cut_l2": 0.42,
        "cpu_calls": 0,
        "cuda_calls": sum(
            item["cuda_calls"] for item in phase_calls.values()
        ),
        "phase_calls": phase_calls,
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "mpi_messages": 4,
        "mpi_scalars": mpi_scalars,
        "cuda_aware_bytes": mpi_scalars * 4,
        "pinned_bytes": 0,
    }


def lifecycle_devices() -> list[dict]:
    return [
        {
            "rank": rank,
            "ordinal": rank,
            "uuid": f"{rank + 1:032x}",
            "name": f"GPU {rank}",
            "compute_major": 8,
            "compute_minor": 0,
            "compatible": True,
        }
        for rank in (0, 1)
    ]


def lifecycle_output(
    *,
    record: dict | None = None,
    devices: list[dict] | None = None,
    applications: int = 2,
) -> str:
    record = lifecycle_record() if record is None else record
    devices = lifecycle_devices() if devices is None else devices
    return "".join(
        [
            *(prefixed(MGB.DEVICE_RECORD_PREFIX, item) for item in devices),
            prefixed(
                MGB.SOURCE_PROFILE_PREFIX,
                {"profile": "single-ez-v1"},
            ),
            prefixed(
                MGB.INITIAL_CONDITION_PREFIX,
                {"profile": "trigonometric-v1"},
            ),
            prefixed(
                MGB.INITIALIZATION_TIMING_PREFIX,
                {
                    "applications_per_rank": applications,
                    "max_seconds": 0.5,
                },
            ),
            prefixed(MGB.RECORD_PREFIX, record),
        ]
    )


def lifecycle_validator_args(log: pathlib.Path) -> list[str]:
    return [
        "--log",
        str(log),
        "--expected-profile",
        "trigonometric-v1",
        "--expected-source-profile",
        "single-ez-v1",
        "--expected-applications-per-rank",
        "2",
        "--expected-ranks",
        "2",
        "--expected-pixels",
        "16",
        "--expected-warmup-steps",
        "2",
        "--expected-steps",
        "2",
        "--expected-transport",
        "cuda-aware",
    ]


class DftRecordTests(unittest.TestCase):
    def test_duplicate_record_key_is_rejected(self) -> None:
        output = BENCHMARK.BENCHMARK_PREFIX + '{"seconds":1,"seconds":2}\n'
        with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
            BENCHMARK.records(output, BENCHMARK.BENCHMARK_PREFIX)

    def test_nonfinite_record_number_is_rejected(self) -> None:
        output = BENCHMARK.BENCHMARK_PREFIX + '{"seconds":NaN}\n'
        with self.assertRaisesRegex(RuntimeError, "non-finite JSON constant"):
            BENCHMARK.records(output, BENCHMARK.BENCHMARK_PREFIX)

    def test_initialization_markers_fail_closed(self) -> None:
        output = sample_output(workload="small", condition="on")
        benchmark = BENCHMARK.exactly_one(
            output, BENCHMARK.BENCHMARK_PREFIX, "benchmark"
        )
        records = BENCHMARK.validate_initialization_markers(output, benchmark)
        self.assertEqual(
            records["initial_condition"], {"profile": "trigonometric-v1"}
        )
        self.assertEqual(
            records["source_profile"], {"profile": "single-ez-v1"}
        )

        source_line = prefixed(
            BENCHMARK.SOURCE_PROFILE_PREFIX,
            {"profile": "single-ez-v1"},
        )
        initial_line = prefixed(
            BENCHMARK.INITIAL_CONDITION_PREFIX,
            {"profile": "trigonometric-v1"},
        )
        timing_line = prefixed(
            BENCHMARK.INITIALIZATION_TIMING_PREFIX,
            {"applications_per_rank": 1, "max_seconds": 0.25},
        )
        invalid_outputs = (
            (output.replace(source_line, ""), "exactly one"),
            (output.replace(source_line, source_line * 2), "exactly one"),
            (
                output.replace(
                    source_line,
                    prefixed(
                        BENCHMARK.SOURCE_PROFILE_PREFIX,
                        {"profile": "dual-electric-v1"},
                    ),
                ),
                "fixed profile",
            ),
            (output.replace(initial_line, ""), "exactly one"),
            (output.replace(initial_line, initial_line * 2), "exactly one"),
            (
                output.replace(
                    initial_line,
                    BENCHMARK.INITIAL_CONDITION_PREFIX + '{"profile":}\n',
                ),
                "Expecting value",
            ),
            (
                output.replace(
                    initial_line,
                    prefixed(
                        BENCHMARK.INITIAL_CONDITION_PREFIX,
                        {"profile": "affine-v1"},
                    ),
                ),
                "fixed profile",
            ),
            (
                output.replace(
                    timing_line,
                    prefixed(
                        BENCHMARK.INITIALIZATION_TIMING_PREFIX,
                        {"applications_per_rank": 2, "max_seconds": 0.25},
                    ),
                ),
                "one application",
            ),
            (
                output.replace(
                    timing_line,
                    prefixed(
                        BENCHMARK.INITIALIZATION_TIMING_PREFIX,
                        {"applications_per_rank": 1, "max_seconds": 0.0},
                    ),
                ),
                "finite and positive",
            ),
            (
                output.replace(
                    timing_line,
                    prefixed(
                        BENCHMARK.INITIALIZATION_TIMING_PREFIX,
                        {
                            "applications_per_rank": 1,
                            "max_seconds": 0.25,
                            "extra": 1,
                        },
                    ),
                ),
                "exact schema",
            ),
        )
        for candidate, message in invalid_outputs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, message):
                    BENCHMARK.validate_initialization_markers(
                        candidate, benchmark
                    )

    def test_unknown_and_mixed_version_markers_are_rejected(self) -> None:
        output = sample_output(workload="small", condition="on")
        BENCHMARK.reject_unknown_gpmeep_markers(output)
        for marker in (
            'gpmeep-initial-condition-v2:{"profile":"trigonometric-v1"}\n',
            'gpmeep-multi-benchmark-v3:{"schema_version":3}\n',
        ):
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(RuntimeError, "mixed-version"):
                    BENCHMARK.reject_unknown_gpmeep_markers(output + marker)

    def test_lifecycle_profile_requires_two_initialization_applications(self) -> None:
        output = sample_output(workload="small", condition="on")
        one = prefixed(
            BENCHMARK.INITIALIZATION_TIMING_PREFIX,
            {"applications_per_rank": 1, "max_seconds": 0.25},
        )
        two = prefixed(
            BENCHMARK.INITIALIZATION_TIMING_PREFIX,
            {"applications_per_rank": 2, "max_seconds": 0.5},
        )
        output = output.replace(one, two)
        benchmark = BENCHMARK.exactly_one(
            output, BENCHMARK.BENCHMARK_PREFIX, "benchmark"
        )
        records = BENCHMARK.validate_initialization_markers(
            output,
            benchmark,
            expected_applications_per_rank=2,
        )
        self.assertEqual(
            records["initialization_timing"]["applications_per_rank"], 2
        )
        with self.assertRaisesRegex(RuntimeError, "one application"):
            BENCHMARK.validate_initialization_markers(output, benchmark)

    def test_standalone_lifecycle_log_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "lifecycle.log"
            log.write_text(lifecycle_output(), encoding="utf-8")
            self.assertEqual(
                VALIDATOR.main(lifecycle_validator_args(log)),
                0,
            )
            with self.assertRaisesRegex(RuntimeError, "application count"):
                wrong_applications = lifecycle_validator_args(log)
                wrong_applications[
                    wrong_applications.index(
                        "--expected-applications-per-rank"
                    )
                    + 1
                ] = "1"
                VALIDATOR.main(
                    wrong_applications
                )
            with self.assertRaisesRegex(RuntimeError, "initial_condition"):
                wrong_profile = lifecycle_validator_args(log)
                wrong_profile[
                    wrong_profile.index("--expected-profile") + 1
                ] = "affine-v1"
                VALIDATOR.main(wrong_profile)

    def test_standalone_lifecycle_validator_rejects_false_gpu_evidence(self) -> None:
        valid = lifecycle_record()
        cases: list[tuple[str, dict, str]] = []

        wrong_ranks = copy.deepcopy(valid)
        wrong_ranks["mpi_ranks"] = 1
        cases.append(("ranks", wrong_ranks, "expected 2"))

        wrong_pixels = copy.deepcopy(valid)
        wrong_pixels["pixels"] = 17
        wrong_pixels["cells"] = 17**3
        wrong_pixels["mcells_per_second"] = (
            wrong_pixels["cells"]
            * wrong_pixels["steps"]
            / wrong_pixels["seconds"]
            / 1e6
        )
        cases.append(("pixels", wrong_pixels, "expected 16"))

        wrong_steps = copy.deepcopy(valid)
        wrong_steps["steps"] = 3
        wrong_steps["mcells_per_second"] = (
            wrong_steps["cells"]
            * wrong_steps["steps"]
            / wrong_steps["seconds"]
            / 1e6
        )
        cases.append(("steps", wrong_steps, "expected 2"))

        wrong_transport = copy.deepcopy(valid)
        wrong_transport["requested_transport"] = "pinned"
        wrong_transport["selected_transport"] = "pinned"
        wrong_transport["cuda_aware_bytes"] = 0
        wrong_transport["pinned_bytes"] = wrong_transport["mpi_scalars"] * 4
        wrong_transport["h2d_bytes"] = 32
        wrong_transport["d2h_bytes"] = 32
        cases.append(("transport", wrong_transport, "expected 'cuda-aware'"))

        cpu_fallback = copy.deepcopy(valid)
        cpu_fallback["phase_calls"]["curl"]["cpu_calls"] = 1
        cpu_fallback["phase_calls"]["curl"]["cuda_calls"] = 0
        cpu_fallback["cpu_calls"] = 1
        cpu_fallback["cuda_calls"] -= 4
        cases.append(("cpu fallback", cpu_fallback, "exclusive CUDA"))

        no_remote_activity = copy.deepcopy(valid)
        no_remote_activity["mpi_messages"] = 0
        cases.append(("remote activity", no_remote_activity, "no MPI boundary"))

        extra_key = copy.deepcopy(valid)
        extra_key["unexpected"] = True
        cases.append(("extra key", extra_key, "key mismatch"))

        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "lifecycle.log"
            for name, record, message in cases:
                with self.subTest(name=name):
                    log.write_text(
                        lifecycle_output(record=record), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(RuntimeError, message):
                        VALIDATOR.main(lifecycle_validator_args(log))

    def test_standalone_lifecycle_validator_requires_distinct_gpus(self) -> None:
        devices = lifecycle_devices()
        devices[1]["uuid"] = devices[0]["uuid"]
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "lifecycle.log"
            log.write_text(
                lifecycle_output(devices=devices), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "distinct physical GPU"):
                VALIDATOR.main(lifecycle_validator_args(log))

    def test_small_enabled_requires_and_accepts_reuse(self) -> None:
        output = sample_output(workload="small", condition="on")
        _, dft, devices = BENCHMARK.validate_sample(
            "on",
            {"name": "small", "pixels": 16, "warmup_steps": 50, "steps": 5000},
            output,
        )
        self.assertEqual(dft["phase_reuses"], 90)
        self.assertEqual(len(devices), 2)

    def test_phase_sharing_sample_rejects_every_multi_monitor_policy_path(self) -> None:
        baseline = sample_output(workload="small", condition="on")
        for field in (
            "automatic_checks",
            "automatic_selected",
            "automatic_rejected",
            "forced_batches",
            "batched_updates",
            "plan_uploads",
            "plan_reuses",
            "metadata_host_to_device_bytes",
        ):
            lines = baseline.splitlines()
            for index, line in enumerate(lines):
                if line.startswith(BENCHMARK.DFT_MULTI_PREFIX):
                    record = json.loads(line[len(BENCHMARK.DFT_MULTI_PREFIX) :])
                    record[field] = 1
                    lines[index] = prefixed(BENCHMARK.DFT_MULTI_PREFIX, record).rstrip("\n")
                    break
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "used multi-monitor"):
                    BENCHMARK.validate_sample(
                        "on",
                        {
                            "name": "small",
                            "pixels": 16,
                            "warmup_steps": 50,
                            "steps": 5000,
                        },
                        "\n".join(lines) + "\n",
                    )

    def test_phase_sharing_sample_rejects_extra_multi_monitor_field(self) -> None:
        lines = sample_output(workload="small", condition="on").splitlines()
        for index, line in enumerate(lines):
            if line.startswith(BENCHMARK.DFT_MULTI_PREFIX):
                record = json.loads(line[len(BENCHMARK.DFT_MULTI_PREFIX) :])
                record["mixed_version_field"] = 0
                lines[index] = prefixed(
                    BENCHMARK.DFT_MULTI_PREFIX, record
                ).rstrip("\n")
                break
        with self.assertRaisesRegex(RuntimeError, "invalid exact schema"):
            BENCHMARK.validate_sample(
                "on",
                {
                    "name": "small",
                    "pixels": 16,
                    "warmup_steps": 50,
                    "steps": 5000,
                },
                "\n".join(lines) + "\n",
            )

    def test_phase_sharing_sample_rejects_negative_multi_monitor_counter(self) -> None:
        lines = sample_output(workload="small", condition="on").splitlines()
        for index, line in enumerate(lines):
            if line.startswith(BENCHMARK.DFT_MULTI_PREFIX):
                record = json.loads(line[len(BENCHMARK.DFT_MULTI_PREFIX) :])
                record["automatic_checks"] = -1
                lines[index] = prefixed(
                    BENCHMARK.DFT_MULTI_PREFIX, record
                ).rstrip("\n")
                break
        with self.assertRaisesRegex(RuntimeError, "invalid exact schema"):
            BENCHMARK.validate_sample(
                "on",
                {
                    "name": "small",
                    "pixels": 16,
                    "warmup_steps": 50,
                    "steps": 5000,
                },
                "\n".join(lines) + "\n",
            )

    def test_small_disabled_requires_no_reuse(self) -> None:
        output = sample_output(workload="small", condition="off")
        BENCHMARK.validate_sample(
            "off",
            {"name": "small", "pixels": 16, "warmup_steps": 50, "steps": 5000},
            output,
        )
        with self.assertRaisesRegex(RuntimeError, "unexpectedly reused"):
            BENCHMARK.validate_sample(
                "off",
                {"name": "small", "pixels": 16, "warmup_steps": 50, "steps": 5000},
                sample_output(workload="small", condition="on"),
            )

    def test_large_is_fixed_as_no_reuse_opportunity(self) -> None:
        output = sample_output(workload="large", condition="on")
        BENCHMARK.validate_sample(
            "on",
            {"name": "large", "pixels": 192, "warmup_steps": 12, "steps": 80},
            output,
        )


class EvidenceHelperTests(unittest.TestCase):
    def test_runner_has_no_duplicate_literal_dictionary_keys(self) -> None:
        source = SCRIPTS / "benchmark-dft-phase-sharing.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        duplicates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, (str, int, float, bool, type(None)))
            ]
            repeated = sorted({key for key in keys if keys.count(key) > 1})
            if repeated:
                duplicates.append((node.lineno, repeated))
        self.assertEqual(duplicates, [])

    def test_ldd_comparison_ignores_only_aslr_addresses(self) -> None:
        first = "\tlibmeep.so.38 => /build/libmeep.so.38 (0x00001000)\n"
        second = "\tlibmeep.so.38 => /build/libmeep.so.38 (0x00002000)\n"
        wrong = "\tlibmeep.so.38 => /installed/libmeep.so.38 (0x00002000)\n"
        self.assertEqual(
            BENCHMARK.normalized_ldd_dependencies(first),
            BENCHMARK.normalized_ldd_dependencies(second),
        )
        self.assertNotEqual(
            BENCHMARK.normalized_ldd_dependencies(first),
            BENCHMARK.normalized_ldd_dependencies(wrong),
        )

    def test_process_monitor_requires_every_expected_rank_pid(self) -> None:
        class FakeProcess:
            def __init__(self, terminal_returncode=-15):
                self.returncode = None
                self.terminal_returncode = terminal_returncode

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = self.terminal_returncode

            def wait(self, timeout):
                return self.returncode

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            complete = root / "complete.log"
            complete_stream = complete.open("wb")
            complete_stream.write(
                b"0 100 C 1 1 gpu-mpi-per\n"
                b"1 101 C 1 1 gpu-mpi-per\n"
                b"0 102 C 1 1 gpu-mpi-per\n"
                b"1 103 C 1 1 gpu-mpi-per\n"
            )
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(), complete_stream, complete, 4
            )
            self.assertTrue(record["pass"])

            handled = root / "handled-sigterm.log"
            handled_stream = handled.open("wb")
            handled_stream.write(complete.read_bytes())
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(0), handled_stream, handled, 4
            )
            self.assertTrue(record["pass"])

            incomplete = root / "incomplete.log"
            incomplete_stream = incomplete.open("wb")
            incomplete_stream.write(
                b"0 200 C 1 1 gpu-mpi-per\n"
                b"1 201 C 1 1 gpu-mpi-per\n"
            )
            record = BENCHMARK.stop_process_monitor(
                FakeProcess(), incomplete_stream, incomplete, 4
            )
            self.assertFalse(record["pass"])

    def test_t90_interval_contains_geometric_mean(self) -> None:
        interval = BENCHMARK.paired_log_interval(
            [0.94, 0.95, 0.96, 0.94, 0.95, 0.96]
        )
        self.assertLess(
            interval["t90_lower_time_ratio"],
            interval["geometric_time_ratio_on_over_off"],
        )
        self.assertGreater(
            interval["t90_upper_time_ratio"],
            interval["geometric_time_ratio_on_over_off"],
        )

    def test_recursive_manifest_seals_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "archive").mkdir()
            (root / "archive" / "artifact").write_bytes(b"artifact")
            (root / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            path = BENCHMARK.write_manifest(root)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["path"] for item in manifest["files"]],
                ["COMPLETE.json", "archive/artifact", "summary.json"],
            )


if __name__ == "__main__":
    unittest.main()
