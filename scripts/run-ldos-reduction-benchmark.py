#!/usr/bin/env python3
"""Publish receipt-bound resident-LDOS reduction/transfer evidence.

This controller deliberately does *not* measure end-to-end FDTD or a CPU/GPU
speedup.  The bound ``gpu-step-db`` mode compares the LDOS update portion of
the same CUDA FDTD workload with resident CUDA reduction enabled and with the
legacy full-field host fallback forced.  Both one-rank and two-rank lanes must
pass; the latter uses two explicitly mapped, distinct physical GPU UUIDs.

``TERMINAL`` is the sole final authority marker.  It is written last and binds
either ``COMPLETE`` or ``FAILED``.  A COMPLETE without a matching TERMINAL is
an interrupted, non-authoritative publication.
"""

from __future__ import annotations

import os
import sys

# This file is intentionally importable by the unit suite, but an authoritative
# controller process must be isolated before *any* further Python imports can
# consult ambient module paths or sitecustomize.  The supported entrypoint is
# ``<system-python> -I -S scripts/run-ldos-reduction-benchmark.py ...``.
if __name__ == "__main__":
    flags = sys.flags
    if not (flags.isolated and flags.no_site and flags.ignore_environment):
        sys.stderr.write(
            "fatal: authoritative LDOS evidence requires Python -I -S\n"
        )
        raise SystemExit(2)
    hostile_environment = sorted(
        key
        for key in os.environ
        if key.upper() in {"PYTHONHOME", "PYTHONPATH"}
        or key.upper().startswith(("LD_", "DYLD_"))
        or "PRELOAD" in key.upper()
        or key.upper() in {"GCONV_PATH", "LOCPATH"}
    )
    if hostile_environment:
        sys.stderr.write(
            "fatal: authoritative LDOS evidence rejects ambient loader/Python "
            "environment: " + ",".join(hostile_environment) + "\n"
        )
        raise SystemExit(2)

import argparse
import copy
import csv
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import math
import pathlib
import platform
import re
import select
import secrets
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import tempfile
import threading
import time
import traceback
import types
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable


def _load_source_module(name: str, path: pathlib.Path) -> types.ModuleType:
    resolved = path.resolve()
    module = types.ModuleType(name)
    module.__file__ = str(resolved)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(resolved.read_bytes(), str(resolved), "exec"), module.__dict__)
    return module


SCRIPT_DIRECTORY = pathlib.Path(
    os.environ.get("GPMEEP_CONTROLLER_SOURCE_DIRECTORY", pathlib.Path(__file__).resolve().parent)
).resolve(strict=True)


def _bootstrapped_source(name: str) -> pathlib.Path:
    descriptor = os.environ.get(
        "GPMEEP_CONTROLLER_SOURCE_FD_" + name.upper().replace("-", "_").replace(".", "_")
    )
    return pathlib.Path(f"/proc/{os.getpid()}/fd/{descriptor}") if descriptor else SCRIPT_DIRECTORY / name


EVIDENCE: types.ModuleType | None = None
PROVENANCE: types.ModuleType | None = None

PROFILE_ID = "m19-ldos-reduction-transfer-v1"
PROFILE = types.MappingProxyType(
    {
        "profile_id": PROFILE_ID,
        "mpi_ranks": (1, 2),
        "pixels": 64,
        "updates": 16,
        "paired_repetitions": 5,
        "minimum_paired_speedup_exclusive": 1.20,
        "runtime_observer_max_gap_seconds": 0.25,
        "runtime_observer_max_duration_seconds": 0.20,
        "maximum_involuntary_context_switches_per_segment": 0,
        "resident_result_d2h_bytes_per_rank": 512,
        "mpi_transport": "pinned",
        "cuda_backend": "strict",
    }
)
PROFILE_SHA256: str | None = None
BENCHMARK_PREFIX = "ldos-transfer-benchmark:"
PAIR_PREFIX = "gpmeep-ldos-pair-v1:"
RANK_PREFIX = "gpmeep-ldos-rank-v1:"
RUNTIME_PREFIX = "gpmeep-ldos-runtime-v1:"
REJECTED_PREFIX = "gpmeep-ldos-rejected-v1:"
PASS_LINE = "PASS: repeated deterministic resident LDOS transfer speedup gate"
UUID_PATTERN = re.compile(
    r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
CUDA_UUID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{16}$")
CANONICAL_UNSIGNED_DECIMAL = r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e-?(?:0|[1-9][0-9]*))?"
CANONICAL_SIGNED_DECIMAL = rf"-?{CANONICAL_UNSIGNED_DECIMAL}"
BENCHMARK_PATTERN = re.compile(
    r"^ldos-transfer-benchmark: "
    r"ranks=(?P<ranks>[0-9]+) "
    r"pixels=(?P<pixels>[0-9]+) "
    r"updates=(?P<updates>[0-9]+) "
    r"repetitions=(?P<repetitions>[0-9]+) "
    rf"minimum-speedup=(?P<minimum>{CANONICAL_UNSIGNED_DECIMAL}) "
    rf"median-speedup=(?P<median>{CANONICAL_UNSIGNED_DECIMAL}) "
    rf"paired-speedups=(?P<speedups>{CANONICAL_UNSIGNED_DECIMAL}(?:,{CANONICAL_UNSIGNED_DECIMAL}){{4}})$"
)
FAIL_SKIP_PATTERN = re.compile(
    r"\b(?:FAIL|FAILED|FAILURE|SKIP|SKIPPED)\b", re.IGNORECASE
)
COMPLETE_MARKER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])COMPLETE(?![A-Za-z0-9_])")
GPMEEP_MARKER_PATTERN = re.compile(r"(gpmeep-[A-Za-z0-9_.-]+:)")
SOURCE_HELPERS = (
    "run-ldos-reduction-benchmark.py",
    "gpmeep_benchmark_evidence.py",
    "gpmeep_provenance.py",
    "gpmeep_qualification_contract.py",
)
CONTROLLER_IDENTITY: dict[str, Any] | None = None
CONTROLLER_SOURCE_HANDLES: dict[str, "StableFile"] = {}
PYTHON_INTERPRETER_HANDLE: "StableFile | None" = None
PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37
PROCESS_GRACE_SECONDS = 5.0
NVML_DEPARTED_CONTEXT_LAG_NS = 100_000_000
NVML_ABSENCE_OBSERVATION_TO_PIDFD_OBSERVATION_NS = 100_000_000
LDOS_READY_MAX_BYTES = 1 << 20
LDOS_HANDSHAKE_MAX_BYTES = 4096
NVML_REQUIRED_SYMBOLS = (
    "nvmlInit_v2",
    "nvmlShutdown",
    "nvmlDeviceGetHandleByUUID",
    "nvmlDeviceGetComputeRunningProcesses_v3",
    "nvmlDeviceGetGraphicsRunningProcesses_v3",
    "nvmlDeviceGetMPSComputeRunningProcesses_v3",
    "nvmlDeviceGetProcessUtilization",
    "nvmlDeviceGetUtilizationRates",
    "nvmlDeviceGetClockInfo",
    "nvmlDeviceGetPowerUsage",
    "nvmlDeviceGetTemperature",
    "nvmlDeviceGetPerformanceState",
    "nvmlDeviceGetMemoryInfo",
)
NVML_ELF_MAX_SIZE_BYTES = 256 * 1024 * 1024
NVML_ELF_MAX_SECTION_COUNT = 1024
NVML_ELF_MAX_PROGRAM_HEADER_COUNT = 1024
NVML_ELF_MAX_PARSE_WORK_BYTES = 64 * 1024 * 1024
NVML_ELF_MAX_C_STRING_BYTES = 4096
SYSTEM_TOOL_ELF_MAX_SIZE_BYTES = 256 * 1024 * 1024
SYSTEM_TOOL_ELF_MAX_PROGRAM_HEADERS = 1024
SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS = 5
NVML_LIBRARY_PLATFORM_CANDIDATES = {
    "x86_64": (
        pathlib.Path("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"),
        pathlib.Path("/lib/x86_64-linux-gnu/libnvidia-ml.so.1"),
    ),
    "aarch64": (
        pathlib.Path("/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1"),
        pathlib.Path("/lib/aarch64-linux-gnu/libnvidia-ml.so.1"),
    ),
}
NVML_LIBRARY_GENERIC_CANDIDATES = (
    pathlib.Path("/usr/lib64/libnvidia-ml.so.1"),
    pathlib.Path("/lib64/libnvidia-ml.so.1"),
    pathlib.Path("/usr/lib/libnvidia-ml.so.1"),
    pathlib.Path("/lib/libnvidia-ml.so.1"),
)
NVML_LIBRARY_WSL_CANDIDATES = (
    pathlib.Path("/usr/lib/wsl/lib/libnvidia-ml.so.1"),
    pathlib.Path("/usr/lib/wsl/drivers/libnvidia-ml.so.1"),
)
NVIDIA_SMI_CANDIDATES = (
    pathlib.Path("/usr/bin/nvidia-smi"),
    pathlib.Path("/usr/local/bin/nvidia-smi"),
    pathlib.Path("/usr/lib/wsl/lib/nvidia-smi"),
)
SYSTEM_RUNTIME_TCB_ROOTS = (
    pathlib.Path("/usr/lib"), pathlib.Path("/lib"),
    pathlib.Path("/usr/lib64"), pathlib.Path("/lib64"),
)
SYS_PIDFD_SEND_SIGNAL = 424
SYS_PIDFD_OPEN = 434
SYS_KCMP_BY_MACHINE = {
    "amd64": 312,
    "x86_64": 312,
    "aarch64": 272,
    "arm64": 272,
    "riscv64": 272,
    "ppc64": 312,
    "ppc64le": 312,
    "s390x": 343,
}
KCMP_FILE = 0


class EvidenceError(RuntimeError):
    """Raised when a benchmark artifact cannot support the narrow claim."""


class ProcessExecutionError(EvidenceError):
    """Post-spawn failure carrying all recoverable process output/evidence."""

    def __init__(
        self, message: str, result: dict[str, Any], original: BaseException
    ) -> None:
        super().__init__(message)
        self.result = result
        self.original = original


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_float(text: str, label: str) -> float:
    if not re.fullmatch(CANONICAL_SIGNED_DECIMAL, text):
        raise EvidenceError(f"{label} is not a canonical decimal")
    try:
        value = float(text)
    except ValueError as exc:
        raise EvidenceError(f"{label} is not a floating-point number") from exc
    if not math.isfinite(value):
        raise EvidenceError(f"{label} is NaN or infinity")
    if value <= 0.0:
        raise EvidenceError(f"{label} is not positive")
    return value


def _canonical_json_object(line: str, prefix: str) -> dict[str, Any]:
    if not line.startswith(prefix):
        raise EvidenceError(f"record does not start with {prefix}")
    payload = line[len(prefix) :]
    if not payload or any(character.isspace() for character in _outside_strings(payload)):
        raise EvidenceError(f"{prefix} payload is not compact canonical JSON")
    _validate_canonical_number_tokens(payload)
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise EvidenceError(f"invalid {prefix} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{prefix} payload is not an object")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _outside_strings(payload: str) -> str:
    result: list[str] = []
    quoted = False
    escaped = False
    for character in payload:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        else:
            result.append(character)
    if quoted or escaped:
        raise EvidenceError("unterminated JSON string")
    return "".join(result)


def _validate_canonical_number_tokens(payload: str) -> None:
    outside = _outside_strings(payload)
    for match in re.finditer(r"(?<![A-Za-z_])[-+0-9][^,\]}:]*", outside):
        token = match.group(0)
        if not re.fullmatch(CANONICAL_SIGNED_DECIMAL, token):
            raise EvidenceError(f"noncanonical JSON number {token!r}")
        if token.startswith("-0") and float(token) == 0.0:
            raise EvidenceError("negative zero is not a canonical JSON number")


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EvidenceError(f"{label} fields differ from the exact schema")


def _require_nonnegative_float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} is not numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0 or (positive and converted == 0.0):
        raise EvidenceError(f"{label} is not a finite {'positive' if positive else 'nonnegative'} value")
    return converted


def _sample_vector(value: Any, label: str) -> list[complex]:
    if not isinstance(value, list) or len(value) != 9:
        raise EvidenceError(f"{label} must contain exactly nine complex samples")
    samples: list[complex] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise EvidenceError(f"{label} sample {index} is not [real,imag]")
        real = _require_nonnegative_float(abs(pair[0]), f"{label} real {index}") if isinstance(pair[0], (int, float)) and not isinstance(pair[0], bool) else None
        imag = _require_nonnegative_float(abs(pair[1]), f"{label} imag {index}") if isinstance(pair[1], (int, float)) and not isinstance(pair[1], bool) else None
        if real is None or imag is None:
            raise EvidenceError(f"{label} sample {index} is not numeric")
        samples.append(complex(float(pair[0]), float(pair[1])))
    return samples


def _sample_digest(samples: list[complex]) -> str:
    digest = 14695981039346656037
    for sample in samples:
        for value in (sample.real, sample.imag):
            for byte in struct.pack("<d", value):
                digest ^= byte
                digest = (digest * 1099511628211) & ((1 << 64) - 1)
    return f"{digest:016x}"


def _sample_errors(reference: list[complex], candidate: list[complex]) -> tuple[float, float]:
    absolute_errors = [abs(left - right) for left, right in zip(reference, candidate)]
    maximum_absolute = max(absolute_errors, default=0.0)
    maximum_relative = max(
        (
            error / max(abs(left), abs(right), sys.float_info.min)
            for left, right, error in zip(reference, candidate, absolute_errors)
        ),
        default=0.0,
    )
    return maximum_absolute, maximum_relative


def _parse_exact_int(text: str, label: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", text):
        raise EvidenceError(f"{label} is not a canonical nonnegative integer")
    return int(text)


def _focused_marker_lines(stdout: str, stderr: str) -> list[tuple[str, str]]:
    """Return isolated focused records and reject embedded/duplicated markers."""

    focused = (
        RANK_PREFIX, PAIR_PREFIX, RUNTIME_PREFIX, REJECTED_PREFIX,
        BENCHMARK_PREFIX, PASS_LINE,
    )
    records: list[tuple[str, str]] = []
    for stream_name, text in (("stdout", stdout), ("stderr", stderr)):
        for line in text.splitlines():
            hits = [marker for marker in focused if marker in line]
            if not hits:
                continue
            if stream_name != "stdout":
                if PASS_LINE in hits:
                    raise EvidenceError(
                        "expected exactly one focused LDOS PASS on stdout and none on stderr"
                    )
                if BENCHMARK_PREFIX in hits:
                    raise EvidenceError(
                        "expected exactly one LDOS benchmark record on stdout and none on stderr"
                    )
                raise EvidenceError("LDOS focused structured evidence is stdout-only")
            if (
                len(hits) != 1
                or line.count(hits[0]) != 1
                or not line.startswith(hits[0])
                or hits[0] == PASS_LINE and line != PASS_LINE
            ):
                raise EvidenceError(
                    "LDOS focused marker is embedded, duplicated, or noncanonical"
                )
            records.append((hits[0], line))
    return records


def _unstructured_failure_token(
    stdout: str, stderr: str,
) -> re.Match[str] | None:
    """Find a failure token outside syntactically canonical JSON records."""

    structured_prefixes = (
        PAIR_PREFIX, RANK_PREFIX, RUNTIME_PREFIX, REJECTED_PREFIX,
    )
    for line in (*stdout.splitlines(), *stderr.splitlines()):
        canonical_structured = False
        for prefix in structured_prefixes:
            if not line.startswith(prefix):
                continue
            try:
                _canonical_json_object(line, prefix)
            except EvidenceError:
                canonical_structured = False
            else:
                canonical_structured = True
            break
        if canonical_structured:
            continue
        token = FAIL_SKIP_PATTERN.search(line)
        if token is not None:
            return token
    return None


def _parse_rejected_repetition_evidence(
    stdout: str, stderr: str, *, expected_ranks: int,
    expected_device_uuids: list[str] | None = None,
    expected_worker_cpu_masks: list[set[int]] | None = None,
) -> dict[str, Any] | None:
    """Validate the complete diagnostic matrix for one rejected fixed attempt."""

    focused_records = _focused_marker_lines(stdout, stderr)
    stdout_lines = stdout.splitlines()
    rejected_lines = [
        line for line in stdout_lines if line.startswith(REJECTED_PREFIX)
    ]
    if any(line.startswith(REJECTED_PREFIX) for line in stderr.splitlines()):
        raise EvidenceError("LDOS rejected evidence markers are stdout-only")
    if not rejected_lines:
        return None
    structured_prefixes = (RANK_PREFIX, PAIR_PREFIX, REJECTED_PREFIX)
    forbidden_success_prefixes = (RUNTIME_PREFIX, BENCHMARK_PREFIX)
    stderr_lines = stderr.splitlines()
    if any(
        any(prefix in line for prefix in structured_prefixes)
        for line in stderr_lines
    ):
        raise EvidenceError("LDOS rejected structured evidence is stdout-only")
    allowed_gpmeep_markers = set(structured_prefixes)
    unknown_markers = [
        marker
        for marker in GPMEEP_MARKER_PATTERN.findall(stdout + "\n" + stderr)
        if marker not in allowed_gpmeep_markers
    ]
    if unknown_markers:
        raise EvidenceError(
            "LDOS rejected stream contains unknown gpmeep marker(s): "
            + ",".join(sorted(set(unknown_markers)))
        )
    for line in (*stdout_lines, *stderr_lines):
        if (
            PASS_LINE in line
            or COMPLETE_MARKER_PATTERN.search(line)
            or any(prefix in line for prefix in forbidden_success_prefixes)
        ):
            raise EvidenceError("LDOS rejected stream contains success evidence")
    if len(rejected_lines) != expected_ranks:
        raise EvidenceError("LDOS rejected repetition rank matrix is incomplete")
    if (
        expected_device_uuids is not None
        and len(expected_device_uuids) < expected_ranks
    ) or (
        expected_worker_cpu_masks is not None
        and len(expected_worker_cpu_masks) < expected_ranks
    ):
        raise EvidenceError("LDOS rejected expected lane binding is incomplete")
    records = [
        _canonical_json_object(line, REJECTED_PREFIX) for line in rejected_lines
    ]
    keys = {
        "schema_version", "rank", "repetition", "order",
        "global_gate_accepted", "device_ordinal", "device_uuid",
        "host_seconds", "resident_seconds", "host_d2h", "resident_d2h",
        "host_cpu_reduction_calls", "host_cuda_reduction_calls",
        "resident_cpu_reduction_calls", "resident_cuda_reduction_calls",
        "resident_cuda_kernel_launches", "resident_result_d2h",
        "max_absolute_error", "max_relative_error", "host_digest",
        "resident_digest", "host_samples", "resident_samples", "cpu_affinity",
        "host_voluntary_context_switches", "host_involuntary_context_switches",
        "resident_voluntary_context_switches",
        "resident_involuntary_context_switches", "host_major_faults",
        "resident_major_faults", "host_minor_faults", "resident_minor_faults",
        "reason_predicates", "host_timed_updates", "resident_timed_updates",
    }
    integer_fields = {
        "schema_version", "rank", "repetition", "device_ordinal", "host_d2h",
        "resident_d2h", "host_cpu_reduction_calls", "host_cuda_reduction_calls",
        "resident_cpu_reduction_calls", "resident_cuda_reduction_calls",
        "resident_cuda_kernel_launches", "resident_result_d2h",
        "host_voluntary_context_switches", "host_involuntary_context_switches",
        "resident_voluntary_context_switches",
        "resident_involuntary_context_switches", "host_major_faults",
        "resident_major_faults", "host_minor_faults", "resident_minor_faults",
    }
    reason_bindings = {
        "host_involuntary_context_switches_nonzero":
            "host_involuntary_context_switches",
        "resident_involuntary_context_switches_nonzero":
            "resident_involuntary_context_switches",
        "host_major_faults_nonzero": "host_major_faults",
        "resident_major_faults_nonzero": "resident_major_faults",
    }
    repetition = records[0].get("repetition")
    if (
        not _is_plain_int(repetition)
        or repetition < 0
        or repetition >= PROFILE["paired_repetitions"]
    ):
        raise EvidenceError("LDOS rejected repetition coordinate is invalid")
    expected_order = "host-resident" if repetition % 2 == 0 else "resident-host"
    rejected_positions = [
        index for index, line in enumerate(stdout_lines)
        if line.startswith(REJECTED_PREFIX)
    ]
    first_rejected_position = rejected_positions[0]
    success_after_rejection = any(
        index > first_rejected_position
        and (
            line.startswith(RANK_PREFIX)
            or line.startswith(PAIR_PREFIX)
            or line.startswith(RUNTIME_PREFIX)
            or line.startswith(BENCHMARK_PREFIX)
            or line.startswith("PASS:")
        )
        for index, line in enumerate(stdout_lines)
    )
    if success_after_rejection:
        raise EvidenceError("LDOS success evidence appeared after rejected repetition")
    accepted_rank_lines = [
        (index, line) for index, line in enumerate(stdout_lines)
        if line.startswith(RANK_PREFIX)
    ]
    accepted_pair_lines = [
        (index, line) for index, line in enumerate(stdout_lines)
        if line.startswith(PAIR_PREFIX)
    ]
    if (
        len(accepted_rank_lines) != repetition * expected_ranks
        or len(accepted_pair_lines) != repetition
        or any(index >= first_rejected_position for index, _ in accepted_rank_lines)
        or any(index >= first_rejected_position for index, _ in accepted_pair_lines)
    ):
        raise EvidenceError(
            "LDOS rejected attempt lacks the exact accepted repetition prefix"
        )
    accepted_markers = [
        line for index, line in enumerate(stdout_lines)
        if index < first_rejected_position
        and (line.startswith(RANK_PREFIX) or line.startswith(PAIR_PREFIX))
    ]
    expected_marker_count = repetition * (expected_ranks + 1)
    if len(accepted_markers) != expected_marker_count:
        raise EvidenceError("LDOS rejected accepted-prefix marker sequence is invalid")
    accepted_rank_coordinates: set[tuple[int, int]] = set()
    marker_index = 0
    for accepted_repetition in range(repetition):
        for expected_rank in range(expected_ranks):
            line = accepted_markers[marker_index]
            marker_index += 1
            if not line.startswith(RANK_PREFIX):
                raise EvidenceError(
                    "LDOS rejected accepted-prefix marker order is invalid"
                )
            accepted = _canonical_json_object(line, RANK_PREFIX)
            rank = accepted.get("rank")
            coordinate_repetition = accepted.get("repetition")
            if (
                not _is_plain_int(rank)
                or not _is_plain_int(coordinate_repetition)
                or rank != expected_rank
                or coordinate_repetition != accepted_repetition
                or (rank, coordinate_repetition) in accepted_rank_coordinates
            ):
                raise EvidenceError(
                    "LDOS rejected accepted-prefix rank coordinate/order is invalid"
                )
            accepted_rank_coordinates.add((rank, coordinate_repetition))
        line = accepted_markers[marker_index]
        marker_index += 1
        if not line.startswith(PAIR_PREFIX):
            raise EvidenceError(
                "LDOS rejected accepted-prefix marker order is invalid"
            )
        accepted = _canonical_json_object(line, PAIR_PREFIX)
        accepted_order = (
            "host-resident" if accepted_repetition % 2 == 0 else "resident-host"
        )
        if (
            not _is_plain_int(accepted.get("repetition"))
            or accepted["repetition"] != accepted_repetition
            or accepted.get("order") != accepted_order
        ):
            raise EvidenceError(
                "LDOS rejected accepted-prefix pair schedule is invalid"
            )
    if accepted_rank_coordinates != {
        (rank, accepted_repetition)
        for accepted_repetition in range(repetition)
        for rank in range(expected_ranks)
    }:
        raise EvidenceError("LDOS rejected accepted-prefix rank matrix is incomplete")
    expected_focused_prefixes: list[str] = []
    for _accepted_repetition in range(repetition):
        expected_focused_prefixes.extend([RANK_PREFIX] * expected_ranks)
        expected_focused_prefixes.append(PAIR_PREFIX)
    expected_focused_prefixes.extend([REJECTED_PREFIX] * expected_ranks)
    if [marker for marker, _line in focused_records] != expected_focused_prefixes:
        raise EvidenceError("LDOS rejected focused marker topology is invalid")
    accepted_prefix = _validate_accepted_prefix_with_success_parser(
        accepted_markers,
        repetitions=repetition,
        expected_ranks=expected_ranks,
        expected_device_uuids=expected_device_uuids,
        expected_worker_cpu_masks=expected_worker_cpu_masks,
    )
    any_reason = False
    for expected_rank, record in enumerate(records):
        _require_exact_keys(record, keys, "LDOS rejected record")
        if any(
            not _is_plain_int(record[name])
            or record[name] < 0
            or record[name] > (1 << 64) - 1
            for name in integer_fields
        ):
            raise EvidenceError("LDOS rejected accounting contains an invalid integer")
        if (
            record["schema_version"] != 1
            or record["rank"] != expected_rank
            or record["repetition"] != repetition
            or record["order"] != expected_order
            or record["global_gate_accepted"] is not False
            or record["device_ordinal"] != expected_rank
            or not isinstance(record["device_uuid"], str)
            or not CUDA_UUID_PATTERN.fullmatch(record["device_uuid"])
        ):
            raise EvidenceError("LDOS rejected rank/order/device binding is invalid")
        affinity = record["cpu_affinity"]
        if (
            not isinstance(affinity, list)
            or not affinity
            or affinity != sorted(set(affinity))
            or any(not _is_plain_int(cpu) or cpu < 0 for cpu in affinity)
        ):
            raise EvidenceError("LDOS rejected CPU affinity is invalid")
        if expected_device_uuids is not None:
            expected_cuda_uuid = expected_device_uuids[expected_rank].removeprefix(
                "GPU-"
            ).replace("-", "")
            if record["device_uuid"] != expected_cuda_uuid:
                raise EvidenceError("LDOS rejected UUID differs from selected device")
        if expected_worker_cpu_masks is not None and affinity != sorted(
            expected_worker_cpu_masks[expected_rank]
        ):
            raise EvidenceError("LDOS rejected affinity differs from lane plan")
        reasons = record["reason_predicates"]
        if not isinstance(reasons, dict) or set(reasons) != set(reason_bindings):
            raise EvidenceError("LDOS rejected reason predicate schema is invalid")
        for reason, counter in reason_bindings.items():
            if not isinstance(reasons[reason], bool) or reasons[reason] != (
                record[counter] != 0
            ):
                raise EvidenceError("LDOS rejected reason predicate is not recomputable")
            any_reason = any_reason or reasons[reason]
        for segment in ("host", "resident"):
            updates = record[f"{segment}_timed_updates"]
            if not isinstance(updates, list) or len(updates) != PROFILE["updates"]:
                raise EvidenceError("LDOS rejected timed-update cardinality is incomplete")
            previous_stop = -1
            for update in updates:
                update_keys = {
                    "start_monotonic_ns", "stop_monotonic_ns", "cpu_before",
                    "cpu_after", "affinity_before", "affinity_after", "elapsed_ns",
                    "voluntary_context_switches", "involuntary_context_switches",
                    "minor_faults", "major_faults",
                }
                if not isinstance(update, dict) or set(update) != update_keys:
                    raise EvidenceError("LDOS rejected timed-update schema is invalid")
                scalar_names = update_keys - {"affinity_before", "affinity_after"}
                if any(
                    not _is_plain_int(update[name])
                    or update[name] < 0
                    or update[name] > (1 << 64) - 1
                    for name in scalar_names
                ):
                    raise EvidenceError("LDOS rejected timed-update integer is invalid")
                if (
                    update["start_monotonic_ns"] <= previous_stop
                    or update["stop_monotonic_ns"] <= update["start_monotonic_ns"]
                    or update["elapsed_ns"]
                    != update["stop_monotonic_ns"] - update["start_monotonic_ns"]
                    or update["cpu_before"] not in affinity
                    or update["cpu_after"] not in affinity
                    or update["affinity_before"] != affinity
                    or update["affinity_after"] != affinity
                ):
                    raise EvidenceError("LDOS rejected timed-update binding is invalid")
                previous_stop = update["stop_monotonic_ns"]
            for counter in (
                "voluntary_context_switches", "involuntary_context_switches",
                "minor_faults", "major_faults",
            ):
                if sum(update[counter] for update in updates) != record[
                    f"{segment}_{counter}"
                ]:
                    raise EvidenceError("LDOS rejected timing sum differs from aggregate")
        first_segment, second_segment = (
            ("host", "resident")
            if expected_order == "host-resident"
            else ("resident", "host")
        )
        if (
            record[f"{first_segment}_timed_updates"][-1]["stop_monotonic_ns"]
            > record[f"{second_segment}_timed_updates"][0]["start_monotonic_ns"]
        ):
            raise EvidenceError(
                "LDOS rejected AB/BA segment timestamps violate scheduled order"
            )
        if (
            record["host_cpu_reduction_calls"] != PROFILE["updates"]
            or record["host_cuda_reduction_calls"] != 0
            or record["resident_cpu_reduction_calls"] != 0
            or record["resident_cuda_reduction_calls"] != PROFILE["updates"]
            or record["resident_cuda_kernel_launches"] != 2 * PROFILE["updates"]
            or record["resident_result_d2h"]
            != PROFILE["resident_result_d2h_bytes_per_rank"]
            or record["resident_d2h"] != record["resident_result_d2h"]
            or record["host_d2h"] <= record["resident_d2h"]
        ):
            raise EvidenceError("LDOS rejected transfer/counter evidence is invalid")
        host_samples = _sample_vector(record["host_samples"], "rejected host samples")
        resident_samples = _sample_vector(
            record["resident_samples"], "rejected resident samples"
        )
        if (
            not isinstance(record["host_digest"], str)
            or not DIGEST_PATTERN.fullmatch(record["host_digest"])
            or not isinstance(record["resident_digest"], str)
            or not DIGEST_PATTERN.fullmatch(record["resident_digest"])
            or _sample_digest(host_samples) != record["host_digest"]
            or _sample_digest(resident_samples) != record["resident_digest"]
        ):
            raise EvidenceError("LDOS rejected physics digest is invalid")
        absolute, relative = _sample_errors(host_samples, resident_samples)
        if not math.isclose(
            _require_nonnegative_float(record["max_absolute_error"], "rejected absolute error"),
            absolute, rel_tol=1e-15, abs_tol=0.0,
        ) or not math.isclose(
            _require_nonnegative_float(record["max_relative_error"], "rejected relative error"),
            relative, rel_tol=1e-15, abs_tol=0.0,
        ):
            raise EvidenceError("LDOS rejected physics error is not recomputable")
        _require_nonnegative_float(record["host_seconds"], "rejected host seconds", positive=True)
        _require_nonnegative_float(
            record["resident_seconds"], "rejected resident seconds", positive=True
        )
    if not any_reason:
        raise EvidenceError("LDOS rejected repetition has no true rejection predicate")
    if len({record["device_uuid"] for record in records}) != expected_ranks:
        raise EvidenceError("LDOS rejected ranks do not bind distinct CUDA UUIDs")
    affinity_sets = [set(record["cpu_affinity"]) for record in records]
    if any(
        first & second
        for index, first in enumerate(affinity_sets)
        for second in affinity_sets[index + 1 :]
    ):
        raise EvidenceError("LDOS rejected rank CPU affinities overlap")
    first_segment, second_segment = (
        ("host", "resident")
        if expected_order == "host-resident"
        else ("resident", "host")
    )
    if max(
        record[f"{first_segment}_timed_updates"][-1]["stop_monotonic_ns"]
        for record in records
    ) > min(
        record[f"{second_segment}_timed_updates"][0]["start_monotonic_ns"]
        for record in records
    ):
        raise EvidenceError(
            "LDOS rejected collective AB/BA segment windows overlap across ranks"
        )
    rejected_first_start = min(
        record[f"{first_segment}_timed_updates"][0]["start_monotonic_ns"]
        for record in records
    )
    accepted_bounds = accepted_prefix["timeline_bounds"]
    if (
        accepted_bounds
        and accepted_bounds[-1]["global_second_stop_monotonic_ns"]
        > rejected_first_start
    ):
        raise EvidenceError(
            "LDOS accepted prefix overlaps the rejected repetition"
        )
    for segment in ("host", "resident"):
        reported = {float(record[f"{segment}_seconds"]) for record in records}
        derived = max(
            sum(update["elapsed_ns"] for update in record[f"{segment}_timed_updates"])
            for record in records
        ) / 1_000_000_000.0
        if len(reported) != 1 or abs(next(iter(reported)) - derived) > 5e-10:
            raise EvidenceError("LDOS rejected seconds differ from raw timed updates")
    physics_fields = (
        "host_samples", "resident_samples", "host_digest", "resident_digest",
        "max_absolute_error", "max_relative_error",
    )
    if any(
        len({
            json.dumps(record[field], separators=(",", ":"), sort_keys=True)
            for record in records
        }) != 1
        for field in physics_fields
    ):
        raise EvidenceError(
            "LDOS rejected globally reduced physics differs across ranks"
        )
    return {
        "schema_version": 1,
        "repetition": repetition,
        "order": expected_order,
        "global_gate_accepted": False,
        "records": records,
        "accepted_prefix": accepted_prefix,
    }


def _cross_bind_rejected_timing(
    rejected: dict[str, Any], *, go_monotonic_ns: int,
    process_finished_monotonic_ns: int,
) -> dict[str, Any]:
    """Bind accepted-prefix and rejected raw intervals to lane endpoints."""

    accepted = rejected.get("accepted_prefix")
    records = rejected.get("records")
    if (
        not isinstance(accepted, dict)
        or not isinstance(records, list)
        or not _is_plain_int(go_monotonic_ns)
        or not 0 < go_monotonic_ns <= (1 << 64) - 1
        or not _is_plain_int(process_finished_monotonic_ns)
        or not 0 < process_finished_monotonic_ns <= (1 << 64) - 1
    ):
        raise EvidenceError("LDOS rejected host timeline endpoints are incomplete")
    all_records = [*accepted.get("rank_accounting", []), *records]
    starts: list[int] = []
    stops: list[int] = []
    for record in all_records:
        if not isinstance(record, dict):
            raise EvidenceError("LDOS rejected host timeline matrix is invalid")
        for segment in ("host", "resident"):
            updates = record.get(f"{segment}_timed_updates")
            if not isinstance(updates, list) or not updates:
                raise EvidenceError("LDOS rejected host timeline matrix is invalid")
            starts.append(updates[0]["start_monotonic_ns"])
            stops.append(updates[-1]["stop_monotonic_ns"])
    if not starts or min(starts) < go_monotonic_ns:
        raise EvidenceError("LDOS rejected timing began before controller GO")
    if max(stops) > process_finished_monotonic_ns:
        raise EvidenceError("LDOS rejected timing exceeds process completion")
    return {
        "go_monotonic_ns": go_monotonic_ns,
        "first_update_start_monotonic_ns": min(starts),
        "last_update_stop_monotonic_ns": max(stops),
        "process_finished_monotonic_ns": process_finished_monotonic_ns,
    }


def _success_timeline_bounds(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive and cross-check collective repetition bounds from raw updates."""

    ranks = parsed.get("ranks")
    repetitions = parsed.get("paired_repetitions")
    pairs = parsed.get("pairs")
    accounting = parsed.get("rank_accounting")
    if (
        not _is_plain_int(ranks)
        or not _is_plain_int(repetitions)
        or not isinstance(pairs, list)
        or not isinstance(accounting, list)
    ):
        raise EvidenceError("LDOS success timeline inputs are invalid")
    by_coordinate = {
        (record.get("rank"), record.get("repetition")): record
        for record in accounting
        if isinstance(record, dict)
    }
    if len(by_coordinate) != ranks * repetitions:
        raise EvidenceError("LDOS success timeline matrix is incomplete")
    bounds: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        pair = pairs[repetition]
        first_segment, second_segment = (
            ("host", "resident")
            if pair.get("order") == "host-resident"
            else ("resident", "host")
        )
        first_starts_by_rank = []
        second_stops_by_rank = []
        for rank in range(ranks):
            record = by_coordinate[(rank, repetition)]
            first_starts_by_rank.append(
                record[f"{first_segment}_timed_updates"][0]["start_monotonic_ns"]
            )
            second_stops_by_rank.append(
                record[f"{second_segment}_timed_updates"][-1]["stop_monotonic_ns"]
            )
        bounds.append({
            "repetition": repetition,
            "first_segment": first_segment,
            "second_segment": second_segment,
            "first_start_monotonic_ns_by_rank": first_starts_by_rank,
            "second_stop_monotonic_ns_by_rank": second_stops_by_rank,
            "global_first_start_monotonic_ns": min(first_starts_by_rank),
            "global_second_stop_monotonic_ns": max(second_stops_by_rank),
        })
    for previous, current in zip(bounds, bounds[1:]):
        if (
            previous["global_second_stop_monotonic_ns"]
            > current["global_first_start_monotonic_ns"]
        ):
            raise EvidenceError(
                "LDOS adjacent collective repetition windows overlap"
            )
    return bounds


def _validate_success_host_timeline(
    parsed: dict[str, Any], host_runtime: dict[str, Any],
    process_finished_monotonic_ns: int,
) -> list[dict[str, Any]]:
    """Bind raw worker intervals to controller GO/RESULTS_READY/process exit."""

    bounds = _success_timeline_bounds(parsed)
    go_ns = host_runtime.get("go_monotonic_ns")
    journal_finished_ns = host_runtime.get("process_finished_monotonic_ns")
    result_records = host_runtime.get("results_ready_records")
    ranks = parsed.get("ranks")
    if (
        not _is_plain_int(go_ns)
        or not 0 < go_ns <= (1 << 64) - 1
        or not _is_plain_int(process_finished_monotonic_ns)
        or not 0 < process_finished_monotonic_ns <= (1 << 64) - 1
        or journal_finished_ns != process_finished_monotonic_ns
        or not isinstance(result_records, list)
        or len(result_records) != ranks
    ):
        raise EvidenceError("LDOS host/worker timeline endpoints are incomplete")
    ready_by_rank = {
        record.get("rank"): record.get("monotonic_ns")
        for record in result_records if isinstance(record, dict)
    }
    if set(ready_by_rank) != set(range(ranks)) or any(
        not _is_plain_int(value) or not 0 < value <= (1 << 64) - 1
        for value in ready_by_rank.values()
    ):
        raise EvidenceError("LDOS RESULTS_READY timeline matrix is invalid")
    first = bounds[0]
    last = bounds[-1]
    for rank in range(ranks):
        if first["first_start_monotonic_ns_by_rank"][rank] < go_ns:
            raise EvidenceError("LDOS worker timing began before controller GO")
        final_stop = last["second_stop_monotonic_ns_by_rank"][rank]
        if (
            final_stop > ready_by_rank[rank]
            or ready_by_rank[rank] > process_finished_monotonic_ns
        ):
            raise EvidenceError(
                "LDOS worker timing does not precede RESULTS_READY/process exit"
            )
    return bounds


def parse_benchmark_output(
    stdout: str,
    stderr: str,
    *,
    returncode: int,
    expected_ranks: int,
    expected_device_uuids: list[str] | None = None,
    _enforce_performance_gate: bool = True,
) -> dict[str, Any]:
    """Parse one focused benchmark lane and reject ambiguous output."""

    if returncode != 0:
        raise EvidenceError(f"LDOS benchmark exited with status {returncode}")
    focused_records = _focused_marker_lines(stdout, stderr)
    combined = stdout + "\n" + stderr
    # Canonical structured records are validated field-by-field below.  Their
    # JSON string values may legitimately contain path components such as
    # "failed-runs" or "skip"; protocol failure tokens remain forbidden in
    # every unstructured diagnostic line.
    token = _unstructured_failure_token(stdout, stderr)
    if token:
        raise EvidenceError(f"LDOS benchmark emitted forbidden {token.group(0)!r}")
    allowed_markers = {PAIR_PREFIX, RANK_PREFIX, RUNTIME_PREFIX, REJECTED_PREFIX}
    unknown_markers = [
        marker
        for marker in GPMEEP_MARKER_PATTERN.findall(combined)
        if marker not in allowed_markers
    ]
    if unknown_markers:
        raise EvidenceError(
            "LDOS benchmark emitted unknown gpmeep marker(s): "
            + ",".join(sorted(set(unknown_markers)))
        )
    if REJECTED_PREFIX in combined:
        raise EvidenceError("LDOS successful output contains a rejected repetition")

    stdout_lines = stdout.splitlines()
    stderr_lines = stderr.splitlines()
    if any(marker in stderr for marker in allowed_markers):
        raise EvidenceError("LDOS structured evidence markers are stdout-only")
    records = [line for line in stdout_lines if line.startswith(BENCHMARK_PREFIX)]
    stderr_records = [
        line for line in stderr_lines if line.startswith(BENCHMARK_PREFIX)
    ]
    if len(records) != 1 or stderr_records:
        raise EvidenceError(
            "expected exactly one LDOS benchmark record on stdout and none on stderr"
        )
    pass_count = sum(line == PASS_LINE for line in stdout_lines)
    stderr_pass_count = sum(line == PASS_LINE for line in stderr_lines)
    if pass_count != 1 or stderr_pass_count:
        raise EvidenceError(
            "expected exactly one focused LDOS PASS on stdout and none on stderr"
        )

    match = BENCHMARK_PATTERN.fullmatch(records[0])
    if not match:
        raise EvidenceError("LDOS benchmark record does not match the exact schema")
    integers = {
        name: _parse_exact_int(match.group(name), name)
        for name in ("ranks", "pixels", "updates", "repetitions")
    }
    expected_integers = {
        "ranks": expected_ranks,
        "pixels": PROFILE["pixels"],
        "updates": PROFILE["updates"],
        "repetitions": PROFILE["paired_repetitions"],
    }
    for name, expected in expected_integers.items():
        if integers[name] != expected:
            raise EvidenceError(
                f"LDOS benchmark {name}={integers[name]} differs from fixed {expected}"
            )
    pair_lines = [line for line in stdout_lines if line.startswith(PAIR_PREFIX)]
    rank_lines = [line for line in stdout_lines if line.startswith(RANK_PREFIX)]
    runtime_lines = [line for line in stdout_lines if line.startswith(RUNTIME_PREFIX)]
    if len(pair_lines) != PROFILE["paired_repetitions"]:
        raise EvidenceError("expected exactly five raw LDOS pair records")
    if len(rank_lines) != PROFILE["paired_repetitions"] * expected_ranks:
        raise EvidenceError("LDOS per-rank accounting record count is incomplete")
    if len(runtime_lines) != expected_ranks:
        raise EvidenceError("LDOS runtime mapping attestation count is incomplete")

    pairs: list[dict[str, Any]] = []
    pair_keys = {
        "repetition", "order", "host_seconds", "resident_seconds", "speedup",
        "max_absolute_error", "max_relative_error", "host_digest",
        "resident_digest", "host_samples", "resident_samples",
    }
    for index, line in enumerate(pair_lines):
        pair = _canonical_json_object(line, PAIR_PREFIX)
        _require_exact_keys(pair, pair_keys, "LDOS pair record")
        if pair["repetition"] != index or not _is_plain_int(pair["repetition"]):
            raise EvidenceError("LDOS pair repetitions are not canonical and ordered")
        expected_order = "host-resident" if index % 2 == 0 else "resident-host"
        if pair["order"] != expected_order:
            raise EvidenceError("LDOS pair AB/BA order differs from the fixed schedule")
        host_seconds = _require_nonnegative_float(
            pair["host_seconds"], "host seconds", positive=True
        )
        resident_seconds = _require_nonnegative_float(
            pair["resident_seconds"], "resident seconds", positive=True
        )
        speedup = _require_nonnegative_float(pair["speedup"], "speedup", positive=True)
        recomputed_speedup = host_seconds / resident_seconds
        if speedup != recomputed_speedup:
            raise EvidenceError("pair speedup is not independently recomputable")
        host_samples = _sample_vector(pair["host_samples"], "host samples")
        resident_samples = _sample_vector(pair["resident_samples"], "resident samples")
        if not isinstance(pair["host_digest"], str) or not DIGEST_PATTERN.fullmatch(pair["host_digest"]):
            raise EvidenceError("host sample digest is invalid")
        if not isinstance(pair["resident_digest"], str) or not DIGEST_PATTERN.fullmatch(pair["resident_digest"]):
            raise EvidenceError("resident sample digest is invalid")
        if _sample_digest(host_samples) != pair["host_digest"] or _sample_digest(resident_samples) != pair["resident_digest"]:
            raise EvidenceError("sample digest is not independently recomputable")
        maximum_absolute, maximum_relative = _sample_errors(host_samples, resident_samples)
        reported_absolute = _require_nonnegative_float(
            pair["max_absolute_error"], "maximum absolute error"
        )
        reported_relative = _require_nonnegative_float(
            pair["max_relative_error"], "maximum relative error"
        )
        if not math.isclose(reported_absolute, maximum_absolute, rel_tol=1e-15, abs_tol=0.0):
            raise EvidenceError("maximum absolute error is not independently recomputable")
        if not math.isclose(reported_relative, maximum_relative, rel_tol=1e-15, abs_tol=0.0):
            raise EvidenceError("maximum relative error is not independently recomputable")
        for sample_index, (host, resident) in enumerate(zip(host_samples, resident_samples)):
            error = abs(host - resident)
            tolerance = 3e-6 + 3e-4 * max(abs(host), abs(resident))
            if error > tolerance:
                raise EvidenceError(f"LDOS pair sample {sample_index} exceeds numerical tolerance")
        pairs.append(
            {
                **pair,
                "host_seconds": host_seconds,
                "resident_seconds": resident_seconds,
                "speedup": recomputed_speedup,
                "max_absolute_error": maximum_absolute,
                "max_relative_error": maximum_relative,
            }
        )
    if len({pair["host_digest"] for pair in pairs}) != 1 or len(
        {pair["resident_digest"] for pair in pairs}
    ) != 1:
        raise EvidenceError(
            "LDOS host/resident samples are not bitwise deterministic across repetitions"
        )

    rank_keys = {
        "rank", "repetition", "device_ordinal", "device_uuid", "host_d2h",
        "resident_d2h", "host_cpu_reduction_calls", "host_cuda_reduction_calls",
        "resident_cpu_reduction_calls", "resident_cuda_reduction_calls",
        "resident_cuda_kernel_launches", "resident_result_d2h", "cpu_affinity",
        "host_voluntary_context_switches", "host_involuntary_context_switches",
        "resident_voluntary_context_switches",
        "resident_involuntary_context_switches",
        "host_major_faults", "resident_major_faults",
        "host_minor_faults", "resident_minor_faults",
        "host_timed_updates", "resident_timed_updates",
    }
    rank_records = [_canonical_json_object(line, RANK_PREFIX) for line in rank_lines]
    by_rank_repetition: dict[tuple[int, int], dict[str, Any]] = {}
    for record in rank_records:
        _require_exact_keys(record, rank_keys, "LDOS rank record")
        integer_fields = rank_keys - {
            "device_uuid", "cpu_affinity", "host_timed_updates",
            "resident_timed_updates",
        }
        if any(
            not _is_plain_int(record[name])
            or record[name] < 0
            or record[name] > (1 << 64) - 1
            for name in integer_fields
        ):
            raise EvidenceError("LDOS rank accounting contains a noncanonical integer")
        rank = record["rank"]
        repetition = record["repetition"]
        if rank >= expected_ranks or repetition >= PROFILE["paired_repetitions"]:
            raise EvidenceError("LDOS rank accounting coordinate is out of range")
        coordinate = (rank, repetition)
        if coordinate in by_rank_repetition:
            raise EvidenceError("duplicate LDOS per-rank accounting coordinate")
        by_rank_repetition[coordinate] = record
        if record["device_ordinal"] != rank:
            raise EvidenceError("worker CUDA ordinal differs from explicit rank mapping")
        if not isinstance(record["device_uuid"], str) or not CUDA_UUID_PATTERN.fullmatch(record["device_uuid"]):
            raise EvidenceError("worker CUDA UUID is not a strict physical UUID")
        if record["host_cpu_reduction_calls"] != PROFILE["updates"] or record["host_cuda_reduction_calls"] != 0:
            raise EvidenceError("host fallback LDOS counters are not exact")
        if record["resident_cpu_reduction_calls"] != 0 or record["resident_cuda_reduction_calls"] != PROFILE["updates"]:
            raise EvidenceError("resident LDOS reduction counters are not exact")
        if record["resident_cuda_kernel_launches"] != 2 * PROFILE["updates"]:
            raise EvidenceError("resident LDOS kernel count is not exact")
        expected_result = PROFILE["resident_result_d2h_bytes_per_rank"]
        if record["resident_result_d2h"] != expected_result or record["resident_d2h"] != expected_result:
            raise EvidenceError("resident per-rank D2H accounting is not exact")
        if record["host_d2h"] <= record["resident_d2h"]:
            raise EvidenceError("host per-rank D2H does not exceed resident D2H")
        affinity = record["cpu_affinity"]
        if (
            not isinstance(affinity, list)
            or not affinity
            or affinity != sorted(set(affinity))
            or any(not _is_plain_int(cpu) or cpu < 0 for cpu in affinity)
        ):
            raise EvidenceError("LDOS worker CPU affinity is not a canonical core mask")
        for segment in ("host", "resident"):
            updates = record[f"{segment}_timed_updates"]
            if not isinstance(updates, list) or len(updates) != PROFILE["updates"]:
                raise EvidenceError("LDOS timed-update cardinality is incomplete")
            previous_stop = -1
            for update in updates:
                if not isinstance(update, dict) or set(update) != {
                    "start_monotonic_ns", "stop_monotonic_ns", "cpu_before",
                    "cpu_after", "affinity_before", "affinity_after",
                    "elapsed_ns", "voluntary_context_switches",
                    "involuntary_context_switches", "minor_faults", "major_faults",
                }:
                    raise EvidenceError("LDOS timed-update schema is invalid")
                integer_names = (
                    "start_monotonic_ns", "stop_monotonic_ns",
                    "cpu_before", "cpu_after",
                    "elapsed_ns", "voluntary_context_switches",
                    "involuntary_context_switches", "minor_faults", "major_faults",
                )
                if any(
                    not _is_plain_int(update[name])
                    or update[name] < 0
                    or update[name] > (1 << 64) - 1
                    for name in integer_names
                ):
                    raise EvidenceError("LDOS timed-update integer is invalid")
                if (
                    update["start_monotonic_ns"] <= previous_stop
                    or update["stop_monotonic_ns"] <= update["start_monotonic_ns"]
                    or update["elapsed_ns"]
                    != update["stop_monotonic_ns"] - update["start_monotonic_ns"]
                    or update["involuntary_context_switches"] != 0
                    or update["major_faults"] != 0
                    or update["cpu_before"] not in affinity
                    or update["cpu_after"] not in affinity
                    or update["cpu_before"] != update["cpu_after"]
                    or update["affinity_before"] != affinity
                    or update["affinity_after"] != affinity
                ):
                    raise EvidenceError("LDOS timed-update CPU/interval binding changed")
                previous_stop = update["stop_monotonic_ns"]
            aggregate_bindings = {
                "voluntary_context_switches": f"{segment}_voluntary_context_switches",
                "involuntary_context_switches": f"{segment}_involuntary_context_switches",
                "minor_faults": f"{segment}_minor_faults",
                "major_faults": f"{segment}_major_faults",
            }
            for update_name, aggregate_name in aggregate_bindings.items():
                if sum(update[update_name] for update in updates) != record[aggregate_name]:
                    raise EvidenceError("LDOS timed-update usage sum differs from aggregate")
        context_switch_gate = PROFILE[
            "maximum_involuntary_context_switches_per_segment"
        ]
        if (
            record["host_involuntary_context_switches"] > context_switch_gate
            or record["resident_involuntary_context_switches"] > context_switch_gate
            or record["host_major_faults"] != 0
            or record["resident_major_faults"] != 0
        ):
            raise EvidenceError("LDOS timing scheduling/fault gate was exceeded")
    if set(by_rank_repetition) != {
        (rank, repetition)
        for rank in range(expected_ranks)
        for repetition in range(PROFILE["paired_repetitions"])
    }:
        raise EvidenceError("LDOS per-rank accounting matrix is incomplete")
    for rank in range(expected_ranks):
        records_for_rank = [by_rank_repetition[(rank, repetition)] for repetition in range(PROFILE["paired_repetitions"])]
        if len({record["host_d2h"] for record in records_for_rank}) != 1 or len({record["resident_d2h"] for record in records_for_rank}) != 1:
            raise EvidenceError("per-rank D2H accounting changed across paired repetitions")
        if len({record["device_uuid"] for record in records_for_rank}) != 1:
            raise EvidenceError("worker CUDA UUID changed across paired repetitions")
        if len({tuple(record["cpu_affinity"]) for record in records_for_rank}) != 1:
            raise EvidenceError("worker CPU affinity changed across paired repetitions")
    for repetition in range(PROFILE["paired_repetitions"]):
        pair = pairs[repetition]
        for segment in ("host", "resident"):
            maximum_rank_ns = max(
                sum(
                    update["elapsed_ns"]
                    for update in by_rank_repetition[(rank, repetition)][
                        f"{segment}_timed_updates"
                    ]
                )
                for rank in range(expected_ranks)
            )
            derived_seconds = maximum_rank_ns / 1_000_000_000.0
            if abs(pair[f"{segment}_seconds"] - derived_seconds) > 5e-10:
                raise EvidenceError(
                    "LDOS pair seconds differ from raw per-update CLOCK_MONOTONIC intervals"
                )
        first_segment, second_segment = (
            ("host", "resident")
            if pair["order"] == "host-resident"
            else ("resident", "host")
        )
        first_stops = [
            by_rank_repetition[(rank, repetition)][
                f"{first_segment}_timed_updates"
            ][-1]["stop_monotonic_ns"]
            for rank in range(expected_ranks)
        ]
        second_starts = [
            by_rank_repetition[(rank, repetition)][
                f"{second_segment}_timed_updates"
            ][0]["start_monotonic_ns"]
            for rank in range(expected_ranks)
        ]
        if any(
            first_stops[rank] > second_starts[rank]
            for rank in range(expected_ranks)
        ):
            raise EvidenceError(
                "LDOS per-rank AB/BA segment timestamps violate scheduled order"
            )
        if max(first_stops) > min(second_starts):
            raise EvidenceError(
                "LDOS collective AB/BA segment windows overlap across ranks"
            )
    rank_masks = [
        set(by_rank_repetition[(rank, 0)]["cpu_affinity"])
        for rank in range(expected_ranks)
    ]
    if any(
        first & second
        for index, first in enumerate(rank_masks)
        for second in rank_masks[index + 1 :]
    ):
        raise EvidenceError("MPI ranks were not pinned to disjoint worker cores")

    runtime_keys = {
        "rank", "device_ordinal", "device_uuid", "libmeep", "mappings",
        "special_mappings",
    }
    runtime_records = [_canonical_json_object(line, RUNTIME_PREFIX) for line in runtime_lines]
    runtime_by_rank: dict[int, dict[str, Any]] = {}
    for record in runtime_records:
        _require_exact_keys(record, runtime_keys, "LDOS runtime record")
        rank = record["rank"]
        if not _is_plain_int(rank) or rank < 0 or rank >= expected_ranks or rank in runtime_by_rank:
            raise EvidenceError("runtime mapping rank is invalid or duplicate")
        if record["device_ordinal"] != rank or not _is_plain_int(record["device_ordinal"]):
            raise EvidenceError("runtime CUDA ordinal differs from explicit rank mapping")
        if record["device_uuid"] != by_rank_repetition[(rank, 0)]["device_uuid"]:
            raise EvidenceError("runtime CUDA UUID differs from LDOS counter record")
        if not isinstance(record["libmeep"], str) or not pathlib.PurePosixPath(record["libmeep"]).is_absolute():
            raise EvidenceError("runtime libmeep path is not absolute")
        mappings = record["mappings"]
        if not isinstance(mappings, list) or not mappings:
            raise EvidenceError("runtime mapping closure is empty")
        identities: set[tuple[int, int]] = set()
        paths: set[str] = set()
        for mapping in mappings:
            if not isinstance(mapping, dict):
                raise EvidenceError("runtime mapping is not an object")
            _require_exact_keys(
                mapping,
                {"path", "device", "inode", "size", "mtime_ns", "ctime_ns"},
                "runtime mapping",
            )
            path = mapping["path"]
            if not isinstance(path, str) or not pathlib.PurePosixPath(path).is_absolute():
                raise EvidenceError("runtime mapping path is not absolute")
            if any(
                not _is_plain_int(mapping[name])
                or mapping[name] < 0
                or mapping[name] > (1 << 64) - 1
                for name in ("device", "inode", "size", "mtime_ns", "ctime_ns")
            ) or mapping["inode"] <= 0 or mapping["size"] <= 0:
                raise EvidenceError("runtime mapping identity is invalid")
            identity = (mapping["device"], mapping["inode"])
            if identity in identities or path in paths:
                raise EvidenceError("runtime mapping attestation contains duplicates")
            identities.add(identity)
            paths.add(path)
        if record["libmeep"] not in paths:
            raise EvidenceError("reported libmeep is absent from runtime mappings")
        special_mappings = record["special_mappings"]
        if not isinstance(special_mappings, list):
            raise EvidenceError("runtime special mappings are not a list")
        special_coordinates: set[tuple[str, int, int]] = set()
        for mapping in special_mappings:
            if not isinstance(mapping, dict):
                raise EvidenceError("runtime special mapping is not an object")
            _require_exact_keys(
                mapping,
                {
                    "path", "kind", "device", "inode", "mode", "rdev",
                    "deleted", "executable",
                },
                "runtime special mapping",
            )
            path = mapping["path"]
            kind = mapping["kind"]
            if (
                not isinstance(path, str)
                or not pathlib.PurePosixPath(path).is_absolute()
                or kind not in {"nvidia-character-device", "ephemeral-data"}
                or not isinstance(mapping["deleted"], bool)
                or mapping["executable"] is not False
                or any(
                    not _is_plain_int(mapping[name])
                    or mapping[name] < 0
                    or mapping[name] > (1 << 64) - 1
                    for name in ("device", "inode", "mode", "rdev")
                )
            ):
                raise EvidenceError("runtime special mapping schema is invalid")
            if kind == "nvidia-character-device":
                if (
                    mapping["deleted"]
                    or mapping["inode"] <= 0
                    or not re.fullmatch(
                        r"/dev/nvidia(?:[0-9]+|ctl|-uvm(?:-tools)?)", path
                    )
                ):
                    raise EvidenceError("NVIDIA character mapping is invalid")
            elif not (
                path == "/dev/zero"
                or path.startswith("/dev/shm/")
                or path.startswith("/memfd:")
            ):
                raise EvidenceError("ephemeral data mapping path is not whitelisted")
            coordinate = (path, mapping["device"], mapping["inode"])
            if coordinate in special_coordinates:
                raise EvidenceError("runtime special mapping contains duplicates")
            special_coordinates.add(coordinate)
        runtime_by_rank[rank] = record
    if set(runtime_by_rank) != set(range(expected_ranks)):
        raise EvidenceError("runtime mapping rank matrix is incomplete")

    expected_rank_coordinates = [
        (rank, repetition)
        for repetition in range(PROFILE["paired_repetitions"])
        for rank in range(expected_ranks)
    ]
    if [
        (record["rank"], record["repetition"]) for record in rank_records
    ] != expected_rank_coordinates:
        raise EvidenceError("LDOS success rank record topology is invalid")
    if [record["rank"] for record in runtime_records] != list(range(expected_ranks)):
        raise EvidenceError("LDOS success runtime record topology is invalid")
    expected_focused_prefixes: list[str] = []
    for _repetition in range(PROFILE["paired_repetitions"]):
        expected_focused_prefixes.extend([RANK_PREFIX] * expected_ranks)
        expected_focused_prefixes.append(PAIR_PREFIX)
    expected_focused_prefixes.extend([RUNTIME_PREFIX] * expected_ranks)
    expected_focused_prefixes.extend([BENCHMARK_PREFIX, PASS_LINE])
    if [marker for marker, _line in focused_records] != expected_focused_prefixes:
        raise EvidenceError("LDOS success focused marker topology is invalid")

    worker_uuids = [runtime_by_rank[rank]["device_uuid"] for rank in range(expected_ranks)]
    if len(set(worker_uuids)) != expected_ranks:
        raise EvidenceError("rank workers did not use distinct physical GPU UUIDs")
    if expected_device_uuids is not None:
        expected_cuda = [value.removeprefix("GPU-").replace("-", "") for value in expected_device_uuids[:expected_ranks]]
        if worker_uuids != expected_cuda:
            raise EvidenceError("worker CUDA UUIDs differ from explicit physical selection")

    speedups = [pair["speedup"] for pair in pairs]
    summary_speedups = [_finite_float(token, f"summary speedup {index}") for index, token in enumerate(match.group("speedups").split(","))]
    if speedups != summary_speedups:
        raise EvidenceError("summary paired speedups differ from raw pair timings")
    minimum = min(speedups)
    median = sorted(speedups)[len(speedups) // 2]
    if _finite_float(match.group("minimum"), "minimum speedup") != minimum:
        raise EvidenceError("reported minimum speedup differs from raw pairs")
    if _finite_float(match.group("median"), "median speedup") != median:
        raise EvidenceError("reported median speedup differs from raw pairs")
    if (
        _enforce_performance_gate
        and minimum <= PROFILE["minimum_paired_speedup_exclusive"]
    ):
        raise EvidenceError("minimum paired LDOS speedup does not exceed 1.20")

    timeline_bounds = _success_timeline_bounds({
        "ranks": integers["ranks"],
        "paired_repetitions": integers["repetitions"],
        "pairs": pairs,
        "rank_accounting": [
            by_rank_repetition[key] for key in sorted(by_rank_repetition)
        ],
    })

    return {
        "schema_version": 1,
        "ranks": integers["ranks"],
        "pixels": integers["pixels"],
        "updates": integers["updates"],
        "paired_repetitions": integers["repetitions"],
        "minimum_speedup": minimum,
        "median_speedup": median,
        "paired_speedups": speedups,
        "pairs": pairs,
        "rank_accounting": [by_rank_repetition[key] for key in sorted(by_rank_repetition)],
        "runtime_attestation": [runtime_by_rank[rank] for rank in range(expected_ranks)],
        "timeline_bounds": timeline_bounds,
        "per_rank_device_to_host_bytes": [
            {
                "rank": rank,
                "host": by_rank_repetition[(rank, 0)]["host_d2h"],
                "resident": by_rank_repetition[(rank, 0)]["resident_d2h"],
            }
            for rank in range(expected_ranks)
        ],
        "focused_pass": True,
    }


def _validate_accepted_prefix_with_success_parser(
    accepted_markers: list[str], *, repetitions: int, expected_ranks: int,
    expected_device_uuids: list[str] | None,
    expected_worker_cpu_masks: list[set[int]] | None,
) -> dict[str, Any]:
    """Reuse the complete success rank/pair validator for a rejected prefix."""

    if repetitions == 0:
        return {
            "repetitions": 0,
            "rank_accounting": [],
            "pairs": [],
            "timeline_bounds": [],
        }
    try:
        rank_templates: dict[int, dict[str, Any]] = {}
        pair_template: dict[str, Any] | None = None
        for line in accepted_markers:
            if line.startswith(RANK_PREFIX):
                record = _canonical_json_object(line, RANK_PREFIX)
                if record.get("repetition") == 0 and _is_plain_int(
                    record.get("rank")
                ):
                    rank_templates[record["rank"]] = record
            else:
                pair = _canonical_json_object(line, PAIR_PREFIX)
                if pair.get("repetition") == 0:
                    pair_template = pair
        if set(rank_templates) != set(range(expected_ranks)) or pair_template is None:
            raise EvidenceError("accepted prefix lacks a complete template repetition")

        synthetic_markers = list(accepted_markers)
        synthetic_pairs = [
            _canonical_json_object(line, PAIR_PREFIX)
            for line in accepted_markers if line.startswith(PAIR_PREFIX)
        ]

        actual_global_max_stop = 0
        for line in accepted_markers:
            if not line.startswith(RANK_PREFIX):
                continue
            accepted_record = _canonical_json_object(line, RANK_PREFIX)
            for segment in ("host", "resident"):
                updates = accepted_record.get(f"{segment}_timed_updates")
                if not isinstance(updates, list):
                    raise EvidenceError(
                        "accepted prefix full validation failed: timed updates "
                        "are not a list"
                    )
                for update in updates:
                    if not isinstance(update, dict) or not _is_plain_int(
                        update.get("stop_monotonic_ns")
                    ):
                        raise EvidenceError(
                            "accepted prefix timed-update stop is invalid"
                        )
                    actual_global_max_stop = max(
                        actual_global_max_stop, update["stop_monotonic_ns"]
                    )

        def rebase_updates(updates: Any, base_ns: int) -> tuple[list[dict[str, Any]], int]:
            if not isinstance(updates, list):
                raise EvidenceError("accepted prefix timed updates are not a list")
            rebased = copy.deepcopy(updates)
            cursor = base_ns
            for update in rebased:
                if not isinstance(update, dict) or not _is_plain_int(
                    update.get("elapsed_ns")
                ):
                    raise EvidenceError(
                        "accepted prefix timed-update duration is invalid"
                    )
                elapsed_ns = update["elapsed_ns"]
                if (
                    elapsed_ns <= 0
                    or cursor > (1 << 64) - 2
                    or elapsed_ns > (1 << 64) - 1 - (cursor + 1)
                ):
                    raise EvidenceError(
                        "accepted prefix synthetic timeline exceeds uint64"
                    )
                update["start_monotonic_ns"] = cursor + 1
                update["stop_monotonic_ns"] = cursor + 1 + elapsed_ns
                if update["stop_monotonic_ns"] > (1 << 64) - 11:
                    raise EvidenceError(
                        "accepted prefix synthetic timeline lacks uint64 headroom"
                    )
                cursor = update["stop_monotonic_ns"] + 10
            return rebased, cursor

        next_repetition_base = actual_global_max_stop + 1_000
        for repetition in range(repetitions, PROFILE["paired_repetitions"]):
            order = "host-resident" if repetition % 2 == 0 else "resident-host"
            first_segment, second_segment = (
                ("host", "resident")
                if order == "host-resident"
                else ("resident", "host")
            )
            generated_by_rank: dict[int, dict[str, Any]] = {}
            first_base = next_repetition_base
            first_stops: list[int] = []
            for rank in range(expected_ranks):
                generated = copy.deepcopy(rank_templates[rank])
                generated["repetition"] = repetition
                updates, stop_ns = rebase_updates(
                    generated[f"{first_segment}_timed_updates"], first_base
                )
                generated[f"{first_segment}_timed_updates"] = updates
                first_stops.append(stop_ns)
                generated_by_rank[rank] = generated
            second_base = max(first_stops) + 1_000
            second_stops: list[int] = []
            for rank in range(expected_ranks):
                generated = generated_by_rank[rank]
                updates, stop_ns = rebase_updates(
                    generated[f"{second_segment}_timed_updates"], second_base
                )
                generated[f"{second_segment}_timed_updates"] = updates
                second_stops.append(stop_ns)
                synthetic_markers.append(RANK_PREFIX + json.dumps(
                    generated, separators=(",", ":"), sort_keys=False
                ))
            pair = copy.deepcopy(pair_template)
            pair["repetition"] = repetition
            pair["order"] = order
            synthetic_pairs.append(pair)
            synthetic_markers.append(PAIR_PREFIX + json.dumps(
                pair, separators=(",", ":"), sort_keys=False
            ))
            next_repetition_base = max(second_stops) + 1_000

        runtime_lines: list[str] = []
        for rank in range(expected_ranks):
            libmeep = f"/accepted-prefix/libmeep-rank-{rank}.so"
            runtime_lines.append(RUNTIME_PREFIX + json.dumps(
                {
                    "rank": rank,
                    "device_ordinal": rank,
                    "device_uuid": rank_templates[rank]["device_uuid"],
                    "libmeep": libmeep,
                    "mappings": [{
                        "path": libmeep, "device": 1, "inode": rank + 1,
                        "size": 1, "mtime_ns": 1, "ctime_ns": 1,
                    }],
                    "special_mappings": [],
                },
                separators=(",", ":"), sort_keys=False,
            ))
        speedups = [
            _require_nonnegative_float(pair.get("speedup"), "accepted prefix speedup", positive=True)
            for pair in synthetic_pairs
        ]
        benchmark = (
            f"{BENCHMARK_PREFIX} ranks={expected_ranks} pixels={PROFILE['pixels']} "
            f"updates={PROFILE['updates']} "
            f"repetitions={PROFILE['paired_repetitions']} "
            f"minimum-speedup={min(speedups):.17g} "
            f"median-speedup={sorted(speedups)[len(speedups) // 2]:.17g} "
            "paired-speedups=" + ",".join(f"{value:.17g}" for value in speedups)
        )
        parsed = parse_benchmark_output(
            "\n".join((*synthetic_markers, *runtime_lines, benchmark, PASS_LINE, "")),
            "",
            returncode=0,
            expected_ranks=expected_ranks,
            expected_device_uuids=expected_device_uuids,
            _enforce_performance_gate=False,
        )
        if expected_worker_cpu_masks is not None:
            for record in parsed["rank_accounting"]:
                if record["repetition"] < repetitions and record[
                    "cpu_affinity"
                ] != sorted(expected_worker_cpu_masks[record["rank"]]):
                    raise EvidenceError(
                        "accepted prefix affinity differs from lane plan"
                    )
        return {
            "repetitions": repetitions,
            "rank_accounting": [
                record for record in parsed["rank_accounting"]
                if record["repetition"] < repetitions
            ],
            "pairs": parsed["pairs"][:repetitions],
            "timeline_bounds": parsed["timeline_bounds"][:repetitions],
        }
    except EvidenceError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise EvidenceError("accepted prefix full validation failed") from error


def _classify_lane_process_failure(
    result: dict[str, Any], *, ranks: int,
    expected_device_uuids: list[str] | None = None,
    expected_worker_cpu_masks: list[set[int]] | None = None,
    go_monotonic_ns: int | None = None,
    process_finished_monotonic_ns: int | None = None,
) -> tuple[EvidenceError | None, dict[str, Any] | None]:
    """Choose the worker/process primary before any success-only teardown gate."""

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    combined = stdout + "\n" + stderr
    forbidden = _unstructured_failure_token(stdout, stderr)
    has_rejected = REJECTED_PREFIX in combined
    rejected: dict[str, Any] | None = None
    rejected_error: BaseException | None = None
    if has_rejected:
        try:
            rejected = _parse_rejected_repetition_evidence(
                result.get("stdout", ""), result.get("stderr", ""),
                expected_ranks=ranks,
                expected_device_uuids=expected_device_uuids,
                expected_worker_cpu_masks=expected_worker_cpu_masks,
            )
            if rejected is not None:
                rejected["timeline_binding"] = _cross_bind_rejected_timing(
                    rejected,
                    go_monotonic_ns=go_monotonic_ns,
                    process_finished_monotonic_ns=(
                        result.get("process_finished_monotonic_ns")
                        if process_finished_monotonic_ns is None
                        else process_finished_monotonic_ns
                    ),
                )
        except BaseException as error:
            rejected_error = error
            rejected = None

    primary: EvidenceError | None = None
    if result.get("monitoring_error_caused_termination"):
        if result.get("callback_error"):
            primary = EvidenceError(
                f"rank-{ranks} telemetry callback failed and terminated the live "
                f"process: {result['callback_error']}"
            )
        elif result.get("runtime_observer_error"):
            primary = EvidenceError(
                f"rank-{ranks} runtime observer failed and terminated the live "
                f"process: {result['runtime_observer_error']}"
            )
        else:
            primary = EvidenceError(
                f"rank-{ranks} monitoring failure terminated the live process"
            )
    elif result.get("timed_out") or result.get("returncode") in (124, 137):
        primary = EvidenceError(f"rank-{ranks} LDOS benchmark timed out")
    elif forbidden is not None or has_rejected:
        details = []
        if result.get("returncode") != 0:
            details.append(f"exit status {result.get('returncode')}")
        if forbidden is not None:
            details.append(f"forbidden {forbidden.group(0)!r}")
        if has_rejected:
            details.append("rejected fixed repetition")
        primary = EvidenceError(
            f"rank-{ranks} LDOS worker primary failure: " + ", ".join(details)
        )
    elif result.get("returncode") != 0:
        primary = EvidenceError(
            f"rank-{ranks} LDOS worker primary failure: "
            f"exit status {result.get('returncode')}"
        )
    elif result.get("callback_error"):
        primary = EvidenceError(
            f"rank-{ranks} telemetry callback failed: {result['callback_error']}"
        )
    elif result.get("runtime_observer_error"):
        primary = EvidenceError(
            f"rank-{ranks} runtime observer failed: "
            f"{result['runtime_observer_error']}"
        )
    elif result.get("containment_error"):
        primary = EvidenceError(
            f"rank-{ranks} process containment failed: "
            f"{result['containment_error']}"
        )
    if primary is not None and result.get("returncode") != 0:
        for label, key in (
            ("telemetry callback", "callback_error"),
            ("runtime observer", "runtime_observer_error"),
            ("containment", "containment_error"),
        ):
            if result.get(key):
                primary.add_note(f"{label} diagnostic: {result[key]}")
    if primary is not None and rejected_error is not None:
        primary.add_note(
            "rejected evidence validation: "
            f"{type(rejected_error).__name__}: {rejected_error}"
        )
    failure_evidence = None if rejected is None else {
        "schema_version": 1,
        "focused_pass": False,
        "rejected_repetition": rejected,
    }
    return primary, failure_evidence


def _classify_lane_process_execution_error(
    error: ProcessExecutionError, *, ranks: int,
    expected_device_uuids: list[str] | None = None,
    expected_worker_cpu_masks: list[set[int]] | None = None,
    go_monotonic_ns: int | None = None,
    process_finished_monotonic_ns: int | None = None,
) -> tuple[EvidenceError | None, dict[str, Any] | None]:
    """Recover a worker/observer primary from a post-spawn control failure."""

    worker_primary, failure_evidence = _classify_lane_process_failure(
        error.result,
        ranks=ranks,
        expected_device_uuids=expected_device_uuids,
        expected_worker_cpu_masks=expected_worker_cpu_masks,
        go_monotonic_ns=go_monotonic_ns,
        process_finished_monotonic_ns=process_finished_monotonic_ns,
    )
    explicit_scientific_fail = any(
        line.startswith("FAIL:")
        for line in (
            *error.result.get("stdout", "").splitlines(),
            *error.result.get("stderr", "").splitlines(),
        )
    )
    independent_worker_evidence = bool(
        failure_evidence is not None or explicit_scientific_fail
    )
    if (
        error.result.get("process_control_error_caused_termination")
        and not independent_worker_evidence
    ):
        primary = EvidenceError(
            f"rank-{ranks} process-control failure terminated the live process: "
            f"{type(error.original).__name__}: {error.original}"
        )
        if worker_primary is not None:
            primary.add_note(f"captured worker-result diagnostic: {worker_primary}")
            for note in getattr(worker_primary, "__notes__", ()):
                primary.add_note(f"worker-result note: {note}")
        return primary, failure_evidence
    primary = worker_primary
    if primary is not None:
        primary.add_note(
            "process-control diagnostic: "
            f"{type(error.original).__name__}: {error.original}"
        )
    return primary, failure_evidence


def parse_device_inventory(text: str) -> list[dict[str, Any]]:
    """Parse the exact nvidia-smi inventory used to resolve UUID selection."""

    rows: list[dict[str, Any]] = []
    try:
        parsed = list(csv.reader(text.splitlines(), skipinitialspace=True))
    except csv.Error as exc:
        raise EvidenceError(f"invalid nvidia-smi CSV: {exc}") from exc
    for row_number, row in enumerate(parsed, 1):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 5:
            raise EvidenceError(
                f"nvidia-smi row {row_number} does not contain five columns"
            )
        index_text, uuid_text, name_text, bus_text, compute_text = (
            value.strip() for value in row
        )
        index = _parse_exact_int(index_text, f"device row {row_number} index")
        if not UUID_PATTERN.fullmatch(uuid_text):
            raise EvidenceError(f"device row {row_number} has an invalid physical UUID")
        if not name_text or not bus_text:
            raise EvidenceError(f"device row {row_number} has incomplete identity")
        compute_match = re.fullmatch(r"([0-9]+)\.([0-9]+)", compute_text)
        if not compute_match:
            raise EvidenceError(f"device row {row_number} has invalid compute capability")
        major, minor = (int(value) for value in compute_match.groups())
        if major < 6:
            raise EvidenceError(f"device row {row_number} is below the CUDA support floor")
        rows.append(
            {
                "index": index,
                "uuid": uuid_text,
                "name": name_text,
                "pci_bus_id": bus_text,
                "compute_major": major,
                "compute_minor": minor,
            }
        )
    if len(rows) < 2:
        raise EvidenceError("two compatible physical NVIDIA GPUs are required")
    if len({row["index"] for row in rows}) != len(rows):
        raise EvidenceError("nvidia-smi inventory contains duplicate indices")
    if len({row["uuid"] for row in rows}) != len(rows):
        raise EvidenceError("nvidia-smi inventory contains duplicate UUIDs")
    return sorted(rows, key=lambda row: row["index"])


def _root_owned_immutable_regular(
    info: os.stat_result, root_uid: int = 0
) -> bool:
    return bool(
        stat.S_ISREG(info.st_mode)
        and info.st_uid == root_uid
        and not info.st_mode & 0o022
    )


def _system_runtime_tcb(path: pathlib.Path) -> dict[str, Any] | None:
    """Bind immutable root-owned host runtime code as an explicit OS TCB."""

    handle = StableFile.open("root-owned system runtime TCB", path)
    try:
        return _system_runtime_tcb_handle(handle)
    finally:
        handle.close()


def _system_runtime_tcb_handle(handle: "StableFile") -> dict[str, Any] | None:
    """Validate the immutable system-runtime TCB on the selected open inode."""

    resolved = handle.path.resolve(strict=True)
    system_owner_uid = pathlib.Path("/").stat().st_uid
    allowed_roots = tuple(
        candidate.resolve(strict=True)
        for candidate in SYSTEM_RUNTIME_TCB_ROOTS
        if candidate.exists()
    )
    if not any(_path_is_within(resolved, root) for root in allowed_roots):
        return None
    opened = os.fstat(handle.descriptor)
    if (
        _fingerprint(opened) != handle.fingerprint
        or not _root_owned_immutable_regular(opened, system_owner_uid)
    ):
        return None
    current = resolved.parent
    while True:
        parent_info = current.stat()
        if parent_info.st_uid != system_owner_uid or parent_info.st_mode & 0o022:
            return None
        if current == current.parent:
            break
        current = current.parent
    verified = handle.verify()
    return {
        "source": "immutable_system_runtime_tcb",
        "system_owner_uid": system_owner_uid,
        "sha256": verified["sha256"],
        "size_bytes": verified["size_bytes"],
    }


def _trusted_system_tool_tcb(path: pathlib.Path) -> dict[str, Any] | None:
    """Validate a root-owned immutable executable below bounded system roots."""

    handle = StableFile.open("root-owned system tool TCB", path)
    try:
        return _trusted_system_tool_tcb_handle(handle)
    finally:
        handle.close()


def _trusted_system_tool_tcb_handle(handle: "StableFile") -> dict[str, Any] | None:
    """Validate an executable system-tool TCB on the selected open inode."""

    resolved = handle.path.resolve(strict=True)
    system_owner_uid = pathlib.Path("/").stat().st_uid
    allowed_roots = tuple(
        candidate.resolve(strict=True)
        for candidate in (
            pathlib.Path("/usr/bin"), pathlib.Path("/usr/local/bin"),
            pathlib.Path("/usr/lib"), pathlib.Path("/lib"),
            pathlib.Path("/usr/lib64"), pathlib.Path("/lib64"),
        )
        if candidate.exists()
    )
    if not any(_path_is_within(resolved, root) for root in allowed_roots):
        return None
    opened = os.fstat(handle.descriptor)
    if (
        _fingerprint(opened) != handle.fingerprint
        or not _root_owned_immutable_regular(opened, system_owner_uid)
        or not opened.st_mode & 0o111
    ):
        return None
    current = resolved.parent
    while True:
        parent_info = current.stat()
        if parent_info.st_uid != system_owner_uid or parent_info.st_mode & 0o022:
            return None
        if current == current.parent:
            break
        current = current.parent
    verified = handle.verify()
    return {
        "source": "immutable_system_tool_tcb",
        "system_owner_uid": system_owner_uid,
        "sha256": verified["sha256"],
        "size_bytes": verified["size_bytes"],
    }


def _matches_runtime_soname(path: pathlib.Path, family: str) -> bool:
    name = path.name.lower()
    patterns = {
        "libmeep.so": r"libmeep\.so(?:\..+)?",
        "libmpi": r"libmpi\.so(?:\..+)?",
        "libpmix": r"libpmix\.so(?:\..+)?",
        "libcuda.so": r"libcuda\.so(?:\..+)?",
        "libcudart.so": r"libcudart\.so(?:\..+)?",
    }
    pattern = patterns.get(family)
    return bool(pattern and re.fullmatch(pattern, name))


def verify_runtime_attestation(
    parsed: dict[str, Any],
    *,
    repo: pathlib.Path,
    receipt: dict[str, Any],
    archived: dict[str, "StableFile"],
) -> list[dict[str, Any]]:
    """Bind every worker mapping to its load-time identity and trusted bytes."""

    manifest = receipt.get("manifests", {}).get("installed_environment")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise EvidenceError("receipt has no installed-environment manifest")
    root_value = manifest.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise EvidenceError("installed-environment manifest root is invalid")
    manifest_root = pathlib.Path(root_value)
    manifest_root = (
        manifest_root if manifest_root.is_absolute() else repo / manifest_root
    ).resolve(strict=True)
    trusted: dict[pathlib.Path, dict[str, Any]] = {}
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise EvidenceError("installed-environment manifest entry is invalid")
        relative = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if (
            not isinstance(relative, str)
            or pathlib.PurePosixPath(relative).is_absolute()
            or ".." in pathlib.PurePosixPath(relative).parts
            or not isinstance(digest, str)
            or not SHA256_PATTERN.fullmatch(digest)
            or not _is_plain_int(size)
            or size < 0
        ):
            raise EvidenceError("installed-environment manifest entry is malformed")
        trusted[(manifest_root / relative).resolve(strict=True)] = {
            "source": "installed_environment",
            "sha256": digest,
            "size_bytes": size,
        }

    try:
        closure = json.loads(
            pathlib.Path(archived["runtime_closure"].proc_path).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError("archived runtime dependency closure is invalid") from exc
    groups = closure.get("groups") if isinstance(closure, dict) else None
    if not isinstance(groups, dict):
        raise EvidenceError("runtime dependency closure groups are invalid")
    for group_name, items in groups.items():
        if not isinstance(group_name, str) or not isinstance(items, list):
            raise EvidenceError("runtime dependency closure group is invalid")
        for item in items:
            if not isinstance(item, dict):
                raise EvidenceError("runtime dependency closure entry is invalid")
            path_value = item.get("path")
            digest = item.get("sha256")
            size = item.get("size_bytes")
            if (
                not isinstance(path_value, str)
                or not pathlib.Path(path_value).is_absolute()
                or not isinstance(digest, str)
                or not SHA256_PATTERN.fullmatch(digest)
                or not _is_plain_int(size)
                or size <= 0
            ):
                raise EvidenceError("runtime dependency closure entry is malformed")
            trusted[pathlib.Path(path_value).resolve(strict=True)] = {
                "source": f"runtime_dependency_closure:{group_name}",
                "sha256": digest,
                "size_bytes": size,
            }
    for label in ("gpu_step_db", "libmeep", "libmeep_soname", "loader"):
        handle = archived[label]
        trusted[handle.path.resolve(strict=True)] = {
            "source": f"archived:{label}",
            "sha256": handle.sha256,
            "size_bytes": handle.size_bytes,
        }

    verified: list[dict[str, Any]] = []
    expected_libmeep = {
        archived[label].path.resolve(strict=True)
        for label in ("libmeep", "libmeep_soname")
    }
    required_families = ("libmeep.so", "libmpi", "libcuda.so", "libcudart.so")
    for runtime in parsed["runtime_attestation"]:
        rank = runtime["rank"]
        if pathlib.Path(runtime["libmeep"]).resolve(strict=True) not in expected_libmeep:
            raise EvidenceError("worker did not map the archived libmeep snapshot")
        rank_records: list[dict[str, Any]] = []
        paths: list[pathlib.Path] = []
        for mapping in runtime["mappings"]:
            path = pathlib.Path(mapping["path"]).resolve(strict=True)
            paths.append(path)
            handle = StableFile.open(f"rank {rank} runtime mapping", path)
            try:
                info = os.fstat(handle.descriptor)
                observed = {
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "size": info.st_size,
                    "mtime_ns": info.st_mtime_ns,
                    "ctime_ns": info.st_ctime_ns,
                }
                expected_identity = {
                    name: mapping[name]
                    for name in (
                        "device", "inode", "size", "mtime_ns", "ctime_ns"
                    )
                }
                if observed != expected_identity:
                    raise EvidenceError(
                        f"rank {rank} runtime mapping changed after it was loaded: {path}"
                    )
                trust = trusted.get(path)
                if trust is None:
                    trust = _system_runtime_tcb_handle(handle)
                    if trust is None:
                        raise EvidenceError(
                            f"rank {rank} mapped an unbound runtime file: {path}"
                        )
                    trusted[path] = trust
                if handle.sha256 != trust["sha256"] or handle.size_bytes != trust["size_bytes"]:
                    raise EvidenceError(f"rank {rank} runtime mapping hash differs from its bound closure: {path}")
                rank_records.append({**mapping, **trust, "sha256": handle.sha256})
            finally:
                handle.close()
        for family in required_families:
            if not any(_matches_runtime_soname(path, family) for path in paths):
                raise EvidenceError(f"rank {rank} runtime closure lacks {family}")
        if archived["loader"].path.resolve(strict=True) not in paths:
            raise EvidenceError(f"rank {rank} did not use the archived ELF loader")
        special_records: list[dict[str, Any]] = []
        for mapping in runtime["special_mappings"]:
            path = pathlib.Path(mapping["path"])
            if mapping["deleted"]:
                if mapping["kind"] != "ephemeral-data":
                    raise EvidenceError("only ephemeral runtime data may be deleted")
                special_records.append(dict(mapping))
                continue
            info = path.stat()
            if (
                info.st_dev != mapping["device"]
                or info.st_ino != mapping["inode"]
                or info.st_mode != mapping["mode"]
                or info.st_rdev != mapping["rdev"]
            ):
                raise EvidenceError(f"rank {rank} special mapping identity changed: {path}")
            if mapping["kind"] == "nvidia-character-device" and not stat.S_ISCHR(
                info.st_mode
            ):
                raise EvidenceError("NVIDIA runtime mapping is not a character device")
            special_records.append(dict(mapping))
        verified.append(
            {"rank": rank, "mappings": rank_records, "special_mappings": special_records}
        )
    if len(verified) == 2:
        first = {(item["path"], item["sha256"]) for item in verified[0]["mappings"]}
        second = {(item["path"], item["sha256"]) for item in verified[1]["mappings"]}
        if first != second:
            raise EvidenceError("rank-two workers did not map the same bound runtime closure")
    return verified


def open_runtime_closure_handles(closure: "StableFile") -> dict[str, "StableFile"]:
    try:
        value = json.loads(pathlib.Path(closure.proc_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError("cannot load archived runtime dependency closure") from exc
    groups = value.get("groups") if isinstance(value, dict) else None
    if not isinstance(groups, dict):
        raise EvidenceError("runtime dependency closure has no groups")
    handles: dict[str, StableFile] = {}
    try:
        for group, records in groups.items():
            if not isinstance(records, list):
                raise EvidenceError(f"runtime closure group {group!r} is invalid")
            for index, record in enumerate(records):
                if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                    raise EvidenceError("runtime closure record path is invalid")
                path = pathlib.Path(record["path"]).resolve(strict=True)
                key = str(path)
                if key in handles:
                    continue
                handle = StableFile.open(f"runtime closure {group}[{index}]", path)
                if handle.sha256 != record.get("sha256") or handle.size_bytes != record.get("size_bytes"):
                    handle.close()
                    raise EvidenceError(f"runtime closure file differs from receipt: {path}")
                handles[key] = handle
        return handles
    except BaseException:
        for handle in handles.values():
            handle.close()
        raise


def verify_mpi_runtime_observations(
    result: dict[str, Any],
    *,
    repo: pathlib.Path,
    receipt: dict[str, Any],
    archived: dict[str, "StableFile"],
    expected_ranks: int,
    expected_library_path: str,
    expected_timeout_seconds: int,
    expected_worker_cpu_masks: list[set[int]],
) -> dict[str, Any]:
    observations = result.get("runtime_observations")
    if not isinstance(observations, list) or not observations:
        raise EvidenceError("MPI runtime process observation is empty")
    attempts = result.get("runtime_observation_attempts")
    if not isinstance(attempts, list) or len(attempts) < 2:
        raise EvidenceError("MPI runtime observation-attempt timeline is incomplete")
    previous_start: int | None = None
    maximum_gap_ns = 0
    maximum_duration_ns = 0
    for attempt in attempts:
        if (
            not isinstance(attempt, dict)
            or set(attempt)
            != {"start_monotonic_ns", "duration_ns", "error", "process_count"}
            or not _is_plain_int(attempt["start_monotonic_ns"])
            or not _is_plain_int(attempt["duration_ns"])
            or not _is_plain_int(attempt["process_count"])
            or attempt["duration_ns"] < 0
            or attempt["process_count"] < 0
            or attempt["error"] is not None
        ):
            raise EvidenceError("MPI runtime observation-attempt timeline is invalid")
        start_ns = attempt["start_monotonic_ns"]
        if previous_start is not None:
            if start_ns <= previous_start:
                raise EvidenceError("MPI runtime observer timestamps are not monotonic")
            maximum_gap_ns = max(maximum_gap_ns, start_ns - previous_start)
        previous_start = start_ns
        maximum_duration_ns = max(maximum_duration_ns, attempt["duration_ns"])
    process_started_ns = result.get("process_started_monotonic_ns")
    process_finished_ns = result.get("process_finished_monotonic_ns")
    if (
        not _is_plain_int(process_started_ns)
        or not _is_plain_int(process_finished_ns)
        or process_started_ns > attempts[0]["start_monotonic_ns"]
        or process_finished_ns
        < attempts[-1]["start_monotonic_ns"] + attempts[-1]["duration_ns"]
    ):
        raise EvidenceError("MPI runtime observer process boundary timestamps are invalid")
    maximum_allowed_gap_ns = int(
        PROFILE["runtime_observer_max_gap_seconds"] * 1_000_000_000
    )
    start_edge_ns = attempts[0]["start_monotonic_ns"] - process_started_ns
    finish_edge_ns = process_finished_ns - (
        attempts[-1]["start_monotonic_ns"] + attempts[-1]["duration_ns"]
    )
    maximum_gap_ns = max(maximum_gap_ns, start_edge_ns, finish_edge_ns)
    if maximum_gap_ns > maximum_allowed_gap_ns:
        raise EvidenceError("MPI runtime observer sampling gap exceeded the fixed gate")
    if maximum_duration_ns > int(
        PROFILE["runtime_observer_max_duration_seconds"] * 1_000_000_000
    ):
        raise EvidenceError("MPI runtime observer duration exceeded the fixed gate")
    trusted: dict[pathlib.Path, tuple[str, int, str]] = {}
    manifest = receipt.get("manifests", {}).get("installed_environment")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise EvidenceError("receipt installed-environment manifest is unavailable")
    root_value = manifest.get("root")
    if not isinstance(root_value, str):
        raise EvidenceError("receipt installed-environment root is invalid")
    manifest_root = pathlib.Path(root_value)
    manifest_root = (manifest_root if manifest_root.is_absolute() else repo / manifest_root).resolve(strict=True)
    for record in manifest["files"]:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise EvidenceError("installed-environment record is invalid")
        path = (manifest_root / record["path"]).resolve(strict=True)
        trusted[path] = (record["sha256"], record["size_bytes"], "installed_environment")
    closure = json.loads(pathlib.Path(archived["runtime_closure"].proc_path).read_text(encoding="utf-8"))
    for group, records in closure.get("groups", {}).items():
        for record in records:
            path = pathlib.Path(record["path"]).resolve(strict=True)
            trusted[path] = (record["sha256"], record["size_bytes"], f"runtime_closure:{group}")
    for label in (
        "timeout", "mpiexec", "loader", "gpu_step_db", "libmeep",
        "libmeep_soname",
    ):
        handle = archived[label]
        trusted[handle.path.resolve(strict=True)] = (
            handle.sha256,
            handle.size_bytes,
            f"archived:{label}",
        )
    if not _is_plain_int(expected_ranks) or expected_ranks not in (1, 2):
        raise EvidenceError("MPI runtime expected-rank cardinality is invalid")
    expected_command = result.get("command")
    root_pid = result.get("root_pid")
    if not isinstance(expected_command, list) or not all(
        isinstance(item, str) for item in expected_command
    ):
        raise EvidenceError("MPI runtime persisted command is invalid")
    authoritative_command = _mpi_command(
        archived["timeout"],
        archived["mpiexec"],
        archived["loader"],
        archived["gpu_step_db"],
        expected_ranks,
        expected_timeout_seconds,
        expected_library_path,
        expected_worker_cpu_masks,
    )
    if expected_command != authoritative_command:
        raise EvidenceError("MPI runtime command differs from fixed held-FD grammar")
    if not _is_plain_int(root_pid) or root_pid <= 0:
        raise EvidenceError("MPI runtime root PID is absent")

    verified_mappings: list[dict[str, Any]] = []
    verified_executables: list[dict[str, Any]] = []
    epoch_mappings: list[dict[pathlib.Path, dict[str, Any]]] = []
    epoch_executables: list[pathlib.Path] = []
    process_epochs: dict[tuple[int, int], list[int]] = {}
    parent_by_process: dict[tuple[int, int], int] = {}
    identities_by_pid: dict[int, set[tuple[int, int]]] = {}
    for observation in observations:
        if (
            not _is_plain_int(observation.get("pid"))
            or not _is_plain_int(observation.get("parent_pid"))
            or not _is_plain_int(observation.get("start_time_ticks"))
            or not SHA256_PATTERN.fullmatch(
                str(observation.get("mapping_epoch_sha256", ""))
            )
            or not isinstance(observation.get("command_line"), list)
            or not all(isinstance(item, str) for item in observation["command_line"])
            or not isinstance(observation.get("executable"), dict)
        ):
            raise EvidenceError("runtime process observation schema is incomplete")
        executable_record = observation["executable"]
        executable_path = pathlib.Path(executable_record.get("path", "")).resolve(strict=True)
        executable_fingerprint = executable_record.get("fingerprint", {})
        executable_handle = StableFile.open(
            "post-lane runtime executable", executable_path
        )
        try:
            executable_trust = trusted.get(executable_path)
            if executable_trust is None:
                system_trust = _system_runtime_tcb_handle(executable_handle)
                if system_trust is None:
                    raise EvidenceError(
                        f"runtime process executed an unbound file: {executable_path}"
                    )
                executable_trust = (
                    system_trust["sha256"],
                    system_trust["size_bytes"],
                    system_trust["source"],
                )
                trusted[executable_path] = executable_trust
            if (
                executable_handle.sha256 != executable_trust[0]
                or executable_handle.size_bytes != executable_trust[1]
                or executable_record.get("size_bytes") != executable_trust[1]
                or executable_fingerprint.get("device")
                != executable_handle.fingerprint[0]
                or executable_fingerprint.get("inode")
                != executable_handle.fingerprint[1]
                or executable_fingerprint.get("mtime_ns")
                != executable_handle.fingerprint[4]
                or executable_fingerprint.get("ctime_ns")
                != executable_handle.fingerprint[5]
            ):
                raise EvidenceError(
                    f"runtime executable identity differs from trusted bytes: {executable_path}"
                )
            verified_executables.append(
                {
                    **executable_handle.record(),
                    "trust_source": executable_trust[2],
                }
            )
        finally:
            executable_handle.close()
        epoch_executables.append(executable_path)
        by_path: dict[pathlib.Path, dict[str, Any]] = {}
        for mapping in observation.get("mappings", []):
            path = pathlib.Path(mapping["path"]).resolve(strict=True)
            mapping_handle = StableFile.open("post-lane runtime mapping", path)
            try:
                expected = trusted.get(path)
                if expected is None:
                    system_trust = _system_runtime_tcb_handle(mapping_handle)
                    if system_trust is None:
                        raise EvidenceError(
                            f"observed MPI runtime mapped an unbound file: {path}"
                        )
                    expected = (
                        system_trust["sha256"],
                        system_trust["size_bytes"],
                        system_trust["source"],
                    )
                    trusted[path] = expected
                digest, size, source = expected
                info = os.fstat(mapping_handle.descriptor)
                if any(
                    mapping[name] != expected_value
                    for name, expected_value in (
                        ("device", info.st_dev),
                        ("inode", info.st_ino),
                        ("size_bytes", info.st_size),
                        ("mtime_ns", info.st_mtime_ns),
                        ("ctime_ns", info.st_ctime_ns),
                    )
                ):
                    raise EvidenceError(
                        f"observed MPI runtime mapping changed after execution: {path}"
                    )
                if mapping_handle.sha256 != digest or mapping_handle.size_bytes != size:
                    raise EvidenceError(
                        f"observed MPI runtime bytes differ from trusted identity: {path}"
                    )
                bound = {
                    **mapping,
                    "sha256": mapping_handle.sha256,
                    "trust_source": source,
                }
            finally:
                mapping_handle.close()
            verified_mappings.append(bound)
            by_path[path] = bound
        epoch_mappings.append(by_path)

        identity = (observation["pid"], observation["start_time_ticks"])
        index = len(epoch_mappings) - 1
        process_epochs.setdefault(identity, []).append(index)
        identities_by_pid.setdefault(identity[0], set()).add(identity)
        previous_parent = parent_by_process.setdefault(identity, observation["parent_pid"])
        if previous_parent != observation["parent_pid"]:
            raise EvidenceError("runtime process parent changed across observation epochs")

    if any(len(identities) != 1 for identities in identities_by_pid.values()):
        raise EvidenceError("runtime process PID reuse makes the execution tree ambiguous")
    identity_by_pid = {
        pid: next(iter(identities)) for pid, identities in identities_by_pid.items()
    }

    def has_main_mapping(mapping: dict[str, Any] | None) -> bool:
        return bool(
            mapping
            and isinstance(mapping.get("segments"), list)
            and any(
                segment.get("offset") == 0
                and isinstance(segment.get("permissions"), str)
                and "r" in segment["permissions"]
                for segment in mapping["segments"]
            )
        )

    timeout_path = archived["timeout"].path.resolve(strict=True)
    loader_path = archived["loader"].path.resolve(strict=True)
    mpiexec_path = archived["mpiexec"].path.resolve(strict=True)
    worker_path = archived["gpu_step_db"].path.resolve(strict=True)

    def direct_loader_target(arguments: list[str]) -> str | None:
        prefix = [
            archived["loader"].proc_path,
            "--inhibit-rpath",
            "",
            "--library-path",
            expected_library_path,
        ]
        if len(arguments) < len(prefix) + 1 or arguments[: len(prefix)] != prefix:
            return None
        return arguments[len(prefix)]

    def path_family(mappings: dict[pathlib.Path, dict[str, Any]], family: str) -> bool:
        return any(_matches_runtime_soname(path, family) for path in mappings)

    timeout_identities: set[tuple[int, int]] = set()
    launcher_identities: set[tuple[int, int]] = set()
    prte_identities: set[tuple[int, int]] = set()
    worker_identities: set[tuple[int, int]] = set()
    complete_timeout_epochs: list[int] = []
    complete_launcher_epochs: list[int] = []
    complete_prte_epochs: list[int] = []
    complete_worker_epochs: list[int] = []
    for identity, indices in process_epochs.items():
        for index in indices:
            observation = observations[index]
            arguments = observation["command_line"]
            mappings = epoch_mappings[index]
            executable_path = epoch_executables[index]
            target = direct_loader_target(arguments)
            if identity[0] == root_pid and executable_path == timeout_path:
                timeout_identities.add(identity)
                if arguments == expected_command and has_main_mapping(
                    mappings.get(timeout_path)
                ):
                    complete_timeout_epochs.append(index)
                continue
            if executable_path == loader_path and target == archived["mpiexec"].proc_path:
                launcher_identities.add(identity)
                if has_main_mapping(mappings.get(loader_path)) and has_main_mapping(
                    mappings.get(mpiexec_path)
                ):
                    complete_launcher_epochs.append(index)
                continue
            if executable_path == loader_path and target == archived["gpu_step_db"].proc_path:
                worker_identities.add(identity)
                if (
                    arguments
                    == [
                        archived["loader"].proc_path,
                        "--inhibit-rpath",
                        "",
                        "--library-path",
                        expected_library_path,
                        archived["gpu_step_db"].proc_path,
                    ]
                    and has_main_mapping(mappings.get(loader_path))
                    and has_main_mapping(mappings.get(worker_path))
                    and path_family(mappings, "libmpi")
                ):
                    complete_worker_epochs.append(index)
                continue
            if executable_path.name in {"prte", "prted"}:
                prte_identities.add(identity)
                if (
                    has_main_mapping(mappings.get(executable_path))
                    and path_family(mappings, "libpmix")
                ):
                    complete_prte_epochs.append(index)

    if len(timeout_identities) != 1 or not complete_timeout_epochs:
        raise EvidenceError("runtime evidence lacks one exact archived timeout-root epoch")
    if len(launcher_identities) > 1:
        raise EvidenceError("runtime evidence contains ambiguous mpiexec launcher identities")
    if len(worker_identities) != expected_ranks:
        raise EvidenceError("runtime worker process cardinality differs from MPI ranks")
    if not prte_identities:
        raise EvidenceError("runtime process evidence did not execute PRTE")
    if {
        (observations[index]["pid"], observations[index]["start_time_ticks"])
        for index in complete_worker_epochs
    } != worker_identities:
        raise EvidenceError(
            "worker proof split argv, loader/main mappings, or MPI across epochs"
        )
    if {
        (observations[index]["pid"], observations[index]["start_time_ticks"])
        for index in complete_prte_epochs
    } != prte_identities:
        raise EvidenceError(
            "PRTE proof split executable/main mappings or MPI/PMIx across epochs"
        )

    timeout_identity = next(iter(timeout_identities))
    launcher_identity = (
        next(iter(launcher_identities)) if launcher_identities else None
    )

    def ancestors(identity: tuple[int, int]) -> list[tuple[int, int]]:
        result_chain: list[tuple[int, int]] = []
        visited = {identity}
        current = identity
        while True:
            parent_pid = parent_by_process[current]
            parent = identity_by_pid.get(parent_pid)
            if parent is None:
                return result_chain
            if parent in visited:
                raise EvidenceError("runtime process execution tree contains a cycle")
            visited.add(parent)
            result_chain.append(parent)
            current = parent

    if launcher_identity is not None:
        launcher_chain = ancestors(launcher_identity)
        if not launcher_chain or launcher_chain[0] != timeout_identity:
            raise EvidenceError("mpiexec launcher is not a direct child of timeout root")
    for identity in prte_identities:
        chain = ancestors(identity)
        if timeout_identity not in chain:
            raise EvidenceError("PRTE parent chain does not terminate at timeout root")
        if (
            launcher_identity is not None
            and identity != launcher_identity
            and launcher_identity not in chain
        ):
            raise EvidenceError("PRTE parent chain bypasses the observed mpiexec launcher")
    for identity in worker_identities:
        chain = ancestors(identity)
        if timeout_identity not in chain or not any(
            ancestor in prte_identities for ancestor in chain
        ):
            raise EvidenceError("worker parent chain does not pass through PRTE/mpiexec")
    def deduplicated(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
        for record in records:
            fingerprint = record.get("fingerprint", record)
            key = (
                record["path"],
                fingerprint.get("device"),
                fingerprint.get("inode"),
                record["sha256"],
                record["trust_source"],
            )
            by_identity[key] = record
        return [by_identity[key] for key in sorted(by_identity)]

    return {
        "schema_version": 1,
        "process_count": len(process_epochs),
        "verified_mapping_count": len(verified_mappings),
        "verified_executable_identities": deduplicated(verified_executables),
        "verified_mapping_identities": deduplicated(verified_mappings),
        "observations": observations,
        "observation_attempts": attempts,
        "process_started_monotonic_ns": process_started_ns,
        "process_finished_monotonic_ns": process_finished_ns,
        "maximum_observer_gap_seconds": maximum_gap_ns / 1_000_000_000,
        "maximum_observer_duration_seconds": maximum_duration_ns / 1_000_000_000,
        "actual_execution_proofs": {
            "bound_launcher_invocation": True,
            "mpiexec_execution_inferred_from_downstream_prte_worker": True,
            "archived_mpiexec_same_epoch_observed": bool(
                launcher_identities and complete_launcher_epochs
            ),
            "archived_worker_mapped": True,
            "archived_loader_mapped": True,
            "prte_executable_mapped": True,
            "pmix_component_mapped": True,
            "mpi_component_mapped": True,
            "coherent_timeout_mpiexec_prte_worker_tree": True,
            "worker_cardinality_matches_ranks": True,
        },
    }


def select_devices(
    inventory: list[dict[str, Any]], requested_uuids: str | None
) -> list[dict[str, Any]]:
    if requested_uuids is None:
        raise EvidenceError("explicit --device-uuids are mandatory; fallback is forbidden")
    else:
        tokens = [token.strip() for token in requested_uuids.split(",")]
        if len(tokens) != 2 or any(not UUID_PATTERN.fullmatch(token) for token in tokens):
            raise EvidenceError("--device-uuids must contain exactly two physical GPU UUIDs")
        if tokens[0] == tokens[1]:
            raise EvidenceError("--device-uuids must name two distinct physical GPUs")
        by_uuid = {row["uuid"]: row for row in inventory}
        try:
            selected = [by_uuid[token] for token in tokens]
        except KeyError as exc:
            raise EvidenceError(f"requested GPU UUID is not visible: {exc.args[0]}") from exc
    if len({row["uuid"] for row in selected}) != 2:
        raise EvidenceError("rank-two LDOS lane requires two distinct GPU UUIDs")
    return [dict(row) for row in selected]


def require_quiescent_devices(
    telemetry: dict[str, Any], selected: list[dict[str, Any]], label: str
) -> None:
    selected_uuids = {item["uuid"] for item in selected}
    competing = [
        item
        for item in telemetry.get("compute_processes", [])
        if item.get("gpu_uuid") in selected_uuids
    ]
    if competing:
        raise EvidenceError(f"selected GPUs are not quiescent at {label}: compute processes exist")
    by_uuid = {item["uuid"]: item for item in telemetry["inventory"]}
    for uuid_value in selected_uuids:
        if by_uuid[uuid_value]["utilization_percent"] != 0.0:
            raise EvidenceError(f"selected GPU {uuid_value} is not idle at {label}")


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(descriptor, 1024 * 1024, offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def _nvml_elf_identity(descriptor: int, size_bytes: int) -> dict[str, Any]:
    """Parse bounded ELF metadata without loading or executing archive bytes."""

    if (
        not _is_plain_int(size_bytes)
        or not 64 <= size_bytes <= NVML_ELF_MAX_SIZE_BYTES
    ):
        raise EvidenceError("archived NVML ELF size is outside the bounded contract")

    parse_work_bytes = 0

    def charge_work(size: int, label: str) -> None:
        nonlocal parse_work_bytes
        if not _is_plain_int(size) or size < 0:
            raise EvidenceError(f"archived NVML ELF {label} work is invalid")
        parse_work_bytes += size
        if parse_work_bytes > NVML_ELF_MAX_PARSE_WORK_BYTES:
            raise EvidenceError("archived NVML ELF aggregate parse-work budget exceeded")

    def read_exact(offset: int, size: int, label: str) -> bytes:
        if (
            not _is_plain_int(offset)
            or not _is_plain_int(size)
            or offset < 0
            or size < 0
            or offset > size_bytes
            or size > size_bytes - offset
        ):
            raise EvidenceError(f"archived NVML ELF {label} range is invalid")
        charge_work(size, label)
        data = os.pread(descriptor, size, offset)
        if len(data) != size:
            raise EvidenceError(f"archived NVML ELF {label} is truncated")
        return data

    identity = read_exact(0, 16, "identity")
    if identity[:4] != b"\x7fELF" or identity[6] != 1:
        raise EvidenceError("archived NVML bytes are not ELF version 1")
    elf_class = identity[4]
    encoding = identity[5]
    if elf_class not in (1, 2) or encoding not in (1, 2):
        raise EvidenceError("archived NVML ELF class/encoding is unsupported")
    endian = "<" if encoding == 1 else ">"
    if elf_class == 2:
        header_format = endian + "HHIQQQIHHHHHH"
        program_format = endian + "IIQQQQQQ"
        section_format = endian + "IIQQQQIIQQ"
        symbol_format = endian + "IBBHQQ"
        dynamic_format = endian + "qQ"
        expected_header_size = 64
        expected_program_size = 56
        expected_section_size = 64
        expected_symbol_size = 24
    else:
        header_format = endian + "HHIIIIIHHHHHH"
        program_format = endian + "IIIIIIII"
        section_format = endian + "IIIIIIIIII"
        symbol_format = endian + "IIIBBH"
        dynamic_format = endian + "iI"
        expected_header_size = 52
        expected_program_size = 32
        expected_section_size = 40
        expected_symbol_size = 16
    header = struct.unpack(
        header_format,
        read_exact(16, struct.calcsize(header_format), "header"),
    )
    (
        elf_type, machine, version, entry, program_offset, section_offset,
        _flags, header_size, program_entry_size, program_count,
        section_entry_size, section_count, section_name_index,
    ) = header
    machine_names = {3: "x86", 40: "arm", 62: "x86_64", 183: "aarch64"}
    if (
        elf_type != 3
        or version != 1
        or machine not in machine_names
        or header_size != expected_header_size
        or program_offset < expected_header_size
        or program_entry_size != expected_program_size
        or not 1 <= program_count <= NVML_ELF_MAX_PROGRAM_HEADER_COUNT
        or section_entry_size != expected_section_size
        or not 1 <= section_count <= NVML_ELF_MAX_SECTION_COUNT
        or not 0 <= section_name_index < section_count
    ):
        raise EvidenceError("archived NVML ELF header contract is invalid")
    program_table_size = program_count * program_entry_size
    program_bytes = read_exact(
        program_offset, program_table_size, "program table"
    )
    word_max = (1 << (64 if elf_class == 2 else 32)) - 1
    programs: list[dict[str, int]] = []
    for index in range(program_count):
        values = struct.unpack_from(
            program_format, program_bytes, index * program_entry_size
        )
        if elf_class == 2:
            kind, flags, offset, virtual_address, _physical_address, file_size, memory_size, alignment = values
        else:
            kind, offset, virtual_address, _physical_address, file_size, memory_size, flags, alignment = values
        if (
            offset > size_bytes
            or file_size > size_bytes - offset
            or file_size > memory_size
            or virtual_address > word_max - memory_size
            or alignment not in (0, 1)
            and (
                alignment & (alignment - 1)
                or virtual_address % alignment != offset % alignment
            )
        ):
            raise EvidenceError(
                "archived NVML ELF program segment range/alignment is invalid"
            )
        programs.append({
            "type": kind,
            "flags": flags,
            "offset": offset,
            "virtual_address": virtual_address,
            "file_size": file_size,
            "memory_size": memory_size,
            "alignment": alignment,
        })
    load_programs = [program for program in programs if program["type"] == 1]
    dynamic_programs = [program for program in programs if program["type"] == 2]
    if (
        not load_programs
        or not any(
            program["file_size"] > 0 and program["flags"] & 1
            for program in load_programs
        )
        or len(dynamic_programs) != 1
    ):
        raise EvidenceError(
            "archived NVML ELF lacks its loadable/dynamic program image"
        )
    dynamic_program = dynamic_programs[0]
    if (
        dynamic_program["file_size"] == 0
        or dynamic_program["file_size"] % struct.calcsize(dynamic_format)
    ):
        raise EvidenceError("archived NVML ELF PT_DYNAMIC is invalid")

    def program_contains(
        container: dict[str, int], item: dict[str, int], *,
        require_file: bool, required_flags: int = 0,
    ) -> bool:
        memory_delta = item["virtual_address"] - container["virtual_address"]
        if (
            memory_delta < 0
            or item["memory_size"] > container["memory_size"] - memory_delta
            or required_flags & ~container["flags"]
        ):
            return False
        if not require_file:
            return True
        file_delta = item["offset"] - container["offset"]
        return bool(
            file_delta >= 0
            and item["file_size"] <= container["file_size"] - file_delta
            and file_delta == memory_delta
        )

    if not any(
        program_contains(
            load, dynamic_program, require_file=True,
            required_flags=dynamic_program["flags"],
        )
        for load in load_programs
    ):
        raise EvidenceError("archived NVML ELF PT_DYNAMIC is outside PT_LOAD")
    section_table_size = section_count * section_entry_size
    section_bytes = read_exact(
        section_offset, section_table_size, "section table"
    )
    sections: list[dict[str, int]] = []
    for index in range(section_count):
        values = struct.unpack_from(
            section_format, section_bytes, index * section_entry_size
        )
        if elf_class == 2:
            name_offset, section_type, section_flags, address, offset, size, link, _info, alignment, entry_size = values
        else:
            name_offset, section_type, section_flags, address, offset, size, link, _info, alignment, entry_size = values
        if (
            section_type != 8
            and (offset > size_bytes or size > size_bytes - offset)
        ) or (
            section_flags & 2 and address > word_max - size
        ) or (
            alignment not in (0, 1)
            and (
                alignment & (alignment - 1)
                or address % alignment
                or section_type != 8 and offset % alignment
            )
        ):
            raise EvidenceError("archived NVML ELF section range is invalid")
        sections.append({
            "name_offset": name_offset,
            "type": section_type,
            "flags": section_flags,
            "address": address,
            "offset": offset,
            "size": size,
            "link": link,
            "alignment": alignment,
            "entry_size": entry_size,
        })

    for section in sections:
        if not section["flags"] & 2 or section["size"] == 0:
            continue
        required_flags = 4
        if section["flags"] & 1:
            required_flags |= 2
        if section["flags"] & 4:
            required_flags |= 1
        section_program = {
            "virtual_address": section["address"],
            "memory_size": section["size"],
            "offset": section["offset"],
            "file_size": 0 if section["type"] == 8 else section["size"],
        }
        if not any(
            program_contains(
                load,
                section_program,
                require_file=section["type"] != 8,
                required_flags=required_flags,
            )
            for load in load_programs
        ):
            raise EvidenceError(
                "archived NVML ELF allocated section is outside compatible PT_LOAD"
            )

    def section_data(index: int, label: str, maximum_size: int) -> bytes:
        if not 0 <= index < len(sections):
            raise EvidenceError(f"archived NVML ELF {label} link is invalid")
        section = sections[index]
        if section["type"] == 8 or section["size"] > maximum_size:
            raise EvidenceError(f"archived NVML ELF {label} is oversized")
        return read_exact(section["offset"], section["size"], label)

    def c_string(table: bytes, offset: int, label: str) -> str:
        if not _is_plain_int(offset) or not 0 <= offset < len(table):
            raise EvidenceError(f"archived NVML ELF {label} offset is invalid")
        search_stop = min(
            len(table), offset + NVML_ELF_MAX_C_STRING_BYTES + 1
        )
        end = table.find(b"\0", offset, search_stop)
        charge_work(search_stop - offset, label)
        if end < 0 or end - offset > NVML_ELF_MAX_C_STRING_BYTES:
            raise EvidenceError(f"archived NVML ELF {label} is unterminated")
        try:
            return table[offset:end].decode("ascii")
        except UnicodeDecodeError as exc:
            raise EvidenceError(f"archived NVML ELF {label} is not ASCII") from exc

    section_names = section_data(
        section_name_index, "section-name table", 4 * 1024 * 1024
    )
    for section in sections:
        section["name"] = c_string(
            section_names, section["name_offset"], "section name"
        )

    dynamic_indices = [
        index for index, section in enumerate(sections) if section["type"] == 6
    ]
    symbol_indices = [
        index for index, section in enumerate(sections) if section["type"] == 11
    ]
    note_indices = [
        index for index, section in enumerate(sections) if section["type"] == 7
    ]
    if len(dynamic_indices) != 1 or len(symbol_indices) != 1:
        raise EvidenceError("archived NVML ELF dynamic metadata cardinality is invalid")
    dynamic_index = dynamic_indices[0]
    symbol_index = symbol_indices[0]
    dynamic = sections[dynamic_index]
    dynamic_entry_size = struct.calcsize(dynamic_format)
    if (
        dynamic["entry_size"] != dynamic_entry_size
        or dynamic["size"] % dynamic_entry_size
        or not 0 <= dynamic["link"] < len(sections)
    ):
        raise EvidenceError("archived NVML ELF dynamic table is invalid")
    dynamic_section_program = {
        "virtual_address": dynamic["address"],
        "memory_size": dynamic["size"],
        "offset": dynamic["offset"],
        "file_size": dynamic["size"],
    }
    if not program_contains(
        dynamic_program, dynamic_section_program, require_file=True
    ):
        raise EvidenceError(
            "archived NVML ELF SHT_DYNAMIC differs from PT_DYNAMIC"
        )
    if sections[dynamic["link"]]["type"] != 3:
        raise EvidenceError("archived NVML ELF dynamic string link is not a string table")
    symbols = sections[symbol_index]
    if (
        not 0 <= symbols["link"] < len(sections)
        or sections[symbols["link"]]["type"] != 3
    ):
        raise EvidenceError("archived NVML ELF symbol string link is not a string table")

    # The same string-table section may legitimately serve both dynamic tables.
    # Distinct relevant file-backed sections, fixed ELF metadata, and the section
    # table may not alias. This prevents a small file from causing repeated parse
    # work through thousands of overlapping section descriptions.
    relevant_indices = {
        section_name_index,
        dynamic_index,
        dynamic["link"],
        symbol_index,
        symbols["link"],
        *note_indices,
    }
    relevant_ranges: list[tuple[int, int, str]] = [
        (0, expected_header_size, "ELF header"),
        (
            program_offset,
            program_offset + program_table_size,
            "program table",
        ),
        (
            section_offset,
            section_offset + section_table_size,
            "section table",
        ),
    ]
    for index in relevant_indices:
        section = sections[index]
        if section["type"] == 8:
            raise EvidenceError("archived NVML ELF relevant section is not file-backed")
        if section["size"]:
            relevant_ranges.append(
                (
                    section["offset"],
                    section["offset"] + section["size"],
                    f"section {index}",
                )
            )
    relevant_ranges.sort()
    for left, right in zip(relevant_ranges, relevant_ranges[1:]):
        if right[0] < left[1]:
            raise EvidenceError(
                "archived NVML ELF relevant file-backed sections overlap"
            )

    dynamic_strings = section_data(
        dynamic["link"], "dynamic string table", 16 * 1024 * 1024
    )
    soname_offsets: list[int] = []
    if dynamic["size"] > 2 * 1024 * 1024:
        raise EvidenceError("archived NVML ELF dynamic table is oversized")
    dynamic_data = read_exact(dynamic["offset"], dynamic["size"], "dynamic table")
    dynamic_terminated = False
    for offset in range(0, len(dynamic_data), dynamic_entry_size):
        tag, value = struct.unpack_from(dynamic_format, dynamic_data, offset)
        if tag == 0:
            dynamic_terminated = True
            break
        if tag == 14:
            soname_offsets.append(value)
    if not dynamic_terminated or len(soname_offsets) != 1:
        raise EvidenceError("archived NVML ELF SONAME cardinality is invalid")
    soname = c_string(dynamic_strings, soname_offsets[0], "SONAME")
    if soname != "libnvidia-ml.so.1":
        raise EvidenceError("archived NVML ELF SONAME is not libnvidia-ml.so.1")

    if (
        symbols["entry_size"] != expected_symbol_size
        or symbols["size"] % expected_symbol_size
        or symbols["size"] > 32 * 1024 * 1024
        or symbols["size"] // expected_symbol_size > 262_144
    ):
        raise EvidenceError("archived NVML ELF dynamic-symbol table is invalid")
    symbol_strings = section_data(
        symbols["link"], "symbol string table", 16 * 1024 * 1024
    )
    symbol_data = read_exact(symbols["offset"], symbols["size"], "dynamic symbols")
    exported: set[str] = set()
    for offset in range(0, len(symbol_data), expected_symbol_size):
        symbol = struct.unpack_from(symbol_format, symbol_data, offset)
        if elf_class == 2:
            name_offset, symbol_info, symbol_other, symbol_section = symbol[:4]
        else:
            name_offset = symbol[0]
            symbol_info, symbol_other, symbol_section = symbol[3:6]
        if (
            name_offset
            and symbol_info >> 4 in (1, 2)
            and symbol_info & 0x0f in (2, 10)
            and symbol_other & 0x03 in (0, 3)
        ):
            symbol_name = c_string(symbol_strings, name_offset, "symbol name")
            valid_definition = (
                0 < symbol_section < len(sections)
                or symbol_section == 0xFFF1  # SHN_ABS
            )
            if symbol_name in NVML_REQUIRED_SYMBOLS and not valid_definition:
                raise EvidenceError(
                    "archived NVML ELF required symbol has an invalid section index"
                )
            if valid_definition:
                exported.add(symbol_name)
    if not set(NVML_REQUIRED_SYMBOLS) <= exported:
        raise EvidenceError("archived NVML ELF lacks a required exported symbol")

    build_ids: list[bytes] = []
    aggregate_note_bytes = 0
    for note_index in note_indices:
        note = sections[note_index]
        aggregate_note_bytes += note["size"]
        if (
            note["size"] > 4 * 1024 * 1024
            or aggregate_note_bytes > 8 * 1024 * 1024
        ):
            raise EvidenceError("archived NVML ELF note section is oversized")
        note_data = read_exact(note["offset"], note["size"], "note section")
        cursor = 0
        while cursor < len(note_data):
            if len(note_data) - cursor < 12:
                raise EvidenceError("archived NVML ELF note is truncated")
            name_size, descriptor_size, note_type = struct.unpack_from(
                endian + "III", note_data, cursor
            )
            cursor += 12
            if name_size > len(note_data) - cursor:
                raise EvidenceError("archived NVML ELF note name is oversized")
            name = note_data[cursor:cursor + name_size]
            padded_name_size = (name_size + 3) & ~3
            if padded_name_size > len(note_data) - cursor:
                raise EvidenceError("archived NVML ELF note name padding is oversized")
            cursor += padded_name_size
            if descriptor_size > len(note_data) - cursor:
                raise EvidenceError("archived NVML ELF note descriptor is oversized")
            note_descriptor = note_data[cursor:cursor + descriptor_size]
            padded_descriptor_size = (descriptor_size + 3) & ~3
            if padded_descriptor_size > len(note_data) - cursor:
                raise EvidenceError(
                    "archived NVML ELF note descriptor padding is oversized"
                )
            cursor += padded_descriptor_size
            if note_type == 3 and name_size == 4 and name == b"GNU\0":
                build_ids.append(note_descriptor)
    if len(build_ids) != 1 or not 8 <= len(build_ids[0]) <= 64:
        raise EvidenceError("archived NVML ELF GNU build-id is absent or ambiguous")
    return {
        "schema_version": 1,
        "elf_class_bits": 64 if elf_class == 2 else 32,
        "endianness": "little" if encoding == 1 else "big",
        "machine": machine_names[machine],
        "machine_id": machine,
        "entry_point": entry,
        "program_header_count": program_count,
        "load_segment_count": len(load_programs),
        "soname": soname,
        "gnu_build_id": build_ids[0].hex(),
        "required_symbols": list(NVML_REQUIRED_SYMBOLS),
    }


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def elf_interpreter(path: pathlib.Path) -> pathlib.Path:
    """Read PT_INTERP without trusting an external inspection utility."""

    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:4] != b"\x7fELF":
            raise EvidenceError(f"benchmark worker is not ELF64: {path}")
        if header[4] != 2 or header[5] != 1:
            raise EvidenceError("benchmark worker must be little-endian ELF64")
        fields = struct.unpack("<16sHHIQQQIHHHHHH", header)
        program_offset, program_entry_size, program_count = fields[5], fields[9], fields[10]
        if program_entry_size < 56 or program_count <= 0:
            raise EvidenceError("benchmark worker has an invalid ELF program table")
        interpreter: pathlib.Path | None = None
        for index in range(program_count):
            stream.seek(program_offset + index * program_entry_size)
            entry = stream.read(program_entry_size)
            if len(entry) != program_entry_size:
                raise EvidenceError("benchmark worker ELF program table is truncated")
            kind, _flags, offset, _vaddr, _paddr, size, _memory, _align = struct.unpack(
                "<IIQQQQQQ", entry[:56]
            )
            if kind != 3:
                continue
            stream.seek(offset)
            raw = stream.read(size)
            if not raw.endswith(b"\0") or b"\0" in raw[:-1]:
                raise EvidenceError("benchmark worker PT_INTERP is malformed")
            try:
                candidate = pathlib.Path(raw[:-1].decode("utf-8")).resolve(strict=True)
            except (UnicodeDecodeError, OSError) as exc:
                raise EvidenceError("benchmark worker PT_INTERP cannot be resolved") from exc
            if interpreter is not None:
                raise EvidenceError("benchmark worker has multiple PT_INTERP entries")
            interpreter = candidate
        if interpreter is None:
            raise EvidenceError("benchmark worker has no PT_INTERP loader")
        return interpreter


@dataclass
class StableFile:
    label: str
    path: pathlib.Path
    descriptor: int
    fingerprint: tuple[int, ...]
    sha256: str
    size_bytes: int
    companions: tuple["StableFile", ...] = ()

    @classmethod
    def open(cls, label: str, path: pathlib.Path) -> "StableFile":
        resolved = pathlib.Path(path).resolve(strict=True)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(resolved, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise EvidenceError(f"{label} is not a regular file: {resolved}")
            if _fingerprint(opened) != _fingerprint(resolved.stat()):
                raise EvidenceError(f"{label} pathname changed while opening: {resolved}")
            return cls(
                label=label,
                path=resolved,
                descriptor=descriptor,
                fingerprint=_fingerprint(opened),
                sha256=_sha256_fd(descriptor),
                size_bytes=opened.st_size,
            )
        except BaseException:
            os.close(descriptor)
            raise

    @classmethod
    def open_named_no_follow(
        cls, label: str, path: pathlib.Path, *, maximum_size_bytes: int
    ) -> "StableFile":
        """Open the exact named inode without resolving a final symlink."""

        absolute = pathlib.Path(path).absolute()
        try:
            named_before = absolute.lstat()
        except OSError as exc:
            raise EvidenceError(f"{label} named path is unavailable") from exc
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(absolute, flags)
        except OSError as exc:
            raise EvidenceError(f"{label} is not an exact regular file") from exc
        try:
            opened = os.fstat(descriptor)
            named_after = absolute.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_size < 0
                or opened.st_size > maximum_size_bytes
                or _fingerprint(named_before) != _fingerprint(opened)
                or _fingerprint(named_after) != _fingerprint(opened)
            ):
                raise EvidenceError(f"{label} named inode changed while opening")
            return cls(
                label=label,
                path=absolute,
                descriptor=descriptor,
                fingerprint=_fingerprint(opened),
                sha256=_sha256_fd(descriptor),
                size_bytes=opened.st_size,
            )
        except BaseException:
            os.close(descriptor)
            raise

    @property
    def proc_path(self) -> str:
        return f"/proc/{os.getpid()}/fd/{self.descriptor}"

    def verify(self) -> dict[str, Any]:
        opened = os.fstat(self.descriptor)
        try:
            named = self.path.stat()
        except FileNotFoundError as exc:
            raise EvidenceError(f"stable {self.label} pathname disappeared") from exc
        if _fingerprint(opened) != self.fingerprint or _fingerprint(named) != self.fingerprint:
            raise EvidenceError(f"stable {self.label} changed during execution")
        if _sha256_fd(self.descriptor) != self.sha256:
            raise EvidenceError(f"stable {self.label} content changed during execution")
        for companion in self.companions:
            companion.verify()
        return self.record()

    def record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "fingerprint": {
                "device": self.fingerprint[0],
                "inode": self.fingerprint[1],
                "mode": self.fingerprint[2],
                "mtime_ns": self.fingerprint[4],
                "ctime_ns": self.fingerprint[5],
            },
        }

    def close(self) -> None:
        # Closing verification-only descriptors cannot invalidate a terminal
        # record already published from their verified bytes. Preserve the
        # historic best-effort contract while trying every companion as well.
        if self.descriptor >= 0:
            try:
                os.close(self.descriptor)
            except OSError:
                pass
            finally:
                self.descriptor = -1
        for companion in self.companions:
            companion.close()
        self.companions = ()


def _git_command(repo: pathlib.Path, *arguments: str) -> bytes:
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise EvidenceError(
            f"controller Git identity command failed: {exc.stderr.decode(errors='replace')}"
        ) from exc
    return completed.stdout


def _controller_source_environment_key(name: str) -> str:
    return "GPMEEP_CONTROLLER_SOURCE_FD_" + name.upper().replace("-", "_").replace(".", "_")


def _bootstrap_authoritative_repo(argv: list[str]) -> pathlib.Path:
    positions = [index for index, value in enumerate(argv) if value == "--authoritative-repo"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise EvidenceError("bootstrap requires exactly one --authoritative-repo")
    value = argv[positions[0] + 1]
    if value.startswith("-"):
        raise EvidenceError("bootstrap authoritative repository value is invalid")
    return pathlib.Path(value).resolve(strict=True)


def _controller_bootstrap(argv: list[str]) -> None:
    """Re-exec exact clean committed controller/helper bytes from held FDs."""

    global CONTROLLER_IDENTITY, CONTROLLER_SOURCE_HANDLES, PYTHON_INTERPRETER_HANDLE
    authoritative_repo = _bootstrap_authoritative_repo(argv)
    metadata_text = os.environ.get("GPMEEP_CONTROLLER_BOOTSTRAP_METADATA")
    if metadata_text:
        try:
            metadata = json.loads(metadata_text)
        except json.JSONDecodeError as exc:
            raise EvidenceError("controller bootstrap metadata is invalid") from exc
        sources = metadata.get("sources") if isinstance(metadata, dict) else None
        if not isinstance(sources, dict) or set(sources) != set(SOURCE_HELPERS):
            raise EvidenceError("controller bootstrap source set is incomplete")
        repo = pathlib.Path(metadata.get("repo", "")).resolve(strict=True)
        if repo != authoritative_repo:
            raise EvidenceError("controller repository differs from authoritative repository")
        top = pathlib.Path(
            _git_command(repo, "rev-parse", "--show-toplevel").decode("utf-8").strip()
        ).resolve(strict=True)
        commit = _git_command(repo, "rev-parse", "HEAD").decode("ascii").strip()
        status = _git_command(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ).decode("utf-8")
        if top != repo or commit != metadata.get("git_commit") or status:
            raise EvidenceError("bootstrapped controller Git identity is not clean and exact")
        handles: dict[str, StableFile] = {}
        interpreter: StableFile | None = None
        try:
            for name, record in sources.items():
                descriptor_text = os.environ.get(_controller_source_environment_key(name))
                if descriptor_text is None or not descriptor_text.isdecimal():
                    raise EvidenceError(f"controller bootstrap descriptor is missing: {name}")
                descriptor = int(descriptor_text)
                info = os.fstat(descriptor)
                fingerprint = tuple(record.get("fingerprint", ()))
                if (
                    len(fingerprint) != 6
                    or _fingerprint(info) != fingerprint
                    or info.st_size != record.get("size_bytes")
                    or _sha256_fd(descriptor) != record.get("sha256")
                ):
                    raise EvidenceError(f"controller bootstrap bytes changed: {name}")
                path = pathlib.Path(record["path"]).resolve(strict=True)
                expected_path = (repo / "scripts" / name).resolve(strict=True)
                if path != expected_path:
                    raise EvidenceError(
                        f"controller source is not the exact scripts/{name} path"
                    )
                try:
                    relative = path.relative_to(repo).as_posix()
                except ValueError as exc:
                    raise EvidenceError(f"controller source escapes its Git repo: {name}") from exc
                committed = _git_command(repo, "show", f"HEAD:{relative}")
                if hashlib.sha256(committed).hexdigest() != record.get("sha256") or len(committed) != record.get("size_bytes"):
                    raise EvidenceError(f"held controller bytes differ from Git HEAD: {name}")
                handles[name] = StableFile(
                    label=f"bootstrapped controller source {name}",
                    path=path,
                    descriptor=descriptor,
                    fingerprint=fingerprint,
                    sha256=record["sha256"],
                    size_bytes=record["size_bytes"],
                )
            interpreter_record = metadata.get("python_interpreter")
            descriptor_text = os.environ.get("GPMEEP_CONTROLLER_PYTHON_FD")
            if not isinstance(interpreter_record, dict) or descriptor_text is None or not descriptor_text.isdecimal():
                raise EvidenceError("held Python interpreter identity is missing")
            descriptor = int(descriptor_text)
            info = os.fstat(descriptor)
            fingerprint = tuple(interpreter_record.get("fingerprint", ()))
            if (
                len(fingerprint) != 6
                or _fingerprint(info) != fingerprint
                or _sha256_fd(descriptor) != interpreter_record.get("sha256")
                or info.st_size != interpreter_record.get("size_bytes")
            ):
                raise EvidenceError("held Python interpreter differs from bootstrap identity")
            interpreter = StableFile(
                label="bootstrapped Python interpreter",
                path=pathlib.Path(interpreter_record["path"]).resolve(strict=True),
                descriptor=descriptor,
                fingerprint=fingerprint,
                sha256=interpreter_record["sha256"],
                size_bytes=interpreter_record["size_bytes"],
            )
            running_script = pathlib.Path(__file__)
            controller_handle = handles["run-ldos-reduction-benchmark.py"]
            if (
                hashlib.sha256(running_script.read_bytes()).hexdigest()
                != controller_handle.sha256
                or running_script.stat().st_size != controller_handle.size_bytes
            ):
                raise EvidenceError("currently executing controller bytes differ from held FD")
            running_python = pathlib.Path("/proc/self/exe").stat()
            held_python = os.fstat(interpreter.descriptor)
            if (running_python.st_dev, running_python.st_ino) != (
                held_python.st_dev,
                held_python.st_ino,
            ):
                raise EvidenceError("current Python interpreter differs from held FD")
            CONTROLLER_SOURCE_HANDLES = handles
            PYTHON_INTERPRETER_HANDLE = interpreter
            CONTROLLER_IDENTITY = {
                "schema_version": 1,
                "git_commit": metadata["git_commit"],
                "git_status_porcelain": [],
                "repo": str(repo),
                "sources": {
                    name: handle.record() for name, handle in handles.items()
                },
                "python_interpreter": interpreter.record(),
                "executed_from_held_descriptor": True,
            }
            return
        except BaseException:
            for handle in handles.values():
                handle.close()
            if interpreter is not None:
                interpreter.close()
            raise

    controller_repo = SCRIPT_DIRECTORY.parent.resolve(strict=True)
    if controller_repo != authoritative_repo:
        raise EvidenceError("controller must be executed from the authoritative Git worktree")
    top = pathlib.Path(
        _git_command(controller_repo, "rev-parse", "--show-toplevel")
        .decode("utf-8")
        .strip()
    ).resolve(strict=True)
    if top != controller_repo:
        raise EvidenceError("controller source directory is not its Git worktree root")
    status = _git_command(
        controller_repo, "status", "--porcelain=v1", "--untracked-files=all"
    ).decode("utf-8")
    if status:
        raise EvidenceError("controller Git worktree must be completely clean")
    commit = _git_command(controller_repo, "rev-parse", "HEAD").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise EvidenceError("controller Git commit identity is invalid")
    handles: dict[str, StableFile] = {}
    interpreter: StableFile | None = None
    try:
        source_records: dict[str, Any] = {}
        for name in SOURCE_HELPERS:
            path = SCRIPT_DIRECTORY / name
            expected_path = controller_repo / "scripts" / name
            if path.resolve(strict=True) != expected_path.resolve(strict=True):
                raise EvidenceError(
                    f"controller source directory is not exact for scripts/{name}"
                )
            handle = StableFile.open(f"controller source {name}", path)
            relative = path.relative_to(controller_repo).as_posix()
            committed = _git_command(controller_repo, "show", f"HEAD:{relative}")
            if hashlib.sha256(committed).hexdigest() != handle.sha256 or len(committed) != handle.size_bytes:
                handle.close()
                raise EvidenceError(f"controller source differs from committed bytes: {name}")
            os.set_inheritable(handle.descriptor, True)
            handles[name] = handle
            source_records[name] = {
                "path": str(handle.path),
                "size_bytes": handle.size_bytes,
                "sha256": handle.sha256,
                "fingerprint": list(handle.fingerprint),
            }
        metadata = {
            "schema_version": 1,
            "repo": str(controller_repo),
            "git_commit": commit,
            "sources": source_records,
        }
        interpreter = StableFile.open(
            "controller Python interpreter",
            pathlib.Path(sys.executable).resolve(strict=True),
        )
        os.set_inheritable(interpreter.descriptor, True)
        metadata["python_interpreter"] = {
            "path": str(interpreter.path),
            "size_bytes": interpreter.size_bytes,
            "sha256": interpreter.sha256,
            "fingerprint": list(interpreter.fingerprint),
        }
        environment = {
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        }
        environment["GPMEEP_CONTROLLER_BOOTSTRAP_METADATA"] = json.dumps(
            metadata, separators=(",", ":"), sort_keys=True
        )
        environment["GPMEEP_CONTROLLER_SOURCE_DIRECTORY"] = str(SCRIPT_DIRECTORY)
        for name, handle in handles.items():
            environment[_controller_source_environment_key(name)] = str(handle.descriptor)
        environment["GPMEEP_CONTROLLER_PYTHON_FD"] = str(interpreter.descriptor)
        controller = handles["run-ldos-reduction-benchmark.py"]
        os.execve(
            interpreter.proc_path,
            [interpreter.path.name, "-I", "-S", controller.proc_path, *argv],
            environment,
        )
    finally:
        for handle in handles.values():
            handle.close()
        if interpreter is not None:
            interpreter.close()


def _verify_controller_identity() -> dict[str, Any]:
    if (
        CONTROLLER_IDENTITY is None
        or not CONTROLLER_SOURCE_HANDLES
        or PYTHON_INTERPRETER_HANDLE is None
    ):
        raise EvidenceError("controller did not execute through the committed-byte bootstrap")
    for handle in CONTROLLER_SOURCE_HANDLES.values():
        handle.verify()
    PYTHON_INTERPRETER_HANDLE.verify()
    repo = pathlib.Path(CONTROLLER_IDENTITY["repo"])
    commit = _git_command(repo, "rev-parse", "HEAD").decode("ascii").strip()
    status = _git_command(repo, "status", "--porcelain=v1", "--untracked-files=all").decode("utf-8")
    if commit != CONTROLLER_IDENTITY["git_commit"] or status:
        raise EvidenceError("controller Git identity changed during execution")
    return dict(CONTROLLER_IDENTITY)


def _initialize_verified_helpers() -> None:
    global EVIDENCE, PROVENANCE, PROFILE_SHA256
    if CONTROLLER_IDENTITY is None or not CONTROLLER_SOURCE_HANDLES:
        raise EvidenceError("refusing to execute helpers before controller bootstrap")
    if EVIDENCE is not None or PROVENANCE is not None:
        raise EvidenceError("controller helpers were initialized more than once")
    EVIDENCE = _load_source_module(
        "gpmeep_ldos_benchmark_evidence",
        pathlib.Path(CONTROLLER_SOURCE_HANDLES["gpmeep_benchmark_evidence.py"].proc_path),
    )
    PROVENANCE = _load_source_module(
        "gpmeep_ldos_provenance",
        pathlib.Path(CONTROLLER_SOURCE_HANDLES["gpmeep_provenance.py"].proc_path),
    )
    PROFILE_SHA256 = PROVENANCE.canonical_sha256(dict(PROFILE))


class MutationSentinel:
    """Kernel-backed recursive mutation watch for receipt-prefix runtime files."""

    _MASK = (
        0x00000002  # IN_MODIFY
        | 0x00000004  # IN_ATTRIB
        | 0x00000008  # IN_CLOSE_WRITE
        | 0x00000040  # IN_MOVED_FROM
        | 0x00000080  # IN_MOVED_TO
        | 0x00000100  # IN_CREATE
        | 0x00000200  # IN_DELETE
        | 0x00000400  # IN_DELETE_SELF
        | 0x00000800  # IN_MOVE_SELF
    )
    _OVERFLOW = 0x00004000

    def __init__(self, root: pathlib.Path):
        if sys.platform != "linux":
            raise EvidenceError("runtime mutation sentinel requires Linux inotify")
        self.root = root.resolve(strict=True)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.inotify_init1.argtypes = [ctypes.c_int]
        libc.inotify_init1.restype = ctypes.c_int
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        libc.inotify_add_watch.restype = ctypes.c_int
        descriptor = libc.inotify_init1(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        if descriptor < 0:
            number = ctypes.get_errno()
            raise EvidenceError(f"cannot create inotify sentinel: {os.strerror(number)}")
        self.descriptor = descriptor
        self._libc = libc
        self.watches: dict[int, pathlib.Path] = {}
        try:
            directories = [self.root]
            directories.extend(
                path for path in self.root.rglob("*") if path.is_dir() and not path.is_symlink()
            )
            for directory in directories:
                watch = libc.inotify_add_watch(
                    self.descriptor,
                    os.fsencode(directory),
                    self._MASK,
                )
                if watch < 0:
                    number = ctypes.get_errno()
                    raise EvidenceError(
                        f"cannot watch runtime directory {directory}: {os.strerror(number)}"
                    )
                self.watches[watch] = directory
        except BaseException as initialization_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                initialization_error.add_note(
                    "mutation-sentinel cleanup after initialization failure also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def finish(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            try:
                payload = os.read(self.descriptor, 1024 * 1024)
            except BlockingIOError:
                break
            if not payload:
                break
            offset = 0
            while offset < len(payload):
                if len(payload) - offset < 16:
                    raise EvidenceError("truncated inotify mutation event")
                watch, mask, cookie, name_length = struct.unpack_from(
                    "iIII", payload, offset
                )
                offset += 16
                raw_name = payload[offset : offset + name_length]
                offset += name_length
                name = raw_name.rstrip(b"\0").decode("utf-8", errors="surrogateescape")
                base = self.watches.get(watch)
                events.append(
                    {
                        "watch": watch,
                        "mask": mask,
                        "cookie": cookie,
                        "path": str(base / name if base is not None and name else base or ""),
                    }
                )
                if mask & self._OVERFLOW:
                    raise EvidenceError("runtime mutation sentinel queue overflowed")
        return events

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _copy_stable_file(
    source: StableFile, destination: pathlib.Path, mode: int
) -> StableFile:
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, mode)
    try:
        offset = 0
        while offset < source.size_bytes:
            block = os.pread(
                source.descriptor,
                min(1024 * 1024, source.size_bytes - offset),
                offset,
            )
            if not block:
                raise EvidenceError(f"short read while archiving {source.label}")
            view = memoryview(block)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            offset += len(block)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(destination, mode)
    archived = StableFile.open(f"archived {source.label}", destination)
    if archived.sha256 != source.sha256 or archived.size_bytes != source.size_bytes:
        archived.close()
        raise EvidenceError(f"archived {source.label} does not match its source")
    return archived


def _resolve_recorded_path(
    repo: pathlib.Path, record: dict[str, Any], label: str
) -> pathlib.Path:
    value = record.get("path")
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"receipt record {label} has no path")
    path = pathlib.Path(value)
    return (path if path.is_absolute() else repo / path).resolve(strict=True)


def _require_receipt_file(
    group: Any, name: str, repo: pathlib.Path, label: str
) -> tuple[dict[str, Any], pathlib.Path]:
    if not isinstance(group, dict) or not isinstance(group.get(name), dict):
        raise EvidenceError(f"authoritative receipt is missing {label}")
    record = group[name]
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(
        expected_hash
    ):
        raise EvidenceError(f"authoritative receipt {label} hash is invalid")
    if not _is_plain_int(expected_size) or expected_size <= 0:
        raise EvidenceError(f"authoritative receipt {label} size is invalid")
    return record, _resolve_recorded_path(repo, record, label)


def validate_receipt_inputs(
    *,
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    expected_receipt_id: str,
    mpiexec_path: pathlib.Path,
) -> dict[str, Any]:
    if not SHA256_PATTERN.fullmatch(expected_receipt_id):
        raise EvidenceError("--expected-receipt-id must be a lowercase SHA-256 value")
    receipt_input = StableFile.open("authoritative build receipt", receipt_path)
    try:
        with tempfile.TemporaryDirectory(prefix="gpmeep-held-receipt-") as directory:
            snapshot_path = pathlib.Path(directory) / "build-provenance.json"
            snapshot = _copy_stable_file(receipt_input, snapshot_path, 0o400)
            try:
                receipt = PROVENANCE.verify_build_receipt(
                    snapshot.path, repo, verify_source=True
                )
            finally:
                snapshot.close()
        receipt_input.verify()
        receipt_record = receipt_input.record()
    except Exception as exc:
        raise EvidenceError(f"authoritative build receipt verification failed: {exc}") from exc
    finally:
        receipt_input.close()
    if receipt.get("build_kind") != "cuda-mpi-python-fp32":
        raise EvidenceError("LDOS benchmark requires a CUDA+MPI+Python FP32 receipt")
    if receipt.get("receipt_id") != expected_receipt_id:
        raise EvidenceError("authoritative build receipt ID differs from the expected ID")

    artifacts = receipt.get("artifacts")
    configuration = receipt.get("configuration_files")
    toolchain = receipt.get("toolchain")
    worker_record, worker = _require_receipt_file(
        artifacts, "gpu_step_db_test", repo, "gpu-step-db artifact"
    )
    lib_record, libmeep = _require_receipt_file(
        artifacts, "libmeep", repo, "build libmeep artifact"
    )
    shim_record, fd_allocation_shim = _require_receipt_file(
        artifacts,
        "fd_allocation_shim",
        repo,
        "FD allocation shim artifact",
    )
    mca_record, mca = _require_receipt_file(
        configuration,
        "openmpi_qualification_params",
        repo,
        "Open MPI MCA configuration",
    )
    closure_record, closure = _require_receipt_file(
        configuration,
        "runtime_dependency_closure",
        repo,
        "runtime dependency closure",
    )
    mpi_record, recorded_mpiexec = _require_receipt_file(
        toolchain, "mpiexec", repo, "mpiexec tool"
    )
    timeout_record, timeout_tool = _require_receipt_file(
        toolchain, "timeout", repo, "timeout tool"
    )
    loader = elf_interpreter(worker)
    loader_handle = StableFile.open("worker ELF interpreter", loader)
    try:
        loader_record = loader_handle.record()
    finally:
        loader_handle.close()
    supplied_mpiexec = mpiexec_path.resolve(strict=True)
    if supplied_mpiexec != recorded_mpiexec:
        raise EvidenceError(
            "supplied mpiexec is not the exact receipt-bound tool pathname"
        )
    supplied = StableFile.open("supplied mpiexec", supplied_mpiexec)
    try:
        if (
            supplied.sha256 != mpi_record["sha256"]
            or supplied.size_bytes != mpi_record["size_bytes"]
        ):
            raise EvidenceError("supplied mpiexec does not match the receipt-bound tool")
    finally:
        supplied.close()
    recorded = StableFile.open("receipt mpiexec", recorded_mpiexec)
    try:
        if (
            recorded.sha256 != mpi_record["sha256"]
            or recorded.size_bytes != mpi_record["size_bytes"]
        ):
            raise EvidenceError("receipt-bound mpiexec changed after receipt verification")
    finally:
        recorded.close()
    return {
        "receipt": receipt,
        "paths": {
            "receipt": receipt_path.resolve(strict=True),
            "gpu_step_db": worker,
            "libmeep": libmeep,
            "fd_allocation_shim": fd_allocation_shim,
            "mpiexec": supplied_mpiexec,
            "timeout": timeout_tool,
            "mca": mca,
            "loader": loader,
            "runtime_closure": closure,
        },
        "records": {
            "receipt": receipt_record,
            "gpu_step_db": worker_record,
            "libmeep": lib_record,
            "fd_allocation_shim": shim_record,
            "mpiexec": mpi_record,
            "timeout": timeout_record,
            "mca": mca_record,
            "loader": loader_record,
            "runtime_closure": closure_record,
        },
    }


def open_validated_input_handles(
    validated: dict[str, Any],
) -> dict[str, StableFile]:
    expected_paths = validated.get("paths")
    expected_records = validated.get("records")
    if not isinstance(expected_paths, dict) or not isinstance(expected_records, dict):
        raise EvidenceError("validated input contract is incomplete")
    if set(expected_paths) != set(expected_records):
        raise EvidenceError("validated input path/record sets differ")
    handles: dict[str, StableFile] = {}
    try:
        for label in sorted(expected_paths):
            handle = StableFile.open(f"validated {label}", expected_paths[label])
            record = expected_records[label]
            if (
                handle.path != pathlib.Path(expected_paths[label]).resolve(strict=True)
                or handle.sha256 != record["sha256"]
                or handle.size_bytes != record["size_bytes"]
            ):
                handle.close()
                raise EvidenceError(
                    f"validated {label} changed before its held handle was acquired"
                )
            handles[label] = handle
        return handles
    except BaseException:
        for handle in handles.values():
            handle.close()
        raise


def reverify_receipt_inputs(
    *,
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    expected_receipt_id: str,
    mpiexec_path: pathlib.Path,
    expected_source: dict[str, Any],
    expected_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checked = validate_receipt_inputs(
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=expected_receipt_id,
        mpiexec_path=mpiexec_path,
    )
    receipt = checked["receipt"]
    if receipt.get("source_end") != expected_source:
        raise EvidenceError("authoritative source identity changed during the benchmark")
    if checked["records"] != expected_records:
        raise EvidenceError("receipt artifact/tool records changed during the benchmark")
    return checked


def _claim_output(output: pathlib.Path, run_id: str) -> None:
    try:
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise EvidenceError("output must be a fresh, absent path") from exc
    EVIDENCE.atomic_write_json(
        output / "RUNNING.json",
        {
            "schema_version": 1,
            "state": "RUNNING",
            "run_id": run_id,
            "profile_id": PROFILE_ID,
            "created_at_utc": _utc_now(),
        },
    )
    _fsync_directory_tree(output)
    parent_descriptor = os.open(
        output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _path_is_within(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _write_text_exclusive(path: pathlib.Path, text: str, mode: int = 0o400) -> None:
    """Atomically publish complete bytes into a previously absent final name."""

    path = pathlib.Path(path).absolute()
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    published = False
    try:
        descriptor = os.open(temporary, flags, mode)
        try:
            os.fchmod(descriptor, mode)
            data = text.encode("utf-8")
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise EvidenceError("exclusive gate write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise EvidenceError("atomic no-replace publication requires renameat2")
        renameat2.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(
            -100, os.fsencode(temporary), -100, os.fsencode(path), 1
        ) != 0:
            error_number = ctypes.get_errno()
            raise EvidenceError(
                f"exclusive gate publication failed: {os.strerror(error_number)}"
            )
        published = True
        _fsync_directory(path.parent)
    finally:
        if not published:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _fsync_directory_tree(root: pathlib.Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    directories.append(root)
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        if directory.is_symlink():
            raise EvidenceError(f"cannot fsync symbolic-link directory: {directory}")
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _fsync_directory(path: pathlib.Path) -> None:
    if path.is_symlink():
        raise EvidenceError(f"cannot fsync symbolic-link directory: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree_nofollow(root: pathlib.Path) -> list[pathlib.Path]:
    """Fsync an existing tree leaf-first without traversing symbolic links."""

    root = pathlib.Path(root)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_descriptor = os.open(root, flags)
    fsynced: list[pathlib.Path] = []
    visited: set[tuple[int, int]] = set()

    def walk(descriptor: int, directory: pathlib.Path) -> None:
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        if identity in visited:
            return
        visited.add(identity)
        children: list[tuple[str, pathlib.Path, os.stat_result]] = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISDIR(info.st_mode):
                    children.append((entry.name, directory / entry.name, info))
        for name, child_path, expected in sorted(
            children, key=lambda item: item[0]
        ):
            try:
                child_descriptor = os.open(name, flags, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno in (
                    errno.ENOENT, errno.ENOTDIR, errno.ELOOP,
                    errno.EACCES, errno.EPERM,
                ):
                    continue
                raise
            try:
                child_opened = os.fstat(child_descriptor)
                if (
                    not stat.S_ISDIR(child_opened.st_mode)
                    or (child_opened.st_dev, child_opened.st_ino)
                    != (expected.st_dev, expected.st_ino)
                ):
                    continue
                walk(child_descriptor, child_path)
            finally:
                os.close(child_descriptor)
        os.fsync(descriptor)
        fsynced.append(directory)

    try:
        walk(root_descriptor, root)
    finally:
        os.close(root_descriptor)
    return fsynced


def _raw_child_subreaper_state() -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    state = ctypes.c_int()
    if libc.prctl(PR_GET_CHILD_SUBREAPER, ctypes.addressof(state), 0, 0, 0) != 0:
        number = ctypes.get_errno()
        raise EvidenceError(f"cannot query child-subreaper state: {os.strerror(number)}")
    return bool(state.value)


def _child_subreaper_state() -> bool:
    return _raw_child_subreaper_state()


def _raw_set_child_subreaper(enabled: bool) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(PR_SET_CHILD_SUBREAPER, int(enabled), 0, 0, 0) != 0:
        number = ctypes.get_errno()
        raise EvidenceError(f"cannot configure child subreaper: {os.strerror(number)}")


def _set_child_subreaper(enabled: bool) -> None:
    _raw_set_child_subreaper(enabled)


_UNRESOLVED_SUBREAPER_TARGET: bool | None = None


def _restore_child_subreaper_state(
    expected: bool, timeout_seconds: float,
) -> list[str]:
    """Restore and independently verify subreaper state without escaping."""

    global _UNRESOLVED_SUBREAPER_TARGET
    _UNRESOLVED_SUBREAPER_TARGET = expected
    diagnostics: list[str] = []
    reported: set[str] = set()

    def record(key: str, message: str) -> None:
        if key not in reported:
            reported.add(key)
            diagnostics.append(message)

    try:
        deadline = time.monotonic() + timeout_seconds
    except BaseException as error:
        deadline = None
        record(
            "restore-clock-init",
            "subreaper restore clock initialization raised "
            f"{type(error).__name__}: {error}",
        )
    for _attempt in range(64):
        for label, setter in (
            ("primary", _set_child_subreaper),
            ("independent", _raw_set_child_subreaper),
        ):
            try:
                setter(expected)
            except BaseException as error:
                record(
                    f"{label}-set",
                    f"{label} subreaper restore raised "
                    f"{type(error).__name__}: {error}",
                )
            try:
                observed = _raw_child_subreaper_state()
            except BaseException as error:
                record(
                    "independent-query",
                    "independent subreaper verification raised "
                    f"{type(error).__name__}: {error}",
                )
            else:
                if observed == expected:
                    _UNRESOLVED_SUBREAPER_TARGET = None
                    return diagnostics
                record(
                    "state-mismatch",
                    "subreaper restore verification observed the wrong state",
                )
        if deadline is not None:
            try:
                if time.monotonic() >= deadline:
                    break
            except BaseException as error:
                record(
                    "restore-clock",
                    "subreaper restore clock raised "
                    f"{type(error).__name__}: {error}",
                )
                deadline = None
        try:
            time.sleep(0.001)
        except BaseException as error:
            record(
                "restore-sleep",
                "subreaper restore sleep raised "
                f"{type(error).__name__}: {error}",
            )
    record("unresolved", "subreaper state could not be restored and verified")
    return diagnostics


def _process_children(pid: int) -> set[int]:
    children: set[int] = set()
    task_root = pathlib.Path(f"/proc/{pid}/task")
    try:
        tasks = tuple(task_root.iterdir())
    except FileNotFoundError:
        return children
    for task in tasks:
        if not task.name.isdecimal():
            continue
        try:
            words = (task / "children").read_text(encoding="ascii").split()
        except FileNotFoundError:
            continue
        for word in words:
            if word.isdecimal():
                children.add(int(word))
    return children


def _process_descendants(pid: int) -> set[int]:
    found: set[int] = set()
    pending = list(_process_children(pid))
    while pending:
        child = pending.pop()
        if child in found:
            continue
        found.add(child)
        pending.extend(_process_children(child) - found)
    return found


def _process_table_descendants(pid: int) -> set[int]:
    """Independently derive descendants from the process-table PPid graph."""

    children_by_parent: dict[int, set[int]] = {}
    try:
        with os.scandir("/proc") as iterator:
            entries = tuple(iterator)
    except OSError as error:
        raise EvidenceError("cannot scan /proc process table") from error
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        candidate_pid = int(entry.name)
        try:
            text = pathlib.Path(entry.path, "stat").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as error:
            if error.errno in (errno.ENOENT, errno.ESRCH):
                continue
            raise EvidenceError(
                f"cannot read process-table stat for PID {candidate_pid}"
            ) from error
        opening = text.find("(")
        close = text.rfind(")")
        if opening < 1 or close <= opening:
            raise EvidenceError(
                f"process-table stat identity for PID {candidate_pid} is invalid"
            )
        fields = text[close + 2 :].split()
        try:
            stat_pid = int(text[:opening].strip())
            parent_pid = int(fields[1])
        except (IndexError, ValueError) as error:
            raise EvidenceError(
                f"process-table stat for PID {candidate_pid} is invalid"
            ) from error
        if stat_pid != candidate_pid:
            raise EvidenceError(
                f"process-table stat identity for PID {candidate_pid} is invalid"
            )
        children_by_parent.setdefault(parent_pid, set()).add(candidate_pid)

    found: set[int] = set()
    pending = list(children_by_parent.get(pid, set()))
    while pending:
        child = pending.pop()
        if child in found:
            continue
        found.add(child)
        pending.extend(children_by_parent.get(child, set()) - found)
    return found


def _open_descriptor_snapshot_once() -> set[int]:
    """Return live FDs after excluding the scanner's transient directory FD."""

    try:
        entries = tuple(pathlib.Path("/proc/self/fd").iterdir())
    except OSError as error:
        raise EvidenceError("cannot snapshot controller descriptors") from error
    descriptors: set[int] = set()
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        descriptor = int(entry.name)
        try:
            os.fstat(descriptor)
        except OSError as error:
            if error.errno == errno.EBADF:
                continue
            raise EvidenceError(
                f"cannot verify controller descriptor {descriptor}"
            ) from error
        descriptors.add(descriptor)
    return descriptors


def _stable_open_descriptor_snapshot() -> set[int]:
    """Require a fixed descriptor set before using allocation recovery."""

    previous: set[int] | None = None
    for _attempt in range(8):
        current = _open_descriptor_snapshot_once()
        if current == previous:
            return current
        previous = current
    raise EvidenceError("controller descriptor set did not stabilize")


FD_ALLOCATION_SHIM_ABI = 1
FD_ALLOCATION_SHIM_FD_LAYOUT = (
    (16 << 32) | (0 << 24) | (4 << 16) | (8 << 8) | 12
)
FD_ALLOCATION_SHIM_PIPE_LAYOUT = (
    (20 << 40) | (0 << 32) | (4 << 24) | (8 << 16) | (12 << 8) | 16
)


class NativeFdResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("completed", ctypes.c_int32),
        ("fd", ctypes.c_int32),
        ("error_number", ctypes.c_int32),
    ]


class NativePipeResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("completed", ctypes.c_int32),
        ("read_fd", ctypes.c_int32),
        ("write_fd", ctypes.c_int32),
        ("error_number", ctypes.c_int32),
    ]


def _verify_fd_allocation_shim_abi_layout() -> None:
    fd_offsets = tuple(
        getattr(NativeFdResult, name).offset
        for name in ("abi_version", "completed", "fd", "error_number")
    )
    pipe_offsets = tuple(
        getattr(NativePipeResult, name).offset
        for name in (
            "abi_version", "completed", "read_fd", "write_fd", "error_number",
        )
    )
    if (
        ctypes.sizeof(NativeFdResult) != 16
        or fd_offsets != (0, 4, 8, 12)
        or ctypes.sizeof(NativePipeResult) != 20
        or pipe_offsets != (0, 4, 8, 12, 16)
    ):
        raise EvidenceError("FD allocation shim ABI layout differs from native C")


class NativeFdAllocationShim:
    """Receipt-bound native allocator with caller-owned result storage."""

    def __init__(self, artifact: StableFile):
        _verify_fd_allocation_shim_abi_layout()
        artifact.verify()
        try:
            library = ctypes.CDLL(artifact.proc_path, mode=ctypes.RTLD_LOCAL)
            abi = library.gpmeep_fd_allocation_shim_abi
            abi.argtypes = []
            abi.restype = ctypes.c_uint32
            fd_layout = library.gpmeep_fd_result_layout
            fd_layout.argtypes = []
            fd_layout.restype = ctypes.c_uint64
            pipe_layout = library.gpmeep_pipe_result_layout
            pipe_layout.argtypes = []
            pipe_layout.restype = ctypes.c_uint64
            duplicate = library.gpmeep_dupfd_cloexec_into
            duplicate.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(NativeFdResult),
            ]
            duplicate.restype = None
            pidfd_open = library.gpmeep_pidfd_open_into
            pidfd_open.argtypes = [
                ctypes.c_int,
                ctypes.c_uint,
                ctypes.POINTER(NativeFdResult),
            ]
            pidfd_open.restype = None
            pipe2 = library.gpmeep_pipe2_into
            pipe2.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(NativePipeResult),
            ]
            pipe2.restype = None
            if abi() != FD_ALLOCATION_SHIM_ABI:
                raise EvidenceError("FD allocation shim ABI differs from controller")
            if (
                fd_layout() != FD_ALLOCATION_SHIM_FD_LAYOUT
                or pipe_layout() != FD_ALLOCATION_SHIM_PIPE_LAYOUT
            ):
                raise EvidenceError(
                    "FD allocation shim native layout differs from controller"
                )
        except BaseException as error:
            raise EvidenceError("cannot load exact FD allocation shim") from error
        artifact.verify()
        self._library = library
        self._duplicate = duplicate
        self._pidfd_open = pidfd_open
        self._pipe2 = pipe2
        self.artifact_identity = artifact.record()

    def duplicate_into(
        self, source_fd: int, minimum_fd: int, result: NativeFdResult,
    ) -> None:
        self._duplicate(source_fd, minimum_fd, ctypes.byref(result))

    def pidfd_open_into(
        self, pid: int, flags: int, result: NativeFdResult,
    ) -> None:
        self._pidfd_open(pid, flags, ctypes.byref(result))

    def pipe2_into(self, flags: int, result: NativePipeResult) -> None:
        self._pipe2(flags, ctypes.byref(result))


_FD_ALLOCATION_SHIM: Any = None


def _require_fd_allocation_shim() -> Any:
    if _FD_ALLOCATION_SHIM is None:
        raise EvidenceError("receipt-bound FD allocation shim is not initialized")
    return _FD_ALLOCATION_SHIM


class NativeOwnedFd(int):
    """An FD generation that closes itself until an explicit closer adopts it."""

    def __new__(cls, descriptor: int, label: str):
        instance = int.__new__(cls, descriptor)
        instance._native_owned = True
        instance._native_label = label
        return instance

    def _disarm(self) -> None:
        self._native_owned = False

    def __del__(self) -> None:
        if not getattr(self, "_native_owned", False):
            return
        # A returned object can be discarded by a trace/signal exception before
        # caller assignment. Its finalizer is the last exact-generation owner.
        # Never let finalizer diagnostics replace the active exception.
        try:
            closed, _diagnostic = _raw_close_owned_descriptor(self)
            if not closed:
                _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(int(self))
        except BaseException:
            # The helper may already have authoritatively closed and disarmed
            # this generation before an asynchronous exception reached this
            # frame. Never retry its bare numeric slot: another thread or a
            # trace callback can already have reused that number. If ownership
            # is still armed, poison and invalidate it so later controller
            # operations fail closed without an ABA close.
            try:
                if getattr(self, "_native_owned", False):
                    _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(int(self))
                    self._native_owned = False
            except BaseException:
                pass
        finally:
            self._native_owned = False


def _disarm_native_owned_descriptor(descriptor: int) -> None:
    if isinstance(descriptor, NativeOwnedFd):
        descriptor._disarm()


def _close_fresh_allocations(
    label: str, descriptors: Iterable[int], primary_error: BaseException,
) -> None:
    notes: list[str] = []
    for descriptor in sorted(set(descriptors)):
        if descriptor < 0:
            continue
        _POISONED_PROBE_DESCRIPTOR_NUMBERS.discard(descriptor)
        closed, diagnostic = _raw_close_owned_descriptor(descriptor)
        if diagnostic is not None:
            notes.append(f"{label} descriptor {descriptor}: {diagnostic}")
        if not closed:
            _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(descriptor)
            notes.append(
                f"{label} descriptor {descriptor} close is ambiguous and poisoned"
            )
    for note in notes:
        primary_error.add_note(note)


def _native_single_fd_allocation(
    label: str,
    invoke: Callable[[Any, NativeFdResult], None],
    validator: Callable[[int], None],
) -> NativeOwnedFd:
    result = NativeFdResult(0, 0, -1, 0)
    call_error: BaseException | None = None
    returned: Any = None
    owner: NativeOwnedFd | None = None
    try:
        try:
            returned = invoke(_require_fd_allocation_shim(), result)
        except BaseException as error:
            call_error = error

        contract_error: BaseException | None = None
        if result.abi_version not in (0, FD_ALLOCATION_SHIM_ABI):
            contract_error = EvidenceError(f"{label} returned the wrong shim ABI")
        elif result.completed not in (0, 1):
            contract_error = EvidenceError(f"{label} completion state is invalid")
        elif returned is not None:
            contract_error = EvidenceError(f"{label} native entrypoint returned data")
        elif result.completed == 1:
            success = result.fd >= 0 and result.error_number == 0
            failure = result.fd == -1 and result.error_number > 0
            if not (success or failure):
                contract_error = EvidenceError(f"{label} native result is inconsistent")
        elif call_error is None:
            contract_error = EvidenceError(f"{label} native call did not complete")

        primary_error = call_error or contract_error
        if primary_error is not None:
            raise primary_error
        if result.fd < 0:
            raise OSError(result.error_number, os.strerror(result.error_number))

        owner = NativeOwnedFd(result.fd, label)
        validator(owner)
        _POISONED_PROBE_DESCRIPTOR_NUMBERS.discard(owner)
        return owner
    except BaseException as active_error:
        descriptors: tuple[int, ...] = (
            (owner,) if owner is not None else ((result.fd,) if result.fd >= 0 else ())
        )
        if descriptors:
            _close_fresh_allocations(label, descriptors, active_error)
        raise


def _native_pipe_allocation() -> tuple[NativeOwnedFd, NativeOwnedFd]:
    result = NativePipeResult(0, 0, -1, -1, 0)
    call_error: BaseException | None = None
    returned: Any = None
    owners: tuple[NativeOwnedFd, NativeOwnedFd] | None = None
    try:
        try:
            returned = _require_fd_allocation_shim().pipe2_into(
                getattr(os, "O_CLOEXEC", 0), result
            )
        except BaseException as error:
            call_error = error

        contract_error: BaseException | None = None
        if result.abi_version not in (0, FD_ALLOCATION_SHIM_ABI):
            contract_error = EvidenceError("readiness pipe returned the wrong shim ABI")
        elif result.completed not in (0, 1):
            contract_error = EvidenceError("readiness pipe completion state is invalid")
        elif returned is not None:
            contract_error = EvidenceError("readiness pipe native entrypoint returned data")
        elif result.completed == 1:
            success = (
                result.read_fd >= 0
                and result.write_fd >= 0
                and result.read_fd != result.write_fd
                and result.error_number == 0
            )
            failure = (
                result.read_fd == -1
                and result.write_fd == -1
                and result.error_number > 0
            )
            if not (success or failure):
                contract_error = EvidenceError("readiness pipe native result is inconsistent")
        elif call_error is None:
            contract_error = EvidenceError("readiness pipe native call did not complete")

        primary_error = call_error or contract_error
        if primary_error is not None:
            raise primary_error
        if result.read_fd < 0 or result.write_fd < 0:
            raise OSError(result.error_number, os.strerror(result.error_number))
        owners = (
            NativeOwnedFd(result.read_fd, "readiness pipe read end"),
            NativeOwnedFd(result.write_fd, "readiness pipe write end"),
        )
        _probe_pipe_validator(owners)
        for descriptor in owners:
            _POISONED_PROBE_DESCRIPTOR_NUMBERS.discard(descriptor)
        return owners
    except BaseException as active_error:
        descriptors = (
            owners
            if owners is not None
            else tuple(
                descriptor
                for descriptor in (result.read_fd, result.write_fd)
                if descriptor >= 0
            )
        )
        if descriptors:
            _close_fresh_allocations(
                "readiness pipe", descriptors, active_error
            )
        raise


def _pidfd_identity(pid: int, descriptor: int) -> tuple[int, tuple[int, ...]]:
    try:
        text = pathlib.Path(f"/proc/self/fdinfo/{descriptor}").read_text(
            encoding="ascii"
        )
    except OSError as error:
        raise EvidenceError("cannot read allocated pidfd identity") from error
    pid_values: list[int] = []
    namespace_values: list[tuple[int, ...]] = []
    for line in text.splitlines():
        if line.startswith("Pid:\t"):
            try:
                pid_values.append(int(line.split("\t", 1)[1]))
            except ValueError as error:
                raise EvidenceError("allocated pidfd Pid field is invalid") from error
        elif line.startswith("NSpid:\t"):
            try:
                namespace_values.append(
                    tuple(int(value) for value in line.split("\t")[1:] if value)
                )
            except ValueError as error:
                raise EvidenceError("allocated pidfd NSpid field is invalid") from error
    if (
        len(pid_values) != 1
        or len(namespace_values) != 1
        or not namespace_values[0]
        or pid_values[0] != pid
        or namespace_values[0][0] != pid
    ):
        raise EvidenceError("allocated pidfd is bound to the wrong PID")
    return pid_values[0], namespace_values[0]


def _pidfd_open(
    pid: int, *, expected_start_time_ticks: int | None = None
) -> NativeOwnedFd:
    if type(pid) is not int or pid <= 0:
        raise ValueError("pidfd target PID must be a positive integer")
    if expected_start_time_ticks is not None and (
        type(expected_start_time_ticks) is not int
        or expected_start_time_ticks <= 0
    ):
        raise ValueError("pidfd expected start epoch must be a positive integer")
    observed_before = _process_start_time_ticks(pid)
    if observed_before is None:
        raise ProcessLookupError(errno.ESRCH, os.strerror(errno.ESRCH), pid)
    bound_epoch = (
        observed_before
        if expected_start_time_ticks is None
        else expected_start_time_ticks
    )
    if observed_before != bound_epoch:
        raise EvidenceError("pidfd target epoch differs before allocation")

    def validate(descriptor: int) -> None:
        _pidfd_identity(pid, descriptor)
        observed_after = _process_start_time_ticks(pid)
        if observed_after != bound_epoch:
            raise EvidenceError("pidfd target epoch changed during allocation")

    return _native_single_fd_allocation(
        "pidfd_open",
        lambda shim, result: shim.pidfd_open_into(pid, 0, result),
        validate,
    )


def _probe_pipe_validator(descriptors: tuple[int, ...]) -> None:
    read_descriptor, write_descriptor = descriptors
    read_state = os.fstat(read_descriptor)
    write_state = os.fstat(write_descriptor)
    if (
        not stat.S_ISFIFO(read_state.st_mode)
        or not stat.S_ISFIFO(write_state.st_mode)
        or (read_state.st_dev, read_state.st_ino)
        != (write_state.st_dev, write_state.st_ino)
        or fcntl.fcntl(read_descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        != os.O_RDONLY
        or fcntl.fcntl(write_descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        != os.O_WRONLY
        or not fcntl.fcntl(read_descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        or not fcntl.fcntl(write_descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    ):
        raise EvidenceError("allocated readiness pipe identity is invalid")


def _allocate_probe_pipe() -> tuple[NativeOwnedFd, NativeOwnedFd]:
    return _native_pipe_allocation()


def _fd_allocation_shim_self_test() -> None:
    baseline = _stable_open_descriptor_snapshot()
    source_descriptor: int | None = None
    allocated: list[int] = []
    primary_error: BaseException | None = None
    try:
        source_descriptor = os.open(
            "/dev/null", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        )
        guard = _allocate_generation_guard(source_descriptor)
        allocated.append(guard)
        pidfd = _pidfd_open(os.getpid())
        allocated.append(pidfd)
        _pidfd_send_signal(pidfd, 0)
        read_descriptor, write_descriptor = _allocate_probe_pipe()
        allocated.extend((read_descriptor, write_descriptor))
    except BaseException as error:
        primary_error = error
    finally:
        cleanup_notes: list[str] = []
        for label, descriptor in [
            *(("allocated", item) for item in reversed(allocated)),
            ("source", source_descriptor),
        ]:
            if descriptor is None:
                continue
            _POISONED_PROBE_DESCRIPTOR_NUMBERS.discard(descriptor)
            closed, diagnostic = _raw_close_owned_descriptor(descriptor)
            if diagnostic is not None:
                cleanup_notes.append(
                    f"FD allocation shim self-test {label} {descriptor}: {diagnostic}"
                )
            if not closed:
                _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(descriptor)
                cleanup_notes.append(
                    f"FD allocation shim self-test {label} {descriptor} close is ambiguous"
                )
        try:
            final = _stable_open_descriptor_snapshot()
        except BaseException as error:
            cleanup_notes.append(
                "FD allocation shim self-test final snapshot raised "
                f"{type(error).__name__}: {error}"
            )
        else:
            if final != baseline:
                cleanup_notes.append("FD allocation shim self-test leaked descriptors")
        if cleanup_notes:
            if primary_error is None:
                primary_error = EvidenceError(cleanup_notes[0])
                cleanup_notes = cleanup_notes[1:]
            for note in cleanup_notes:
                primary_error.add_note(note)
    if primary_error is not None:
        raise primary_error


def _initialize_fd_allocation_shim(artifact: StableFile) -> None:
    global _FD_ALLOCATION_SHIM
    if _FD_ALLOCATION_SHIM is not None:
        raise EvidenceError("FD allocation shim was already initialized")
    shim = NativeFdAllocationShim(artifact)
    _FD_ALLOCATION_SHIM = shim
    try:
        _fd_allocation_shim_self_test()
        artifact.verify()
    except BaseException:
        _FD_ALLOCATION_SHIM = None
        raise


def _bind_loaded_fd_allocation_shim_archive(
    source: StableFile, archived: StableFile,
) -> None:
    """Prove the pre-probe loaded shim is the exact later archive source."""

    source_identity = source.verify()
    archived_identity = archived.verify()
    shim_identity = getattr(_require_fd_allocation_shim(), "artifact_identity", None)
    if shim_identity != source_identity:
        raise EvidenceError("loaded FD allocation shim identity changed before archive")
    if (
        archived_identity["sha256"] != source_identity["sha256"]
        or archived_identity["size_bytes"] != source_identity["size_bytes"]
    ):
        raise EvidenceError("archived FD allocation shim differs from loaded bytes")


def _raw_waitpid_nohang(pid: int) -> tuple[int, int]:
    """Call libc waitpid independently of the Python wrapper."""

    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.waitpid
    function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    function.restype = ctypes.c_int
    status = ctypes.c_int()
    result = function(pid, ctypes.byref(status), os.WNOHANG)
    if result < 0:
        number = ctypes.get_errno()
        if number == errno.EINTR:
            raise InterruptedError(number, os.strerror(number))
        if number == errno.ECHILD:
            raise ChildProcessError(number, os.strerror(number))
        raise OSError(number, os.strerror(number))
    return int(result), int(status.value)


def _waitpid_nohang(pid: int) -> tuple[int, int]:
    try:
        return os.waitpid(pid, os.WNOHANG)
    except InterruptedError:
        return _raw_waitpid_nohang(pid)


_RETAINED_PROBE_DESCRIPTORS: dict[int, int | None] = {}
_POISONED_PROBE_DESCRIPTOR_NUMBERS: set[int] = set()


def _same_open_file_description(
    descriptor: int, guard_descriptor: int,
) -> tuple[bool | None, str | None]:
    """Compare open-file descriptions, not same-resource metadata."""

    try:
        syscall_number = SYS_KCMP_BY_MACHINE.get(platform.machine().lower())
        if syscall_number is None:
            return None, "kcmp syscall ABI is unavailable on this architecture"
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        result = libc.syscall(
            syscall_number,
            os.getpid(),
            os.getpid(),
            KCMP_FILE,
            descriptor,
            guard_descriptor,
        )
        if result == 0:
            return True, None
        if result > 0:
            return False, None
        number = ctypes.get_errno()
        if number == errno.EBADF:
            return False, "kcmp observed a closed descriptor"
        return None, (
            f"kcmp raised OSError: [Errno {number}] {os.strerror(number)}"
        )
    except BaseException as error:
        return None, f"kcmp raised {type(error).__name__}: {error}"


def _raw_close_owned_descriptor(descriptor: int) -> tuple[bool, str | None]:
    """Issue one raw close; any kernel return is authoritative on Linux."""

    if isinstance(descriptor, NativeOwnedFd) and not getattr(
        descriptor, "_native_owned", False
    ):
        return True, "native descriptor owner was already disarmed"
    if descriptor in _POISONED_PROBE_DESCRIPTOR_NUMBERS:
        return False, f"raw close refused poisoned descriptor {descriptor}"
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.close
        function.argtypes = [ctypes.c_int]
        function.restype = ctypes.c_int
        result = function(int(descriptor))
        if result == 0:
            _disarm_native_owned_descriptor(descriptor)
            return True, None
        number = ctypes.get_errno()
        if number == errno.EBADF:
            _disarm_native_owned_descriptor(descriptor)
            return True, "raw close observed EBADF"
        # Linux releases the descriptor before reporting later close/writeback
        # errors. Never retry the raw number after a kernel close return.
        _disarm_native_owned_descriptor(descriptor)
        return True, (
            "raw close returned OSError after ownership release: "
            f"[Errno {number}] {os.strerror(number)}"
        )
    except BaseException as error:
        # The syscall's state is ambiguous across an exception boundary. The
        # owner must not retry its bare number from __del__, where an ABA reuse
        # could close an unrelated generation.
        try:
            _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(int(descriptor))
            _disarm_native_owned_descriptor(descriptor)
        except BaseException:
            pass
        return False, f"raw close raised {type(error).__name__}: {error}"


def _close_native_owned_descriptor_once(label: str, descriptor: int) -> None:
    closed, diagnostic = _raw_close_owned_descriptor(descriptor)
    if not closed:
        _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(int(descriptor))
        message = f"{label} close is ambiguous and poisoned"
        if diagnostic is not None:
            message += f": {diagnostic}"
        raise EvidenceError(message)


def _generation_guard_validator(
    source_descriptor: int, guard_descriptor: int,
) -> None:
    if not fcntl.fcntl(guard_descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC:
        raise EvidenceError("generation guard is not close-on-exec")
    same_generation, diagnostic = _same_open_file_description(
        source_descriptor, guard_descriptor
    )
    if same_generation is not True:
        message = "generation guard is not bound to the source description"
        if diagnostic is not None:
            message += f": {diagnostic}"
        raise EvidenceError(message)


def _allocate_generation_guard(source_descriptor: int) -> NativeOwnedFd:
    return _native_single_fd_allocation(
        "generation guard",
        lambda shim, result: shim.duplicate_into(
            source_descriptor, 0, result
        ),
        lambda guard_descriptor: _generation_guard_validator(
            source_descriptor, guard_descriptor
        ),
    )


def _close_descriptor_verified(
    descriptor: int,
) -> tuple[bool, list[str]]:
    """Close an FD using a duplicate open-file-description generation guard."""

    diagnostics: list[str] = []
    if descriptor in _POISONED_PROBE_DESCRIPTOR_NUMBERS:
        _disarm_native_owned_descriptor(descriptor)
        return True, [
            f"descriptor {descriptor} is poisoned; local reference invalidated"
        ]
    try:
        guard_descriptor = _allocate_generation_guard(descriptor)
    except OSError as error:
        if error.errno == errno.EBADF:
            _disarm_native_owned_descriptor(descriptor)
            return True, diagnostics
        diagnostics.append(
            f"generation-guard duplication raised {type(error).__name__}: {error}"
        )
        closed, raw_diagnostic = _raw_close_owned_descriptor(descriptor)
        if raw_diagnostic is not None:
            diagnostics.append(raw_diagnostic)
        if not closed:
            _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(descriptor)
            diagnostics.append(
                f"descriptor {descriptor} close is ambiguous and poisoned"
            )
            return True, diagnostics
        return True, diagnostics
    except BaseException as error:
        diagnostics.append(
            f"generation-guard duplication raised {type(error).__name__}: {error}"
        )
        closed, raw_diagnostic = _raw_close_owned_descriptor(descriptor)
        if raw_diagnostic is not None:
            diagnostics.append(raw_diagnostic)
        if not closed:
            _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(descriptor)
            diagnostics.append(
                f"descriptor {descriptor} close is ambiguous and poisoned"
            )
            return True, diagnostics
        return True, diagnostics

    # The atomic dup proves that this numeric slot now denotes a fresh guard
    # owned by this function.  Rebind any stale poison from an older descriptor
    # generation before registering and eventually closing the guard.
    _POISONED_PROBE_DESCRIPTOR_NUMBERS.discard(guard_descriptor)

    # Guard ownership is recorded immediately after the atomic CLOEXEC dup.
    # Every later escape therefore leaves a non-reusable capability reachable.
    _RETAINED_PROBE_DESCRIPTORS[guard_descriptor] = descriptor
    candidate_resolved = False
    candidate_retry_safe = False
    candidate_poisoned = False
    try:
        try:
            os.close(descriptor)
        except OSError as error:
            if error.errno == errno.EBADF:
                candidate_resolved = True
            else:
                diagnostics.append(
                    f"os.close raised {type(error).__name__}: {error}"
                )
        except BaseException as error:
            diagnostics.append(f"os.close raised {type(error).__name__}: {error}")
        else:
            candidate_resolved = True
            _disarm_native_owned_descriptor(descriptor)

        if not candidate_resolved:
            same_generation, comparison_diagnostic = _same_open_file_description(
                descriptor, guard_descriptor
            )
            if comparison_diagnostic is not None:
                diagnostics.append(
                    f"generation comparison: {comparison_diagnostic}"
                )
            if same_generation is True:
                candidate_resolved, raw_diagnostic = _raw_close_owned_descriptor(
                    descriptor
                )
                if raw_diagnostic is not None:
                    diagnostics.append(raw_diagnostic)
                if not candidate_resolved:
                    post_same, post_diagnostic = _same_open_file_description(
                        descriptor, guard_descriptor
                    )
                    if post_diagnostic is not None:
                        diagnostics.append(
                            f"post-close generation comparison: {post_diagnostic}"
                        )
                    if post_same is True:
                        candidate_retry_safe = True
                    elif post_same is False:
                        candidate_resolved = True
                    else:
                        candidate_poisoned = True
            elif same_generation is False:
                candidate_resolved = True
                diagnostics.append(
                    "descriptor number was closed or reused; replacement was not closed"
                )
            else:
                # No raw fallback has been attempted. The held guard makes a
                # later kcmp retry generation-safe even if the primary close
                # actually closed/reused the candidate number.
                candidate_retry_safe = True
    except BaseException as error:
        diagnostics.append(
            "descriptor generation cleanup raised "
            f"{type(error).__name__}: {error}"
        )
        candidate_poisoned = True

    if candidate_retry_safe:
        return False, diagnostics
    if candidate_poisoned:
        _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(descriptor)
        diagnostics.append(
            f"descriptor {descriptor} close generation is ambiguous and poisoned"
        )
    if candidate_resolved or candidate_poisoned:
        _disarm_native_owned_descriptor(descriptor)
    _RETAINED_PROBE_DESCRIPTORS[guard_descriptor] = None

    guard_closed, guard_diagnostic = _raw_close_owned_descriptor(
        guard_descriptor
    )
    if guard_diagnostic is not None:
        diagnostics.append(f"generation guard: {guard_diagnostic}")
    if guard_closed:
        _RETAINED_PROBE_DESCRIPTORS.pop(guard_descriptor, None)
    else:
        # An ambiguous close of the sole generation guard is never retried by
        # number. Poisoning blocks all resolver continuation without risking
        # an unrelated future owner.
        _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(guard_descriptor)
        _RETAINED_PROBE_DESCRIPTORS.pop(guard_descriptor, None)
        diagnostics.append(
            f"generation guard {guard_descriptor} close is ambiguous and poisoned"
        )
    if candidate_poisoned or not guard_closed:
        # The local numeric reference is invalidated even though global poison
        # records unresolved ownership and prevents resolver continuation.
        return True, diagnostics
    return candidate_resolved, diagnostics


def _drain_retained_probe_descriptors() -> list[str]:
    diagnostics: list[str] = []
    for guard_descriptor, candidate_descriptor in tuple(
        sorted(_RETAINED_PROBE_DESCRIPTORS.items())
    ):
        if guard_descriptor in _POISONED_PROBE_DESCRIPTOR_NUMBERS:
            diagnostics.append(
                f"retained guard {guard_descriptor} is poisoned and was not retried"
            )
            _RETAINED_PROBE_DESCRIPTORS.pop(guard_descriptor, None)
            continue
        candidate_resolved = candidate_descriptor is None
        if candidate_descriptor is not None:
            if candidate_descriptor in _POISONED_PROBE_DESCRIPTOR_NUMBERS:
                diagnostics.append(
                    f"retained candidate {candidate_descriptor} is poisoned "
                    "and was not retried"
                )
                candidate_resolved = True
                same_generation = None
                comparison_diagnostic = None
            else:
                same_generation, comparison_diagnostic = (
                    _same_open_file_description(
                        candidate_descriptor, guard_descriptor
                    )
                )
            if comparison_diagnostic is not None:
                diagnostics.append(
                    f"retained guard {guard_descriptor}: {comparison_diagnostic}"
                )
            if candidate_resolved:
                pass
            elif same_generation is True:
                candidate_resolved, raw_diagnostic = _raw_close_owned_descriptor(
                    candidate_descriptor
                )
                if raw_diagnostic is not None:
                    diagnostics.append(
                        f"retained candidate {candidate_descriptor}: "
                        f"{raw_diagnostic}"
                    )
                if not candidate_resolved:
                    post_same, post_diagnostic = _same_open_file_description(
                        candidate_descriptor, guard_descriptor
                    )
                    if post_diagnostic is not None:
                        diagnostics.append(
                            f"retained post-close guard {guard_descriptor}: "
                            f"{post_diagnostic}"
                        )
                    if post_same is True:
                        diagnostics.append(
                            f"retained candidate {candidate_descriptor} remains open"
                        )
                        continue
                    if post_same is False:
                        candidate_resolved = True
                    else:
                        _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(
                            candidate_descriptor
                        )
                        candidate_resolved = True
                        diagnostics.append(
                            f"retained candidate {candidate_descriptor} close is "
                            "ambiguous and poisoned"
                        )
            elif same_generation is False:
                candidate_resolved = True
                diagnostics.append(
                    f"retained candidate number {candidate_descriptor} was "
                    "closed or reused; replacement was not closed"
                )
        if not candidate_resolved:
            diagnostics.append(
                f"retained candidate {candidate_descriptor} remains unresolved"
            )
            continue

        if candidate_descriptor is not None:
            _disarm_native_owned_descriptor(candidate_descriptor)

        guard_closed, guard_diagnostic = _raw_close_owned_descriptor(
            guard_descriptor
        )
        if guard_diagnostic is not None:
            diagnostics.append(
                f"retained generation guard {guard_descriptor}: "
                f"{guard_diagnostic}"
            )
        _RETAINED_PROBE_DESCRIPTORS.pop(guard_descriptor, None)
        if not guard_closed:
            _POISONED_PROBE_DESCRIPTOR_NUMBERS.add(guard_descriptor)
            diagnostics.append(
                f"retained generation guard {guard_descriptor} close is "
                "ambiguous and poisoned"
            )
    return diagnostics


def _pidfd_send_signal(descriptor: int, number: int) -> None:
    stdlib_function = getattr(signal, "pidfd_send_signal", None)
    if callable(stdlib_function):
        stdlib_function(descriptor, number, None, 0)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "pidfd_send_signal", None)
    if function is not None:
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(descriptor, number, None, 0)
    else:
        if platform.machine().lower() not in {
            "aarch64", "amd64", "arm64", "ppc64", "ppc64le", "riscv64", "s390x", "x86_64"
        }:
            raise OSError(errno.ENOSYS, "no verified pidfd_send_signal syscall ABI")
        libc.syscall.restype = ctypes.c_long
        result = libc.syscall(SYS_PIDFD_SEND_SIGNAL, descriptor, number, 0, 0)
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _signal_pidfds(handles: dict[int, int], number: int) -> list[str]:
    errors: list[str] = []
    for pid, descriptor in handles.items():
        try:
            _pidfd_send_signal(descriptor, number)
        except ProcessLookupError:
            continue
        except OSError as exc:
            errors.append(f"pidfd signal {number} failed for {pid}: {exc}")
    return errors


def _snapshot_pidfds(
    baseline: set[int], *, do_not_reap: set[int] | None = None
) -> dict[int, int]:
    handles: dict[int, int] = {}
    for pid in sorted(_process_descendants(os.getpid()) - baseline):
        try:
            descriptor = _pidfd_open(pid)
        except ProcessLookupError:
            continue
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        if poller.poll(0):
            _close_native_owned_descriptor_once("finished descendant pidfd", descriptor)
            if do_not_reap is None or pid not in do_not_reap:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
        else:
            handles[pid] = descriptor
    return handles


def _close_pidfds(handles: dict[int, int]) -> None:
    for descriptor in handles.values():
        try:
            _close_native_owned_descriptor_once("descendant pidfd", descriptor)
        except BaseException:
            pass


def _contain_process_tree(
    process: subprocess.Popen[str], baseline: set[int], *, terminate: bool
) -> tuple[set[int], list[str]]:
    tracked: dict[int, int] = {}
    errors: list[str] = []

    def merge() -> int:
        added = 0
        for pid, descriptor in _snapshot_pidfds(
            baseline, do_not_reap={process.pid}
        ).items():
            if pid in tracked:
                _close_native_owned_descriptor_once(
                    "duplicate descendant pidfd", descriptor
                )
            else:
                tracked[pid] = descriptor
                added += 1
        return added

    merge()
    if terminate:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        errors.extend(_signal_pidfds(tracked, signal.SIGTERM))
        time.sleep(0.05)
    if terminate:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + PROCESS_GRACE_SECONDS
    empty_rounds = 0
    while time.monotonic() < deadline:
        added = merge()
        errors.extend(_signal_pidfds(tracked, signal.SIGKILL))
        active: set[int] = set()
        for pid, descriptor in tracked.items():
            poller = select.poll()
            poller.register(descriptor, select.POLLIN)
            if not poller.poll(0):
                active.add(pid)
        if not active and added == 0:
            empty_rounds += 1
            if empty_rounds >= 2:
                break
        else:
            empty_rounds = 0
        time.sleep(0.01)
    survivors: set[int] = set()
    for pid, descriptor in tracked.items():
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        if not poller.poll(0):
            survivors.add(pid)
        if pid != process.pid:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
    _close_pidfds(tracked)
    return survivors, errors


def _observe_runtime_processes(
    root_pid: int,
    seen: set[tuple[int, int, str]],
    hash_cache: dict[tuple[int, ...], str],
) -> tuple[list[dict[str, Any]], int]:
    call_started_ns = time.monotonic_ns()
    observations: list[dict[str, Any]] = []
    process_count = 0
    for pid in sorted({root_pid} | _process_descendants(root_pid)):
        try:
            stat_text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
            close = stat_text.rfind(")")
            stat_fields = stat_text[close + 2 :].split()
            parent_pid = int(stat_fields[1])
            start_time = int(stat_fields[19])
            command_line = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            maps_lines = pathlib.Path(f"/proc/{pid}/maps").read_text(
                encoding="utf-8", errors="surrogateescape"
            ).splitlines()
            executable_path = pathlib.Path(f"/proc/{pid}/exe").resolve(strict=True)
            executable_descriptor = os.open(
                f"/proc/{pid}/exe",
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                executable_info = os.fstat(executable_descriptor)
                executable = {
                    "label": f"runtime process {pid} executable",
                    "path": str(executable_path),
                    "size_bytes": executable_info.st_size,
                    "sha256": None,
                    "fingerprint": {
                        "device": executable_info.st_dev,
                        "inode": executable_info.st_ino,
                        "mode": executable_info.st_mode,
                        "mtime_ns": executable_info.st_mtime_ns,
                        "ctime_ns": executable_info.st_ctime_ns,
                    },
                }
            finally:
                os.close(executable_descriptor)
        except (FileNotFoundError, ProcessLookupError):
            continue
        process_count += 1
        mapping_by_identity: dict[tuple[int, int], dict[str, Any]] = {}
        try:
            for line in maps_lines:
                fields = line.split(maxsplit=5)
                if len(fields) != 6 or not fields[5].startswith("/"):
                    continue
                path_text = re.sub(
                    r"\\([0-7]{3})",
                    lambda match: chr(int(match.group(1), 8)),
                    fields[5],
                )
                clean_path_text = path_text.removesuffix(" (deleted)")
                clean_path = pathlib.Path(clean_path_text)
                name = clean_path.name.lower()
                relevant = (
                    clean_path == executable_path
                    or name in {"mpiexec", "prte", "prted"}
                    or "gpu-step-db" in name
                    or "ld-linux" in name
                    or "ld.so" in name
                    or any(
                        _matches_runtime_soname(clean_path, family)
                        for family in (
                            "libmeep.so", "libmpi", "libpmix", "libcuda.so",
                            "libcudart.so",
                        )
                    )
                )
                if not relevant:
                    continue
                if path_text.endswith(" (deleted)"):
                    raise EvidenceError(
                        f"runtime process {pid} has a deleted relevant executable/library mapping"
                    )
                major_text, minor_text = fields[3].split(":", 1)
                mapped_device = os.makedev(int(major_text, 16), int(minor_text, 16))
                mapped_inode = int(fields[4])
                path = pathlib.Path(path_text).resolve(strict=True)
                descriptor = os.open(
                    path,
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    info = os.fstat(descriptor)
                    if info.st_dev != mapped_device or info.st_ino != mapped_inode:
                        raise EvidenceError(
                            f"runtime process {pid} mapping pathname differs from mapped inode"
                        )
                    identity = (info.st_dev, info.st_ino)
                    segment = {
                        "permissions": fields[1],
                        "offset": int(fields[2], 16),
                    }
                    if identity in mapping_by_identity:
                        mapping_by_identity[identity]["segments"].append(segment)
                        continue
                    mapping_by_identity[identity] = {
                            "path": str(path),
                            "device": info.st_dev,
                            "inode": info.st_ino,
                            "size_bytes": info.st_size,
                            "mtime_ns": info.st_mtime_ns,
                            "ctime_ns": info.st_ctime_ns,
                            "sha256": None,
                            "segments": [segment],
                        }
                finally:
                    os.close(descriptor)
        except (FileNotFoundError, ProcessLookupError):
            continue
        mappings = list(mapping_by_identity.values())
        decoded_command = [item.decode("utf-8", errors="replace") for item in command_line if item]
        epoch = hashlib.sha256(
            json.dumps(
                {
                    "command_line": decoded_command,
                    "executable": executable,
                    "mappings": [
                        (item["device"], item["inode"], item["sha256"])
                        for item in sorted(mappings, key=lambda item: item["path"])
                    ],
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        coordinate = (pid, start_time, epoch)
        if coordinate in seen:
            continue
        seen.add(coordinate)
        observations.append(
            {
                "pid": pid,
                "parent_pid": parent_pid,
                "start_time_ticks": start_time,
                "command_line": decoded_command,
                "mapping_epoch_sha256": epoch,
                "executable": executable,
                "observed_at_monotonic_ns": call_started_ns,
                "mappings": sorted(mappings, key=lambda item: item["path"]),
            }
        )
    call_duration_ns = time.monotonic_ns() - call_started_ns
    for observation in observations:
        observation["observation_duration_ns"] = call_duration_ns
    return observations, process_count


def _proc_stat_cpu_counters(cpus: Iterable[int]) -> dict[int, dict[str, int]]:
    requested = set(cpus)
    result: dict[int, dict[str, int]] = {}
    for line in pathlib.Path("/proc/stat").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"cpu[0-9]+", fields[0]):
            continue
        cpu = int(fields[0][3:])
        if cpu not in requested:
            continue
        values = [int(value) for value in fields[1:]]
        if len(values) < 5:
            raise EvidenceError("/proc/stat CPU record is incomplete")
        result[cpu] = {
            "total_ticks": sum(values),
            "idle_ticks": values[3] + values[4],
        }
    if set(result) != requested:
        raise EvidenceError("selected CPU disappeared from /proc/stat")
    return result


def _parse_linux_cpu_list(text: str) -> set[int]:
    result: set[int] = set()
    for token in text.strip().split(","):
        if not token:
            continue
        bounds = token.split("-", 1)
        if not all(bound.isdecimal() for bound in bounds):
            raise EvidenceError("Linux CPU-list syntax is invalid")
        first = int(bounds[0])
        last = int(bounds[-1])
        if last < first:
            raise EvidenceError("Linux CPU-list range is backwards")
        result.update(range(first, last + 1))
    if not result:
        raise EvidenceError("Linux CPU-list is empty")
    return result


def _canonical_linux_cpu_list(cpus: Iterable[int]) -> str:
    ordered = sorted(set(cpus))
    if not ordered or any(not _is_plain_int(cpu) or cpu < 0 for cpu in ordered):
        raise EvidenceError("Linux CPU-list values are invalid")
    ranges: list[str] = []
    first = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = cpu
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def _physical_cpu_core_plan(
    ranks: int, allowed_override: set[int] | None = None
) -> dict[str, Any]:
    allowed = set(os.sched_getaffinity(0)) if allowed_override is None else set(allowed_override)
    groups: dict[tuple[int, int], set[int]] = {}
    topology: list[dict[str, Any]] = []
    for cpu in sorted(allowed):
        root = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((root / "physical_package_id").read_text().strip())
            core = int((root / "core_id").read_text().strip())
            siblings = _parse_linux_cpu_list(
                (root / "thread_siblings_list").read_text()
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise EvidenceError("cannot resolve physical CPU topology") from exc
        if not siblings <= allowed:
            continue
        groups.setdefault((package, core), set()).update(siblings)
    for (package, core), siblings in sorted(groups.items()):
        topology.append(
            {
                "package_id": package,
                "core_id": core,
                "logical_cpus": sorted(siblings),
            }
        )
    if len(topology) < ranks + 1:
        raise EvidenceError("LDOS evidence needs distinct full physical cores for controller/ranks")
    chosen = topology[: ranks + 1]
    return {
        "allowed_cpus": sorted(allowed),
        "available_full_cores": topology,
        "controller_core": chosen[0],
        "worker_cores": chosen[1:],
    }


def _process_start_time_ticks(pid: int) -> int | None:
    try:
        text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    close = text.rfind(")")
    fields = text[close + 2 :].split()
    if close < 0 or len(fields) < 20:
        raise EvidenceError(f"process {pid} /proc/stat record is invalid")
    return int(fields[19])


def _process_task_cpu_state(pid: int) -> dict[str, Any] | None:
    task_root = pathlib.Path(f"/proc/{pid}/task")
    try:
        task_paths = sorted(task_root.iterdir(), key=lambda path: int(path.name))
    except (FileNotFoundError, ProcessLookupError):
        return None
    ticks = 0
    affinities: dict[str, list[int]] = {}
    for task_path in task_paths:
        tid = int(task_path.name)
        try:
            text = (task_path / "stat").read_text(encoding="ascii")
            affinity = sorted(os.sched_getaffinity(tid))
        except (FileNotFoundError, ProcessLookupError):
            return None
        close = text.rfind(")")
        fields = text[close + 2 :].split()
        if close < 0 or len(fields) < 13:
            raise EvidenceError(f"worker task {tid} /proc/stat record is invalid")
        ticks += int(fields[11]) + int(fields[12])
        affinities[str(tid)] = affinity
    return {
        "start_time_ticks": _process_start_time_ticks(pid),
        "ticks": ticks,
        "task_affinities": affinities,
    }


def _set_all_process_task_affinity(pid: int, mask: set[int]) -> dict[str, Any]:
    stable_snapshots = 0
    previous_tasks: set[str] | None = None
    for _attempt in range(8):
        state = _process_task_cpu_state(pid)
        if state is None:
            stable_snapshots = 0
            previous_tasks = None
            continue
        for tid_text in state["task_affinities"]:
            os.sched_setaffinity(int(tid_text), mask)
        verified = _process_task_cpu_state(pid)
        if verified is None:
            stable_snapshots = 0
            previous_tasks = None
            continue
        tasks = set(verified["task_affinities"])
        if all(
            set(value) == mask for value in verified["task_affinities"].values()
        ) and tasks == set(state["task_affinities"]):
            stable_snapshots = stable_snapshots + 1 if tasks == previous_tasks else 1
            previous_tasks = tasks
            if stable_snapshots >= 2:
                return verified
        else:
            stable_snapshots = 0
            previous_tasks = None
    raise EvidenceError("controller native task affinity did not reach a fixed point")


def _restore_process_task_affinities(
    pid: int, original: dict[str, list[int]], default_mask: set[int]
) -> None:
    stable_snapshots = 0
    previous_tasks: set[str] | None = None
    for _attempt in range(8):
        state = _process_task_cpu_state(pid)
        if state is None:
            stable_snapshots = 0
            previous_tasks = None
            continue
        for tid_text in state["task_affinities"]:
            desired = set(original.get(tid_text, sorted(default_mask)))
            try:
                os.sched_setaffinity(int(tid_text), desired)
            except ProcessLookupError:
                # A disappearing native thread is covered by the next coherent
                # snapshot; an arriving one receives the original process mask.
                continue
        verified = _process_task_cpu_state(pid)
        if verified is None:
            stable_snapshots = 0
            previous_tasks = None
            continue
        tasks = set(verified["task_affinities"])
        restored = all(
            set(mask) == set(original.get(tid_text, sorted(default_mask)))
            for tid_text, mask in verified["task_affinities"].items()
        )
        if restored and tasks == set(state["task_affinities"]):
            stable_snapshots = stable_snapshots + 1 if tasks == previous_tasks else 1
            previous_tasks = tasks
            if stable_snapshots >= 2:
                return
        else:
            stable_snapshots = 0
            previous_tasks = None
    raise EvidenceError("controller native task affinity restore did not reach a fixed point")


def _parse_pressure_file(name: str) -> dict[str, dict[str, float | int]]:
    path = pathlib.Path(f"/proc/pressure/{name}")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except FileNotFoundError as exc:
        raise EvidenceError(f"Linux {name} PSI telemetry is unavailable") from exc
    result: dict[str, dict[str, float | int]] = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {"some", "full"}:
            raise EvidenceError(f"Linux {name} PSI record is malformed")
        values: dict[str, float | int] = {}
        for field in fields[1:]:
            key, separator, value = field.partition("=")
            if not separator or key not in {"avg10", "avg60", "avg300", "total"}:
                raise EvidenceError(f"Linux {name} PSI field is malformed")
            try:
                values[key] = int(value) if key == "total" else float(value)
            except ValueError as exc:
                raise EvidenceError(f"Linux {name} PSI value is malformed") from exc
        if set(values) != {"avg10", "avg60", "avg300", "total"}:
            raise EvidenceError(f"Linux {name} PSI record is incomplete")
        result[fields[0]] = values
    required = {"some", "full"} if name != "cpu" else {"some"}
    if not required <= set(result):
        raise EvidenceError(f"Linux {name} PSI classes are incomplete")
    return result


class RawTelemetryJournal:
    """Append-only raw lane evidence which survives controller failures."""

    def __init__(self, path: pathlib.Path, header: dict[str, Any]) -> None:
        self.path = path
        self.sequence = 0
        self.closed = False
        self.write_poisoned = False
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        self.descriptor = os.open(path, flags, 0o600)
        try:
            self.append("header", header)
            os.fsync(self.descriptor)
            self._sync_parent_directory()
        except BaseException:
            os.close(self.descriptor)
            self.closed = True
            raise

    def _sync_parent_directory(self) -> None:
        descriptor = os.open(
            self.path.parent,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append(self, record_type: str, payload: dict[str, Any]) -> None:
        if self.closed:
            raise EvidenceError("telemetry journal is already closed")
        record = {
            "schema_version": 1,
            "sequence": self.sequence,
            "record_type": record_type,
            "payload": payload,
        }
        data = (
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        record_offset = os.lseek(self.descriptor, 0, os.SEEK_CUR)
        offset = 0
        try:
            while offset < len(data):
                written = os.write(self.descriptor, data[offset:])
                if written <= 0:
                    raise EvidenceError("telemetry journal write made no progress")
                offset += written
        except BaseException as write_error:
            try:
                os.ftruncate(self.descriptor, record_offset)
                os.lseek(self.descriptor, record_offset, os.SEEK_SET)
            except BaseException as rollback_error:
                self.write_poisoned = True
                write_error.add_note(
                    "telemetry rollback raised "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
            raise
        self.sequence += 1

    def close(self, state: str, error: str | None = None) -> dict[str, Any]:
        if not self.closed:
            try:
                if not self.write_poisoned:
                    self.append(
                        "terminal",
                        {"state": state, "error": error, "closed_at_utc": _utc_now()},
                    )
                os.fdatasync(self.descriptor)
            finally:
                os.close(self.descriptor)
                self.closed = True
            self._sync_parent_directory()
        handle = StableFile.open("closed raw telemetry journal", self.path)
        try:
            return handle.record()
        finally:
            handle.close()


def _nvml_archive_content_identity(archive: StableFile) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "relative_path": "../archive/system/lib/libnvidia-ml.so.1",
        "size_bytes": archive.size_bytes,
        "sha256": archive.sha256,
        "elf_nvml_identity": _nvml_elf_identity(
            archive.descriptor, archive.size_bytes
        ),
    }


def _validate_archived_nvml_identity(
    journal_path: pathlib.Path, identity: Any
) -> bool:
    """Verify portable fixed-name archive bytes without executing them."""

    journal_path = pathlib.Path(journal_path)
    if not (
        isinstance(identity, dict)
        and set(identity)
        == {
            "schema_version", "relative_path", "size_bytes", "sha256",
            "elf_nvml_identity",
        }
        and _is_plain_int(identity["schema_version"])
        and identity["schema_version"] == 1
        and identity["relative_path"]
        == "../archive/system/lib/libnvidia-ml.so.1"
        and _is_plain_int(identity["size_bytes"])
        and identity["size_bytes"] > 0
        and isinstance(identity["sha256"], str)
        and SHA256_PATTERN.fullmatch(identity["sha256"])
        and isinstance(identity["elf_nvml_identity"], dict)
    ):
        return False
    archive_path = (
        journal_path.parent.parent
        / "archive" / "system" / "lib" / "libnvidia-ml.so.1"
    )
    archive_directories = (
        journal_path.parent.parent,
        journal_path.parent.parent / "archive",
        journal_path.parent.parent / "archive" / "system",
        journal_path.parent.parent / "archive" / "system" / "lib",
    )
    for index, directory in enumerate(archive_directories):
        try:
            info = directory.lstat()
        except OSError:
            return False
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o002
            or index == 0 and info.st_mode & 0o020
        ):
            return False
    try:
        named_before = archive_path.lstat()
    except OSError:
        return False
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(archive_path, flags)
    except OSError:
        return False
    try:
        opened = os.fstat(descriptor)
        digest = _sha256_fd(descriptor)
        opened_after = os.fstat(descriptor)
        try:
            named_after = archive_path.lstat()
        except OSError:
            return False
        return bool(
            stat.S_ISREG(opened.st_mode)
            and stat.S_IMODE(opened.st_mode) == 0o400
            and opened.st_nlink == 1
            and opened.st_uid == os.geteuid()
            and _fingerprint(named_before) == _fingerprint(opened)
            and _fingerprint(named_after) == _fingerprint(opened)
            and _fingerprint(opened_after) == _fingerprint(opened)
            and opened.st_size == identity["size_bytes"]
            and digest == identity["sha256"]
            and _nvml_elf_identity(descriptor, opened.st_size)
            == identity["elf_nvml_identity"]
        )
    except EvidenceError:
        return False
    finally:
        os.close(descriptor)


def _validate_journal_interval_sequence(rows: Iterable[dict[str, Any]]) -> None:
    """Validate the controller's one-writer timestamped causal sequence."""

    previous_interval_end_ns: int | None = None
    for row in rows:
        record_type = row["record_type"]
        payload = row["payload"]
        if record_type == "sample":
            started_ns = payload["sample_started_monotonic_ns"]
            finished_ns = payload["sample_finished_monotonic_ns"]
            if (
                payload["monotonic_ns"] != started_ns
                or finished_ns < started_ns
                or previous_interval_end_ns is not None
                and started_ns < previous_interval_end_ns
            ):
                raise EvidenceError(
                    "raw telemetry sample/event intervals overlap or move backwards"
                )
            previous_interval_end_ns = finished_ns
            continue
        event_ns: int | None = None
        if record_type in {"go", "release", "worker-exit", "process-exit"}:
            event_ns = payload["monotonic_ns"]
        elif record_type == "results-ready":
            event_ns = payload["observed_monotonic_ns"]
        if event_ns is None:
            continue
        if (
            not _is_plain_int(event_ns)
            or not 0 < event_ns <= (1 << 64) - 1
            or previous_interval_end_ns is not None
            and event_ns < previous_interval_end_ns
        ):
            raise EvidenceError(
                "raw telemetry sample/event intervals overlap or move backwards"
            )
        previous_interval_end_ns = event_ns


def validate_lane_journal(
    path: pathlib.Path,
    *,
    expected_header: dict[str, Any] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Independently derive all host-lane gates from closed raw JSONL."""

    path = pathlib.Path(path)
    try:
        named_before = path.lstat()
    except OSError as exc:
        raise EvidenceError("raw telemetry journal path is unavailable") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(
            "raw telemetry journal changed or is a symbolic link"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or _fingerprint(opened) != _fingerprint(named_before)
        ):
            raise EvidenceError("raw telemetry journal metadata is invalid")
        data = os.pread(descriptor, opened.st_size + 1, 0)
        digest = _sha256_fd(descriptor)
        try:
            named_after = path.lstat()
        except OSError as exc:
            raise EvidenceError("raw telemetry journal disappeared while reading") from exc
        if (
            len(data) != opened.st_size
            or hashlib.sha256(data).hexdigest() != digest
            or _fingerprint(os.fstat(descriptor)) != _fingerprint(opened)
            or _fingerprint(named_after) != _fingerprint(opened)
        ):
            raise EvidenceError("raw telemetry journal changed while reading")
    finally:
        os.close(descriptor)
    if not data or not data.endswith(b"\n"):
        raise EvidenceError("raw telemetry journal is truncated")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError("raw telemetry journal is not UTF-8") from exc
    rows: list[dict[str, Any]] = []

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    for sequence, line in enumerate(text.splitlines()):
        try:
            row = json.loads(line, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise EvidenceError("raw telemetry journal contains invalid JSON") from exc
        if (
            not isinstance(row, dict)
            or set(row) != {"schema_version", "sequence", "record_type", "payload"}
            or not _is_plain_int(row["schema_version"])
            or row["schema_version"] != 1
            or not _is_plain_int(row["sequence"])
            or row["sequence"] != sequence
            or not isinstance(row["record_type"], str)
            or not isinstance(row["payload"], dict)
        ):
            raise EvidenceError("raw telemetry journal sequence/schema is invalid")
        canonical = json.dumps(row, separators=(",", ":"), sort_keys=True)
        if canonical != line:
            raise EvidenceError("raw telemetry journal record is not canonical")
        rows.append(row)
    terminals = [row for row in rows if row["record_type"] == "terminal"]
    if rows[0]["record_type"] != "header":
        raise EvidenceError("raw telemetry journal does not begin with its header")
    if require_complete:
        if (
            len(rows) < 4
            or len(terminals) != 1
            or rows[-1] is not terminals[0]
            or terminals[0]["payload"].get("state") != "sealed"
            or terminals[0]["payload"].get("error") is not None
        ):
            raise EvidenceError("raw telemetry journal has no unique complete terminal")
        if set(terminals[0]["payload"]) != {"state", "error", "closed_at_utc"}:
            raise EvidenceError("raw telemetry journal terminal schema is invalid")
        if not isinstance(terminals[0]["payload"]["closed_at_utc"], str) or not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z",
            terminals[0]["payload"]["closed_at_utc"],
        ):
            raise EvidenceError("raw telemetry journal terminal time is invalid")
    elif terminals:
        raise EvidenceError("open raw telemetry journal already has a terminal")
    header = rows[0]["payload"]
    required_header = {
        "nonce", "ranks", "controller_cpu_mask", "worker_cpu_masks",
        "expected_device_uuids", "online_cpu_list", "sample_period_ns",
        "allowed_cpus", "physical_topology_plan",
        "nvml_archive_identity",
        "nvml_baseline_process_utilization",
        "nvml_baseline_current_processes",
        "nvml_device_uuids",
        "controller_pid", "controller_start_time_ticks",
        "departed_nvml_lag_limit_ns",
        "nvml_absence_observation_to_pidfd_observation_limit_ns",
    }
    if set(header) != required_header or expected_header is not None and header != expected_header:
        raise EvidenceError("raw telemetry journal header differs from lane plan")

    ranks = header["ranks"]
    if (
        not _is_plain_int(ranks)
        or ranks not in (1, 2)
        or not isinstance(header["nonce"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", header["nonce"])
        or not _is_plain_int(header["controller_pid"])
        or header["controller_pid"] <= 0
        or not _is_plain_int(header["controller_start_time_ticks"])
        or header["controller_start_time_ticks"] <= 0
        or not _is_plain_int(header["sample_period_ns"])
        or header["sample_period_ns"] != 50_000_000
        or not _is_plain_int(header["departed_nvml_lag_limit_ns"])
        or header["departed_nvml_lag_limit_ns"] != NVML_DEPARTED_CONTEXT_LAG_NS
        or not _is_plain_int(
            header["nvml_absence_observation_to_pidfd_observation_limit_ns"]
        )
        or header["nvml_absence_observation_to_pidfd_observation_limit_ns"]
        != NVML_ABSENCE_OBSERVATION_TO_PIDFD_OBSERVATION_NS
        or not _validate_archived_nvml_identity(
            path, header["nvml_archive_identity"]
        )
        or not isinstance(header["controller_cpu_mask"], list)
        or not header["controller_cpu_mask"]
        or header["controller_cpu_mask"] != sorted(set(header["controller_cpu_mask"]))
        or any(
            not _is_plain_int(cpu) or cpu < 0
            for cpu in header["controller_cpu_mask"]
        )
        or not isinstance(header["worker_cpu_masks"], list)
        or len(header["worker_cpu_masks"]) != ranks
        or any(
            not isinstance(mask, list)
            or not mask
            or mask != sorted(set(mask))
            or any(not _is_plain_int(cpu) or cpu < 0 for cpu in mask)
            for mask in header["worker_cpu_masks"]
        )
        or [value.removeprefix("GPU-").replace("-", "") for value in header["nvml_device_uuids"]]
        != header["expected_device_uuids"]
        or any(header["nvml_baseline_process_utilization"].get(value) for value in header["nvml_device_uuids"])
        or any(
            any(rows for rows in header["nvml_baseline_current_processes"].get(value, {}).values())
            for value in header["nvml_device_uuids"]
        )
    ):
        raise EvidenceError("raw telemetry journal header cardinality is invalid")
    nvml_uuids = header["nvml_device_uuids"]
    expected_uuids = header["expected_device_uuids"]
    if (
        not isinstance(nvml_uuids, list)
        or len(nvml_uuids) != 2
        or len(set(nvml_uuids)) != 2
        or any(not isinstance(value, str) or not UUID_PATTERN.fullmatch(value) for value in nvml_uuids)
        or not isinstance(expected_uuids, list)
        or len(expected_uuids) != 2
        or len(set(expected_uuids)) != 2
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value)
            for value in expected_uuids
        )
        or set(header["nvml_baseline_process_utilization"]) != set(nvml_uuids)
        or any(
            not isinstance(header["nvml_baseline_process_utilization"][uuid_value], list)
            or header["nvml_baseline_process_utilization"][uuid_value]
            for uuid_value in nvml_uuids
        )
        or set(header["nvml_baseline_current_processes"]) != set(nvml_uuids)
        or any(
            not isinstance(header["nvml_baseline_current_processes"][uuid_value], dict)
            or set(header["nvml_baseline_current_processes"][uuid_value])
            != {"compute", "graphics", "mps"}
            or any(
                not isinstance(rows, list) or rows
                for rows in header["nvml_baseline_current_processes"][uuid_value].values()
            )
            for uuid_value in nvml_uuids
        )
    ):
        raise EvidenceError("raw telemetry journal two-GPU baseline contract is invalid")
    allowed_record_types = {
        "header", "ready", "sample", "go", "results-ready", "release",
        "worker-exit", "process-exit", "terminal"
    }
    if (
        any(row["record_type"] not in allowed_record_types for row in rows)
        or sum(row["record_type"] == "header" for row in rows) != 1
    ):
        raise EvidenceError("raw telemetry journal contains an unknown record type")
    ready_rows = [row["payload"] for row in rows if row["record_type"] == "ready"]
    results_ready_events = [
        row for row in rows if row["record_type"] == "results-ready"
    ]
    results_ready_rows = [row["payload"] for row in results_ready_events]
    go_rows = [row["payload"] for row in rows if row["record_type"] == "go"]
    release_events = [row for row in rows if row["record_type"] == "release"]
    release_rows = [row["payload"] for row in release_events]
    exit_rows = [row["payload"] for row in rows if row["record_type"] == "process-exit"]
    worker_exit_events = [
        row for row in rows if row["record_type"] == "worker-exit"
    ]
    worker_exit_rows = [row["payload"] for row in worker_exit_events]
    sample_rows = [row for row in rows if row["record_type"] == "sample"]
    timeline = [row["payload"] for row in sample_rows]
    if (
        len(ready_rows) != ranks
        or {record.get("rank") for record in ready_rows} != set(range(ranks))
        or len(go_rows) != 1
        or len(results_ready_rows) != ranks
        or {record.get("rank") for record in results_ready_rows}
        != set(range(ranks))
        or len(release_rows) != 1
        or go_rows[0].get("clean_affinity_polls", 0) < 2
        or go_rows[0].get("clean_gpu_polls", 0) < 2
        or len(exit_rows) != 1
        or len(timeline) < 2
    ):
        raise EvidenceError("raw telemetry journal ready/GO coverage is incomplete")
    go_sequence = next(row["sequence"] for row in rows if row["record_type"] == "go")
    first_sample_sequence = sample_rows[0]["sequence"]
    release_sequence = release_events[0]["sequence"]
    exit_sequence = next(
        row["sequence"] for row in rows if row["record_type"] == "process-exit"
    )
    if (
        any(
            row["sequence"] >= first_sample_sequence
            for row in rows if row["record_type"] == "ready"
        )
        or any(
            event["sequence"] <= go_sequence
            or event["sequence"] >= release_sequence
            for event in results_ready_events
        )
        or release_sequence <= max(
            event["sequence"] for event in results_ready_events
        )
        or exit_sequence <= go_sequence
        or any(
            event["sequence"] <= release_sequence
            or event["sequence"] >= exit_sequence
            for event in worker_exit_events
        )
        or any(
            not isinstance(sample.get("go_released"), bool)
            or not isinstance(sample.get("after_process_exit"), bool)
            for sample in timeline
        )
    ):
        raise EvidenceError("raw telemetry journal event ordering/types are invalid")
    ready_keys = {
        "schema_version", "rank", "pid", "start_time_ticks", "device_ordinal",
        "device_uuid", "nonce", "task_affinities", "file",
    }
    if any(set(record) != ready_keys for record in ready_rows):
        raise EvidenceError("raw telemetry ready schema is invalid")
    if set(go_rows[0]) != {
        "monotonic_ns", "clean_affinity_polls", "clean_gpu_polls"
    } or set(release_rows[0]) != {
        "monotonic_ns", "results_ready_ranks", "file"
    } or set(exit_rows[0]) != {"monotonic_ns"}:
        raise EvidenceError("raw telemetry GO/exit schema is invalid")
    if (
        not _is_plain_int(go_rows[0]["monotonic_ns"])
        or not 0 < go_rows[0]["monotonic_ns"] <= (1 << 64) - 1
        or not _is_plain_int(go_rows[0]["clean_affinity_polls"])
        or go_rows[0]["clean_affinity_polls"] < 2
        or not _is_plain_int(go_rows[0]["clean_gpu_polls"])
        or go_rows[0]["clean_gpu_polls"] < 2
        or not _is_plain_int(exit_rows[0]["monotonic_ns"])
        or not 0 < exit_rows[0]["monotonic_ns"] <= (1 << 64) - 1
        or not _is_plain_int(release_rows[0]["monotonic_ns"])
        or not 0 < release_rows[0]["monotonic_ns"] <= (1 << 64) - 1
        or release_rows[0]["monotonic_ns"] <= go_rows[0]["monotonic_ns"]
        or release_rows[0]["results_ready_ranks"] != list(range(ranks))
        or any(
            not _is_plain_int(rank)
            for rank in release_rows[0]["results_ready_ranks"]
        )
        or exit_rows[0]["monotonic_ns"] <= release_rows[0]["monotonic_ns"]
    ):
        raise EvidenceError("raw telemetry GO/exit values are invalid")
    if (
        len(worker_exit_rows) != ranks
        or {record.get("rank") for record in worker_exit_rows} != set(range(ranks))
        or any(
            set(record) != {"rank", "pid", "start_time_ticks", "monotonic_ns"}
            for record in worker_exit_rows
        )
    ):
        raise EvidenceError("raw worker-exit coverage/schema is invalid")
    worker_exit_times: list[int] = []
    for record in worker_exit_rows:
        ready = next(item for item in ready_rows if item["rank"] == record["rank"])
        if (
            not _is_plain_int(record["rank"])
            or not _is_plain_int(record["pid"])
            or record["pid"] <= 0
            or not _is_plain_int(record["start_time_ticks"])
            or record["start_time_ticks"] <= 0
            or record["pid"] != ready["pid"]
            or record["start_time_ticks"] != ready["start_time_ticks"]
            or not _is_plain_int(record["monotonic_ns"])
            or not 0 < record["monotonic_ns"] <= (1 << 64) - 1
            or record["monotonic_ns"] < release_rows[0]["monotonic_ns"]
            or record["monotonic_ns"] > exit_rows[0]["monotonic_ns"]
        ):
            raise EvidenceError("raw worker-exit identity/time differs from ready")
        worker_exit_times.append(record["monotonic_ns"])
    if worker_exit_times != sorted(worker_exit_times):
        raise EvidenceError("raw worker-exit timestamps moved backwards")
    sample_keys = {
        "monotonic_ns", "scheduled_monotonic_ns", "sample_started_monotonic_ns",
        "sample_finished_monotonic_ns", "go_released", "descendant_pids",
        "worker_process_state", "process_task_state", "cpu_counters", "current_frequency_khz",
        "controller_process_state",
        "governors", "loadavg", "pressure", "nvml", "after_process_exit",
        "departed_worker_ranks",
    }
    if any(set(sample) != sample_keys for sample in timeline):
        raise EvidenceError("raw telemetry sample schema is invalid")
    timestamps = [sample.get("monotonic_ns") for sample in timeline]
    if any(
        not _is_plain_int(value) or not 0 < value <= (1 << 64) - 1
        for value in timestamps
    ):
        raise EvidenceError("raw telemetry journal sample time is invalid")
    if any(
        not _is_plain_int(sample.get(name))
        or not 0 < sample[name] <= (1 << 64) - 1
        for sample in timeline
        for name in (
            "scheduled_monotonic_ns", "sample_started_monotonic_ns",
            "sample_finished_monotonic_ns",
        )
    ):
        raise EvidenceError("raw telemetry journal sample interval is invalid")
    gaps_ns = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    maximum_gap_ns = max(gaps_ns, default=0)
    if not gaps_ns or any(gap <= 0 or gap > 100_000_000 for gap in gaps_ns):
        raise EvidenceError("raw telemetry journal sample gap exceeds 100 ms")
    scheduled = [sample["scheduled_monotonic_ns"] for sample in timeline]
    if any(not 0 < value <= (1 << 64) - 1 for value in scheduled) or any(
        current - previous != 50_000_000
        for previous, current in zip(scheduled, scheduled[1:])
    ):
        raise EvidenceError("raw telemetry cadence is not absolute 50 ms")
    ready_by_rank = {record["rank"]: record for record in ready_rows}
    for rank, record in ready_by_rank.items():
        if (
            not _is_plain_int(record.get("schema_version"))
            or record["schema_version"] != 1
            or not _is_plain_int(record.get("rank"))
            or record["rank"] != rank
            or not _is_plain_int(record.get("pid"))
            or record["pid"] <= 0
            or not _is_plain_int(record.get("start_time_ticks"))
            or record["start_time_ticks"] <= 0
            or not _is_plain_int(record.get("device_ordinal"))
            or record["device_ordinal"] != rank
            or not isinstance(record.get("device_uuid"), str)
            or record.get("device_uuid") != header["expected_device_uuids"][rank]
            or not isinstance(record.get("nonce"), str)
            or record.get("nonce") != header["nonce"]
        ):
            raise EvidenceError("raw telemetry ready rank/device/nonce binding differs")
    results_ready_by_rank = {
        record["rank"]: record for record in results_ready_rows
    }
    results_ready_keys = {
        "schema_version", "state", "rank", "pid", "start_time_ticks",
        "device_ordinal", "device_uuid", "nonce", "monotonic_ns",
        "observed_monotonic_ns", "file",
    }
    for rank, record in results_ready_by_rank.items():
        ready = ready_by_rank[rank]
        if (
            set(record) != results_ready_keys
            or not _is_plain_int(record["schema_version"])
            or record["schema_version"] != 1
            or record["state"] != "RESULTS_READY"
            or not _is_plain_int(record["rank"])
            or record["rank"] != rank
            or not _is_plain_int(record["pid"])
            or record["pid"] != ready["pid"]
            or not _is_plain_int(record["start_time_ticks"])
            or record["start_time_ticks"] != ready["start_time_ticks"]
            or not _is_plain_int(record["device_ordinal"])
            or record["device_ordinal"] != ready["device_ordinal"]
            or record["device_uuid"] != ready["device_uuid"]
            or record["nonce"] != header["nonce"]
            or not _is_plain_int(record["monotonic_ns"])
            or not 0 < record["monotonic_ns"] <= (1 << 64) - 1
            or not _is_plain_int(record["observed_monotonic_ns"])
            or not 0 < record["observed_monotonic_ns"] <= (1 << 64) - 1
            or not (
                go_rows[0]["monotonic_ns"] < record["monotonic_ns"]
                <= record["observed_monotonic_ns"]
                <= release_rows[0]["monotonic_ns"]
            )
        ):
            raise EvidenceError("raw RESULTS_READY rank/epoch/time binding differs")

    # The controller is the sole journal writer.  Its timestamped events and
    # sampling intervals therefore form one non-overlapping causal sequence.
    _validate_journal_interval_sequence(rows)
    worker_masks = [set(mask) for mask in header["worker_cpu_masks"]]
    controller_mask = set(header["controller_cpu_mask"])
    role_masks = [controller_mask, *worker_masks]
    if any(
        first & second
        for index, first in enumerate(role_masks)
        for second in role_masks[index + 1 :]
    ):
        raise EvidenceError("raw controller/worker CPU masks overlap")
    allowed_cpus = header["allowed_cpus"]
    plan = header["physical_topology_plan"]

    try:
        online_cpus = (
            _parse_linux_cpu_list(header["online_cpu_list"])
            if isinstance(header["online_cpu_list"], str)
            else set()
        )
        canonical_online_cpu_list = _canonical_linux_cpu_list(online_cpus)
    except EvidenceError:
        online_cpus = set()
        canonical_online_cpu_list = ""

    def exact_core_record(value: Any) -> bool:
        return bool(
            isinstance(value, dict)
            and set(value) == {"package_id", "core_id", "logical_cpus"}
            and _is_plain_int(value["package_id"])
            and value["package_id"] >= 0
            and _is_plain_int(value["core_id"])
            and value["core_id"] >= 0
            and isinstance(value["logical_cpus"], list)
            and value["logical_cpus"]
            and value["logical_cpus"] == sorted(set(value["logical_cpus"]))
            and all(
                _is_plain_int(cpu) and cpu >= 0 for cpu in value["logical_cpus"]
            )
        )

    if (
        not isinstance(allowed_cpus, list)
        or not allowed_cpus
        or allowed_cpus != sorted(set(allowed_cpus))
        or any(not _is_plain_int(cpu) or cpu < 0 for cpu in allowed_cpus)
        or not isinstance(header["online_cpu_list"], str)
        or header["online_cpu_list"] != canonical_online_cpu_list
        or not set(allowed_cpus) <= online_cpus
        or not set().union(*role_masks) <= set(allowed_cpus)
        or not isinstance(plan, dict)
        or set(plan)
        != {"allowed_cpus", "available_full_cores", "controller_core", "worker_cores"}
        or plan["allowed_cpus"] != allowed_cpus
        or not exact_core_record(plan["controller_core"])
        or set(plan["controller_core"]["logical_cpus"]) != controller_mask
        or not isinstance(plan["worker_cores"], list)
        or len(plan["worker_cores"]) < ranks
        or any(not exact_core_record(core) for core in plan["worker_cores"])
        or [set(core["logical_cpus"]) for core in plan["worker_cores"][:ranks]]
        != worker_masks
        or not isinstance(plan["available_full_cores"], list)
        or any(not exact_core_record(core) for core in plan["available_full_cores"])
        or plan["controller_core"] not in plan["available_full_cores"]
        or any(core not in plan["available_full_cores"] for core in plan["worker_cores"])
    ):
        raise EvidenceError("raw physical CPU topology plan is invalid")
    monitored_cpus = sorted(controller_mask.union(*worker_masks))
    monitored_cpu_keys = {str(cpu) for cpu in monitored_cpus}
    worker_pid_to_rank = {
        record["pid"]: rank for rank, record in ready_by_rank.items()
    }
    allowed_by_uuid = {
        uuid_value: {} for uuid_value in header["nvml_device_uuids"]
    }
    for rank, record in ready_by_rank.items():
        allowed_by_uuid[header["nvml_device_uuids"][rank]][record["pid"]] = (
            record["start_time_ticks"]
        )

    def exact_task_affinities(
        value: Any, *, leader_pid: int, expected_mask: set[int]
    ) -> bool:
        if not isinstance(value, dict) or not value or str(leader_pid) not in value:
            return False
        for tid_text, mask in value.items():
            if (
                not isinstance(tid_text, str)
                or not re.fullmatch(r"[1-9][0-9]*", tid_text)
                or not isinstance(mask, list)
                or mask != sorted(expected_mask)
                or any(not _is_plain_int(cpu) or cpu < 0 for cpu in mask)
            ):
                return False
        return True

    def exact_process_state(
        value: Any, *, leader_pid: int, expected_mask: set[int]
    ) -> bool:
        return bool(
            isinstance(value, dict)
            and set(value) == {"start_time_ticks", "ticks", "task_affinities"}
            and _is_plain_int(value["start_time_ticks"])
            and value["start_time_ticks"] > 0
            and _is_plain_int(value["ticks"])
            and value["ticks"] >= 0
            and exact_task_affinities(
                value["task_affinities"],
                leader_pid=leader_pid,
                expected_mask=expected_mask,
            )
        )

    def exact_pressure(value: Any) -> bool:
        def exact_pressure_class(record: Any) -> bool:
            return bool(
                isinstance(record, dict)
                and set(record) == {"avg10", "avg60", "avg300", "total"}
                and all(
                    type(record[name]) is float
                    and math.isfinite(record[name])
                    and 0.0 <= record[name] <= 100.0
                    for name in ("avg10", "avg60", "avg300")
                )
                and _is_plain_int(record["total"])
                and 0 <= record["total"] <= (1 << 64) - 1
            )

        if not isinstance(value, dict) or set(value) != {"cpu", "memory", "io"}:
            return False
        if (
            not isinstance(value["cpu"], dict)
            or set(value["cpu"]) not in ({"some"}, {"some", "full"})
            or not isinstance(value["memory"], dict)
            or set(value["memory"]) != {"some", "full"}
            or not isinstance(value["io"], dict)
            or set(value["io"]) != {"some", "full"}
        ):
            return False
        return all(
            exact_pressure_class(record)
            for pressure_file in value.values()
            for record in pressure_file.values()
        )

    def exact_ready_file(record: Any, rank: int) -> bool:
        if not isinstance(record, dict) or set(record) != {
            "label", "path", "size_bytes", "sha256", "fingerprint"
        }:
            return False
        fingerprint = record.get("fingerprint")
        expected_path = path.parent / f"rank-{rank}.ready"
        ready = ready_by_rank[rank]
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
        expected_bytes = json.dumps(
            ordered_ready, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        if not (
            record["label"] == f"LDOS rank {rank} ready record"
            and isinstance(record["path"], str)
            and pathlib.Path(record["path"]) == expected_path
            and _is_plain_int(record["size_bytes"])
            and record["size_bytes"] == len(expected_bytes)
            and isinstance(record["sha256"], str)
            and record["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
            and isinstance(fingerprint, dict)
            and set(fingerprint)
            == {"device", "inode", "mode", "mtime_ns", "ctime_ns"}
            and all(_is_plain_int(value) and value >= 0 for value in fingerprint.values())
            and fingerprint["device"] > 0
            and fingerprint["inode"] > 0
            and stat.S_ISREG(fingerprint["mode"])
            and stat.S_IMODE(fingerprint["mode"]) == 0o600
        ):
            return False
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(expected_path, flags)
        except OSError:
            return False
        try:
            opened = os.fstat(descriptor)
            actual_fingerprint = {
                "device": opened.st_dev,
                "inode": opened.st_ino,
                "mode": opened.st_mode,
                "mtime_ns": opened.st_mtime_ns,
                "ctime_ns": opened.st_ctime_ns,
            }
            if opened.st_size != len(expected_bytes):
                return False
            data = os.pread(descriptor, opened.st_size + 1, 0)
            try:
                named = expected_path.lstat()
            except OSError:
                return False
            opened_after = os.fstat(descriptor)
            try:
                named_after = expected_path.lstat()
            except OSError:
                return False
            return bool(
                _fingerprint(opened) == _fingerprint(named)
                and actual_fingerprint == fingerprint
                and opened.st_nlink == 1
                and opened.st_uid == os.geteuid()
                and opened.st_size == len(expected_bytes)
                and data == expected_bytes
                and hashlib.sha256(data).hexdigest() == record["sha256"]
                and _fingerprint(opened_after) == _fingerprint(opened)
                and _fingerprint(named_after) == _fingerprint(opened)
            )
        finally:
            os.close(descriptor)

    def exact_bound_file(
        record: Any, *, expected_path: pathlib.Path, expected_label: str,
        expected_bytes: bytes, expected_mode: int,
    ) -> bool:
        if not isinstance(record, dict) or set(record) != {
            "label", "path", "size_bytes", "sha256", "fingerprint"
        }:
            return False
        fingerprint = record.get("fingerprint")
        if not (
            record["label"] == expected_label
            and isinstance(record["path"], str)
            and pathlib.Path(record["path"]) == expected_path
            and record["size_bytes"] == len(expected_bytes)
            and record["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
            and isinstance(fingerprint, dict)
            and set(fingerprint)
            == {"device", "inode", "mode", "mtime_ns", "ctime_ns"}
            and all(_is_plain_int(value) and value >= 0 for value in fingerprint.values())
            and fingerprint["device"] > 0
            and fingerprint["inode"] > 0
            and stat.S_ISREG(fingerprint["mode"])
            and stat.S_IMODE(fingerprint["mode"]) == expected_mode
        ):
            return False
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(expected_path, flags)
        except OSError:
            return False
        try:
            opened = os.fstat(descriptor)
            try:
                named = expected_path.lstat()
            except OSError:
                return False
            actual_fingerprint = {
                "device": opened.st_dev,
                "inode": opened.st_ino,
                "mode": opened.st_mode,
                "mtime_ns": opened.st_mtime_ns,
                "ctime_ns": opened.st_ctime_ns,
            }
            if opened.st_size != len(expected_bytes):
                return False
            data = os.pread(descriptor, opened.st_size + 1, 0)
            return bool(
                _fingerprint(opened) == _fingerprint(named)
                and actual_fingerprint == fingerprint
                and opened.st_nlink == 1
                and opened.st_uid == os.geteuid()
                and opened.st_size == len(expected_bytes)
                and data == expected_bytes
                and _fingerprint(os.fstat(descriptor)) == _fingerprint(opened)
                and _fingerprint(expected_path.lstat()) == _fingerprint(opened)
            )
        finally:
            os.close(descriptor)

    for rank, record in ready_by_rank.items():
        if (
            not exact_task_affinities(
                record["task_affinities"],
                leader_pid=record["pid"],
                expected_mask=worker_masks[rank],
            )
            or not exact_ready_file(record["file"], rank)
        ):
            raise EvidenceError("raw telemetry ready task/file evidence is invalid")
    for rank, record in results_ready_by_rank.items():
        worker_payload = {
            key: record[key]
            for key in (
                "schema_version", "state", "rank", "pid", "start_time_ticks",
                "device_ordinal", "device_uuid", "nonce", "monotonic_ns",
            )
        }
        expected_bytes = json.dumps(
            worker_payload, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        if not exact_bound_file(
            record["file"],
            expected_path=path.parent / f"rank-{rank}.results-ready",
            expected_label=f"LDOS rank {rank} results-ready record",
            expected_bytes=expected_bytes,
            expected_mode=0o600,
        ):
            raise EvidenceError("raw RESULTS_READY file evidence is invalid")
    release_bytes = (
        json.dumps(
            {
                "schema_version": 1,
                "nonce": header["nonce"],
                "ranks": ranks,
                "state": "RELEASE",
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if not exact_bound_file(
        release_rows[0]["file"],
        expected_path=path.parent / "RELEASE",
        expected_label="LDOS RELEASE gate",
        expected_bytes=release_bytes,
        expected_mode=0o400,
    ):
        raise EvidenceError("raw RELEASE file evidence is invalid")
    nvml_sample_keys = {
        "scheduled_monotonic_ns", "started_monotonic_ns", "finished_monotonic_ns",
        "lateness_ns", "duration_ns", "devices", "outsiders",
    }
    nvml_device_keys = {
        "uuid", "process_lists", "process_utilization_since_cursor", "utilization",
        "graphics_clock_mhz", "memory_clock_mhz", "power_mw", "temperature_c",
        "pstate", "memory_bytes",
    }
    post_go_process_epochs: dict[int, int] = {}
    post_go_descendants: set[int] | None = None
    first_nvml_absence_by_rank: dict[int, int] = {}
    first_nvml_absence_sequence_by_rank: dict[int, int] = {}
    last_utilization_cursor = {
        uuid_value: 0 for uuid_value in header["nvml_device_uuids"]
    }

    def exact_affinity_sample(sample: dict[str, Any]) -> bool:
        descendants = sample.get("descendant_pids")
        states = sample.get("process_task_state")
        if not isinstance(descendants, list) or not isinstance(states, dict):
            return False
        if set(states) != {str(pid) for pid in descendants}:
            return False
        if not set(worker_pid_to_rank) <= set(descendants):
            return False
        for pid in descendants:
            state = states[str(pid)]
            rank = worker_pid_to_rank.get(pid)
            mask = controller_mask if rank is None else worker_masks[rank]
            if (
                not exact_process_state(
                    state, leader_pid=pid, expected_mask=mask
                )
            ):
                return False
            if rank is not None and state.get("start_time_ticks") != ready_by_rank[rank]["start_time_ticks"]:
                return False
        return True

    for sample_row in sample_rows:
        sample = sample_row["payload"]
        expected_departed_ranks = sorted(
            event["payload"]["rank"]
            for event in worker_exit_events
            if event["sequence"] < sample_row["sequence"]
        )
        descendants = sample["descendant_pids"]
        process_states = sample["process_task_state"]
        if (
            not isinstance(descendants, list)
            or descendants != sorted(set(descendants))
            or set(process_states) != {str(pid) for pid in descendants}
        ):
            raise EvidenceError("raw process/task state cardinality differs from descendants")
        controller_state = sample["controller_process_state"]
        if (
            not exact_process_state(
                controller_state,
                leader_pid=header["controller_pid"],
                expected_mask=controller_mask,
            )
            or controller_state["start_time_ticks"]
            != header["controller_start_time_ticks"]
        ):
            raise EvidenceError("raw controller native task affinity changed")
        departed_ranks = sample["departed_worker_ranks"]
        if (
            not isinstance(departed_ranks, list)
            or departed_ranks != sorted(set(departed_ranks))
            or any(not _is_plain_int(rank) or rank < 0 or rank >= ranks for rank in departed_ranks)
            or departed_ranks != expected_departed_ranks
            or sample["go_released"] != (sample_row["sequence"] > go_sequence)
            or sample["after_process_exit"] != (
                sample_row["sequence"] > exit_sequence
            )
            or any(
                sample["sample_started_monotonic_ns"]
                < next(
                    record["monotonic_ns"]
                    for record in worker_exit_rows if record["rank"] == rank
                )
                for rank in departed_ranks
            )
        ):
            raise EvidenceError("raw worker-exit rank state is invalid")
        if (
            not isinstance(sample["worker_process_state"], dict)
            or set(sample["worker_process_state"])
            != {str(rank) for rank in range(ranks)}
        ):
            raise EvidenceError("raw worker-state rank schema is invalid")
        for pid in descendants:
            state = process_states[str(pid)]
            rank = worker_pid_to_rank.get(pid)
            expected_mask = controller_mask if rank is None else worker_masks[rank]
            if state is None:
                if not sample["after_process_exit"]:
                    raise EvidenceError("raw live descendant has no task state")
            elif not exact_process_state(
                state, leader_pid=pid, expected_mask=expected_mask
            ):
                raise EvidenceError("raw descendant task state schema is invalid")
        for rank, ready_record in ready_by_rank.items():
            expected_worker_state = process_states.get(str(ready_record["pid"]))
            if sample["worker_process_state"][str(rank)] != expected_worker_state:
                raise EvidenceError("raw worker state differs from process task state")
        frequencies = sample["current_frequency_khz"]
        governors = sample["governors"]
        cpu_counters = sample["cpu_counters"]
        allowed_governors = {
            "performance", "powersave", "ondemand", "conservative",
            "schedutil", "userspace",
        }
        if (
            not isinstance(frequencies, dict)
            or set(frequencies) != monitored_cpu_keys
            or any(
                value is not None
                and (not _is_plain_int(value) or value < 0)
                for value in frequencies.values()
            )
            or not isinstance(governors, dict)
            or set(governors) != monitored_cpu_keys
            or any(
                value is not None and value not in allowed_governors
                for value in governors.values()
            )
            or not isinstance(sample["loadavg"], str)
            or not re.fullmatch(
                r"[0-9]+(?:\.[0-9]+)? [0-9]+(?:\.[0-9]+)? "
                r"[0-9]+(?:\.[0-9]+)? [0-9]+/[0-9]+ [0-9]+",
                sample["loadavg"],
            )
            or not isinstance(cpu_counters, dict)
            or set(cpu_counters) != monitored_cpu_keys
            or any(
                not isinstance(value, dict)
                or set(value) != {"total_ticks", "idle_ticks"}
                or any(not _is_plain_int(item) or item < 0 for item in value.values())
                or value["idle_ticks"] > value["total_ticks"]
                for value in cpu_counters.values()
            )
            or not exact_pressure(sample["pressure"])
        ):
            raise EvidenceError("raw host telemetry schema/value is invalid")
        if (
            sample_row["sequence"] < go_sequence
            and sample["sample_finished_monotonic_ns"] > go_rows[0]["monotonic_ns"]
        ) or (
            go_sequence < sample_row["sequence"] < exit_sequence
            and (
                sample["sample_started_monotonic_ns"] < go_rows[0]["monotonic_ns"]
                or sample["sample_finished_monotonic_ns"]
                > exit_rows[0]["monotonic_ns"]
            )
        ) or (
            sample_row["sequence"] < release_sequence
            and sample["sample_finished_monotonic_ns"]
            > release_rows[0]["monotonic_ns"]
        ) or (
            release_sequence < sample_row["sequence"] < exit_sequence
            and sample["sample_started_monotonic_ns"]
            < release_rows[0]["monotonic_ns"]
        ) or (
            sample_row["sequence"] > exit_sequence
            and sample["sample_started_monotonic_ns"]
            < exit_rows[0]["monotonic_ns"]
        ):
            raise EvidenceError("raw telemetry sample crossed an event timestamp")
        if sample.get("go_released") and not sample.get("after_process_exit"):
            current_descendants = set(descendants)
            if post_go_descendants is None:
                post_go_descendants = current_descendants
            expected_descendants = post_go_descendants - {
                ready_by_rank[rank]["pid"] for rank in departed_ranks
            }
            if current_descendants != expected_descendants:
                raise EvidenceError("raw post-GO process tree membership changed")
            for pid in descendants:
                state = process_states[str(pid)]
                rank = worker_pid_to_rank.get(pid)
                expected_mask = controller_mask if rank is None else worker_masks[rank]
                if (
                    not isinstance(state, dict)
                    or not _is_plain_int(state.get("start_time_ticks"))
                    or state["start_time_ticks"] <= 0
                    or not isinstance(state.get("task_affinities"), dict)
                    or not state["task_affinities"]
                    or any(
                        set(mask) != expected_mask
                        for mask in state["task_affinities"].values()
                    )
                ):
                    raise EvidenceError("raw controller/worker task affinity is invalid")
                if rank is not None and state["start_time_ticks"] != ready_by_rank[rank]["start_time_ticks"]:
                    raise EvidenceError("raw worker process epoch changed")
                previous_epoch = post_go_process_epochs.setdefault(
                    pid, state["start_time_ticks"]
                )
                if previous_epoch != state["start_time_ticks"]:
                    raise EvidenceError("raw post-GO process epoch changed")
            active_worker_pids = {
                pid for pid, rank in worker_pid_to_rank.items()
                if rank not in departed_ranks
            }
            if not active_worker_pids <= set(descendants):
                raise EvidenceError("raw worker disappeared before process exit")
        nvml = sample["nvml"]
        if (
            not isinstance(nvml, dict)
            or set(nvml) != nvml_sample_keys
            or any(
                not _is_plain_int(nvml.get(name))
                for name in (
                    "scheduled_monotonic_ns", "started_monotonic_ns",
                    "finished_monotonic_ns", "lateness_ns", "duration_ns",
                )
            )
            or not 0 < nvml["scheduled_monotonic_ns"] <= (1 << 64) - 1
            or not 0 < nvml["started_monotonic_ns"] <= (1 << 64) - 1
            or not 0 < nvml["finished_monotonic_ns"] <= (1 << 64) - 1
            or not 0 <= nvml["lateness_ns"] <= (1 << 64) - 1
            or not 0 <= nvml["duration_ns"] <= (1 << 64) - 1
            or not isinstance(nvml["outsiders"], list)
            or nvml["scheduled_monotonic_ns"] != sample["scheduled_monotonic_ns"]
            or not (
                sample["sample_started_monotonic_ns"]
                <= nvml["started_monotonic_ns"]
                <= nvml["finished_monotonic_ns"]
                <= sample["sample_finished_monotonic_ns"]
            )
            or nvml["duration_ns"]
            != nvml["finished_monotonic_ns"] - nvml["started_monotonic_ns"]
            or nvml["started_monotonic_ns"] < nvml["scheduled_monotonic_ns"]
            or sample["sample_started_monotonic_ns"]
            < sample["scheduled_monotonic_ns"]
            or nvml["lateness_ns"]
            != nvml["started_monotonic_ns"] - nvml["scheduled_monotonic_ns"]
            or sample["sample_started_monotonic_ns"]
            != sample["monotonic_ns"]
            or sample["sample_finished_monotonic_ns"]
            < sample["sample_started_monotonic_ns"]
            or not isinstance(nvml["devices"], list)
            or [item.get("uuid") for item in nvml["devices"]]
            != header["nvml_device_uuids"]
            or nvml["lateness_ns"] >= 50_000_000
            or nvml["duration_ns"] >= 50_000_000
        ):
            raise EvidenceError("raw NVML sample timing/schema is invalid")
        for device in nvml["devices"]:
            if set(device) != nvml_device_keys or set(device["process_lists"]) != {
                "compute", "graphics", "mps"
            }:
                raise EvidenceError("raw NVML device schema is invalid")
            process_keys = {
                "pid", "used_gpu_memory_bytes", "gpu_instance_id",
                "compute_instance_id",
                "observed_start_time_ticks",
            }
            utilization_keys = {
                "pid", "timestamp_us", "sm_utilization_percent",
                "memory_utilization_percent", "encoder_utilization_percent",
                "decoder_utilization_percent", "observed_start_time_ticks",
            }
            if any(
                not isinstance(process_rows, list)
                or any(set(record) != process_keys for record in process_rows)
                for process_rows in device["process_lists"].values()
            ) or not isinstance(
                device["process_utilization_since_cursor"], list
            ) or any(
                set(record) != utilization_keys
                for record in device["process_utilization_since_cursor"]
            ):
                raise EvidenceError("raw NVML process-row schema is invalid")
            if (
                set(device["utilization"]) != {"gpu_percent", "memory_percent"}
                or set(device["memory_bytes"]) != {"total", "free", "used"}
            ):
                raise EvidenceError("raw NVML scalar schema is invalid")
            scalar_values = [
                device["graphics_clock_mhz"], device["memory_clock_mhz"],
                device["power_mw"], device["temperature_c"], device["pstate"],
                *device["memory_bytes"].values(),
                *device["utilization"].values(),
            ]
            if (
                any(not _is_plain_int(value) or value < 0 for value in scalar_values)
                or any(
                    value > (1 << 32) - 1
                    for value in (
                        device["graphics_clock_mhz"], device["memory_clock_mhz"],
                        device["power_mw"], device["temperature_c"],
                        device["pstate"],
                    )
                )
                or any(
                    value > (1 << 64) - 1
                    for value in device["memory_bytes"].values()
                )
                or any(value > 100 for value in device["utilization"].values())
                or device["pstate"] > 15
                or device["memory_bytes"]["free"] + device["memory_bytes"]["used"]
                != device["memory_bytes"]["total"]
            ):
                raise EvidenceError("raw NVML scalar value is invalid")
            for process_rows in device["process_lists"].values():
                if len({record["pid"] for record in process_rows}) != len(process_rows):
                    raise EvidenceError("raw NVML process list contains duplicate PIDs")
                if any(
                    not _is_plain_int(record["pid"])
                    or not 1 <= record["pid"] <= (1 << 32) - 1
                    or not _is_plain_int(record["used_gpu_memory_bytes"])
                    or not 0 <= record["used_gpu_memory_bytes"] <= (1 << 64) - 1
                    or not _is_plain_int(record["gpu_instance_id"])
                    or not 0 <= record["gpu_instance_id"] <= (1 << 32) - 1
                    or not _is_plain_int(record["compute_instance_id"])
                    or not 0 <= record["compute_instance_id"] <= (1 << 32) - 1
                    or (
                        record["observed_start_time_ticks"] is not None
                        and (
                            not _is_plain_int(record["observed_start_time_ticks"])
                            or record["observed_start_time_ticks"] <= 0
                        )
                    )
                    for record in process_rows
                ):
                    raise EvidenceError("raw NVML process-list value is invalid")
            uuid_value = device["uuid"]
            utilization_rows = device["process_utilization_since_cursor"]
            coordinates = {
                (record["pid"], record["timestamp_us"])
                for record in utilization_rows
            }
            if len(coordinates) != len(utilization_rows) or any(
                not _is_plain_int(record["pid"])
                or not 1 <= record["pid"] <= (1 << 32) - 1
                or not _is_plain_int(record["timestamp_us"])
                or not 0 <= record["timestamp_us"] <= (1 << 64) - 1
                or record["timestamp_us"] <= last_utilization_cursor[uuid_value]
                or any(
                    not _is_plain_int(record[name]) or not 0 <= record[name] <= 100
                    for name in (
                        "sm_utilization_percent", "memory_utilization_percent",
                        "encoder_utilization_percent", "decoder_utilization_percent",
                    )
                )
                or (
                    record["observed_start_time_ticks"] is not None
                    and (
                        not _is_plain_int(record["observed_start_time_ticks"])
                        or record["observed_start_time_ticks"] <= 0
                    )
                )
                for record in utilization_rows
            ):
                raise EvidenceError("raw NVML utilization cursor/value is invalid")
            if utilization_rows:
                last_utilization_cursor[uuid_value] = max(
                    record["timestamp_us"] for record in utilization_rows
                )
            if uuid_value not in allowed_by_uuid:
                raise EvidenceError("raw NVML device UUID is outside the lane")
        devices_by_uuid = {
            device["uuid"]: device for device in nvml["devices"]
        }
        if sample_row["sequence"] > release_sequence:
            for rank, ready in ready_by_rank.items():
                rows_for_rank = devices_by_uuid[
                    header["nvml_device_uuids"][rank]
                ]["process_lists"]["compute"]
                if rows_for_rank and not (
                    len(rows_for_rank) == 1
                    and rows_for_rank[0]["pid"] == ready["pid"]
                    and rows_for_rank[0]["observed_start_time_ticks"]
                    == ready["start_time_ticks"]
                ):
                    raise EvidenceError(
                        "raw post-RELEASE NVML worker binding changed"
                    )
        derived_outsiders = _derive_nvml_outsiders(
            nvml,
            allowed_by_uuid,
            after_process_exit=bool(sample["after_process_exit"]),
            departed_by_uuid={
                header["nvml_device_uuids"][rank]: {ready_by_rank[rank]["pid"]}
                for rank in departed_ranks
            },
        )
        if nvml["outsiders"] != derived_outsiders:
            raise EvidenceError("raw NVML outsider verdict differs from process rows")
        if derived_outsiders:
            raise EvidenceError("raw NVML process row crossed its UUID/epoch binding")
        current_allowed_by_uuid = {
            uuid_value: dict(records) for uuid_value, records in allowed_by_uuid.items()
        }
        for rank in departed_ranks:
            current_allowed_by_uuid[header["nvml_device_uuids"][rank]].pop(
                ready_by_rank[rank]["pid"], None
            )
        departed_lag_by_uuid = {
            uuid_value: {} for uuid_value in header["nvml_device_uuids"]
        }
        worker_exit_by_rank = {
            record["rank"]: record for record in worker_exit_rows
        }
        for rank in departed_ranks:
            exit_record = worker_exit_by_rank[rank]
            if _within_departed_nvml_lag(
                nvml["finished_monotonic_ns"],
                exit_record["monotonic_ns"],
                header["departed_nvml_lag_limit_ns"],
            ):
                departed_lag_by_uuid[header["nvml_device_uuids"][rank]][
                    ready_by_rank[rank]["pid"]
                ] = ready_by_rank[rank]["start_time_ticks"]
        if sample_row["sequence"] > release_sequence:
            for rank, ready in ready_by_rank.items():
                rows_for_rank = devices_by_uuid[
                    header["nvml_device_uuids"][rank]
                ]["process_lists"]["compute"]
                if rows_for_rank and rank in first_nvml_absence_by_rank:
                    raise EvidenceError("raw NVML worker context reappeared")
                if not rows_for_rank:
                    first_nvml_absence_by_rank.setdefault(
                        rank, nvml["finished_monotonic_ns"]
                    )
                    first_nvml_absence_sequence_by_rank.setdefault(
                        rank, sample_row["sequence"]
                    )
        if sample.get("go_released") and not sample.get("after_process_exit"):
            if sample_row["sequence"] < release_sequence:
                if not _nvml_current_presence_with_departed_lag(
                    nvml,
                    current_allowed_by_uuid,
                    departed_lag_by_uuid,
                    after_process_exit=False,
                ):
                    raise EvidenceError("raw NVML worker presence changed before RELEASE")
            else:
                expected_with_absence = {
                    uuid_value: dict(records)
                    for uuid_value, records in current_allowed_by_uuid.items()
                }
                for rank, ready in ready_by_rank.items():
                    rows_for_rank = devices_by_uuid[
                        header["nvml_device_uuids"][rank]
                    ]["process_lists"]["compute"]
                    if rank in departed_ranks:
                        continue
                    if not rows_for_rank:
                        expected_with_absence[
                            header["nvml_device_uuids"][rank]
                        ].pop(ready["pid"], None)
                if not _nvml_current_presence_with_departed_lag(
                    nvml,
                    expected_with_absence,
                    departed_lag_by_uuid,
                    after_process_exit=False,
                ):
                    raise EvidenceError("raw post-RELEASE NVML lifecycle is invalid")
    worker_exit_by_rank = {record["rank"]: record for record in worker_exit_rows}
    absence_observation_to_pidfd_observation_ns: list[int | None] = []
    pidfd_observation_to_first_nvml_absence_observation_ns: list[int | None] = []
    absence_pidfd_observation_order_by_rank: list[str] = []
    first_nvml_absence_observation_finished_ns_by_rank: list[int] = []
    pidfd_readiness_observed_ns_by_rank: list[int] = []
    worker_exit_event_by_rank = {
        event["payload"]["rank"]: event for event in worker_exit_events
    }
    for rank in range(ranks):
        absence_ns = first_nvml_absence_by_rank.get(rank)
        if absence_ns is None:
            raise EvidenceError(
                "raw NVML telemetry never observed the post-RELEASE worker absence"
            )
        first_nvml_absence_observation_finished_ns_by_rank.append(absence_ns)
        pidfd_ns = worker_exit_by_rank[rank]["monotonic_ns"]
        pidfd_readiness_observed_ns_by_rank.append(pidfd_ns)
        if (
            first_nvml_absence_sequence_by_rank[rank]
            > worker_exit_event_by_rank[rank]["sequence"]
        ):
            # Pidfd readiness was observed first. This absence observation is
            # still latched to reject any subsequent context reappearance, but
            # there is no absence-before-pidfd interval to report or gate.
            absence_pidfd_observation_order_by_rank.append(
                "pidfd-observed-first"
            )
            absence_observation_to_pidfd_observation_ns.append(None)
            reverse_delta = absence_ns - pidfd_ns
            if reverse_delta < 0:
                raise EvidenceError(
                    "raw pidfd/NVML observation order contradicts its timestamps"
                )
            pidfd_observation_to_first_nvml_absence_observation_ns.append(
                reverse_delta
            )
            continue
        absence_pidfd_observation_order_by_rank.append("absence-observed-first")
        pidfd_observation_to_first_nvml_absence_observation_ns.append(None)
        delta = pidfd_ns - absence_ns
        if not 0 <= delta <= header[
            "nvml_absence_observation_to_pidfd_observation_limit_ns"
        ]:
            raise EvidenceError(
                "raw NVML absence observation exceeded the pidfd observation bound"
            )
        absence_observation_to_pidfd_observation_ns.append(delta)
    valid_intervals = 0
    valid_intervals_by_rank = [0] * ranks
    maximum_external_busy_fraction = 0.0
    first_post_go = next(
        (sample for sample in timeline if sample.get("go_released")), None
    )
    if (
        first_post_go is None
        or first_post_go.get("sample_started_monotonic_ns", 0)
        < go_rows[0].get("monotonic_ns", 0)
    ):
        raise EvidenceError("raw telemetry post-GO coverage begins before GO")
    process_exit_ns = exit_rows[0]["monotonic_ns"]
    drains = [
        sample for sample in timeline
        if sample.get("after_process_exit")
        and sample["sample_started_monotonic_ns"] >= process_exit_ns
    ]
    if len(drains) < 2:
        raise EvidenceError("raw NVML telemetry lacks two post-exit drain samples")
    for sample in drains[:2]:
        if not _nvml_current_presence_exact(
            sample["nvml"], allowed_by_uuid, after_process_exit=True
        ):
            raise EvidenceError("raw NVML post-exit current process list is nonempty")
    go_sequence = next(
        row["sequence"] for row in rows if row["record_type"] == "go"
    )
    pre_go_samples = [
        row["payload"] for row in rows
        if row["record_type"] == "sample" and row["sequence"] < go_sequence
    ]
    if len(pre_go_samples) < 2 or any(
        not _nvml_current_presence_exact(
            sample["nvml"], allowed_by_uuid, after_process_exit=False
        )
        for sample in pre_go_samples[-2:]
    ):
        raise EvidenceError("raw NVML GO lacks two exact worker-presence samples")
    if any(not exact_affinity_sample(sample) for sample in pre_go_samples[-2:]):
        raise EvidenceError("raw GO lacks two exact controller/worker affinity samples")
    last_results_ready_sequence = max(
        event["sequence"] for event in results_ready_events
    )
    release_qualification = [
        row["payload"] for row in sample_rows
        if last_results_ready_sequence < row["sequence"] < release_sequence
        and _results_ready_precede_nvml_sample(
            results_ready_rows, row["payload"]["nvml"]
        )
        and row["payload"]["sample_finished_monotonic_ns"]
        <= release_rows[0]["monotonic_ns"]
    ]
    if not release_qualification or not any(
        exact_affinity_sample(sample)
        and _nvml_current_presence_exact(
            sample["nvml"], allowed_by_uuid, after_process_exit=False
        )
        for sample in release_qualification
    ):
        raise EvidenceError("raw RELEASE lacks a qualified live NVML/affinity sample")
    for sample in timeline:
        if not sample.get("go_released"):
            continue
        states = sample.get("worker_process_state")
        if not isinstance(states, dict):
            raise EvidenceError("raw telemetry worker states are invalid")
        for rank in range(ranks):
            state = states.get(str(rank))
            if state is None:
                continue
            if (
                state.get("start_time_ticks")
                != ready_by_rank[rank].get("start_time_ticks")
                or not isinstance(state.get("task_affinities"), dict)
                or any(
                    set(mask) != worker_masks[rank]
                    for mask in state["task_affinities"].values()
                )
            ):
                raise EvidenceError("raw telemetry worker epoch/affinity changed")
    for previous, current in zip(timeline, timeline[1:]):
        if not previous.get("go_released") or not current.get("go_released"):
            continue
        for rank, cpu_mask in enumerate(worker_masks):
            first_state = previous["worker_process_state"].get(str(rank))
            second_state = current["worker_process_state"].get(str(rank))
            if first_state is None or second_state is None:
                continue
            try:
                total = sum(
                    current["cpu_counters"][str(cpu)]["total_ticks"]
                    - previous["cpu_counters"][str(cpu)]["total_ticks"]
                    for cpu in cpu_mask
                )
                idle = sum(
                    current["cpu_counters"][str(cpu)]["idle_ticks"]
                    - previous["cpu_counters"][str(cpu)]["idle_ticks"]
                    for cpu in cpu_mask
                )
                worker = second_state["ticks"] - first_state["ticks"]
            except (KeyError, TypeError) as exc:
                raise EvidenceError("raw telemetry CPU counters are incomplete") from exc
            if total <= 0 or idle < 0 or worker < 0:
                raise EvidenceError("host CPU telemetry counters moved backwards")
            external = max(0, max(0, total - idle) - worker)
            maximum_external_busy_fraction = max(
                maximum_external_busy_fraction, external / total
            )
            valid_intervals += 1
            valid_intervals_by_rank[rank] += 1
    if any(count < 1 for count in valid_intervals_by_rank):
        raise EvidenceError("host CPU telemetry did not cover each worker rank")
    if maximum_external_busy_fraction > 0.05:
        raise EvidenceError("worker CPU experienced excessive external contention")
    go_samples = [sample for sample in timeline if sample.get("go_released")]
    try:
        duration_us = (
            go_samples[-1]["monotonic_ns"] - go_samples[0]["monotonic_ns"]
        ) / 1000.0
        if duration_us <= 0:
            raise EvidenceError("raw telemetry PSI duration is empty")
        some_fractions = {
            name: (
                go_samples[-1]["pressure"][name]["some"]["total"]
                - go_samples[0]["pressure"][name]["some"]["total"]
            ) / duration_us
            for name in ("cpu", "memory", "io")
        }
        full_fractions = {
            name: (
                go_samples[-1]["pressure"][name]["full"]["total"]
                - go_samples[0]["pressure"][name]["full"]["total"]
            ) / duration_us
            for name in ("memory", "io")
        }
        if any(value < 0 for value in (*some_fractions.values(), *full_fractions.values())):
            raise EvidenceError("raw telemetry PSI totals moved backwards")
        maximum_psi_some = 100.0 * max(some_fractions.values())
        maximum_psi_full = 100.0 * max(full_fractions.values())
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError("raw telemetry PSI records are incomplete") from exc
    if maximum_psi_some > 5.0 or maximum_psi_full > 1.0:
        raise EvidenceError("Linux PSI contention gate was exceeded")
    return {
        "schema_version": 1,
        "journal_header": header,
        "nonce": header["nonce"],
        "online_cpu_list": header["online_cpu_list"],
        "controller_cpu_mask": sorted(controller_mask),
        "worker_cpu_masks": [sorted(mask) for mask in worker_masks],
        "ready_records": sorted(ready_rows, key=lambda record: record["rank"]),
        "results_ready_records": sorted(
            results_ready_rows, key=lambda record: record["rank"]
        ),
        "go_monotonic_ns": go_rows[0].get("monotonic_ns"),
        "release_monotonic_ns": release_rows[0].get("monotonic_ns"),
        "process_finished_monotonic_ns": exit_rows[0].get("monotonic_ns"),
        "nvml_absence_observation_to_pidfd_observation_ns_by_rank": (
            absence_observation_to_pidfd_observation_ns
        ),
        "nvml_absence_pidfd_observation_order_by_rank": (
            absence_pidfd_observation_order_by_rank
        ),
        "pidfd_observation_to_first_nvml_absence_observation_ns_by_rank": (
            pidfd_observation_to_first_nvml_absence_observation_ns
        ),
        "first_nvml_absence_observation_finished_ns_by_rank": (
            first_nvml_absence_observation_finished_ns_by_rank
        ),
        "pidfd_readiness_observed_ns_by_rank": (
            pidfd_readiness_observed_ns_by_rank
        ),
        "timeline": timeline,
        "sample_count": len(timeline),
        "maximum_sample_gap_seconds": maximum_gap_ns / 1_000_000_000,
        "valid_worker_cpu_intervals": valid_intervals,
        "valid_worker_cpu_intervals_by_rank": valid_intervals_by_rank,
        "maximum_external_busy_fraction": maximum_external_busy_fraction,
        "maximum_external_busy_fraction_exclusive": 0.05,
        "maximum_psi_some_interval_percent": maximum_psi_some,
        "maximum_psi_full_interval_percent": maximum_psi_full,
        "psi_some_interval_percent": {
            name: 100.0 * value for name, value in some_fractions.items()
        },
        "psi_full_interval_percent": {
            name: 100.0 * value for name, value in full_fractions.items()
        },
        "psi_some_maximum_percent": 5.0,
        "psi_full_maximum_percent": 1.0,
    }


class LaneControlMonitor:
    """Pre-measurement worker pinning and continuous host-load evidence."""

    def __init__(
        self,
        *,
        directory: pathlib.Path,
        ranks: int,
        controller_cpu_mask: set[int],
        worker_cpu_masks: list[set[int]],
        expected_device_uuids: list[str],
        physical_topology_plan: dict[str, Any],
        nvml_sampler: NvmlSampler,
        nvml_library_archive: StableFile,
    ) -> None:
        self.directory = directory
        self.ranks = ranks
        self.controller_cpu_mask = set(controller_cpu_mask)
        self.worker_cpu_masks = [set(mask) for mask in worker_cpu_masks[:ranks]]
        self.expected_device_uuids = [
            value.removeprefix("GPU-").replace("-", "")
            for value in expected_device_uuids[:ranks]
        ]
        self.physical_topology_plan = physical_topology_plan
        self.nvml_sampler = nvml_sampler
        self.nvml_library_archive = nvml_library_archive
        self.nvml_device_uuids = list(nvml_sampler.handles)
        self.nonce = secrets.token_hex(32)
        role_masks = [self.controller_cpu_mask, *self.worker_cpu_masks]
        if any(
            first & second
            for index, first in enumerate(role_masks)
            for second in role_masks[index + 1 :]
        ):
            raise EvidenceError("controller and worker physical-core masks overlap")
        self.online_cpu_list = pathlib.Path(
            "/sys/devices/system/cpu/online"
        ).read_text(encoding="ascii").strip()
        self.controller_pid = os.getpid()
        self.controller_start_time_ticks = _process_start_time_ticks(
            self.controller_pid
        )
        if self.controller_start_time_ticks is None:
            raise EvidenceError("controller process identity is unavailable")
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.journal_header = {
                "nonce": self.nonce,
                "ranks": self.ranks,
                "controller_cpu_mask": sorted(self.controller_cpu_mask),
                "worker_cpu_masks": [sorted(mask) for mask in self.worker_cpu_masks],
                "expected_device_uuids": [
                    value.removeprefix("GPU-").replace("-", "")
                    for value in self.nvml_device_uuids
                ],
                "online_cpu_list": self.online_cpu_list,
                "allowed_cpus": self.physical_topology_plan["allowed_cpus"],
                "physical_topology_plan": self.physical_topology_plan,
                "controller_pid": self.controller_pid,
                "controller_start_time_ticks": self.controller_start_time_ticks,
                "nvml_archive_identity": _nvml_archive_content_identity(
                    self.nvml_library_archive
                ),
                "nvml_baseline_process_utilization": (
                    self.nvml_sampler.baseline_process_utilization
                ),
                "nvml_baseline_current_processes": (
                    self.nvml_sampler.baseline_current_processes
                ),
                "nvml_device_uuids": self.nvml_device_uuids,
                "sample_period_ns": 50_000_000,
                "departed_nvml_lag_limit_ns": NVML_DEPARTED_CONTEXT_LAG_NS,
                "nvml_absence_observation_to_pidfd_observation_limit_ns": (
                    NVML_ABSENCE_OBSERVATION_TO_PIDFD_OBSERVATION_NS
                ),
            }
        self.journal = RawTelemetryJournal(
            directory / "telemetry.raw.jsonl",
            self.journal_header,
        )
        self.journal_record: dict[str, Any] | None = None
        self.ready: dict[int, dict[str, Any]] = {}
        self.results_ready: dict[int, dict[str, Any]] = {}
        self.ready_pidfds: dict[int, int] = {}
        self.worker_exits: dict[int, dict[str, Any]] = {}
        self.go_monotonic_ns: int | None = None
        self.release_monotonic_ns: int | None = None
        self.first_nvml_absence_ns: dict[int, int] = {}
        self.lifecycle_issues: list[str] = []
        self.clean_affinity_polls = 0
        self.clean_gpu_polls = 0
        self.timeline: list[dict[str, Any]] = []
        self.sample_origin_ns: int | None = None
        self.next_sample_index = 0
        self.process_finished_monotonic_ns: int | None = None

    def _load_ready(self, rank: int) -> dict[str, Any] | None:
        path = self.directory / f"rank-{rank}.ready"
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        handle = StableFile.open_named_no_follow(
            f"LDOS rank {rank} ready record", path,
            maximum_size_bytes=LDOS_READY_MAX_BYTES,
        )
        try:
            ready_info = os.fstat(handle.descriptor)
            if (
                stat.S_IMODE(ready_info.st_mode) != 0o600
                or ready_info.st_nlink != 1
                or ready_info.st_uid != os.geteuid()
            ):
                raise EvidenceError("LDOS worker ready record metadata is invalid")
            try:
                record = json.loads(
                    pathlib.Path(handle.proc_path).read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                raise EvidenceError("LDOS worker ready record is invalid JSON") from exc
            if (
                not isinstance(record, dict)
                or set(record)
                != {
                    "schema_version", "rank", "pid", "start_time_ticks",
                    "device_ordinal", "device_uuid", "nonce", "task_affinities",
                }
                or record["schema_version"] != 1
                or record["rank"] != rank
                or record["device_ordinal"] != rank
                or not _is_plain_int(record["pid"])
                or record["pid"] <= 0
                or not _is_plain_int(record["start_time_ticks"])
                or record["start_time_ticks"] <= 0
                or record["device_uuid"] != self.expected_device_uuids[rank]
                or record["nonce"] != self.nonce
                or not isinstance(record["task_affinities"], dict)
            ):
                raise EvidenceError("LDOS worker ready record differs from lane binding")
            state = _process_task_cpu_state(record["pid"])
            if state is None:
                raise EvidenceError("LDOS worker vanished while loading ready record")
            if (
                _process_start_time_ticks(record["pid"])
                != record["start_time_ticks"]
                or record["task_affinities"] != state["task_affinities"]
                or any(
                    set(mask) != self.worker_cpu_masks[rank]
                    for mask in state["task_affinities"].values()
                )
            ):
                raise EvidenceError("LDOS worker ready process identity is stale")
            return {**record, "file": handle.record()}
        finally:
            handle.close()

    def _load_results_ready(self, rank: int) -> dict[str, Any] | None:
        path = self.directory / f"rank-{rank}.results-ready"
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        handle = StableFile.open_named_no_follow(
            f"LDOS rank {rank} results-ready record", path,
            maximum_size_bytes=LDOS_HANDSHAKE_MAX_BYTES,
        )
        try:
            info = os.fstat(handle.descriptor)
            if (
                handle.path != path.absolute()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
                or info.st_uid != os.geteuid()
            ):
                raise EvidenceError("LDOS RESULTS_READY metadata is invalid")
            data = os.pread(handle.descriptor, info.st_size + 1, 0)
            try:
                record = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_json_object)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise EvidenceError("LDOS RESULTS_READY is invalid JSON") from exc
            expected_keys = {
                "schema_version", "state", "rank", "pid", "start_time_ticks",
                "device_ordinal", "device_uuid", "nonce", "monotonic_ns",
            }
            ready = self.ready.get(rank)
            observed_monotonic_ns = time.monotonic_ns()
            if (
                ready is None
                or not isinstance(record, dict)
                or set(record) != expected_keys
                or not _is_plain_int(record["schema_version"])
                or record["schema_version"] != 1
                or record["state"] != "RESULTS_READY"
                or not _is_plain_int(record["rank"])
                or record["rank"] != rank
                or not _is_plain_int(record["pid"])
                or record["pid"] != ready["pid"]
                or not _is_plain_int(record["start_time_ticks"])
                or record["start_time_ticks"] != ready["start_time_ticks"]
                or not _is_plain_int(record["device_ordinal"])
                or record["device_ordinal"] != ready["device_ordinal"]
                or record["device_uuid"] != ready["device_uuid"]
                or record["nonce"] != self.nonce
                or not _is_plain_int(record["monotonic_ns"])
                or not 0 < record["monotonic_ns"] <= (1 << 64) - 1
                or self.go_monotonic_ns is None
                or record["monotonic_ns"] <= self.go_monotonic_ns
                or record["monotonic_ns"] > observed_monotonic_ns
            ):
                raise EvidenceError("LDOS RESULTS_READY differs from lane binding")
            state = _process_task_cpu_state(record["pid"])
            if (
                state is None
                or state["start_time_ticks"] != record["start_time_ticks"]
                or any(
                    set(mask) != self.worker_cpu_masks[rank]
                    for mask in state["task_affinities"].values()
                )
            ):
                raise EvidenceError("LDOS RESULTS_READY process epoch is not live")
            ordered = {
                "schema_version": 1,
                "state": "RESULTS_READY",
                "rank": rank,
                "pid": record["pid"],
                "start_time_ticks": record["start_time_ticks"],
                "device_ordinal": record["device_ordinal"],
                "device_uuid": record["device_uuid"],
                "nonce": self.nonce,
                "monotonic_ns": record["monotonic_ns"],
            }
            expected_bytes = json.dumps(
                ordered, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            if data != expected_bytes:
                raise EvidenceError("LDOS RESULTS_READY bytes are not canonical")
            return {
                **record,
                "observed_monotonic_ns": observed_monotonic_ns,
                "file": handle.record(),
            }
        finally:
            handle.close()

    def _publish_release(self) -> None:
        if self.release_monotonic_ns is not None:
            raise EvidenceError("LDOS RELEASE was already published")
        release_path = self.directory / "RELEASE"
        release_text = json.dumps(
            {
                "schema_version": 1,
                "nonce": self.nonce,
                "ranks": self.ranks,
                "state": "RELEASE",
            },
            separators=(",", ":"),
        ) + "\n"
        release_publication_lower_bound_ns = time.monotonic_ns()
        _write_text_exclusive(release_path, release_text, 0o400)
        handle = StableFile.open_named_no_follow(
            "LDOS RELEASE gate", release_path,
            maximum_size_bytes=LDOS_HANDSHAKE_MAX_BYTES,
        )
        try:
            info = os.fstat(handle.descriptor)
            if (
                handle.path != release_path.absolute()
                or stat.S_IMODE(info.st_mode) != 0o400
                or info.st_nlink != 1
                or info.st_uid != os.geteuid()
                or os.pread(handle.descriptor, info.st_size + 1, 0)
                != release_text.encode("utf-8")
            ):
                raise EvidenceError("LDOS RELEASE gate identity is invalid")
            self.release_monotonic_ns = release_publication_lower_bound_ns
            self.journal.append(
                "release",
                {
                    "monotonic_ns": self.release_monotonic_ns,
                    "results_ready_ranks": sorted(self.results_ready),
                    "file": handle.record(),
                },
            )
        finally:
            handle.close()

    def _active_nvml_rank_presence(
        self, nvml_sample: dict[str, Any]
    ) -> dict[int, bool]:
        devices = {
            device.get("uuid"): device for device in nvml_sample.get("devices", [])
        }
        if set(devices) != set(self.nvml_device_uuids):
            raise EvidenceError("NVML device matrix changed during LDOS lane")
        result: dict[int, bool] = {}
        for rank, ready in self.ready.items():
            if rank in self.worker_exits:
                continue
            lists = devices[self.nvml_device_uuids[rank]].get("process_lists", {})
            if lists.get("graphics") or lists.get("mps"):
                raise EvidenceError("NVML observed a non-compute worker context")
            compute = lists.get("compute")
            if not isinstance(compute, list):
                raise EvidenceError("NVML compute process list is incomplete")
            if not compute:
                result[rank] = False
                continue
            if len(compute) != 1 or (
                compute[0].get("pid") != ready["pid"]
                or compute[0].get("observed_start_time_ticks")
                != ready["start_time_ticks"]
            ):
                raise EvidenceError("NVML worker PID/epoch differs from lane binding")
            result[rank] = True
        return result

    def _require_absence_deadlines(self, timestamp_ns: int) -> None:
        for rank, absence_ns in self.first_nvml_absence_ns.items():
            if (
                rank not in self.worker_exits
                and timestamp_ns - absence_ns
                > NVML_ABSENCE_OBSERVATION_TO_PIDFD_OBSERVATION_NS
            ):
                raise EvidenceError(
                    "LDOS NVML absence observation exceeded the pidfd observation bound"
                )

    def _observe_nvml_lifecycle_and_maybe_release(
        self, nvml_sample: dict[str, Any], *, presence_exact: bool
    ) -> None:
        devices = {
            device.get("uuid"): device for device in nvml_sample.get("devices", [])
        }
        for rank in self.worker_exits:
            ready = self.ready[rank]
            device = devices.get(self.nvml_device_uuids[rank])
            if device is None:
                raise EvidenceError("NVML device matrix changed during LDOS lane")
            lists = device.get("process_lists", {})
            compute = lists.get("compute")
            if (
                not isinstance(compute, list)
                or lists.get("graphics")
                or lists.get("mps")
            ):
                raise EvidenceError("NVML observed an invalid departed worker context")
            exact_row = bool(
                len(compute) == 1
                and compute[0].get("pid") == ready["pid"]
                and compute[0].get("observed_start_time_ticks")
                == ready["start_time_ticks"]
            )
            if compute and not exact_row:
                raise EvidenceError("NVML departed worker PID/epoch binding changed")
            if compute and rank in self.first_nvml_absence_ns:
                raise EvidenceError(
                    "LDOS NVML worker context reappeared after teardown began"
                )
            if not compute:
                self.first_nvml_absence_ns.setdefault(
                    rank, nvml_sample["finished_monotonic_ns"]
                )
        active_presence = self._active_nvml_rank_presence(nvml_sample)
        for rank, present in active_presence.items():
            if not present:
                self.first_nvml_absence_ns.setdefault(
                    rank, nvml_sample["finished_monotonic_ns"]
                )
            elif rank in self.first_nvml_absence_ns:
                raise EvidenceError(
                    "LDOS NVML worker context reappeared after teardown began"
                )
        if self.release_monotonic_ns is not None:
            return
        if not presence_exact:
            issue = "NVML worker presence changed before RESULTS_READY/RELEASE"
            if issue not in self.lifecycle_issues:
                self.lifecycle_issues.append(issue)
            return
        if (
            set(self.results_ready) == set(range(self.ranks))
            and not self.worker_exits
            and not self.lifecycle_issues
            and _results_ready_precede_nvml_sample(
                self.results_ready.values(), nvml_sample
            )
        ):
            self._publish_release()

    def _record_worker_exits(self) -> None:
        if self.go_monotonic_ns is None:
            return
        for rank, record in self.ready.items():
            if rank in self.worker_exits:
                continue
            observed_start = _process_start_time_ticks(record["pid"])
            if (
                observed_start is not None
                and observed_start != record["start_time_ticks"]
            ):
                raise EvidenceError("LDOS worker PID was reused before exit harvest")
            poller = select.poll()
            poller.register(self.ready_pidfds[rank], select.POLLIN)
            if not poller.poll(0):
                continue
            # Bind the endpoint to a fresh observation taken only after this
            # rank's descriptor has positively reported readiness. A callback
            # entry timestamp can predate delayed rank polling.
            observed_monotonic_ns = time.monotonic_ns()
            # pidfd readiness is the authoritative exit transition for the
            # exact descriptor-bound epoch. /proc can still expose that epoch
            # as a zombie until its parent (or this subreaper) performs wait().
            exit_record = {
                "rank": rank,
                "pid": record["pid"],
                "start_time_ticks": record["start_time_ticks"],
                "monotonic_ns": observed_monotonic_ns,
            }
            self.worker_exits[rank] = exit_record
            self.journal.append("worker-exit", exit_record)
            absence_ns = self.first_nvml_absence_ns.get(rank)
            if absence_ns is not None and not (
                0 <= observed_monotonic_ns - absence_ns
                <= NVML_ABSENCE_OBSERVATION_TO_PIDFD_OBSERVATION_NS
            ):
                raise EvidenceError(
                    "LDOS NVML absence observation was not followed by a bounded "
                    "pidfd readiness observation"
                )
        self._require_absence_deadlines(time.monotonic_ns())

    def poll(self, root_pid: int) -> None:
        self._record_worker_exits()
        self._require_absence_deadlines(time.monotonic_ns())
        if self.go_monotonic_ns is None:
            # Popen invokes the callback before MPI workers normally have time
            # to publish READY.  Do not start or advance the telemetry cadence
            # until every rank has a descriptor-bound epoch; otherwise the
            # first sample cannot contain the exact full-rank state matrix.
            for rank in range(self.ranks):
                if rank not in self.ready:
                    record = self._load_ready(rank)
                    if record is not None:
                        self.ready[rank] = record
                        self.ready_pidfds[rank] = _pidfd_open(
                            record["pid"],
                            expected_start_time_ticks=record["start_time_ticks"],
                        )
                        self.journal.append("ready", record)
            if len(self.ready) != self.ranks:
                return
        poll_now_ns = time.monotonic_ns()
        if self.sample_origin_ns is None:
            self.sample_origin_ns = poll_now_ns
        scheduled_ns = self.sample_origin_ns + self.next_sample_index * 50_000_000
        if poll_now_ns < scheduled_ns:
            return
        if (
            self.go_monotonic_ns is not None
            and len(self.worker_exits) == self.ranks
            and _process_start_time_ticks(root_pid) is None
            and self.process_finished_monotonic_ns is None
        ):
            # _run_process performs one final callback after wait/communicate.
            # Record the worker pidfd transitions, but wait for drain() to bind
            # the root exit before collecting the two post-exit samples.
            return
        self.next_sample_index += 1
        if self.go_monotonic_ns is not None:
            for rank in range(self.ranks):
                if rank not in self.results_ready:
                    record = self._load_results_ready(rank)
                    if record is not None:
                        self.results_ready[rank] = record
                        self.journal.append("results-ready", record)
        # Take the sample endpoint only after every same-poll lifecycle event
        # has received its own fresh observation and journal record.  This
        # keeps the one-writer journal intervals physically non-overlapping.
        timestamp_ns = time.monotonic_ns()
        departed_worker_pids = {
            self.ready[rank]["pid"] for rank in self.worker_exits
        }
        descendants = (
            {root_pid} | _process_descendants(root_pid)
        ) - departed_worker_pids
        if self.go_monotonic_ns is None:
            if any(record["pid"] not in descendants for record in self.ready.values()):
                raise EvidenceError("LDOS ready PID is not in the contained MPI tree")
            worker_by_pid = {
                record["pid"]: rank for rank, record in self.ready.items()
            }
            stable = True
            for pid in sorted(descendants):
                expected_mask = self.worker_cpu_masks[worker_by_pid[pid]] if pid in worker_by_pid else self.controller_cpu_mask
                state = _process_task_cpu_state(pid)
                if state is None:
                    stable = False
                    continue
                if any(
                    set(mask) != expected_mask
                    for mask in state["task_affinities"].values()
                ):
                    stable = False
            self.clean_affinity_polls = self.clean_affinity_polls + 1 if stable else 0

        process_task_states = {
            str(pid): _process_task_cpu_state(pid) for pid in sorted(descendants)
        }
        controller_process_state = _process_task_cpu_state(self.controller_pid)
        if controller_process_state is None or any(
            set(mask) != self.controller_cpu_mask
            for mask in controller_process_state["task_affinities"].values()
        ):
            raise EvidenceError("controller native task affinity changed")
        worker_states = {
            str(rank): process_task_states.get(str(record["pid"]))
            for rank, record in self.ready.items()
        }
        if self.go_monotonic_ns is not None:
            worker_by_pid = {
                record["pid"]: rank for rank, record in self.ready.items()
            }
            for rank, record in self.ready.items():
                state = worker_states[str(rank)]
                if (
                    state is None
                    and self.process_finished_monotonic_ns is None
                    and rank not in self.worker_exits
                ):
                    issue = f"rank {rank} vanished before its pidfd exit observation"
                    if issue not in self.lifecycle_issues:
                        self.lifecycle_issues.append(issue)
                if state is not None and (
                    state["start_time_ticks"] != record["start_time_ticks"]
                    or any(
                        set(mask) != self.worker_cpu_masks[rank]
                        for mask in state["task_affinities"].values()
                    )
                ):
                    raise EvidenceError("post-GO worker task affinity/epoch changed")
            for pid in descendants - set(worker_by_pid):
                state = _process_task_cpu_state(pid)
                if state is not None and any(
                    set(mask) != self.controller_cpu_mask
                    for mask in state["task_affinities"].values()
                ):
                    raise EvidenceError("post-GO controller/PRTE affinity changed")
        frequencies: dict[str, int | None] = {}
        governors: dict[str, str | None] = {}
        monitored_cpus = sorted(
            self.controller_cpu_mask.union(*self.worker_cpu_masks)
        )
        for cpu in monitored_cpus:
            frequency_path = pathlib.Path(
                f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
            )
            governor_path = pathlib.Path(
                f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            )
            try:
                frequencies[str(cpu)] = int(frequency_path.read_text().strip())
            except (FileNotFoundError, OSError, ValueError):
                frequencies[str(cpu)] = None
            try:
                governors[str(cpu)] = governor_path.read_text().strip()
            except (FileNotFoundError, OSError):
                governors[str(cpu)] = None
        allowed_process_epochs_by_uuid = {
            uuid_value: {} for uuid_value in self.nvml_device_uuids
        }
        for rank, record in self.ready.items():
            allowed_process_epochs_by_uuid[self.nvml_device_uuids[rank]][
                record["pid"]
            ] = record["start_time_ticks"]
        current_process_epochs_by_uuid = {
            uuid_value: dict(records)
            for uuid_value, records in allowed_process_epochs_by_uuid.items()
        }
        departed_by_uuid = {
            uuid_value: set() for uuid_value in self.nvml_device_uuids
        }
        for rank, exit_record in self.worker_exits.items():
            uuid_value = self.nvml_device_uuids[rank]
            current_process_epochs_by_uuid[uuid_value].pop(exit_record["pid"], None)
            departed_by_uuid[uuid_value].add(exit_record["pid"])
        after_process_exit = (
            self.process_finished_monotonic_ns is not None
            and timestamp_ns >= self.process_finished_monotonic_ns
        )
        nvml_sample = self.nvml_sampler.sample(
            allowed_process_epochs_by_uuid=allowed_process_epochs_by_uuid,
            scheduled_monotonic_ns=scheduled_ns,
            after_process_exit=after_process_exit,
            departed_processes_by_uuid=departed_by_uuid,
        )
        departed_lag_epochs_by_uuid = {
            uuid_value: {} for uuid_value in self.nvml_device_uuids
        }
        for rank, exit_record in self.worker_exits.items():
            if _within_departed_nvml_lag(
                nvml_sample["finished_monotonic_ns"],
                exit_record["monotonic_ns"],
            ):
                departed_lag_epochs_by_uuid[self.nvml_device_uuids[rank]][
                    exit_record["pid"]
                ] = exit_record["start_time_ticks"]
        retroactive_outsiders: list[dict[str, Any]] = []
        if len(self.ready) == self.ranks:
            for previous in self.timeline:
                retroactive_outsiders.extend(
                    _derive_nvml_outsiders(
                        previous["nvml"],
                        allowed_process_epochs_by_uuid,
                        after_process_exit=previous["after_process_exit"],
                        departed_by_uuid={
                            self.nvml_device_uuids[rank]: {
                                self.ready[rank]["pid"]
                                for rank in previous["departed_worker_ranks"]
                            }
                        },
                    )
                )
            retroactive_outsiders.extend(
                _derive_nvml_outsiders(
                    nvml_sample,
                    allowed_process_epochs_by_uuid,
                    after_process_exit=after_process_exit,
                    departed_by_uuid=departed_by_uuid,
                )
            )
            presence_exact = _nvml_current_presence_with_departed_lag(
                nvml_sample,
                current_process_epochs_by_uuid,
                departed_lag_epochs_by_uuid,
                after_process_exit=after_process_exit,
            )
            self.clean_gpu_polls = (
                self.clean_gpu_polls + 1
                if not retroactive_outsiders and presence_exact
                else 0
            )
        else:
            self.clean_gpu_polls = 0
        sample = {
                "monotonic_ns": timestamp_ns,
                "scheduled_monotonic_ns": scheduled_ns,
                "sample_started_monotonic_ns": timestamp_ns,
                "go_released": (
                    self.go_monotonic_ns is not None
                    and timestamp_ns >= self.go_monotonic_ns
                ),
                "descendant_pids": sorted(descendants),
                "process_task_state": process_task_states,
                "controller_process_state": controller_process_state,
                "worker_process_state": worker_states,
                "cpu_counters": {
                    str(cpu): counters
                    for cpu, counters in _proc_stat_cpu_counters(
                        monitored_cpus
                    ).items()
                },
                "current_frequency_khz": frequencies,
                "governors": governors,
                "loadavg": pathlib.Path("/proc/loadavg").read_text(
                    encoding="ascii"
                ).strip(),
                "pressure": {
                    name: _parse_pressure_file(name)
                    for name in ("cpu", "memory", "io")
                },
                "nvml": nvml_sample,
                "after_process_exit": after_process_exit,
                "departed_worker_ranks": sorted(self.worker_exits),
            }
        sample["sample_finished_monotonic_ns"] = time.monotonic_ns()
        self.timeline.append(sample)
        self.journal.append("sample", sample)
        if retroactive_outsiders:
            raise EvidenceError(
                "NVML observed a process outside exact ready PID/UUID epochs"
            )
        if self.go_monotonic_ns is not None:
            self._observe_nvml_lifecycle_and_maybe_release(
                nvml_sample, presence_exact=presence_exact
            )
        if (
            self.go_monotonic_ns is None
            and self.clean_affinity_polls >= 2
            and self.clean_gpu_polls >= 2
        ):
            # This is a conservative publication lower bound. A worker may
            # observe the file immediately after rename, before the controller
            # can take a post-write timestamp.
            go_publication_lower_bound_ns = time.monotonic_ns()
            _write_text_exclusive(
                self.directory / "GO",
                json.dumps(
                    {"schema_version": 1, "nonce": self.nonce, "ranks": self.ranks},
                    separators=(",", ":"),
                )
                + "\n",
                0o400,
            )
            self.go_monotonic_ns = go_publication_lower_bound_ns
            self.journal.append(
                "go",
                {
                    "monotonic_ns": self.go_monotonic_ns,
                    "clean_affinity_polls": self.clean_affinity_polls,
                    "clean_gpu_polls": self.clean_gpu_polls,
                },
            )

    def drain(self, root_pid: int, process_finished_monotonic_ns: int) -> None:
        deadline = time.monotonic() + 1.0
        while len(self.worker_exits) != self.ranks and time.monotonic() < deadline:
            self._record_worker_exits()
            if len(self.worker_exits) != self.ranks:
                time.sleep(0.005)
        if self.release_monotonic_ns is None or len(self.results_ready) != self.ranks:
            raise EvidenceError("MPI success lacked complete RESULTS_READY/RELEASE lifecycle")
        if self.lifecycle_issues:
            raise EvidenceError("; ".join(self.lifecycle_issues))
        if len(self.worker_exits) != self.ranks:
            raise EvidenceError("MPI root exited before every worker pidfd transition was recorded")
        if any(
            record["monotonic_ns"] > process_finished_monotonic_ns
            for record in self.worker_exits.values()
        ):
            raise EvidenceError("worker-exit observation occurred after MPI root exit")
        self.process_finished_monotonic_ns = process_finished_monotonic_ns
        self.journal.append(
            "process-exit", {"monotonic_ns": process_finished_monotonic_ns}
        )
        target = len(self.timeline) + 2
        deadline = time.monotonic() + 1.0
        while len(self.timeline) < target:
            if time.monotonic() >= deadline:
                raise EvidenceError("NVML exit drain sampling timed out")
            self.poll(root_pid)
            if len(self.timeline) < target:
                time.sleep(0.005)

    def diagnose_failure_teardown(
        self, root_pid: int, process_finished_monotonic_ns: int
    ) -> list[str]:
        """Best-effort failure harvest which can never replace worker status."""

        diagnostics = list(self.lifecycle_issues)
        try:
            deadline = time.monotonic() + 0.25
            while len(self.worker_exits) != self.ranks and time.monotonic() < deadline:
                self._record_worker_exits()
                if len(self.worker_exits) != self.ranks:
                    time.sleep(0.005)
        except BaseException as error:
            diagnostics.append(f"pidfd harvest: {type(error).__name__}: {error}")
        missing = sorted(set(range(self.ranks)) - set(self.worker_exits))
        if missing:
            diagnostics.append(f"missing worker pidfd exits: {missing}")
        try:
            self.process_finished_monotonic_ns = process_finished_monotonic_ns
            self.journal.append(
                "process-exit", {"monotonic_ns": process_finished_monotonic_ns}
            )
        except BaseException as error:
            diagnostics.append(f"process-exit journal: {type(error).__name__}: {error}")
        return diagnostics

    def finish(self) -> dict[str, Any]:
        if (
            self.go_monotonic_ns is None
            or len(self.ready) != self.ranks
            or self.release_monotonic_ns is None
            or len(self.results_ready) != self.ranks
        ):
            raise EvidenceError("LDOS workers never reached the pinned ready/GO gate")
        if pathlib.Path("/sys/devices/system/cpu/online").read_text(
            encoding="ascii"
        ).strip() != self.online_cpu_list:
            raise EvidenceError("online CPU topology changed during the lane")
        if _physical_cpu_core_plan(
            max(PROFILE["mpi_ranks"]), set(self.journal_header["allowed_cpus"])
        ) != self.physical_topology_plan:
            raise EvidenceError("physical CPU topology changed during the lane")
        source_identity = self.nvml_sampler.library_file.verify()
        archive_identity = self.nvml_library_archive.verify()
        if (
            source_identity != self.nvml_sampler.library_identity
            or archive_identity != self.nvml_library_archive.record()
            or source_identity["sha256"] != archive_identity["sha256"]
            or source_identity["size_bytes"] != archive_identity["size_bytes"]
            or _nvml_archive_content_identity(self.nvml_library_archive)
            != self.journal_header["nvml_archive_identity"]
        ):
            raise EvidenceError("held/archived NVML library identity changed")
        # Validate the still-open bytes first. Any failure is followed by abort(),
        # so only a completely validated lane receives a complete terminal.
        summary = validate_lane_journal(
            self.journal.path,
            expected_header=self.journal_header,
            require_complete=False,
        )
        self.journal_record = self.journal.close("sealed")
        final = validate_lane_journal(
            self.journal.path,
            expected_header=self.journal_header,
            require_complete=True,
        )
        if final != summary:
            raise EvidenceError("closed telemetry journal changed its derived verdict")
        final["raw_journal"] = self.journal_record
        for descriptor in self.ready_pidfds.values():
            _close_native_owned_descriptor_once("lane READY pidfd", descriptor)
        self.ready_pidfds.clear()
        return final

    def abort(self, error: BaseException | str) -> dict[str, Any]:
        primary_error: BaseException | None = None
        try:
            if self.journal_record is None:
                self.journal_record = self.journal.close("error", str(error))
        except BaseException as close_error:
            primary_error = close_error
        finally:
            descriptor_errors: list[str] = []
            for rank, descriptor in self.ready_pidfds.items():
                try:
                    _close_native_owned_descriptor_once(
                        f"rank {rank} READY pidfd", descriptor
                    )
                except BaseException as descriptor_error:
                    descriptor_errors.append(
                        f"rank {rank} pidfd close: "
                        f"{type(descriptor_error).__name__}: {descriptor_error}"
                    )
            self.ready_pidfds.clear()
            if descriptor_errors:
                message = "; ".join(descriptor_errors)
                if primary_error is not None:
                    primary_error.add_note(message)
                else:
                    primary_error = EvidenceError(message)
        if primary_error is not None:
            raise primary_error
        if self.journal_record is None:
            raise EvidenceError("lane abort produced no raw journal record")
        return self.journal_record


def _run_process(
    *,
    command: list[str],
    environment: dict[str, str],
    cwd: pathlib.Path,
    timeout_seconds: int,
    executable: str | None = None,
    pass_fds: tuple[int, ...] = (),
    observe_runtime: bool = False,
    poll_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    if threading.active_count() != 1:
        raise EvidenceError("process containment requires a single-threaded controller")
    start = time.monotonic()
    started_at = _utc_now()
    previous_subreaper = _child_subreaper_state()
    if not previous_subreaper:
        try:
            _set_child_subreaper(True)
            if not _raw_child_subreaper_state():
                raise EvidenceError(
                    "process child-subreaper enable did not reach the kernel"
                )
        except BaseException as error:
            restore_diagnostics = _restore_child_subreaper_state(
                previous_subreaper, PROCESS_GRACE_SECONDS
            )
            if restore_diagnostics:
                error.add_note("; ".join(restore_diagnostics))
            raise
    process: subprocess.Popen[str] | None = None
    process_started_monotonic_ns: int | None = None
    timed_out = False
    containment_error: str | None = None
    runtime_observations: list[dict[str, Any]] = []
    runtime_observation_attempts: list[dict[str, Any]] = []
    runtime_observation_seen: set[tuple[int, int, str]] = set()
    runtime_hash_cache: dict[tuple[int, ...], str] = {}
    callback_error: str | None = None
    runtime_observer_error: str | None = None
    monitoring_error_caused_termination = False
    process_control_error_caused_termination = False
    next_runtime_observation_ns: int | None = None
    baseline: set[int] = set()
    stdout = ""
    stderr = ""
    completed_result: dict[str, Any] | None = None

    def invoke_poll_callback(root_pid: int) -> bool:
        nonlocal callback_error
        if poll_callback is None or callback_error is not None:
            return callback_error is None
        try:
            poll_callback(root_pid)
            return True
        except BaseException as error:
            callback_error = f"{type(error).__name__}: {error}"
            return False

    def sample_runtime_processes(root_pid: int) -> bool:
        nonlocal runtime_observer_error
        if runtime_observer_error is not None:
            return False
        sample_start_ns = time.monotonic_ns()
        error_text: str | None = None
        process_count = 0
        try:
            new_observations, process_count = _observe_runtime_processes(
                root_pid,
                runtime_observation_seen,
                runtime_hash_cache,
            )
            runtime_observations.extend(new_observations)
        except BaseException as error:
            error_text = f"{type(error).__name__}: {error}"
            runtime_observer_error = error_text
            return False
        finally:
            runtime_observation_attempts.append(
                {
                    "start_monotonic_ns": sample_start_ns,
                    "duration_ns": time.monotonic_ns() - sample_start_ns,
                    "error": error_text,
                    "process_count": process_count,
                }
            )
        return True

    def build_result() -> dict[str, Any]:
        finished_ns = time.monotonic_ns()
        return {
            "command": command,
            "executable": executable,
            "environment": dict(sorted(environment.items())),
            "cwd": str(cwd),
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "process_wall_seconds": time.monotonic() - start,
            "process_started_monotonic_ns": process_started_monotonic_ns,
            "process_finished_monotonic_ns": finished_ns,
            "returncode": None if process is None else process.poll(),
            "root_pid": None if process is None else process.pid,
            "timed_out": timed_out,
            "containment_error": containment_error,
            "callback_error": callback_error,
            "runtime_observer_error": runtime_observer_error,
            "monitoring_error_caused_termination": (
                monitoring_error_caused_termination
            ),
            "process_control_error_caused_termination": (
                process_control_error_caused_termination
            ),
            "runtime_observations": runtime_observations,
            "runtime_observation_attempts": runtime_observation_attempts,
            "timeout_seconds": timeout_seconds,
            "stdout": stdout,
            "stderr": stderr,
        }
    try:
        baseline = _process_descendants(os.getpid())
        if baseline:
            raise EvidenceError("process containment found pre-existing descendants")
        probe = _pidfd_open(os.getpid())
        try:
            _pidfd_send_signal(probe, 0)
        finally:
            _close_native_owned_descriptor_once("controller self pidfd", probe)
        process_started_monotonic_ns = time.monotonic_ns()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
            executable=executable,
            pass_fds=pass_fds,
        )
        next_runtime_observation_ns = time.monotonic_ns() + 50_000_000
        invoke_poll_callback(process.pid)
        if observe_runtime:
            sample_runtime_processes(process.pid)
        deadline = time.monotonic() + timeout_seconds
        timeout_exception: subprocess.TimeoutExpired | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                stdout, stderr = process.communicate(
                    timeout=(
                        min(0.005 if poll_callback is not None else 0.05, remaining)
                        if observe_runtime or poll_callback is not None
                        else remaining
                    )
                )
                break
            except subprocess.TimeoutExpired as exc:
                timeout_exception = exc
                invoke_poll_callback(process.pid)
                now_ns = time.monotonic_ns()
                if (
                    observe_runtime
                    and next_runtime_observation_ns is not None
                    and now_ns >= next_runtime_observation_ns
                ):
                    sample_runtime_processes(process.pid)
                    while next_runtime_observation_ns <= now_ns:
                        next_runtime_observation_ns += 50_000_000
                if callback_error is not None or runtime_observer_error is not None:
                    break
                if observe_runtime or poll_callback is not None:
                    continue
                break
        monitoring_error = callback_error or runtime_observer_error
        if process.poll() is None and monitoring_error is not None:
            monitoring_error_caused_termination = True
            survivors, errors = _contain_process_tree(process, baseline, terminate=True)
            try:
                stdout, stderr = process.communicate(timeout=PROCESS_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            if survivors or errors:
                containment_error = (
                    "callback-error containment failed; survivors="
                    + ",".join(str(pid) for pid in sorted(survivors))
                    + "; errors=" + " | ".join(errors)
                )
        elif process.poll() is None:
            exc = timeout_exception or subprocess.TimeoutExpired(command, timeout_seconds)
            timed_out = True
            survivors, errors = _contain_process_tree(process, baseline, terminate=True)
            try:
                stdout, stderr = process.communicate(timeout=PROCESS_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
            if survivors or errors:
                containment_error = (
                    "timeout containment failed; survivors="
                    + ",".join(str(pid) for pid in sorted(survivors))
                    + "; errors="
                    + " | ".join(errors)
                )
        else:
            # Give a lane monitor one unscheduled post-wait observation so its
            # worker pidfds are committed before the root-exit timestamp.
            invoke_poll_callback(process.pid)
            if observe_runtime:
                sample_runtime_processes(process.pid)
            leaked = _snapshot_pidfds(baseline, do_not_reap={process.pid})
            leaked_pids = set(leaked)
            _close_pidfds(leaked)
            if leaked_pids:
                survivors, errors = _contain_process_tree(
                    process, baseline, terminate=False
                )
                containment_error = (
                    "command left surviving descendants; observed="
                    + ",".join(str(pid) for pid in sorted(leaked_pids))
                    + "; survivors="
                    + ",".join(str(pid) for pid in sorted(survivors))
                    + "; errors="
                    + " | ".join(errors)
                )
        completed_result = build_result()
        return completed_result
    except ProcessExecutionError:
        raise
    except BaseException as process_error:
        if process is None:
            raise
        emergency_errors: list[str] = []
        try:
            process_was_live = process.poll() is None
            if process_was_live:
                process_control_error_caused_termination = True
            survivors, errors = _contain_process_tree(
                process, baseline, terminate=process_was_live
            )
            if survivors or errors:
                containment_error = (
                    "exception containment survivors="
                    + ",".join(str(pid) for pid in sorted(survivors))
                    + "; errors=" + " | ".join(errors)
                )
        except BaseException as emergency_error:
            emergency_errors.append(
                f"containment: {type(emergency_error).__name__}: {emergency_error}"
            )
        try:
            recovered_stdout, recovered_stderr = process.communicate(
                timeout=PROCESS_GRACE_SECONDS
            )
            stdout = recovered_stdout or stdout
            stderr = recovered_stderr or stderr
        except BaseException as drain_error:
            emergency_errors.append(
                f"stream drain: {type(drain_error).__name__}: {drain_error}"
            )
            for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                if stream is None:
                    continue
                try:
                    recovered = stream.read()
                    if name == "stdout" and recovered:
                        stdout += recovered
                    elif name == "stderr" and recovered:
                        stderr += recovered
                except BaseException as read_error:
                    emergency_errors.append(
                        f"{name} read: {type(read_error).__name__}: {read_error}"
                    )
        completed_result = build_result()
        if emergency_errors:
            process_error.add_note("; ".join(emergency_errors))
        raise ProcessExecutionError(
            f"post-spawn process execution failed: {type(process_error).__name__}: {process_error}",
            completed_result,
            process_error,
        ) from process_error
    finally:
        cleanup_failures: list[str] = []
        if process is not None:
            try:
                survivors, errors = _contain_process_tree(
                    process, baseline, terminate=process.poll() is None
                )
                if survivors or errors:
                    cleanup_failures.append(
                        "emergency containment survivors="
                        + ",".join(str(pid) for pid in sorted(survivors))
                        + "; errors=" + " | ".join(errors)
                    )
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    "emergency containment raised "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        if not previous_subreaper:
            try:
                restore_diagnostics = _restore_child_subreaper_state(
                    previous_subreaper, PROCESS_GRACE_SECONDS
                )
                cleanup_failures.extend(
                    f"subreaper restore: {item}"
                    for item in restore_diagnostics
                )
                if _UNRESOLVED_SUBREAPER_TARGET is not None:
                    cleanup_failures.append(
                        "subreaper restore remains unresolved"
                    )
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    "subreaper restore raised "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        if cleanup_failures:
            message = "; ".join(cleanup_failures)
            active_error = sys.exc_info()[1]
            if isinstance(active_error, ProcessExecutionError):
                prior = active_error.result.get("containment_error")
                active_error.result["containment_error"] = (
                    message if not prior else f"{prior}; {message}"
                )
                active_error.add_note(message)
            elif active_error is not None:
                active_error.add_note(message)
            elif completed_result is not None:
                prior = completed_result.get("containment_error")
                completed_result["containment_error"] = (
                    message if not prior else f"{prior}; {message}"
                )
                cleanup_error = EvidenceError(message)
                raise ProcessExecutionError(
                    "process cleanup failed after output capture",
                    completed_result,
                    cleanup_error,
                ) from cleanup_error
            else:
                raise EvidenceError(message)


def _contain_subreaper_descendants(
    baseline: set[int], timeout_seconds: float,
) -> tuple[bool, list[str]]:
    """Kill/reap every post-baseline child epoch to a stable empty fixed point."""

    tracked: dict[int, dict[str, int | None]] = {}
    observed_any = False
    errors: list[str] = []
    reported_errors: set[str] = set()
    empty_rounds = 0

    def record_error(key: str, message: str) -> None:
        if key not in reported_errors:
            reported_errors.add(key)
            errors.append(message)

    def current_descendants() -> tuple[set[int], bool]:
        try:
            return _process_descendants(os.getpid()) - baseline, True
        except BaseException as error:
            record_error(
                "enumeration",
                "descendant enumeration raised "
                f"{type(error).__name__}: {error}",
            )
        try:
            return _process_table_descendants(os.getpid()) - baseline, True
        except BaseException as error:
            record_error(
                "fallback-enumeration",
                "fallback descendant enumeration raised "
                f"{type(error).__name__}: {error}",
            )
            return set(tracked), False

    def close_pidfd(pid: int, descriptor: int, *, final: bool = False) -> bool:
        closed, diagnostics = _close_descriptor_verified(descriptor)
        for diagnostic in diagnostics:
            record_error(
                f"{'final-' if final else ''}pidfd-close-{pid}-{diagnostic}",
                f"descendant pidfd close for {pid}: {diagnostic}",
            )
        if not closed:
            record_error(
                f"{'final-' if final else ''}pidfd-close-unresolved-{pid}",
                f"descendant pidfd {descriptor} for {pid} remains open",
            )
        return closed

    def direct_kill(pid: int, start_time_ticks: int | None) -> None:
        try:
            observed_start = _process_start_time_ticks(pid)
        except BaseException as error:
            observed_start = None
            record_error(
                f"start-read-kill-{pid}",
                "descendant start-time read raised "
                f"{type(error).__name__} for {pid}: {error}",
            )
        if start_time_ticks is not None and observed_start != start_time_ticks:
            if observed_start is not None:
                record_error(
                    f"epoch-{pid}",
                    f"descendant PID {pid} changed epoch before direct kill",
                )
                return
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as error:
            record_error(
                f"direct-kill-{pid}",
                "descendant direct kill raised "
                f"{type(error).__name__} for {pid}: {error}",
            )

    def reap_if_dead(pid: int, start_time_ticks: int | None) -> bool:
        try:
            observed_pid, _status = _waitpid_nohang(pid)
            return observed_pid == pid
        except InterruptedError as error:
            record_error(
                f"wait-interrupted-{pid}",
                f"descendant waitpid remained interrupted for {pid}: {error}",
            )
            return False
        except ChildProcessError:
            try:
                current_start = _process_start_time_ticks(pid)
            except BaseException as error:
                current_start = start_time_ticks
                record_error(
                    f"start-read-wait-{pid}",
                    "descendant post-wait start-time read raised "
                    f"{type(error).__name__} for {pid}: {error}",
                )
            if current_start is None:
                return True
            record_error(
                f"not-waitable-{pid}",
                f"adopted descendant {pid} is not waitable",
            )
            return False
        except BaseException as error:
            record_error(
                f"wait-{pid}",
                "descendant waitpid raised "
                f"{type(error).__name__} for {pid}: {error}",
            )
            direct_kill(pid, start_time_ticks)
            return False

    def track_pids(current: set[int]) -> int:
        added = 0
        for pid in sorted(current):
            if pid in tracked:
                continue
            try:
                start_time_ticks = _process_start_time_ticks(pid)
            except BaseException as error:
                start_time_ticks = None
                record_error(
                    f"start-read-open-{pid}",
                    "descendant initial start-time read raised "
                    f"{type(error).__name__} for {pid}: {error}",
                )
            try:
                descriptor: int | None = _pidfd_open(
                    pid, expected_start_time_ticks=start_time_ticks
                )
            except ProcessLookupError:
                continue
            except BaseException as error:
                descriptor = None
                record_error(
                    f"pidfd-open-{pid}",
                    "descendant pidfd_open raised "
                    f"{type(error).__name__} for {pid}: {error}",
                )
            tracked[pid] = {
                "descriptor": descriptor,
                "start_time_ticks": start_time_ticks,
            }
            added += 1
        return added

    def terminate_and_reap_tracked() -> bool:
        for pid, record in tuple(tracked.items()):
            descriptor = record["descriptor"]
            if (
                descriptor is not None
                and descriptor in _POISONED_PROBE_DESCRIPTOR_NUMBERS
            ):
                record_error(
                    f"poisoned-pidfd-{pid}",
                    f"descendant pidfd {descriptor} for {pid} is poisoned; "
                    "using PID/start-epoch fallback",
                )
                record["descriptor"] = None
                descriptor = None
            if descriptor is not None:
                try:
                    _pidfd_send_signal(descriptor, signal.SIGKILL)
                    continue
                except ProcessLookupError:
                    pass
                except BaseException as error:
                    record_error(
                        f"pidfd-kill-{pid}",
                        "descendant pidfd kill raised "
                        f"{type(error).__name__} for {pid}: {error}",
                    )
            direct_kill(pid, record["start_time_ticks"])

        active = False
        for pid, record in tuple(tracked.items()):
            descriptor = record["descriptor"]
            ready = descriptor is None
            if descriptor is not None:
                try:
                    poller = select.poll()
                    poller.register(descriptor, select.POLLIN)
                    ready = bool(poller.poll(0))
                except BaseException as error:
                    record_error(
                        f"pidfd-poll-{pid}",
                        "descendant pidfd poll raised "
                        f"{type(error).__name__} for {pid}: {error}",
                    )
                    direct_kill(pid, record["start_time_ticks"])
                    ready = True
            if not ready or not reap_if_dead(pid, record["start_time_ticks"]):
                active = True
                continue
            if descriptor is not None and not close_pidfd(pid, descriptor):
                active = True
                continue
            tracked.pop(pid, None)
        return active

    try:
        deadline = time.monotonic() + timeout_seconds
    except BaseException as error:
        deadline = None
        record_error(
            "deadline-init",
            "descendant containment deadline initialization raised "
            f"{type(error).__name__}: {error}",
        )

    try:
        while deadline is not None:
            try:
                if time.monotonic() >= deadline:
                    break
            except BaseException as error:
                record_error(
                    "deadline-read",
                    "descendant containment deadline read raised "
                    f"{type(error).__name__}: {error}",
                )
                break
            current, enumeration_ok = current_descendants()
            if current:
                observed_any = True
            added = track_pids(current)
            active = terminate_and_reap_tracked()
            remaining, remaining_enumeration_ok = current_descendants()
            if (
                enumeration_ok
                and remaining_enumeration_ok
                and not remaining
                and not tracked
                and added == 0
                and not active
            ):
                empty_rounds += 1
                if empty_rounds >= 3:
                    break
            else:
                empty_rounds = 0
            try:
                time.sleep(0.005)
            except BaseException as error:
                record_error(
                    "containment-sleep",
                    "descendant containment sleep raised "
                    f"{type(error).__name__}: {error}",
                )
    except BaseException as error:
        # The unconditional finalizer below retains the local PID/FD registry
        # and runs even if an otherwise-unanticipated Python exception occurs.
        record_error(
            "containment-pass",
            "descendant containment pass raised "
            f"{type(error).__name__}: {error}",
        )
    finally:
        # Fixed-count emergency rounds avoid depending on the clock/sleep paths
        # that may have caused the primary containment pass to abort.
        final_empty_rounds = 0
        for _attempt in range(64):
            try:
                current, enumeration_ok = current_descendants()
                if current:
                    observed_any = True
                added = track_pids(current)
                active = terminate_and_reap_tracked()
                remaining, remaining_enumeration_ok = current_descendants()
                if (
                    enumeration_ok
                    and remaining_enumeration_ok
                    and not remaining
                    and not tracked
                    and added == 0
                    and not active
                ):
                    final_empty_rounds += 1
                    if final_empty_rounds >= 3:
                        break
                else:
                    final_empty_rounds = 0
            except BaseException as error:
                final_empty_rounds = 0
                record_error(
                    "containment-finalizer",
                    "descendant containment finalizer raised "
                    f"{type(error).__name__}: {error}",
                )
            try:
                time.sleep(0.005)
            except BaseException as error:
                record_error(
                    "containment-finalizer-sleep",
                    "descendant containment finalizer sleep raised "
                    f"{type(error).__name__}: {error}",
                )

    try:
        survivors, final_enumeration_ok = current_descendants()
    except BaseException as error:
        survivors = set(tracked)
        final_enumeration_ok = False
        record_error(
            "final-enumeration-escape",
            "final descendant enumeration raised "
            f"{type(error).__name__}: {error}",
        )
    if not final_enumeration_ok:
        record_error(
            "unverified-empty",
            "subreaper descendant emptiness could not be verified",
        )
    if survivors:
        record_error(
            "timeout",
            "subreaper descendant containment timed out; survivors="
            + ",".join(str(pid) for pid in sorted(survivors)),
        )
    if tracked:
        record_error(
            "unresolved-tracked-ownership",
            "subreaper containment retained unresolved PID/pidfd ownership="
            + ",".join(str(pid) for pid in sorted(tracked)),
        )
    for pid, record in tuple(tracked.items()):
        descriptor = record["descriptor"]
        if descriptor is not None:
            for _attempt in range(8):
                try:
                    closed = close_pidfd(pid, descriptor, final=True)
                except BaseException as error:
                    closed = False
                    record_error(
                        f"final-close-escape-{pid}",
                        "final descendant pidfd close raised "
                        f"{type(error).__name__} for {pid}: {error}",
                    )
                if closed:
                    tracked.pop(pid, None)
                    break
    return observed_any, errors


def _probe_nvml_loadability(
    library: StableFile, *, cdll_factory: Callable[..., Any] = ctypes.CDLL,
) -> None:
    """Boundedly load exact held NVML bytes outside the controller process."""

    global _UNRESOLVED_SUBREAPER_TARGET

    if threading.active_count() != 1:
        raise EvidenceError("NVML loadability probe requires a single-threaded controller")
    if _POISONED_PROBE_DESCRIPTOR_NUMBERS:
        raise EvidenceError(
            "NVML probe cannot start after ambiguous descriptor close: "
            + ",".join(
                str(item)
                for item in sorted(_POISONED_PROBE_DESCRIPTOR_NUMBERS)
            )
        )
    if _UNRESOLVED_SUBREAPER_TARGET is not None:
        restore_diagnostics = _restore_child_subreaper_state(
            _UNRESOLVED_SUBREAPER_TARGET,
            SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS,
        )
        if _UNRESOLVED_SUBREAPER_TARGET is not None:
            error = EvidenceError(
                "NVML probe cannot start with unresolved child-subreaper state"
            )
            if restore_diagnostics:
                error.add_note("; ".join(restore_diagnostics))
            raise error
    if _RETAINED_PROBE_DESCRIPTORS:
        retained_diagnostics = _drain_retained_probe_descriptors()
        if _RETAINED_PROBE_DESCRIPTORS:
            error = EvidenceError(
                "NVML probe cannot start with unresolved descriptor ownership"
            )
            if retained_diagnostics:
                error.add_note("; ".join(retained_diagnostics))
            raise error
    library.verify()
    baseline = _process_descendants(os.getpid())
    if baseline:
        raise EvidenceError("NVML loadability probe found pre-existing descendants")
    previous_subreaper = _child_subreaper_state()
    subreaper_changed = False
    ready_read: int | None = None
    ready_write: int | None = None
    child_pid: int | None = None
    child_pidfd: int | None = None
    child_reaped = False
    child_status: int | None = None
    group_ready = False
    readiness_error: BaseException | None = None
    primary_error: BaseException | None = None
    cleanup_errors: list[str] = []
    cleanup_error_keys: set[str] = set()
    descendants_observed = False

    def record_cleanup_error(key: str, message: str) -> None:
        if key not in cleanup_error_keys:
            cleanup_error_keys.add(key)
            cleanup_errors.append(message)

    def refresh_group_ready() -> None:
        nonlocal group_ready, readiness_error
        if group_ready or ready_read is None:
            return
        for _attempt in range(8):
            try:
                marker = os.read(ready_read, 1)
            except InterruptedError:
                continue
            except BlockingIOError:
                return
            except BaseException as error:
                if readiness_error is None:
                    readiness_error = error
                return
            if marker == b"R":
                group_ready = True
            return
        if readiness_error is None:
            readiness_error = EvidenceError(
                "NVML readiness read remained interrupted"
            )

    def kill_exact_child() -> None:
        if child_pid is None or child_reaped:
            return
        try:
            if child_pidfd is not None:
                try:
                    _pidfd_send_signal(child_pidfd, signal.SIGKILL)
                    return
                except BaseException as error:
                    record_cleanup_error(
                        "exact-child-pidfd-kill",
                        "NVML exact-child pidfd kill raised "
                        f"{type(error).__name__}: {error}",
                    )
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as error:
            record_cleanup_error(
                "exact-child-direct-kill",
                f"NVML exact-child kill raised {type(error).__name__}: {error}",
            )

    def reap_exact_child(deadline: float | None) -> None:
        nonlocal child_reaped, child_status
        if child_pid is None or child_reaped:
            return
        clock_healthy = deadline is not None
        for attempt in range(1024):
            if child_reaped:
                return
            if clock_healthy:
                try:
                    if time.monotonic() >= deadline:
                        break
                except BaseException as error:
                    clock_healthy = False
                    record_cleanup_error(
                        "exact-child-reap-clock",
                        "NVML exact-child reap clock raised "
                        f"{type(error).__name__}: {error}",
                    )
            elif attempt >= 64:
                break
            # Retry termination so transient pidfd/direct-kill failures cannot
            # lead to an unbounded blocking wait.
            kill_exact_child()
            try:
                observed_pid, observed_status = _waitpid_nohang(child_pid)
                if observed_pid == child_pid:
                    child_status = observed_status
                    child_reaped = True
                    return
            except InterruptedError:
                continue
            except ChildProcessError:
                child_reaped = True
                return
            except BaseException as error:
                record_cleanup_error(
                    "exact-child-waitpid",
                    "NVML exact-child waitpid raised "
                    f"{type(error).__name__}: {error}",
                )
            try:
                time.sleep(0.005)
            except BaseException as error:
                record_cleanup_error(
                    "exact-child-reap-sleep",
                    "NVML exact-child reap sleep raised "
                    f"{type(error).__name__}: {error}",
                )
        if not child_reaped:
            record_cleanup_error(
                "exact-child-reap-timeout",
                "NVML exact-child bounded termination/reap timed out",
            )

    def close_probe_descriptor(label: str, descriptor: int) -> None:
        closed, diagnostics = _close_descriptor_verified(descriptor)
        for diagnostic in diagnostics:
            record_cleanup_error(
                f"{label}-close-{diagnostic}",
                f"NVML {label} close: {diagnostic}",
            )
        if not closed:
            record_cleanup_error(
                f"{label}-close-unresolved",
                f"NVML {label} descriptor {descriptor} remains open",
            )

    try:
        if not previous_subreaper:
            # Mark restoration responsibility before the setter: a setter can
            # change kernel state and then raise.
            subreaper_changed = True
            _set_child_subreaper(True)
            if not _raw_child_subreaper_state():
                raise EvidenceError(
                    "NVML child-subreaper enable did not reach the kernel"
                )
        ready_read, ready_write = _allocate_probe_pipe()
        os.set_inheritable(ready_read, False)
        os.set_inheritable(ready_write, False)
        os.set_blocking(ready_read, False)
        child_pid = os.fork()
        if child_pid == 0:
            try:
                os.close(ready_read)
                os.setsid()
                if os.write(ready_write, b"R") != 1:
                    raise OSError("short NVML probe readiness write")
                os.close(ready_write)
                loaded = cdll_factory(
                    f"/proc/self/fd/{library.descriptor}", mode=ctypes.RTLD_LOCAL
                )
                for symbol_name in NVML_REQUIRED_SYMBOLS:
                    getattr(loaded, symbol_name)
                if _process_descendants(os.getpid()):
                    raise EvidenceError("NVML loadability probe created descendants")
            except BaseException:
                os._exit(1)
            os._exit(0)

        _close_native_owned_descriptor_once(
            "NVML readiness parent write end", ready_write
        )
        ready_write = None
        try:
            child_pidfd = _pidfd_open(child_pid)
        except ProcessLookupError:
            child_pidfd = None
        except OSError as error:
            raise EvidenceError("NVML loadability probe could not bind child pidfd") from error

        deadline = time.monotonic() + SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS
        while True:
            refresh_group_ready()
            if readiness_error is not None:
                raise readiness_error
            try:
                observed_pid, child_status = _waitpid_nohang(child_pid)
            except InterruptedError:
                observed_pid = 0
            if observed_pid == child_pid:
                child_reaped = True
                break
            if time.monotonic() >= deadline:
                raise EvidenceError("NVML loadability probe timed out")
            time.sleep(0.005)
    except BaseException as error:
        primary_error = error
    finally:
        try:
            cleanup_deadline: float | None = (
                time.monotonic() + SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS
            )
        except BaseException as error:
            cleanup_deadline = None
            record_cleanup_error(
                "cleanup-deadline",
                "NVML cleanup deadline initialization raised "
                f"{type(error).__name__}: {error}",
            )
        # Exact-child termination is deliberately first: optional handshake
        # reads and group operations can fail without skipping the bound epoch.
        if child_pid not in (None, 0) and not child_reaped:
            kill_exact_child()
        refresh_group_ready()
        if child_pid not in (None, 0) and group_ready:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException as error:
                record_cleanup_error(
                    "probe-group-kill",
                    "NVML probe-group kill raised "
                    f"{type(error).__name__}: {error}",
                )
        reap_exact_child(cleanup_deadline)
        try:
            for label, descriptor in (
                ("readiness read", ready_read),
                ("readiness write", ready_write),
                ("child pidfd", child_pidfd),
            ):
                if descriptor is None:
                    continue
                try:
                    close_probe_descriptor(label, descriptor)
                except BaseException as error:
                    record_cleanup_error(
                        f"{label}-close-stage",
                        f"NVML {label} close stage raised "
                        f"{type(error).__name__}: {error}",
                    )
        except BaseException as error:
            record_cleanup_error(
                "descriptor-close-stage",
                "NVML descriptor-close stage raised "
                f"{type(error).__name__}: {error}",
            )
        try:
            if subreaper_changed or previous_subreaper:
                observed, errors = _contain_subreaper_descendants(
                    baseline, SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS
                )
                descendants_observed = observed
                cleanup_errors.extend(errors)
        except BaseException as error:
            record_cleanup_error(
                "descendant-containment",
                "NVML descendant containment raised "
                f"{type(error).__name__}: {error}",
            )
        finally:
            try:
                retained_diagnostics = _drain_retained_probe_descriptors()
                for diagnostic in retained_diagnostics:
                    record_cleanup_error(
                        f"retained-descriptor-{diagnostic}", diagnostic
                    )
                if _RETAINED_PROBE_DESCRIPTORS:
                    record_cleanup_error(
                        "retained-descriptor-unresolved",
                        "NVML cleanup retains unresolved descriptor ownership: "
                        + ",".join(
                            str(item)
                            for item in sorted(_RETAINED_PROBE_DESCRIPTORS)
                        ),
                    )
                if _POISONED_PROBE_DESCRIPTOR_NUMBERS:
                    record_cleanup_error(
                        "poisoned-descriptor-unresolved",
                        "NVML cleanup has ambiguous poisoned descriptor numbers: "
                        + ",".join(
                            str(item)
                            for item in sorted(
                                _POISONED_PROBE_DESCRIPTOR_NUMBERS
                            )
                        ),
                    )
            except BaseException as error:
                record_cleanup_error(
                    "retained-descriptor-drain",
                    "NVML retained-descriptor drain raised "
                    f"{type(error).__name__}: {error}",
                )
            if subreaper_changed:
                try:
                    restore_diagnostics = _restore_child_subreaper_state(
                        previous_subreaper,
                        SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS,
                    )
                    for diagnostic in restore_diagnostics:
                        record_cleanup_error(
                            f"subreaper-restore-{diagnostic}",
                            f"NVML child-subreaper restore: {diagnostic}",
                        )
                except BaseException as error:
                    _UNRESOLVED_SUBREAPER_TARGET = previous_subreaper
                    record_cleanup_error(
                        "subreaper-restore-stage",
                        "NVML child-subreaper restore stage raised "
                        f"{type(error).__name__}: {error}",
                    )
                    try:
                        _raw_set_child_subreaper(previous_subreaper)
                        if _raw_child_subreaper_state() == previous_subreaper:
                            _UNRESOLVED_SUBREAPER_TARGET = None
                    except BaseException as fallback_error:
                        record_cleanup_error(
                            "subreaper-restore-stage-fallback",
                            "NVML independent subreaper restore stage raised "
                            f"{type(fallback_error).__name__}: {fallback_error}",
                        )
                if _UNRESOLVED_SUBREAPER_TARGET is not None:
                    record_cleanup_error(
                        "subreaper-restore-unresolved",
                        "NVML child-subreaper state remains unresolved",
                    )

    if primary_error is None and readiness_error is not None:
        primary_error = readiness_error
    if cleanup_errors:
        message = "; ".join(cleanup_errors)
        if primary_error is not None:
            primary_error.add_note(message)
            raise primary_error
        raise EvidenceError(f"NVML loadability cleanup failed: {message}")
    if descendants_observed:
        descendant_error = EvidenceError(
            "NVML loadability probe created contained descendant processes"
        )
        if primary_error is not None:
            primary_error.add_note(str(descendant_error))
            raise primary_error
        raise descendant_error
    if primary_error is not None:
        raise primary_error
    if (
        child_status is None
        or not os.WIFEXITED(child_status)
        or os.WEXITSTATUS(child_status) != 0
    ):
        raise EvidenceError(
            "NVML exact held image failed isolated load/required-symbol probe"
        )
    library.verify()


def _probe_nvidia_smi_startup(
    tool: StableFile, *, cwd: pathlib.Path = pathlib.Path("/")
) -> None:
    """Start exact held tool bytes with their held trusted interpreter."""

    if len(tool.companions) > 1:
        raise EvidenceError("nvidia-smi has an ambiguous held interpreter")
    interpreter = tool.companions[0] if tool.companions else None
    tool.verify()
    if interpreter is None:
        command = ["nvidia-smi", "--help"]
        executable = tool.proc_path
        pass_fds = (tool.descriptor,)
    else:
        interpreter.verify()
        command = [interpreter.path.name, tool.proc_path, "--help"]
        executable = interpreter.proc_path
        pass_fds = (tool.descriptor, interpreter.descriptor)
    result = _run_process(
        command=command,
        environment={
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
        cwd=cwd,
        timeout_seconds=SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS,
        executable=executable,
        pass_fds=pass_fds,
    )
    if (
        result["timed_out"]
        or result.get("containment_error")
        or result.get("callback_error")
        or result.get("runtime_observer_error")
        or result["returncode"] != 0
        or not result["stdout"].startswith("NVIDIA System Management Interface -- v")
        or "nvidia-smi [OPTION" not in result["stdout"]
        or "--help" not in result["stdout"]
        or not result["stdout"].strip()
        or result["stderr"].strip()
        or len(result["stdout"].encode("utf-8")) > 1 << 20
        or len(result["stderr"].encode("utf-8")) > 1 << 20
    ):
        raise EvidenceError("nvidia-smi exact held image failed bounded startup probe")
    tool.verify()


def _nvidia_smi_command(
    tool: StableFile, arguments: list[str]
) -> tuple[list[str], str, tuple[int, ...]]:
    if len(tool.companions) > 1:
        raise EvidenceError("nvidia-smi has an ambiguous held interpreter")
    if not tool.companions:
        return ["nvidia-smi", *arguments], tool.proc_path, (tool.descriptor,)
    interpreter = tool.companions[0]
    return (
        [interpreter.path.name, tool.proc_path, *arguments],
        interpreter.proc_path,
        (tool.descriptor, interpreter.descriptor),
    )


def _nvidia_smi_inventory_arguments() -> list[str]:
    return [
        "--query-gpu=index,uuid,name,pci.bus_id,compute_cap,driver_version,pstate,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]


def _parse_telemetry_inventory(text: str) -> list[dict[str, Any]]:
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    result: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 1):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 15:
            raise EvidenceError(f"telemetry row {row_number} does not contain 15 columns")
    # Reuse the exact legacy identity parser once all five-column rows exist.
    identity_stream = io.StringIO()
    identity_writer = csv.writer(identity_stream, lineterminator="\n")
    identity_writer.writerows(
        item[:5] for item in rows if item and any(value.strip() for value in item)
    )
    identities = parse_device_inventory(identity_stream.getvalue())
    identity_by_uuid = {item["uuid"]: item for item in identities}
    for row_number, row in enumerate(rows, 1):
        if not row or all(not item.strip() for item in row):
            continue
        values = [item.strip() for item in row]
        if len(values) != 15:
            raise EvidenceError(f"telemetry row {row_number} does not contain 15 columns")
        identity = dict(identity_by_uuid[values[1]])
        if not values[5] or not re.fullmatch(r"P(?:[0-9]|1[0-5])", values[6]):
            raise EvidenceError(f"telemetry row {row_number} has invalid driver/P-state")
        numeric_names = (
            "graphics_clock_mhz", "memory_clock_mhz", "power_draw_w",
            "power_limit_w", "temperature_c", "utilization_percent",
            "memory_used_mib", "memory_total_mib",
        )
        numeric_values = []
        for name, value in zip(numeric_names, values[7:]):
            if not re.fullmatch(CANONICAL_UNSIGNED_DECIMAL, value):
                raise EvidenceError(f"telemetry row {row_number} {name} is not canonical")
            numeric_values.append(
                _require_nonnegative_float(float(value), f"telemetry row {row_number} {name}")
            )
        identity.update(
            {
                "driver_version": values[5],
                "pstate": values[6],
                **dict(zip(numeric_names, numeric_values)),
            }
        )
        result.append(identity)
    return sorted(result, key=lambda item: item["index"])


def _parse_compute_processes(text: str) -> list[dict[str, Any]]:
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    result: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 1):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != 4:
            raise EvidenceError(f"compute-process row {row_number} is malformed")
        gpu_uuid, pid_text, process_name, used_text = (item.strip() for item in row)
        if not UUID_PATTERN.fullmatch(gpu_uuid) or not process_name:
            raise EvidenceError(f"compute-process row {row_number} has invalid identity")
        if not re.fullmatch(CANONICAL_UNSIGNED_DECIMAL, used_text):
            raise EvidenceError(f"compute-process row {row_number} memory is not canonical")
        result.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": _parse_exact_int(pid_text, "compute-process pid"),
                "process_name": process_name,
                "used_gpu_memory_mib": _require_nonnegative_float(
                    float(used_text), "compute-process memory"
                )
            }
        )
    return result


class NvmlProcessInfoV3(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


class NvmlProcessUtilizationSample(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("timestamp_us", ctypes.c_ulonglong),
        ("sm_utilization", ctypes.c_uint),
        ("memory_utilization", ctypes.c_uint),
        ("encoder_utilization", ctypes.c_uint),
        ("decoder_utilization", ctypes.c_uint),
    ]


class NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlSampler:
    """Direct, receipt-recorded NVML sampling with no hot-path hashing."""

    SUCCESS = 0
    NOT_SUPPORTED = 3
    NOT_FOUND = 6
    INSUFFICIENT_SIZE = 7

    def __init__(
        self,
        library: StableFile,
        device_uuids: list[str],
        *,
        cdll_factory: Callable[..., Any] = ctypes.CDLL,
    ) -> None:
        if ctypes.sizeof(NvmlProcessInfoV3) != 24 or (
            NvmlProcessInfoV3.used_gpu_memory.offset,
            NvmlProcessInfoV3.gpu_instance_id.offset,
            NvmlProcessInfoV3.compute_instance_id.offset,
        ) != (8, 16, 20):
            raise EvidenceError("NVML v3 process ABI layout differs from expected LP64")
        if ctypes.sizeof(NvmlProcessUtilizationSample) != 32 or (
            NvmlProcessUtilizationSample.timestamp_us.offset,
            NvmlProcessUtilizationSample.sm_utilization.offset,
        ) != (8, 16):
            raise EvidenceError("NVML process-utilization ABI layout differs from expected LP64")
        self.library_file = library
        self.library_identity = library.record()
        self.library = cdll_factory(library.proc_path, mode=ctypes.RTLD_LOCAL)
        self.closed = False

        def bind(name: str, arguments: list[Any]) -> Any:
            try:
                function = getattr(self.library, name)
            except AttributeError as exc:
                raise EvidenceError(f"NVML symbol is unavailable: {name}") from exc
            function.argtypes = arguments
            function.restype = ctypes.c_int
            return function

        device = ctypes.c_void_p
        self.init = bind("nvmlInit_v2", [])
        self.shutdown = bind("nvmlShutdown", [])
        self.handle_by_uuid = bind(
            "nvmlDeviceGetHandleByUUID",
            [ctypes.c_char_p, ctypes.POINTER(device)],
        )
        process_arguments = [
            device,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(NvmlProcessInfoV3),
        ]
        self.process_functions = {
            "compute": bind("nvmlDeviceGetComputeRunningProcesses_v3", process_arguments),
            "graphics": bind("nvmlDeviceGetGraphicsRunningProcesses_v3", process_arguments),
            "mps": bind("nvmlDeviceGetMPSComputeRunningProcesses_v3", process_arguments),
        }
        self.process_utilization = bind(
            "nvmlDeviceGetProcessUtilization",
            [
                device,
                ctypes.POINTER(NvmlProcessUtilizationSample),
                ctypes.POINTER(ctypes.c_uint),
                ctypes.c_ulonglong,
            ],
        )
        self.utilization = bind(
            "nvmlDeviceGetUtilizationRates",
            [device, ctypes.POINTER(NvmlUtilization)],
        )
        self.clock = bind(
            "nvmlDeviceGetClockInfo",
            [device, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
        )
        self.power = bind(
            "nvmlDeviceGetPowerUsage", [device, ctypes.POINTER(ctypes.c_uint)]
        )
        self.temperature = bind(
            "nvmlDeviceGetTemperature",
            [device, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
        )
        self.pstate = bind(
            "nvmlDeviceGetPerformanceState",
            [device, ctypes.POINTER(ctypes.c_uint)],
        )
        self.memory = bind(
            "nvmlDeviceGetMemoryInfo", [device, ctypes.POINTER(NvmlMemory)]
        )
        self._require_success(self.init(), "nvmlInit_v2")
        try:
            self.handles: dict[str, ctypes.c_void_p] = {}
            self.utilization_cursors: dict[str, int] = {}
            self.baseline_process_utilization: dict[str, list[dict[str, int]]] = {}
            self.baseline_current_processes: dict[
                str, dict[str, list[dict[str, int]]]
            ] = {}
            for uuid_value in device_uuids:
                handle = device()
                self._require_success(
                    self.handle_by_uuid(uuid_value.encode("ascii"), ctypes.byref(handle)),
                    f"NVML handle lookup for {uuid_value}",
                )
                self.handles[uuid_value] = handle
                current = {
                    name: self._process_rows(function, handle)
                    for name, function in self.process_functions.items()
                }
                self.baseline_current_processes[uuid_value] = current
                if any(rows for rows in current.values()):
                    raise EvidenceError(
                        "NVML baseline current process lists are not empty"
                    )
                baseline = self._process_utilization_rows(handle, 0)
                self.baseline_process_utilization[uuid_value] = baseline
                if baseline:
                    raise EvidenceError(
                        "NVML baseline process-utilization history is not empty"
                    )
                self.utilization_cursors[uuid_value] = max(
                    (row["timestamp_us"] for row in baseline), default=0
                )
        except BaseException as initialization_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                initialization_error.add_note(
                    "NVML cleanup after initialization failure also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def _require_success(self, result: int, label: str) -> None:
        if result != self.SUCCESS:
            raise EvidenceError(f"{label} failed with NVML status {result}")

    def _process_rows(self, function: Any, handle: ctypes.c_void_p) -> list[dict[str, int]]:
        count = ctypes.c_uint(0)
        result = function(handle, ctypes.byref(count), None)
        if result == self.SUCCESS and count.value == 0:
            return []
        if result != self.INSUFFICIENT_SIZE or count.value == 0:
            raise EvidenceError(f"NVML process size query failed with status {result}")
        capacity = count.value
        for _attempt in range(3):
            records = (NvmlProcessInfoV3 * capacity)()
            count = ctypes.c_uint(capacity)
            result = function(handle, ctypes.byref(count), records)
            if result == self.SUCCESS:
                if count.value > capacity:
                    raise EvidenceError("NVML process retrieval exceeded its capacity")
                if any(records[index].pid == 0 for index in range(count.value)):
                    raise EvidenceError("NVML process retrieval returned PID zero")
                if len({records[index].pid for index in range(count.value)}) != count.value:
                    raise EvidenceError("NVML process retrieval returned duplicate PIDs")
                return [
                    {
                        "pid": int(records[index].pid),
                        "used_gpu_memory_bytes": int(records[index].used_gpu_memory),
                        "gpu_instance_id": int(records[index].gpu_instance_id),
                        "compute_instance_id": int(records[index].compute_instance_id),
                    }
                    for index in range(count.value)
                ]
            if result != self.INSUFFICIENT_SIZE or count.value <= capacity:
                raise EvidenceError(f"NVML process retrieval failed with status {result}")
            capacity = count.value
        raise EvidenceError("NVML process retrieval did not stabilize")

    def _process_utilization_rows(
        self, handle: ctypes.c_void_p, cursor: int
    ) -> list[dict[str, int]]:
        count = ctypes.c_uint(0)
        result = self.process_utilization(
            handle, None, ctypes.byref(count), ctypes.c_ulonglong(cursor)
        )
        if result in (self.SUCCESS, self.NOT_FOUND) and count.value == 0:
            return []
        if result != self.INSUFFICIENT_SIZE or count.value == 0:
            raise EvidenceError(
                f"NVML process-utilization size query failed with status {result}"
            )
        capacity = count.value
        for _attempt in range(3):
            records = (NvmlProcessUtilizationSample * capacity)()
            count = ctypes.c_uint(capacity)
            result = self.process_utilization(
                handle, records, ctypes.byref(count), ctypes.c_ulonglong(cursor)
            )
            if result == self.NOT_FOUND:
                return []
            if result == self.SUCCESS:
                if count.value > capacity:
                    raise EvidenceError(
                        "NVML process-utilization retrieval exceeded its capacity"
                    )
                coordinates = {
                    (records[index].pid, records[index].timestamp_us)
                    for index in range(count.value)
                }
                if (
                    len(coordinates) != count.value
                    or any(
                        records[index].pid == 0
                        or records[index].timestamp_us <= cursor
                        or max(
                            records[index].sm_utilization,
                            records[index].memory_utilization,
                            records[index].encoder_utilization,
                            records[index].decoder_utilization,
                        ) > 100
                        for index in range(count.value)
                    )
                ):
                    raise EvidenceError(
                        "NVML process-utilization rows violate identity/range/cursor gates"
                    )
                return [
                    {
                        "pid": int(records[index].pid),
                        "timestamp_us": int(records[index].timestamp_us),
                        "sm_utilization_percent": int(records[index].sm_utilization),
                        "memory_utilization_percent": int(records[index].memory_utilization),
                        "encoder_utilization_percent": int(records[index].encoder_utilization),
                        "decoder_utilization_percent": int(records[index].decoder_utilization),
                    }
                    for index in range(count.value)
                ]
            if result != self.INSUFFICIENT_SIZE or count.value <= capacity:
                raise EvidenceError(
                    f"NVML process-utilization retrieval failed with status {result}"
                )
            capacity = count.value
        raise EvidenceError("NVML process-utilization retrieval did not stabilize")

    def _scalar(self, function: Any, handle: ctypes.c_void_p, *prefix: int) -> int:
        value = ctypes.c_uint()
        self._require_success(
            function(handle, *prefix, ctypes.byref(value)), "NVML scalar query"
        )
        return int(value.value)

    def sample(
        self,
        *,
        allowed_process_epochs_by_uuid: dict[str, dict[int, int]],
        scheduled_monotonic_ns: int,
        after_process_exit: bool = False,
        departed_processes_by_uuid: dict[str, set[int]] | None = None,
    ) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        devices: list[dict[str, Any]] = []
        outsiders: list[dict[str, Any]] = []
        departed_processes_by_uuid = departed_processes_by_uuid or {}
        for uuid_value, handle in self.handles.items():
            process_lists = {
                name: self._process_rows(function, handle)
                for name, function in self.process_functions.items()
            }
            utilization_rows = self._process_utilization_rows(
                handle, self.utilization_cursors[uuid_value]
            )
            if utilization_rows:
                self.utilization_cursors[uuid_value] = max(
                    self.utilization_cursors[uuid_value],
                    max(row["timestamp_us"] for row in utilization_rows),
                )
            allowed = allowed_process_epochs_by_uuid.get(uuid_value, {})
            for source, rows in process_lists.items():
                for row in rows:
                    pid = row["pid"]
                    expected_start = allowed.get(pid)
                    current_start = _process_start_time_ticks(pid)
                    row["observed_start_time_ticks"] = current_start
                    if (
                        source != "compute"
                        or expected_start is None
                        or (
                            current_start is None
                            and pid not in departed_processes_by_uuid.get(
                                uuid_value, set()
                            )
                        )
                        or (
                            current_start is not None
                            and current_start != expected_start
                        )
                    ):
                        outsiders.append(
                            {
                                "uuid": uuid_value,
                                "source": source,
                                "pid": pid,
                                "expected_start_time_ticks": expected_start,
                                "observed_start_time_ticks": current_start,
                            }
                        )
            for row in utilization_rows:
                pid = row["pid"]
                expected_start = allowed.get(pid)
                current_start = _process_start_time_ticks(pid)
                row["observed_start_time_ticks"] = current_start
                if expected_start is None or (
                    current_start is None and not after_process_exit
                    and pid not in departed_processes_by_uuid.get(uuid_value, set())
                ) or (
                    current_start is not None and current_start != expected_start
                ):
                    outsiders.append(
                        {
                            "uuid": uuid_value,
                            "source": "process-utilization",
                            "pid": pid,
                            "expected_start_time_ticks": expected_start,
                            "observed_start_time_ticks": current_start,
                        }
                    )
            rates = NvmlUtilization()
            self._require_success(
                self.utilization(handle, ctypes.byref(rates)),
                "NVML utilization query",
            )
            memory = NvmlMemory()
            self._require_success(
                self.memory(handle, ctypes.byref(memory)), "NVML memory query"
            )
            devices.append(
                {
                    "uuid": uuid_value,
                    "process_lists": process_lists,
                    "process_utilization_since_cursor": utilization_rows,
                    "utilization": {
                        "gpu_percent": int(rates.gpu),
                        "memory_percent": int(rates.memory),
                    },
                    "graphics_clock_mhz": self._scalar(self.clock, handle, 0),
                    "memory_clock_mhz": self._scalar(self.clock, handle, 2),
                    "power_mw": self._scalar(self.power, handle),
                    "temperature_c": self._scalar(self.temperature, handle, 0),
                    "pstate": self._scalar(self.pstate, handle),
                    "memory_bytes": {
                        "total": int(memory.total),
                        "free": int(memory.free),
                        "used": int(memory.used),
                    },
                }
            )
        finished_ns = time.monotonic_ns()
        sample = {
            "scheduled_monotonic_ns": scheduled_monotonic_ns,
            "started_monotonic_ns": started_ns,
            "finished_monotonic_ns": finished_ns,
            "lateness_ns": max(0, started_ns - scheduled_monotonic_ns),
            "duration_ns": finished_ns - started_ns,
            "devices": devices,
            "outsiders": outsiders,
        }
        return sample

    def close(self) -> None:
        if not self.closed:
            result = self.shutdown()
            self._require_success(result, "nvmlShutdown")
            self.closed = True


def _derive_nvml_outsiders(
    nvml_sample: dict[str, Any],
    allowed_by_uuid: dict[str, dict[int, int]],
    *,
    after_process_exit: bool,
    departed_by_uuid: dict[str, set[int]] | None = None,
) -> list[dict[str, Any]]:
    departed_by_uuid = departed_by_uuid or {}
    outsiders: list[dict[str, Any]] = []
    for device in nvml_sample.get("devices", []):
        uuid_value = device.get("uuid")
        allowed = allowed_by_uuid.get(uuid_value, {})
        for source, rows in device.get("process_lists", {}).items():
            for row in rows:
                expected = allowed.get(row.get("pid"))
                observed = row.get("observed_start_time_ticks")
                if (
                    source != "compute"
                    or expected is None
                    or (
                        observed is None
                        and row.get("pid") not in departed_by_uuid.get(
                            uuid_value, set()
                        )
                    )
                    or (observed is not None and observed != expected)
                ):
                    outsiders.append(
                        {
                            "uuid": uuid_value,
                            "source": source,
                            "pid": row.get("pid"),
                            "expected_start_time_ticks": expected,
                            "observed_start_time_ticks": observed,
                        }
                    )
        for row in device.get("process_utilization_since_cursor", []):
            expected = allowed.get(row.get("pid"))
            observed = row.get("observed_start_time_ticks")
            if expected is None or (
                observed is None
                and not after_process_exit
                and row.get("pid") not in departed_by_uuid.get(uuid_value, set())
            ) or (
                observed is not None and observed != expected
            ):
                outsiders.append(
                    {
                        "uuid": uuid_value,
                        "source": "process-utilization",
                        "pid": row.get("pid"),
                        "expected_start_time_ticks": expected,
                        "observed_start_time_ticks": observed,
                    }
                )
    return outsiders


def _nvml_current_presence_exact(
    nvml_sample: dict[str, Any],
    allowed_by_uuid: dict[str, dict[int, int]],
    *,
    after_process_exit: bool,
) -> bool:
    devices = {
        device.get("uuid"): device for device in nvml_sample.get("devices", [])
    }
    if set(devices) != set(allowed_by_uuid):
        return False
    for uuid_value, allowed in allowed_by_uuid.items():
        lists = devices[uuid_value].get("process_lists", {})
        if lists.get("graphics") or lists.get("mps"):
            return False
        observed = {row.get("pid") for row in lists.get("compute", [])}
        expected = set() if after_process_exit else set(allowed)
        if observed != expected:
            return False
    return True


def _nvml_current_presence_with_departed_lag(
    nvml_sample: dict[str, Any],
    active_by_uuid: dict[str, dict[int, int]],
    departed_lag_by_uuid: dict[str, dict[int, int]],
    *,
    after_process_exit: bool,
) -> bool:
    if after_process_exit:
        return _nvml_current_presence_exact(
            nvml_sample, active_by_uuid, after_process_exit=True
        )
    devices = {
        device.get("uuid"): device for device in nvml_sample.get("devices", [])
    }
    if set(devices) != set(active_by_uuid):
        return False
    for uuid_value, active in active_by_uuid.items():
        lists = devices[uuid_value].get("process_lists", {})
        if lists.get("graphics") or lists.get("mps"):
            return False
        compute = lists.get("compute")
        if not isinstance(compute, list):
            return False
        observed = {
            row.get("pid"): row.get("observed_start_time_ticks") for row in compute
        }
        if len(observed) != len(compute):
            return False
        if any(observed.get(pid) != epoch for pid, epoch in active.items()):
            return False
        extras = set(observed) - set(active)
        lag = departed_lag_by_uuid.get(uuid_value, {})
        if any(
            pid not in lag or observed[pid] not in (None, lag[pid])
            for pid in extras
        ):
            return False
    return True


def _results_ready_precede_nvml_sample(
    records: Iterable[dict[str, Any]], nvml_sample: dict[str, Any]
) -> bool:
    started_ns = nvml_sample.get("started_monotonic_ns")
    records = list(records)
    return bool(
        records
        and _is_plain_int(started_ns)
        and all(
            _is_plain_int(record.get("observed_monotonic_ns"))
            and record["observed_monotonic_ns"] <= started_ns
            for record in records
        )
    )


def _within_departed_nvml_lag(
    nvml_finished_monotonic_ns: int, pidfd_observed_monotonic_ns: int,
    limit_ns: int = NVML_DEPARTED_CONTEXT_LAG_NS,
) -> bool:
    if not all(
        _is_plain_int(value)
        for value in (
            nvml_finished_monotonic_ns, pidfd_observed_monotonic_ns, limit_ns
        )
    ):
        return False
    delta = nvml_finished_monotonic_ns - pidfd_observed_monotonic_ns
    return limit_ns >= 0 and 0 <= delta <= limit_ns


def _normalized_machine(machine: str | None = None) -> str:
    value = (platform.machine() if machine is None else machine).lower()
    if value in {"amd64", "x86_64"}:
        return "x86_64"
    if value in {"i386", "i486", "i586", "i686", "x86"}:
        return "x86"
    if value in {"aarch64", "arm64"}:
        return "aarch64"
    if value.startswith("arm"):
        return "arm"
    return value


def _native_machine(
    machine: str | None = None, pointer_bits: int | None = None,
) -> str:
    """Return a supported 64-bit controller userspace ELF machine."""

    normalized = _normalized_machine(machine)
    bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    native_lp64 = (
        ctypes.sizeof(ctypes.c_void_p) == 8
        and ctypes.sizeof(ctypes.c_long) == 8
        and ctypes.sizeof(ctypes.c_uint) == 4
        and ctypes.sizeof(ctypes.c_ulonglong) == 8
    )
    if bits != 64 or pointer_bits is None and not native_lp64:
        raise EvidenceError(
            "LDOS physical evidence requires a 64-bit LP64 controller userspace"
        )
    if normalized not in {"x86_64", "aarch64"}:
        raise EvidenceError(
            f"LDOS physical evidence does not support controller machine {normalized!r}"
        )
    return normalized


def _nvml_library_candidates(
    machine: str | None = None, is_wsl: bool | None = None, *,
    pointer_bits: int | None = None,
) -> tuple[pathlib.Path, ...]:
    normalized = _native_machine(machine, pointer_bits)
    if is_wsl is None:
        is_wsl = _is_wsl_kernel()
    # Both supported LP64 machines use the same bounded WSL locations; the
    # still-open candidate's ELF machine check rejects the wrong architecture.
    wsl_candidates = NVML_LIBRARY_WSL_CANDIDATES
    ordered = (
        *(wsl_candidates if is_wsl else ()),
        *NVML_LIBRARY_PLATFORM_CANDIDATES.get(normalized, ()),
        *NVML_LIBRARY_GENERIC_CANDIDATES,
        *(wsl_candidates if not is_wsl else ()),
    )
    return tuple(dict.fromkeys(ordered))


def _is_wsl_kernel() -> bool:
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open("/proc/sys/kernel/osrelease", flags)
        data = os.read(descriptor, 4097)
        if len(data) > 4096:
            return False
        value = data.decode("ascii")
    except (OSError, UnicodeError):
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return "microsoft" in value.lower()


def _nvidia_smi_candidates(is_wsl: bool | None = None) -> tuple[pathlib.Path, ...]:
    if is_wsl is None:
        is_wsl = _is_wsl_kernel()
    if not is_wsl:
        return NVIDIA_SMI_CANDIDATES
    wsl = pathlib.Path("/usr/lib/wsl/lib/nvidia-smi")
    return (wsl, *(path for path in NVIDIA_SMI_CANDIDATES if path != wsl))


def _elf_identity(descriptor: int, size_bytes: int) -> dict[str, Any]:
    if (
        not _is_plain_int(size_bytes)
        or not 52 <= size_bytes <= SYSTEM_TOOL_ELF_MAX_SIZE_BYTES
    ):
        raise EvidenceError("system tool ELF header is truncated")
    identity = os.pread(descriptor, 16, 0)
    if (
        len(identity) != 16
        or identity[:4] != b"\x7fELF"
        or identity[6] != 1
    ):
        raise EvidenceError("system tool is not a supported ELF executable")
    elf_class = identity[4]
    encoding = identity[5]
    if elf_class not in (1, 2) or encoding not in (1, 2):
        raise EvidenceError("system tool ELF encoding is unsupported")
    endian = "<" if encoding == 1 else ">"
    if elf_class == 2:
        header_format = endian + "HHIQQQIHHHHHH"
        program_format = endian + "IIQQQQQQ"
        expected_header_size = 64
        expected_program_size = 56
    else:
        header_format = endian + "HHIIIIIHHHHHH"
        program_format = endian + "IIIIIIII"
        expected_header_size = 52
        expected_program_size = 32
    header_bytes = os.pread(descriptor, struct.calcsize(header_format), 16)
    if len(header_bytes) != struct.calcsize(header_format):
        raise EvidenceError("system tool ELF header is truncated")
    (
        elf_type, machine, version, entry, program_offset, _section_offset,
        _flags, header_size, program_entry_size, program_count,
        _section_entry_size, _section_count, _section_name_index,
    ) = struct.unpack(header_format, header_bytes)
    if (
        elf_type not in (2, 3)
        or version != 1
        or header_size != expected_header_size
        or program_offset < expected_header_size
        or program_entry_size != expected_program_size
        or not 1 <= program_count <= SYSTEM_TOOL_ELF_MAX_PROGRAM_HEADERS
        or program_offset > size_bytes
        or program_count > (size_bytes - program_offset) // program_entry_size
    ):
        raise EvidenceError("system tool ELF header/program table is invalid")
    table_size = program_count * program_entry_size
    program_bytes = os.pread(descriptor, table_size, program_offset)
    if len(program_bytes) != table_size:
        raise EvidenceError("system tool ELF program table is truncated")
    word_max = (1 << (64 if elf_class == 2 else 32)) - 1
    programs: list[dict[str, int]] = []
    for index in range(program_count):
        values = struct.unpack_from(
            program_format, program_bytes, index * program_entry_size
        )
        if elf_class == 2:
            kind, segment_flags, file_offset, virtual_address, _paddr, file_size, memory_size, alignment = values
        else:
            kind, file_offset, virtual_address, _paddr, file_size, memory_size, segment_flags, alignment = values
        if (
            file_offset > size_bytes
            or file_size > size_bytes - file_offset
            or file_size > memory_size
            or virtual_address > word_max - memory_size
            or alignment not in (0, 1)
            and (
                alignment & (alignment - 1)
                or virtual_address % alignment != file_offset % alignment
            )
        ):
            raise EvidenceError("system tool ELF program segment is outside the file")
        programs.append({
            "type": kind,
            "flags": segment_flags,
            "offset": file_offset,
            "virtual_address": virtual_address,
            "file_size": file_size,
            "memory_size": memory_size,
        })
    load_programs = [program for program in programs if program["type"] == 1]
    dynamic_programs = [program for program in programs if program["type"] == 2]
    interpreter_programs = [
        program for program in programs if program["type"] == 3
    ]
    executable_load_ranges = [
        (
            program["virtual_address"],
            program["virtual_address"] + program["file_size"],
        )
        for program in load_programs
        if program["file_size"] > 0 and program["flags"] & 1
    ]
    if (
        entry == 0
        or not any(start <= entry < stop for start, stop in executable_load_ranges)
    ):
        raise EvidenceError("system tool ELF has no executable file-backed PT_LOAD")
    if len(dynamic_programs) > 1:
        raise EvidenceError("system tool ELF has multiple PT_DYNAMIC segments")
    if len(interpreter_programs) > 1:
        raise EvidenceError("system tool ELF has multiple PT_INTERP segments")

    def program_inside_load(item: dict[str, int]) -> bool:
        for load in load_programs:
            memory_delta = item["virtual_address"] - load["virtual_address"]
            file_delta = item["offset"] - load["offset"]
            if (
                memory_delta >= 0
                and file_delta == memory_delta
                and item["memory_size"] <= load["memory_size"] - memory_delta
                and item["file_size"] <= load["file_size"] - file_delta
                and not item["flags"] & ~load["flags"]
            ):
                return True
        return False

    for program in (*dynamic_programs, *interpreter_programs):
        if program["file_size"] == 0 or not program_inside_load(program):
            raise EvidenceError(
                "system tool ELF auxiliary segment is outside compatible PT_LOAD"
            )
    interpreter_path: str | None = None
    if interpreter_programs:
        interpreter_program = interpreter_programs[0]
        interpreter_offset = interpreter_program["offset"]
        interpreter_size = interpreter_program["file_size"]
        if not 2 <= interpreter_size <= 4096:
            raise EvidenceError("system tool ELF PT_INTERP size is invalid")
        interpreter = os.pread(descriptor, interpreter_size, interpreter_offset)
        if (
            len(interpreter) != interpreter_size
            or not interpreter.endswith(b"\0")
            or b"\0" in interpreter[:-1]
        ):
            raise EvidenceError("system tool ELF PT_INTERP is malformed")
        try:
            interpreter_path = interpreter[:-1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise EvidenceError("system tool ELF PT_INTERP is not ASCII") from exc
        interpreter_components = interpreter_path.split("/")[1:]
        if (
            not interpreter_path.startswith("/")
            or "//" in interpreter_path
            or not interpreter_components
            or any(component in {"", ".", ".."} for component in interpreter_components)
        ):
            raise EvidenceError("system tool ELF PT_INTERP path is not absolute")
    names = {3: "x86", 40: "arm", 62: "x86_64", 183: "aarch64"}
    if machine not in names:
        raise EvidenceError("system tool ELF machine is unsupported")
    return {
        "machine": names[machine],
        "elf_class_bits": 64 if elf_class == 2 else 32,
        "endianness": "little" if encoding == 1 else "big",
        "elf_type": "ET_EXEC" if elf_type == 2 else "ET_DYN",
        "entry_point": entry,
        "program_header_count": program_count,
        "load_segment_count": len(load_programs),
        "dynamic_segment_count": len(dynamic_programs),
        "interpreter_path": interpreter_path,
    }


def _elf_machine(descriptor: int, size_bytes: int) -> str:
    """Compatibility accessor for tests and non-selection callers."""

    return _elf_identity(descriptor, size_bytes)["machine"]


def _require_native_elf(identity: dict[str, Any], label: str) -> None:
    if (
        identity.get("machine") != _native_machine()
        or identity.get("elf_class_bits") != struct.calcsize("P") * 8
        or identity.get("endianness") != sys.byteorder
    ):
        raise EvidenceError(
            f"{label} ELF machine/class/encoding differs from the controller platform"
        )


def _resolve_nvml() -> StableFile:
    # Establish the ABI floor before touching any candidate pathname.  The
    # ctypes NVML structures below intentionally implement the LP64 ABI only.
    _native_machine()
    rejection_notes: list[str] = []
    for candidate in _nvml_library_candidates():
        if not candidate.is_file():
            continue
        handle: StableFile | None = None
        try:
            handle = StableFile.open("candidate system NVML", candidate)
            if _system_runtime_tcb_handle(handle) is None:
                raise EvidenceError("candidate is outside the immutable system TCB")
            identity = _nvml_elf_identity(handle.descriptor, handle.size_bytes)
            _require_native_elf(identity, "candidate NVML")
            _probe_nvml_loadability(handle)
            handle.label = "trusted system NVML"
            selected = handle
            handle = None
            return selected
        except (OSError, EvidenceError) as error:
            if _POISONED_PROBE_DESCRIPTOR_NUMBERS:
                ownership_error = EvidenceError(
                    "NVML resolver has ambiguous poisoned descriptor ownership: "
                    + ",".join(
                        str(item)
                        for item in sorted(
                            _POISONED_PROBE_DESCRIPTOR_NUMBERS
                        )
                    )
                )
                ownership_error.add_note(str(error))
                raise ownership_error from error
            if _UNRESOLVED_SUBREAPER_TARGET is not None:
                restore_diagnostics = _restore_child_subreaper_state(
                    _UNRESOLVED_SUBREAPER_TARGET,
                    SYSTEM_CANDIDATE_PROBE_TIMEOUT_SECONDS,
                )
                if _UNRESOLVED_SUBREAPER_TARGET is not None:
                    ownership_error = EvidenceError(
                        "NVML resolver retains unresolved child-subreaper state"
                    )
                    ownership_error.add_note(str(error))
                    if restore_diagnostics:
                        ownership_error.add_note(
                            "; ".join(restore_diagnostics)
                        )
                    raise ownership_error from error
            if _RETAINED_PROBE_DESCRIPTORS:
                retained_diagnostics = _drain_retained_probe_descriptors()
                if _RETAINED_PROBE_DESCRIPTORS:
                    ownership_error = EvidenceError(
                        "NVML resolver retains unresolved descriptor ownership"
                    )
                    ownership_error.add_note(str(error))
                    if retained_diagnostics:
                        ownership_error.add_note(
                            "; ".join(retained_diagnostics)
                        )
                    raise ownership_error from error
            rejection_notes.append(
                f"{candidate}: {type(error).__name__}: {error}"
            )
        finally:
            if handle is not None:
                handle.close()
    error = EvidenceError(
        "no bounded libnvidia-ml.so.1 candidate passed TCB/ELF validation"
    )
    if rejection_notes:
        error.add_note("; ".join(rejection_notes))
    raise error


def _query_devices(
    tool: StableFile,
    home: pathlib.Path,
    timeout_seconds: int,
    *,
    output: pathlib.Path,
    label: str,
) -> dict[str, Any]:
    environment = {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    inventory_command, inventory_executable, inventory_descriptors = (
        _nvidia_smi_command(tool, _nvidia_smi_inventory_arguments())
    )
    try:
        result = _run_process(
            command=inventory_command,
            environment=environment,
            cwd=home,
            timeout_seconds=timeout_seconds,
            executable=inventory_executable,
            pass_fds=inventory_descriptors,
        )
    except ProcessExecutionError as error:
        _persist_failed_process_execution(
            output, f"device-telemetry-{label}-inventory-error", error
        )
        raise
    process_arguments = [
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    process_command, process_executable, process_descriptors = (
        _nvidia_smi_command(tool, process_arguments)
    )
    try:
        processes = _run_process(
            command=process_command,
            environment=environment,
            cwd=home,
            timeout_seconds=timeout_seconds,
            executable=process_executable,
            pass_fds=process_descriptors,
        )
    except ProcessExecutionError as error:
        _persist_failed_process_execution(
            output, f"device-telemetry-{label}-compute-error", error
        )
        raise
    raw = {"schema_version": 1, "inventory_query": result, "compute_process_query": processes}
    EVIDENCE.atomic_write_json(output / f"device-telemetry-{label}-raw.json", raw)
    for name, query in (("inventory", result), ("compute process", processes)):
        if query.get("containment_error"):
            raise EvidenceError(f"nvidia-smi {name} containment failed: {query['containment_error']}")
        if query["timed_out"] or query["returncode"] != 0:
            raise EvidenceError(f"nvidia-smi {name} query failed: {query['stderr']}")
        if query["stderr"].strip():
            raise EvidenceError(f"nvidia-smi {name} query emitted stderr")
    result["inventory"] = _parse_telemetry_inventory(result["stdout"])
    result["compute_processes"] = _parse_compute_processes(processes["stdout"])
    return result


def _prefix_from_mpiexec(path: pathlib.Path) -> pathlib.Path:
    parent = path.resolve().parent
    if parent.name != "bin":
        raise EvidenceError("receipt-bound mpiexec must reside below a prefix bin directory")
    return parent.parent


def _base_environment(
    *,
    output: pathlib.Path,
    prefix: pathlib.Path,
    archived_lib_directory: pathlib.Path,
    archived_mca: StableFile,
    selected_devices: list[dict[str, Any]],
) -> dict[str, str]:
    openmpi_components = prefix / "lib" / "openmpi"
    pmix_components = prefix / "lib" / "pmix"
    if not openmpi_components.is_dir() or not pmix_components.is_dir():
        raise EvidenceError("receipt MPI/PMIx component directories are absent")
    home = output / "home"
    home.mkdir(mode=0o700, exist_ok=True)
    if not home.is_dir():
        raise EvidenceError("isolated benchmark HOME is not a directory")
    temporary = output / "runtime-tmp"
    cache = output / "runtime-cache"
    temporary.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    return {
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "TMP": str(temporary),
        "TEMP": str(temporary),
        "XDG_CACHE_HOME": str(cache),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": f"{archived_lib_directory}:{prefix / 'lib'}",
        "OPAL_PREFIX": str(prefix),
        "OMP_NUM_THREADS": "1",
        "OMPI_MCA_mca_base_param_files": archived_mca.proc_path,
        "OMPI_MCA_mca_base_component_path": str(openmpi_components),
        "PMIX_MCA_mca_base_param_files": archived_mca.proc_path,
        "PMIX_MCA_mca_base_component_path": str(pmix_components),
        "PRTE_MCA_mca_base_param_files": archived_mca.proc_path,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(row["uuid"] for row in selected_devices),
        "MEEP_GPU_BACKEND": "cuda",
        "MEEP_GPU_STRICT": "1",
        "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
        "MEEP_GPU_MPI_TRANSPORT": "pinned",
        "MEEP_GPU_TEST_LDOS_BENCHMARK_ONLY": "1",
    }


def _mpi_command(
    timeout: StableFile,
    mpiexec: StableFile,
    loader: StableFile,
    worker: StableFile,
    ranks: int,
    timeout_seconds: int,
    library_path: str,
    worker_cpu_masks: list[set[int]],
) -> list[str]:
    if len(worker_cpu_masks) != ranks or any(not mask for mask in worker_cpu_masks):
        raise EvidenceError("MPI worker CPU-mask plan is invalid")
    if any(
        first & second
        for index, first in enumerate(worker_cpu_masks)
        for second in worker_cpu_masks[index + 1 :]
    ):
        raise EvidenceError("MPI worker CPU masks overlap")
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        f"{timeout_seconds}s",
        loader.proc_path,
        "--inhibit-rpath",
        "",
        "--library-path",
        library_path,
        mpiexec.proc_path,
        "--map-by",
        "slot",
        "--bind-to",
        "none",
    ]
    for rank in range(ranks):
        if rank:
            command.append(":")
        command.extend(
            (
                "-n",
                "1",
                "-x",
                f"MEEP_GPU_DEVICE={rank}",
                "-x",
                "MEEP_GPU_LDOS_CPU_LIST="
                + ",".join(str(cpu) for cpu in sorted(worker_cpu_masks[rank])),
                loader.proc_path,
                "--inhibit-rpath",
                "",
                "--library-path",
                library_path,
                worker.proc_path,
            )
        )
    return command


def _persist_process_result(
    output: pathlib.Path,
    label: str,
    result: dict[str, Any],
    parsed: dict[str, Any] | None,
    *,
    validation_error: str | None = None,
) -> dict[str, Any]:
    stdout_path = output / f"{label}.stdout.log"
    stderr_path = output / f"{label}.stderr.log"
    combined_path = output / f"{label}.log"
    record_path = output / f"{label}.json"
    _write_text_exclusive(stdout_path, result["stdout"])
    _write_text_exclusive(stderr_path, result["stderr"])
    combined = (
        "command: "
        + shlex.join(result["command"])
        + f"\nreturncode: {result['returncode']}\n"
        + f"process_wall_seconds: {result['process_wall_seconds']:.9f}\n"
        + "===== environment =====\n"
        + json.dumps(result["environment"], indent=2, sort_keys=True)
        + "\n===== stdout =====\n"
        + result["stdout"]
        + ("\n" if result["stdout"] and not result["stdout"].endswith("\n") else "")
        + "===== stderr =====\n"
        + result["stderr"]
        + ("\n" if result["stderr"] and not result["stderr"].endswith("\n") else "")
    )
    _write_text_exclusive(combined_path, combined)
    record = {
        "schema_version": 1,
        "label": label,
        "command": result["command"],
        "executable": result.get("executable"),
        "environment": result["environment"],
        "cwd": result["cwd"],
        "started_at_utc": result["started_at_utc"],
        "finished_at_utc": result["finished_at_utc"],
        "process_wall_seconds": result["process_wall_seconds"],
        "process_started_monotonic_ns": result.get(
            "process_started_monotonic_ns"
        ),
        "process_finished_monotonic_ns": result.get(
            "process_finished_monotonic_ns"
        ),
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
        "callback_error": result.get("callback_error"),
        "runtime_observer_error": result.get("runtime_observer_error"),
        "monitoring_error_caused_termination": result.get(
            "monitoring_error_caused_termination", False
        ),
        "process_control_error_caused_termination": result.get(
            "process_control_error_caused_termination", False
        ),
        "timeout_seconds": result["timeout_seconds"],
        "stdout": _relative_file_record(stdout_path, output),
        "stderr": _relative_file_record(stderr_path, output),
        "combined_log": _relative_file_record(combined_path, output),
        "benchmark": parsed,
        "validation_error": validation_error,
    }
    EVIDENCE.atomic_write_json(record_path, record)
    return {
        "record": record,
        "record_file": _relative_file_record(record_path, output),
    }


def _persist_failed_process_execution(
    output: pathlib.Path, label: str, error: ProcessExecutionError
) -> None:
    try:
        _persist_process_result(
            output,
            label,
            error.result,
            None,
            validation_error=str(error),
        )
    except BaseException as persistence_error:
        error.add_note(
            "partial process-result persistence raised "
            f"{type(persistence_error).__name__}: {persistence_error}"
        )


def _add_lane_failure_teardown_diagnostics(
    primary: EvidenceError, lane_monitor: LaneControlMonitor,
    root_pid: Any, finished_ns: Any,
) -> None:
    """Attach best-effort teardown facts without ever replacing the primary."""

    if not _is_plain_int(root_pid) or not _is_plain_int(finished_ns):
        primary.add_note(
            "teardown diagnostic: process identity/timestamp was unavailable"
        )
        return
    try:
        diagnostics = lane_monitor.diagnose_failure_teardown(root_pid, finished_ns)
    except BaseException as diagnostic_error:
        primary.add_note(
            "teardown diagnostic raised "
            f"{type(diagnostic_error).__name__}: {diagnostic_error}"
        )
        return
    for diagnostic in diagnostics:
        primary.add_note(f"teardown diagnostic: {diagnostic}")


def _raise_lane_result_primary(
    primary: EvidenceError, *, result: dict[str, Any],
    lane_monitor: LaneControlMonitor, output: pathlib.Path | None = None,
    label: str | None = None,
    failure_evidence: dict[str, Any] | None = None,
) -> None:
    _add_lane_failure_teardown_diagnostics(
        primary,
        lane_monitor,
        result.get("root_pid"),
        result.get("process_finished_monotonic_ns"),
    )
    if output is not None and label is not None:
        try:
            _persist_process_result(
                output,
                label,
                result,
                failure_evidence,
                validation_error=str(primary),
            )
            setattr(primary, "_lane_result_persisted", True)
        except BaseException as persistence_error:
            primary.add_note(
                "process-result persistence raised "
                f"{type(persistence_error).__name__}: {persistence_error}"
            )
    raise primary


def _raise_lane_process_execution_error(
    *, output: pathlib.Path, label: str, error: ProcessExecutionError,
    lane_monitor: LaneControlMonitor, ranks: int,
    expected_device_uuids: list[str],
    expected_worker_cpu_masks: list[set[int]],
) -> None:
    """Persist recovered evidence and raise the classified lane primary."""

    primary, failure_evidence = _classify_lane_process_execution_error(
        error,
        ranks=ranks,
        expected_device_uuids=expected_device_uuids,
        expected_worker_cpu_masks=expected_worker_cpu_masks,
        go_monotonic_ns=getattr(lane_monitor, "go_monotonic_ns", None),
        process_finished_monotonic_ns=error.result.get(
            "process_finished_monotonic_ns"
        ),
    )
    if primary is None:
        _persist_failed_process_execution(output, label, error)
        raise error
    _add_lane_failure_teardown_diagnostics(
        primary,
        lane_monitor,
        error.result.get("root_pid"),
        error.result.get("process_finished_monotonic_ns"),
    )
    try:
        _persist_process_result(
            output,
            label,
            error.result,
            failure_evidence,
            validation_error=str(primary),
        )
    except BaseException as persistence_error:
        primary.add_note(
            "process-result persistence raised "
            f"{type(persistence_error).__name__}: {persistence_error}"
        )
    raise primary from error


def _persist_attempt(
    *,
    output: pathlib.Path,
    label: str,
    command: list[str],
    environment: dict[str, str],
    cwd: pathlib.Path,
    timeout_seconds: int,
) -> pathlib.Path:
    path = output / f"{label}.attempt.json"
    if path.exists():
        raise EvidenceError(f"duplicate command attempt label: {label}")
    EVIDENCE.atomic_write_json(
        path,
        {
            "schema_version": 1,
            "state": "ATTEMPT",
            "label": label,
            "created_at_utc": _utc_now(),
            "command": command,
            "environment": dict(sorted(environment.items())),
            "cwd": str(cwd),
            "timeout_seconds": timeout_seconds,
        },
    )
    return path


def _relative_file_record(path: pathlib.Path, root: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink():
        raise EvidenceError(f"evidence file is a symbolic link: {path}")
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        relative = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"evidence file escapes the output root: {path}") from exc
    handle = StableFile.open("evidence artifact", resolved_path)
    try:
        return {
            "path": relative,
            "size_bytes": handle.size_bytes,
            "sha256": handle.sha256,
        }
    finally:
        handle.close()


def _duplicate_stable_file(source: StableFile, label: str) -> StableFile:
    source.verify()
    descriptor = os.dup(source.descriptor)
    try:
        os.set_inheritable(descriptor, False)
        info = os.fstat(descriptor)
        return StableFile(
            label=label,
            path=source.path,
            descriptor=descriptor,
            fingerprint=_fingerprint(info),
            sha256=source.sha256,
            size_bytes=source.size_bytes,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _archive_inputs(
    *,
    output: pathlib.Path,
    held_inputs: dict[str, StableFile],
    nvidia_smi: StableFile,
    nvml_library: StableFile,
) -> tuple[dict[str, StableFile], dict[str, StableFile]]:
    originals: dict[str, StableFile] = dict(held_inputs)
    archived: dict[str, StableFile] = {}
    if set(held_inputs) != {
        "receipt", "gpu_step_db", "libmeep", "fd_allocation_shim",
        "mpiexec", "timeout",
        "mca", "loader", "runtime_closure",
    }:
        raise EvidenceError("held receipt input set is incomplete")
    originals["nvidia_smi"] = nvidia_smi
    originals["nvml_library"] = nvml_library
    destinations = {
        "receipt": output / "archive" / "input" / "build-provenance.json",
        "gpu_step_db": output / "archive" / "runtime" / "bin" / "gpu-step-db",
        "libmeep": output / "archive" / "runtime" / "lib" / "libmeep.so.38.0.0",
        "fd_allocation_shim": (
            output
            / "archive"
            / "runtime"
            / "lib"
            / "libgpmeep-fd-allocation-shim.so.1"
        ),
        "mpiexec": output / "archive" / "runtime" / "bin" / "mpiexec",
        "timeout": output / "archive" / "runtime" / "bin" / "timeout",
        "mca": output / "archive" / "runtime" / "config" / "openmpi-qualification-mca-params.conf",
        "runtime_closure": output / "archive" / "runtime" / "config" / "runtime-dependency-closure.json",
        "nvidia_smi": output / "archive" / "runtime" / "bin" / "nvidia-smi",
        "nvml_library": (
            output / "archive" / "system" / "lib" / "libnvidia-ml.so.1"
        ),
        "loader": output / "archive" / "runtime" / "lib" / held_inputs["loader"].path.name,
        **{
            f"source_{name}": output / "archive" / "source" / name
            for name in SOURCE_HELPERS
        },
    }
    executable_labels = {
        "gpu_step_db",
        "mpiexec",
        "timeout",
        "nvidia_smi",
        "loader",
        "source_run-ldos-reduction-benchmark.py",
    }
    try:
        for name in SOURCE_HELPERS:
            source = CONTROLLER_SOURCE_HANDLES.get(name)
            if source is None:
                raise EvidenceError(f"held controller source is absent: {name}")
            originals[f"source_{name}"] = _duplicate_stable_file(
                source, f"held controller source archive input {name}"
            )
        for source in originals.values():
            source.verify()
        for label, source in originals.items():
            mode = 0o500 if label in executable_labels else 0o400
            archived[label] = _copy_stable_file(source, destinations[label], mode)
        lib_alias = output / "archive" / "runtime" / "lib" / "libmeep.so.38"
        os.link(destinations["libmeep"], lib_alias)
        if EVIDENCE.sha256_file(lib_alias) != archived["libmeep"].sha256:
            raise EvidenceError("archived libmeep SONAME alias is inconsistent")
        # Creating the SONAME hard link legitimately changes the inode ctime.
        # Refresh the held fingerprint only after the fixed alias exists.
        archived["libmeep"].close()
        archived["libmeep"] = StableFile.open(
            "archived libmeep", destinations["libmeep"]
        )
        archived["libmeep_soname"] = StableFile.open(
            "archived libmeep SONAME alias", lib_alias
        )
        manifest = {
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "originals": {label: handle.record() for label, handle in originals.items()},
            "snapshots": {label: handle.record() for label, handle in archived.items()},
            "libmeep_soname_alias": _relative_file_record(lib_alias, output),
        }
        EVIDENCE.atomic_write_json(output / "archive" / "archive-manifest.json", manifest)
        return originals, archived
    except BaseException:
        for handle in (*originals.values(), *archived.values()):
            handle.close()
        raise


def _artifact_records(output: pathlib.Path, excluded: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output).as_posix()
        if path.is_symlink():
            raise EvidenceError(f"symbolic links are not allowed in evidence: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise EvidenceError(f"non-regular evidence entry is not allowed: {relative}")
        if relative in excluded:
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise EvidenceError(f"non-regular evidence artifact is not allowed: {relative}")
        records.append(
            _relative_file_record(path, output)
        )
    return records


def _verify_relative_records(
    output: pathlib.Path, records: Iterable[dict[str, Any]]
) -> None:
    for expected in records:
        path_value = expected.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise EvidenceError("artifact manifest contains an invalid path")
        path = output / path_value
        if _relative_file_record(path, output) != expected:
            raise EvidenceError(f"evidence artifact changed before publication: {path_value}")


def _failure_artifact_records(
    output: pathlib.Path, excluded: set[str]
) -> list[dict[str, Any]]:
    """Describe partial state without following hostile or broken links.

    Success publication rejects every non-regular entry.  Failure publication
    must remain possible precisely when such an entry caused the rejection,
    so this diagnostic manifest uses lstat and hashes regular files only.
    """

    records: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        record: dict[str, Any] = {
            "path": relative,
            "size_bytes": info.st_size,
            "mode": info.st_mode,
        }
        if stat.S_ISREG(info.st_mode):
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags)
            except OSError as exc:
                raise EvidenceError(
                    f"partial evidence entry changed while opening: {relative}"
                ) from exc
            try:
                opened = os.fstat(descriptor)
                try:
                    named_after_open = path.lstat()
                except OSError as exc:
                    raise EvidenceError(
                        f"partial evidence entry disappeared while hashing: {relative}"
                    ) from exc
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _fingerprint(opened) != _fingerprint(info)
                    or _fingerprint(named_after_open) != _fingerprint(opened)
                ):
                    raise EvidenceError(
                        f"partial evidence entry changed while hashing: {relative}"
                    )
                digest = _sha256_fd(descriptor)
                os.fsync(descriptor)
                try:
                    named_after_hash = path.lstat()
                except OSError as exc:
                    raise EvidenceError(
                        f"partial evidence entry disappeared after hashing: {relative}"
                    ) from exc
                if (
                    _fingerprint(os.fstat(descriptor)) != _fingerprint(opened)
                    or _fingerprint(named_after_hash) != _fingerprint(opened)
                ):
                    raise EvidenceError(
                        f"partial evidence entry changed after hashing: {relative}"
                    )
            finally:
                os.close(descriptor)
            record.update(
                {
                    "kind": "regular",
                    "sha256": digest,
                    "device": opened.st_dev,
                    "inode": opened.st_ino,
                }
            )
        elif stat.S_ISLNK(info.st_mode):
            record.update({"kind": "symlink", "target": os.readlink(path)})
        else:
            record["kind"] = "non-regular"
        records.append(record)
    return records


def _revalidate_sample(output: pathlib.Path, sample: dict[str, Any]) -> dict[str, Any]:
    record_path = output / sample["record_file"]["path"]
    if _relative_file_record(record_path, output) != sample["record_file"]:
        raise EvidenceError("sample record changed before publication")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    for stream_name in ("stdout", "stderr", "combined_log"):
        stream_path = output / record[stream_name]["path"]
        if _relative_file_record(stream_path, output) != record[stream_name]:
            raise EvidenceError(f"sample {stream_name} changed before publication")
    stdout = (output / record["stdout"]["path"]).read_text(encoding="utf-8")
    stderr = (output / record["stderr"]["path"]).read_text(encoding="utf-8")
    parsed = parse_benchmark_output(
        stdout,
        stderr,
        returncode=record["returncode"],
        expected_ranks=record["benchmark"]["ranks"],
    )
    recorded_base = dict(record["benchmark"])
    verified_runtime = recorded_base.pop("verified_runtime_closure", None)
    verified_mpi_runtime = recorded_base.pop("mpi_runtime_execution", None)
    verified_host_runtime = recorded_base.pop("host_runtime_control", None)
    if parsed != recorded_base:
        raise EvidenceError("sample parser result changed before publication")
    if verified_runtime is not None:
        parsed["verified_runtime_closure"] = verified_runtime
    if verified_mpi_runtime is not None:
        parsed["mpi_runtime_execution"] = verified_mpi_runtime
    if verified_host_runtime is not None:
        raw_record = verified_host_runtime.get("raw_journal")
        if not isinstance(raw_record, dict) or not isinstance(
            raw_record.get("path"), str
        ):
            raise EvidenceError("sample raw telemetry journal binding is absent")
        raw_path = output / raw_record["path"]
        if _relative_file_record(raw_path, output) != raw_record:
            raise EvidenceError("sample raw telemetry journal changed before publication")
        revalidated_host = validate_lane_journal(
            raw_path,
            expected_header=verified_host_runtime.get("journal_header"),
            require_complete=True,
        )
        revalidated_host["raw_journal"] = raw_record
        if revalidated_host != verified_host_runtime:
            raise EvidenceError("sample host telemetry verdict changed before publication")
        _validate_success_host_timeline(
            parsed,
            revalidated_host,
            record.get("process_finished_monotonic_ns"),
        )
        parsed["host_runtime_control"] = revalidated_host
    return {
        "record_file": sample["record_file"],
        "benchmark": parsed,
    }


def _read_current_json_artifact(
    path: pathlib.Path,
    output: pathlib.Path,
    label: str,
    *,
    maximum_size: int = 64 * 1024 * 1024,
) -> tuple[Any, dict[str, Any]]:
    """Read one current-path JSON artifact through a held no-follow handle."""

    path = pathlib.Path(path)
    try:
        named_before = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"{label} current path is unavailable") from exc
    if (
        not stat.S_ISREG(named_before.st_mode)
        or named_before.st_nlink != 1
        or named_before.st_uid != os.geteuid()
    ):
        raise EvidenceError(f"{label} current path is not a single-link regular file")
    handle = StableFile.open(label, path)
    try:
        if handle.fingerprint != _fingerprint(named_before):
            raise EvidenceError(f"{label} current path changed while opening")
        if handle.size_bytes > maximum_size:
            raise EvidenceError(f"{label} exceeds its bounded JSON size")
        data = os.pread(handle.descriptor, handle.size_bytes + 1, 0)
        if len(data) != handle.size_bytes:
            raise EvidenceError(f"{label} changed while reading")
        handle.verify()
        try:
            named_after = path.lstat()
        except OSError as exc:
            raise EvidenceError(f"{label} current path disappeared") from exc
        opened_after = os.fstat(handle.descriptor)
        if (
            not stat.S_ISREG(named_after.st_mode)
            or named_after.st_nlink != 1
            or named_after.st_uid != os.geteuid()
            or not stat.S_ISREG(opened_after.st_mode)
            or opened_after.st_nlink != 1
            or opened_after.st_uid != os.geteuid()
            or _fingerprint(named_after) != handle.fingerprint
            or _fingerprint(opened_after) != handle.fingerprint
        ):
            raise EvidenceError(f"{label} current path changed after reading")
        try:
            value = json.loads(
                data.decode("utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {token}")
                ),
                object_pairs_hook=_unique_json_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise EvidenceError(f"{label} is not strict JSON") from exc
        relative = handle.path.relative_to(output.resolve(strict=True)).as_posix()
        return value, {
            "path": relative,
            "size_bytes": handle.size_bytes,
            "sha256": handle.sha256,
        }
    finally:
        handle.close()


def _verify_published_nvml_cross_binding(
    output: pathlib.Path,
    report: Any,
    records_by_path: dict[str, dict[str, Any]],
) -> None:
    """Bind current archive bytes to report and every raw lane journal."""

    archive_relative = "archive/system/lib/libnvidia-ml.so.1"
    archive_record = records_by_path.get(archive_relative)
    if archive_record is None:
        raise EvidenceError("artifact manifest omits the archived NVML library")
    if not isinstance(report, dict):
        raise EvidenceError("published report is not an object")
    device_binding = report.get("device_binding")
    samples = report.get("samples")
    if not isinstance(device_binding, dict) or not isinstance(samples, list) or not samples:
        raise EvidenceError("published report lacks NVML/raw-journal bindings")
    trusted_archive = device_binding.get("trusted_nvml_archive")
    if not (
        isinstance(trusted_archive, dict)
        and set(trusted_archive)
        == {"label", "path", "size_bytes", "sha256", "fingerprint"}
        and isinstance(trusted_archive["path"], str)
        and _is_plain_int(trusted_archive["size_bytes"])
        and isinstance(trusted_archive["sha256"], str)
        and SHA256_PATTERN.fullmatch(trusted_archive["sha256"])
        and isinstance(trusted_archive["fingerprint"], dict)
    ):
        raise EvidenceError("published report archived NVML record is invalid")

    archive_path = output / archive_relative
    archive_handle = StableFile.open("final current archived NVML", archive_path)
    try:
        current = archive_handle.record()
        if (
            pathlib.Path(trusted_archive["path"]).resolve(strict=True)
            != archive_handle.path
            or trusted_archive["size_bytes"] != current["size_bytes"]
            or trusted_archive["sha256"] != current["sha256"]
            or trusted_archive["fingerprint"] != current["fingerprint"]
            or archive_record
            != {
                "path": archive_relative,
                "size_bytes": current["size_bytes"],
                "sha256": current["sha256"],
            }
        ):
            raise EvidenceError("report/artifact-manifest archived NVML binding changed")
        semantic_identity = _nvml_archive_content_identity(archive_handle)
        archive_handle.verify()
    finally:
        archive_handle.close()

    archive_manifest, archive_manifest_record = _read_current_json_artifact(
        output / "archive" / "archive-manifest.json",
        output,
        "final archive manifest",
    )
    if (
        records_by_path.get("archive/archive-manifest.json")
        != archive_manifest_record
        or not isinstance(archive_manifest, dict)
        or not isinstance(archive_manifest.get("snapshots"), dict)
        or archive_manifest["snapshots"].get("nvml_library") != trusted_archive
    ):
        raise EvidenceError("archive manifest/report NVML binding changed")

    raw_paths: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("benchmark"), dict):
            raise EvidenceError("published sample lacks a benchmark object")
        host_control = sample["benchmark"].get("host_runtime_control")
        if not isinstance(host_control, dict):
            raise EvidenceError("published sample lacks host runtime control")
        raw_record = host_control.get("raw_journal")
        journal_header = host_control.get("journal_header")
        if not (
            isinstance(raw_record, dict)
            and set(raw_record) == {"path", "size_bytes", "sha256"}
            and isinstance(raw_record["path"], str)
            and isinstance(journal_header, dict)
            and journal_header.get("nvml_archive_identity") == semantic_identity
        ):
            raise EvidenceError("published raw journal/NVML identity binding is invalid")
        raw_path = raw_record["path"]
        if records_by_path.get(raw_path) != raw_record:
            raise EvidenceError("artifact manifest/raw journal binding changed")
        validated = validate_lane_journal(
            output / raw_path,
            expected_header=journal_header,
            require_complete=True,
        )
        if validated.get("journal_header") != journal_header:
            raise EvidenceError("raw journal header changed during final publication")
        raw_paths.append(raw_path)
    if len(raw_paths) != len(set(raw_paths)):
        raise EvidenceError("published report repeats a raw telemetry journal")


def _reverify_final_artifact_manifest(
    output: pathlib.Path,
    expected_manifest: dict[str, Any],
    expected_report: dict[str, Any],
) -> dict[str, Any]:
    """Reopen the finalized manifest and all files through current paths."""

    manifest, manifest_record = _read_current_json_artifact(
        output / "artifacts.sha256.json", output, "final artifact manifest"
    )
    if manifest != expected_manifest or not isinstance(manifest, dict):
        raise EvidenceError("final artifact manifest changed before publication")
    if set(manifest) != {
        "schema_version", "algorithm", "run_id", "profile_id", "records",
        "records_sha256",
    }:
        raise EvidenceError("final artifact manifest schema is invalid")
    records = manifest["records"]
    if (
        not isinstance(records, list)
        or any(
            not isinstance(record, dict)
            or set(record) != {"path", "size_bytes", "sha256"}
            for record in records
        )
        or manifest["records_sha256"] != PROVENANCE.canonical_sha256(records)
    ):
        raise EvidenceError("final artifact manifest records are invalid")
    records_by_path = {
        record["path"]: record
        for record in records
        if isinstance(record.get("path"), str)
    }
    if len(records_by_path) != len(records):
        raise EvidenceError("final artifact manifest paths are invalid or repeated")
    _verify_relative_records(output, records)
    current_report, report_record = _read_current_json_artifact(
        output / "report.json", output, "final report"
    )
    if (
        records_by_path.get("report.json") != report_record
        or PROVENANCE.canonical_sha256(current_report)
        != PROVENANCE.canonical_sha256(expected_report)
    ):
        raise EvidenceError("final report changed before publication")
    _verify_published_nvml_cross_binding(
        output, current_report, records_by_path
    )
    return manifest_record


def _current_success_running_marker(
    output: pathlib.Path, run_id: str
) -> dict[str, Any]:
    marker, _record = _read_current_json_artifact(
        output / "RUNNING.json", output, "success RUNNING marker"
    )
    if not (
        isinstance(marker, dict)
        and set(marker)
        == {"schema_version", "state", "run_id", "profile_id", "created_at_utc"}
        and marker["schema_version"] == 1
        and marker["state"] == "RUNNING"
        and marker["run_id"] == run_id
        and marker["profile_id"] == PROFILE_ID
        and isinstance(marker["created_at_utc"], str)
    ):
        raise EvidenceError("success RUNNING marker is not current")
    return marker


def _require_exact_success_files(
    output: pathlib.Path, expected: set[str]
) -> None:
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != expected:
        raise EvidenceError("unbound artifact appeared during success publication")


def _retire_success_running_marker(
    output: pathlib.Path, run_id: str, expected_marker: dict[str, Any]
) -> None:
    if _current_success_running_marker(output, run_id) != expected_marker:
        raise EvidenceError("RUNNING marker changed before success retirement")
    (output / "RUNNING.json").unlink()
    _fsync_directory(output)


def _publish_success(
    *, output: pathlib.Path, run_id: str, summary: dict[str, Any], report: dict[str, Any]
) -> None:
    running_marker = _current_success_running_marker(output, run_id)
    summary_path = output / "summary.json"
    report_path = output / "report.json"
    markdown_path = output / "report.md"
    EVIDENCE.atomic_write_json(summary_path, summary)
    EVIDENCE.atomic_write_json(report_path, report)
    markdown = (
        "# Receipt-bound LDOS reduction/transfer evidence\n\n"
        "- State: **COMPLETE**\n"
        f"- Receipt ID: `{report['build_receipt']['receipt_id']}`\n"
        "- Comparison: CUDA-resident LDOS reduction/transfer versus the forced "
        "legacy full-field host LDOS fallback in the same CUDA FDTD workload.\n"
        "- This is **not** end-to-end FDTD, CPU-vs-GPU, general-Meep, or "
        "multi-GPU-scaling speed evidence.\n\n"
        "| MPI ranks | Minimum paired speedup | Median paired speedup | Per-rank host/resident D2H |\n"
        "| ---: | ---: | ---: | --- |\n"
        + "".join(
            f"| {lane['ranks']} | {lane['minimum_speedup']:.9g}x | "
            f"{lane['median_speedup']:.9g}x | "
            + ", ".join(
                f"r{item['rank']}={item['host']}/{item['resident']}"
                for item in lane["per_rank_device_to_host_bytes"]
            )
            + " |\n"
            for lane in summary["lanes"]
        )
    )
    EVIDENCE.atomic_write_text(markdown_path, markdown)
    records = _artifact_records(
        output,
        {"RUNNING.json", "artifacts.sha256.json", "COMPLETE", "FAILED", "TERMINAL"},
    )
    manifest = {
        "schema_version": 1,
        "algorithm": "sha256",
        "run_id": run_id,
        "profile_id": PROFILE_ID,
        "records": records,
        "records_sha256": PROVENANCE.canonical_sha256(records),
    }
    manifest_path = output / "artifacts.sha256.json"
    EVIDENCE.atomic_write_json(manifest_path, manifest)
    _fsync_directory(output)
    manifest_record = _reverify_final_artifact_manifest(output, manifest, report)
    complete = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "profile_id": PROFILE_ID,
        "profile_sha256": PROFILE_SHA256,
        "claim_scope": report["claim_scope"],
        "receipt_id": report["build_receipt"]["receipt_id"],
        "summary": _relative_file_record(summary_path, output),
        "report": _relative_file_record(report_path, output),
        "artifact_manifest": manifest_record,
        "artifact_records_sha256": manifest["records_sha256"],
    }
    complete_path = output / "COMPLETE"
    EVIDENCE.atomic_write_json(complete_path, complete)
    _fsync_directory_tree(output)
    expected = {record["path"] for record in records} | {
        "artifacts.sha256.json",
        "COMPLETE",
        "RUNNING.json",
    }
    _require_exact_success_files(output, expected)
    manifest_record = _reverify_final_artifact_manifest(output, manifest, report)
    if manifest_record != complete["artifact_manifest"]:
        complete_path.unlink(missing_ok=True)
        raise EvidenceError("artifact manifest changed before terminal publication")
    current_complete, complete_record = _read_current_json_artifact(
        complete_path, output, "final COMPLETE authority"
    )
    if current_complete != complete:
        complete_path.unlink(missing_ok=True)
        raise EvidenceError("COMPLETE authority changed before terminal publication")
    terminal = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "authority": complete_record,
        "published_at_utc": _utc_now(),
    }
    EVIDENCE.atomic_write_json(output / "TERMINAL", terminal)
    _fsync_directory_tree(output)
    if _reverify_final_artifact_manifest(output, manifest, report) != manifest_record:
        raise EvidenceError("artifact manifest changed after terminal publication")
    final_complete, final_complete_record = _read_current_json_artifact(
        complete_path, output, "published COMPLETE authority"
    )
    final_terminal, _terminal_record = _read_current_json_artifact(
        output / "TERMINAL", output, "published TERMINAL authority"
    )
    if (
        final_complete != complete
        or final_complete_record != terminal["authority"]
        or final_terminal != terminal
    ):
        raise EvidenceError("published COMPLETE/TERMINAL authority changed")
    _require_exact_success_files(output, expected | {"TERMINAL"})
    if _current_success_running_marker(output, run_id) != running_marker:
        raise EvidenceError("RUNNING marker changed before success retirement")
    _retire_success_running_marker(output, run_id, running_marker)
    _require_exact_success_files(output, (expected - {"RUNNING.json"}) | {"TERMINAL"})
    retired_complete, retired_complete_record = _read_current_json_artifact(
        complete_path, output, "retired COMPLETE authority"
    )
    retired_terminal, _retired_terminal_record = _read_current_json_artifact(
        output / "TERMINAL", output, "retired TERMINAL authority"
    )
    if (
        retired_complete != complete
        or retired_complete_record != terminal["authority"]
        or retired_terminal != terminal
    ):
        raise EvidenceError("success authority changed during RUNNING retirement")


def _publish_failure(output: pathlib.Path, run_id: str, error: BaseException) -> None:
    terminal_path = output / "TERMINAL"
    running_path = output / "RUNNING.json"

    def quarantine(path: pathlib.Path) -> pathlib.Path | None:
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        while True:
            destination = output / f"superseded-{path.name}-{secrets.token_hex(16)}"
            try:
                destination.lstat()
            except FileNotFoundError:
                break
        os.replace(path, destination)
        _fsync_directory(output)
        return destination

    def read_running_marker() -> dict[str, Any] | None:
        try:
            named = running_path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1:
            return None
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(running_path, flags)
        except OSError:
            return None
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_uid != os.geteuid()
                or _fingerprint(opened) != _fingerprint(named)
            ):
                return None
            data = os.pread(descriptor, opened.st_size + 1, 0)
            if len(data) != opened.st_size:
                return None
            try:
                marker = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            if _fingerprint(running_path.lstat()) != _fingerprint(opened):
                return None
            return marker if isinstance(marker, dict) else None
        finally:
            os.close(descriptor)

    def running_marker_is_current(marker: dict[str, Any] | None) -> bool:
        return bool(
            marker is not None
            and set(marker)
            == {"schema_version", "state", "run_id", "profile_id", "created_at_utc"}
            and marker["schema_version"] == 1
            and marker["state"] == "RUNNING"
            and marker["run_id"] == run_id
            and marker["profile_id"] == PROFILE_ID
            and isinstance(marker["created_at_utc"], str)
        )

    def ensure_running_marker() -> None:
        if running_marker_is_current(read_running_marker()):
            return
        quarantine(running_path)
        marker = {
            "schema_version": 1,
            "state": "RUNNING",
            "run_id": run_id,
            "profile_id": PROFILE_ID,
            "created_at_utc": _utc_now(),
        }
        _write_text_exclusive(
            running_path,
            json.dumps(marker, separators=(",", ":"), sort_keys=True) + "\n",
            0o400,
        )
        _fsync_directory(output)
        if read_running_marker() != marker:
            raise EvidenceError("recreated RUNNING marker failed exact verification")

    try:
        # Do not follow or recursively delete attacker-created publication
        # entries. Renaming even a non-empty directory is recoverable and frees
        # the authoritative basename for a regular file.
        ensure_running_marker()
        quarantine(terminal_path)
        quarantine(output / "COMPLETE")
        quarantine(output / "FAILED")
        partial = _failure_artifact_records(
            output, {"RUNNING.json", "FAILED", "TERMINAL", "COMPLETE"}
        )
        _fsync_directory_tree_nofollow(output)
        if _failure_artifact_records(
            output, {"RUNNING.json", "FAILED", "TERMINAL", "COMPLETE"}
        ) != partial:
            raise EvidenceError(
                "partial evidence changed across failure durability barrier"
            )
        failure = {
            "schema_version": 1,
            "state": "FAILED",
            "run_id": run_id,
            "profile_id": PROFILE_ID,
            "profile_sha256": PROFILE_SHA256,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "partial_artifacts": partial,
            "partial_artifacts_sha256": PROVENANCE.canonical_sha256(partial),
            "failed_at_utc": _utc_now(),
        }
        failed_path = output / "FAILED"
        EVIDENCE.atomic_write_json(failed_path, failure)
        _fsync_directory(output)
        terminal = {
            "schema_version": 1,
            "state": "FAILED",
            "run_id": run_id,
            "authority": _relative_file_record(failed_path, output),
            "published_at_utc": _utc_now(),
        }
        EVIDENCE.atomic_write_json(terminal_path, terminal)
        _fsync_directory(output)
        _fsync_directory(output.parent)
        current_terminal, _terminal_record = _read_current_json_artifact(
            terminal_path, output, "FAILED terminal"
        )
        current_failure, current_failure_record = _read_current_json_artifact(
            failed_path, output, "FAILED authority"
        )
        if current_terminal != terminal:
            raise EvidenceError("FAILED terminal changed before RUNNING retirement")
        if (
            current_failure != failure
            or current_failure_record != terminal["authority"]
        ):
            raise EvidenceError("FAILED authority changed before RUNNING retirement")
        if not running_marker_is_current(read_running_marker()):
            raise EvidenceError("RUNNING marker changed before retirement")
        running_path.unlink()
        _fsync_directory(output)
        retired_terminal, _retired_terminal_record = _read_current_json_artifact(
            terminal_path, output, "retired FAILED terminal"
        )
        retired_failure, retired_failure_record = _read_current_json_artifact(
            failed_path, output, "retired FAILED authority"
        )
        if (
            retired_terminal != terminal
            or retired_failure != failure
            or retired_failure_record != terminal["authority"]
        ):
            raise EvidenceError("FAILED authority changed during RUNNING retirement")
    except BaseException as publication_error:
        recovery_errors: list[str] = []
        try:
            quarantine(terminal_path)
        except BaseException as recovery_error:
            recovery_errors.append(
                f"terminal quarantine: {type(recovery_error).__name__}: {recovery_error}"
            )
        try:
            ensure_running_marker()
            _fsync_directory(output)
            _fsync_directory(output.parent)
        except BaseException as recovery_error:
            recovery_errors.append(
                f"RUNNING recovery: {type(recovery_error).__name__}: {recovery_error}"
            )
        if recovery_errors:
            publication_error.add_note("; ".join(recovery_errors))
        raise


def _resolve_nvidia_smi() -> StableFile:
    # Fail before probing candidate paths when the controller cannot satisfy
    # the sampler/tool ABI floor.
    _native_machine()
    rejection_notes: list[str] = []
    for candidate in _nvidia_smi_candidates():
        if not candidate.is_file():
            continue
        handle: StableFile | None = None
        interpreter: StableFile | None = None
        try:
            handle = StableFile.open("candidate system nvidia-smi", candidate)
            if _trusted_system_tool_tcb_handle(handle) is None:
                raise EvidenceError(
                    "candidate is outside the immutable system-tool TCB"
                )
            identity = _elf_identity(handle.descriptor, handle.size_bytes)
            _require_native_elf(identity, "candidate nvidia-smi")
            interpreter_path = identity.get("interpreter_path")
            if interpreter_path is not None:
                interpreter = StableFile.open(
                    "candidate nvidia-smi interpreter",
                    pathlib.Path(interpreter_path),
                )
                if _trusted_system_tool_tcb_handle(interpreter) is None:
                    raise EvidenceError(
                        "candidate nvidia-smi interpreter is outside the immutable TCB"
                    )
                interpreter_identity = _elf_identity(
                    interpreter.descriptor, interpreter.size_bytes
                )
                _require_native_elf(
                    interpreter_identity, "candidate nvidia-smi interpreter"
                )
                if (
                    interpreter_identity["elf_type"] != "ET_DYN"
                    or interpreter_identity["interpreter_path"] is not None
                    or interpreter_identity["dynamic_segment_count"] != 1
                    or interpreter_identity["load_segment_count"] < 1
                ):
                    raise EvidenceError(
                        "candidate nvidia-smi interpreter is not a standalone dynamic loader"
                    )
                handle.companions = (interpreter,)
                interpreter = None
            _probe_nvidia_smi_startup(handle)
            handle.label = "trusted system nvidia-smi"
            selected = handle
            handle = None
            return selected
        except (OSError, EvidenceError) as error:
            rejection_notes.append(
                f"{candidate}: {type(error).__name__}: {error}"
            )
        finally:
            if interpreter is not None:
                interpreter.close()
            if handle is not None:
                handle.close()
    error = EvidenceError(
        "no bounded nvidia-smi candidate passed trusted-file validation"
    )
    if rejection_notes:
        error.add_note("; ".join(rejection_notes))
    raise error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-repo", type=pathlib.Path, required=True)
    parser.add_argument("--build-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--expected-receipt-id", required=True)
    parser.add_argument("--mpiexec", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--device-uuids",
        required=True,
        help="two comma-separated physical GPU UUIDs in rank-0/rank-1 order",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args(argv)


def _validate_invocation(
    args: argparse.Namespace,
) -> tuple[pathlib.Path, pathlib.Path]:
    repo = args.authoritative_repo.resolve(strict=True)
    if not (repo / ".git").exists():
        raise EvidenceError("--authoritative-repo is not a Git worktree")
    output = args.output.resolve(strict=False)
    if _path_is_within(output, repo):
        raise EvidenceError("evidence output must be outside the authoritative source tree")
    if not _is_plain_int(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise EvidenceError("--timeout-seconds must be a positive integer")
    if not output.parent.is_dir():
        raise EvidenceError("evidence output parent directory must already exist")
    return repo, output


def execute(
    args: argparse.Namespace,
    run_id: str,
    *,
    repo: pathlib.Path,
    output: pathlib.Path,
) -> int:

    global _FD_ALLOCATION_SHIM

    controller_identity_start = _verify_controller_identity()
    receipt_path = args.build_receipt.resolve(strict=True)
    validated = validate_receipt_inputs(
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=args.expected_receipt_id,
        mpiexec_path=args.mpiexec,
    )
    held_inputs: dict[str, StableFile] = {}
    nvidia_smi: StableFile | None = None
    nvml_library: StableFile | None = None
    nvml_system_tcb: dict[str, Any] | None = None
    originals: dict[str, StableFile] = {}
    archived: dict[str, StableFile] = {}
    closure_handles: dict[str, StableFile] = {}
    sentinel: MutationSentinel | None = None
    active_lane_monitor: LaneControlMonitor | None = None
    active_nvml_sampler: NvmlSampler | None = None
    controller_original_task_affinities: dict[str, list[int]] = {}
    controller_default_affinity: set[int] = set()
    controller_affinity_active = False
    try:
        held_inputs = open_validated_input_handles(validated)
        # nvidia-smi/NVML candidate probes themselves allocate pidfds, pipes,
        # and generation guards. Initialize from the already receipt-verified,
        # held artifact before either resolver can execute a candidate. The
        # archive is created only after those resolvers identify their inputs;
        # its exact copy is cross-bound immediately below.
        _initialize_fd_allocation_shim(held_inputs["fd_allocation_shim"])
        nvidia_smi = _resolve_nvidia_smi()
        nvml_library = _resolve_nvml()
        nvml_system_tcb = _system_runtime_tcb_handle(nvml_library)
        if nvml_system_tcb is None:
            raise EvidenceError("NVML library is outside the immutable system runtime TCB")
        originals, archived = _archive_inputs(
            output=output,
            held_inputs=held_inputs,
            nvidia_smi=nvidia_smi,
            nvml_library=nvml_library,
        )
        _bind_loaded_fd_allocation_shim_archive(
            held_inputs["fd_allocation_shim"],
            archived["fd_allocation_shim"],
        )
        archived_interpreters: list[StableFile] = []
        try:
            for index, interpreter in enumerate(nvidia_smi.companions):
                archived_interpreters.append(
                    _duplicate_stable_file(
                        interpreter,
                        f"held archived nvidia-smi interpreter {index}",
                    )
                )
            archived["nvidia_smi"].companions = tuple(archived_interpreters)
        except BaseException:
            for interpreter in archived_interpreters:
                interpreter.close()
            raise
        _probe_nvml_loadability(archived["nvml_library"])
        home = output / "home"
        home.mkdir(mode=0o700)
        _probe_nvidia_smi_startup(archived["nvidia_smi"], cwd=home)
        prefix = _prefix_from_mpiexec(validated["paths"]["mpiexec"])
        closure_handles = open_runtime_closure_handles(archived["runtime_closure"])
        sentinel = MutationSentinel(prefix)
        device_before = _query_devices(
            archived["nvidia_smi"],
            home,
            min(args.timeout_seconds, 60),
            output=output,
            label="before",
        )
        selected = select_devices(device_before["inventory"], args.device_uuids)
        require_quiescent_devices(device_before, selected, "benchmark start")
        environment = _base_environment(
            output=output,
            prefix=prefix,
            archived_lib_directory=output / "archive" / "runtime" / "lib",
            archived_mca=archived["mca"],
            selected_devices=selected,
        )
        device_before_path = output / "device-inventory-before.json"
        EVIDENCE.atomic_write_json(device_before_path, device_before)

        library_path = f"{output / 'archive' / 'runtime' / 'lib'}:{prefix / 'lib'}"
        held_descriptors = tuple(
            handle.descriptor for handle in archived.values()
        )
        preflight_command = [
            archived["loader"].path.name,
            "--inhibit-rpath",
            "",
            "--library-path",
            library_path,
            "--list",
            archived["gpu_step_db"].proc_path,
        ]
        try:
            preflight = _run_process(
                command=preflight_command,
                executable=archived["loader"].proc_path,
                pass_fds=held_descriptors,
                environment=environment,
                cwd=home,
                timeout_seconds=min(args.timeout_seconds, 60),
            )
        except ProcessExecutionError as error:
            _persist_failed_process_execution(
                output, "loader-preflight-process-error", error
            )
            raise
        preflight_error: str | None = None
        if preflight["timed_out"] or preflight.get("containment_error"):
            preflight_error = "ELF-loader preflight timed out or violated containment"
        elif preflight["returncode"] != 0 or preflight["stderr"].strip():
            preflight_error = "ELF-loader preflight failed"
        elif str(archived["libmeep"].path.parent) not in preflight["stdout"]:
            preflight_error = "ELF-loader preflight did not resolve archived libmeep"
        _persist_process_result(
            output,
            "loader-preflight",
            preflight,
            {"archived_libmeep_resolved": preflight_error is None},
            validation_error=preflight_error,
        )
        if preflight_error:
            raise EvidenceError(preflight_error)

        samples: list[dict[str, Any]] = []
        topology_plan = _physical_cpu_core_plan(max(PROFILE["mpi_ranks"]))
        controller_cpu_mask = set(
            topology_plan["controller_core"]["logical_cpus"]
        )
        worker_cpu_masks_all = [
            set(record["logical_cpus"])
            for record in topology_plan["worker_cores"]
        ]
        original_controller_affinity = set(os.sched_getaffinity(0))
        original_controller_state = _process_task_cpu_state(os.getpid())
        if original_controller_state is None:
            raise EvidenceError("controller native task state is unavailable")
        controller_original_task_affinities = original_controller_state[
            "task_affinities"
        ]
        controller_default_affinity = original_controller_affinity
        for ranks in PROFILE["mpi_ranks"]:
            worker_cpu_masks = worker_cpu_masks_all[:ranks]
            control_directory = output / f"ldos-rank-{ranks}-control"
            lane_environment = dict(environment)
            lane_environment["MEEP_GPU_LDOS_CONTROL_DIR"] = str(control_directory)
            # Set the cleanup obligation before the first per-TID mutation:
            # the pinning helper can fail after changing only a subset of TIDs.
            controller_affinity_active = True
            _set_all_process_task_affinity(os.getpid(), controller_cpu_mask)
            nvml_sampler = NvmlSampler(
                nvml_library, [row["uuid"] for row in selected]
            )
            active_nvml_sampler = nvml_sampler
            _set_all_process_task_affinity(os.getpid(), controller_cpu_mask)
            lane_monitor = LaneControlMonitor(
                directory=control_directory,
                ranks=ranks,
                controller_cpu_mask=controller_cpu_mask,
                worker_cpu_masks=worker_cpu_masks,
                expected_device_uuids=[row["uuid"] for row in selected],
                physical_topology_plan=topology_plan,
                nvml_sampler=nvml_sampler,
                nvml_library_archive=archived["nvml_library"],
            )
            active_lane_monitor = lane_monitor
            lane_environment["MEEP_GPU_LDOS_NONCE"] = lane_monitor.nonce
            command = _mpi_command(
                archived["timeout"],
                archived["mpiexec"],
                archived["loader"],
                archived["gpu_step_db"],
                ranks,
                args.timeout_seconds,
                library_path,
                worker_cpu_masks,
            )
            _persist_attempt(
                output=output,
                label=f"ldos-rank-{ranks}",
                command=command,
                environment=lane_environment,
                cwd=home,
                timeout_seconds=args.timeout_seconds + 20,
            )
            try:
                result = _run_process(
                    command=command,
                    executable=archived["timeout"].proc_path,
                    pass_fds=held_descriptors,
                    environment=lane_environment,
                    cwd=home,
                    timeout_seconds=args.timeout_seconds + 20,
                    observe_runtime=True,
                    poll_callback=lane_monitor.poll,
                )
            except ProcessExecutionError as error:
                _raise_lane_process_execution_error(
                    output=output,
                    label=f"ldos-rank-{ranks}",
                    error=error,
                    lane_monitor=lane_monitor,
                    ranks=ranks,
                    expected_device_uuids=[row["uuid"] for row in selected],
                    expected_worker_cpu_masks=worker_cpu_masks,
                )
            except BaseException:
                # The outer lifecycle guard owns abort/close so both are still
                # attempted even if one cleanup action itself fails.
                raise
            parsed_for_persistence: dict[str, Any] | None = None
            try:
                primary, parsed_for_persistence = _classify_lane_process_failure(
                    result,
                    ranks=ranks,
                    expected_device_uuids=[row["uuid"] for row in selected],
                    expected_worker_cpu_masks=worker_cpu_masks,
                    go_monotonic_ns=lane_monitor.go_monotonic_ns,
                    process_finished_monotonic_ns=result.get(
                        "process_finished_monotonic_ns"
                    ),
                )
                if primary is not None:
                    _raise_lane_result_primary(
                        primary,
                        result=result,
                        lane_monitor=lane_monitor,
                        output=output,
                        label=f"ldos-rank-{ranks}",
                        failure_evidence=parsed_for_persistence,
                    )
                parsed = parse_benchmark_output(
                    result["stdout"],
                    result["stderr"],
                    returncode=result["returncode"],
                    expected_ranks=ranks,
                    expected_device_uuids=[row["uuid"] for row in selected],
                )
                lane_monitor.drain(
                    result["root_pid"], result["process_finished_monotonic_ns"]
                )
                observed_masks = [
                    parsed["rank_accounting"][rank * PROFILE["paired_repetitions"]][
                        "cpu_affinity"
                    ]
                    for rank in range(ranks)
                ]
                if observed_masks != [sorted(mask) for mask in worker_cpu_masks]:
                    raise EvidenceError("worker-reported CPU masks differ from topology plan")
                parsed["verified_runtime_closure"] = verify_runtime_attestation(
                    parsed,
                    repo=repo,
                    receipt=validated["receipt"],
                    archived=archived,
                )
                parsed["mpi_runtime_execution"] = verify_mpi_runtime_observations(
                    result,
                    repo=repo,
                    receipt=validated["receipt"],
                    archived=archived,
                    expected_ranks=ranks,
                    expected_library_path=library_path,
                    expected_timeout_seconds=args.timeout_seconds,
                    expected_worker_cpu_masks=worker_cpu_masks,
                )
                host_runtime_control = lane_monitor.finish()
                _validate_success_host_timeline(
                    parsed,
                    host_runtime_control,
                    result["process_finished_monotonic_ns"],
                )
                host_runtime_control["raw_journal"] = _relative_file_record(
                    lane_monitor.journal.path, output
                )
                parsed["host_runtime_control"] = host_runtime_control
            except BaseException as error:
                if not getattr(error, "_lane_result_persisted", False):
                    try:
                        _persist_process_result(
                            output,
                            f"ldos-rank-{ranks}",
                            result,
                            parsed_for_persistence,
                            validation_error=str(error),
                        )
                    except BaseException as persistence_error:
                        error.add_note(
                            "process-result persistence raised "
                            f"{type(persistence_error).__name__}: {persistence_error}"
                        )
                raise
            samples.append(
                _persist_process_result(
                    output, f"ldos-rank-{ranks}", result, parsed
                )
            )
            nvml_sampler.close()
            _restore_process_task_affinities(
                os.getpid(), controller_original_task_affinities,
                controller_default_affinity,
            )
            controller_affinity_active = False
            active_nvml_sampler = None
            active_lane_monitor = None

        device_after = _query_devices(
            archived["nvidia_smi"],
            home,
            min(args.timeout_seconds, 60),
            output=output,
            label="after",
        )
        require_quiescent_devices(device_after, selected, "benchmark end")
        static_fields = (
            "index", "uuid", "name", "pci_bus_id", "compute_major",
            "compute_minor", "driver_version", "power_limit_w", "memory_total_mib",
        )
        before_static = [
            {name: item[name] for name in static_fields}
            for item in device_before["inventory"]
        ]
        after_static = [
            {name: item[name] for name in static_fields}
            for item in device_after["inventory"]
        ]
        if after_static != before_static:
            raise EvidenceError("physical GPU/driver identity changed during the benchmark")
        device_after_path = output / "device-inventory-after.json"
        EVIDENCE.atomic_write_json(device_after_path, device_after)

        mutation_events = sentinel.finish()
        sentinel_path = output / "runtime-mutation-sentinel.json"
        EVIDENCE.atomic_write_json(
            sentinel_path,
            {
                "schema_version": 1,
                "root": str(prefix),
                "watch_count": len(sentinel.watches),
                "events": mutation_events,
            },
        )
        if mutation_events:
            raise EvidenceError("receipt runtime prefix changed during the benchmark")

        end_validation = reverify_receipt_inputs(
            repo=repo,
            receipt_path=receipt_path,
            expected_receipt_id=args.expected_receipt_id,
            mpiexec_path=args.mpiexec,
            expected_source=validated["receipt"]["source_end"],
            expected_records=validated["records"],
        )
        stable_originals = {
            label: handle.verify() for label, handle in originals.items()
        }
        stable_snapshots = {
            label: handle.verify() for label, handle in archived.items()
        }
        stable_closure = [
            handle.verify() for handle in closure_handles.values()
        ]
        nvidia_smi_interpreter_record: dict[str, Any] | None = None
        nvidia_smi_archive_interpreter_record: dict[str, Any] | None = None
        nvidia_smi_interpreter_tcb: dict[str, Any] | None = None
        nvidia_smi_interpreter_elf: dict[str, Any] | None = None
        if nvidia_smi.companions:
            if len(nvidia_smi.companions) != 1 or len(
                archived["nvidia_smi"].companions
            ) != 1:
                raise EvidenceError("held nvidia-smi interpreter binding is ambiguous")
            source_interpreter = nvidia_smi.companions[0]
            archive_interpreter = archived["nvidia_smi"].companions[0]
            nvidia_smi_interpreter_record = source_interpreter.verify()
            nvidia_smi_archive_interpreter_record = archive_interpreter.verify()
            if (
                nvidia_smi_interpreter_record["sha256"]
                != nvidia_smi_archive_interpreter_record["sha256"]
                or nvidia_smi_interpreter_record["size_bytes"]
                != nvidia_smi_archive_interpreter_record["size_bytes"]
            ):
                raise EvidenceError("nvidia-smi interpreter duplicate changed")
            nvidia_smi_interpreter_tcb = _trusted_system_tool_tcb_handle(
                source_interpreter
            )
            if nvidia_smi_interpreter_tcb is None:
                raise EvidenceError("nvidia-smi interpreter left its immutable TCB")
            nvidia_smi_interpreter_elf = _elf_identity(
                source_interpreter.descriptor, source_interpreter.size_bytes
            )
        controller_identity_end = _verify_controller_identity()
        if controller_identity_end != controller_identity_start:
            raise EvidenceError("controller source identity changed during benchmark")
        revalidated_samples = [
            _revalidate_sample(output, sample) for sample in samples
        ]
        lanes = [sample["benchmark"] for sample in revalidated_samples]
        if [lane["ranks"] for lane in lanes] != list(PROFILE["mpi_ranks"]):
            raise EvidenceError("LDOS rank lane matrix is incomplete or out of order")

        claim_scope = {
            "comparison": (
                "CUDA-resident LDOS reduction/transfer versus forced legacy "
                "full-field host LDOS fallback within the same CUDA FDTD workload"
            ),
            "valid_for_ldos_reduction_transfer_speedup": True,
            "valid_for_end_to_end_fdtd_speedup": False,
            "valid_for_cpu_vs_gpu_speedup": False,
            "valid_for_general_meep_speedup": False,
            "valid_for_multi_gpu_scaling": False,
        }
        summary = {
            "schema_version": 1,
            "state": "COMPLETE",
            "run_id": run_id,
            "profile_id": PROFILE_ID,
            "profile_sha256": PROFILE_SHA256,
            "controller_source_identity": controller_identity_end,
            "claim_scope": claim_scope,
            "receipt_id": args.expected_receipt_id,
            "selected_gpu_uuids": [row["uuid"] for row in selected],
            "lanes": lanes,
            "all_gates_pass": True,
        }
        report = {
            "schema_version": 1,
            "state": "COMPLETE",
            "run_id": run_id,
            "generated_at_utc": _utc_now(),
            "profile": dict(PROFILE),
            "profile_sha256": PROFILE_SHA256,
            "controller_source_identity": controller_identity_end,
            "claim_scope": claim_scope,
            "build_receipt": {
                "path": str(receipt_path),
                "receipt_id": args.expected_receipt_id,
                "build_input_id": validated["receipt"].get("build_input_id"),
                "artifact_set_id": validated["receipt"].get("artifact_set_id"),
                "source_end": validated["receipt"]["source_end"],
                "archived_receipt": _relative_file_record(
                    output / "archive" / "input" / "build-provenance.json", output
                ),
            },
            "device_binding": {
                "selection_mechanism": (
                    "CUDA_VISIBLE_DEVICES=<uuid0,uuid1> plus per-app-context "
                    "MEEP_GPU_DEVICE=0/1"
                ),
                "selected": selected,
                "distinct_physical_uuids": len({row["uuid"] for row in selected}) == 2,
                "strict_cuda_compatibility_proof": (
                    "both explicitly mapped ranks completed the strict-CUDA focused benchmark"
                ),
                "inventory_before": _relative_file_record(device_before_path, output),
                "inventory_after": _relative_file_record(device_after_path, output),
                "trusted_nvidia_smi": stable_snapshots["nvidia_smi"],
                "trusted_nvidia_smi_interpreter": nvidia_smi_interpreter_record,
                "trusted_nvidia_smi_archive_interpreter": (
                    nvidia_smi_archive_interpreter_record
                ),
                "trusted_nvidia_smi_interpreter_tcb": nvidia_smi_interpreter_tcb,
                "trusted_nvidia_smi_interpreter_elf": nvidia_smi_interpreter_elf,
                "trusted_nvml_library": nvml_library.record(),
                "trusted_nvml_archive": stable_snapshots["nvml_library"],
                "trusted_nvml_system_tcb": nvml_system_tcb,
                "trusted_nvml_end_verification": nvml_library.verify(),
            },
            "cpu_topology_binding": topology_plan,
            "samples": revalidated_samples,
            "input_reverification": {
                "receipt_id": end_validation["receipt"]["receipt_id"],
                "source_unchanged": True,
                "artifact_records_unchanged": True,
                "stable_original_handles": stable_originals,
                "stable_snapshot_handles": stable_snapshots,
                "stable_runtime_closure_count": len(stable_closure),
                "stable_runtime_closure_sha256": PROVENANCE.canonical_sha256(
                    stable_closure
                ),
                "mutation_sentinel": _relative_file_record(sentinel_path, output),
            },
            "summary": summary,
        }
        # A COMPLETE terminal is published only after every held resource and
        # mutation observer has closed successfully. Cleanup failure therefore
        # remains a FAILED run with RUNNING still available to the publisher.
        if sentinel is not None:
            sentinel.close()
            sentinel = None
        for handle in closure_handles.values():
            handle.close()
        closure_handles.clear()
        for handle in held_inputs.values():
            handle.close()
        held_inputs.clear()
        for handle in (*originals.values(), *archived.values()):
            handle.close()
        originals.clear()
        archived.clear()
        if nvidia_smi is not None:
            nvidia_smi.close()
            nvidia_smi = None
        if nvml_library is not None:
            nvml_library.close()
            nvml_library = None
        _publish_success(output=output, run_id=run_id, summary=summary, report=report)
        return 0
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_failures: list[str] = []

        def cleanup(label: str, action: Callable[[], Any]) -> None:
            try:
                action()
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    f"{label}: {type(cleanup_error).__name__}: {cleanup_error}"
                )

        if active_lane_monitor is not None:
            cleanup(
                "lane telemetry abort",
                lambda: active_lane_monitor.abort(
                    "controller unwound before lane commit"
                ),
            )
        if active_nvml_sampler is not None:
            cleanup("NVML sampler close", active_nvml_sampler.close)
        if controller_affinity_active:
            cleanup(
                "controller affinity restore",
                lambda: _restore_process_task_affinities(
                    os.getpid(), controller_original_task_affinities,
                    controller_default_affinity,
                ),
            )
        if sentinel is not None:
            cleanup("mutation sentinel close", sentinel.close)
        for label, handle in closure_handles.items():
            cleanup(f"runtime closure {label} close", handle.close)
        for label, handle in held_inputs.items():
            cleanup(f"held input {label} close", handle.close)
        if nvidia_smi is not None:
            cleanup("nvidia-smi close", nvidia_smi.close)
        if nvml_library is not None:
            cleanup("NVML library close", nvml_library.close)
        for label, handle in originals.items():
            cleanup(f"original {label} close", handle.close)
        for label, handle in archived.items():
            cleanup(f"archived {label} close", handle.close)
        _FD_ALLOCATION_SHIM = None
        if cleanup_failures:
            message = "; ".join(cleanup_failures)
            if primary_error is not None:
                primary_error.add_note(message)
            else:
                raise EvidenceError(message)


def main(argv: list[str] | None = None) -> int:
    invocation = list(sys.argv[1:] if argv is None else argv)
    _controller_bootstrap(invocation)
    _initialize_verified_helpers()
    args = parse_args(invocation)
    run_id = f"ldos-{uuid.uuid4().hex}"
    output = args.output.resolve(strict=False)
    prepared = False
    try:
        repo, output = _validate_invocation(args)
        _claim_output(output, run_id)
        prepared = True
        return execute(args, run_id, repo=repo, output=output)
    except BaseException as error:
        if prepared:
            try:
                _publish_failure(output, run_id, error)
            except BaseException as publication_error:
                print(
                    f"fatal: benchmark failed ({error}); failure publication also failed: "
                    f"{publication_error}",
                    file=sys.stderr,
                )
                return 2
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
