#!/usr/bin/env python3
"""Run and strictly gate the gpmeep fixed-work multi-GPU benchmark.

The C++ worker emits one versioned JSON record.  This controller deliberately
does not attempt to recover partial CSV rows: malformed, duplicate, non-finite,
zero-signal, or workload-mismatched evidence is rejected before it reaches the
strong-scaling summary.
"""

from __future__ import annotations

import ctypes
import csv
import errno
import io
import json
import math
import os
import pathlib
import platform
import re
import select
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

from gpmeep_provenance import (
    ProvenanceError,
    file_record,
    sha256_file,
    verify_build_receipt,
)
from gpmeep_benchmark_evidence import atomic_write_json, atomic_write_text


RECORD_PREFIX = "gpmeep-multi-benchmark-v4:"
DEVICE_RECORD_PREFIX = "gpmeep-multi-device-v1:"
INITIAL_CONDITION_PREFIX = "gpmeep-initial-condition-v1:"
SOURCE_PROFILE_PREFIX = "gpmeep-source-profile-v1:"
INITIALIZATION_TIMING_PREFIX = "gpmeep-initialize-field-timing-v1:"
SCHEMA_VERSION = 4
WORKER_STDOUT_GPMEEP_PREFIXES = (
    RECORD_PREFIX,
    DEVICE_RECORD_PREFIX,
    "gpmeep-multi-process-v1:",
    SOURCE_PROFILE_PREFIX,
    INITIAL_CONDITION_PREFIX,
    INITIALIZATION_TIMING_PREFIX,
    "gpmeep-boundary-phase-graph-v1:",
    "gpmeep-eager-mpi-v1:",
    "gpmeep-receive-pingpong-v1:",
    "gpmeep-boundary-eh-overlap-v1:",
    "gpmeep-halo-curl-overlap-v1:",
    "gpmeep-tile-coalescing-v1:",
    "gpmeep-phase-batch-policy-v1:",
    "gpmeep-curl-phase-replay-v1:",
    "gpmeep-boundary-descriptor-replay-v1:",
    "gpmeep-dft-phase-sharing-v1:",
    "gpmeep-dft-multi-monitor-batch-v1:",
    "gpmeep-dft-warmup-plan-v1:",
)
PROBE_COUNT = 18
REQUIRED_CUDA_PHASES = ("curl", "update_eh", "source", "boundary", "dft")
NOT_APPLICABLE_PHASES = ("polarization",)
ALL_PHASES = REQUIRED_CUDA_PHASES + NOT_APPLICABLE_PHASES
PHASE_EXACT_KEYS = {"cpu_calls", "cuda_calls", "expectation"}
RELEASE_PROFILE = {
    "quick": False,
    "pixels": 192,
    "warmup_steps": 12,
    "measured_steps": 80,
    "minimum_repeats": 3,
    "timing_statistic": "median",
    "transport": "pinned",
    "minimum_parallel_efficiency": 0.55,
    "maximum_observable_relative_error": 0.002,
    "loop_tile_base_db": 128,
    "bfast": False,
    "source_enabled": True,
    "source_profile": "single-ez-v1",
    "overlap_material": False,
    "completion_policy": "waitsome",
    "initial_condition": "trigonometric-v1",
}
FIXED_WORKER_PROFILE = {
    key: RELEASE_PROFILE[key]
    for key in (
        "loop_tile_base_db",
        "bfast",
        "source_enabled",
        "overlap_material",
        "completion_policy",
        "initial_condition",
    )
}
TERMINATION_GRACE_SECONDS = 5.0
PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37
SYS_PIDFD_SEND_SIGNAL = 424
SYS_PIDFD_OPEN = 434
RUN_LOGGED_LOCK = threading.Lock()
CANONICAL_MPI_ENVIRONMENT_KEYS = frozenset(
    {
        "OMPI_MCA_mca_base_param_files",
        "OMPI_MCA_mca_base_component_path",
        "PMIX_MCA_mca_base_param_files",
        "PMIX_MCA_mca_base_component_path",
        "PRTE_MCA_mca_base_param_files",
    }
)
EXACT_KEYS = {
    "schema_version",
    "mpi_ranks",
    "pixels",
    "cells",
    "warmup_steps",
    "steps",
    "loop_tile_base_db",
    "bfast",
    "source_enabled",
    "overlap_material",
    "seconds",
    "mcells_per_second",
    "requested_transport",
    "selected_transport",
    "completion_policy",
    "initial_condition",
    "cut_probe_values",
    "cut_probe_l2",
    "ez_l2",
    "ez_weighted_checksum",
    "energy",
    "dft_norm",
    "remote_cut_scalars",
    "remote_cut_l2",
    "cpu_calls",
    "cuda_calls",
    "phase_calls",
    "h2d_bytes",
    "d2h_bytes",
    "mpi_messages",
    "mpi_scalars",
    "cuda_aware_bytes",
    "pinned_bytes",
}
INTEGER_KEYS = {
    "schema_version",
    "mpi_ranks",
    "pixels",
    "cells",
    "warmup_steps",
    "steps",
    "loop_tile_base_db",
    "remote_cut_scalars",
    "cpu_calls",
    "cuda_calls",
    "h2d_bytes",
    "d2h_bytes",
    "mpi_messages",
    "mpi_scalars",
    "cuda_aware_bytes",
    "pinned_bytes",
}
POSITIVE_FLOAT_KEYS = {
    "seconds",
    "mcells_per_second",
    "cut_probe_l2",
    "ez_l2",
    "ez_weighted_checksum",
    "energy",
    "dft_norm",
}


class EvidenceError(RuntimeError):
    """The worker output cannot serve as benchmark evidence."""


def verify_file_record(
    record: dict[str, Any], repo: pathlib.Path, label: str
) -> pathlib.Path:
    """Re-hash a recorded artifact immediately before authority is published."""

    if type(record) is not dict or set(record) != {"path", "size_bytes", "sha256"}:
        raise EvidenceError(f"{label} has an invalid file-record schema")
    if type(record["path"]) is not str or not record["path"]:
        raise EvidenceError(f"{label} has no path")
    path = pathlib.Path(record["path"])
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        raise EvidenceError(f"{label} is absent: {path}")
    before = path.stat()
    if type(record["size_bytes"]) is not int or record["size_bytes"] != before.st_size:
        raise EvidenceError(f"{label} size differs from its recorded value")
    if type(record["sha256"]) is not str or not re.fullmatch(
        r"[0-9a-f]{64}", record["sha256"]
    ):
        raise EvidenceError(f"{label} has an invalid SHA-256")
    actual_sha256 = sha256_file(path)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise EvidenceError(f"{label} changed while it was being verified")
    if actual_sha256 != record["sha256"]:
        raise EvidenceError(f"{label} SHA-256 differs from its recorded value")
    return path.resolve()


def _reject_json_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def normalize_transport(value: str) -> str:
    if value in ("", "auto"):
        return "auto"
    if value in ("pinned", "host"):
        return "pinned"
    if value in ("cuda-aware", "device"):
        return "cuda-aware"
    raise EvidenceError(
        f"invalid MPI transport {value!r}; expected auto, pinned, or cuda-aware"
    )


def _require_int(record: dict[str, Any], key: str, *, positive: bool = False) -> int:
    value = record[key]
    if type(value) is not int:
        raise EvidenceError(f"{key} must be an integer, got {value!r}")
    if positive and value <= 0:
        raise EvidenceError(f"{key} must be positive, got {value}")
    if not positive and value < 0:
        raise EvidenceError(f"{key} must be nonnegative, got {value}")
    return value


def _require_finite_number(
    record: dict[str, Any], key: str, *, positive: bool = False
) -> float:
    value = record[key]
    if type(value) not in (int, float):
        raise EvidenceError(f"{key} must be numeric, got {value!r}")
    promoted = float(value)
    if not math.isfinite(promoted):
        raise EvidenceError(f"{key} must be finite, got {value!r}")
    if positive and promoted <= 0:
        raise EvidenceError(f"{key} must be positive, got {value!r}")
    return promoted


def extract_record(
    stdout: str,
    *,
    expected_ranks: int | None = None,
    expected_pixels: int | None = None,
    expected_warmup_steps: int | None = None,
    expected_steps: int | None = None,
    expected_transport: str | None = None,
    expected_worker_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract and validate exactly one worker record from stdout."""

    payloads = [
        line[len(RECORD_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(RECORD_PREFIX)
    ]
    if len(payloads) != 1:
        raise EvidenceError(
            f"expected exactly one {RECORD_PREFIX!r} record, found {len(payloads)}"
        )
    try:
        value = json.loads(
            payloads[0],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except EvidenceError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise EvidenceError(f"malformed benchmark JSON: {error}") from error
    if type(value) is not dict:
        raise EvidenceError("benchmark record must be a JSON object")
    record: dict[str, Any] = value
    if set(record) != EXACT_KEYS:
        missing = sorted(EXACT_KEYS - set(record))
        extra = sorted(set(record) - EXACT_KEYS)
        raise EvidenceError(
            f"benchmark record key mismatch: missing={missing}, extra={extra}"
        )

    for key in INTEGER_KEYS:
        _require_int(
            record,
            key,
            positive=key
            in {"mpi_ranks", "pixels", "cells", "warmup_steps", "steps"},
        )
    for key in POSITIVE_FLOAT_KEYS:
        _require_finite_number(record, key, positive=True)
    _require_finite_number(record, "remote_cut_l2")

    if record["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError(
            f"schema_version must be {SCHEMA_VERSION}, got {record['schema_version']!r}"
        )
    if record["cells"] != record["pixels"] ** 3:
        raise EvidenceError("cells must equal pixels cubed")
    if expected_ranks is not None and record["mpi_ranks"] != expected_ranks:
        raise EvidenceError(
            f"worker reported {record['mpi_ranks']} ranks, expected {expected_ranks}"
        )
    expected_work = {
        "pixels": expected_pixels,
        "warmup_steps": expected_warmup_steps,
        "steps": expected_steps,
    }
    for key, expected in expected_work.items():
        if expected is not None and record[key] != expected:
            raise EvidenceError(
                f"worker reported {key}={record[key]}, expected {expected}"
            )

    requested = record["requested_transport"]
    selected = record["selected_transport"]
    if type(requested) is not str or requested not in {
        "auto",
        "pinned",
        "cuda-aware",
    }:
        raise EvidenceError(f"invalid requested_transport {requested!r}")
    if type(selected) is not str or selected not in {
        "none",
        "pinned",
        "cuda-aware",
    }:
        raise EvidenceError(f"invalid selected_transport {selected!r}")
    if expected_transport is not None:
        normalized_expected = normalize_transport(expected_transport)
        if requested != normalized_expected:
            raise EvidenceError(
                f"worker reported requested transport {requested!r}, "
                f"expected {normalized_expected!r}"
            )
    worker_profile = (
        FIXED_WORKER_PROFILE
        if expected_worker_profile is None
        else expected_worker_profile
    )
    if type(worker_profile) is not dict or set(worker_profile) != set(
        FIXED_WORKER_PROFILE
    ):
        raise EvidenceError(
            "expected worker profile must define exactly the fixed worker keys"
        )
    for key, expected in worker_profile.items():
        if type(expected) not in (bool, int, str):
            raise EvidenceError(
                f"expected worker profile {key!r} has an invalid value type"
            )
        actual = record[key]
        if type(expected) is bool:
            if type(actual) is not bool:
                raise EvidenceError(f"{key} must be a JSON boolean")
        elif type(expected) is int:
            if type(actual) is not int:
                raise EvidenceError(f"{key} must be an integer")
        elif type(actual) is not str:
            raise EvidenceError(f"{key} must be a string")
        if actual != expected:
            raise EvidenceError(
                f"worker profile requires {key}={expected!r}, got {actual!r}"
            )

    probes = record["cut_probe_values"]
    if type(probes) is not list or len(probes) != PROBE_COUNT:
        raise EvidenceError(
            f"cut_probe_values must contain exactly {PROBE_COUNT} complex values"
        )
    computed_probe_squared = 0.0
    for index, pair in enumerate(probes):
        if type(pair) is not list or len(pair) != 2:
            raise EvidenceError(f"cut probe {index} must be [real, imag]")
        for part in pair:
            if type(part) not in (int, float) or not math.isfinite(float(part)):
                raise EvidenceError(
                    f"cut probe {index} contains a non-finite/non-numeric value"
                )
        computed_probe_squared += float(pair[0]) ** 2 + float(pair[1]) ** 2
    computed_probe_l2 = math.sqrt(computed_probe_squared)
    if computed_probe_l2 <= 0:
        raise EvidenceError("cut probe vector is identically zero")
    if not math.isclose(
        float(record["cut_probe_l2"]),
        computed_probe_l2,
        rel_tol=2e-12,
        abs_tol=1e-14,
    ):
        raise EvidenceError("cut_probe_l2 does not match cut_probe_values")

    expected_rate = (
        record["cells"] * record["steps"] / float(record["seconds"]) / 1e6
    )
    if not math.isclose(
        float(record["mcells_per_second"]),
        expected_rate,
        rel_tol=2e-12,
        abs_tol=1e-12,
    ):
        raise EvidenceError("mcells_per_second is inconsistent with fixed work")

    phase_calls = record["phase_calls"]
    if type(phase_calls) is not dict or set(phase_calls) != set(ALL_PHASES):
        actual = set(phase_calls) if type(phase_calls) is dict else set()
        raise EvidenceError(
            "phase_calls key mismatch: "
            f"missing={sorted(set(ALL_PHASES) - actual)}, "
            f"extra={sorted(actual - set(ALL_PHASES))}"
        )
    phase_cpu_calls = 0
    phase_cuda_calls = 0
    for phase in ALL_PHASES:
        item = phase_calls[phase]
        if type(item) is not dict or set(item) != PHASE_EXACT_KEYS:
            raise EvidenceError(f"phase_calls[{phase!r}] has an invalid schema")
        cpu = _require_int(item, "cpu_calls")
        cuda = _require_int(item, "cuda_calls")
        expectation = item["expectation"]
        expected = (
            "cuda_required"
            if phase in REQUIRED_CUDA_PHASES
            else "not_applicable"
        )
        if expectation != expected:
            raise EvidenceError(
                f"phase {phase!r} expectation must be {expected!r}"
            )
        if phase in REQUIRED_CUDA_PHASES and (cpu != 0 or cuda <= 0):
            raise EvidenceError(
                f"required phase {phase!r} violated exclusive CUDA dispatch"
            )
        if phase in NOT_APPLICABLE_PHASES and (cpu != 0 or cuda != 0):
            raise EvidenceError(
                f"not-applicable phase {phase!r} recorded dispatch activity"
            )
        phase_cpu_calls += cpu
        phase_cuda_calls += cuda
    if record["cpu_calls"] != phase_cpu_calls:
        raise EvidenceError("cpu_calls does not equal the phase_calls aggregate")
    if record["cuda_calls"] != phase_cuda_calls:
        raise EvidenceError("cuda_calls does not equal the phase_calls aggregate")
    if record["cpu_calls"] != 0 or record["cuda_calls"] <= 0:
        raise EvidenceError("worker violated the exclusive CUDA dispatch contract")

    ranks = record["mpi_ranks"]
    if ranks == 1:
        zero_keys = (
            "mpi_messages",
            "mpi_scalars",
            "cuda_aware_bytes",
            "pinned_bytes",
            "remote_cut_scalars",
        )
        if any(record[key] != 0 for key in zero_keys) or float(
            record["remote_cut_l2"]
        ) != 0.0:
            raise EvidenceError("one-rank record contains remote MPI/cut activity")
        if selected != "none":
            raise EvidenceError("one-rank record must select transport 'none'")
    else:
        if record["mpi_messages"] <= 0 or record["mpi_scalars"] <= 0:
            raise EvidenceError("multi-rank record exchanged no MPI boundary data")
        if record["remote_cut_scalars"] <= 0 or float(
            record["remote_cut_l2"]
        ) <= 0:
            raise EvidenceError("multi-rank record has vacuous remote-cut evidence")
        used_cuda_aware = record["cuda_aware_bytes"] > 0
        used_pinned = record["pinned_bytes"] > 0
        if used_cuda_aware == used_pinned:
            raise EvidenceError(
                "multi-rank record must use exactly one MPI transport"
            )
        inferred = "cuda-aware" if used_cuda_aware else "pinned"
        if selected != inferred:
            raise EvidenceError(
                "selected_transport disagrees with the transport byte counters"
            )
        if requested != "auto" and selected != requested:
            raise EvidenceError("forced and selected MPI transports differ")
        expected_transport_bytes = record["mpi_scalars"] * 4
        actual_transport_bytes = (
            record["cuda_aware_bytes"] + record["pinned_bytes"]
        )
        if actual_transport_bytes != expected_transport_bytes:
            raise EvidenceError(
                "MPI transport bytes must equal mpi_scalars times FP32 size"
            )
        if selected == "pinned" and (
            record["h2d_bytes"] <= 0 or record["d2h_bytes"] <= 0
        ):
            raise EvidenceError(
                "pinned MPI transport requires nonzero H2D and D2H activity"
            )
    return record


def reject_unknown_worker_markers(stdout: str, stderr: str) -> None:
    for line in stdout.replace("\r", "\n").splitlines():
        position = line.find("gpmeep-")
        if position >= 0 and not any(
            line.startswith(prefix, position)
            for prefix in WORKER_STDOUT_GPMEEP_PREFIXES
        ):
            raise EvidenceError(
                "worker stdout contains an unknown or mixed-version gpmeep marker"
            )
    if any("gpmeep-" in line for line in stderr.replace("\r", "\n").splitlines()):
        raise EvidenceError("worker stderr contains a reserved gpmeep marker")


def _extract_exact_marker(
    stdout: str, prefix: str, label: str, exact_keys: set[str]
) -> dict[str, Any]:
    payloads = [
        line[len(prefix) :]
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]
    if len(payloads) != 1:
        raise EvidenceError(
            f"expected exactly one {label} marker, found {len(payloads)}"
        )
    try:
        value = json.loads(
            payloads[0],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except EvidenceError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise EvidenceError(f"malformed {label} marker: {error}") from error
    if type(value) is not dict or set(value) != exact_keys:
        raise EvidenceError(f"{label} marker has an invalid exact schema")
    return value


def extract_initialization_records(
    stdout: str,
    *,
    expected_profile: str,
    expected_applications_per_rank: int = 1,
) -> dict[str, Any]:
    if (
        type(expected_applications_per_rank) is not int
        or expected_applications_per_rank <= 0
    ):
        raise EvidenceError(
            "expected initialization application count must be a positive integer"
        )
    profile_record = _extract_exact_marker(
        stdout, INITIAL_CONDITION_PREFIX, "initial-condition", {"profile"}
    )
    profile = profile_record["profile"]
    if type(profile) is not str or profile != expected_profile:
        raise EvidenceError(
            "initial-condition marker differs from the expected worker profile"
        )
    timing = _extract_exact_marker(
        stdout,
        INITIALIZATION_TIMING_PREFIX,
        "initialize-field timing",
        {"applications_per_rank", "max_seconds"},
    )
    applications = timing["applications_per_rank"]
    seconds = timing["max_seconds"]
    if (
        type(applications) is not int
        or applications != expected_applications_per_rank
    ):
        raise EvidenceError(
            "initialize-field timing has an invalid application count"
        )
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(float(seconds))
        or seconds <= 0
    ):
        raise EvidenceError(
            "initialize-field timing max_seconds must be finite and positive"
        )
    return {
        "initial_condition": profile_record,
        "initialize_field_timing": {
            "applications_per_rank": applications,
            "max_seconds": float(seconds),
            "timing_domain": "untimed setup; excluded from FDTD seconds",
        },
    }


def extract_source_profile_record(
    stdout: str, *, expected_profile: str
) -> dict[str, str]:
    record = _extract_exact_marker(
        stdout, SOURCE_PROFILE_PREFIX, "source-profile", {"profile"}
    )
    profile = record["profile"]
    if type(expected_profile) is not str or expected_profile not in {
        "single-ez-v1",
        "dual-electric-v1",
    }:
        raise EvidenceError("expected source profile is invalid")
    if type(profile) is not str or profile != expected_profile:
        raise EvidenceError(
            "source-profile marker differs from the expected worker profile"
        )
    return record


def extract_device_records(stdout: str, expected_ranks: int) -> list[dict[str, Any]]:
    """Require one physical-device identity record from every MPI rank."""

    payloads = [
        line[len(DEVICE_RECORD_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(DEVICE_RECORD_PREFIX)
    ]
    if len(payloads) != expected_ranks:
        raise EvidenceError(
            f"expected {expected_ranks} device identity records, found {len(payloads)}"
        )
    required_keys = {
        "rank",
        "ordinal",
        "uuid",
        "name",
        "compute_major",
        "compute_minor",
        "compatible",
    }
    records: list[dict[str, Any]] = []
    for payload in payloads:
        try:
            value = json.loads(
                payload,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except EvidenceError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise EvidenceError(f"malformed device identity JSON: {error}") from error
        if type(value) is not dict or set(value) != required_keys:
            raise EvidenceError("device identity record has an invalid schema")
        for key in ("rank", "ordinal", "compute_major", "compute_minor"):
            if type(value[key]) is not int or value[key] < 0:
                raise EvidenceError(f"device identity {key} must be nonnegative")
        if value["compute_major"] <= 0:
            raise EvidenceError("device identity compute capability is invalid")
        if value["compatible"] is not True:
            raise EvidenceError("MPI rank selected an incompatible CUDA device")
        if type(value["uuid"]) is not str or not re.fullmatch(
            r"[0-9a-f]{32}", value["uuid"]
        ):
            raise EvidenceError("device identity uuid must be 32 lowercase hex digits")
        if type(value["name"]) is not str or not value["name"].strip():
            raise EvidenceError("device identity name is absent")
        records.append(value)
    records.sort(key=lambda item: item["rank"])
    if [item["rank"] for item in records] != list(range(expected_ranks)):
        raise EvidenceError("device identity ranks are missing or duplicated")
    identifiers = [item["uuid"] for item in records]
    if len(set(identifiers)) != expected_ranks:
        raise EvidenceError(
            "MPI benchmark ranks did not report distinct physical GPU UUIDs"
        )
    return records


def parse_rank_list(text: str) -> list[int]:
    pieces = [item.strip() for item in text.split(",")]
    if not pieces or any(not re.fullmatch(r"[1-9][0-9]*", item) for item in pieces):
        raise EvidenceError(
            "MEEP_GPU_MULTI_RANKS must be a comma-separated list of positive integers"
        )
    ranks = [int(item) for item in pieces]
    if len(set(ranks)) != len(ranks):
        raise EvidenceError("MEEP_GPU_MULTI_RANKS contains duplicates")
    if 1 not in ranks:
        raise EvidenceError("MEEP_GPU_MULTI_RANKS must include a one-GPU baseline")
    return ranks


def parse_probability(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise EvidenceError(f"{name} must be numeric") from error
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
        raise EvidenceError(f"{name} must be finite and between 0 and 1")
    return parsed


def parse_positive_int(name: str, value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise EvidenceError(f"{name} must be a positive integer")
    return int(value)


def parse_boolean_flag(name: str, value: str) -> bool:
    if value not in {"0", "1"}:
        raise EvidenceError(f"{name} must be 0 or 1")
    return value == "1"


def normalized_error(candidate: float, reference: float, floor: float = 1e-8) -> float:
    return abs(candidate - reference) / max(
        abs(candidate), abs(reference), floor
    )


def compare_record(
    record: dict[str, Any], reference: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    probe_errors = []
    for candidate_pair, reference_pair in zip(
        record["cut_probe_values"], reference["cut_probe_values"]
    ):
        delta = math.hypot(
            float(candidate_pair[0]) - float(reference_pair[0]),
            float(candidate_pair[1]) - float(reference_pair[1]),
        )
        scale = max(
            math.hypot(float(candidate_pair[0]), float(candidate_pair[1])),
            math.hypot(float(reference_pair[0]), float(reference_pair[1])),
            1e-8,
        )
        probe_errors.append(delta / scale)
    errors = {
        "cut_probe_max_relative_error": max(probe_errors, default=0.0),
        "cut_probe_l2_relative_error": normalized_error(
            float(record["cut_probe_l2"]), float(reference["cut_probe_l2"])
        ),
        "ez_l2_relative_error": normalized_error(
            float(record["ez_l2"]), float(reference["ez_l2"])
        ),
        "ez_weighted_checksum_relative_error": normalized_error(
            float(record["ez_weighted_checksum"]),
            float(reference["ez_weighted_checksum"]),
        ),
        "energy_relative_error": normalized_error(
            float(record["energy"]), float(reference["energy"])
        ),
        "dft_relative_error": normalized_error(
            float(record["dft_norm"]), float(reference["dft_norm"])
        ),
    }
    errors["maximum_observable_relative_error"] = max(errors.values())
    errors["observable_gate"] = (
        errors["maximum_observable_relative_error"] <= tolerance
    )
    return errors


def validate_sample_matrix(
    samples: list[dict[str, Any]],
    executed_ranks: list[int],
    repeat_count: int,
) -> None:
    """Require a complete repeat matrix and stable physical-GPU mappings."""

    expected_repeats = list(range(1, repeat_count + 1))
    mappings: dict[int, tuple[str, ...]] = {}
    for ranks in executed_ranks:
        rank_samples = [sample for sample in samples if sample["mpi_ranks"] == ranks]
        actual_repeats = sorted(sample["repeat_index"] for sample in rank_samples)
        if actual_repeats != expected_repeats:
            raise EvidenceError(
                f"rank count {ranks} has incomplete/duplicate repeats: "
                f"expected {expected_repeats}, got {actual_repeats}"
            )
        for sample in rank_samples:
            if sample["record"]["mpi_ranks"] != ranks:
                raise EvidenceError("sample envelope and worker mpi_ranks differ")
            uuids = tuple(device["uuid"] for device in sample["devices"])
            if ranks not in mappings:
                mappings[ranks] = uuids
            elif mappings[ranks] != uuids:
                raise EvidenceError(
                    f"physical GPU mapping changed across {ranks}-rank repeats"
                )

    if 1 not in mappings or len(mappings[1]) != 1:
        raise EvidenceError("sample matrix has no unique one-GPU device baseline")
    largest_rank_count = max(mappings)
    canonical_mapping = mappings[largest_rank_count]
    for ranks, mapping in mappings.items():
        if mapping != canonical_mapping[:ranks]:
            raise EvidenceError(
                f"physical GPU mapping for {ranks} ranks is not the canonical "
                f"prefix of the {largest_rank_count}-rank mapping"
            )


def summarize_samples(
    samples: list[dict[str, Any]],
    executed_ranks: list[int],
    repeat_count: int,
    observable_tolerance: float,
    minimum_efficiency: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Compare every repeat and summarize timing with per-rank medians."""

    validate_sample_matrix(samples, executed_ranks, repeat_count)
    reference_samples = [sample for sample in samples if sample["mpi_ranks"] == 1]
    reference_samples.sort(key=lambda sample: sample["repeat_index"])
    reference = reference_samples[0]["record"]
    baseline_seconds = statistics.median(
        float(sample["record"]["seconds"]) for sample in reference_samples
    )
    error_keys = (
        "cut_probe_max_relative_error",
        "cut_probe_l2_relative_error",
        "ez_l2_relative_error",
        "ez_weighted_checksum_relative_error",
        "energy_relative_error",
        "dft_relative_error",
        "maximum_observable_relative_error",
    )
    comparisons: list[dict[str, Any]] = []
    overall_gate = True
    for ranks in executed_ranks:
        rank_samples = [sample for sample in samples if sample["mpi_ranks"] == ranks]
        rank_samples.sort(key=lambda sample: sample["repeat_index"])
        sample_comparisons: list[dict[str, Any]] = []
        for sample in rank_samples:
            comparison = compare_record(
                sample["record"], reference, observable_tolerance
            )
            sample["comparison_vs_reference"] = comparison
            sample_comparisons.append(comparison)
        seconds_samples = [
            float(sample["record"]["seconds"]) for sample in rank_samples
        ]
        seconds = statistics.median(seconds_samples)
        speedup = baseline_seconds / seconds
        efficiency = speedup / ranks
        all_observables_pass = all(
            comparison["observable_gate"] for comparison in sample_comparisons
        )
        performance_gate = ranks == 1 or efficiency >= minimum_efficiency
        gate = bool(all_observables_pass and performance_gate)
        summary = {
            "mpi_ranks": ranks,
            "repeat_count": repeat_count,
            "seconds_samples": seconds_samples,
            "seconds": seconds,
            "seconds_minimum": min(seconds_samples),
            "seconds_maximum": max(seconds_samples),
            "seconds_statistic": "median",
            "speedup_vs_1gpu": speedup,
            "parallel_efficiency": efficiency,
            **{
                key: max(comparison[key] for comparison in sample_comparisons)
                for key in error_keys
            },
            "all_sample_observable_gate": all_observables_pass,
            "performance_gate": performance_gate,
            "gate": "PASS" if gate else "FAIL",
        }
        comparisons.append(summary)
        overall_gate = overall_gate and gate
    return comparisons, overall_gate


def qualification_gates(
    comparisons_pass: bool,
    requested_ranks: list[int],
    executed_ranks: list[int],
    skipped_ranks: list[dict[str, Any]],
    require_all: bool,
    *,
    workload: dict[str, Any] | None = None,
    repeat_count: int = 1,
    timing_statistic: str = "single",
    transport: str = "auto",
    minimum_efficiency: float | None = None,
    observable_tolerance: float | None = None,
) -> dict[str, Any]:
    """Separate executed-scope development evidence from release qualification."""

    development_pass = comparisons_pass and bool(executed_ranks)
    release_rank_set = [1, 2, 4]
    release_ranks_complete = (
        requested_ranks == release_rank_set
        and executed_ranks == release_rank_set
        and not skipped_ranks
    )
    observed_profile = {
        "quick": workload.get("quick") if workload is not None else None,
        "pixels": workload.get("pixels") if workload is not None else None,
        "warmup_steps": (
            workload.get("warmup_steps") if workload is not None else None
        ),
        "measured_steps": (
            workload.get("measured_steps") if workload is not None else None
        ),
        "repeat_count": repeat_count,
        "timing_statistic": timing_statistic,
        "transport": transport,
        "minimum_parallel_efficiency": minimum_efficiency,
        "maximum_observable_relative_error": observable_tolerance,
        "source_profile": (
            workload.get("source_profile") if workload is not None else None
        ),
        **{
            key: workload.get(key) if workload is not None else None
            for key in FIXED_WORKER_PROFILE
        },
    }
    profile_failures: list[str] = []
    if workload is None:
        profile_failures.append("workload profile is absent")
    else:
        for key in (
            "quick", "pixels", "warmup_steps", "measured_steps",
            "source_profile",
            *FIXED_WORKER_PROFILE,
        ):
            observed = workload.get(key)
            required = RELEASE_PROFILE[key]
            if type(observed) is not type(required) or observed != required:
                profile_failures.append(
                    f"{key} must be {required!r} with type "
                    f"{type(required).__name__}, got {observed!r}"
                )
    if repeat_count < RELEASE_PROFILE["minimum_repeats"]:
        profile_failures.append(
            "repeat_count must be at least "
            f"{RELEASE_PROFILE['minimum_repeats']}, got {repeat_count}"
        )
    if timing_statistic != RELEASE_PROFILE["timing_statistic"]:
        profile_failures.append(
            "timing_statistic must be "
            f"{RELEASE_PROFILE['timing_statistic']!r}, got {timing_statistic!r}"
        )
    if transport != RELEASE_PROFILE["transport"]:
        profile_failures.append(
            f"transport must be {RELEASE_PROFILE['transport']!r}, got {transport!r}"
        )
    if (
        minimum_efficiency is None
        or minimum_efficiency < RELEASE_PROFILE["minimum_parallel_efficiency"]
    ):
        profile_failures.append(
            "minimum_parallel_efficiency must be at least "
            f"{RELEASE_PROFILE['minimum_parallel_efficiency']!r}, "
            f"got {minimum_efficiency!r}"
        )
    if (
        observable_tolerance is None
        or observable_tolerance
        > RELEASE_PROFILE["maximum_observable_relative_error"]
    ):
        profile_failures.append(
            "maximum_observable_relative_error must be at most "
            f"{RELEASE_PROFILE['maximum_observable_relative_error']!r}, "
            f"got {observable_tolerance!r}"
        )
    profile_pass = not profile_failures
    release_pass = (
        development_pass
        and require_all
        and release_ranks_complete
        and profile_pass
    )
    return {
        "development": {
            "pass": development_pass,
            "scope": "executed_requested_ranks",
            "executed_ranks": list(executed_ranks),
        },
        "release": {
            "pass": release_pass,
            "scope": "required_physical_1_2_4_gpu_scaling",
            "required_ranks": release_rank_set,
            "all_required_ranks_executed": release_ranks_complete,
            "require_all": require_all,
            "profile_pass": profile_pass,
            "required_profile": dict(RELEASE_PROFILE),
            "observed_profile": observed_profile,
            "profile_failures": profile_failures,
        },
        "qualification": (
            "release" if release_pass else "development-only"
            if development_pass
            else "failed"
        ),
    }


def publish_complete(
    repo: pathlib.Path,
    report_path: pathlib.Path,
    complete_path: pathlib.Path,
    report: dict[str, Any],
    *,
    controller_pass: bool,
    release_pass: bool,
) -> dict[str, Any]:
    """Atomically publish the report followed by its hash-bound authority marker."""

    atomic_write_json(report_path, report)
    marker = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": report["run_id"],
        "report": file_record(report_path, repo),
        "controller_pass": controller_pass,
        "release_pass": release_pass,
    }
    atomic_write_json(complete_path, marker)
    return marker


def _clean_run_prefix(
    mamba: pathlib.Path,
    mamba_root: pathlib.Path,
    prefix: pathlib.Path,
    environment: dict[str, str],
) -> list[str]:
    command = [str(mamba), "--no-rc", "run", "--clean-env"]
    for key in sorted(environment):
        command.extend(("--env", f"{key}={environment[key]}"))
    command.extend(
        ("--root-prefix", str(mamba_root), "--prefix", str(prefix))
    )
    return command


def _timed_mpi_command(
    prefix: list[str], timeout_seconds: int, ranks: int, executable: pathlib.Path
) -> list[str]:
    return [
        *prefix,
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        f"{timeout_seconds}s",
        "mpirun",
        "--bind-to",
        "none",
        "-np",
        str(ranks),
        str(executable),
    ]


def _write_run_log(
    path: pathlib.Path,
    command: list[str],
    returncode: int | str,
    elapsed: float,
    stdout: str,
    stderr: str,
) -> None:
    atomic_write_text(
        path,
        "command: "
        + shlex.join(command)
        + f"\nreturncode: {returncode}\nelapsed_seconds: {elapsed:.9f}\n"
        + "===== stdout =====\n"
        + stdout
        + ("\n" if stdout and not stdout.endswith("\n") else "")
        + "===== stderr =====\n"
        + stderr
        + ("\n" if stderr and not stderr.endswith("\n") else ""),
    )


def _child_subreaper_state() -> bool:
    if sys.platform != "linux":
        raise EvidenceError(
            "multi-GPU process containment requires Linux child-subreaper support"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    state = ctypes.c_int()
    result = libc.prctl(
        PR_GET_CHILD_SUBREAPER,
        ctypes.addressof(state),
        0,
        0,
        0,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise EvidenceError(
            "cannot query Linux child-subreaper state: "
            f"[Errno {error_number}] {os.strerror(error_number)}"
        )
    return bool(state.value)


def _set_child_subreaper(enabled: bool) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    result = libc.prctl(PR_SET_CHILD_SUBREAPER, int(enabled), 0, 0, 0)
    if result != 0:
        error_number = ctypes.get_errno()
        raise EvidenceError(
            "cannot configure Linux child-subreaper containment: "
            f"[Errno {error_number}] {os.strerror(error_number)}"
        )


def _process_children(pid: int, *, require_exists: bool = False) -> set[int]:
    children: set[int] = set()
    task_root = pathlib.Path(f"/proc/{pid}/task")
    try:
        tasks = tuple(task_root.iterdir())
    except FileNotFoundError as error:
        if require_exists:
            raise EvidenceError(
                f"required process task tree is unavailable for pid {pid}"
            ) from error
        return children
    except OSError as error:
        raise EvidenceError(
            f"cannot enumerate process tasks for pid {pid}: {error}"
        ) from error
    for task in tasks:
        if not task.name.isdecimal():
            continue
        try:
            words = (task / "children").read_text(encoding="ascii").split()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise EvidenceError(
                f"cannot enumerate child processes for pid {pid}: {error}"
            ) from error
        for word in words:
            if word.isdecimal():
                children.add(int(word))
    return children


def _process_descendants(root_pid: int) -> set[int]:
    descendants: set[int] = set()
    pending = list(_process_children(root_pid, require_exists=True))
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(_process_children(pid) - descendants)
    return descendants


def _pidfd_has_exited(pidfd: int) -> bool:
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    return bool(poller.poll(0))


def _pidfd_open(pid: int) -> int:
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.pidfd_open
    except AttributeError:
        function = None
    if function is not None:
        function.argtypes = [ctypes.c_int, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(pid, 0)
        if result < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        return int(result)
    if platform.machine().lower() not in {
        "aarch64",
        "amd64",
        "arm64",
        "ppc64",
        "ppc64le",
        "riscv64",
        "s390x",
        "x86_64",
    }:
        raise OSError(
            errno.ENOSYS,
            f"no verified pidfd_open ABI for {platform.machine()}",
        )
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(SYS_PIDFD_OPEN, pid, 0)
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return int(result)


def _pidfd_send_signal(pidfd: int, signal_number: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(pidfd, signal_number, None, 0)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.pidfd_send_signal
    except AttributeError:
        function = None
    if function is not None:
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(pidfd, signal_number, None, 0)
        if result < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        return
    if platform.machine().lower() not in {
        "aarch64",
        "amd64",
        "arm64",
        "ppc64",
        "ppc64le",
        "riscv64",
        "s390x",
        "x86_64",
    }:
        raise OSError(
            errno.ENOSYS,
            f"no verified pidfd_send_signal ABI for {platform.machine()}",
        )
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(
        SYS_PIDFD_SEND_SIGNAL,
        pidfd,
        signal_number,
        0,
        0,
    )
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _snapshot_run_pidfds(
    baseline: set[int], errors: list[str] | None = None
) -> dict[int, int]:
    handles: dict[int, int] = {}
    try:
        descendants = _process_descendants(os.getpid()) - baseline
    except BaseException as error:
        if errors is None:
            raise
        errors.append(
            f"process-tree scan failed: {type(error).__name__}: {error}"
        )
        return handles
    for pid in sorted(descendants):
        try:
            pidfd = _pidfd_open(pid)
        except OSError as error:
            if error.errno in (errno.ENOENT, errno.ESRCH):
                continue
            if errors is not None:
                errors.append(f"cannot open pidfd for run descendant {pid}: {error}")
                continue
            for open_pidfd in handles.values():
                try:
                    os.close(open_pidfd)
                except OSError:
                    pass
            raise EvidenceError(
                f"cannot open pidfd for run descendant {pid}: {error}"
            ) from error
        try:
            exited = _pidfd_has_exited(pidfd)
        except OSError as error:
            os.close(pidfd)
            if errors is not None:
                errors.append(f"cannot poll pidfd for run descendant {pid}: {error}")
                continue
            for open_pidfd in handles.values():
                try:
                    os.close(open_pidfd)
                except OSError:
                    pass
            raise EvidenceError(
                f"cannot poll pidfd for run descendant {pid}: {error}"
            ) from error
        if exited:
            os.close(pidfd)
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            continue
        handles[pid] = pidfd
    return handles


def _signal_pidfds(
    handles: dict[int, int], signal_number: int
) -> list[str]:
    errors: list[str] = []
    for pid, pidfd in handles.items():
        try:
            _pidfd_send_signal(pidfd, signal_number)
        except OSError as error:
            if error.errno != errno.ESRCH:
                errors.append(
                    f"pidfd signal {signal_number} failed for {pid}: {error}"
                )
    return errors


def _wait_pidfds(handles: dict[int, int], timeout_seconds: float) -> set[int]:
    if not handles:
        return set()
    poller = select.poll()
    by_fd = {pidfd: pid for pid, pidfd in handles.items()}
    for pidfd in by_fd:
        try:
            poller.register(pidfd, select.POLLIN)
        except OSError:
            continue
    deadline = time.monotonic() + timeout_seconds
    active = set(handles)
    while active:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            events = poller.poll(max(1, math.ceil(remaining * 1000)))
        except OSError:
            break
        for pidfd, _event in events:
            pid = by_fd[pidfd]
            active.discard(pid)
            try:
                poller.unregister(pidfd)
            except (KeyError, OSError):
                pass
    return active


def _reap_run_pids(pids: set[int], timeout_seconds: float) -> None:
    pending = set(pids)
    deadline = time.monotonic() + timeout_seconds
    while pending and time.monotonic() < deadline:
        for pid in tuple(pending):
            try:
                waited, _status = os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                try:
                    exists = pathlib.Path(f"/proc/{pid}").exists()
                except OSError:
                    exists = True
                if not exists:
                    pending.discard(pid)
                continue
            if waited == pid:
                pending.discard(pid)
        if pending:
            time.sleep(0.01)


def _close_pidfds(handles: dict[int, int]) -> None:
    for pidfd in handles.values():
        try:
            os.close(pidfd)
        except OSError:
            pass


def _captured_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _signal_run_group(proc: subprocess.Popen[str], signal_number: int) -> None:
    try:
        os.killpg(proc.pid, signal_number)
    except OSError:
        pass


def _terminate_run_processes_impl(
    proc: subprocess.Popen[str],
    baseline: set[int],
    partial_stdout: str = "",
    partial_stderr: str = "",
) -> tuple[str, str, set[int], list[str]]:
    stdout = partial_stdout
    stderr = partial_stderr
    tracked: dict[int, int] = {}
    cleanup_errors: list[str] = []

    def merge_snapshot() -> int:
        try:
            more = _snapshot_run_pidfds(baseline, cleanup_errors)
        except BaseException as error:
            cleanup_errors.append(
                f"process-tree scan failed: {type(error).__name__}: {error}"
            )
            return 0
        added = 0
        for pid, pidfd in more.items():
            if pid in tracked:
                try:
                    os.close(pidfd)
                except OSError:
                    pass
            else:
                tracked[pid] = pidfd
                added += 1
        return added

    merge_snapshot()
    if proc.poll() is None and proc.pid not in tracked:
        cleanup_errors.append(
            "live launcher was absent from the process-tree containment scan"
        )
    _signal_run_group(proc, signal.SIGTERM)
    cleanup_errors.extend(_signal_pidfds(tracked, signal.SIGTERM))
    try:
        new_stdout, new_stderr = proc.communicate(
            timeout=TERMINATION_GRACE_SECONDS
        )
        stdout = new_stdout if new_stdout is not None else stdout
        stderr = new_stderr if new_stderr is not None else stderr
    except subprocess.TimeoutExpired as error:
        if error.stdout is not None:
            stdout = _captured_text(error.stdout)
        if error.stderr is not None:
            stderr = _captured_text(error.stderr)
    except BaseException:
        pass

    # TERM may make the launcher exit and close its pipes while a rank that
    # ignored TERM or called setsid() remains alive.  Always rescan the
    # subreaper-owned tree and force-kill every run-owned pidfd.
    merge_snapshot()
    _signal_run_group(proc, signal.SIGKILL)
    cleanup_errors.extend(_signal_pidfds(tracked, signal.SIGKILL))
    try:
        new_stdout, new_stderr = proc.communicate(
            timeout=TERMINATION_GRACE_SECONDS
        )
        stdout = new_stdout if new_stdout is not None else stdout
        stderr = new_stderr if new_stderr is not None else stderr
    except subprocess.TimeoutExpired as error:
        if error.stdout is not None:
            stdout = _captured_text(error.stdout)
        if error.stderr is not None:
            stderr = _captured_text(error.stderr)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except OSError:
                pass
    except BaseException:
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except OSError:
                pass

    # A rank can fork between the first post-TERM scan and SIGKILL.  Iterate
    # to a stable, empty subreaper tree so every late child receives its own
    # pidfd-bound SIGKILL before containment is released.
    fixed_point_deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    empty_rounds = 0
    while time.monotonic() < fixed_point_deadline:
        added = merge_snapshot()
        cleanup_errors.extend(_signal_pidfds(tracked, signal.SIGKILL))
        active: set[int] = set()
        for pid, pidfd in tracked.items():
            try:
                if not _pidfd_has_exited(pidfd):
                    active.add(pid)
            except OSError as error:
                cleanup_errors.append(
                    f"pidfd poll failed for {pid}: {error}"
                )
                active.add(pid)
        if not active and added == 0:
            empty_rounds += 1
            if empty_rounds >= 2:
                break
        else:
            empty_rounds = 0
        time.sleep(0.01)

    try:
        proc.wait(timeout=TERMINATION_GRACE_SECONDS)
    except BaseException:
        try:
            proc.kill()
        except BaseException:
            pass
        try:
            proc.wait(timeout=TERMINATION_GRACE_SECONDS)
        except BaseException:
            pass
    try:
        survivors = _wait_pidfds(tracked, TERMINATION_GRACE_SECONDS)
    except BaseException:
        survivors = set(tracked)
    try:
        _reap_run_pids(set(tracked), TERMINATION_GRACE_SECONDS)
    except BaseException:
        pass
    _close_pidfds(tracked)
    return stdout, stderr, survivors, cleanup_errors


def _terminate_run_processes(
    proc: subprocess.Popen[str],
    baseline: set[int],
    partial_stdout: str = "",
    partial_stderr: str = "",
) -> tuple[str, str, set[int], list[str]]:
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    result: tuple[str, str, set[int], list[str]] | None = None
    try:
        result = _terminate_run_processes_impl(
            proc, baseline, partial_stdout, partial_stderr
        )
        return result
    finally:
        # This independent last-resort sweep deliberately does not call the
        # helpers used above: even a second Ctrl-C or an unexpected helper
        # exception must still reach group/pidfd KILL, drain, wait, and reap.
        emergency_errors: list[str] = []
        try:
            emergency = _snapshot_run_pidfds(baseline, emergency_errors)
        except BaseException:
            emergency = {}
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except BaseException:
            pass
        for pidfd in emergency.values():
            try:
                _pidfd_send_signal(pidfd, signal.SIGKILL)
            except BaseException:
                pass
        emergency_deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
        empty_rounds = 0
        while time.monotonic() < emergency_deadline:
            try:
                more = _snapshot_run_pidfds(baseline, emergency_errors)
            except BaseException:
                more = {}
            added = 0
            for pid, pidfd in more.items():
                if pid in emergency:
                    try:
                        os.close(pidfd)
                    except BaseException:
                        pass
                else:
                    emergency[pid] = pidfd
                    added += 1
            active: set[int] = set()
            for pid, pidfd in emergency.items():
                try:
                    _pidfd_send_signal(pidfd, signal.SIGKILL)
                    if not _pidfd_has_exited(pidfd):
                        active.add(pid)
                except ProcessLookupError:
                    pass
                except BaseException:
                    active.add(pid)
            if not active and added == 0:
                empty_rounds += 1
                if empty_rounds >= 2:
                    break
            else:
                empty_rounds = 0
            time.sleep(0.01)
        try:
            proc.kill()
        except BaseException:
            pass
        try:
            proc.communicate(timeout=TERMINATION_GRACE_SECONDS)
        except BaseException:
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except BaseException:
                    pass
            if proc.stderr is not None:
                try:
                    proc.stderr.close()
                except BaseException:
                    pass
        try:
            proc.wait(timeout=TERMINATION_GRACE_SECONDS)
        except BaseException:
            pass
        try:
            _wait_pidfds(emergency, TERMINATION_GRACE_SECONDS)
            _reap_run_pids(set(emergency), TERMINATION_GRACE_SECONDS)
        except BaseException:
            pass
        _close_pidfds(emergency)
        if result is not None:
            result[3].extend(emergency_errors)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def run_logged(
    command: list[str], log_path: pathlib.Path, outer_timeout: int
) -> dict[str, Any]:
    if not RUN_LOGGED_LOCK.acquire(blocking=False):
        raise EvidenceError("run_logged process containment is already active")
    try:
        if threading.active_count() != 1:
            raise EvidenceError(
                "run_logged requires a single-threaded controller so unrelated "
                "children cannot enter its subreaper ownership boundary"
            )
        return _run_logged_single(command, log_path, outer_timeout)
    finally:
        RUN_LOGGED_LOCK.release()


def _run_logged_single(
    command: list[str], log_path: pathlib.Path, outer_timeout: int
) -> dict[str, Any]:
    started = time.perf_counter()
    previous_subreaper = _child_subreaper_state()
    if not previous_subreaper:
        _set_child_subreaper(True)
    baseline: set[int] = set()
    proc: subprocess.Popen[str] | None = None
    try:
        baseline = _process_descendants(os.getpid())
        if baseline:
            raise EvidenceError(
                "run_logged requires no pre-existing child processes; found "
                + ",".join(str(pid) for pid in sorted(baseline))
            )
        pidfd_probe = _pidfd_open(os.getpid())
        try:
            _pidfd_send_signal(pidfd_probe, 0)
        finally:
            os.close(pidfd_probe)
        proc = subprocess.Popen(
            command,
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=outer_timeout)
        except subprocess.TimeoutExpired as error:
            stdout, stderr, survivors, cleanup_errors = _terminate_run_processes(
                proc,
                baseline,
                _captured_text(error.stdout),
                _captured_text(error.stderr),
            )
            if survivors:
                stderr += (
                    ("\n" if stderr and not stderr.endswith("\n") else "")
                    + "containment survivors after SIGKILL: "
                    + ",".join(str(pid) for pid in sorted(survivors))
                    + "\n"
                )
            if cleanup_errors:
                stderr += (
                    ("\n" if stderr and not stderr.endswith("\n") else "")
                    + "containment errors: "
                    + " | ".join(cleanup_errors)
                    + "\n"
                )
            containment_failed = bool(survivors or cleanup_errors)
            state = (
                "CONTAINMENT_FAILURE"
                if containment_failed
                else "PYTHON_TIMEOUT"
            )
            elapsed = time.perf_counter() - started
            _write_run_log(
                log_path, command, state, elapsed, stdout, stderr
            )
            if containment_failed:
                raise EvidenceError(
                    f"command timeout containment failed; see {log_path}"
                ) from error
            raise EvidenceError(
                f"command exceeded outer timeout; see {log_path}"
            ) from error
        except BaseException:
            _terminate_run_processes(proc, baseline)
            raise

        try:
            leaked = _snapshot_run_pidfds(baseline)
        except BaseException as error:
            stdout, stderr, survivors, cleanup_errors = _terminate_run_processes(
                proc, baseline, stdout, stderr
            )
            detail = [
                f"post-run containment scan failed: {type(error).__name__}: {error}",
                *cleanup_errors,
            ]
            if survivors:
                detail.append(
                    "survivors="
                    + ",".join(str(pid) for pid in sorted(survivors))
                )
            stderr += (
                ("\n" if stderr and not stderr.endswith("\n") else "")
                + "containment errors: "
                + " | ".join(detail)
                + "\n"
            )
            elapsed = time.perf_counter() - started
            _write_run_log(
                log_path,
                command,
                "CONTAINMENT_FAILURE",
                elapsed,
                stdout,
                stderr,
            )
            raise EvidenceError(
                f"post-run containment verification failed; see {log_path}"
            ) from error
        leaked_pids = set(leaked)
        _close_pidfds(leaked)
        if leaked_pids:
            stdout, stderr, survivors, cleanup_errors = _terminate_run_processes(
                proc, baseline, stdout, stderr
            )
            detail = sorted(survivors or leaked_pids)
            stderr += (
                ("\n" if stderr and not stderr.endswith("\n") else "")
                + "unexpected surviving descendants: "
                + ",".join(str(pid) for pid in detail)
                + "\n"
            )
            if cleanup_errors:
                stderr += (
                    "containment errors: "
                    + " | ".join(cleanup_errors)
                    + "\n"
                )
            elapsed = time.perf_counter() - started
            _write_run_log(
                log_path, command, "DESCENDANT_LEAK", elapsed, stdout, stderr
            )
            raise EvidenceError(
                f"command left surviving descendants; see {log_path}"
            )

        elapsed = time.perf_counter() - started
        _write_run_log(
            log_path,
            command,
            proc.returncode,
            elapsed,
            stdout,
            stderr,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "stdout": stdout,
            "stderr": stderr,
            "log": str(log_path),
        }
    finally:
        if not previous_subreaper:
            _set_child_subreaper(False)


def _device_count(stdout: str) -> int:
    matches = []
    for line in stdout.splitlines():
        match = re.fullmatch(r"compatible_cuda_devices=([0-9]+)", line)
        if match:
            matches.append(int(match.group(1)))
    if len(matches) != 1:
        raise EvidenceError(
            f"expected exactly one CUDA device-count record, found {len(matches)}"
        )
    return matches[0]


def load_build_receipt(repo: pathlib.Path, build_dir: pathlib.Path) -> dict[str, Any]:
    """Require and verify the MPI build receipt and benchmark ELFs."""

    path = build_dir / "build-provenance.json"
    if not path.is_file():
        raise EvidenceError(
            "required MPI build receipt is absent: "
            f"{path}; rebuild with scripts/build-meep-cuda-mpi.sh"
        )
    try:
        value = verify_build_receipt(path, repo)
    except ProvenanceError as error:
        raise EvidenceError(f"invalid MPI build receipt {path}: {error}") from error
    if value.get("build_kind") != "cuda-mpi-python-fp32":
        raise EvidenceError("MPI build receipt has the wrong build_kind")
    receipt_id = value.get("receipt_id")
    if type(receipt_id) is not str or not re.fullmatch(r"[0-9a-f]{64}", receipt_id):
        raise EvidenceError("MPI build receipt has no valid receipt_id")
    artifacts = value.get("artifacts")
    if type(artifacts) is not dict:
        raise EvidenceError("MPI build receipt contains no artifact map")
    verified_artifacts: dict[str, dict[str, Any]] = {}
    for name in ("gpu_mpi_performance", "gpu_step_db_test"):
        item = artifacts.get(name)
        if type(item) is not dict or type(item.get("path")) is not str:
            raise EvidenceError(f"MPI build receipt is missing artifact {name!r}")
        artifact_path = pathlib.Path(item["path"])
        if not artifact_path.is_absolute():
            artifact_path = repo / artifact_path
        if not artifact_path.is_file():
            raise EvidenceError(f"receipt artifact is absent: {artifact_path}")
        actual_sha256 = sha256_file(artifact_path)
        if item.get("sha256") != actual_sha256:
            raise EvidenceError(f"receipt artifact hash mismatch: {artifact_path}")
        verified_artifacts[name] = {
            "path": str(artifact_path.resolve()),
            "sha256": actual_sha256,
        }
    return {
        "status": "VERIFIED",
        "path": str(path),
        "receipt_file": file_record(path, repo),
        "receipt_id": receipt_id,
        "build_input_id": value.get("build_input_id"),
        "artifact_set_id": value.get("artifact_set_id"),
        "verified_artifacts": verified_artifacts,
    }


def _forwarded_base_environment(repo: pathlib.Path) -> dict[str, str]:
    home_value = os.environ.get("HOME")
    if not home_value or not pathlib.Path(home_value).is_dir():
        raise EvidenceError("Open MPI requires the launch user's real HOME")
    openmpi_params = repo / "environment" / "openmpi-qualification-mca-params.conf"
    prefix = repo / ".envs" / "meep-gpu-cuda-mpi"
    ompi_components = prefix / "lib" / "openmpi"
    pmix_components = prefix / "lib" / "pmix"
    if not openmpi_params.is_file():
        raise EvidenceError(
            f"source-bound Open MPI parameter file is missing: {openmpi_params}"
        )
    for path, label in (
        (ompi_components, "Open MPI component path"),
        (pmix_components, "PMIx component path"),
    ):
        if not path.is_dir():
            raise EvidenceError(f"{label} is missing: {path}")
    environment = {
        "HOME": home_value,
        "CCACHE_DIR": str(repo / ".micromamba" / "cache" / "ccache"),
        "OMP_NUM_THREADS": "1",
        "OMPI_MCA_mca_base_param_files": str(openmpi_params),
        "OMPI_MCA_mca_base_component_path": str(ompi_components),
        "PMIX_MCA_mca_base_param_files": str(openmpi_params),
        "PMIX_MCA_mca_base_component_path": str(pmix_components),
        "PRTE_MCA_mca_base_param_files": str(openmpi_params),
    }
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "MEEP_GPU_ALLOW_OVERSUBSCRIBE",
    ):
        if key in os.environ:
            environment[key] = os.environ[key]
    return environment


def _mpi_transport_environment(
    base_environment: dict[str, str], transport: str
) -> dict[str, str]:
    environment = dict(base_environment)
    environment["MEEP_GPU_MPI_TRANSPORT"] = transport
    if transport == "cuda-aware":
        environment.update(
            {
                "OMPI_MCA_opal_cuda_support": "true",
                "OMPI_MCA_btl": os.environ.get(
                    "MEEP_GPU_MPI_MCA_BTL", "self,smcuda,tcp"
                ),
                "UCX_MEMTYPE_CACHE": "n",
            }
        )
    return environment


def _device_query_environment(
    base_environment: dict[str, str],
) -> dict[str, str]:
    # The worker's device-query branch exits before parsing or allocating the
    # fixed workload and emits no v4 benchmark/initialization records. Keep
    # this identity-only command independent of workload-profile inputs.
    missing = CANONICAL_MPI_ENVIRONMENT_KEYS - base_environment.keys()
    if missing:
        raise EvidenceError(
            "device query is missing canonical MPI configuration: "
            + ",".join(sorted(missing))
        )
    allowed = {
        "HOME",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        *CANONICAL_MPI_ENVIRONMENT_KEYS,
    }
    return {
        key: value for key, value in base_environment.items() if key in allowed
    }


def controller() -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    mamba = repo / ".tools" / "micromamba"
    mamba_root = repo / ".micromamba"
    prefix = repo / ".envs" / "meep-gpu-cuda-mpi"
    build_dir = repo / "build" / "meep-cuda-mpi-python-fp32"
    for path, label in ((mamba, "Micromamba"),):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise EvidenceError(f"{label} is missing or not executable: {path}")
    if not prefix.is_dir():
        raise EvidenceError(f"isolated CUDA/MPI environment is missing: {prefix}")
    build_receipt = load_build_receipt(repo, build_dir)
    benchmark = pathlib.Path(
        build_receipt["verified_artifacts"]["gpu_mpi_performance"]["path"]
    )
    dft_test = pathlib.Path(
        build_receipt["verified_artifacts"]["gpu_step_db_test"]["path"]
    )
    for path, label in (
        (benchmark, "receipt-verified multi-GPU benchmark ELF"),
        (dft_test, "receipt-verified DFT/CW MPI regression ELF"),
    ):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise EvidenceError(f"{label} is missing or not executable: {path}")

    result_dir = pathlib.Path(
        os.environ.get("MEEP_GPU_MULTI_RESULT_DIR", repo / "benchmark-results")
    ).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"multi-gpu-{timestamp}"
    raw_path = result_dir / f"{stem}.csv"
    summary_path = result_dir / f"{stem}-summary.csv"
    report_path = result_dir / f"{stem}-report.json"
    complete_path = result_dir / f"{stem}-COMPLETE.json"
    combined_log_path = result_dir / f"{stem}.log"

    requested_ranks = parse_rank_list(
        os.environ.get("MEEP_GPU_MULTI_RANKS", "1,2,4")
    )
    minimum_efficiency = parse_probability(
        "MEEP_GPU_MULTI_MIN_EFFICIENCY",
        os.environ.get("MEEP_GPU_MULTI_MIN_EFFICIENCY", "0.55"),
    )
    observable_tolerance = parse_probability(
        "MEEP_GPU_MULTI_OBSERVABLE_TOLERANCE",
        os.environ.get("MEEP_GPU_MULTI_OBSERVABLE_TOLERANCE", "0.002"),
    )
    require_all = parse_boolean_flag(
        "MEEP_GPU_MULTI_REQUIRE_ALL",
        os.environ.get("MEEP_GPU_MULTI_REQUIRE_ALL", "1"),
    )
    transport = normalize_transport(os.environ.get("MEEP_GPU_MPI_TRANSPORT", "pinned"))
    mpi_timeout = parse_positive_int(
        "MEEP_GPU_MULTI_TIMEOUT_SECONDS",
        os.environ.get("MEEP_GPU_MULTI_TIMEOUT_SECONDS", "600"),
    )
    regression_timeout = parse_positive_int(
        "MEEP_GPU_MULTI_REGRESSION_TIMEOUT_SECONDS",
        os.environ.get("MEEP_GPU_MULTI_REGRESSION_TIMEOUT_SECONDS", "300"),
    )
    failure_timeout = parse_positive_int(
        "MEEP_GPU_MULTI_FAILURE_TIMEOUT_SECONDS",
        os.environ.get("MEEP_GPU_MULTI_FAILURE_TIMEOUT_SECONDS", "20"),
    )
    quick = parse_boolean_flag(
        "MEEP_GPU_MULTI_QUICK",
        os.environ.get("MEEP_GPU_MULTI_QUICK", "0"),
    )
    pixels = parse_positive_int(
        "MEEP_GPU_MULTI_PIXELS",
        os.environ.get("MEEP_GPU_MULTI_PIXELS", "96" if quick else "192"),
    )
    warmup_steps = parse_positive_int(
        "MEEP_GPU_MULTI_WARMUP_STEPS",
        os.environ.get("MEEP_GPU_MULTI_WARMUP_STEPS", "6" if quick else "12"),
    )
    measured_steps = parse_positive_int(
        "MEEP_GPU_MULTI_STEPS",
        os.environ.get("MEEP_GPU_MULTI_STEPS", "20" if quick else "80"),
    )
    repeat_count = parse_positive_int(
        "MEEP_GPU_MULTI_REPEATS",
        os.environ.get("MEEP_GPU_MULTI_REPEATS", "1" if quick else "3"),
    )

    base_environment = _forwarded_base_environment(repo)
    query_environment = _device_query_environment(base_environment)
    query_command = [
        *_clean_run_prefix(mamba, mamba_root, prefix, query_environment),
        "env",
        "MEEP_GPU_MULTI_QUERY_DEVICE_COUNT=1",
        str(benchmark),
    ]
    query = run_logged(
        query_command,
        result_dir / f"{stem}-device-query.log",
        outer_timeout=60,
    )
    available_gpus = _device_count(query["stdout"])
    if query["returncode"] not in ({0} if available_gpus else {77}):
        raise EvidenceError("CUDA runtime device-count query returned an inconsistent status")
    if available_gpus < 1:
        raise EvidenceError("no physical CUDA GPU is visible")
    query_log_artifact = file_record(pathlib.Path(query["log"]), repo)

    regressions: list[dict[str, Any]] = []
    if available_gpus >= 2:
        regression_environment = _mpi_transport_environment(
            base_environment, "pinned"
        )
        success_environment = dict(regression_environment)
        success_environment["MEEP_GPU_TEST_DFT_NORM_ONLY"] = "1"
        success_command = _timed_mpi_command(
            _clean_run_prefix(mamba, mamba_root, prefix, success_environment),
            regression_timeout,
            2,
            dft_test,
        )
        success = run_logged(
            success_command,
            result_dir / f"{stem}-dft-norm-success.log",
            outer_timeout=regression_timeout + 30,
        )
        marker = "PASS: contiguous and persistent CUDA DFT norms agree"
        if success["returncode"] != 0 or success["stdout"].count(marker) != 1:
            raise EvidenceError("two-GPU DFT norm correctness gate failed")
        regressions.append(
            {
                "name": "two_gpu_dft_norm",
                "status": "PASS",
                "command": success["command"],
                "log": success["log"],
                "log_artifact": file_record(pathlib.Path(success["log"]), repo),
            }
        )

        timing_environment = _mpi_transport_environment(
            regression_environment, transport
        )
        timing_environment["MEEP_GPU_TEST_BOUNDARY_TIMING_ONLY"] = "1"
        timing_command = _timed_mpi_command(
            _clean_run_prefix(
                mamba, mamba_root, prefix, timing_environment
            ),
            regression_timeout,
            2,
            dft_test,
        )
        timing = run_logged(
            timing_command,
            result_dir / f"{stem}-boundary-timing-success.log",
            outer_timeout=regression_timeout + 30,
        )
        timing_marker = (
            "PASS: distributed CUDA boundary timing separates MPI completion"
        )
        if (
            timing["returncode"] != 0
            or timing["stdout"].count(timing_marker) != 1
        ):
            raise EvidenceError(
                "two-GPU CUDA boundary timing correctness gate failed"
            )
        regressions.append(
            {
                "name": "two_gpu_boundary_timing",
                "status": "PASS",
                "command": timing["command"],
                "log": timing["log"],
                "log_artifact": file_record(
                    pathlib.Path(timing["log"]), repo
                ),
            }
        )

        failure_cases = (
            (
                "dft_norm_rank_failure",
                "MEEP_GPU_TEST_DFT_NORM_RANK_FAILURE",
                "rank-local failure during a distributed DFT norm reduction",
            ),
            (
                "cw_rank_failure",
                "MEEP_GPU_TEST_CW_RANK_FAILURE",
                "rank-local failure during distributed solve_cw",
            ),
            (
                "integrate2_swapped_ownership",
                "MEEP_GPU_TEST_INTEGRATE2_SWAPPED_OWNERSHIP",
                "fields must have identical per-chunk process ownership",
            ),
            (
                "integrate2_rank_failure",
                "MEEP_GPU_TEST_INTEGRATE2_RANK_FAILURE",
                "rank-local failure during distributed integrate2",
            ),
        )
        forbidden_failure_statuses = {0, 124, 125, 126, 127, 137}
        for name, variable, diagnostic in failure_cases:
            failure_environment = dict(regression_environment)
            failure_environment[variable] = "1"
            failure_command = _timed_mpi_command(
                _clean_run_prefix(
                    mamba, mamba_root, prefix, failure_environment
                ),
                failure_timeout,
                2,
                dft_test,
            )
            failure = run_logged(
                failure_command,
                result_dir / f"{stem}-{name}.log",
                outer_timeout=failure_timeout + 30,
            )
            combined_output = failure["stdout"] + failure["stderr"]
            if (
                failure["returncode"] in forbidden_failure_statuses
                or combined_output.count(diagnostic) < 1
            ):
                raise EvidenceError(f"expected MPI abort gate {name} failed")
            regressions.append(
                {
                    "name": name,
                    "status": "PASS",
                    "returncode": failure["returncode"],
                    "command": failure["command"],
                    "log": failure["log"],
                    "log_artifact": file_record(
                        pathlib.Path(failure["log"]), repo
                    ),
                }
            )
    else:
        regressions.append(
            {
                "name": "two_gpu_mpi_regressions",
                "status": "SKIP",
                "reason": "requires two visible compatible CUDA devices",
            }
        )

    skipped_ranks: list[dict[str, Any]] = []
    executable_ranks: list[int] = []
    for ranks in requested_ranks:
        if ranks > available_gpus:
            skipped_ranks.append(
                {
                    "ranks": ranks,
                    "reason": (
                        f"requested {ranks} ranks but only {available_gpus} "
                        "compatible GPUs are visible"
                    ),
                }
            )
            continue
        executable_ranks.append(ranks)
    if not executable_ranks:
        raise EvidenceError("no requested multi-GPU rank count could be run")

    records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []
    for repeat_index in range(1, repeat_count + 1):
        for ranks in executable_ranks:
            run_environment = _mpi_transport_environment(
                base_environment, transport
            )
            run_environment.update(
                {
                    "MEEP_GPU_MPI_COMPLETION": "waitsome",
                    "MEEP_GPU_MULTI_PIXELS": str(pixels),
                    "MEEP_GPU_MULTI_WARMUP_STEPS": str(warmup_steps),
                    "MEEP_GPU_MULTI_STEPS": str(measured_steps),
                    "MEEP_GPU_MULTI_LOOP_TILE_BASE_DB": "128",
                    "MEEP_GPU_MULTI_BFAST": "0",
                    "MEEP_GPU_MULTI_DISABLE_SOURCE": "0",
                    "MEEP_GPU_MULTI_SOURCE_PROFILE": (
                        RELEASE_PROFILE["source_profile"]
                    ),
                    "MEEP_GPU_MULTI_OVERLAP_MATERIAL": "0",
                    "MEEP_GPU_MULTI_INITIAL_CONDITION": (
                        RELEASE_PROFILE["initial_condition"]
                    ),
                }
            )
            if quick:
                run_environment["MEEP_GPU_MULTI_QUICK"] = "1"
            if "MEEP_GPU_MULTI_PRINT_TIMES" in os.environ:
                run_environment["MEEP_GPU_MULTI_PRINT_TIMES"] = os.environ[
                    "MEEP_GPU_MULTI_PRINT_TIMES"
                ]
            command = _timed_mpi_command(
                _clean_run_prefix(mamba, mamba_root, prefix, run_environment),
                mpi_timeout,
                ranks,
                benchmark,
            )
            run = run_logged(
                command,
                result_dir
                / f"{stem}-repeat-{repeat_index:02d}-{ranks}-ranks.log",
                outer_timeout=mpi_timeout + 30,
            )
            if run["returncode"] != 0:
                raise EvidenceError(
                    f"repeat {repeat_index}, {ranks}-rank benchmark failed "
                    f"with status {run['returncode']}"
                )
            reject_unknown_worker_markers(run["stdout"], run["stderr"])
            record = extract_record(
                run["stdout"],
                expected_ranks=ranks,
                expected_pixels=pixels,
                expected_warmup_steps=warmup_steps,
                expected_steps=measured_steps,
                expected_transport=transport,
            )
            initialization_records = extract_initialization_records(
                run["stdout"],
                expected_profile=RELEASE_PROFILE["initial_condition"],
            )
            source_profile_record = extract_source_profile_record(
                run["stdout"],
                expected_profile=RELEASE_PROFILE["source_profile"],
            )
            if (
                initialization_records["initial_condition"]["profile"]
                != record["initial_condition"]
            ):
                raise EvidenceError(
                    "initial-condition marker and worker record differ"
                )
            device_records = extract_device_records(run["stdout"], ranks)
            log_artifact = file_record(pathlib.Path(run["log"]), repo)
            records.append(record)
            sample = {
                "repeat_index": repeat_index,
                "mpi_ranks": ranks,
                "record": record,
                "source_profile": source_profile_record,
                **initialization_records,
                "devices": device_records,
                "command": run["command"],
                "returncode": run["returncode"],
                "elapsed_seconds": run["elapsed_seconds"],
                "log": run["log"],
                "log_artifact": log_artifact,
            }
            samples.append(sample)
            run_entries.append(
                {
                    "repeat_index": repeat_index,
                    "ranks": ranks,
                    "command": run["command"],
                    "returncode": run["returncode"],
                    "elapsed_seconds": run["elapsed_seconds"],
                    "log": run["log"],
                    "log_artifact": log_artifact,
                    "devices": device_records,
                    **initialization_records,
                }
            )

    executed_ranks = list(executable_ranks)
    comparisons, overall_gate = summarize_samples(
        samples,
        executed_ranks,
        repeat_count,
        observable_tolerance,
        minimum_efficiency,
    )
    workload = {
        "pixels": pixels,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "quick": quick,
        **FIXED_WORKER_PROFILE,
        "source_profile": RELEASE_PROFILE["source_profile"],
    }
    qualification = qualification_gates(
        overall_gate,
        requested_ranks,
        executed_ranks,
        skipped_ranks,
        require_all,
        workload=workload,
        repeat_count=repeat_count,
        timing_statistic="median",
        transport=transport,
        minimum_efficiency=minimum_efficiency,
        observable_tolerance=observable_tolerance,
    )
    missing_required = bool(skipped_ranks and require_all)
    controller_pass = (
        qualification["release"]["pass"]
        if require_all
        else qualification["development"]["pass"]
    )

    raw_fields = [
        "repeat_index",
        "mpi_ranks",
        "pixels",
        "cells",
        "warmup_steps",
        "steps",
        "loop_tile_base_db",
        "bfast",
        "source_enabled",
        "overlap_material",
        "seconds",
        "mcells_per_second",
        "requested_transport",
        "selected_transport",
        "completion_policy",
        "initial_condition",
        "cut_probe_l2",
        "ez_l2",
        "ez_weighted_checksum",
        "energy",
        "dft_norm",
        "remote_cut_scalars",
        "remote_cut_l2",
        "cpu_calls",
        "cuda_calls",
        *[
            f"{phase}_{backend}_calls"
            for phase in ALL_PHASES
            for backend in ("cpu", "cuda")
        ],
        "h2d_bytes",
        "d2h_bytes",
        "mpi_messages",
        "mpi_scalars",
        "cuda_aware_bytes",
        "pinned_bytes",
    ]
    raw_stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        raw_stream, fieldnames=raw_fields, extrasaction="ignore"
    )
    writer.writeheader()
    raw_rows: list[dict[str, Any]] = []
    for sample in samples:
        row = {"repeat_index": sample["repeat_index"], **sample["record"]}
        for phase in ALL_PHASES:
            for backend in ("cpu", "cuda"):
                row[f"{phase}_{backend}_calls"] = sample["record"][
                    "phase_calls"
                ][phase][f"{backend}_calls"]
        raw_rows.append(row)
    writer.writerows(raw_rows)
    atomic_write_text(raw_path, raw_stream.getvalue())
    summary_fields = list(comparisons[0])
    summary_stream = io.StringIO(newline="")
    writer = csv.DictWriter(summary_stream, fieldnames=summary_fields)
    writer.writeheader()
    summary_rows = []
    for comparison in comparisons:
        row = dict(comparison)
        row["seconds_samples"] = json.dumps(
            row["seconds_samples"], separators=(",", ":")
        )
        summary_rows.append(row)
    writer.writerows(summary_rows)
    atomic_write_text(summary_path, summary_stream.getvalue())

    log_paths = [pathlib.Path(entry["log"]) for entry in run_entries]
    log_paths.extend(
        pathlib.Path(item["log"])
        for item in regressions
        if "log" in item
    )
    log_paths.insert(0, pathlib.Path(query["log"]))
    combined_log = "".join(
        f"===== {path.name} =====\n" + path.read_text(encoding="utf-8")
        for path in log_paths
    )
    atomic_write_text(combined_log_path, combined_log)

    artifact_records = {
        "raw_csv": file_record(raw_path, repo),
        "summary_csv": file_record(summary_path, repo),
        "combined_log": file_record(combined_log_path, repo),
    }
    recorded_logs = [("device query log", query_log_artifact)]
    recorded_logs.extend(
        (f"MPI regression log {item['name']}", item["log_artifact"])
        for item in regressions
        if "log_artifact" in item
    )
    recorded_logs.extend(
        (
            f"repeat {sample['repeat_index']} {sample['mpi_ranks']}-rank log",
            sample["log_artifact"],
        )
        for sample in samples
    )
    for label, record in [*recorded_logs, *artifact_records.items()]:
        verify_file_record(record, repo, label)
    end_build_receipt = load_build_receipt(repo, build_dir)
    if end_build_receipt != build_receipt:
        raise EvidenceError("MPI build receipt or verified artifacts changed during the run")
    end_integrity = {
        "status": "VERIFIED",
        "build_receipt_id": end_build_receipt["receipt_id"],
        "individual_log_count": len(recorded_logs),
        "generated_artifact_count": len(artifact_records),
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "state": "COMPLETE",
        "run_id": stem,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": {
            # `pass` is deliberately release qualification, never a partial
            # 1/2-GPU result. Development runs have their own explicit gate.
            "pass": qualification["release"]["pass"],
            "controller_pass": controller_pass,
            "missing_required_ranks": missing_required,
            **qualification,
        },
        "request": {
            "requested_ranks": requested_ranks,
            "executed_ranks": executed_ranks,
            "skipped_ranks": skipped_ranks,
            "available_compatible_gpus": available_gpus,
            "require_all": require_all,
            "transport": transport,
            "repeat_count": repeat_count,
            "timing_statistic": "median",
            "workload": workload,
            "thresholds": {
                "minimum_parallel_efficiency": minimum_efficiency,
                "maximum_observable_relative_error": observable_tolerance,
                "mpi_timeout_seconds": mpi_timeout,
                "regression_timeout_seconds": regression_timeout,
                "failure_timeout_seconds": failure_timeout,
            },
        },
        "device_query": {
            "command": query["command"],
            "returncode": query["returncode"],
            "log": query["log"],
            "log_artifact": query_log_artifact,
        },
        "build_receipt": build_receipt,
        "end_integrity": end_integrity,
        "mpi_regressions": regressions,
        "runs": run_entries,
        "samples": samples,
        "records": records,
        "comparisons": comparisons,
        "artifacts": artifact_records,
    }
    publish_complete(
        repo,
        report_path,
        complete_path,
        report,
        controller_pass=controller_pass,
        release_pass=qualification["release"]["pass"],
    )

    print(f"Multi-GPU raw CSV written to {raw_path}")
    print(f"Multi-GPU scaling summary written to {summary_path}")
    print(f"Multi-GPU report written to {report_path}")
    print(f"Multi-GPU completion marker written to {complete_path}")
    print(f"Multi-GPU log written to {combined_log_path}")
    with summary_path.open(encoding="utf-8") as stream:
        sys.stdout.write(stream.read())
    if missing_required:
        print(
            "error: one or more required GPU rank counts were unavailable",
            file=sys.stderr,
        )
    return 0 if controller_pass else 1


def main() -> int:
    try:
        return controller()
    except EvidenceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
