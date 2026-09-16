#!/usr/bin/env python3
"""Run the receipt-bound two-rank gpmeep Python/MPI correctness matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pathlib
import pwd
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from gpmeep_benchmark_evidence import atomic_write_json, atomic_write_text
from gpmeep_provenance import (
    ProvenanceError,
    StatHashCache,
    file_record,
    source_snapshot,
    verify_build_receipt,
)


PROBE_PREFIX = "gpmeep-mpi-python-probe-v1:"
DIVIDE_PREFIX = "gpmeep-divide-mpi-validation-v1:"
SCHEMA_VERSION = 1
RECEIPT_KIND = "cuda-mpi-python-fp32"
LANES = (
    "cpu-hidden",
    "cuda-pinned-waitsome",
    "cuda-aware-waitall",
)
WORKLOAD = {
    "resolution": 20,
    "cell_size": [4.0, 4.0, 0.0],
    "pml_thickness": 0.5,
    "run_until": 8.0,
    "expected_timesteps": 320,
}
PHASES = {
    "curl": ("dispatch", "curl"),
    "update_eh": ("field_updates", "update_eh"),
    "polarization": ("polarizations", "polarization"),
    "source": ("sources", "source"),
    "boundary": ("boundaries", "boundary"),
    "dft": ("dfts", "dft"),
}
REQUIRED_PHASES = ("curl", "update_eh", "source", "boundary", "dft")
FINAL_FILES = (
    "report.json",
    "report.md",
    "artifacts.sha256.json",
    "COMPLETE",
    "FAILED.json",
)


class EvidenceError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EvidenceError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def reject_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON constant {value!r}")


def load_json_text(text: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except EvidenceError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise EvidenceError(f"malformed JSON: {error}") from error


def finite_number(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def canonical_field_sha256(values: list[float]) -> str:
    import struct

    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def normalize_uuid(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    normalized = normalized.replace("-", "")
    require(bool(re.fullmatch(r"[0-9a-f]{32}", normalized)), "invalid GPU UUID")
    return normalized


def extract_probe(stdout: str) -> dict[str, Any]:
    payloads = [
        line[len(PROBE_PREFIX) :]
        for line in stdout.replace("\r", "\n").splitlines()
        if line.startswith(PROBE_PREFIX)
    ]
    require(len(payloads) == 1, f"expected one probe record, found {len(payloads)}")
    record = load_json_text(payloads[0])
    require(type(record) is dict, "probe record is not an object")
    return record


def validate_file_record(
    record: Any, label: str, *, allow_empty: bool = False
) -> None:
    require(
        type(record) is dict
        and set(record) == {"path", "size_bytes", "sha256"},
        f"invalid {label} file record",
    )
    require(type(record["path"]) is str and record["path"], f"invalid {label} path")
    require(
        type(record["size_bytes"]) is int
        and (
            record["size_bytes"] >= 0
            if allow_empty
            else record["size_bytes"] > 0
        ),
        f"invalid {label} size",
    )
    require(bool(re.fullmatch(r"[0-9a-f]{64}", record["sha256"])),
            f"invalid {label} SHA-256")


def validate_runtime_against_receipt(
    runtime: dict[str, Any], receipt: dict[str, Any], repo: pathlib.Path
) -> dict[str, Any]:
    expected = {
        "python": receipt.get("toolchain", {}).get("python"),
        "extension": receipt.get("artifacts", {}).get("python_extension"),
        "libmeep": receipt.get("artifacts", {}).get("libmeep"),
    }
    require(set(runtime) == set(expected), "runtime/receipt artifact schema mismatch")
    for name, actual in runtime.items():
        validate_file_record(actual, f"runtime {name}")
        recorded = expected[name]
        require(type(recorded) is dict, f"receipt has no {name} artifact")
        recorded_path = recorded.get("path")
        require(type(recorded_path) is str and recorded_path,
                f"receipt {name} path is invalid")
        path = pathlib.Path(recorded_path)
        if not path.is_absolute():
            path = repo / path
        actual_path = pathlib.Path(actual["path"])
        if not actual_path.is_absolute():
            actual_path = repo / actual_path
        require(
            actual_path.resolve(strict=True) == path.resolve(strict=True)
            and actual["size_bytes"] == recorded.get("size_bytes")
            and actual["sha256"] == recorded.get("sha256"),
            f"runtime {name} does not match the build receipt",
        )
    return {"pass": True, "artifacts": sorted(expected)}


def phase_value(statistics: dict[str, Any], phase: str, backend: str, suffix: str) -> int:
    group, stem = PHASES[phase]
    value = statistics.get(group, {}).get(f"{backend}_{stem}_{suffix}")
    require(type(value) is int and value >= 0,
            f"invalid {group}.{backend}_{stem}_{suffix}")
    return value


def validate_metrics(metrics: Any) -> dict[str, Any]:
    expected_keys = {
        "timesteps",
        "meep_time",
        "shape",
        "field_values",
        "field_sha256",
        "field_sum",
        "field_l1",
        "field_l2",
        "field_maximum_absolute",
        "field_weighted_checksum",
        "flux_values",
    }
    require(type(metrics) is dict and set(metrics) == expected_keys,
            "probe metrics schema mismatch")
    require(metrics["timesteps"] == WORKLOAD["expected_timesteps"],
            "probe timestep contract mismatch")
    require(finite_number(metrics["meep_time"]),
            "invalid probe Meep time")
    require(math.isclose(float(metrics["meep_time"]), WORKLOAD["run_until"],
                         rel_tol=0.0, abs_tol=1e-12),
            "probe Meep-time contract mismatch")
    require(metrics["shape"] == [80, 80], "probe field shape mismatch")
    values = metrics["field_values"]
    require(type(values) is list and len(values) == 6400,
            "probe field vector length mismatch")
    require(all(finite_number(value) for value in values),
            "probe field vector contains a non-finite value")
    promoted = [float(value) for value in values]
    require(metrics["field_sha256"] == canonical_field_sha256(promoted),
            "probe field SHA-256 mismatch")
    for name in (
        "field_sum",
        "field_l1",
        "field_l2",
        "field_maximum_absolute",
        "field_weighted_checksum",
    ):
        require(finite_number(metrics[name]), f"invalid probe metric {name}")
    require(metrics["field_l1"] > 0 and metrics["field_l2"] > 0
            and metrics["field_maximum_absolute"] > 0,
            "probe field signal is zero")
    recomputed = {
        "field_sum": math.fsum(promoted),
        "field_l1": math.fsum(abs(value) for value in promoted),
        "field_l2": math.sqrt(math.fsum(value * value for value in promoted)),
        "field_maximum_absolute": max(abs(value) for value in promoted),
        "field_weighted_checksum": math.fsum(
            (index + 1) * value for index, value in enumerate(promoted)
        ),
    }
    require(
        all(
            math.isclose(
                float(metrics[name]), value, rel_tol=2e-12, abs_tol=2e-12
            )
            for name, value in recomputed.items()
        ),
        "probe field summaries disagree with the complete field vector",
    )
    fluxes = metrics["flux_values"]
    require(type(fluxes) is list and len(fluxes) == 3
            and all(finite_number(value) for value in fluxes),
            "invalid probe flux vector")
    require(any(abs(float(value)) > 1e-12 for value in fluxes),
            "probe flux signal is zero")
    return metrics


def validate_probe_record(
    record: dict[str, Any],
    *,
    lane: str,
    receipt_id: str,
    nonce: str,
    visible_devices: str,
) -> dict[str, Any]:
    require(
        set(record)
        == {
            "schema_version",
            "lane",
            "receipt_id",
            "nonce",
            "mpi_ranks",
            "workload",
            "metrics",
            "rank_records",
        },
        "probe record schema mismatch",
    )
    require(record["schema_version"] == SCHEMA_VERSION, "probe schema mismatch")
    require(record["lane"] == lane, "probe lane mismatch")
    require(record["receipt_id"] == receipt_id, "probe receipt mismatch")
    require(record["nonce"] == nonce, "probe nonce mismatch")
    require(record["mpi_ranks"] == 2, "probe rank count mismatch")
    require(record["workload"] == WORKLOAD, "probe workload mismatch")
    metrics = validate_metrics(record["metrics"])
    ranks = record["rank_records"]
    require(type(ranks) is list and len(ranks) == 2, "probe rank records incomplete")
    require(sorted(item.get("rank") for item in ranks) == [0, 1],
            "probe rank identities incomplete or duplicated")
    expected_backend = "cpu" if lane == "cpu-hidden" else "cuda"
    active_totals = {
        phase: {"calls": 0, "points": 0} for phase in PHASES
    }
    identifiers: list[str] = []
    ordinals: list[int] = []
    pids: list[int] = []
    affinities: set[tuple[str, int, int]] = set()
    runtime_reference = None
    for rank in sorted(ranks, key=lambda item: item["rank"]):
        require(rank.get("metrics") == metrics, "rank numerical metrics differ")
        require(rank.get("active_backend") == expected_backend
                and rank.get("requested_backend") == expected_backend,
                "rank selected the wrong backend")
        require(type(rank.get("pid")) is int and rank["pid"] > 1,
                "invalid rank PID")
        pids.append(rank["pid"])
        affinity = rank.get("physical_affinity")
        require(type(affinity) is list and len(affinity) == 1
                and type(affinity[0]) is dict,
                "rank is not bound to exactly one physical CPU core")
        core = affinity[0]
        require(
            set(core) == {"host", "package_id", "core_id"}
            and type(core["host"]) is str
            and type(core["package_id"]) is int
            and type(core["core_id"]) is int,
            "invalid physical CPU-core identity",
        )
        core_key = (core["host"], core["package_id"], core["core_id"])
        require(core_key not in affinities, "MPI ranks overlap on a CPU")
        affinities.add(core_key)
        require(finite_number(rank.get("local_seconds"))
                and rank["local_seconds"] > 0,
                "invalid rank-local time")
        runtime = rank.get("runtime")
        require(type(runtime) is dict and set(runtime) == {"python", "extension", "libmeep"},
                "runtime schema mismatch")
        for name, item in runtime.items():
            validate_file_record(item, f"runtime {name}")
        if runtime_reference is None:
            runtime_reference = runtime
        require(runtime == runtime_reference, "rank runtime artifacts differ")
        environment = rank.get("environment")
        require(type(environment) is dict, "rank environment is absent")
        require(environment.get("CUDA_VISIBLE_DEVICES") == visible_devices,
                "rank CUDA visibility differs from controller")
        require(environment.get("MEEP_GPU_BACKEND") == expected_backend,
                "rank backend environment mismatch")
        require(environment.get("OMP_NUM_THREADS") == "1",
                "rank OpenMP budget mismatch")
        require(environment.get("MEEP_GPU_ALLOW_OVERSUBSCRIBE") == "0",
                "rank GPU oversubscription policy mismatch")
        if lane == "cuda-pinned-waitsome":
            require(environment.get("MEEP_GPU_MPI_TRANSPORT") == "pinned"
                    and environment.get("MEEP_GPU_MPI_COMPLETION") == "waitsome",
                    "pinned/waitsome environment mismatch")
        if lane == "cuda-aware-waitall":
            require(environment.get("MEEP_GPU_MPI_TRANSPORT") == "cuda-aware"
                    and environment.get("MEEP_GPU_MPI_COMPLETION") == "waitall",
                    "CUDA-aware/waitall environment mismatch")
        statistics = rank.get("statistics")
        require(type(statistics) is dict, "rank statistics are absent")
        startup_statistics = rank.get("startup_statistics")
        require(type(startup_statistics) is dict,
                "rank startup statistics are absent")
        opposite = "cuda" if expected_backend == "cpu" else "cpu"
        for phase in PHASES:
            require(phase_value(statistics, phase, opposite, "calls") == 0
                    and phase_value(statistics, phase, opposite, "points") == 0,
                    f"{lane} used {opposite} fallback in {phase}")
            for suffix in ("calls", "points"):
                active_totals[phase][suffix] += phase_value(
                    statistics, phase, expected_backend, suffix
                )
        runtime_stats = statistics.get("runtime", {})
        runtime_values = [
            runtime_stats.get(name)
            for name in (
                "runtime_availability_probes",
                "runtime_device_enumerations",
                "runtime_device_selections",
            )
        ]
        require(all(type(value) is int and value >= 0 for value in runtime_values),
                "invalid CUDA runtime statistics")
        multi = statistics.get("multi_gpu", {})
        for name in (
            "mpi_messages",
            "mpi_scalars",
            "cuda_aware_bytes",
            "pinned_staging_bytes",
            "pinned_device_to_host_bytes",
            "pinned_host_to_device_bytes",
        ):
            require(type(multi.get(name)) is int and multi[name] >= 0,
                    f"invalid multi-GPU counter {name}")
        completion = statistics.get("mpi_completion", {})
        require(
            type(completion.get("mpi_waitsome_executions")) is int
            and completion["mpi_waitsome_executions"] >= 0
            and type(completion.get("mpi_waitall_executions")) is int
            and completion["mpi_waitall_executions"] >= 0,
            "invalid MPI completion counters",
        )
        if expected_backend == "cpu":
            startup_runtime = startup_statistics.get("runtime", {})
            require(
                [startup_runtime.get(name) for name in (
                    "runtime_availability_probes",
                    "runtime_device_enumerations",
                    "runtime_device_selections",
                )] == [0, 0, 0],
                "CPU-hidden import touched the CUDA runtime",
            )
            require(runtime_values == [0, 0, 0],
                    "CPU-hidden lane touched the CUDA runtime")
            require(rank.get("selected_device") == -1
                    and rank.get("selected_device_identifier") == "",
                    "CPU-hidden lane retained a CUDA device")
            require(all(multi[name] == 0 for name in multi),
                    "CPU-hidden lane recorded GPU MPI transport")
            require(all(value == 0 for value in completion.values()),
                    "CPU-hidden lane recorded a GPU MPI completion branch")
        else:
            require(all(value > 0 for value in runtime_values),
                    "CUDA lane did not discover and select a device")
            identifier = rank.get("selected_device_identifier")
            require(type(identifier) is str and identifier,
                    "CUDA rank has no physical identifier")
            identifiers.append(normalize_uuid(identifier))
            require(type(rank.get("selected_device")) is int
                    and rank["selected_device"] >= 0,
                    "CUDA rank has no selected ordinal")
            ordinals.append(rank["selected_device"])
            require(multi["mpi_messages"] > 0 and multi["mpi_scalars"] > 0,
                    "CUDA rank recorded no MPI halo traffic")
            if lane == "cuda-pinned-waitsome":
                require(multi["pinned_staging_bytes"] > 0
                        and multi["pinned_device_to_host_bytes"] > 0
                        and multi["pinned_host_to_device_bytes"] > 0
                        and multi["cuda_aware_bytes"] == 0,
                        "pinned lane transport counters are inconsistent")
                require(completion["mpi_waitsome_executions"] > 0
                        and completion["mpi_waitall_executions"] == 0,
                        "pinned lane did not execute only MPI_Waitsome")
            else:
                require(multi["cuda_aware_bytes"] > 0
                        and multi["pinned_staging_bytes"] == 0
                        and multi["pinned_device_to_host_bytes"] == 0
                        and multi["pinned_host_to_device_bytes"] == 0,
                        "CUDA-aware lane transport counters are inconsistent")
                require(completion["mpi_waitall_executions"] > 0
                        and completion["mpi_waitsome_executions"] == 0,
                        "CUDA-aware lane did not execute only MPI_Waitall")
    require(len(set(pids)) == 2, "rank PIDs are duplicated")
    for phase in REQUIRED_PHASES:
        require(active_totals[phase]["calls"] > 0
                and active_totals[phase]["points"] > 0,
                f"{lane} did not execute {phase}")
    if expected_backend == "cuda":
        require(len(set(identifiers)) == 2, "CUDA ranks selected duplicate GPUs")
        require(sorted(ordinals) == [0, 1], "CUDA ranks did not select ordinals 0 and 1")
    return {
        "pass": True,
        "lane": lane,
        "pids": sorted(pids),
        "physical_affinity": [list(value) for value in sorted(affinities)],
        "device_identifiers": identifiers,
        "active_phase_calls": active_totals,
        "runtime": runtime_reference,
    }


def compare_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    validate_metrics(reference)
    validate_metrics(candidate)
    require(reference["timesteps"] == candidate["timesteps"]
            and reference["meep_time"] == candidate["meep_time"]
            and reference["shape"] == candidate["shape"],
            "CPU/CUDA workload identity differs")
    left = reference["field_values"]
    right = candidate["field_values"]
    errors = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    limits = [2e-5 + 5e-4 * max(abs(float(a)), abs(float(b)))
              for a, b in zip(left, right)]
    require(all(error <= limit for error, limit in zip(errors, limits)),
            "CPU/CUDA pointwise field tolerance failed")
    flux_errors = [
        abs(float(a) - float(b))
        for a, b in zip(reference["flux_values"], candidate["flux_values"])
    ]
    flux_limits = [
        1e-6 + 1e-4 * max(abs(float(a)), abs(float(b)))
        for a, b in zip(reference["flux_values"], candidate["flux_values"])
    ]
    require(all(error <= limit for error, limit in zip(flux_errors, flux_limits)),
            "CPU/CUDA flux tolerance failed")
    return {
        "pass": True,
        "field_count": len(left),
        "maximum_field_absolute_error": max(errors),
        "maximum_field_scaled_error": max(
            error / limit if limit else 0.0
            for error, limit in zip(errors, limits)
        ),
        "maximum_flux_absolute_error": max(flux_errors),
        "maximum_flux_scaled_error": max(
            error / limit if limit else 0.0
            for error, limit in zip(flux_errors, flux_limits)
        ),
    }


def parse_unittest_log(text: str) -> dict[str, Any]:
    identities = re.findall(
        r"test_divide_parallel_processes \((?:__main__|test_divide_mpi_processes)\."
        r"TestDivideParallelProcesses\.test_divide_parallel_processes\)",
        text,
    )
    require(len(identities) == 2, "divide MPI test identity count mismatch")
    require(len(re.findall(r"Ran 1 test in [0-9.]+s", text)) == 2,
            "divide MPI executed-test count mismatch")
    require(len(re.findall(r"^OK$", text, flags=re.MULTILINE)) == 2,
            "divide MPI OK count mismatch")
    require(not re.search(
        r"^(?:FAILED|ERROR)(?:\s|$)|\b(?:failures|errors|skipped)=\d+",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    ),
            "divide MPI test skipped or failed")
    require(text.count("Using MPI version") == 1,
            "divide MPI runtime banner mismatch")
    return {"pass": True, "rank_reports": 2, "tests_per_rank": 1, "skips": 0}


def load_divide_telemetry(
    text: str, telemetry_path: pathlib.Path
) -> dict[str, Any]:
    payloads = [
        load_json_text(line[len(DIVIDE_PREFIX):])
        for line in text.splitlines()
        if line.startswith(DIVIDE_PREFIX)
    ]
    require(len(payloads) == 1 and type(payloads[0]) is dict
            and set(payloads[0]) == {"schema_version", "sha256"}
            and payloads[0]["schema_version"] == 1
            and bool(re.fullmatch(r"[0-9a-f]{64}", payloads[0]["sha256"])),
            "divide MPI telemetry witness mismatch")
    require(telemetry_path.is_file(), "divide MPI telemetry file is absent")
    raw = telemetry_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == payloads[0]["sha256"],
            "divide MPI telemetry file hash mismatch")
    try:
        encoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError("divide MPI telemetry is not UTF-8") from error
    telemetry = load_json_text(encoded)
    require(type(telemetry) is dict and set(telemetry) == {"records"}
            and type(telemetry["records"]) is list
            and len(telemetry["records"]) == 2,
            "divide MPI telemetry record count mismatch")
    return {
        "records": telemetry["records"],
        "sha256": payloads[0]["sha256"],
    }


def validate_divide_telemetry(
    records: list[dict[str, Any]], *, backend: str, nonce: str
) -> dict[str, Any]:
    require(sorted(record.get("world_rank") for record in records) == [0, 1],
            "divide MPI telemetry ranks mismatch")
    identifiers: list[str] = []
    pids: list[int] = []
    for record in records:
        require(record.get("schema_version") == 1
                and record.get("nonce") == nonce,
                "divide MPI telemetry identity mismatch")
        require(record.get("expected_backend") == backend
                and record.get("active_backend") == backend
                and record.get("requested_backend") == backend,
                "divide MPI backend telemetry mismatch")
        require(type(record.get("pid")) is int and record["pid"] > 1,
                "divide MPI PID telemetry mismatch")
        pids.append(record["pid"])
        statistics = record.get("statistics")
        startup = record.get("startup_statistics")
        require(type(statistics) is dict and type(startup) is dict,
                "divide MPI statistics telemetry missing")
        opposite = "cuda" if backend == "cpu" else "cpu"
        active_calls = 0
        active_points = 0
        for phase in PHASES:
            require(phase_value(statistics, phase, opposite, "calls") == 0
                    and phase_value(statistics, phase, opposite, "points") == 0,
                    f"divide MPI used {opposite} fallback in {phase}")
            active_calls += phase_value(statistics, phase, backend, "calls")
            active_points += phase_value(statistics, phase, backend, "points")
        require(active_calls > 0 and active_points > 0,
                "divide MPI recorded no active backend work")
        runtime = statistics.get("runtime", {})
        runtime_values = [runtime.get(name) for name in (
            "runtime_availability_probes",
            "runtime_device_enumerations",
            "runtime_device_selections",
        )]
        require(all(type(value) is int and value >= 0 for value in runtime_values),
                "divide MPI runtime telemetry invalid")
        if backend == "cpu":
            startup_runtime = startup.get("runtime", {})
            require([startup_runtime.get(name) for name in (
                "runtime_availability_probes",
                "runtime_device_enumerations",
                "runtime_device_selections",
            )] == [0, 0, 0] and runtime_values == [0, 0, 0],
                    "divide CPU lane touched the CUDA runtime")
            require(record.get("selected_device") == -1
                    and record.get("selected_device_identifier") == "",
                    "divide CPU lane selected a CUDA device")
        else:
            require(all(value > 0 for value in runtime_values),
                    "divide CUDA lane did not initialize a CUDA device")
            require(type(record.get("selected_device")) is int
                    and record["selected_device"] == record["world_rank"],
                    "divide CUDA lane selected an invalid ordinal")
            identifier = record.get("selected_device_identifier")
            require(type(identifier) is str and identifier,
                    "divide CUDA lane has no device UUID")
            identifiers.append(normalize_uuid(identifier))
    require(len(set(pids)) == 2, "divide MPI rank PIDs are duplicated")
    if backend == "cuda":
        require(len(set(identifiers)) == 2,
                "divide CUDA ranks selected duplicate physical GPUs")
    return {
        "pass": True,
        "rank_reports": 2,
        "pids": sorted(pids),
        "device_identifiers": identifiers,
    }


def parse_inventory(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        columns = [item.strip() for item in line.split(",")]
        require(len(columns) == 4, "invalid NVIDIA inventory row")
        rows.append(
            {
                "index": int(columns[0]),
                "uuid": normalize_uuid(columns[1]),
                "name": columns[2],
                "driver_version": columns[3],
            }
        )
    require(rows, "NVIDIA inventory is empty")
    return rows


def parse_pmon(
    text: str,
    *,
    expected_pid_to_gpu: dict[int, int],
) -> dict[str, Any]:
    observed: dict[int, set[int]] = {}
    unrelated: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        columns = stripped.split()
        if len(columns) < 3 or columns[1] == "-":
            continue
        try:
            gpu = int(columns[0])
            pid = int(columns[1])
        except ValueError:
            unrelated.append(line)
            continue
        if pid not in expected_pid_to_gpu or expected_pid_to_gpu[pid] != gpu:
            unrelated.append(line)
            continue
        observed.setdefault(pid, set()).add(gpu)
    require(not unrelated, "NVIDIA process monitor observed an unrelated process")
    require(set(observed) == set(expected_pid_to_gpu),
            "NVIDIA process monitor missed a CUDA rank")
    require(all(values == {expected_pid_to_gpu[pid]} for pid, values in observed.items()),
            "CUDA rank moved between physical GPUs")
    return {
        "pass": True,
        "expected_pid_to_gpu": {
            str(pid): gpu for pid, gpu in sorted(expected_pid_to_gpu.items())
        },
        "observed_pids": sorted(observed),
        "unrelated_rows": unrelated,
    }


def validate_rank_device_bindings(
    samples: list[dict[str, Any]],
    requested_devices: list[int],
    inventory_by_index: dict[int, dict[str, Any]],
) -> dict[int, int]:
    expected_pid_to_gpu: dict[int, int] = {}
    all_pids: list[int] = []
    for sample in samples:
        ranks = sorted(
            sample["record"]["rank_records"], key=lambda item: item["rank"]
        )
        require(len(ranks) == len(requested_devices),
                "rank/device binding cardinality mismatch")
        for rank_record, physical_gpu in zip(ranks, requested_devices):
            rank = rank_record["rank"]
            require(rank_record["selected_device"] == rank,
                    "rank selected an unexpected visible CUDA ordinal")
            require(
                normalize_uuid(rank_record["selected_device_identifier"])
                == inventory_by_index[physical_gpu]["uuid"],
                "rank CUDA UUID does not match its physical NVIDIA ordinal",
            )
            pid = rank_record["pid"]
            all_pids.append(pid)
            expected_pid_to_gpu[pid] = physical_gpu
    require(len(set(all_pids)) == len(all_pids),
            "CUDA probe PID was reused across monitored lanes")
    return expected_pid_to_gpu


def validate_divide_device_bindings(
    divide_result: dict[str, Any],
    requested_devices: list[int],
    inventory_by_index: dict[int, dict[str, Any]],
) -> None:
    records = sorted(divide_result["telemetry"], key=lambda item: item["world_rank"])
    require(len(records) == len(requested_devices),
            "divide CUDA device binding cardinality mismatch")
    for record, physical_gpu in zip(records, requested_devices):
        require(
            normalize_uuid(record["selected_device_identifier"])
            == inventory_by_index[physical_gpu]["uuid"],
            "divide CUDA UUID does not match its physical NVIDIA ordinal",
        )


def run_logged(
    command: list[str],
    *,
    environment: dict[str, str],
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            env=environment,
            cwd=stdout_path.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        atomic_write_text(stdout_path, stdout)
        atomic_write_text(stderr_path, stderr)
        raise EvidenceError(f"command timed out after {timeout}s") from error
    atomic_write_text(stdout_path, completed.stdout)
    atomic_write_text(stderr_path, completed.stderr)
    atomic_write_text(
        stdout_path.parent / "command.txt",
        f"command={shlex.join(command)}\nreturncode={completed.returncode}\n"
        f"elapsed_seconds={time.perf_counter() - started:.9f}\n",
    )
    require(completed.returncode == 0,
            f"command failed with status {completed.returncode}: {shlex.join(command)}")
    return completed


def nvidia_output(arguments: list[str]) -> str:
    executable = pathlib.Path("/usr/bin/nvidia-smi")
    require(executable.is_file() and os.access(executable, os.X_OK),
            "trusted /usr/bin/nvidia-smi is unavailable")
    completed = subprocess.run(
        [str(executable), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=nvidia_environment(),
        timeout=30,
        check=False,
    )
    require(completed.returncode == 0,
            f"nvidia-smi failed: {completed.stderr.strip()}")
    return completed.stdout


def nvidia_environment() -> dict[str, str]:
    # Do not let controller-side preload or library search settings influence
    # the independent NVIDIA inventory/process witness.
    return {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}


def nvidia_smi_identity() -> dict[str, Any]:
    executable = pathlib.Path("/usr/bin/nvidia-smi").resolve(strict=True)
    return {
        "path": str(executable),
        "size_bytes": executable.stat().st_size,
        "sha256": sha256_file(executable),
    }


def gpu_snapshot() -> dict[str, Any]:
    inventory_text = nvidia_output(
        [
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader",
        ]
    )
    apps = nvidia_output(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    return {
        "captured_at_utc": utc_now(),
        "nvidia_smi": nvidia_smi_identity(),
        "inventory_raw": inventory_text,
        "inventory": parse_inventory(inventory_text),
        "compute_apps_raw": apps,
        "idle": not apps.strip(),
    }


def validate_stable_gpu_witness(
    before: dict[str, Any], after: dict[str, Any]
) -> None:
    require(after.get("inventory") == before.get("inventory"),
            "NVIDIA inventory changed during validation")
    require(after.get("nvidia_smi") == before.get("nvidia_smi"),
            "nvidia-smi executable identity changed during validation")


def require_private_directory(
    path: pathlib.Path, *, create: bool = False, allow_existing: bool = False
) -> pathlib.Path:
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError as error:
            if not allow_existing:
                raise EvidenceError(
                    f"private directory already exists: {path}"
                ) from error
    try:
        identity = path.lstat()
    except OSError as error:
        raise EvidenceError(f"private directory is absent: {path}") from error
    require(
        stat.S_ISDIR(identity.st_mode) and not stat.S_ISLNK(identity.st_mode),
        f"private directory is not a real directory: {path}",
    )
    require(
        identity.st_uid == os.getuid(),
        f"private directory is not owned by the current user: {path}",
    )
    os.chmod(path, 0o700, follow_symlinks=False)
    flags = (
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise EvidenceError(f"cannot securely open private directory: {path}") from error
    try:
        opened = os.fstat(descriptor)
        require(
            stat.S_ISDIR(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (identity.st_dev, identity.st_ino),
            f"private directory identity changed: {path}",
        )
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)
    return path


def copy_receipt_bound_file(
    record: dict[str, Any], repo: pathlib.Path, destination: pathlib.Path
) -> pathlib.Path:
    require(
        set(record) == {"path", "size_bytes", "sha256"}
        and isinstance(record["path"], str)
        and type(record["size_bytes"]) is int
        and record["size_bytes"] >= 0
        and isinstance(record["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
        "invalid receipt-bound file record",
    )
    source = pathlib.Path(record["path"])
    if not source.is_absolute():
        source = repo / source
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise EvidenceError(
            f"cannot securely open receipt-bound file: {source}: {error}"
        ) from error
    try:
        identity_before = os.fstat(descriptor)
        require(
            stat.S_ISREG(identity_before.st_mode),
            "receipt-bound path is not a regular file",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        identity_after = os.fstat(descriptor)
        require(
            (identity_before.st_dev, identity_before.st_ino,
             identity_before.st_size, identity_before.st_mtime_ns)
            == (identity_after.st_dev, identity_after.st_ino,
                identity_after.st_size, identity_after.st_mtime_ns),
            "receipt-bound file changed while it was read",
        )
    finally:
        os.close(descriptor)
    require(
        len(payload) == record["size_bytes"]
        and hashlib.sha256(payload).hexdigest() == record["sha256"],
        "receipt-bound file bytes do not match the build receipt",
    )

    require_private_directory(destination.parent)
    destination_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        output_descriptor = os.open(destination, destination_flags, 0o400)
    except OSError as error:
        raise EvidenceError(
            f"cannot create private receipt-bound copy: {destination}: {error}"
        ) from error
    try:
        os.fchmod(output_descriptor, 0o400)
        offset = 0
        while offset < len(payload):
            offset += os.write(output_descriptor, payload[offset:])
        os.fsync(output_descriptor)
    finally:
        os.close(output_descriptor)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    require(
        destination.stat().st_size == record["size_bytes"]
        and sha256_file(destination) == record["sha256"],
        "private receipt-bound copy failed verification",
    )
    return destination


def clean_environment(
    repo: pathlib.Path,
    output: pathlib.Path,
    receipt: dict[str, Any],
    lane: str,
    visible_devices: str,
    nonce: str,
    *,
    profile_lane: str | None = None,
) -> dict[str, str]:
    prefix = repo / ".envs" / "meep-gpu-cuda-mpi"
    build_dir = repo / receipt["build_dir"]
    fontconfig_record = receipt.get("configuration_files", {}).get(
        "qualification_fontconfig"
    )
    require(
        isinstance(fontconfig_record, dict)
        and isinstance(fontconfig_record.get("path"), str),
        "build receipt has no qualification Fontconfig record",
    )
    account_home = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir)
    runtime = output / lane / "runtime"
    profile = profile_lane or lane
    require_private_directory(output)
    require_private_directory(output / lane)
    require_private_directory(runtime, create=True, allow_existing=True)
    for directory in (runtime / "home", runtime / "cache", runtime / "config"):
        require_private_directory(directory, create=True, allow_existing=True)
    fontconfig_file = copy_receipt_bound_file(
        fontconfig_record,
        repo,
        runtime / "config" / "qualification-fontconfig.conf",
    )
    environment = {
        "HOME": str(runtime / "home"),
        "XDG_CACHE_HOME": str(runtime / "cache"),
        "XDG_CONFIG_HOME": str(runtime / "config"),
        "TMPDIR": str(runtime),
        "PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": f"{build_dir / 'src' / '.libs'}:{prefix / 'lib'}",
        "PYTHONPATH": str(build_dir / "python"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "CUDA_CACHE_DISABLE": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(runtime / "config" / "matplotlib"),
        "FONTCONFIG_FILE": str(fontconfig_file),
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MEEP_GPU_ALLOW_OVERSUBSCRIBE": "0",
        "CUDA_VISIBLE_DEVICES": visible_devices,
        "GPMEEP_MPI_PROBE_LANE": lane,
        "GPMEEP_MPI_PROBE_RECEIPT_ID": receipt["receipt_id"],
        "GPMEEP_MPI_PROBE_NONCE": nonce,
    }
    # Open MPI needs the real account home to find a valid launch identity,
    # while application caches remain redirected into the evidence tree.
    environment["HOME"] = str(account_home)
    recorded = receipt.get("configuration", {}).get("environment", {})
    for name in (
        "OMPI_MCA_mca_base_component_path",
        "OMPI_MCA_mca_base_param_files",
        "PMIX_MCA_mca_base_component_path",
        "PMIX_MCA_mca_base_param_files",
        "PRTE_MCA_mca_base_param_files",
    ):
        value = recorded.get(name)
        if isinstance(value, str) and value:
            environment[name] = value
    if profile == "cpu-hidden":
        environment.update(
            {
                "MEEP_GPU_BACKEND": "cpu",
                "MEEP_GPU_MPI_TRANSPORT": "pinned",
                "MEEP_GPU_MPI_COMPLETION": "waitsome",
            }
        )
    elif profile == "cuda-pinned-waitsome":
        environment.update(
            {
                "MEEP_GPU_BACKEND": "cuda",
                "MEEP_GPU_STRICT": "1",
                "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
                "MEEP_GPU_MPI_TRANSPORT": "pinned",
                "MEEP_GPU_MPI_COMPLETION": "waitsome",
            }
        )
    else:
        environment.update(
            {
                "MEEP_GPU_BACKEND": "cuda",
                "MEEP_GPU_STRICT": "1",
                "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
                "MEEP_GPU_MPI_TRANSPORT": "cuda-aware",
                "MEEP_GPU_MPI_COMPLETION": "waitall",
                "OMPI_MCA_opal_cuda_support": "true",
                "OMPI_MCA_btl": "self,smcuda,tcp",
                "UCX_MEMTYPE_CACHE": "n",
            }
        )
    return environment


def mpi_command(repo: pathlib.Path, script: pathlib.Path, *arguments: str) -> list[str]:
    prefix = repo / ".envs" / "meep-gpu-cuda-mpi" / "bin"
    return [
        str(prefix / "mpiexec"),
        "--bind-to",
        "core",
        "--map-by",
        "core",
        "-n",
        "2",
        str(prefix / "python"),
        str(script),
        *arguments,
    ]


def run_probe(
    repo: pathlib.Path,
    output: pathlib.Path,
    receipt: dict[str, Any],
    lane: str,
    visible_devices: str,
    timeout: int,
) -> dict[str, Any]:
    directory = output / lane
    require_private_directory(output)
    require_private_directory(directory, create=True)
    nonce = uuid.uuid4().hex
    environment = clean_environment(
        repo, output, receipt, lane, visible_devices, nonce
    )
    completed = run_logged(
        mpi_command(repo, repo / "scripts" / "mpi-python-validation-probe.py"),
        environment=environment,
        stdout_path=directory / "stdout.log",
        stderr_path=directory / "stderr.log",
        timeout=timeout,
    )
    record = extract_probe(completed.stdout)
    validation = validate_probe_record(
        record,
        lane=lane,
        receipt_id=receipt["receipt_id"],
        nonce=nonce,
        visible_devices=visible_devices,
    )
    validation["receipt_runtime_binding"] = validate_runtime_against_receipt(
        validation["runtime"], receipt, repo
    )
    atomic_write_json(directory / "result.json", record)
    return {
        "lane": lane,
        "nonce": nonce,
        "record": record,
        "validation": validation,
        "files": {
            name: file_record(directory / name, repo)
            for name in ("command.txt", "stdout.log", "stderr.log", "result.json")
        },
    }


def run_divide_test(
    repo: pathlib.Path,
    output: pathlib.Path,
    receipt: dict[str, Any],
    *,
    backend: str,
    visible_devices: str,
    timeout: int,
) -> dict[str, Any]:
    lane = f"divide-{backend}"
    directory = output / lane
    require_private_directory(output)
    require_private_directory(directory, create=True)
    nonce = uuid.uuid4().hex
    environment = clean_environment(
        repo, output, receipt,
        lane,
        visible_devices,
        nonce,
        profile_lane=(
            "cpu-hidden" if backend == "cpu" else "cuda-pinned-waitsome"
        ),
    )
    for name in (
        "GPMEEP_MPI_PROBE_LANE",
        "GPMEEP_MPI_PROBE_RECEIPT_ID",
        "GPMEEP_MPI_PROBE_NONCE",
    ):
        environment.pop(name, None)
    environment["GPMEEP_DIVIDE_VALIDATION_NONCE"] = nonce
    environment["GPMEEP_DIVIDE_VALIDATION_BACKEND"] = backend
    telemetry_path = directory / "telemetry.json"
    require(not telemetry_path.exists(), "divide telemetry target already exists")
    environment["GPMEEP_DIVIDE_TELEMETRY_PATH"] = str(telemetry_path.resolve())
    completed = run_logged(
        mpi_command(
            repo,
            repo / "python" / "tests" / "test_divide_mpi_processes.py",
            "-v",
        ),
        environment=environment,
        stdout_path=directory / "stdout.log",
        stderr_path=directory / "stderr.log",
        timeout=timeout,
    )
    combined = completed.stdout + "\n" + completed.stderr
    validation = parse_unittest_log(combined)
    telemetry_bundle = load_divide_telemetry(combined, telemetry_path)
    telemetry = telemetry_bundle["records"]
    telemetry_validation = validate_divide_telemetry(
        telemetry, backend=backend, nonce=nonce
    )
    result = {
        "backend": backend,
        "nonce": nonce,
        "validation": validation,
        "telemetry_validation": telemetry_validation,
        "telemetry": telemetry,
    }
    atomic_write_json(directory / "result.json", result)
    files = {
        name: file_record(directory / name, repo)
        for name in (
            "command.txt", "stdout.log", "stderr.log", "telemetry.json",
            "result.json",
        )
    }
    require(files["telemetry.json"]["sha256"] == telemetry_bundle["sha256"],
            "divide MPI telemetry changed after validation")
    return {
        **result,
        "files": files,
    }


def artifact_manifest(output: pathlib.Path, repo: pathlib.Path) -> dict[str, Any]:
    require_private_directory(output)
    entries = sorted(output.rglob("*"))
    for path in entries:
        require(not path.is_symlink(), f"evidence tree contains a symlink: {path}")
    paths = [
        path for path in entries
        if path.is_file() and path.name not in {"state.json", *FINAL_FILES}
    ]
    records = [file_record(path, repo) for path in paths]
    return {
        "schema_version": 1,
        "root": str(output),
        "file_count": len(records),
        "files": records,
    }


def verify_artifact_manifest(
    manifest: dict[str, Any],
    repo: pathlib.Path,
    *,
    expected_root: pathlib.Path,
) -> None:
    require(
        type(manifest) is dict
        and set(manifest) == {"schema_version", "root", "file_count", "files"}
        and manifest.get("schema_version") == 1
        and isinstance(manifest.get("root"), str),
        "artifact manifest schema mismatch",
    )
    root = pathlib.Path(manifest["root"])
    require(root.is_absolute(), "artifact manifest root is not absolute")
    require_private_directory(root)
    require_private_directory(expected_root)
    require(
        root.resolve(strict=True) == expected_root.resolve(strict=True),
        "artifact manifest root differs from the evidence directory",
    )
    records = manifest.get("files")
    require(type(records) is list and manifest.get("file_count") == len(records),
            "artifact manifest count mismatch")
    entries = sorted(root.rglob("*"))
    for path in entries:
        require(not path.is_symlink(), f"evidence tree contains a symlink: {path}")
    paths = [
        path for path in entries
        if path.is_file() and path.name not in {"state.json", *FINAL_FILES}
    ]
    current_records = [file_record(path, repo) for path in paths]
    require(current_records == records, "artifact manifest file set changed")
    for record in current_records:
        validate_file_record(record, "artifact", allow_empty=True)
        path = pathlib.Path(record["path"])
        if not path.is_absolute():
            path = repo / path
        require(path.is_file() and file_record(path, repo) == record,
                f"artifact changed: {record['path']}")


def record_path(record: dict[str, Any], repo: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(record["path"])
    return path if path.is_absolute() else repo / path


def verify_complete_publication(output: pathlib.Path, repo: pathlib.Path) -> None:
    require(
        not (output / "FAILED.json").exists(),
        "COMPLETE evidence contains contradictory FAILED.json",
    )
    complete = load_json_text((output / "COMPLETE").read_text(encoding="utf-8"))
    state = load_json_text((output / "state.json").read_text(encoding="utf-8"))
    require(type(complete) is dict and state == complete,
            "COMPLETE/state publication mismatch")
    require(set(complete) == {
        "schema_version", "state", "run_id", "build_receipt_id",
        "source_snapshot_sha256", "report", "report_markdown",
        "artifact_manifest",
    } and complete["schema_version"] == 1 and complete["state"] == "COMPLETE",
            "COMPLETE marker schema mismatch")
    for name in ("report", "report_markdown", "artifact_manifest"):
        validate_file_record(complete[name], f"COMPLETE {name}")
        path = record_path(complete[name], repo)
        require(path.is_file() and file_record(path, repo) == complete[name],
                f"COMPLETE {name} hash mismatch")
    manifest = load_json_text(
        record_path(complete["artifact_manifest"], repo).read_text(encoding="utf-8")
    )
    verify_artifact_manifest(manifest, repo, expected_root=output)
    report = load_json_text(
        record_path(complete["report"], repo).read_text(encoding="utf-8")
    )
    require(type(report) is dict and report.get("state") == "COMPLETE"
            and report.get("run_id") == complete["run_id"],
            "COMPLETE report identity mismatch")
    require(report.get("build_receipt", {}).get("receipt_id")
            == complete["build_receipt_id"],
            "COMPLETE receipt identity mismatch")
    require(report.get("source_snapshot", {}).get("sha256")
            == complete["source_snapshot_sha256"],
            "COMPLETE source identity mismatch")
    require(report.get("artifact_manifest") == complete["artifact_manifest"]
            and report.get("report_markdown") == complete["report_markdown"],
            "COMPLETE report closure mismatch")


def prepare_output(output: pathlib.Path, run_id: str) -> None:
    try:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError as error:
        raise EvidenceError(f"output directory already exists: {output}") from error
    require_private_directory(output)
    atomic_write_json(
        output / "state.json",
        {"schema_version": 1, "state": "RUNNING", "run_id": run_id},
    )


def mark_failed(output: pathlib.Path, run_id: str, error: BaseException) -> None:
    for name in ("report.json", "report.md", "artifacts.sha256.json", "COMPLETE"):
        (output / name).unlink(missing_ok=True)
    value = {
        "schema_version": 1,
        "state": "FAILED",
        "run_id": run_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }
    atomic_write_json(output / "FAILED.json", value)
    atomic_write_json(output / "state.json", value)


def execute(args: argparse.Namespace, repo: pathlib.Path, run_id: str) -> None:
    receipt_path = args.build_receipt.resolve()
    hash_cache = StatHashCache()
    snapshot_before = source_snapshot(repo, hash_cache=hash_cache)
    try:
        receipt = verify_build_receipt(receipt_path, repo, hash_cache=hash_cache)
    except ProvenanceError as error:
        raise EvidenceError(f"invalid build receipt: {error}") from error
    require(receipt.get("build_kind") == RECEIPT_KIND,
            "controller requires a CUDA+MPI+Python FP32 receipt")
    require(receipt.get("source_end") == snapshot_before,
            "build receipt does not match current source")
    copied_receipt = args.output / "build-provenance.json"
    atomic_write_text(copied_receipt, receipt_path.read_text(encoding="utf-8"))
    require(file_record(copied_receipt, repo)["sha256"] == sha256_file(receipt_path),
            "copied build receipt differs from its source")

    requested_devices = [int(value) for value in args.devices.split(",")]
    require(len(requested_devices) == 2 and len(set(requested_devices)) == 2
            and all(value >= 0 for value in requested_devices),
            "--devices must name two distinct nonnegative ordinals")
    visible_devices = ",".join(str(value) for value in requested_devices)
    pre = gpu_snapshot()
    require(pre["idle"], "GPU was not idle before MPI Python validation")
    inventory_by_index = {item["index"]: item for item in pre["inventory"]}
    require(all(index in inventory_by_index for index in requested_devices),
            "requested GPU ordinal is absent")
    atomic_write_json(args.output / "gpu-pre.json", pre)

    cpu = run_probe(
        repo, args.output, receipt, "cpu-hidden", "", args.timeout
    )

    pmon_path = args.output / "gpu-pmon.log"
    with pmon_path.open("wb") as pmon_stream:
        monitor = subprocess.Popen(
            ["/usr/bin/nvidia-smi", "pmon", "-s", "um", "-d", "1"],
            stdout=pmon_stream,
            stderr=subprocess.STDOUT,
            env=nvidia_environment(),
        )
        try:
            pinned = run_probe(
                repo,
                args.output,
                receipt,
                "cuda-pinned-waitsome",
                visible_devices,
                args.timeout,
            )
            aware = run_probe(
                repo,
                args.output,
                receipt,
                "cuda-aware-waitall",
                visible_devices,
                args.timeout,
            )
        finally:
            if monitor.poll() is None:
                monitor.send_signal(signal.SIGTERM)
                try:
                    monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait(timeout=5)
            pmon_stream.flush()
            os.fsync(pmon_stream.fileno())
    require(monitor.returncode in (0, -signal.SIGTERM),
            "NVIDIA process monitor failed")

    expected_identifiers = {
        inventory_by_index[index]["uuid"] for index in requested_devices
    }
    expected_pid_to_gpu = validate_rank_device_bindings(
        [pinned, aware], requested_devices, inventory_by_index
    )
    monitor_validation = parse_pmon(
        pmon_path.read_text(encoding="utf-8", errors="replace"),
        expected_pid_to_gpu=expected_pid_to_gpu,
    )
    atomic_write_json(
        args.output / "gpu-monitor.json",
        {
            **monitor_validation,
            "returncode": monitor.returncode,
            "raw_sha256": sha256_file(pmon_path),
        },
    )

    divide_cpu = run_divide_test(
        repo, args.output, receipt,
        backend="cpu", visible_devices="", timeout=args.timeout,
    )
    divide_cuda = run_divide_test(
        repo, args.output, receipt,
        backend="cuda", visible_devices=visible_devices, timeout=args.timeout,
    )
    validate_divide_device_bindings(
        divide_cuda, requested_devices, inventory_by_index
    )
    post = gpu_snapshot()
    require(post["idle"], "GPU was not idle after MPI Python validation")
    validate_stable_gpu_witness(pre, post)
    atomic_write_json(args.output / "gpu-post.json", post)

    comparisons = {
        "cpu_vs_cuda_pinned": compare_metrics(
            cpu["record"]["metrics"], pinned["record"]["metrics"]
        ),
        "cpu_vs_cuda_aware": compare_metrics(
            cpu["record"]["metrics"], aware["record"]["metrics"]
        ),
        "cuda_pinned_vs_cuda_aware": compare_metrics(
            pinned["record"]["metrics"], aware["record"]["metrics"]
        ),
    }

    end_receipt = verify_build_receipt(
        receipt_path, repo, hash_cache=StatHashCache()
    )
    snapshot_after = source_snapshot(repo, hash_cache=StatHashCache())
    require(end_receipt == receipt and snapshot_after == snapshot_before,
            "receipt or source changed during validation")
    manifest = artifact_manifest(args.output, repo)
    atomic_write_json(args.output / "artifacts.sha256.json", manifest)
    verify_artifact_manifest(manifest, repo, expected_root=args.output)
    manifest_record = file_record(args.output / "artifacts.sha256.json", repo)
    producer_records = {
        name: file_record(repo / "scripts" / name, repo)
        for name in (
            "run-mpi-python-validation.py",
            "mpi-python-validation-probe.py",
        )
    }
    gates = {
        "receipt_and_source_closure": True,
        "cpu_hidden_no_cuda_runtime": True,
        "two_distinct_physical_gpus": True,
        "strict_cuda_no_cpu_fallback": True,
        "pinned_waitsome_transport": True,
        "cuda_aware_waitall_transport": True,
        "pointwise_fp32_numerical_agreement": True,
        "divide_parallel_processes_cpu": True,
        "divide_parallel_processes_cuda": True,
        "exclusive_gpu_process_monitor": True,
    }
    report = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "build_receipt": {
            "receipt_id": receipt["receipt_id"],
            "original": file_record(receipt_path, repo),
            "copied": file_record(copied_receipt, repo),
            "build_input_id": receipt.get("build_input_id"),
            "artifact_set_id": receipt.get("artifact_set_id"),
        },
        "source_snapshot": snapshot_after,
        "producers": producer_records,
        "devices": {
            "requested_ordinals": requested_devices,
            "identifiers": sorted(expected_identifiers),
            "pre": file_record(args.output / "gpu-pre.json", repo),
            "monitor": file_record(args.output / "gpu-monitor.json", repo),
            "post": file_record(args.output / "gpu-post.json", repo),
        },
        "lanes": {
            sample["lane"]: {
                "nonce": sample["nonce"],
                "validation": sample["validation"],
                "files": sample["files"],
            }
            for sample in (cpu, pinned, aware)
        },
        "divide_parallel_processes": {
            "cpu": divide_cpu,
            "cuda": divide_cuda,
        },
        "comparisons": comparisons,
        "gate": {**gates, "pass": all(gates.values())},
        "performance_evidence": {
            "valid_for_speed_gate": False,
            "reason": "single correctness samples with monitor hold time and tiny domains",
        },
        "artifact_manifest": manifest_record,
    }
    require(report["gate"]["pass"] is True,
            "refusing to publish failed MPI Python gates")
    markdown = (
        "# gpmeep MPI Python validation\n\n"
        f"- Receipt: `{receipt['receipt_id']}`\n"
        "- Matrix: CPU hidden; CUDA pinned/waitsome; CUDA-aware/waitall\n"
        "- MPI ranks: 2\n"
        "- Distinct physical GPUs: 2\n"
        "- Pointwise field values compared: 6400 per lane pair\n"
        "- divide_parallel_processes: CPU and CUDA PASS\n"
        "- Performance evidence: no (correctness-only)\n"
    )
    atomic_write_text(args.output / "report.md", markdown)
    report["report_markdown"] = file_record(args.output / "report.md", repo)
    atomic_write_json(args.output / "report.json", report)
    # Re-hash the full raw closure immediately before publishing authority.
    verify_artifact_manifest(manifest, repo, expected_root=args.output)
    final_receipt = verify_build_receipt(receipt_path, repo)
    require(final_receipt == receipt and source_snapshot(repo) == snapshot_before,
            "final receipt/source reverification failed")
    require(
        not (args.output / "FAILED.json").exists(),
        "refusing to publish COMPLETE beside FAILED.json",
    )
    marker = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "build_receipt_id": receipt["receipt_id"],
        "source_snapshot_sha256": snapshot_before["sha256"],
        "report": file_record(args.output / "report.json", repo),
        "report_markdown": file_record(args.output / "report.md", repo),
        "artifact_manifest": manifest_record,
    }
    atomic_write_json(args.output / "COMPLETE", marker)
    atomic_write_json(args.output / "state.json", marker)
    verify_complete_publication(args.output, repo)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--build-receipt", type=pathlib.Path, required=True)
    result.add_argument("--output", type=pathlib.Path, required=True)
    result.add_argument("--devices", default="0,1")
    result.add_argument("--timeout", type=int, default=600)
    return result


def main() -> int:
    args = parser().parse_args()
    args.output = args.output.resolve()
    args.build_receipt = args.build_receipt.resolve()
    run_id = f"mpi-python-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex}"
    repo = pathlib.Path(__file__).resolve().parent.parent
    lock_path = pathlib.Path("/tmp/gpmeep-mpi-python-validation.lock")
    with lock_path.open("a+") as lock:
        prepared = False
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("error: another MPI Python validation holds the GPU lock", file=sys.stderr)
            return 2
        try:
            require(args.timeout > 0, "--timeout must be positive")
            try:
                args.output.relative_to(repo)
            except ValueError:
                pass
            else:
                raise EvidenceError("authoritative output must be outside the source tree")
            prepare_output(args.output, run_id)
            prepared = True
            execute(args, repo, run_id)
        except BaseException as error:
            # Never invalidate or overwrite an existing evidence directory
            # merely because prepare_output rejected it.  Only this run's
            # freshly created directory is eligible for FAILED publication.
            if prepared:
                mark_failed(args.output, run_id, error)
            print(f"error: {error}", file=sys.stderr)
            return 1
    print(f"MPI Python validation COMPLETE: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
