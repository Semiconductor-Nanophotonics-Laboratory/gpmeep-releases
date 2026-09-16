#!/usr/bin/env python3
"""Validators for receipt-bound MPI adjoint benchmark records.

The legacy standalone CLI is deliberately unable to publish COMPLETE because
arbitrary pre-existing inputs cannot prove launch provenance or repetition.
Use ``run-mpi-adjoint-benchmark.py`` for the fixed release qualification.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import platform
import re
import stat
import sys
import traceback
import uuid
from typing import Any

from gpmeep_benchmark_evidence import (
    atomic_write_json,
    atomic_write_text,
    canonical_float64_sha256,
    compare_gradient_vectors,
    sha256_file,
    validate_gradient_record,
)
from gpmeep_provenance import (
    StatHashCache,
    canonical_sha256,
    git_output,
    source_snapshot,
    tree_manifest,
    verify_build_receipt,
)


FINAL_ARTIFACTS = ("report.json", "report.md", "COMPLETE", "FAILED.json")
POSTPROCESSOR_SOURCES = (
    "scripts/benchmark-adjoint.py",
    "scripts/compare-mpi-adjoint-benchmarks.py",
    "scripts/gpmeep_benchmark_evidence.py",
    "scripts/gpmeep_provenance.py",
)
TIMING_FIELDS = (
    "update_design_seconds",
    "forward_seconds",
    "adjoint_seconds",
    "gradient_seconds",
)
QUALIFICATION_PROFILE = "m19-cpu-gpu-adjoint-release-v4"
PHASE_DFT_CALL_LIMITS = {"forward": 200, "adjoint": 256}
PHASE_CUDA_COUNTERS = ("curl", "update_eh", "source", "boundary", "dft")
PHASE_COUNTER_STATISTICS = {
    "curl": ("dispatch", "curl"),
    "update_eh": ("field_updates", "update_eh"),
    "source": ("sources", "source"),
    "boundary": ("boundaries", "boundary"),
    "dft": ("dfts", "dft"),
    "polarization": ("polarizations", "polarization"),
}
PHASE_REQUIRED_PER_RANK_CUDA_COUNTERS = (
    "curl",
    "update_eh",
    "boundary",
    "dft",
)
CUDA_COUNTERS = (
    ("dispatch", "cuda_curl_calls"),
    ("field_updates", "cuda_update_eh_calls"),
    ("sources", "cuda_source_calls"),
    ("boundaries", "cuda_boundary_calls"),
    ("dfts", "cuda_dft_calls"),
)
CPU_COUNTERS = (
    ("dispatch", "cpu_curl_calls"),
    ("field_updates", "cpu_update_eh_calls"),
    ("polarizations", "cpu_polarization_calls"),
    ("sources", "cpu_source_calls"),
    ("boundaries", "cpu_boundary_calls"),
    ("dfts", "cpu_dft_calls"),
)
WORKLOAD_KEYS = (
    "resolution",
    "run_time",
    "cell_size",
    "design_resolution",
    "design_shape",
    "design_variables",
    "yee_cells",
    "weights_sha256",
    "direction_sha256",
    "material_gradient_path",
)
DEVICE_IDENTIFIER_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
TRUSTED_CUDA_DRIVER_ROOTS = (
    pathlib.Path("/usr/lib"),
    pathlib.Path("/usr/lib64"),
    pathlib.Path("/lib"),
    pathlib.Path("/lib64"),
    pathlib.Path("/usr/local/nvidia"),
    pathlib.Path("/run/nvidia"),
    pathlib.Path("/opt/nvidia"),
    pathlib.Path("/usr/lib/wsl"),
)
TRUSTED_NVIDIA_SMI_ROOTS = (
    pathlib.Path("/usr/bin"),
    pathlib.Path("/usr/sbin"),
    pathlib.Path("/bin"),
    pathlib.Path("/sbin"),
    pathlib.Path("/usr/local/cuda/bin"),
    pathlib.Path("/usr/lib/wsl/lib"),
)
RANK_ENVIRONMENT_SEMANTIC_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "GPMEEP_VALIDATION_STRICT_CUDA",
    "MEEP_GPU_BACKEND",
    "MEEP_GPU_MPI_COMPLETION",
    "MEEP_GPU_MPI_TRANSPORT",
    "MEEP_GPU_STRICT",
    "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS",
    "MEEP_MATERIAL_GRADIENT_PATH",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "OMPI_COMM_WORLD_NODE_RANK",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_MCA_mca_base_component_path",
    "OMPI_MCA_mca_base_param_files",
    "PMIX_HOSTNAME",
    "PMIX_MCA_mca_base_component_path",
    "PMIX_MCA_mca_base_param_files",
    "PMIX_RANK",
    "PRTE_MCA_mca_base_param_files",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-report", type=pathlib.Path, required=True)
    parser.add_argument("--multi-report", type=pathlib.Path, required=True)
    parser.add_argument("--build-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expected-multi-ranks", type=int, default=2)
    parser.add_argument(
        "--expected-material-gradient-path",
        choices=("legacy", "auto", "analytic-required"),
        default="analytic-required",
    )
    parser.add_argument("--objective-atol", type=float, default=1e-8)
    parser.add_argument("--gradient-atol", type=float, default=2e-7)
    parser.add_argument("--gradient-rtol", type=float, default=5e-5)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    parser.add_argument("--maximum-dft-call-multiplier", type=float, default=2.0)
    return parser.parse_args(argv)


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise RuntimeError(f"{label} must be finite and positive")
    return converted


def _load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _validate_runtime_record(
    record: Any,
    label: str,
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, str]:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} runtime record is absent")
    path_value = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"{label} runtime path is absent")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"{label} runtime SHA-256 is invalid")
    path = pathlib.Path(path_value).resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} runtime file is absent: {path}")
    current_digest = (
        hash_cache.digest(path) if hash_cache is not None else sha256_file(path)
    )
    if current_digest != digest:
        raise RuntimeError(f"{label} runtime file changed after measurement")
    return {"path": str(path), "sha256": digest}


def _validate_package_record(
    record: Any,
    label: str,
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} Python-package manifest is absent")
    root_value = record.get("root")
    count = record.get("file_count")
    digest = record.get("sha256")
    if not isinstance(root_value, str) or not root_value:
        raise RuntimeError(f"{label} Python-package root is absent")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RuntimeError(f"{label} Python-package file count is invalid")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"{label} Python-package SHA-256 is invalid")
    root = pathlib.Path(root_value).resolve()
    current = tree_manifest(root, root.parent, hash_cache=hash_cache)
    if current["file_count"] != count or current["sha256"] != digest:
        raise RuntimeError(f"{label} Python package changed after measurement")
    return {"root": str(root), "file_count": count, "sha256": digest}


def validate_runtime_closure(
    result: dict[str, Any],
    label: str,
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    if hash_cache is None:
        hash_cache = StatHashCache()
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError(f"{label} runtime closure is absent")
    closure = {
        name: _validate_runtime_record(
            runtime.get(name), f"{label}.{name}", hash_cache=hash_cache
        )
        for name in ("python_executable", "meep_module", "extension", "libmeep")
    }
    closure["python_package"] = _validate_package_record(
        runtime.get("python_package"),
        f"{label}.python_package",
        hash_cache=hash_cache,
    )
    loaded = runtime.get("loaded_libraries")
    required_loaded = {
        "mpi",
        "cuda_driver",
        "cuda_runtime",
        "hdf5",
        "environment",
    }
    if not isinstance(loaded, dict) or set(loaded) != required_loaded:
        raise RuntimeError(f"{label} loaded-library closure is incomplete")
    validated_loaded: dict[str, list[dict[str, str]]] = {}
    for group in sorted(required_loaded):
        records = loaded.get(group)
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"{label}.{group} loaded-library evidence is absent")
        validated_loaded[group] = []
        seen_paths: set[str] = set()
        for index, record in enumerate(records):
            validated = _validate_runtime_record(
                record,
                f"{label}.loaded_libraries.{group}[{index}]",
                hash_cache=hash_cache,
            )
            path = pathlib.Path(validated["path"])
            size = record.get("size_bytes") if isinstance(record, dict) else None
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or path.stat().st_size != size
                or validated["path"] in seen_paths
            ):
                raise RuntimeError(
                    f"{label}.{group} loaded-library record is invalid or duplicated"
                )
            seen_paths.add(validated["path"])
            validated_loaded[group].append(validated)
    environment_records = {
        (record["path"], record["sha256"])
        for record in validated_loaded["environment"]
    }
    for group in ("mpi", "cuda_runtime", "hdf5"):
        required_records = {
            (record["path"], record["sha256"])
            for record in validated_loaded[group]
        }
        if not required_records.issubset(environment_records):
            raise RuntimeError(
                f"{label}.{group} records are absent from the complete mapped-environment set"
            )
    closure["loaded_libraries"] = validated_loaded
    return closure


def runtime_environment_independent_projection(runtime: dict[str, Any]) -> dict[str, Any]:
    """Return identities that must match across heterogeneous MPI ranks.

    The complete mapped environment and the system CUDA driver can legitimately
    differ by rank because transport plugins are loaded lazily and nodes may
    expose different protected driver paths.  Every such record is validated
    separately against the receipt or trusted system roots.
    """

    projection = {key: value for key, value in runtime.items() if key != "loaded_libraries"}
    loaded = runtime.get("loaded_libraries", {})
    projection["loaded_libraries"] = {
        key: value
        for key, value in loaded.items()
        if key not in {"environment", "cuda_driver"}
    }
    return projection


def validate_rank_nvidia_driver(
    evidence: Any, selected_identifier: str, label: str
) -> dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != {
        "hostname",
        "executable",
        "inventory",
        "topology",
    }:
        raise RuntimeError(f"{label} rank-local NVIDIA driver evidence is absent")
    if not isinstance(evidence.get("hostname"), str) or not evidence["hostname"]:
        raise RuntimeError(f"{label} rank hostname is absent")
    executable = evidence.get("executable")
    if not isinstance(executable, dict) or set(executable) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise RuntimeError(f"{label} nvidia-smi executable record is invalid")
    executable_path = pathlib.Path(str(executable.get("path", ""))).resolve()
    protected_roots = [
        root.resolve() for root in TRUSTED_NVIDIA_SMI_ROOTS if root.is_dir()
    ]
    if (
        not executable_path.is_file()
        or not any(
            executable_path == root or root in executable_path.parents
            for root in protected_roots
        )
        or executable_path.stat().st_size != executable.get("size_bytes")
        or sha256_file(executable_path) != executable.get("sha256")
        or any(
            path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            for path in (executable_path.parent, executable_path)
        )
    ):
        raise RuntimeError(f"{label} nvidia-smi path/hash is not protected")

    identifiers: set[str] = set()
    inventory = evidence.get("inventory")
    topology = evidence.get("topology")
    for probe_name, probe in (("inventory", inventory), ("topology", topology)):
        if (
            not isinstance(probe, dict)
            or probe.get("exit_code") != 0
            or not isinstance(probe.get("stdout"), str)
            or not isinstance(probe.get("command"), list)
            or not probe["command"]
            or pathlib.Path(str(probe["command"][0])).resolve() != executable_path
        ):
            raise RuntimeError(f"{label} NVIDIA {probe_name} probe failed")
    for line in inventory["stdout"].splitlines():
        columns = [value.strip() for value in line.split(",")]
        if len(columns) != 6:
            raise RuntimeError(f"{label} NVIDIA inventory row is malformed")
        normalized = re.sub(
            r"[^0-9a-f]", "", columns[1].lower().removeprefix("gpu-")
        )
        if DEVICE_IDENTIFIER_PATTERN.fullmatch(normalized) is None:
            raise RuntimeError(f"{label} NVIDIA inventory UUID is unsupported")
        identifiers.add(normalized)
    for raw_uuid in re.findall(
        r"UUID:\s*((?:GPU|MIG)-[0-9A-Fa-f-]+)", topology["stdout"]
    ):
        normalized = re.sub(r"[^0-9a-f]", "", raw_uuid.lower().split("-", 1)[1])
        if DEVICE_IDENTIFIER_PATTERN.fullmatch(normalized):
            identifiers.add(normalized)
    if selected_identifier not in identifiers:
        raise RuntimeError(
            f"{label} CUDA runtime UUID is absent from its rank-local driver inventory"
        )
    return {
        "hostname": evidence["hostname"],
        "executable": dict(executable),
        "inventory_identifiers": sorted(identifiers),
        "selected_identifier": selected_identifier,
    }


def _counter(statistics: Any, group: str, field: str, label: str) -> int:
    value = statistics.get(group, {}).get(field) if isinstance(statistics, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} counter {group}.{field} is absent or invalid")
    return value


def validate_gradient_summaries(result: dict[str, Any], label: str) -> None:
    gradient = validate_gradient_record(result)
    payload = result["result"]
    l2 = math.sqrt(math.fsum(value * value for value in gradient))
    total = math.fsum(gradient)
    raw_direction = [math.sin(index * 0.61803398875) for index in range(len(gradient))]
    direction_norm = math.sqrt(math.fsum(value * value for value in raw_direction))
    if not math.isfinite(direction_norm) or direction_norm <= 0:
        raise RuntimeError(f"{label} deterministic gradient direction is invalid")
    projection = math.fsum(
        value * direction_value / direction_norm
        for value, direction_value in zip(gradient, raw_direction)
    )
    expected = {
        "gradient_l2": l2,
        "gradient_sum": total,
        "gradient_projection": projection,
    }
    for name, recomputed in expected.items():
        recorded = payload.get(name)
        if (
            isinstance(recorded, bool)
            or not isinstance(recorded, (int, float))
            or not math.isfinite(float(recorded))
            or not math.isclose(
                float(recorded), recomputed, rel_tol=2.0e-13, abs_tol=2.0e-13
            )
        ):
            raise RuntimeError(f"{label} {name} summary is stale or invalid")


def validate_rank_cuda_coverage(
    rank: dict[str, Any], *, world_size: int, label: str
) -> dict[str, Any]:
    backend = rank.get("backend")
    if not isinstance(backend, dict):
        raise RuntimeError(f"{label} per-rank backend evidence is absent")
    expected_backend = (
        backend.get("requested") == "cuda"
        and backend.get("active") == "cuda"
        and rank.get("active_backend") == "cuda"
        and backend.get("compiled") is True
        and backend.get("runtime_available") is True
        and backend.get("single_precision") is True
        and backend.get("strict_cuda_marker") is True
    )
    if not expected_backend:
        raise RuntimeError(f"{label} did not use strict FP32 CUDA")
    process_environment = rank.get("process_environment")
    if not isinstance(process_environment, dict) or not process_environment or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in process_environment.items()
    ):
        raise RuntimeError(f"{label} actual process environment is absent")
    world_rank = rank.get("world_rank")
    required_rank_environment = {
        "OMPI_COMM_WORLD_RANK": str(world_rank),
        "OMPI_COMM_WORLD_SIZE": str(world_size),
        "PMIX_RANK": str(world_rank),
    }
    if any(
        process_environment.get(name) != value
        for name, value in required_rank_environment.items()
    ):
        raise RuntimeError(f"{label} MPI rank environment is absent or inconsistent")
    semantic_process_environment = {
        name: process_environment[name]
        for name in RANK_ENVIRONMENT_SEMANTIC_KEYS
        if name in process_environment
    }

    selected_ordinal = backend.get("selected_device")
    if (
        isinstance(selected_ordinal, bool)
        or not isinstance(selected_ordinal, int)
        or selected_ordinal < 0
        or rank.get("selected_device") != selected_ordinal
    ):
        raise RuntimeError(f"{label} selected CUDA ordinal is invalid or inconsistent")
    inventory = backend.get("devices")
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError(f"{label} CUDA device inventory is absent")
    inventory_by_ordinal: dict[int, dict[str, Any]] = {}
    inventory_identifiers: set[str] = set()
    for index, device in enumerate(inventory):
        if not isinstance(device, dict):
            raise RuntimeError(f"{label} CUDA inventory entry {index} is invalid")
        ordinal = device.get("ordinal")
        capability = device.get("compute_capability")
        identifier_value = device.get("identifier")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            or ordinal in inventory_by_ordinal
            or not isinstance(capability, (list, tuple))
            or len(capability) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in capability
            )
            or not isinstance(device.get("compatible"), bool)
            or isinstance(device.get("multiprocessor_count"), bool)
            or not isinstance(device.get("multiprocessor_count"), int)
            or device["multiprocessor_count"] <= 0
            or isinstance(device.get("max_threads_per_block"), bool)
            or not isinstance(device.get("max_threads_per_block"), int)
            or device["max_threads_per_block"] <= 0
            or isinstance(device.get("global_memory_bytes"), bool)
            or not isinstance(device.get("global_memory_bytes"), int)
            or device["global_memory_bytes"] <= 0
            or not isinstance(device.get("name"), str)
            or not device["name"]
            or not isinstance(identifier_value, str)
            or DEVICE_IDENTIFIER_PATTERN.fullmatch(identifier_value) is None
            or identifier_value in inventory_identifiers
        ):
            raise RuntimeError(f"{label} CUDA inventory entry {index} is invalid")
        inventory_by_ordinal[ordinal] = device
        inventory_identifiers.add(identifier_value)
    if selected_ordinal not in inventory_by_ordinal:
        raise RuntimeError(f"{label} selected CUDA ordinal is absent from inventory")
    if inventory_by_ordinal[selected_ordinal]["compatible"] is not True:
        raise RuntimeError(f"{label} selected CUDA device is not build-compatible")

    statistics = rank.get("final_statistics")
    cpu_calls = sum(
        _counter(statistics, group, field, label) for group, field in CPU_COUNTERS
    )
    cuda_calls = {
        field: _counter(statistics, group, field, label)
        for group, field in CUDA_COUNTERS
    }
    if cpu_calls != 0:
        raise RuntimeError(f"{label} recorded {cpu_calls} CPU fallback calls")
    missing_cuda = [name for name, value in cuda_calls.items() if value <= 0]
    if missing_cuda:
        raise RuntimeError(f"{label} has no CUDA work for {', '.join(missing_cuda)}")

    mpi_messages = _counter(statistics, "multi_gpu", "mpi_messages", label)
    mpi_scalars = _counter(statistics, "multi_gpu", "mpi_scalars", label)
    cuda_aware = _counter(statistics, "multi_gpu", "cuda_aware_bytes", label)
    pinned = _counter(statistics, "multi_gpu", "pinned_staging_bytes", label)
    pinned_d2h = _counter(
        statistics, "multi_gpu", "pinned_device_to_host_bytes", label
    )
    pinned_h2d = _counter(
        statistics, "multi_gpu", "pinned_host_to_device_bytes", label
    )
    transport = "none"
    if world_size > 1:
        if (
            mpi_messages <= 0
            or mpi_scalars <= 0
            or not (bool(cuda_aware) ^ bool(pinned))
            or cuda_aware + pinned != mpi_scalars * 4
            or (pinned > 0 and (pinned_d2h <= 0 or pinned_h2d <= 0))
            or pinned_d2h + pinned_h2d != pinned
        ):
            raise RuntimeError(f"{label} has no measured MPI GPU-halo traffic")
        transport = "cuda-aware" if cuda_aware else "pinned-staging"
    elif any(
        value != 0
        for value in (
            mpi_messages,
            mpi_scalars,
            cuda_aware,
            pinned,
            pinned_d2h,
            pinned_h2d,
        )
    ):
        raise RuntimeError(f"{label} singleton run unexpectedly recorded MPI traffic")

    phases = rank.get("phases")
    if not isinstance(phases, dict):
        raise RuntimeError(f"{label} rank-local phase evidence is absent")
    phase_coverage: dict[str, Any] = {}
    phase_statistics_records: list[dict[str, Any]] = []
    for phase_name in ("forward", "adjoint"):
        phase = phases.get(phase_name)
        if not isinstance(phase, dict):
            raise RuntimeError(f"{label}.{phase_name} phase evidence is absent")
        counters = phase.get("backend_counters")
        if not isinstance(counters, dict):
            raise RuntimeError(f"{label}.{phase_name} counters are absent")
        cpu = counters.get("cpu")
        cuda = counters.get("cuda")
        if not isinstance(cpu, dict) or not isinstance(cuda, dict):
            raise RuntimeError(f"{label}.{phase_name} backend counters are absent")
        expected_phase_counters = set(PHASE_CUDA_COUNTERS) | {"polarization"}
        if set(cpu) != expected_phase_counters or set(cuda) != expected_phase_counters:
            raise RuntimeError(f"{label}.{phase_name} counter schema is incomplete")
        statistics_delta = phase.get("statistics_delta")
        if not isinstance(statistics_delta, dict):
            raise RuntimeError(f"{label}.{phase_name} statistics delta is absent")
        phase_statistics_records.append(statistics_delta)
        for counter_name, values in cpu.items():
            if (
                not isinstance(values, dict)
                or set(values) != {"calls", "points"}
                or values.get("calls") != 0
                or values.get("points") != 0
            ):
                raise RuntimeError(
                    f"{label}.{phase_name} recorded CPU {counter_name} fallback"
                )
        measured: dict[str, dict[str, int]] = {}
        for counter_name in expected_phase_counters:
            values = cuda.get(counter_name)
            if not isinstance(values, dict) or set(values) != {"calls", "points"}:
                raise RuntimeError(
                    f"{label}.{phase_name} CUDA {counter_name} evidence is absent"
                )
            calls = values.get("calls")
            points = values.get("points")
            if (
                isinstance(calls, bool)
                or not isinstance(calls, int)
                or isinstance(points, bool)
                or not isinstance(points, int)
                or calls < 0
                or points < 0
            ):
                raise RuntimeError(
                    f"{label}.{phase_name} CUDA {counter_name} calls/points are invalid"
                )
            group, stem = PHASE_COUNTER_STATISTICS[counter_name]
            statistics_group = statistics_delta.get(group)
            if not isinstance(statistics_group, dict) or any(
                statistics_group.get(f"{backend_name}_{stem}_{field}")
                != backend_values[field]
                for backend_name, backend_values in (("cpu", cpu[counter_name]), ("cuda", values))
                for field in ("calls", "points")
            ):
                raise RuntimeError(
                    f"{label}.{phase_name} {counter_name} backend view/statistics disagree"
                )
            if counter_name in PHASE_CUDA_COUNTERS:
                measured[counter_name] = {"calls": calls, "points": points}
        for counter_name in PHASE_REQUIRED_PER_RANK_CUDA_COUNTERS:
            if (
                measured[counter_name]["calls"] <= 0
                or measured[counter_name]["points"] <= 0
            ):
                raise RuntimeError(
                    f"{label}.{phase_name} CUDA {counter_name} calls/points are absent"
                )
        dft_limit = PHASE_DFT_CALL_LIMITS[phase_name]
        if measured["dft"]["calls"] > dft_limit:
            raise RuntimeError(
                f"{label}.{phase_name} DFT calls exceed fixed limit {dft_limit}"
            )
        phase_statistics = statistics_delta.get("multi_gpu", {})
        phase_cuda_aware = phase_statistics.get("cuda_aware_bytes")
        phase_pinned = phase_statistics.get("pinned_staging_bytes")
        phase_messages = phase_statistics.get("mpi_messages")
        phase_scalars = phase_statistics.get("mpi_scalars")
        phase_pinned_d2h = phase_statistics.get("pinned_device_to_host_bytes")
        phase_pinned_h2d = phase_statistics.get("pinned_host_to_device_bytes")
        phase_transport_values = {
            "mpi_messages": phase_messages,
            "mpi_scalars": phase_scalars,
            "cuda_aware_bytes": phase_cuda_aware,
            "pinned_staging_bytes": phase_pinned,
            "pinned_device_to_host_bytes": phase_pinned_d2h,
            "pinned_host_to_device_bytes": phase_pinned_h2d,
        }
        for counter_name, value in phase_transport_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(
                    f"{label}.{phase_name} MPI {counter_name} is invalid"
                )
        if world_size > 1:
            if (
                phase_messages <= 0
                or phase_scalars <= 0
                or not (bool(phase_cuda_aware) ^ bool(phase_pinned))
                or phase_cuda_aware + phase_pinned != phase_scalars * 4
                or (
                    phase_pinned > 0
                    and (phase_pinned_d2h <= 0 or phase_pinned_h2d <= 0)
                )
                or phase_pinned_d2h + phase_pinned_h2d != phase_pinned
            ):
                raise RuntimeError(
                    f"{label}.{phase_name} did not use exactly one measured MPI transport"
                )
            phase_transport = "cuda-aware" if phase_cuda_aware else "pinned-staging"
            if phase_transport != transport:
                raise RuntimeError(f"{label} changed MPI transport between phases")
        else:
            if any(
                value != 0
                for value in (
                    phase_messages,
                    phase_scalars,
                    phase_cuda_aware,
                    phase_pinned,
                    phase_pinned_d2h,
                    phase_pinned_h2d,
                )
            ):
                raise RuntimeError(
                    f"{label}.{phase_name} singleton phase recorded MPI traffic"
                )
        phase_coverage[phase_name] = measured

    for counter_name, (group, stem) in PHASE_COUNTER_STATISTICS.items():
        final_group = statistics.get(group) if isinstance(statistics, dict) else None
        if not isinstance(final_group, dict):
            raise RuntimeError(f"{label} final {counter_name} statistics are absent")
        for backend_name in ("cpu", "cuda"):
            for field in ("calls", "points"):
                final_value = final_group.get(f"{backend_name}_{stem}_{field}")
                phase_sum = sum(
                    phase[group][f"{backend_name}_{stem}_{field}"]
                    for phase in phase_statistics_records
                )
                if final_value != phase_sum:
                    raise RuntimeError(
                        f"{label} final {counter_name} statistics differ from phase sum"
                    )
    for field in (
        "mpi_messages",
        "mpi_scalars",
        "cuda_aware_bytes",
        "pinned_staging_bytes",
        "pinned_device_to_host_bytes",
        "pinned_host_to_device_bytes",
    ):
        if statistics["multi_gpu"].get(field) != sum(
            phase["multi_gpu"][field] for phase in phase_statistics_records
        ):
            raise RuntimeError(f"{label} final MPI {field} differs from phase sum")

    identifier = rank.get("selected_device_identifier")
    if (
        not isinstance(identifier, str)
        or DEVICE_IDENTIFIER_PATTERN.fullmatch(identifier) is None
    ):
        raise RuntimeError(f"{label} physical CUDA/MIG device identifier is absent")
    if identifier != backend.get("selected_device_identifier"):
        raise RuntimeError(f"{label} per-rank device identifiers disagree")
    if inventory_by_ordinal[selected_ordinal]["identifier"] != identifier:
        raise RuntimeError(f"{label} selected CUDA identifier disagrees with inventory")
    return {
        "cpu_fallback_calls": cpu_calls,
        "cuda_calls": cuda_calls,
        "mpi_messages": mpi_messages,
        "mpi_scalars": mpi_scalars,
        "cuda_aware_bytes": cuda_aware,
        "pinned_staging_bytes": pinned,
        "pinned_device_to_host_bytes": pinned_d2h,
        "pinned_host_to_device_bytes": pinned_h2d,
        "selected_device_identifier": identifier,
        "selected_device": selected_ordinal,
        "device_inventory_sha256": canonical_sha256(inventory),
        "process_environment_sha256": canonical_sha256(
            semantic_process_environment
        ),
        "process_environment_full_sha256": canonical_sha256(process_environment),
        "process_environment_semantic": semantic_process_environment,
        "dft_calls": cuda_calls["cuda_dft_calls"],
        "transport": transport,
        "phases": phase_coverage,
    }


def validate_distributed_record(
    result: dict[str, Any],
    *,
    expected_world_size: int,
    label: str,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    if hash_cache is None:
        hash_cache = StatHashCache()
    if result.get("schema_version") != 4:
        raise RuntimeError(f"{label} benchmark schema is unsupported")
    validate_gradient_summaries(result, label)
    objective = result.get("result", {}).get("objective")
    if isinstance(objective, bool) or not isinstance(objective, (int, float)):
        raise RuntimeError(f"{label} objective is absent")
    if not math.isfinite(float(objective)):
        raise RuntimeError(f"{label} objective is non-finite")

    distributed = result.get("distributed")
    if not isinstance(distributed, dict):
        raise RuntimeError(f"{label} distributed evidence is absent")
    if distributed.get("world_size") != expected_world_size:
        raise RuntimeError(f"{label} world size does not match the requested rank count")
    if distributed.get("timing_semantics") != (
        "barrier-delimited maximum rank elapsed per phase"
    ):
        raise RuntimeError(f"{label} timing is not barrier-delimited max-rank time")
    gate = distributed.get("gate")
    if not isinstance(gate, dict) or gate.get("pass") is not True or any(
        gate.get(name) is not True
        for name in ("objective", "gradient", "strict_cuda_coverage")
    ):
        raise RuntimeError(f"{label} producer-side distributed gate did not pass")

    ranks = distributed.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != expected_world_size:
        raise RuntimeError(f"{label} rank evidence is incomplete")
    if [item.get("world_rank") for item in ranks] != list(range(expected_world_size)):
        raise RuntimeError(f"{label} rank evidence is duplicate or out of order")

    top_runtime = validate_runtime_closure(
        result, label, hash_cache=hash_cache
    )
    rank_coverage: list[dict[str, Any]] = []
    rank_gradients: list[list[float]] = []
    rank_runtimes: list[dict[str, Any]] = []
    for rank_index, rank in enumerate(ranks):
        rank_label = f"{label}.rank{rank_index}"
        if not isinstance(rank, dict):
            raise RuntimeError(f"{rank_label} evidence is not an object")
        rank_runtime = validate_runtime_closure(
            rank, rank_label, hash_cache=hash_cache
        )
        if rank_index == 0:
            if rank_runtime != top_runtime:
                raise RuntimeError(
                    f"{rank_label} differs from the rank-zero runtime closure"
                )
        elif runtime_environment_independent_projection(
            rank_runtime
        ) != runtime_environment_independent_projection(top_runtime):
            raise RuntimeError(
                f"{rank_label} loaded different receipt-controlled runtime identities"
            )
        rank_runtimes.append(rank_runtime)
        gradient = rank.get("gradient")
        if not isinstance(gradient, list):
            raise RuntimeError(f"{rank_label} full gradient is absent")
        try:
            gradient_values = [float(value) for value in gradient]
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{rank_label} gradient is malformed") from error
        if not all(math.isfinite(value) for value in gradient_values):
            raise RuntimeError(f"{rank_label} gradient is non-finite")
        if rank.get("gradient_sha256") != canonical_float64_sha256(gradient_values):
            raise RuntimeError(f"{rank_label} gradient digest is stale")
        rank_gradients.append(gradient_values)
        rank_coverage.append(
            validate_rank_cuda_coverage(
                rank, world_size=expected_world_size, label=rank_label
            )
        )

    identifiers = [item["selected_device_identifier"] for item in rank_coverage]
    identifiers_unique = len(set(identifiers)) == expected_world_size
    if distributed.get("selected_devices_unique") is not identifiers_unique:
        raise RuntimeError(f"{label} selected-device uniqueness marker is stale")
    if not identifiers_unique:
        raise RuntimeError(f"{label} ranks did not use distinct physical CUDA/MIG devices")

    transports = {item["transport"] for item in rank_coverage}
    if len(transports) != 1:
        raise RuntimeError(f"{label} ranks used different MPI GPU-halo transports")
    for phase_name in ("forward", "adjoint"):
        source_calls = sum(
            item["phases"][phase_name]["source"]["calls"]
            for item in rank_coverage
        )
        source_points = sum(
            item["phases"][phase_name]["source"]["points"]
            for item in rank_coverage
        )
        if source_calls <= 0 or source_points <= 0:
            raise RuntimeError(
                f"{label}.{phase_name} has no aggregate CUDA source coverage"
            )

    top_gradient = validate_gradient_record(result)
    if any(values != top_gradient for values in rank_gradients):
        raise RuntimeError(f"{label} rank gradients are not bitwise identical")
    top_objective = float(objective)
    if any(float(rank.get("objective")) != top_objective for rank in ranks):
        raise RuntimeError(f"{label} rank objectives are not bitwise identical")
    if result.get("workload", {}).get("final_statistics") != ranks[0].get(
        "final_statistics"
    ):
        raise RuntimeError(f"{label} top-level statistics are not rank-zero statistics")

    timing = result.get("timing")
    if not isinstance(timing, dict):
        raise RuntimeError(f"{label} timing evidence is absent")
    rank_workload_wall_seconds = [
        _finite_positive(
            rank.get("timing", {}).get("workload_wall_seconds"),
            f"{label}.rank{rank_index}.timing.workload_wall_seconds",
        )
        for rank_index, rank in enumerate(ranks)
    ]
    workload_wall_seconds = _finite_positive(
        timing.get("workload_wall_seconds"),
        f"{label}.timing.workload_wall_seconds",
    )
    if workload_wall_seconds != max(rank_workload_wall_seconds):
        raise RuntimeError(
            f"{label} workload_wall_seconds is not the exact max-rank time"
        )
    if timing.get("rank_local_workload_wall_seconds") != rank_workload_wall_seconds[0]:
        raise RuntimeError(
            f"{label} rank-local workload wall is not the exact rank-zero time"
        )
    phase_maxima: dict[str, float] = {}
    for field in TIMING_FIELDS:
        phase_name = field.removesuffix("_seconds")
        rank_values = [
            _finite_positive(rank.get("timing", {}).get(field), f"{label}.{field}")
            for rank in ranks
        ]
        maximum = max(rank_values)
        recorded = _finite_positive(timing.get(field), f"{label}.timing.{field}")
        if recorded != maximum:
            raise RuntimeError(f"{label} {field} is not the exact max-rank time")
        if any(
            rank.get("phases", {}).get(phase_name, {}).get("elapsed_seconds")
            != rank_value
            for rank, rank_value in zip(ranks, rank_values)
        ):
            raise RuntimeError(
                f"{label} {phase_name} rank timing/phase record differs"
            )
        if (
            result.get("workload", {})
            .get("phases", {})
            .get(phase_name, {})
            .get("elapsed_seconds")
            != maximum
        ):
            raise RuntimeError(
                f"{label} {phase_name} top timing/phase record differs"
            )
        phase_maxima[field] = maximum
    elapsed = _finite_positive(timing.get("elapsed_seconds"), f"{label}.elapsed")
    if not math.isclose(
        elapsed, sum(phase_maxima.values()), rel_tol=1e-14, abs_tol=1e-14
    ):
        raise RuntimeError(f"{label} total time is not the sum of max-rank phases")
    for rank_index, (rank, rank_workload_wall) in enumerate(
        zip(ranks, rank_workload_wall_seconds)
    ):
        rank_elapsed = _finite_positive(
            rank.get("timing", {}).get("elapsed_seconds"),
            f"{label}.rank{rank_index}.timing.elapsed_seconds",
        )
        rank_phase_sum = math.fsum(
            _finite_positive(
                rank.get("timing", {}).get(field),
                f"{label}.rank{rank_index}.{field}",
            )
            for field in TIMING_FIELDS
        )
        if not math.isclose(
            rank_elapsed, rank_phase_sum, rel_tol=1e-14, abs_tol=1e-14
        ):
            raise RuntimeError(
                f"{label}.rank{rank_index} elapsed time is not its phase sum"
            )
        if rank_workload_wall < rank_elapsed:
            raise RuntimeError(
                f"{label}.rank{rank_index} workload wall time is shorter than "
                "rank elapsed time"
            )
        if rank_index == 0 and timing.get("rank_local_elapsed_seconds") != rank_elapsed:
            raise RuntimeError(
                f"{label} rank-local elapsed is not the exact rank-zero time"
            )

    return {
        "runtime": top_runtime,
        "rank_runtimes": rank_runtimes,
        "rank_coverage": rank_coverage,
        "device_identifiers": identifiers,
        "workload_wall_seconds": workload_wall_seconds,
        "elapsed_seconds": elapsed,
        "objective": top_objective,
    }


def validate_qualification_binding(
    result: dict[str, Any],
    *,
    expected_nonce: str,
    expected_run_id: str,
    expected_receipt_id: str,
    expected_source_snapshot_sha256: str,
    expected_producer_sha256: str,
    expected_sample_kind: str,
    expected_sample_lane: str,
    expected_sample_iteration: int,
    label: str,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    qualification = result.get("qualification")
    expected = {
        "mode": "qualification",
        "profile": QUALIFICATION_PROFILE,
        "run_nonce": expected_nonce,
        "run_id": expected_run_id,
        "build_receipt_id": expected_receipt_id,
        "source_snapshot_sha256": expected_source_snapshot_sha256,
        "producer_sha256": expected_producer_sha256,
        "sample_kind": expected_sample_kind,
        "sample_lane": expected_sample_lane,
        "sample_iteration": expected_sample_iteration,
    }
    if qualification != expected:
        raise RuntimeError(f"{label} qualification binding is stale or forged")
    producer = result.get("producer")
    if not isinstance(producer, dict):
        raise RuntimeError(f"{label} producer evidence is absent")
    if producer.get("sha256") != expected_producer_sha256:
        raise RuntimeError(f"{label} producer digest does not match its binding")
    producer_path = pathlib.Path(str(producer.get("path", ""))).resolve()
    producer_digest = (
        hash_cache.digest(producer_path)
        if hash_cache is not None and producer_path.is_file()
        else sha256_file(producer_path)
        if producer_path.is_file()
        else None
    )
    if producer_digest != expected_producer_sha256:
        raise RuntimeError(f"{label} executing producer changed after measurement")
    command = producer.get("command")
    if not isinstance(command, list) or str(producer_path) not in command:
        raise RuntimeError(f"{label} producer command is absent or inconsistent")
    for timestamp_name in ("started_at_utc", "completed_at_utc"):
        timestamp = producer.get(timestamp_name)
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise RuntimeError(f"{label} {timestamp_name} is absent")
    environment = producer.get("environment")
    if not isinstance(environment, dict):
        raise RuntimeError(f"{label} producer environment is absent")
    return qualification


def compare_workloads(single: dict[str, Any], multi: dict[str, Any]) -> dict[str, Any]:
    single_workload = single.get("workload")
    multi_workload = multi.get("workload")
    if not isinstance(single_workload, dict) or not isinstance(multi_workload, dict):
        raise RuntimeError("workload evidence is absent")
    comparisons: list[dict[str, Any]] = []
    for key in WORKLOAD_KEYS:
        left = single_workload.get(key)
        right = multi_workload.get(key)
        comparisons.append(
            {
                "metric": key,
                "single": left,
                "multi": right,
                "pass": left is not None and left == right,
            }
        )
    for phase in ("forward", "adjoint"):
        for metric in ("timesteps", "end_time"):
            left = single_workload.get("phases", {}).get(phase, {}).get(metric)
            right = multi_workload.get("phases", {}).get(phase, {}).get(metric)
            comparisons.append(
                {
                    "metric": f"{phase}.{metric}",
                    "single": left,
                    "multi": right,
                    "pass": left is not None and left == right,
                }
            )
    return {
        "pass": all(item["pass"] for item in comparisons),
        "comparisons": comparisons,
    }


def measured_dependency_projection_matches(
    group: str,
    captured_projection: list[tuple[str, str]],
    runtime_projection: list[tuple[str, str]],
) -> bool:
    """Match one measured dependency group against its sealed build closure."""

    if group != "hdf5":
        return runtime_projection == captured_projection
    captured_set = set(captured_projection)
    runtime_set = set(runtime_projection)
    hdf5_soname = re.compile(r"libhdf5(?:_hl)?\.so(?:\.[0-9]+)*\Z")
    core_hdf5_soname = re.compile(r"libhdf5\.so(?:\.[0-9]+)*\Z")
    all_names_are_hdf5 = all(
        hdf5_soname.fullmatch(pathlib.Path(path).name) is not None
        for path, _sha256 in (*captured_projection, *runtime_projection)
    )
    has_core_hdf5 = any(
        core_hdf5_soname.fullmatch(pathlib.Path(path).name) is not None
        for path, _sha256 in runtime_projection
    )
    return (
        bool(runtime_projection)
        and all_names_are_hdf5
        and len(captured_set) == len(captured_projection)
        and len(runtime_set) == len(runtime_projection)
        and runtime_set.issubset(captured_set)
        and has_core_hdf5
    )


def validate_receipt_runtime(
    receipt: dict[str, Any],
    runtime: dict[str, Any],
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    if hash_cache is None:
        hash_cache = StatHashCache()
    if receipt.get("build_kind") != "cuda-mpi-python-fp32":
        raise RuntimeError("build receipt is not a CUDA+MPI+Python FP32 build")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("build receipt artifact set is absent")

    def matching_artifacts(prefix: str, digest: str) -> list[str]:
        return sorted(
            name
            for name, record in artifacts.items()
            if isinstance(record, dict)
            and pathlib.Path(str(record.get("path", ""))).name.startswith(prefix)
            and record.get("sha256") == digest
        )

    extension_matches = matching_artifacts(
        "_meep.so", runtime["extension"]["sha256"]
    )
    libmeep_matches = matching_artifacts(
        "libmeep.so", runtime["libmeep"]["sha256"]
    )
    if not extension_matches or not libmeep_matches:
        raise RuntimeError("measured extension/libmeep do not match the build receipt")

    manifests = receipt.get("manifests")
    installed_python = (
        manifests.get("installed_python") if isinstance(manifests, dict) else None
    )
    package = runtime.get("python_package")
    if not isinstance(installed_python, dict) or not isinstance(package, dict):
        raise RuntimeError("installed Python package manifest is absent")
    installed_root_value = installed_python.get("root")
    if not isinstance(installed_root_value, str):
        raise RuntimeError("receipt installed-Python root is invalid")
    installed_root = pathlib.Path(installed_root_value)
    if not installed_root.is_absolute():
        installed_root = pathlib.Path(str(receipt.get("repo", ""))) / installed_root
    if (
        pathlib.Path(package["root"]).resolve() != installed_root.resolve()
        or package["file_count"] != installed_python.get("file_count")
        or package["sha256"] != installed_python.get("sha256")
    ):
        raise RuntimeError("measured Python package does not match the build receipt")

    module_path = pathlib.Path(runtime["meep_module"]["path"]).resolve()
    if module_path.parent != pathlib.Path(package["root"]).resolve():
        raise RuntimeError("loaded meep module lies outside the receipt-bound package")
    receipt_module = next(
        (
            record
            for record in installed_python.get("files", [])
            if isinstance(record, dict) and record.get("path") == "__init__.py"
        ),
        None,
    )
    if (
        not isinstance(receipt_module, dict)
        or receipt_module.get("sha256") != runtime["meep_module"]["sha256"]
    ):
        raise RuntimeError("loaded meep module does not match the package manifest")

    toolchain = receipt.get("toolchain")
    receipt_python = toolchain.get("python") if isinstance(toolchain, dict) else None
    if not isinstance(receipt_python, dict) or not isinstance(receipt_python.get("path"), str):
        raise RuntimeError("receipt Python toolchain record is absent")
    receipt_python_path = pathlib.Path(receipt_python["path"])
    if not receipt_python_path.is_absolute():
        receipt_python_path = pathlib.Path(str(receipt.get("repo", ""))) / receipt_python_path
    receipt_python_path = receipt_python_path.resolve()
    if receipt_python_path != pathlib.Path(
        runtime["python_executable"]["path"]
    ).resolve():
        raise RuntimeError("benchmark used a different Python executable than the build")
    configuration_files = receipt.get("configuration_files")
    dependency_record = (
        configuration_files.get("runtime_dependency_closure")
        if isinstance(configuration_files, dict)
        else None
    )
    if not isinstance(dependency_record, dict) or not isinstance(
        dependency_record.get("path"), str
    ):
        raise RuntimeError("receipt runtime-dependency closure is absent")
    dependency_path = pathlib.Path(dependency_record["path"])
    if not dependency_path.is_absolute():
        dependency_path = pathlib.Path(str(receipt.get("repo", ""))) / dependency_path
    try:
        captured_value = json.loads(dependency_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read receipt runtime-dependency closure") from error
    installed_libmeep_record = artifacts.get("installed_libmeep")
    if not isinstance(installed_libmeep_record, dict) or not isinstance(
        installed_libmeep_record.get("path"), str
    ):
        raise RuntimeError("receipt installed libmeep artifact is absent")
    installed_libmeep_path = pathlib.Path(installed_libmeep_record["path"])
    if not installed_libmeep_path.is_absolute():
        installed_libmeep_path = (
            pathlib.Path(str(receipt.get("repo", ""))) / installed_libmeep_path
        )
    captured = validate_captured_runtime_dependency_closure(
        captured_value,
        receipt_python_path.parent.parent,
        installed_libmeep_path.resolve().parent.parent,
        hash_cache=hash_cache,
    )
    environment_prefix = receipt_python_path.parent.parent.resolve()
    installed_environment = (
        manifests.get("installed_environment")
        if isinstance(manifests, dict)
        else None
    )
    if (
        not isinstance(installed_environment, dict)
        or not isinstance(installed_environment.get("root"), str)
        or not isinstance(installed_environment.get("files"), list)
    ):
        raise RuntimeError("receipt installed-environment manifest is absent")
    manifest_root = pathlib.Path(installed_environment["root"])
    if not manifest_root.is_absolute():
        manifest_root = pathlib.Path(str(receipt.get("repo", ""))) / manifest_root
    if manifest_root.resolve() != environment_prefix:
        raise RuntimeError("receipt installed-environment root is wrong")
    manifest_records: dict[str, tuple[int, str]] = {}
    for record in installed_environment["files"]:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or isinstance(record.get("size_bytes"), bool)
            or not isinstance(record.get("size_bytes"), int)
            or not isinstance(record.get("sha256"), str)
        ):
            raise RuntimeError("receipt installed-environment record is invalid")
        resolved = str((environment_prefix / record["path"]).resolve())
        identity = (record["size_bytes"], record["sha256"])
        previous = manifest_records.get(resolved)
        if previous is not None and previous != identity:
            raise RuntimeError("receipt environment aliases have different identities")
        manifest_records[resolved] = identity

    def require_environment_membership(
        records: list[dict[str, Any]], label: str
    ) -> None:
        for record in records:
            path = pathlib.Path(str(record.get("path", ""))).resolve()
            identity = (path.stat().st_size, record.get("sha256"))
            if manifest_records.get(str(path)) != identity:
                raise RuntimeError(
                    f"{label} file is absent or different in the receipt environment: {path}"
                )

    runtime_groups = runtime.get("loaded_libraries")
    if not isinstance(runtime_groups, dict):
        raise RuntimeError("measured loaded-library groups are absent")
    require_environment_membership(
        captured["groups"]["environment"], "build-captured mapped environment"
    )
    require_environment_membership(
        runtime_groups.get("environment", []), "measured mapped environment"
    )
    for group in ("mpi", "cuda_runtime", "hdf5"):
        captured_records = captured["groups"][group]
        captured_projection = sorted(
            (str(pathlib.Path(record["path"]).resolve()), record["sha256"])
            for record in captured_records
        )
        runtime_projection = sorted(
            (str(pathlib.Path(record["path"]).resolve()), record["sha256"])
            for record in runtime_groups.get(group, [])
        )
        # The build closure is captured after importing both ``meep`` and
        # ``meep.mpb``.  MPB maps libhdf5_hl in addition to libhdf5, while a
        # normal Meep/adjoint process legitimately maps only the core HDF5
        # library.  Keep exact equality for MPI and CUDA, but require the
        # measured HDF5 closure to be a non-empty, core-containing subset of
        # the receipt closure.  Paths and hashes still have to match the
        # sealed environment exactly, so this does not admit an unrecorded
        # HDF5 implementation.
        if not measured_dependency_projection_matches(
            group, captured_projection, runtime_projection
        ):
            raise RuntimeError(
                f"measured {group} libraries differ from the receipt closure"
            )
    library_origins = validate_loaded_library_origins(
        runtime, environment_prefix
    )
    return {
        "receipt_id": receipt["receipt_id"],
        "build_input_id": receipt["build_input_id"],
        "artifact_set_id": receipt["artifact_set_id"],
        "extension_artifacts": extension_matches,
        "libmeep_artifacts": libmeep_matches,
        "python_package_manifest": "installed_python",
        "installed_environment_manifest": {
            "name": "installed_environment",
            "file_count": installed_environment.get("file_count"),
            "sha256": installed_environment.get("sha256"),
        },
        "loaded_library_origins": library_origins,
    }


def validate_loaded_library_origins(
    runtime: dict[str, Any], environment_prefix: pathlib.Path
) -> dict[str, Any]:
    loaded = runtime.get("loaded_libraries")
    if not isinstance(loaded, dict):
        raise RuntimeError("loaded-library origin evidence is absent")
    environment_prefix = environment_prefix.resolve()
    isolated: dict[str, list[str]] = {}
    for group in ("environment", "mpi", "cuda_runtime", "hdf5"):
        records = loaded.get(group)
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"loaded {group} libraries are absent")
        isolated[group] = []
        for record in records:
            path = pathlib.Path(str(record.get("path", ""))).resolve()
            try:
                path.relative_to(environment_prefix)
            except ValueError as error:
                raise RuntimeError(
                    f"loaded {group} library is outside the isolated environment"
                ) from error
            isolated[group].append(str(path))
    trusted_roots = [
        root.resolve() for root in TRUSTED_CUDA_DRIVER_ROOTS if root.is_dir()
    ]
    driver_paths: list[str] = []
    driver_records = loaded.get("cuda_driver")
    if not isinstance(driver_records, list):
        raise RuntimeError("loaded CUDA driver origin is absent")
    for record in driver_records:
        path = pathlib.Path(str(record.get("path", ""))).resolve()
        matching_root = next(
            (root for root in trusted_roots if path == root or root in path.parents),
            None,
        )
        if (
            not path.is_file()
            or matching_root is None
            or not path.name.startswith("libcuda.so")
        ):
            raise RuntimeError("loaded CUDA driver is outside protected system paths")
        protected_chain = [path]
        current = path.parent
        while True:
            protected_chain.append(current)
            if current == matching_root:
                break
            current = current.parent
        if any(
            item.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            for item in protected_chain
        ):
            raise RuntimeError("loaded CUDA driver path is group/world writable")
        driver_paths.append(str(path))
    if not driver_paths:
        raise RuntimeError("loaded CUDA driver origin is absent")
    return {
        "environment_prefix": str(environment_prefix),
        "isolated": {
            group: paths
            for group, paths in isolated.items()
            if group != "environment"
        },
        "mapped_environment_file_count": len(isolated["environment"]),
        "mapped_environment_files_sha256": canonical_sha256(
            isolated["environment"]
        ),
        "cuda_driver": driver_paths,
    }


def validate_captured_runtime_dependency_closure(
    closure: Any,
    environment_prefix: pathlib.Path,
    installed_prefix: pathlib.Path,
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    if hash_cache is None:
        hash_cache = StatHashCache()
    if (
        not isinstance(closure, dict)
        or closure.get("schema_version") != 3
        or pathlib.Path(str(closure.get("environment_prefix", ""))).resolve()
        != environment_prefix.resolve()
        or pathlib.Path(str(closure.get("installed_prefix", ""))).resolve()
        != installed_prefix.resolve()
        or not isinstance(closure.get("groups"), dict)
        or set(closure["groups"])
        != {
            "mpi", "cuda_driver", "cuda_runtime", "hdf5", "environment",
            "installed",
        }
    ):
        raise RuntimeError("receipt runtime-dependency closure is invalid")
    groups = closure["groups"]
    for group, records in groups.items():
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"receipt runtime-dependency group {group} is empty")
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "path",
                "size_bytes",
                "sha256",
            }:
                raise RuntimeError("receipt runtime-dependency record is invalid")
            path = pathlib.Path(str(record["path"])).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != record["size_bytes"]
                or hash_cache.digest(path) != record["sha256"]
            ):
                raise RuntimeError(f"receipt runtime-dependency {group} changed")
    origins = validate_loaded_library_origins(
        {"loaded_libraries": groups}, environment_prefix
    )
    environment_records = {
        (str(pathlib.Path(record["path"]).resolve()), record["sha256"])
        for record in groups["environment"]
    }
    for group in ("mpi", "cuda_runtime", "hdf5"):
        named_records = {
            (str(pathlib.Path(record["path"]).resolve()), record["sha256"])
            for record in groups[group]
        }
        if not named_records.issubset(environment_records):
            raise RuntimeError(
                f"receipt runtime-dependency {group} is absent from mapped environment"
            )
    installed_records = {
        (str(pathlib.Path(record["path"]).resolve()), record["sha256"])
        for record in groups["installed"]
    }
    installed_root = installed_prefix.resolve()
    if not installed_records:
        raise RuntimeError("receipt installed runtime closure has a wrong origin")
    for path, _digest in installed_records:
        try:
            pathlib.Path(path).resolve().relative_to(installed_root)
        except ValueError as error:
            raise RuntimeError(
                "receipt installed runtime closure has a wrong origin"
            ) from error
    required_installed = ("_meep.so", "_mpb.so", "libmeep.so", "libpympb.so")
    if any(
        not any(pathlib.Path(path).name.startswith(prefix) for path, _ in installed_records)
        for prefix in required_installed
    ):
        raise RuntimeError("receipt installed runtime closure is incomplete")
    return {"groups": groups, "origins": origins}


def prepare_output(output: pathlib.Path, run_id: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in FINAL_ARTIFACTS:
        (output / name).unlink(missing_ok=True)
    atomic_write_json(
        output / "state.json",
        {"schema_version": 1, "state": "RUNNING", "run_id": run_id},
    )


def mark_failed(output: pathlib.Path, run_id: str, error: BaseException) -> None:
    for name in ("report.json", "report.md", "COMPLETE"):
        (output / name).unlink(missing_ok=True)
    failure = {
        "schema_version": 1,
        "state": "FAILED",
        "run_id": run_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }
    atomic_write_json(output / "FAILED.json", failure)
    atomic_write_json(output / "state.json", failure)


def publish_complete(
    output: pathlib.Path, run_id: str, report: dict[str, Any], markdown: str
) -> None:
    del output, run_id, report, markdown
    raise RuntimeError(
        "standalone MPI comparison cannot publish COMPLETE; use the fixed runner"
    )


def execute(args: argparse.Namespace, repo: pathlib.Path, run_id: str) -> int:
    del args, repo, run_id
    raise RuntimeError(
        "standalone MPI comparison cannot qualify a release; use "
        "run-mpi-adjoint-benchmark.py"
    )

    # Kept below as unreachable historical comparison/reporting code so older
    # evidence remains understandable.  No callable path can publish it as a
    # release qualification.
    if args.expected_multi_ranks < 2:
        raise ValueError("expected-multi-ranks must be at least two")
    if args.objective_atol < 0 or args.gradient_atol < 0 or args.gradient_rtol < 0:
        raise ValueError("numeric tolerances must be nonnegative")
    if args.minimum_speedup <= 1:
        raise ValueError("minimum-speedup must be greater than one")
    if args.maximum_dft_call_multiplier < 1:
        raise ValueError("maximum-dft-call-multiplier must be at least one")

    snapshot_before = source_snapshot(repo)
    receipt = verify_build_receipt(args.build_receipt.resolve(), repo)
    single_path = args.single_report.resolve()
    multi_path = args.multi_report.resolve()
    input_records = {
        "single": {"path": str(single_path), "sha256": sha256_file(single_path)},
        "multi": {"path": str(multi_path), "sha256": sha256_file(multi_path)},
    }
    single = _load_json(single_path, "single-rank report")
    multi = _load_json(multi_path, "multi-rank report")
    single_validation = validate_distributed_record(
        single, expected_world_size=1, label="single"
    )
    multi_validation = validate_distributed_record(
        multi, expected_world_size=args.expected_multi_ranks, label="multi"
    )
    if single_validation["runtime"] != multi_validation["runtime"]:
        raise RuntimeError("single- and multi-rank runs loaded different runtime closures")
    receipt_gate = validate_receipt_runtime(receipt, single_validation["runtime"])

    workload_gate = compare_workloads(single, multi)
    if single.get("workload", {}).get("material_gradient_path") != (
        args.expected_material_gradient_path
    ):
        raise RuntimeError("benchmark used an unexpected MaterialGrid gradient path")

    objective_error = abs(
        single_validation["objective"] - multi_validation["objective"]
    )
    objective_gate = objective_error <= args.objective_atol
    expected_count = single.get("workload", {}).get("design_variables")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise RuntimeError("design-variable count is invalid")
    gradient_gate = compare_gradient_vectors(
        single,
        multi,
        atol=args.gradient_atol,
        rtol=args.gradient_rtol,
        expected_count=expected_count,
    )

    single_dft_calls = single_validation["rank_coverage"][0]["dft_calls"]
    multi_dft_calls = [
        rank["dft_calls"] for rank in multi_validation["rank_coverage"]
    ]
    dft_limit = max(1.0, float(single_dft_calls)) * args.maximum_dft_call_multiplier
    dft_gate = {
        "pass": single_dft_calls > 0 and max(multi_dft_calls) <= dft_limit,
        "single_rank_calls": single_dft_calls,
        "multi_rank_calls": multi_dft_calls,
        "maximum_allowed_per_rank": dft_limit,
        "maximum_call_multiplier": args.maximum_dft_call_multiplier,
    }

    speedup = (
        single_validation["elapsed_seconds"]
        / multi_validation["elapsed_seconds"]
    )
    performance_gate = {
        "pass": speedup >= args.minimum_speedup,
        "single_seconds": single_validation["elapsed_seconds"],
        "multi_seconds": multi_validation["elapsed_seconds"],
        "speedup_single_over_multi": speedup,
        "minimum_speedup": args.minimum_speedup,
    }
    gates = {
        "workload": workload_gate["pass"],
        "objective": objective_gate,
        "gradient": gradient_gate["pass"],
        "dft_decimation": dft_gate["pass"],
        "performance": performance_gate["pass"],
        "provenance": True,
        "strict_cuda": True,
    }
    gates["pass"] = all(gates.values())
    if not gates["pass"]:
        failed = ", ".join(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"MPI adjoint comparison gates failed: {failed}")

    snapshot_after = source_snapshot(repo)
    if snapshot_after != snapshot_before or snapshot_after != receipt.get("source_end"):
        raise RuntimeError("source changed or differs from the verified build receipt")
    postprocessor = {
        "git_head": git_output(repo, "rev-parse", "HEAD").strip(),
        "git_status_porcelain": git_output(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines(),
        "source_snapshot": snapshot_after,
        "source_sha256": {
            relative: sha256_file(repo / relative)
            for relative in POSTPROCESSOR_SOURCES
        },
        "python": {
            "executable": str(pathlib.Path(sys.executable).resolve()),
            "sha256": sha256_file(pathlib.Path(sys.executable).resolve()),
            "version": sys.version,
        },
        "platform": platform.platform(),
        "command": [
            str(pathlib.Path(sys.executable).resolve()),
            str(pathlib.Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    }
    report = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "inputs": input_records,
        "build_receipt": receipt_gate,
        "postprocessor": postprocessor,
        "workload": workload_gate,
        "objective": {
            "pass": objective_gate,
            "single": single_validation["objective"],
            "multi": multi_validation["objective"],
            "absolute_error": objective_error,
            "atol": args.objective_atol,
        },
        "gradient": gradient_gate,
        "dft_decimation": dft_gate,
        "performance": performance_gate,
        "devices": {
            "single": single_validation["device_identifiers"],
            "multi": multi_validation["device_identifiers"],
        },
        "coverage": {
            "single": single_validation["rank_coverage"],
            "multi": multi_validation["rank_coverage"],
        },
        "gate": gates,
    }
    markdown = (
        "# gpmeep MPI adjoint comparison\n\n"
        f"- State: **COMPLETE**\n"
        f"- Ranks: 1 vs {args.expected_multi_ranks}\n"
        f"- Objective absolute error: `{objective_error:.9g}`\n"
        f"- Gradient max absolute error: `{gradient_gate['max_absolute_error']:.9g}`\n"
        f"- Gradient relative L2 error: `{gradient_gate['relative_l2_error']:.9g}`\n"
        f"- DFT calls/rank: `{single_dft_calls}` vs `{multi_dft_calls}`\n"
        f"- Strong-scaling speedup: `{speedup:.6f}x` "
        f"(required `{args.minimum_speedup:.6f}x`)\n"
        f"- Build receipt: `{receipt['receipt_id']}`\n"
    )
    publish_complete(args.output, run_id, report, markdown)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = uuid.uuid4().hex
    prepare_output(args.output, run_id)
    error = RuntimeError(
        "standalone input comparison is diagnostic-only and cannot qualify a release; "
        "run scripts/run-mpi-adjoint-benchmark.py instead"
    )
    mark_failed(args.output, run_id, error)
    print(f"error: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
