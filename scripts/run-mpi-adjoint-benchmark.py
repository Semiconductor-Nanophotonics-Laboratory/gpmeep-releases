#!/usr/bin/env python3
"""Run the fixed, receipt-bound gpmeep MPI adjoint qualification profile.

This is the only entry point allowed to publish an MPI-adjoint COMPLETE
marker.  It launches every raw sample itself, binds it to a fresh nonce and
the verified build receipt, and uses non-overridable release thresholds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shlex
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import types
import uuid
from typing import Any


def _load_source_module(name: str, path: pathlib.Path) -> types.ModuleType:
    """Load audited local tooling from source bytes, never adjacent .pyc."""

    resolved = path.resolve()
    module = types.ModuleType(name)
    module.__file__ = str(resolved)
    module.__package__ = ""
    sys.modules[name] = module
    source = resolved.read_bytes()
    exec(compile(source, str(resolved), "exec"), module.__dict__)
    return module


_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
_EVIDENCE = _load_source_module(
    "gpmeep_benchmark_evidence", _SCRIPT_DIRECTORY / "gpmeep_benchmark_evidence.py"
)
_PROVENANCE = _load_source_module(
    "gpmeep_provenance", _SCRIPT_DIRECTORY / "gpmeep_provenance.py"
)
_BUILD_QUALIFICATION = _load_source_module(
    "gpmeep_qualification_contract_for_adjoint",
    _SCRIPT_DIRECTORY / "gpmeep_qualification_contract.py",
)
atomic_write_json = _EVIDENCE.atomic_write_json
atomic_write_text = _EVIDENCE.atomic_write_text
canonical_float64_sha256 = _EVIDENCE.canonical_float64_sha256
compare_gradient_vectors = _EVIDENCE.compare_gradient_vectors
sha256_file = _EVIDENCE.sha256_file
validate_gradient_record = _EVIDENCE.validate_gradient_record
canonical_sha256 = _PROVENANCE.canonical_sha256
command_probe = _PROVENANCE.command_probe
file_record = _PROVENANCE.file_record
git_output = _PROVENANCE.git_output
source_snapshot = _PROVENANCE.source_snapshot
verify_build_receipt = _PROVENANCE.verify_build_receipt
StatHashCache = _PROVENANCE.StatHashCache


PROFILE_ID = "m19-cpu-gpu-adjoint-release-v4"
BUILD_QUALIFICATION_CONTRACT = "gpmeep-cuda-mpi-python-fp32-v2"
RELEASE_INITIAL_CONDITION = "trigonometric-v1"
RELEASE_SOURCE_PROFILE = "single-ez-v1"
PINNED_MICROMAMBA_SHA256 = (
    "9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82"
)
ALLOWED_CANONICAL_BUILD_ENVIRONMENT_KEYS = {
    "ADDR2LINE",
    "AR",
    "AS",
    "BUILD",
    "CC",
    "CC_FOR_BUILD",
    "CFLAGS",
    "CMAKE_ARGS",
    "CMAKE_PREFIX_PATH",
    "CONDA_BUILD_SYSROOT",
    "CONDA_DEFAULT_ENV",
    "CONDA_PKGS_DIRS",
    "CONDA_PREFIX",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_SHLVL",
    "CONDA_TOOLCHAIN_BUILD",
    "CONDA_TOOLCHAIN_HOST",
    "CPP",
    "CPPFLAGS",
    "CXX",
    "CXXFILT",
    "CXX_FOR_BUILD",
    "CXXFLAGS",
    "DEBUG_CFLAGS",
    "DEBUG_CPPFLAGS",
    "DEBUG_CXXFLAGS",
    "DEBUG_FFLAGS",
    "DEBUG_FORTRANFLAGS",
    "ELFEDIT",
    "F77",
    "F90",
    "F95",
    "FC",
    "FC_FOR_BUILD",
    "FFLAGS",
    "FORTRANFLAGS",
    "GCC",
    "GCC_AR",
    "GCC_NM",
    "GCC_RANLIB",
    "GFORTRAN",
    "GPMEEP_FRESH_ENV_NONCE",
    "GPROF",
    "GSETTINGS_SCHEMA_DIR",
    "GSETTINGS_SCHEMA_DIR_CONDA_BACKUP",
    "GXX",
    "HOME",
    "HOST",
    "LC_CTYPE",
    "LD",
    "LDFLAGS",
    "MAMBA_EXE",
    "MAMBA_ROOT_PREFIX",
    "MEEP_GPU_ALLOW_OVERSUBSCRIBE",
    "MEEP_GPU_CUDA_ARCHS",
    "MEEP_GPU_DEVICE",
    "MEEP_GPU_FAST_MATH",
    "MEEP_GPU_MAKE_JOBS",
    "MEEP_GPU_MPI_COMPLETION",
    "MEEP_GPU_MPI_TRANSPORT",
    "MEEP_GPU_MULTI_INITIAL_CONDITION",
    "MEEP_GPU_MULTI_SOURCE_PROFILE",
    "MESON_ARGS",
    "MPLBACKEND",
    "MPLCONFIGDIR",
    "NM",
    "NVCC_PREPEND_FLAGS",
    "OBJCOPY",
    "OBJDUMP",
    "OMPI_MCA_mca_base_component_path",
    "OMPI_MCA_mca_base_param_files",
    "OMPI_MCA_opal_cuda_support",
    "PATH",
    "PMIX_MCA_mca_base_component_path",
    "PMIX_MCA_mca_base_param_files",
    "PRTE_MCA_mca_base_param_files",
    "PWD",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONPYCACHEPREFIX",
    "PYTHONSAFEPATH",
    "RANLIB",
    "READELF",
    "SHLVL",
    "SIZE",
    "STRINGS",
    "STRIP",
    "UCX_MEMTYPE_CACHE",
    "XDG_CACHE_HOME",
    "XML_CATALOG_FILES",
    "_",
    "_CONDA_PYTHON_SYSCONFIGDATA_NAME",
    "build_alias",
    "host_alias",
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
}
RECORDED_BUILD_ENVIRONMENT_KEYS = {
    "MEEP_GPU_MAKE_JOBS",
    "MEEP_GPU_FAST_MATH",
    "MEEP_GPU_CUDA_ARCHS",
    "MEEP_GPU_DEVICE",
    "MEEP_GPU_ALLOW_OVERSUBSCRIBE",
    "MEEP_GPU_MPI_TRANSPORT",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CC",
    "CXX",
    "FC",
    "F77",
    "NVCC",
    "CUDAHOSTCXX",
    "MPICXX",
    "OMPI_MCA_opal_cuda_support",
    "OMPI_MCA_mca_base_param_files",
    "OMPI_MCA_mca_base_component_path",
    "PMIX_MCA_mca_base_param_files",
    "PMIX_MCA_mca_base_component_path",
    "PRTE_MCA_mca_base_param_files",
    "UCX_MEMTYPE_CACHE",
    "GPMEEP_FRESH_ENV_NONCE",
}
FORWARDED_OPTIONAL_BUILD_ENVIRONMENT_KEYS = (
    "MEEP_GPU_MAKE_JOBS",
    "MEEP_GPU_FAST_MATH",
    "MEEP_GPU_CUDA_ARCHS",
    "MEEP_GPU_DEVICE",
    "MEEP_GPU_ALLOW_OVERSUBSCRIBE",
    "MEEP_GPU_MPI_TRANSPORT",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "OMPI_MCA_opal_cuda_support",
    "UCX_MEMTYPE_CACHE",
)
PROFILE = types.MappingProxyType({
    "profile_id": PROFILE_ID,
    "resolution": 64,
    "run_time": 40.0,
    "cell_size": 32.0,
    "design_resolution": 20,
    "design_shape": [41, 41],
    "design_variables": 1681,
    "yee_cells": 4194304,
    "weights_sha256": "441c6df996857239098db8694ec724f6e3ec2312042fa2a667c650d3c8c58ada",
    "direction_sha256": "581786e703e71a2548224e070d2b9da0bab5de69476f9be5357441175323fb1a",
    "forward_timesteps": 6614,
    "forward_end_time": 51.671875,
    "forward_dft_calls": 152,
    "forward_dft_points": 1314268,
    "adjoint_timesteps": 19840,
    "adjoint_end_time": 155.0,
    "adjoint_dft_calls": 211,
    "adjoint_dft_points": 3620971,
    "multi_ranks": 2,
    "cpu_ranks": 2,
    "cpu_omp_threads": 4,
    "cpu_config_id": "2x4",
    "cpu_required_physical_cores": 8,
    "cpu_baseline_scope": "host-specific-full-physical-core",
    "warmups": 1,
    "repeats": 5,
    "material_gradient_path": "analytic-required",
    "oracle_material_gradient_path": "legacy",
    "objective_atol": 1.0e-8,
    "gradient_atol": 2.0e-7,
    "gradient_rtol": 5.0e-5,
    "minimum_median_speedup": 1.25,
    "minimum_conservative_speedup": 1.15,
    "minimum_cpu_gpu_median_speedup": 1.25,
    "minimum_cpu_gpu_conservative_speedup": 1.15,
    "maximum_timing_cv": 0.05,
    "maximum_cpu_pressure_some_fraction": 0.01,
    "maximum_pressure_audit_overhead_seconds": 0.25,
    "maximum_dft_point_ratio": 1.05,
    "timeout_seconds": 600,
    "timeout_kill_grace_seconds": 5,
})
PROFILE_CANONICAL_SHA256 = (
    "fef918a4cca4f97b17c523b648abf3b9b95012a1d1860dc381cc29a22a4ec3e1"
)

CPU_ORACLE_LANE = "cpu-legacy-oracle"
CPU_BENCHMARK_LANE = "cpu-2x4"
CUDA_LANES = ("cuda-single", "cuda-multi")
ANALYTIC_LANES = (CPU_BENCHMARK_LANE, *CUDA_LANES)
ALL_LANES = (CPU_ORACLE_LANE, *ANALYTIC_LANES)
MEASURED_LANES = ANALYTIC_LANES
CPU_STATISTICS_SCHEMA_SHA256 = (
    "bc721bd60d95bdc6674c1f65d1c16254e7832b35fbaa70d2f43d45037d0fac23"
)
_CPU_HOST_AUDIT_MODULE: types.ModuleType | None = None
_ACTIVE_HOST_CONTENTION_AUDIT: Any | None = None
MEASURED_ORDERS = (
    (CPU_BENCHMARK_LANE, "cuda-single", "cuda-multi"),
    ("cuda-multi", "cuda-single", CPU_BENCHMARK_LANE),
    ("cuda-single", "cuda-multi", CPU_BENCHMARK_LANE),
    (CPU_BENCHMARK_LANE, "cuda-multi", "cuda-single"),
)


def _require_lane(lane: str) -> str:
    if lane not in ALL_LANES:
        raise RuntimeError(f"unsupported qualification lane: {lane!r}")
    return lane


def _lane_ranks(lane: str) -> int:
    _require_lane(lane)
    if lane in (CPU_BENCHMARK_LANE, "cuda-multi"):
        return int(PROFILE["cpu_ranks"] if lane == CPU_BENCHMARK_LANE else PROFILE["multi_ranks"])
    return 1


def _cpu_host_audit_module() -> types.ModuleType:
    global _CPU_HOST_AUDIT_MODULE
    if _CPU_HOST_AUDIT_MODULE is None:
        _CPU_HOST_AUDIT_MODULE = _load_source_module(
            "gpmeep_mpi_cpu_host_audit",
            _SCRIPT_DIRECTORY / "run-cpu-parallel-benchmark.py",
        )
    return _CPU_HOST_AUDIT_MODULE


def _cpu_pressure_state() -> dict[str, int]:
    try:
        pressure_text = pathlib.Path("/proc/pressure/cpu").read_text(
            encoding="utf-8"
        )
    except OSError as error:
        raise RuntimeError("Linux CPU pressure evidence is unavailable") from error
    totals: dict[str, int] = {}
    for line in pressure_text.splitlines():
        columns = line.split()
        if not columns or columns[0] not in {"some", "full"}:
            continue
        total = next(
            (
                value.split("=", 1)[1]
                for value in columns[1:]
                if value.startswith("total=")
            ),
            None,
        )
        if total is None:
            raise RuntimeError("Linux CPU pressure total is absent")
        totals[columns[0]] = int(total)
    if set(totals) != {"some", "full"}:
        raise RuntimeError("Linux CPU pressure schema is incomplete")
    return {
        "monotonic_ns": time.monotonic_ns(),
        "some_total_usec": totals["some"],
        "full_total_usec": totals["full"],
    }


class _HostContentionAudit:
    def __init__(self) -> None:
        self.segments: list[dict[str, Any]] = []
        self.active = False

    def begin(
        self, *, label: str, lane: str, sample_kind: str, iteration: int
    ) -> dict[str, Any]:
        if self.active:
            raise RuntimeError("CPU pressure audit segment overlapped")
        self.active = True
        return {
            "label": label,
            "lane": lane,
            "sample_kind": sample_kind,
            "iteration": iteration,
            "start": _cpu_pressure_state(),
        }

    def end(self, token: dict[str, Any]) -> None:
        if not self.active:
            raise RuntimeError("CPU pressure audit segment is not active")
        end = _cpu_pressure_state()
        self.active = False
        start = token["start"]
        elapsed_usec = (end["monotonic_ns"] - start["monotonic_ns"]) / 1000.0
        some_delta = end["some_total_usec"] - start["some_total_usec"]
        full_delta = end["full_total_usec"] - start["full_total_usec"]
        if elapsed_usec <= 0 or some_delta < 0 or full_delta < 0:
            raise RuntimeError("CPU pressure audit counters are invalid")
        self.segments.append(
            {
                **token,
                "end": end,
                "elapsed_usec": elapsed_usec,
                "some_pressure_fraction": some_delta / elapsed_usec,
                "full_pressure_fraction": full_delta / elapsed_usec,
            }
        )


def capture_host_audit_start(repo: pathlib.Path) -> dict[str, Any]:
    global _ACTIVE_HOST_CONTENTION_AUDIT
    if _ACTIVE_HOST_CONTENTION_AUDIT is not None:
        raise RuntimeError("CPU pressure audit is already active")
    record = _cpu_host_audit_module()._capture_topology_record(repo)
    topology = record.get("normalized", {})
    if topology.get("physical_core_count") != PROFILE["cpu_required_physical_cores"]:
        raise RuntimeError(
            "fixed CPU performance baseline requires exactly "
            f"{PROFILE['cpu_required_physical_cores']} allowed physical cores; "
            "run a topology-derived CPU tuning profile for this host before "
            "defining another release profile"
        )
    tracker = _HostContentionAudit()
    _ACTIVE_HOST_CONTENTION_AUDIT = tracker
    return {"topology": record, "contention_tracker": tracker}


def complete_host_audit(
    start: dict[str, Any], samples: list[dict[str, Any]], *, repo: pathlib.Path
) -> dict[str, Any]:
    global _ACTIVE_HOST_CONTENTION_AUDIT
    module = _cpu_host_audit_module()
    topology_record = start.get("topology")
    tracker = start.get("contention_tracker")
    if (
        not isinstance(topology_record, dict)
        or not isinstance(tracker, _HostContentionAudit)
        or tracker is not _ACTIVE_HOST_CONTENTION_AUDIT
        or tracker.active
    ):
        raise RuntimeError("CPU pressure audit state is invalid")
    _ACTIVE_HOST_CONTENTION_AUDIT = None
    topology = topology_record.get("normalized", {})
    logical_cpus = topology.get("available_logical_cpus")
    end_power = module.capture_power_load_state(logical_cpus)
    record = {
        "schema_version": 1,
        "scope": PROFILE["cpu_baseline_scope"],
        "hostname": platform.node(),
        "cpu_model": topology_record.get("summary", {}).get("Model name"),
        "topology": topology_record,
        "topology_identity_sha256": canonical_sha256(
            module._topology_identity(topology_record)
        ),
        "contention_segments": tracker.segments,
        "power_load_state_end": end_power,
        "power_load_comparability": module.validate_power_load_comparability(
            topology_record["power_load_state_start"], end_power
        ),
    }
    validate_host_audit(record, samples=samples, repo=repo, verify_live=True)
    return record


def validate_host_audit(
    record: Any,
    *,
    samples: list[dict[str, Any]],
    repo: pathlib.Path,
    verify_live: bool,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "scope",
        "hostname",
        "cpu_model",
        "topology",
        "topology_identity_sha256",
        "contention_segments",
        "power_load_state_end",
        "power_load_comparability",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected_keys
        or record.get("schema_version") != 1
        or record.get("scope") != PROFILE["cpu_baseline_scope"]
        or not isinstance(record.get("hostname"), str)
        or not record["hostname"]
        or not isinstance(record.get("cpu_model"), str)
        or not record["cpu_model"]
    ):
        raise RuntimeError("CPU/GPU host audit schema is invalid")
    module = _cpu_host_audit_module()
    topology_record = record.get("topology")
    topology = topology_record.get("normalized") if isinstance(topology_record, dict) else None
    if (
        not isinstance(topology, dict)
        or topology.get("physical_core_count")
        != PROFILE["cpu_required_physical_cores"]
        or record.get("topology_identity_sha256")
        != canonical_sha256(module._topology_identity(topology_record))
        or topology_record.get("summary", {}).get("Model name") != record["cpu_model"]
    ):
        raise RuntimeError("CPU/GPU host topology identity is invalid")
    comparability = module.validate_power_load_comparability(
        topology_record.get("power_load_state_start"),
        record.get("power_load_state_end"),
    )
    if record.get("power_load_comparability") != comparability:
        raise RuntimeError("CPU/GPU host power/load comparison is stale")
    segments = record.get("contention_segments")
    expected_segment_identities = [
        (
            sample.get("artifact_label")
            or f"{sample.get('lane')}-{sample.get('sample_kind')}-{sample.get('iteration')}",
            sample.get("lane"),
            sample.get("sample_kind"),
            sample.get("iteration"),
        )
        for sample in samples
    ]
    if (
        not isinstance(segments, list)
        or any(not isinstance(segment, dict) for segment in segments)
        or [
            (
                segment.get("label"),
                segment.get("lane"),
                segment.get("sample_kind"),
                segment.get("iteration"),
            )
            for segment in segments
        ]
        != expected_segment_identities
    ):
        raise RuntimeError("CPU pressure audit sample matrix is invalid")
    for segment, sample in zip(segments, samples):
        start_pressure = segment.get("start")
        end_pressure = segment.get("end")
        elapsed_usec = segment.get("elapsed_usec")
        pressure_keys = {"monotonic_ns", "some_total_usec", "full_total_usec"}
        if (
            set(segment)
            != {
                "label",
                "lane",
                "sample_kind",
                "iteration",
                "start",
                "end",
                "elapsed_usec",
                "some_pressure_fraction",
                "full_pressure_fraction",
            }
            or not isinstance(start_pressure, dict)
            or not isinstance(end_pressure, dict)
            or set(start_pressure) != pressure_keys
            or set(end_pressure) != pressure_keys
            or isinstance(elapsed_usec, bool)
            or not isinstance(elapsed_usec, (int, float))
            or elapsed_usec <= 0
        ):
            raise RuntimeError("CPU pressure audit segment schema is invalid")
        expected_elapsed = (
            end_pressure["monotonic_ns"] - start_pressure["monotonic_ns"]
        ) / 1000.0
        some_delta = (
            end_pressure["some_total_usec"] - start_pressure["some_total_usec"]
        )
        full_delta = (
            end_pressure["full_total_usec"] - start_pressure["full_total_usec"]
        )
        if elapsed_usec != expected_elapsed or some_delta < 0 or full_delta < 0:
            raise RuntimeError("CPU pressure audit counters are invalid")
        process_usec = _finite_positive(
            sample.get("process_seconds"), "CPU pressure audited process time"
        ) * 1.0e6
        overhead_usec = elapsed_usec - process_usec
        if (
            overhead_usec < 0
            or overhead_usec
            > PROFILE["maximum_pressure_audit_overhead_seconds"] * 1.0e6
        ):
            raise RuntimeError("CPU pressure audit does not enclose the child process")
        expected_some = some_delta / elapsed_usec
        expected_full = full_delta / elapsed_usec
        if (
            segment.get("some_pressure_fraction") != expected_some
            or segment.get("full_pressure_fraction") != expected_full
            or (
                segment["lane"] == CPU_BENCHMARK_LANE
                and expected_some
                > PROFILE["maximum_cpu_pressure_some_fraction"]
            )
        ):
            raise RuntimeError("CPU baseline experienced scheduler contention")
    maximum_load = 1.25 * PROFILE["cpu_required_physical_cores"]
    if any(
        comparability[epoch]["load_1m"] > maximum_load
        for epoch in ("start_load", "end_load")
    ):
        raise RuntimeError("CPU/GPU host load exceeded the fixed comparability limit")
    cpu_samples = [sample for sample in samples if sample.get("lane") == CPU_BENCHMARK_LANE]
    if not cpu_samples:
        raise RuntimeError("CPU/GPU host audit has no CPU performance samples")
    expected_cores = {
        (core["socket"], core["core"])
        for core in topology.get("cores", [])
        if isinstance(core, dict)
    }
    for sample in cpu_samples:
        affinity = sample.get("validation", {}).get("affinity", {})
        used = {tuple(core) for core in affinity.get("used_physical_cores", [])}
        if (
            affinity.get("hostname") != record["hostname"]
            or used != expected_cores
            or len(used) != PROFILE["cpu_required_physical_cores"]
        ):
            raise RuntimeError("CPU baseline did not cover the audited host physical cores")
    for sample in samples:
        if sample.get("lane") not in CUDA_LANES:
            continue
        drivers = sample.get("validation", {}).get("rank_driver")
        if (
            not isinstance(drivers, list)
            or not drivers
            or drivers[0].get("hostname") != record["hostname"]
        ):
            raise RuntimeError("CPU and CUDA rank zero were measured on different hosts")
    if verify_live:
        live = module._capture_topology_record(repo)
        if (
            module._topology_identity(live) != module._topology_identity(topology_record)
            or live.get("summary", {}).get("Model name") != record["cpu_model"]
            or platform.node() != record["hostname"]
        ):
            raise RuntimeError("live CPU host topology changed during qualification")
    return {
        "pass": True,
        "hostname": record["hostname"],
        "cpu_model": record["cpu_model"],
        "physical_cores": PROFILE["cpu_required_physical_cores"],
        "topology_identity_sha256": record["topology_identity_sha256"],
        "maximum_cpu_pressure_some_fraction": PROFILE[
            "maximum_cpu_pressure_some_fraction"
        ],
        "maximum_observed_cpu_pressure_some_fraction": max(
            segment["some_pressure_fraction"]
            for segment in segments
            if segment["lane"] == CPU_BENCHMARK_LANE
        ),
        "power_load_comparability": comparability,
    }


def bind_host_audit(
    comparison: dict[str, Any],
    record: Any,
    *,
    samples: list[dict[str, Any]],
    repo: pathlib.Path,
    verify_live: bool,
) -> dict[str, Any]:
    if "host" in comparison or "host_topology_power_load" in comparison.get("gate", {}):
        raise RuntimeError("CPU/GPU host audit was already bound")
    host = validate_host_audit(
        record, samples=samples, repo=repo, verify_live=verify_live
    )
    comparison["host"] = host
    comparison["gate"]["host_topology_power_load"] = True
    comparison["gate"]["pass"] = all(
        value for name, value in comparison["gate"].items() if name != "pass"
    )
    return host


def _assert_fixed_profile(
    expected_object: Any = PROFILE,
    expected_sha256: str = PROFILE_CANONICAL_SHA256,
) -> None:
    if (
        PROFILE is not expected_object
        or not isinstance(PROFILE, types.MappingProxyType)
        or canonical_sha256(dict(PROFILE)) != expected_sha256
    ):
        raise RuntimeError("fixed MPI adjoint qualification profile was modified in memory")
PREFIX = "gpmeep-adjoint-benchmark:"
UNQUALIFIED_REPORT_NAME = "UNQUALIFIED_REPORT.json"
FINAL_ARTIFACTS = (
    "report.json",
    "report.md",
    "COMPLETE",
    "FAILED.json",
    UNQUALIFIED_REPORT_NAME,
)
RECORDED_ENVIRONMENT_KEYS = (
    "CUDA_CACHE_DISABLE",
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "GPMEEP_VALIDATION_STRICT_CUDA",
    "MEEP_GPU_BACKEND",
    "MEEP_GPU_MPI_COMPLETION",
    "MEEP_GPU_MPI_TRANSPORT",
    "MEEP_GPU_STRICT",
    "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS",
    "MEEP_MATERIAL_GRADIENT_PATH",
    "MEEP_MATERIAL_GRADIENT_STATS",
    "MPLBACKEND",
    "MPLCONFIGDIR",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "FONTCONFIG_FILE",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "OMPI_MCA_mca_base_component_path",
    "OMPI_MCA_mca_base_param_files",
    "PMIX_MCA_mca_base_component_path",
    "PMIX_MCA_mca_base_param_files",
    "PRTE_MCA_mca_base_param_files",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONPYCACHEPREFIX",
    "PYTHONPATH",
    "PATH",
    "HOME",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
)
FORBIDDEN_INFLUENCE_VARIABLES = {
    "LD_PRELOAD",
    "OMPI_MCA_btl",
    "OMPI_MCA_pml",
    "OMPI_MCA_mtl",
    "OMPI_MCA_coll",
    "UCX_TLS",
    "UCX_NET_DEVICES",
    "UCX_MEMTYPE_CACHE",
}
INFLUENTIAL_ENVIRONMENT_PREFIXES = (
    "MEEP_",
    "GPMEEP_",
    "CUDA_",
    "OMP_",
    "OPENBLAS_",
    "MKL_",
    "NUMEXPR_",
    "BLIS_",
    "VECLIB_",
    "UCX_",
    "NCCL_",
    "HDF5_",
    "FONTCONFIG_",
    "PYTHON",
    "LD_",
    "OMPI_MCA_",
    "PMIX_MCA_",
    "PRTE_MCA_",
)
ALLOWED_LAUNCHER_INJECTED_INFLUENCE_VARIABLES = {
    "CUDA_VISIBLE_DEVICES",
    "OMPI_MCA_PREFIXES",
    "OMPI_MCA_cpu_type",
    "OMPI_MCA_initial_wdir",
    "OMPI_MCA_num_procs",
}
LAUNCHER_CONSUMED_ENVIRONMENT_KEYS = {
    "PRTE_MCA_mca_base_param_files",
}
RUNNER_SOURCES = (
    "scripts/audit-conda-prefix.py",
    "scripts/benchmark-adjoint.py",
    "scripts/capture-runtime-dependency-closure.py",
    "scripts/capture-build-environment.py",
    "scripts/compile-conda-relocated-bytecode.py",
    "scripts/compare-mpi-adjoint-benchmarks.py",
    "scripts/fresh-environment-attestation.py",
    "scripts/gpmeep_benchmark_evidence.py",
    "scripts/gpmeep_cpu_topology_policy.py",
    "scripts/gpmeep_provenance.py",
    "scripts/gpmeep_qualification_contract.py",
    "scripts/run-cpu-parallel-benchmark.py",
    "scripts/run-mpi-adjoint-benchmark.py",
    "scripts/verify-mpi-adjoint-evidence.py",
    "scripts/verify-protected-bash.py",
    "python/tests/test_adjoint_default_material_grid.py",
)
REQUIRED_CONFIGURATION_FILES = {
    "config_h",
    "config_status",
    "environment_explicit",
    "openmpi_qualification_params",
    "prte_mca_params",
    "prte_default_hostfile",
    "runtime_dependency_closure",
    "fresh_environment_attestation",
    "conda_prefix_content_audit",
    "micromamba",
    "canonical_build_environment",
    "cuda_runtime_cmake_cache",
    "cuda_runtime_flags_stamp",
    "qualification_fontconfig",
}
REQUIRED_ARTIFACTS = {
    "python_extension",
    "libmeep",
    "installed_python_extension",
    "installed_libmeep",
    "installed_mpb_extension",
    "installed_libpympb",
    "gpu_backend_test",
    "gpu_step_db_test",
    "gpu_mpi_performance",
}
REQUIRED_MANIFESTS = {
    "build_home",
    "in_place_python",
    "installed_python",
    "installed_environment",
    "installed_prefix",
    "qualification_logs",
    "qualification_home",
}
REQUIRED_TOOLS = {
    "autoreconf",
    "c++",
    "make",
    "nvcc",
    "python",
    "swig",
    "mpicxx",
    "mpiexec",
    "h5pcc",
    "timeout",
}
REQUIRED_QUALIFICATION_LOGS = {
    "in-place-singleton-import.log",
    "in-place-two-rank-import.log",
    "cpp-mpi-test-suite.log",
    "gpu-step-db-full-waitall-two-rank.log",
    "comms-manager-waitall-failure-two-rank.log",
    "eager-pingpong-waitall-two-rank.log",
    "comms-manager-completion-policy-mismatch-two-rank.log",
    "comms-manager-invalid-completion-policy-two-rank.log",
    "python-gpu-backend-singleton.log",
    "dft-decimation-two-rank.log",
    "dft-norm-rank-failure-two-rank.log",
    "integrate2-rank-failure-two-rank.log",
    "installed-singleton-import.log",
    "installed-two-rank-import.log",
    "installed-adjoint-default-material-grid.log",
    "runtime-dependency-closure.log",
}
EXPECTED_MPI_FAILURE_LOGS = {
    "comms-manager-completion-policy-mismatch-two-rank.log": (
        "distributed CUDA ranks selected incompatible MPI completion policies"
    ),
    "comms-manager-invalid-completion-policy-two-rank.log": (
        "invalid MEEP_GPU_MPI_COMPLETION='invalid'"
    ),
    "dft-norm-rank-failure-two-rank.log": (
        "rank-local failure during a distributed DFT norm reduction"
    ),
    "integrate2-rank-failure-two-rank.log": (
        "rank-local failure during distributed integrate2"
    ),
}
FIXED_CONFIGURE_ARGUMENTS = (
    "--enable-maintainer-mode",
    "--enable-shared",
    "--enable-single",
    "--enable-cuda",
    "{cuda_arch}",
    "--with-openmp",
    "--with-mpi",
    "--with-python",
    "--without-scheme",
    "{prefix}",
    "--disable-cuda-fast-math",
)
TRUSTED_NVIDIA_SMI_DIRECTORIES = (
    pathlib.Path("/usr/bin"),
    pathlib.Path("/usr/sbin"),
    pathlib.Path("/bin"),
    pathlib.Path("/sbin"),
    pathlib.Path("/usr/local/cuda/bin"),
    pathlib.Path("/usr/lib/wsl/lib"),
)
TRUSTED_TIMEOUT_PATHS = (
    pathlib.Path("/usr/bin/timeout"),
    pathlib.Path("/bin/timeout"),
)
FD_TEST_METHODS = (
    "test_default_material_grid_legacy_matches_analytic",
    "test_projected_grid_auto_uses_legacy_semantics",
    "test_analytic_required_rejects_every_unsupported_class",
)
FD_DIRECTION_PREFIX = "gpmeep-fd-direction:"
FD_DIRECTIONS = (
    "cosine",
    "quasiperiodic-sine",
    "component-6",
    "component-12",
    "component-18",
)
EXPECTED_SAMPLE_KEYS = {
    "lane",
    "sample_kind",
    "iteration",
    "completion_policy",
    "capture_field_times",
    "artifact_label",
    "ranks",
    "nonce",
    "process_seconds",
    "result",
    "validation",
    "receipt_gate",
    "process_environment_gate",
    "lazy_import_gate",
    "material_gradient_gate",
    "result_file",
    "result_sha256",
    "stdout_log",
    "stdout_sha256",
    "stderr_log",
    "stderr_sha256",
    "timing_file",
    "timing_sha256",
    "command",
    "environment",
}


class _RunPublicationCapability:
    """One-run controller ledger used to authorize final publication.

    Entries are retained as canonical JSON bytes so later mutation of a sample,
    timing record, or caller-owned container cannot change what the controller
    observed.  The capability is sealed before validation and can authorize at
    most one publication attempt.
    """

    __slots__ = (
        "_output",
        "_run_id",
        "_measurements",
        "_artifacts",
        "_sealed",
        "_consumed",
    )

    def __init__(self, output: pathlib.Path, run_id: str):
        self._output = str(_canonical_output_directory(output))
        self._run_id = run_id
        self._measurements: dict[str, bytes] = {}
        self._artifacts: dict[str, bytes] = {}
        self._sealed = False
        self._consumed = False

    @staticmethod
    def _encode(record: dict[str, Any]) -> bytes:
        return json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    @staticmethod
    def _decode(record: bytes) -> dict[str, Any]:
        value = json.loads(record)
        if not isinstance(value, dict):
            raise RuntimeError("controller ledger record is not an object")
        return value

    def record_measurement(self, key: str, record: dict[str, Any]) -> None:
        if self._sealed or key in self._measurements:
            raise RuntimeError("controller measurement ledger is sealed or duplicated")
        if (
            record.get("run_id") != self._run_id
            or record.get("output") != self._output
        ):
            raise RuntimeError("controller measurement belongs to another run")
        self._measurements[key] = self._encode(record)

    def record_artifacts(self, key: str, record: dict[str, Any]) -> None:
        if (
            self._sealed
            or key not in self._measurements
            or key in self._artifacts
        ):
            raise RuntimeError("controller artifact ledger is incomplete or duplicated")
        self._artifacts[key] = self._encode(record)

    def seal(self) -> None:
        if self._sealed or set(self._measurements) != set(self._artifacts):
            raise RuntimeError("controller publication ledger is incomplete")
        self._sealed = True

    def snapshot(
        self, output: pathlib.Path, run_id: str, *, consume: bool = False
    ) -> dict[str, dict[str, dict[str, Any]]]:
        canonical_output = str(_canonical_output_directory(output))
        if (
            not self._sealed
            or canonical_output != self._output
            or run_id != self._run_id
        ):
            raise RuntimeError("run-instance publication capability is invalid")
        if consume:
            if self._consumed:
                raise RuntimeError("run-instance publication capability was already consumed")
            self._consumed = True
        return {
            "measurements": {
                key: self._decode(value) for key, value in self._measurements.items()
            },
            "artifacts": {
                key: self._decode(value) for key, value in self._artifacts.items()
            },
        }


def _qualification_fontconfig_text(
    environment_prefix: pathlib.Path, _build_home: pathlib.Path
) -> str:
    return (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n'
        '<fontconfig>\n'
        f'  <dir>{environment_prefix}/fonts</dir>\n'
        '  <include ignore_missing="yes">'
        f'{environment_prefix}/etc/fonts/conf.d</include>\n'
        '  <cachedir prefix="xdg">fontconfig</cachedir>\n'
        '</fontconfig>\n'
    )


class CommandTimeout(RuntimeError):
    def __init__(self, message: str, stdout: str, stderr: str):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def _load_comparator():
    path = pathlib.Path(__file__).resolve().with_name(
        "compare-mpi-adjoint-benchmarks.py"
    )
    return _load_source_module("gpmeep_mpi_comparator", path)


COMPARATOR = _load_comparator()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-receipt",
        type=pathlib.Path,
        default=repo
        / "build"
        / "meep-cuda-mpi-python-fp32"
        / "build-provenance.json",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=repo / "benchmark-results" / PROFILE_ID,
    )
    parser.add_argument("--profile", choices=(PROFILE_ID,), default=PROFILE_ID)
    return parser.parse_args(argv)


def _resolve_recorded_path(value: str, repo: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _explicit_packages(path: pathlib.Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot read explicit environment lock {path}") from error
    packages = sorted(line.strip() for line in lines if "://" in line)
    if not packages or len(packages) != len(set(packages)):
        raise RuntimeError(f"explicit environment lock is empty or duplicated: {path}")
    return packages


def _reproduce_clean_build_environment(
    *,
    repo: pathlib.Path,
    environment_prefix: pathlib.Path,
    build_dir: pathlib.Path,
    recorded_environment: dict[str, Any],
) -> dict[str, str]:
    """Re-run the pinned clean activation and capture every exported value."""

    mamba = (repo / ".tools" / "micromamba").resolve()
    mamba_root = (repo / ".micromamba").resolve()
    mamba_cache = mamba_root / "cache"
    mamba_packages = mamba_root / "pkgs"
    home = str(
        mamba_cache
        / f"build-home-{recorded_environment['GPMEEP_FRESH_ENV_NONCE']}"
    )
    openmpi_params = (
        repo / "environment" / "openmpi-qualification-mca-params.conf"
    ).resolve()
    run_environment = [
        "--clean-env",
        "--env",
        f"HOME={home}",
        "--env",
        f"XDG_CACHE_HOME={home}/cache",
        "--env",
        f"MAMBA_ROOT_PREFIX={mamba_root}",
        "--env",
        f"CONDA_PKGS_DIRS={mamba_packages}",
        "--env",
        f"MPLCONFIGDIR={home}/matplotlib",
        "--env",
        "MPLBACKEND=Agg",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONPYCACHEPREFIX=/dev/null",
        "--env",
        "PYTHONSAFEPATH=1",
        "--env",
        f"MEEP_GPU_MULTI_INITIAL_CONDITION={RELEASE_INITIAL_CONDITION}",
        "--env",
        f"MEEP_GPU_MULTI_SOURCE_PROFILE={RELEASE_SOURCE_PROFILE}",
        "--env",
        f"GPMEEP_FRESH_ENV_NONCE={recorded_environment['GPMEEP_FRESH_ENV_NONCE']}",
        "--env",
        f"OMPI_MCA_mca_base_param_files={openmpi_params}",
        "--env",
        f"OMPI_MCA_mca_base_component_path={environment_prefix / 'lib' / 'openmpi'}",
        "--env",
        f"PMIX_MCA_mca_base_param_files={openmpi_params}",
        "--env",
        f"PMIX_MCA_mca_base_component_path={environment_prefix / 'lib' / 'pmix'}",
        "--env",
        f"PRTE_MCA_mca_base_param_files={openmpi_params}",
    ]
    for name in FORWARDED_OPTIONAL_BUILD_ENVIRONMENT_KEYS:
        value = recorded_environment.get(name)
        if value is not None:
            if not isinstance(value, str) or "\x00" in value:
                raise RuntimeError(f"build receipt environment {name} is invalid")
            run_environment.extend(("--env", f"{name}={value}"))
    with tempfile.TemporaryDirectory(prefix="gpmeep-build-env-", dir="/tmp") as temporary:
        temporary_root = pathlib.Path(temporary)
        output = temporary_root / "environment.json"
        process_mamba_root = temporary_root / "micromamba-root"
        base_environment = {
            "HOME": str(temporary_root),
            "PATH": "/usr/bin:/bin",
            "XDG_CACHE_HOME": str(process_mamba_root / "cache"),
            "MAMBA_ROOT_PREFIX": str(process_mamba_root),
            "CONDA_PKGS_DIRS": str(mamba_packages),
        }
        command = [
            str(mamba),
            "--no-rc",
            "run",
            *run_environment,
            "--root-prefix",
            str(process_mamba_root),
            "--prefix",
            str(environment_prefix),
            "/bin/bash",
            "-c",
            '/usr/bin/python3 -I -E "$1" --output "$2"',
            "bash",
            str(repo / "scripts" / "capture-build-environment.py"),
            str(output),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=repo,
                env=base_environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("cannot reproduce the canonical build environment") from error
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                "cannot reproduce the canonical build environment"
                + (f": {diagnostic}" if diagnostic else "")
            )
        try:
            value = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("clean build-environment reproduction is invalid") from error
    environment = value.get("environment") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schema_version", "environment", "normalized_shell_keys"}
        or value.get("schema_version") != 2
        or value.get("normalized_shell_keys") != ["SHLVL", "_"]
        or not isinstance(environment, dict)
        or not all(
            isinstance(name, str) and isinstance(item, str)
            for name, item in environment.items()
        )
    ):
        raise RuntimeError("clean build-environment reproduction is invalid")
    return environment


def _rerun_conda_prefix_audit(
    *, repo: pathlib.Path, environment_prefix: pathlib.Path,
    lock: pathlib.Path, package_cache: pathlib.Path
) -> dict[str, Any]:
    """Re-audit installed bytes with the receipt-bound current auditor."""

    with tempfile.TemporaryDirectory(prefix="gpmeep-prefix-audit-", dir="/tmp") as temporary:
        output = pathlib.Path(temporary) / "audit.json"
        try:
            completed = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-I",
                    "-E",
                    str(repo / "scripts" / "audit-conda-prefix.py"),
                    "--prefix",
                    str(environment_prefix),
                    "--lock",
                    str(lock),
                    "--package-cache",
                    str(package_cache),
                    "--output",
                    str(output),
                ],
                cwd=repo,
                env={"HOME": "/tmp", "PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("cannot rerun the conda-prefix byte audit") from error
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                "current conda-prefix byte audit failed"
                + (f": {diagnostic}" if diagnostic else "")
            )
        try:
            value = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("current conda-prefix byte audit is invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError("current conda-prefix byte audit is invalid")
    return value


def validate_build_receipt_contract(
    receipt: dict[str, Any],
    repo: pathlib.Path,
    *,
    hash_cache: Any | None = None,
) -> dict[str, Any]:
    configuration = receipt.get("configuration")
    configuration_files = receipt.get("configuration_files")
    artifacts = receipt.get("artifacts")
    manifests = receipt.get("manifests")
    toolchain = receipt.get("toolchain")
    if not all(
        isinstance(value, dict)
        for value in (
            configuration,
            configuration_files,
            artifacts,
            manifests,
            toolchain,
        )
    ):
        raise RuntimeError("build receipt qualification closure is incomplete")
    if configuration.get("qualification_contract") != BUILD_QUALIFICATION_CONTRACT:
        raise RuntimeError("build receipt has the wrong qualification contract")
    builder = configuration.get("builder")
    expected_builder = (repo / "scripts" / "build-meep-cuda-mpi-python.sh").resolve()
    if (
        not isinstance(builder, dict)
        or not isinstance(builder.get("path"), str)
        or _resolve_recorded_path(builder["path"], repo) != expected_builder
    ):
        raise RuntimeError("build receipt was not produced by the fixed builder")

    configure_argv = configuration.get("configure_argv")
    if not isinstance(configure_argv, list) or not all(
        isinstance(value, str) for value in configure_argv
    ):
        raise RuntimeError("build receipt configure argument list is invalid")
    cuda_arch_args = [
        value for value in configure_argv if value.startswith("--with-cuda-arch=")
    ]
    prefix_args = [value for value in configure_argv if value.startswith("--prefix=")]
    if len(cuda_arch_args) != 1 or len(prefix_args) != 1:
        raise RuntimeError("build receipt CUDA architecture/install prefix is ambiguous")
    cuda_arch = cuda_arch_args[0]
    if re.fullmatch(r"--with-cuda-arch=[A-Za-z0-9_.+,;-]+", cuda_arch) is None:
        raise RuntimeError("build receipt CUDA architecture is malformed")
    expected_install = (repo / "install" / "meep-cuda-mpi-python-fp32").resolve()
    if pathlib.Path(prefix_args[0].split("=", 1)[1]).resolve() != expected_install:
        raise RuntimeError("build receipt install prefix is not the fixed release prefix")
    expected_configure = [
        value.format(cuda_arch=cuda_arch, prefix=f"--prefix={expected_install}")
        for value in FIXED_CONFIGURE_ARGUMENTS
    ]
    if configure_argv != expected_configure:
        raise RuntimeError(
            "build receipt configure arguments are not the exact CUDA/MPI/Python FP32 contract"
        )

    named_groups = (
        ("configuration files", configuration_files, REQUIRED_CONFIGURATION_FILES),
        ("artifacts", artifacts, REQUIRED_ARTIFACTS),
        ("manifests", manifests, REQUIRED_MANIFESTS),
        ("toolchain", toolchain, REQUIRED_TOOLS),
    )
    for label, records, required in named_groups:
        missing = sorted(required - set(records))
        if missing:
            raise RuntimeError(f"build receipt {label} missing: {', '.join(missing)}")

    try:
        _BUILD_QUALIFICATION.validate_release_cuda_math_policy(receipt, repo)
    except _BUILD_QUALIFICATION.QualificationContractError as error:
        raise RuntimeError(f"build receipt CUDA math policy is invalid: {error}") from error

    micromamba_record = configuration_files["micromamba"]
    expected_micromamba = (repo / ".tools" / "micromamba").resolve()
    if (
        not isinstance(micromamba_record, dict)
        or not isinstance(micromamba_record.get("path"), str)
        or _resolve_recorded_path(micromamba_record["path"], repo).resolve()
        != expected_micromamba
        or micromamba_record.get("sha256") != PINNED_MICROMAMBA_SHA256
    ):
        raise RuntimeError("build receipt Micromamba binary is not the pinned tool")

    config_h_record = configuration_files["config_h"]
    config_status_record = configuration_files["config_status"]
    openmpi_params_record = configuration_files["openmpi_qualification_params"]
    prte_params_record = configuration_files["prte_mca_params"]
    prte_hostfile_record = configuration_files["prte_default_hostfile"]
    if not all(
        isinstance(record, dict) and isinstance(record.get("path"), str)
        for record in (
            config_h_record,
            config_status_record,
            openmpi_params_record,
            prte_params_record,
            prte_hostfile_record,
        )
    ):
        raise RuntimeError("build receipt configure/MPI parameter records are invalid")
    config_h_path = _resolve_recorded_path(config_h_record["path"], repo)
    config_status_path = _resolve_recorded_path(config_status_record["path"], repo)
    expected_openmpi_params = (
        repo / "environment" / "openmpi-qualification-mca-params.conf"
    ).resolve()
    if _resolve_recorded_path(openmpi_params_record["path"], repo).resolve() != (
        expected_openmpi_params
    ):
        raise RuntimeError("build receipt uses the wrong Open MPI parameter file")
    try:
        config_h = config_h_path.read_text(encoding="utf-8")
        config_status = config_status_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError("cannot read receipt-bound configure outputs") from error
    for macro in ("HAVE_CUDA", "HAVE_MPI", "HAVE_OPENMP"):
        if re.search(rf"^#define {macro} 1$", config_h, re.MULTILINE) is None:
            raise RuntimeError(f"receipt-bound config.h does not enable {macro}")
    if re.search(r"^#define HAVE_LIBCTL(?:\s|$)", config_h, re.MULTILINE):
        raise RuntimeError("receipt-bound config.h unexpectedly enables Scheme/libctl")
    configure_line = next(
        (line.strip() for line in config_status.splitlines() if line.startswith("  set X /bin/bash")),
        None,
    )
    if configure_line is None:
        raise RuntimeError("receipt-bound config.status lacks the configure invocation")
    try:
        status_tokens = shlex.split(configure_line)
    except ValueError as error:
        raise RuntimeError("receipt-bound config.status configure invocation is malformed") from error
    if len(status_tokens) < 5 or status_tokens[:3] != ["set", "X", "/bin/bash"]:
        raise RuntimeError("receipt-bound config.status configure invocation is malformed")
    status_configure: list[str] = []
    for token in status_tokens[4:]:
        if not token.startswith("--"):
            break
        status_configure.append(token)
    if status_configure != configure_argv:
        raise RuntimeError("receipt and config.status configure arguments disagree")
    required_status = {
        'S["MEEP_HAVE_CUDA"]="1"',
        'S["MEEP_SINGLE"]="1"',
        'S["WITH_PYTHON_FALSE"]="#"',
        'S["WITH_PYTHON_TRUE"]=""',
        'S["WITH_SCHEME_FALSE"]=""',
        'S["WITH_SCHEME_TRUE"]="#"',
        'D["HAVE_MPI"]=" 1"',
        'D["HAVE_CUDA"]=" 1"',
        'D["HAVE_OPENMP"]=" 1"',
    }
    status_lines = {line.strip() for line in config_status.splitlines()}
    missing_status = sorted(required_status - status_lines)
    if missing_status:
        raise RuntimeError(
            "receipt-bound config.status feature state is incomplete: "
            + ", ".join(missing_status)
        )

    environment_record = configuration_files["environment_explicit"]
    lock_record = configuration.get("lockfiles", {}).get("environment_lock")
    if not isinstance(environment_record, dict) or not isinstance(
        environment_record.get("path"), str
    ):
        raise RuntimeError("build receipt explicit environment record is invalid")
    if not isinstance(lock_record, dict) or not isinstance(lock_record.get("path"), str):
        raise RuntimeError("build receipt canonical environment lock is invalid")
    environment_path = _resolve_recorded_path(environment_record["path"], repo)
    lock_path = _resolve_recorded_path(lock_record["path"], repo)
    if _explicit_packages(environment_path) != _explicit_packages(lock_path):
        raise RuntimeError("receipt environment does not exactly match its lock")

    qualification_manifest = manifests["qualification_logs"]
    files = qualification_manifest.get("files") if isinstance(
        qualification_manifest, dict
    ) else None
    if not isinstance(files, list):
        raise RuntimeError("build receipt qualification-log manifest is invalid")
    by_name = {
        record.get("path"): record
        for record in files
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    if (
        len(by_name) != len(files)
        or any(pathlib.PurePosixPath(name).name != name for name in by_name)
    ):
        raise RuntimeError(
            "build receipt qualification-log manifest has duplicate or nested paths"
        )
    missing_logs = sorted(REQUIRED_QUALIFICATION_LOGS - set(by_name))
    if missing_logs:
        raise RuntimeError(
            "build receipt qualification logs missing: " + ", ".join(missing_logs)
        )
    log_root_value = qualification_manifest.get("root")
    if not isinstance(log_root_value, str):
        raise RuntimeError("build receipt qualification-log root is invalid")
    log_root = _resolve_recorded_path(log_root_value, repo)
    for name in REQUIRED_QUALIFICATION_LOGS:
        marker = f"gpmeep-qualification:{name}:PASS"
        try:
            content = (log_root / name).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as error:
            raise RuntimeError(f"cannot read qualification log {name}") from error
        if marker not in content.splitlines():
            raise RuntimeError(f"qualification log {name} lacks its PASS marker")
        expected_diagnostic = EXPECTED_MPI_FAILURE_LOGS.get(name)
        if expected_diagnostic is not None:
            prefix = f"gpmeep-expected-mpi-failure:{name}:status="
            attestations = [line for line in content.splitlines() if line.startswith(prefix)]
            if len(attestations) != 1:
                raise RuntimeError(
                    f"qualification log {name} lacks one expected-failure attestation"
                )
            status_text, separator, diagnostic = attestations[0][len(prefix) :].partition(
                ":diagnostic="
            )
            try:
                status = int(status_text)
            except ValueError as error:
                raise RuntimeError(
                    f"qualification log {name} has an invalid failure status"
                ) from error
            if (
                not separator
                or not 1 <= status <= 123
                or diagnostic != expected_diagnostic
                or not any(
                    expected_diagnostic in line
                    for line in content.splitlines()
                    if not line.startswith(prefix)
                )
            ):
                raise RuntimeError(
                    f"qualification log {name} failed for an unbound reason"
                )

    qualification_home_manifest = manifests["qualification_home"]
    build_dir_value = receipt.get("build_dir")
    if (
        not isinstance(qualification_home_manifest, dict)
        or not isinstance(qualification_home_manifest.get("root"), str)
        or not isinstance(build_dir_value, str)
    ):
        raise RuntimeError("build receipt isolated qualification HOME is invalid")
    qualification_home = _resolve_recorded_path(
        qualification_home_manifest["root"], repo
    ).resolve()
    expected_qualification_home = (
        pathlib.Path(build_dir_value).resolve() / "qualification-home"
    )
    home_files = qualification_home_manifest.get("files")
    if (
        qualification_home != expected_qualification_home
        or not isinstance(home_files, list)
        or [record.get("path") for record in home_files if isinstance(record, dict)]
        != [".gpmeep-empty-home"]
        or sorted(path.name for path in qualification_home.iterdir())
        != [".gpmeep-empty-home"]
        or (qualification_home / ".gpmeep-empty-home").read_text(encoding="utf-8")
        != "gpmeep isolated MPI qualification home\n"
    ):
        raise RuntimeError("build receipt qualification HOME was contaminated")

    python_record = toolchain["python"]
    if not isinstance(python_record, dict) or not isinstance(
        python_record.get("path"), str
    ):
        raise RuntimeError("build receipt Python tool record is invalid")
    environment_prefix = _resolve_recorded_path(python_record["path"], repo).parent.parent
    installed_environment_manifest = manifests["installed_environment"]
    if (
        not isinstance(installed_environment_manifest, dict)
        or not isinstance(installed_environment_manifest.get("root"), str)
        or _resolve_recorded_path(
            installed_environment_manifest["root"], repo
        ).resolve()
        != environment_prefix.resolve()
        or not isinstance(installed_environment_manifest.get("files"), list)
        or not installed_environment_manifest["files"]
    ):
        raise RuntimeError("build receipt installed-environment manifest is invalid")
    installed_prefix_manifest = manifests["installed_prefix"]
    installed_libmeep_record = artifacts["installed_libmeep"]
    if (
        not isinstance(installed_prefix_manifest, dict)
        or not isinstance(installed_prefix_manifest.get("root"), str)
        or not isinstance(installed_prefix_manifest.get("files"), list)
        or not installed_prefix_manifest["files"]
        or not isinstance(installed_libmeep_record, dict)
        or not isinstance(installed_libmeep_record.get("path"), str)
    ):
        raise RuntimeError("build receipt installed-prefix manifest is invalid")
    installed_prefix = _resolve_recorded_path(
        installed_prefix_manifest["root"], repo
    ).resolve()
    expected_installed_prefix = _resolve_recorded_path(
        installed_libmeep_record["path"], repo
    ).resolve().parent.parent
    if installed_prefix != expected_installed_prefix:
        raise RuntimeError("build receipt installed-prefix root is wrong")
    for artifact_name in (
        "installed_python_extension",
        "installed_libmeep",
        "installed_mpb_extension",
        "installed_libpympb",
    ):
        record = artifacts[artifact_name]
        try:
            _resolve_recorded_path(record["path"], repo).resolve().relative_to(
                installed_prefix
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"build receipt {artifact_name} is outside the install prefix"
            ) from error
    expected_build_environment = {
        "OMPI_MCA_mca_base_param_files": str(expected_openmpi_params),
        "OMPI_MCA_mca_base_component_path": str(environment_prefix / "lib" / "openmpi"),
        "PMIX_MCA_mca_base_param_files": str(expected_openmpi_params),
        "PMIX_MCA_mca_base_component_path": str(environment_prefix / "lib" / "pmix"),
        "PRTE_MCA_mca_base_param_files": str(expected_openmpi_params),
    }
    recorded_build_environment = configuration.get("environment")
    if (
        not isinstance(recorded_build_environment, dict)
        or set(recorded_build_environment) != RECORDED_BUILD_ENVIRONMENT_KEYS
        or any(
            recorded_build_environment.get(name) != value
            for name, value in expected_build_environment.items()
        )
    ):
        raise RuntimeError("build receipt lacks the fixed Open MPI environment")
    fresh_nonce = recorded_build_environment.get("GPMEEP_FRESH_ENV_NONCE")
    if not isinstance(fresh_nonce, str) or re.fullmatch(
        r"[0-9a-f]{64}", fresh_nonce
    ) is None:
        raise RuntimeError("build receipt fresh-environment nonce is invalid")

    canonical_environment_record = configuration_files[
        "canonical_build_environment"
    ]
    if not isinstance(canonical_environment_record, dict) or not isinstance(
        canonical_environment_record.get("path"), str
    ):
        raise RuntimeError("build receipt canonical build environment is absent")
    canonical_environment_path = _resolve_recorded_path(
        canonical_environment_record["path"], repo
    ).resolve()
    expected_canonical_environment_path = (
        pathlib.Path(build_dir_value).resolve()
        / "canonical-build-environment.json"
    )
    if canonical_environment_path != expected_canonical_environment_path:
        raise RuntimeError("build receipt canonical build-environment path is wrong")
    try:
        canonical_environment_value = json.loads(
            canonical_environment_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read canonical build environment") from error
    if (
        not isinstance(canonical_environment_value, dict)
        or set(canonical_environment_value)
        != {"schema_version", "environment", "normalized_shell_keys"}
        or canonical_environment_value.get("schema_version") != 2
        or canonical_environment_value.get("normalized_shell_keys")
        != ["SHLVL", "_"]
        or not isinstance(canonical_environment_value.get("environment"), dict)
    ):
        raise RuntimeError("build receipt canonical build environment is invalid")
    canonical_environment = canonical_environment_value["environment"]
    if not all(
        isinstance(name, str)
        and name
        and "\x00" not in name
        and isinstance(value, str)
        and "\x00" not in value
        for name, value in canonical_environment.items()
    ):
        raise RuntimeError("canonical build environment contains invalid entries")
    unexpected_environment = sorted(
        set(canonical_environment) - ALLOWED_CANONICAL_BUILD_ENVIRONMENT_KEYS
    )
    if unexpected_environment:
        raise RuntimeError(
            "canonical build environment contains unapproved variables: "
            + ", ".join(unexpected_environment)
        )
    required_environment = {
        "CC",
        "CC_FOR_BUILD",
        "CFLAGS",
        "CPPFLAGS",
        "CXX",
        "CXX_FOR_BUILD",
        "CXXFLAGS",
        "F77",
        "FC",
        "FC_FOR_BUILD",
        "FFLAGS",
        "HOME",
        "LDFLAGS",
        "MAMBA_EXE",
        "MAMBA_ROOT_PREFIX",
        "MEEP_GPU_MULTI_INITIAL_CONDITION",
        "MEEP_GPU_MULTI_SOURCE_PROFILE",
        "MPLBACKEND",
        "MPLCONFIGDIR",
        "NVCC_PREPEND_FLAGS",
        "PATH",
        "PWD",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PYTHONPYCACHEPREFIX",
        "PYTHONSAFEPATH",
        "SHLVL",
        "CONDA_DEFAULT_ENV",
        "CONDA_PKGS_DIRS",
        "CONDA_PREFIX",
        "CONDA_SHLVL",
        "GPMEEP_FRESH_ENV_NONCE",
        "_",
        "XDG_CACHE_HOME",
    }
    missing_environment = sorted(required_environment - set(canonical_environment))
    if missing_environment:
        raise RuntimeError(
            "canonical build environment is incomplete: "
            + ", ".join(missing_environment)
        )
    expected_compilers = {
        "CC": environment_prefix / "bin" / "x86_64-conda-linux-gnu-cc",
        "CC_FOR_BUILD": environment_prefix
        / "bin"
        / "x86_64-conda-linux-gnu-cc",
        "CXX": environment_prefix / "bin" / "x86_64-conda-linux-gnu-c++",
        "CXX_FOR_BUILD": environment_prefix
        / "bin"
        / "x86_64-conda-linux-gnu-c++",
        "F77": environment_prefix
        / "bin"
        / "x86_64-conda-linux-gnu-gfortran",
        "FC": environment_prefix / "bin" / "x86_64-conda-linux-gnu-gfortran",
        "FC_FOR_BUILD": environment_prefix
        / "bin"
        / "x86_64-conda-linux-gnu-gfortran",
    }
    if any(
        canonical_environment.get(name) != str(path)
        for name, path in expected_compilers.items()
    ):
        raise RuntimeError("canonical build environment uses the wrong compilers")
    expected_mamba_root = (repo / ".micromamba").resolve()
    expected_path = ":".join(
        str(path)
        for path in (
            environment_prefix / "bin",
            expected_mamba_root / "condabin",
            pathlib.Path("/usr/local/sbin"),
            pathlib.Path("/usr/local/bin"),
            pathlib.Path("/usr/sbin"),
            pathlib.Path("/usr/bin"),
            pathlib.Path("/sbin"),
            pathlib.Path("/bin"),
        )
    )
    expected_build_home = (
        expected_mamba_root / "cache" / f"build-home-{fresh_nonce}"
    )
    fixed_environment_values = {
        "CONDA_DEFAULT_ENV": str(environment_prefix),
        "CONDA_PKGS_DIRS": str(expected_mamba_root / "pkgs"),
        "CONDA_PREFIX": str(environment_prefix),
        "CONDA_SHLVL": "1",
        "HOME": str(expected_build_home),
        "MAMBA_EXE": str(expected_micromamba),
        "MAMBA_ROOT_PREFIX": str(expected_mamba_root),
        "MEEP_GPU_MULTI_INITIAL_CONDITION": RELEASE_INITIAL_CONDITION,
        "MEEP_GPU_MULTI_SOURCE_PROFILE": RELEASE_SOURCE_PROFILE,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(expected_build_home / "matplotlib"),
        "PATH": expected_path,
        "PWD": str(repo.resolve()),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": "/dev/null",
        "PYTHONSAFEPATH": "1",
        "SHLVL": "<shell-managed>",
        "_": "<shell-managed>",
        "XDG_CACHE_HOME": str(expected_build_home / "cache"),
    }
    if any(
        canonical_environment.get(name) != value
        for name, value in fixed_environment_values.items()
    ):
        raise RuntimeError("canonical build environment is not the clean worker environment")
    build_home_manifest = manifests["build_home"]
    build_home = pathlib.Path(canonical_environment["HOME"]).resolve()
    build_home_files = (
        build_home_manifest.get("files")
        if isinstance(build_home_manifest, dict)
        else None
    )
    marker_name = ".gpmeep-empty-build-home"
    try:
        build_home_stat = build_home.stat()
        build_home_marker = (build_home / marker_name).read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError("cannot read isolated build HOME") from error
    if (
        not isinstance(build_home_manifest, dict)
        or not isinstance(build_home_manifest.get("root"), str)
        or _resolve_recorded_path(build_home_manifest["root"], repo).resolve()
        != build_home
        or not isinstance(build_home_files, list)
        or marker_name
        not in {
            record.get("path")
            for record in build_home_files
            if isinstance(record, dict)
        }
        or not build_home.is_dir()
        or stat.S_IMODE(build_home_stat.st_mode) != 0o700
        or build_home_stat.st_uid != os.getuid()
        or build_home_marker != "gpmeep isolated build home\n"
    ):
        raise RuntimeError("build receipt isolated build HOME is invalid")
    qualification_fontconfig_record = configuration_files[
        "qualification_fontconfig"
    ]
    expected_qualification_fontconfig = (
        pathlib.Path(build_dir_value).resolve()
        / "qualification-fontconfig.conf"
    )
    if (
        not isinstance(qualification_fontconfig_record, dict)
        or not isinstance(qualification_fontconfig_record.get("path"), str)
        or _resolve_recorded_path(
            qualification_fontconfig_record["path"], repo
        ).resolve()
        != expected_qualification_fontconfig
        or not (expected_build_home / "cache" / "fontconfig").is_dir()
    ):
        raise RuntimeError("build receipt qualification fontconfig is invalid")
    try:
        qualification_fontconfig = expected_qualification_fontconfig.read_text(
            encoding="utf-8"
        )
    except OSError as error:
        raise RuntimeError("cannot read qualification fontconfig") from error
    if qualification_fontconfig != _qualification_fontconfig_text(
        environment_prefix.resolve(), expected_build_home.resolve()
    ):
        raise RuntimeError("build receipt qualification fontconfig is not isolated")
    expected_nvcc_prepend = [
        f"-ccbin={environment_prefix / 'bin' / 'x86_64-conda-linux-gnu-c++'}"
    ]
    if shlex.split(canonical_environment["NVCC_PREPEND_FLAGS"]) != expected_nvcc_prepend:
        raise RuntimeError("canonical build environment changed NVCC implicit flags")
    for name in ("CFLAGS", "CPPFLAGS", "CXXFLAGS", "FFLAGS", "LDFLAGS"):
        value = canonical_environment[name]
        optimization_tokens = (
            {"-O2", "-Wl,-O2"} if name == "LDFLAGS" else {"-O2"}
        )
        if (
            optimization_tokens.isdisjoint(shlex.split(value))
            or str(environment_prefix) not in value
            or "-ffast-math" in value
            or "-Ofast" in value
        ):
            raise RuntimeError(f"canonical build environment has unsafe {name}")
    if any(
        recorded_build_environment.get(name) != canonical_environment.get(name)
        for name in recorded_build_environment
    ):
        raise RuntimeError("receipt summary and canonical build environment disagree")
    requested_fast_math = canonical_environment.get("MEEP_GPU_FAST_MATH", "OFF")
    if (
        requested_fast_math != "OFF"
        or configuration.get("environment", {}).get("MEEP_GPU_FAST_MATH") != "OFF"
        or configure_argv.count("--disable-cuda-fast-math") != 1
        or "--enable-cuda-fast-math" in configure_argv
    ):
        raise RuntimeError(
            "canonical build environment and fast-math OFF contract disagree"
        )
    requested_cuda_arch = canonical_environment.get("MEEP_GPU_CUDA_ARCHS", "AUTO")
    if cuda_arch != f"--with-cuda-arch={requested_cuda_arch}":
        raise RuntimeError("canonical build environment and CUDA architecture disagree")
    reproduced_environment = _reproduce_clean_build_environment(
        repo=repo.resolve(),
        environment_prefix=environment_prefix.resolve(),
        build_dir=pathlib.Path(build_dir_value).resolve(),
        recorded_environment=recorded_build_environment,
    )
    if canonical_environment != reproduced_environment:
        raise RuntimeError(
            "canonical build environment differs from a fresh clean activation"
        )
    fresh_record = configuration_files["fresh_environment_attestation"]
    if (
        not isinstance(fresh_record, dict)
        or not isinstance(fresh_record.get("path"), str)
    ):
        raise RuntimeError("build receipt fresh-environment attestation is absent")
    fresh_path = _resolve_recorded_path(fresh_record["path"], repo).resolve()
    expected_fresh_path = (
        pathlib.Path(build_dir_value).resolve() / f"cuda-mpi-{fresh_nonce}.json"
    )
    try:
        fresh_value = json.loads(fresh_path.read_text(encoding="utf-8"))
        fresh_mode = stat.S_IMODE(fresh_path.stat().st_mode)
        started_at = dt.datetime.fromisoformat(
            str(receipt.get("started_at_utc", "")).replace("Z", "+00:00")
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError("cannot validate fresh-environment attestation") from error
    fresh_created_ns = fresh_value.get("created_at_unix_ns") if isinstance(
        fresh_value, dict
    ) else None
    fresh_created = (
        dt.datetime.fromtimestamp(fresh_created_ns / 1.0e9, tz=dt.timezone.utc)
        if isinstance(fresh_created_ns, int)
        else None
    )
    if (
        fresh_path != expected_fresh_path
        or fresh_mode != 0o600
        or not isinstance(fresh_value, dict)
        or set(fresh_value)
        != {
            "schema_version",
            "kind",
            "nonce",
            "created_at_unix_ns",
            "creator_pid",
            "repo",
            "environment_prefix",
            "lock",
            "explicit_package_count",
            "environment_manifest",
        }
        or fresh_value.get("schema_version") != 1
        or fresh_value.get("kind") != "fresh-exact-lock-environment"
        or fresh_value.get("nonce") != fresh_nonce
        or fresh_value.get("repo") != str(repo.resolve())
        or fresh_value.get("environment_prefix") != str(environment_prefix.resolve())
        or fresh_value.get("lock") != lock_record
        or fresh_value.get("explicit_package_count")
        != len(_explicit_packages(lock_path))
        or fresh_value.get("environment_manifest") != installed_environment_manifest
        or not isinstance(fresh_value.get("creator_pid"), int)
        or fresh_value["creator_pid"] <= 0
        or fresh_created is None
        or fresh_created > started_at
        or (started_at - fresh_created).total_seconds() > 900
    ):
        raise RuntimeError("build receipt fresh-environment attestation is invalid")
    prefix_audit_record = configuration_files["conda_prefix_content_audit"]
    if not isinstance(prefix_audit_record, dict) or not isinstance(
        prefix_audit_record.get("path"), str
    ):
        raise RuntimeError("build receipt conda-prefix byte audit is absent")
    prefix_audit_path = _resolve_recorded_path(
        prefix_audit_record["path"], repo
    ).resolve()
    if prefix_audit_path != (
        pathlib.Path(build_dir_value).resolve() / "conda-prefix-content-audit.json"
    ):
        raise RuntimeError("build receipt conda-prefix byte-audit path is wrong")
    try:
        prefix_audit = json.loads(prefix_audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read conda-prefix byte audit") from error
    expected_package_records = sorted(
        (
            {
                "url": package.rpartition("#")[0],
                "sha256": package.rpartition("#")[2],
            }
            for package in _explicit_packages(lock_path)
        ),
        key=lambda record: record["url"],
    )
    if (
        not isinstance(prefix_audit, dict)
        or prefix_audit.get("schema_version") != 4
        or prefix_audit.get("pass") is not True
        or prefix_audit.get("environment_prefix") != str(environment_prefix.resolve())
        or prefix_audit.get("lock")
        != {
            "path": str(lock_path.resolve()),
            "package_count": len(expected_package_records),
            "sha256": lock_record.get("sha256"),
        }
        or prefix_audit.get("metadata_package_count")
        != len(expected_package_records)
        or not isinstance(prefix_audit.get("package_cache"), dict)
        or prefix_audit["package_cache"].get("archive_count")
        != len(expected_package_records)
        or not isinstance(prefix_audit["package_cache"].get("path"), str)
        or not pathlib.Path(prefix_audit["package_cache"]["path"]).is_absolute()
        or prefix_audit.get("archive_snapshot")
        != {
            "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
            "hash_and_parse_same_private_snapshot": True,
            "descriptor_retained_until_terminal_verification": True,
            "pathname_fingerprint_rechecked": True,
        }
        or not isinstance(prefix_audit.get("archive_decoder"), dict)
        or prefix_audit["archive_decoder"].get("path") != "/usr/bin/zstd"
        or prefix_audit["archive_decoder"].get("timeout_seconds") != 120
        or not isinstance(prefix_audit.get("archive_bound_hardlink_count"), int)
        or prefix_audit["archive_bound_hardlink_count"] <= 0
        or not isinstance(prefix_audit.get("relocated_hardlink_count"), int)
        or prefix_audit["relocated_hardlink_count"] < 0
        or not isinstance(prefix_audit.get("relocated_source_bytecode"), list)
        or not prefix_audit["relocated_source_bytecode"]
        or prefix_audit.get("relocated_source_bytecode_count")
        != len(prefix_audit["relocated_source_bytecode"])
        or any(
            not isinstance(item, dict)
            or item.get("current_state") != "relocated-source-compiled"
            or item.get("current_sha256") != item.get("derived_sha256")
            for item in prefix_audit["relocated_source_bytecode"]
        )
        or not isinstance(prefix_audit.get("generated_source_bytecode"), list)
        or not prefix_audit["generated_source_bytecode"]
        or prefix_audit.get("generated_source_bytecode_count")
        != len(prefix_audit["generated_source_bytecode"])
        or any(
            not isinstance(item, dict)
            or item.get("current_state") != "source-compiled"
            or item.get("current_sha256") != item.get("derived_sha256")
            for item in prefix_audit["generated_source_bytecode"]
        )
        or prefix_audit.get("package_records") != expected_package_records
        or isinstance(prefix_audit.get("hashed_installed_path_count"), bool)
        or not isinstance(prefix_audit.get("hashed_installed_path_count"), int)
        or prefix_audit["hashed_installed_path_count"] <= 0
        or not isinstance(prefix_audit.get("owned_path_count"), int)
        or prefix_audit["owned_path_count"]
        < prefix_audit["hashed_installed_path_count"]
        or not isinstance(prefix_audit.get("generated_entry_points"), list)
        or not isinstance(prefix_audit.get("softlinks"), list)
        or prefix_audit.get("softlink_count") != len(prefix_audit["softlinks"])
    ):
        raise RuntimeError("build receipt conda-prefix byte audit is invalid")
    current_prefix_audit = _rerun_conda_prefix_audit(
        repo=repo.resolve(),
        environment_prefix=environment_prefix.resolve(),
        lock=lock_path.resolve(),
        package_cache=pathlib.Path(prefix_audit["package_cache"]["path"]).resolve(),
    )
    if current_prefix_audit != prefix_audit:
        raise RuntimeError(
            "receipt conda-prefix audit differs from the current protected re-audit"
        )
    expected_prte_files = {
        "prte_mca_params": environment_prefix / "etc" / "prte-mca-params.conf",
        "prte_default_hostfile": environment_prefix / "etc" / "prte-default-hostfile",
    }
    for name, expected_path in expected_prte_files.items():
        record = configuration_files[name]
        if _resolve_recorded_path(record["path"], repo).resolve() != expected_path.resolve():
            raise RuntimeError(f"build receipt {name} uses the wrong PRRTE file")
    dependency_record = configuration_files["runtime_dependency_closure"]
    if not isinstance(dependency_record, dict) or not isinstance(
        dependency_record.get("path"), str
    ):
        raise RuntimeError("build receipt runtime-dependency closure is invalid")
    dependency_path = _resolve_recorded_path(dependency_record["path"], repo).resolve()
    expected_dependency_path = (
        pathlib.Path(build_dir_value).resolve() / "runtime-dependency-closure.json"
    )
    if dependency_path != expected_dependency_path:
        raise RuntimeError("build receipt runtime-dependency closure path is wrong")
    try:
        dependency_closure = json.loads(dependency_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read build runtime-dependency closure") from error
    COMPARATOR.validate_captured_runtime_dependency_closure(
        dependency_closure,
        environment_prefix,
        _resolve_recorded_path(
            artifacts["installed_libmeep"]["path"], repo
        ).resolve().parent.parent,
        hash_cache=hash_cache,
    )
    for override_name in (
        "openmpi-mca-params-override.conf",
        "pmix-mca-params-override.conf",
        "prte-mca-params-override.conf",
    ):
        if (environment_prefix / "etc" / override_name).exists():
            raise RuntimeError("MPI override parameter file appeared after the build")
    for name in REQUIRED_TOOLS:
        record = toolchain[name]
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError(f"build receipt tool {name} is invalid")
        tool_path = _resolve_recorded_path(record["path"], repo)
        if name == "timeout":
            trusted_timeouts = {
                candidate.resolve()
                for candidate in TRUSTED_TIMEOUT_PATHS
                if candidate.is_file()
            }
            if tool_path.resolve() not in trusted_timeouts:
                raise RuntimeError("build receipt timeout is not a protected system tool")
        else:
            try:
                tool_path.relative_to(environment_prefix)
            except ValueError as error:
                raise RuntimeError(
                    f"build receipt tool {name} is outside the isolated environment"
                ) from error

    return {
        "pass": True,
        "contract": BUILD_QUALIFICATION_CONTRACT,
        "builder": str(expected_builder),
        "environment_prefix": str(environment_prefix),
        "openmpi_qualification_params": str(expected_openmpi_params),
        # The smaller required set above identifies the release-critical logs
        # whose contents receive additional semantic checks here.  Report the
        # complete receipt-bound directory, including centrally validated API
        # and materialization evidence, instead of presenting that subset as
        # if it were the entire build qualification.
        "qualification_logs": sorted(by_name),
    }


def _receipt_runtime(
    receipt: dict[str, Any], repo: pathlib.Path
) -> dict[str, pathlib.Path]:
    artifacts = receipt.get("artifacts")
    manifests = receipt.get("manifests")
    toolchain = receipt.get("toolchain")
    configuration_files = receipt.get("configuration_files")
    if not all(
        isinstance(value, dict)
        for value in (artifacts, manifests, toolchain, configuration_files)
    ):
        raise RuntimeError("build receipt runtime closure is incomplete")

    def artifact(name: str) -> pathlib.Path:
        record = artifacts.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError(f"build receipt artifact {name!r} is absent")
        return _resolve_recorded_path(record["path"], repo)

    installed = manifests.get("installed_python")
    if not isinstance(installed, dict) or not isinstance(installed.get("root"), str):
        raise RuntimeError("build receipt installed-Python manifest is absent")
    package_root = _resolve_recorded_path(installed["root"], repo)
    python_record = toolchain.get("python")
    mpiexec_record = toolchain.get("mpiexec")
    fontconfig_record = configuration_files.get("qualification_fontconfig")
    if not isinstance(python_record, dict) or not isinstance(
        python_record.get("path"), str
    ):
        raise RuntimeError("build receipt Python tool is absent")
    if not isinstance(mpiexec_record, dict) or not isinstance(
        mpiexec_record.get("path"), str
    ):
        raise RuntimeError("build receipt mpiexec tool is absent")
    if not isinstance(fontconfig_record, dict) or not isinstance(
        fontconfig_record.get("path"), str
    ):
        raise RuntimeError("build receipt qualification fontconfig is absent")
    runtime = {
        "python": _resolve_recorded_path(python_record["path"], repo).resolve(),
        "mpiexec": _resolve_recorded_path(mpiexec_record["path"], repo).resolve(),
        "package_root": package_root,
        "package_parent": package_root.parent,
        "extension": artifact("installed_python_extension"),
        "libmeep": artifact("installed_libmeep"),
        "qualification_fontconfig": _resolve_recorded_path(
            fontconfig_record["path"], repo
        ).resolve(),
    }
    for name, path in runtime.items():
        expected_directory = name in ("package_root", "package_parent")
        if expected_directory and not path.is_dir():
            raise RuntimeError(f"receipt runtime directory is absent: {path}")
        if not expected_directory and not path.is_file():
            raise RuntimeError(f"receipt runtime file is absent: {path}")
    return runtime


def _extract_pointer(stdout: str) -> dict[str, Any]:
    matches = [line[len(PREFIX) :] for line in stdout.splitlines() if line.startswith(PREFIX)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one raw-result pointer, found {len(matches)}")
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError as error:
        raise RuntimeError("raw-result pointer is malformed") from error
    if not isinstance(value, dict):
        raise RuntimeError("raw-result pointer is not an object")
    return value


def _extract_material_gradient_stats(stdout: str, *, lane: str) -> dict[str, Any]:
    _require_lane(lane)
    prefix = "material-gradient-stats: "
    lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(
            f"{lane} expected exactly one material-gradient dispatch record"
        )
    match = re.fullmatch(
        r"material-gradient-stats: points=(\d+) forced_legacy_points=(\d+) "
        r"analytic_points=(\d+) edge_legacy_points=(\d+) unsupported_points=(\d+) "
        r"requests=(\d+) analytic_requests=(\d+) legacy_requests=(\d+)",
        lines[0],
    )
    if match is None:
        raise RuntimeError(f"{lane} material-gradient dispatch record is malformed")
    names = (
        "points",
        "forced_legacy_points",
        "analytic_points",
        "edge_legacy_points",
        "unsupported_points",
        "requests",
        "analytic_requests",
        "legacy_requests",
    )
    values = {name: int(value) for name, value in zip(names, match.groups())}
    if lane == CPU_ORACLE_LANE:
        passed = (
            values["points"] > 0
            and values["forced_legacy_points"] == values["points"]
            and values["analytic_points"] == 0
            and values["analytic_requests"] == 0
            and values["legacy_requests"] > 0
            and values["requests"] == values["legacy_requests"]
        )
    elif lane in ANALYTIC_LANES:
        passed = (
            values["points"] > 0
            and values["forced_legacy_points"] == 0
            and values["analytic_points"] == values["points"]
            and values["edge_legacy_points"] == 0
            and values["unsupported_points"] == 0
            and values["legacy_requests"] == 0
            and values["analytic_requests"] > 0
            and values["requests"] == values["analytic_requests"]
        )
    if not passed:
        raise RuntimeError(f"{lane} used the wrong material-gradient implementation")
    return {"pass": True, **values}


def _validate_actual_process_environments(
    result: dict[str, Any], launched: dict[str, str], *, label: str
) -> dict[str, Any]:
    producer_environment = result.get("producer", {}).get("environment")
    ranks = result.get("distributed", {}).get("ranks")
    if not isinstance(producer_environment, dict) or not isinstance(ranks, list):
        raise RuntimeError(f"{label} actual process-environment evidence is absent")
    actual_environments: list[dict[str, str]] = []
    for rank_index, rank in enumerate(ranks):
        environment = rank.get("process_environment") if isinstance(rank, dict) else None
        if not isinstance(environment, dict) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in environment.items()
        ):
            raise RuntimeError(f"{label}.rank{rank_index} process environment is invalid")
        fixed_launch = {
            name: value
            for name, value in launched.items()
            if name != "CUDA_VISIBLE_DEVICES"
            and name not in LAUNCHER_CONSUMED_ENVIRONMENT_KEYS
        }
        if any(environment.get(name) != value for name, value in fixed_launch.items()):
            raise RuntimeError(f"{label}.rank{rank_index} changed the fixed launch environment")
        for name in LAUNCHER_CONSUMED_ENVIRONMENT_KEYS:
            if name in environment and environment[name] != launched.get(name):
                raise RuntimeError(
                    f"{label}.rank{rank_index} changed launcher-consumed environment {name}"
                )
        if environment.get("PMIX_MCA_mca_base_param_files") != launched.get(
            "OMPI_MCA_mca_base_param_files"
        ):
            raise RuntimeError(
                f"{label}.rank{rank_index} changed the fixed PMIx parameter file"
            )
        rank_visible_devices = environment.get("CUDA_VISIBLE_DEVICES")
        if rank_visible_devices is not None and (
            not rank_visible_devices
            or re.fullmatch(r"[A-Za-z0-9_./:+-]+(?:,[A-Za-z0-9_./:+-]+)*", rank_visible_devices)
            is None
        ):
            raise RuntimeError(
                f"{label}.rank{rank_index} has malformed rank-local CUDA visibility"
            )
        forbidden = sorted(FORBIDDEN_INFLUENCE_VARIABLES & set(environment))
        if forbidden:
            raise RuntimeError(
                f"{label}.rank{rank_index} inherited forbidden runtime controls: "
                + ", ".join(forbidden)
            )
        injected_influence = sorted(
            name
            for name in environment
            if name not in launched
            and name not in ALLOWED_LAUNCHER_INJECTED_INFLUENCE_VARIABLES
            and name.startswith(INFLUENTIAL_ENVIRONMENT_PREFIXES)
        )
        if injected_influence:
            raise RuntimeError(
                f"{label}.rank{rank_index} gained unlaunched runtime controls: "
                + ", ".join(injected_influence)
            )
        actual_environments.append(environment)
    if not actual_environments or producer_environment != actual_environments[0]:
        raise RuntimeError(f"{label} producer/rank-zero environments disagree")
    pycache_prefix = pathlib.Path(launched.get("PYTHONPYCACHEPREFIX", ""))
    if (
        launched.get("PYTHONDONTWRITEBYTECODE") != "1"
        or not pycache_prefix.is_dir()
        or any(
            path.is_symlink() or not path.is_dir()
            for path in pycache_prefix.rglob("*")
        )
    ):
        raise RuntimeError(f"{label} executable bytecode isolation is absent or contaminated")
    qualification_home = pathlib.Path(launched.get("HOME", ""))
    home_marker = qualification_home / ".gpmeep-empty-home"
    if (
        not qualification_home.is_dir()
        or sorted(path.name for path in qualification_home.iterdir())
        != [home_marker.name]
        or home_marker.read_text(encoding="utf-8")
        != "gpmeep isolated MPI qualification home\n"
    ):
        raise RuntimeError(f"{label} isolated MPI qualification HOME was modified")
    return {
        "pass": True,
        "launch_environment": dict(sorted(launched.items())),
        "rank_environment_sha256": [
            canonical_sha256(environment) for environment in actual_environments
        ],
        "rank_environment_extra_keys": [
            sorted(set(environment) - set(launched))
            for environment in actual_environments
        ],
    }


def _validate_lane_process_environment(
    result: dict[str, Any], launched: dict[str, str], *, lane: str, label: str
) -> None:
    _require_lane(lane)
    ranks = result.get("distributed", {}).get("ranks")
    if not isinstance(ranks, list):
        raise RuntimeError(f"{label} rank process environments are absent")
    environments = [
        rank.get("process_environment") if isinstance(rank, dict) else None
        for rank in ranks
    ]
    if any(not isinstance(environment, dict) for environment in environments):
        raise RuntimeError(f"{label} rank process environments are invalid")
    if lane in (CPU_ORACLE_LANE, CPU_BENCHMARK_LANE):
        forbidden = {
            "MEEP_GPU_STRICT",
            "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS",
            "GPMEEP_VALIDATION_STRICT_CUDA",
            "CUDA_VISIBLE_DEVICES",
        }
        if forbidden.intersection(launched) or any(
            forbidden.intersection(environment) for environment in environments
        ):
            raise RuntimeError(f"{label} CPU lane inherited CUDA runtime controls")
        expected_omp = (
            {
                "OMP_NUM_THREADS": str(PROFILE["cpu_omp_threads"]),
                "OMP_DYNAMIC": "FALSE",
                "OMP_PLACES": "cores",
                "OMP_PROC_BIND": "close",
            }
            if lane == CPU_BENCHMARK_LANE
            else {
                "OMP_NUM_THREADS": "1",
                "OMP_DYNAMIC": "FALSE",
                "OMP_PROC_BIND": "FALSE",
            }
        )
        if any(
            environment.get(name) != value
            for environment in environments
            for name, value in expected_omp.items()
        ):
            raise RuntimeError(f"{label} CPU OpenMP controls differ from the fixed lane")
    elif lane in CUDA_LANES:
        expected_cuda = {
            "MEEP_GPU_BACKEND": "cuda",
            "MEEP_GPU_STRICT": "1",
            "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
            "GPMEEP_VALIDATION_STRICT_CUDA": "1",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
        }
        if any(
            environment.get(name) != value
            for environment in environments
            for name, value in expected_cuda.items()
        ):
            raise RuntimeError(f"{label} CUDA/OpenMP controls differ from the fixed lane")


def _child_environment(
    runtime: dict[str, pathlib.Path],
    *,
    lane: str,
    output: pathlib.Path,
    pycache_namespace: str,
    prepare_filesystem: bool = True,
) -> dict[str, str]:
    _require_lane(lane)
    environment_prefix = runtime["python"].parent.parent
    output = pathlib.Path(os.path.abspath(os.fspath(output)))
    temporary = output / "tmp"
    matplotlib = output / "matplotlib"
    pycache = output / "pycache" / pycache_namespace
    xdg_cache = output / "cache" / pycache_namespace
    qualification_home = output / "home" / pycache_namespace
    openmpi_params = (
        pathlib.Path(__file__).resolve().parent.parent
        / "environment"
        / "openmpi-qualification-mca-params.conf"
    ).resolve()
    if type(prepare_filesystem) is not bool:
        raise RuntimeError("child environment filesystem mode is invalid")
    if prepare_filesystem:
        temporary.mkdir(parents=True, exist_ok=True)
        matplotlib.mkdir(parents=True, exist_ok=True)
        xdg_cache.mkdir(parents=True, exist_ok=True)
    else:
        for path, label in (
            (temporary, "temporary directory"),
            (matplotlib, "Matplotlib directory"),
            (xdg_cache, "XDG cache directory"),
        ):
            try:
                metadata = path.lstat()
            except OSError as error:
                raise RuntimeError(
                    f"qualification {label} is absent during read-only replay"
                ) from error
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(
                    f"qualification {label} is invalid during read-only replay"
                )
    if pycache.exists():
        metadata = pycache.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or any(
                path.is_symlink() or not path.is_dir()
                for path in pycache.rglob("*")
            )
        ):
            raise RuntimeError("qualification Python bytecode cache is contaminated")
    elif prepare_filesystem:
        pycache.mkdir(parents=True)
    else:
        raise RuntimeError(
            "qualification Python bytecode cache is absent during read-only replay"
        )
    home_marker = qualification_home / ".gpmeep-empty-home"
    if qualification_home.exists():
        metadata = qualification_home.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or sorted(path.name for path in qualification_home.iterdir())
            != [home_marker.name]
            or home_marker.is_symlink()
            or not home_marker.is_file()
            or home_marker.read_text(encoding="utf-8")
            != "gpmeep isolated MPI qualification home\n"
        ):
            raise RuntimeError("isolated MPI qualification HOME is contaminated")
    elif prepare_filesystem:
        qualification_home.mkdir(parents=True)
        home_marker.write_text(
            "gpmeep isolated MPI qualification home\n", encoding="utf-8"
        )
    else:
        raise RuntimeError(
            "isolated MPI qualification HOME is absent during read-only replay"
        )
    if not openmpi_params.is_file():
        raise RuntimeError("fixed Open MPI qualification parameter file is absent")
    for override_name in (
        "openmpi-mca-params-override.conf",
        "pmix-mca-params-override.conf",
        "prte-mca-params-override.conf",
    ):
        if (environment_prefix / "etc" / override_name).exists():
            raise RuntimeError(f"MPI override parameter file is not allowed: {override_name}")
    environment = {
            "CUDA_CACHE_DISABLE": "1",
            "PYTHONPATH": os.pathsep.join(
                (str(pathlib.Path(__file__).resolve().parent), str(runtime["package_parent"]))
            ),
            "PATH": os.pathsep.join(
                (str(environment_prefix / "bin"), "/usr/bin", "/bin")
            ),
            "HOME": str(qualification_home),
            "TMPDIR": str(temporary),
            "XDG_CACHE_HOME": str(xdg_cache),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "LD_LIBRARY_PATH": os.pathsep.join(
                (str(runtime["libmeep"].parent), str(environment_prefix / "lib"))
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(pycache),
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
            "OMPI_MCA_mca_base_param_files": str(openmpi_params),
            "OMPI_MCA_mca_base_component_path": str(
                environment_prefix / "lib" / "openmpi"
            ),
            "PMIX_MCA_mca_base_component_path": str(
                environment_prefix / "lib" / "pmix"
            ),
            "PMIX_MCA_mca_base_param_files": str(openmpi_params),
            "PRTE_MCA_mca_base_param_files": str(openmpi_params),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "FONTCONFIG_FILE": str(runtime["qualification_fontconfig"]),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(matplotlib),
            "MEEP_GPU_MPI_COMPLETION": "waitsome",
            "MEEP_GPU_MPI_TRANSPORT": "auto",
            "MEEP_MATERIAL_GRADIENT_STATS": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if lane in CUDA_LANES and visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    if lane in (CPU_ORACLE_LANE, CPU_BENCHMARK_LANE):
        environment.update(
            {
                "MEEP_GPU_BACKEND": "cpu",
                "MEEP_MATERIAL_GRADIENT_PATH": (
                    PROFILE["oracle_material_gradient_path"]
                    if lane == CPU_ORACLE_LANE
                    else PROFILE["material_gradient_path"]
                ),
            }
        )
        if lane == CPU_BENCHMARK_LANE:
            environment.update(
                {
                    "OMP_NUM_THREADS": str(PROFILE["cpu_omp_threads"]),
                    "OMP_PLACES": "cores",
                    "OMP_PROC_BIND": "close",
                }
            )
    elif lane in CUDA_LANES:
        environment.update(
            {
                "MEEP_GPU_BACKEND": "cuda",
                "MEEP_GPU_STRICT": "1",
                "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
                "GPMEEP_VALIDATION_STRICT_CUDA": "1",
                "MEEP_MATERIAL_GRADIENT_PATH": PROFILE["material_gradient_path"],
            }
        )
    unexpected = sorted(set(environment) - set(RECORDED_ENVIRONMENT_KEYS))
    if unexpected or not all(isinstance(value, str) for value in environment.values()):
        raise RuntimeError(
            "fixed child environment contains unrecorded values: "
            + ", ".join(unexpected)
        )
    return environment


def _run_command(
    command: list[str],
    *,
    cwd: pathlib.Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(
                timeout=PROFILE["timeout_kill_grace_seconds"]
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise CommandTimeout(
            f"process group exceeded the fixed {timeout_seconds}s timeout",
            stdout or "",
            stderr or "",
        ) from error
    return process.returncode, stdout, stderr, time.perf_counter() - started


def _fixed_workload_gate(result: dict[str, Any], *, lane: str) -> None:
    _require_lane(lane)
    workload = result.get("workload")
    if not isinstance(workload, dict):
        raise RuntimeError(f"{lane} workload evidence is absent")
    expected = {
        "resolution": PROFILE["resolution"],
        "run_time": PROFILE["run_time"],
        "cell_size": PROFILE["cell_size"],
        "design_resolution": PROFILE["design_resolution"],
        "design_shape": PROFILE["design_shape"],
        "design_variables": PROFILE["design_variables"],
        "yee_cells": PROFILE["yee_cells"],
        "weights_sha256": PROFILE["weights_sha256"],
        "direction_sha256": PROFILE["direction_sha256"],
        "material_gradient_path": (
            PROFILE["oracle_material_gradient_path"]
            if lane == CPU_ORACLE_LANE
            else PROFILE["material_gradient_path"]
        ),
    }
    for name, value in expected.items():
        if workload.get(name) != value:
            raise RuntimeError(
                f"{lane} changed fixed workload {name}: {workload.get(name)!r} != {value!r}"
            )
    for phase_name in ("forward", "adjoint"):
        phase = workload.get("phases", {}).get(phase_name, {})
        for suffix in ("timesteps", "end_time"):
            expected_value = PROFILE[f"{phase_name}_{suffix}"]
            if phase.get(suffix) != expected_value:
                raise RuntimeError(
                    f"{lane} changed fixed {phase_name} {suffix}: "
                    f"{phase.get(suffix)!r} != {expected_value!r}"
                )


def _validate_lazy_import_contract(
    result: dict[str, Any], *, label: str
) -> dict[str, Any]:
    record = result.get("producer", {}).get("lazy_import_contract")
    expected_roots = ["jax", "matplotlib", "scipy"]
    expected_core = {
        "meep.Simulation",
        "meep.adjoint.DesignRegion",
        "meep.adjoint.FourierFields",
        "meep.adjoint.OptimizationProblem",
    }
    expected_lazy = {
        "meep.Animate2D",
        "meep.plot2D",
        "meep.adjoint.FilteredSource",
        "meep.adjoint.MeepJaxWrapper",
        "meep.adjoint.conic_filter",
    }
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "schema_version",
            "capture_point",
            "deferred_module_roots",
            "deferred_module_loaded",
            "core_surface",
            "lazy_surface_discoverable",
            "pass",
        }
        or record.get("schema_version") != 1
        or record.get("capture_point")
        != "immediately-after-meep-and-meep-adjoint-import"
        or record.get("deferred_module_roots") != expected_roots
        or record.get("deferred_module_loaded")
        != {name: False for name in expected_roots}
        or set(record.get("core_surface", {})) != expected_core
        or any(value is not True for value in record["core_surface"].values())
        or set(record.get("lazy_surface_discoverable", {})) != expected_lazy
        or any(
            value is not True
            for value in record["lazy_surface_discoverable"].values()
        )
        or record.get("pass") is not True
    ):
        raise RuntimeError(f"{label} lazy import contract is absent or failed")
    return dict(record)


def _cpu_counter(statistics: Any, group: str, field: str) -> int:
    value = statistics.get(group, {}).get(field) if isinstance(statistics, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"CPU oracle counter {group}.{field} is invalid")
    return value


def validate_cpu_oracle(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("schema_version") != 4:
        raise RuntimeError("CPU oracle raw schema is unsupported")
    COMPARATOR.validate_gradient_summaries(result, "cpu-oracle")
    gradient = validate_gradient_record(result)
    if not gradient or max(abs(value) for value in gradient) <= 1.0e-12:
        raise RuntimeError("CPU oracle gradient is all zero")
    runtime = COMPARATOR.validate_runtime_closure(result, "cpu-oracle")
    distributed = result.get("distributed")
    if not isinstance(distributed, dict) or distributed.get("world_size") != 1:
        raise RuntimeError("CPU oracle is not a one-rank MPI record")
    ranks = distributed.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 1:
        raise RuntimeError("CPU oracle rank evidence is incomplete")
    rank = ranks[0]
    rank_runtime = COMPARATOR.validate_runtime_closure(rank, "cpu-oracle.rank0")
    if rank_runtime != runtime:
        raise RuntimeError("CPU oracle rank loaded a different runtime closure")
    rank_gradient = rank.get("gradient")
    if (
        not isinstance(rank_gradient, list)
        or canonical_float64_sha256(rank_gradient)
        != result.get("result", {}).get("gradient_sha256")
    ):
        raise RuntimeError("CPU oracle rank gradient disagrees with rank zero")
    backend = rank.get("backend")
    if (
        not isinstance(backend, dict)
        or backend.get("requested") != "cpu"
        or backend.get("active") != "cpu"
        or backend.get("single_precision") is not True
        or backend.get("strict_cuda_marker") is not False
    ):
        raise RuntimeError("CPU oracle did not use the FP32 CPU backend")
    statistics_record = rank.get("final_statistics")
    cpu_calls = sum(
        _cpu_counter(statistics_record, group, field)
        for group, field in (
            ("dispatch", "cpu_curl_calls"),
            ("field_updates", "cpu_update_eh_calls"),
            ("sources", "cpu_source_calls"),
            ("boundaries", "cpu_boundary_calls"),
            ("dfts", "cpu_dft_calls"),
        )
    )
    cuda_calls = sum(
        _cpu_counter(statistics_record, group, field)
        for group, field in (
            ("dispatch", "cuda_curl_calls"),
            ("field_updates", "cuda_update_eh_calls"),
            ("sources", "cuda_source_calls"),
            ("boundaries", "cuda_boundary_calls"),
            ("dfts", "cuda_dft_calls"),
        )
    )
    if cpu_calls <= 0 or cuda_calls != 0:
        raise RuntimeError("CPU oracle coverage is absent or contains CUDA work")
    multi_gpu = statistics_record.get("multi_gpu", {})
    if any(
        multi_gpu.get(name) != 0
        for name in (
            "mpi_messages",
            "mpi_scalars",
            "cuda_aware_bytes",
            "pinned_staging_bytes",
            "pinned_device_to_host_bytes",
            "pinned_host_to_device_bytes",
        )
    ):
        raise RuntimeError("one-rank CPU oracle unexpectedly recorded MPI traffic")
    return {
        "runtime": runtime,
        "rank_runtimes": [rank_runtime],
        "gradient": gradient,
        "objective": float(result["result"]["objective"]),
    }


def _finite_positive(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RuntimeError(f"{label} is not finite and positive")
    return float(value)


def _linux_cpu_core(cpu: int, *, label: str) -> tuple[int, int, tuple[int, ...]]:
    topology = pathlib.Path("/sys/devices/system/cpu") / f"cpu{cpu}" / "topology"
    try:
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
        siblings_text = (topology / "thread_siblings_list").read_text().strip()
    except (OSError, ValueError) as error:
        raise RuntimeError(f"{label} cannot resolve Linux CPU topology for cpu{cpu}") from error
    siblings: set[int] = set()
    for token in siblings_text.split(","):
        bounds = token.split("-", 1)
        try:
            first = int(bounds[0])
            last = int(bounds[-1])
        except ValueError as error:
            raise RuntimeError(f"{label} has malformed CPU sibling topology") from error
        if first < 0 or last < first:
            raise RuntimeError(f"{label} has malformed CPU sibling topology")
        siblings.update(range(first, last + 1))
    if cpu not in siblings:
        raise RuntimeError(f"{label} CPU is absent from its sibling topology")
    return package, core, tuple(sorted(siblings))


def _validate_cpu_affinity(result: dict[str, Any], *, label: str) -> dict[str, Any]:
    ranks = result.get("distributed", {}).get("ranks")
    if not isinstance(ranks, list) or len(ranks) != PROFILE["cpu_ranks"]:
        raise RuntimeError(f"{label} CPU affinity rank matrix is absent")
    used_cores: set[tuple[int, int]] = set()
    used_logical: set[int] = set()
    hostnames: set[str] = set()
    rank_records: list[dict[str, Any]] = []
    for rank_index, rank in enumerate(ranks):
        affinity = rank.get("cpu_affinity") if isinstance(rank, dict) else None
        completion = rank.get("completion_thread_cpu_affinity") if isinstance(rank, dict) else None
        thread_map = rank.get("process_thread_cpu_affinities") if isinstance(rank, dict) else None
        hostname = rank.get("hostname") if isinstance(rank, dict) else None
        if (
            not isinstance(affinity, list)
            or affinity != sorted(set(affinity))
            or not affinity
            or not isinstance(completion, list)
            or completion != sorted(set(completion))
            or not completion
            or not isinstance(thread_map, dict)
            or set(thread_map) != {"thread_count", "logical_cpu_union", "threads"}
            or not isinstance(hostname, str)
            or not hostname
        ):
            raise RuntimeError(f"{label}.rank{rank_index} CPU affinity evidence is invalid")
        threads = thread_map["threads"]
        if (
            isinstance(thread_map["thread_count"], bool)
            or not isinstance(thread_map["thread_count"], int)
            or thread_map["thread_count"] < PROFILE["cpu_omp_threads"]
            or not isinstance(threads, list)
            or len(threads) != thread_map["thread_count"]
        ):
            raise RuntimeError(f"{label}.rank{rank_index} thread-affinity inventory is invalid")
        thread_ids: set[int] = set()
        placements: list[tuple[int, ...]] = []
        logical_union: set[int] = set()
        core_places: dict[tuple[int, int], tuple[int, ...]] = {}
        for thread in threads:
            thread_id = thread.get("thread_id") if isinstance(thread, dict) else None
            logical = thread.get("logical_cpus") if isinstance(thread, dict) else None
            if (
                isinstance(thread_id, bool)
                or not isinstance(thread_id, int)
                or thread_id <= 0
                or thread_id in thread_ids
                or not isinstance(logical, list)
                or logical != sorted(set(logical))
                or not logical
                or any(isinstance(cpu, bool) or not isinstance(cpu, int) for cpu in logical)
            ):
                raise RuntimeError(f"{label}.rank{rank_index} thread affinity row is invalid")
            thread_ids.add(thread_id)
            identities = [_linux_cpu_core(cpu, label=label) for cpu in logical]
            core_ids = {(package, core) for package, core, _ in identities}
            sibling_sets = {siblings for _, _, siblings in identities}
            if len(core_ids) != 1 or len(sibling_sets) != 1 or tuple(logical) != next(iter(sibling_sets)):
                raise RuntimeError(f"{label}.rank{rank_index} is not bound to exact physical-core places")
            identity = next(iter(core_ids))
            core_places[identity] = tuple(logical)
            placements.append(tuple(logical))
            logical_union.update(logical)
        if (
            thread_map["logical_cpu_union"] != sorted(logical_union)
            or not set(affinity).issubset(logical_union)
            or not set(completion).issubset(logical_union)
            or len(core_places) != PROFILE["cpu_omp_threads"]
            or any(place not in placements for place in core_places.values())
            or used_cores.intersection(core_places)
            or used_logical.intersection(logical_union)
        ):
            raise RuntimeError(f"{label}.rank{rank_index} CPU core placement is incomplete or overlapping")
        used_cores.update(core_places)
        used_logical.update(logical_union)
        hostnames.add(hostname)
        rank_records.append(
            {
                "world_rank": rank_index,
                "hostname": hostname,
                "physical_cores": [list(identity) for identity in sorted(core_places)],
                "logical_cpu_union": sorted(logical_union),
                "thread_count": thread_map["thread_count"],
            }
        )
    if len(hostnames) != 1:
        raise RuntimeError(f"{label} CPU baseline unexpectedly crossed hosts")
    producer = result.get("producer", {})
    if (
        producer.get("hostname") != next(iter(hostnames))
        or
        producer.get("cpu_affinity") != ranks[0].get("cpu_affinity")
        or producer.get("completion_thread_cpu_affinity")
        != ranks[0].get("completion_thread_cpu_affinity")
        or producer.get("process_thread_cpu_affinities")
        != ranks[0].get("process_thread_cpu_affinities")
    ):
        raise RuntimeError(f"{label} producer/rank-zero CPU affinity evidence differs")
    return {
        "pass": True,
        "binding": "core",
        "omp_threads": PROFILE["cpu_omp_threads"],
        "hostname": next(iter(hostnames)),
        "rank_affinity": rank_records,
        "used_logical_cpus": sorted(used_logical),
        "used_physical_cores": [list(identity) for identity in sorted(used_cores)],
    }


def _validate_cpu_rank_coverage(rank: dict[str, Any], *, label: str) -> dict[str, Any]:
    backend = rank.get("backend")
    if (
        not isinstance(backend, dict)
        or backend.get("requested") != "cpu"
        or backend.get("active") != "cpu"
        or rank.get("active_backend") != "cpu"
        or backend.get("compiled") is not True
        or backend.get("single_precision") is not True
        or backend.get("strict_cuda_marker") is not False
    ):
        raise RuntimeError(f"{label} did not use the FP32 CPU backend")
    final_statistics = rank.get("final_statistics")
    if not isinstance(final_statistics, dict):
        raise RuntimeError(f"{label} final CPU statistics are absent")
    schema = {
        group: sorted(values)
        for group, values in sorted(final_statistics.items())
        if isinstance(group, str) and isinstance(values, dict)
    }
    if (
        len(schema) != len(final_statistics)
        or canonical_sha256(schema) != CPU_STATISTICS_SCHEMA_SHA256
    ):
        raise RuntimeError(f"{label} final CPU statistics schema is incomplete")
    final_flat: dict[tuple[str, str], int] = {}
    for group, values in final_statistics.items():
        for field, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"{label}.{group}.{field} CPU statistic is invalid")
            if (
                field.startswith("cuda_")
                or field
                in {
                    "device_to_host_bytes",
                    "host_to_device_bytes",
                    "ldos_full_field_device_to_host_bytes_avoided",
                }
                or group
                in {"multi_gpu", "mpi_completion", "resident", "runtime", "tile_coalescing"}
                or field.startswith("cpu_polarization_")
            ) and value != 0:
                raise RuntimeError(
                    f"{label}.{group}.{field} contains CUDA work or dispersive activity"
                )
            final_flat[(group, field)] = value
    cpu_calls = {
        field: _cpu_counter(final_statistics, group, field)
        for group, field in (
            ("dispatch", "cpu_curl_calls"),
            ("field_updates", "cpu_update_eh_calls"),
            ("sources", "cpu_source_calls"),
            ("boundaries", "cpu_boundary_calls"),
            ("dfts", "cpu_dft_calls"),
        )
    }
    cuda_calls = {
        field: _cpu_counter(final_statistics, group, field)
        for group, field in (
            ("dispatch", "cuda_curl_calls"),
            ("field_updates", "cuda_update_eh_calls"),
            ("sources", "cuda_source_calls"),
            ("boundaries", "cuda_boundary_calls"),
            ("dfts", "cuda_dft_calls"),
        )
    }
    if any(cpu_calls[name] <= 0 for name in ("cpu_curl_calls", "cpu_update_eh_calls", "cpu_boundary_calls")) or any(cuda_calls.values()):
        raise RuntimeError(f"{label} CPU coverage is absent or contains CUDA work")
    multi_gpu = final_statistics.get("multi_gpu", {}) if isinstance(final_statistics, dict) else {}
    if any(
        multi_gpu.get(name) != 0
        for name in (
            "mpi_messages",
            "mpi_scalars",
            "cuda_aware_bytes",
            "pinned_staging_bytes",
            "pinned_device_to_host_bytes",
            "pinned_host_to_device_bytes",
        )
    ):
        raise RuntimeError(f"{label} CPU baseline contains GPU-halo accounting")
    phases = rank.get("phases")
    if not isinstance(phases, dict) or set(phases) != {
        "update_design",
        "forward",
        "adjoint",
        "gradient",
    }:
        raise RuntimeError(f"{label} phase accounting is incomplete")
    measured: dict[str, Any] = {}
    phase_flat: dict[str, dict[tuple[str, str], int]] = {}
    for phase_name, phase in phases.items():
        statistics_delta = phase.get("statistics_delta") if isinstance(phase, dict) else None
        phase_schema = (
            {
                group: sorted(values)
                for group, values in sorted(statistics_delta.items())
                if isinstance(group, str) and isinstance(values, dict)
            }
            if isinstance(statistics_delta, dict)
            else {}
        )
        if canonical_sha256(phase_schema) != CPU_STATISTICS_SCHEMA_SHA256:
            raise RuntimeError(f"{label}.{phase_name} CPU statistics schema is incomplete")
        flattened: dict[tuple[str, str], int] = {}
        for group, values in statistics_delta.items():
            for field, value in values.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise RuntimeError(f"{label}.{phase_name}.{group}.{field} is invalid")
                flattened[(group, field)] = value
        phase_flat[phase_name] = flattened
    if {
        path: sum(phase_flat[phase][path] for phase in phases)
        for path in final_flat
    } != final_flat:
        raise RuntimeError(f"{label} final CPU statistics differ from exact phase sum")
    if any(
        value
        for phase in ("update_design", "gradient")
        for value in phase_flat[phase].values()
    ):
        raise RuntimeError(f"{label} non-FDTD phase unexpectedly recorded backend work")
    for phase_name in ("forward", "adjoint"):
        counters = phases.get(phase_name, {}).get("backend_counters")
        if not isinstance(counters, dict) or set(counters) != {"cpu", "cuda"}:
            raise RuntimeError(f"{label}.{phase_name} backend counters are absent")
        cpu = counters["cpu"]
        cuda = counters["cuda"]
        expected = {"curl", "update_eh", "source", "boundary", "dft", "polarization"}
        if not isinstance(cpu, dict) or not isinstance(cuda, dict) or set(cpu) != expected or set(cuda) != expected:
            raise RuntimeError(f"{label}.{phase_name} backend counter schema is incomplete")
        normalized: dict[str, dict[str, int]] = {}
        statistic_mapping = {
            "curl": ("dispatch", "curl"),
            "update_eh": ("field_updates", "update_eh"),
            "source": ("sources", "source"),
            "boundary": ("boundaries", "boundary"),
            "dft": ("dfts", "dft"),
            "polarization": ("polarizations", "polarization"),
        }
        for name in expected:
            cpu_values = cpu[name]
            cuda_values = cuda[name]
            if any(
                not isinstance(values, dict)
                or set(values) != {"calls", "points"}
                or any(isinstance(values[field], bool) or not isinstance(values[field], int) or values[field] < 0 for field in ("calls", "points"))
                for values in (cpu_values, cuda_values)
            ):
                raise RuntimeError(f"{label}.{phase_name}.{name} counters are invalid")
            if cuda_values != {"calls": 0, "points": 0} or ((cpu_values["calls"] == 0) != (cpu_values["points"] == 0)):
                raise RuntimeError(f"{label}.{phase_name}.{name} CPU/CUDA accounting disagrees")
            group, stem = statistic_mapping[name]
            statistics_delta = phases[phase_name]["statistics_delta"][group]
            if any(
                values[field]
                != statistics_delta[f"{backend_name}_{stem}_{field}"]
                for backend_name, values in (
                    ("cpu", cpu_values),
                    ("cuda", cuda_values),
                )
                for field in ("calls", "points")
            ):
                raise RuntimeError(
                    f"{label}.{phase_name}.{name} backend view/statistics disagree"
                )
            normalized[name] = dict(cpu_values)
        for name in ("curl", "update_eh", "boundary"):
            if normalized[name]["calls"] <= 0 or normalized[name]["points"] <= 0:
                raise RuntimeError(f"{label}.{phase_name} CPU {name} coverage is absent")
        measured[phase_name] = normalized
    return {
        "cpu_calls": cpu_calls,
        "cuda_calls": cuda_calls,
        "phases": measured,
        "final_statistics_sha256": canonical_sha256(final_statistics),
        "phase_statistics_sha256": {
            phase: canonical_sha256(phases[phase]["statistics_delta"])
            for phase in phases
        },
    }


def validate_cpu_benchmark_record(result: dict[str, Any], *, label: str) -> dict[str, Any]:
    if result.get("schema_version") != 4:
        raise RuntimeError(f"{label} benchmark schema is unsupported")
    COMPARATOR.validate_gradient_summaries(result, label)
    gradient = validate_gradient_record(result)
    objective = result.get("result", {}).get("objective")
    if (
        not gradient
        or max(abs(value) for value in gradient) <= 1.0e-12
        or isinstance(objective, bool)
        or not isinstance(objective, (int, float))
        or not math.isfinite(float(objective))
    ):
        raise RuntimeError(f"{label} objective/full gradient is invalid")
    distributed = result.get("distributed")
    if (
        not isinstance(distributed, dict)
        or distributed.get("world_size") != PROFILE["cpu_ranks"]
        or distributed.get("timing_semantics") != "barrier-delimited maximum rank elapsed per phase"
    ):
        raise RuntimeError(f"{label} CPU distributed evidence is invalid")
    gate = distributed.get("gate")
    if not isinstance(gate, dict) or any(gate.get(name) is not True for name in ("pass", "objective", "gradient", "strict_cuda_coverage")):
        raise RuntimeError(f"{label} producer-side distributed gate did not pass")
    ranks = distributed.get("ranks")
    if not isinstance(ranks, list) or [rank.get("world_rank") for rank in ranks if isinstance(rank, dict)] != list(range(PROFILE["cpu_ranks"])):
        raise RuntimeError(f"{label} CPU rank evidence is incomplete")
    runtime = COMPARATOR.validate_runtime_closure(result, label)
    rank_runtimes: list[dict[str, Any]] = []
    rank_coverage: list[dict[str, Any]] = []
    for rank_index, rank in enumerate(ranks):
        rank_label = f"{label}.rank{rank_index}"
        rank_runtime = COMPARATOR.validate_runtime_closure(rank, rank_label)
        if rank_index == 0:
            if rank_runtime != runtime:
                raise RuntimeError(f"{rank_label} runtime differs from rank zero")
        elif COMPARATOR.runtime_environment_independent_projection(rank_runtime) != COMPARATOR.runtime_environment_independent_projection(runtime):
            raise RuntimeError(f"{rank_label} loaded a different controlled runtime")
        rank_gradient = rank.get("gradient")
        if (
            not isinstance(rank_gradient, list)
            or canonical_float64_sha256(rank_gradient) != result.get("result", {}).get("gradient_sha256")
            or float(rank.get("objective")) != float(objective)
        ):
            raise RuntimeError(f"{rank_label} objective/full gradient differs")
        rank_runtimes.append(rank_runtime)
        rank_coverage.append(_validate_cpu_rank_coverage(rank, label=rank_label))
    for phase_name in ("forward", "adjoint"):
        for name in ("source", "dft"):
            if sum(item["phases"][phase_name][name]["calls"] for item in rank_coverage) <= 0 or sum(item["phases"][phase_name][name]["points"] for item in rank_coverage) <= 0:
                raise RuntimeError(f"{label}.{phase_name} aggregate CPU {name} coverage is absent")
    if result.get("workload", {}).get("final_statistics") != ranks[0].get("final_statistics"):
        raise RuntimeError(f"{label} top/rank-zero statistics differ")
    top_phases = result.get("workload", {}).get("phases")
    if not isinstance(top_phases, dict) or any(
        top_phases.get(phase, {}).get(field)
        != ranks[0].get("phases", {}).get(phase, {}).get(field)
        for phase in ("update_design", "forward", "adjoint", "gradient")
        for field in ("statistics_delta", "backend_counters")
    ):
        raise RuntimeError(f"{label} top/rank-zero phase statistics differ")
    top_backend = result.get("backend")
    if not isinstance(top_backend, dict) or top_backend != ranks[0].get("backend"):
        raise RuntimeError(f"{label} top/rank-zero backend evidence differs")
    timing = result.get("timing")
    if not isinstance(timing, dict):
        raise RuntimeError(f"{label} timing evidence is absent")
    rank_wall = [_finite_positive(rank.get("timing", {}).get("workload_wall_seconds"), f"{label}.rank{index}.workload_wall") for index, rank in enumerate(ranks)]
    workload_wall = _finite_positive(timing.get("workload_wall_seconds"), f"{label}.workload_wall")
    if workload_wall != max(rank_wall) or timing.get("rank_local_workload_wall_seconds") != rank_wall[0]:
        raise RuntimeError(f"{label} workload wall time is not exact max-rank time")
    phase_names = ("update_design", "forward", "adjoint", "gradient")
    phase_maxima: dict[str, float] = {}
    rank_elapsed: list[float] = []
    for phase_name in phase_names:
        values = [_finite_positive(rank.get("timing", {}).get(f"{phase_name}_seconds"), f"{label}.rank{index}.{phase_name}") for index, rank in enumerate(ranks)]
        phase_maxima[phase_name] = max(values)
        if timing.get(f"{phase_name}_seconds") != phase_maxima[phase_name]:
            raise RuntimeError(f"{label} {phase_name} timing is not exact max-rank time")
        if (
            any(
                rank.get("phases", {}).get(phase_name, {}).get("elapsed_seconds")
                != value
                for rank, value in zip(ranks, values)
            )
            or top_phases.get(phase_name, {}).get("elapsed_seconds")
            != phase_maxima[phase_name]
        ):
            raise RuntimeError(
                f"{label} {phase_name} timing/phase elapsed differs"
            )
    for index, (rank, wall) in enumerate(zip(ranks, rank_wall)):
        elapsed = _finite_positive(rank.get("timing", {}).get("elapsed_seconds"), f"{label}.rank{index}.elapsed")
        rank_phase_sum = math.fsum(
            _finite_positive(
                rank.get("timing", {}).get(f"{phase_name}_seconds"),
                f"{label}.rank{index}.{phase_name}",
            )
            for phase_name in phase_names
        )
        if not math.isclose(
            elapsed, rank_phase_sum, rel_tol=1e-14, abs_tol=1e-14
        ):
            raise RuntimeError(f"{label}.rank{index} elapsed is not its phase sum")
        if wall < elapsed:
            raise RuntimeError(f"{label}.rank{index} workload wall is shorter than phase elapsed")
        rank_elapsed.append(elapsed)
    if timing.get("rank_local_elapsed_seconds") != rank_elapsed[0]:
        raise RuntimeError(f"{label} rank-local elapsed is not rank-zero elapsed")
    elapsed = _finite_positive(timing.get("elapsed_seconds"), f"{label}.elapsed")
    if not math.isclose(elapsed, sum(phase_maxima.values()), rel_tol=1e-14, abs_tol=1e-14):
        raise RuntimeError(f"{label} elapsed time is not the sum of phase maxima")
    affinity = _validate_cpu_affinity(result, label=label)
    return {
        "runtime": runtime,
        "rank_runtimes": rank_runtimes,
        "rank_coverage": rank_coverage,
        "affinity": affinity,
        "workload_wall_seconds": workload_wall,
        "rank_workload_wall_seconds": rank_wall,
        "elapsed_seconds": elapsed,
        "rank_elapsed_seconds": rank_elapsed,
        "objective": float(objective),
    }


def _validate_receipt_rank_runtimes(
    receipt: dict[str, Any],
    validation: dict[str, Any],
    *,
    hash_cache: Any | None = None,
) -> dict[str, Any]:
    rank_runtimes = validation.get("rank_runtimes")
    if not isinstance(rank_runtimes, list) or not rank_runtimes:
        raise RuntimeError("validated rank-runtime closure is absent")
    gates = [
        COMPARATOR.validate_receipt_runtime(
            receipt, runtime, hash_cache=hash_cache
        )
        for runtime in rank_runtimes
    ]
    result = dict(gates[0])
    result["rank_count"] = len(gates)
    result["rank_loaded_library_origins"] = [
        gate["loaded_library_origins"] for gate in gates
    ]
    return result


def _validate_cuda_rank_driver_evidence(
    result: dict[str, Any], validation: dict[str, Any], *, label: str
) -> list[dict[str, Any]]:
    ranks = result.get("distributed", {}).get("ranks")
    identifiers = validation.get("device_identifiers")
    if (
        not isinstance(ranks, list)
        or not isinstance(identifiers, list)
        or len(ranks) != len(identifiers)
    ):
        raise RuntimeError(f"{label} rank-local driver evidence matrix is incomplete")
    validated: list[dict[str, Any]] = []
    producer_hostname = result.get("producer", {}).get("hostname")
    for rank_index, (rank, identifier) in enumerate(zip(ranks, identifiers)):
        rank_label = f"{label}.rank{rank_index}"
        driver = COMPARATOR.validate_rank_nvidia_driver(
            rank.get("nvidia_driver"), identifier, rank_label
        )
        if (
            not isinstance(rank.get("hostname"), str)
            or not rank["hostname"]
            or driver.get("hostname") != rank["hostname"]
            or (rank_index == 0 and producer_hostname != rank["hostname"])
        ):
            raise RuntimeError(f"{rank_label} rank/driver hostname binding differs")
        validated.append(driver)
    return validated


def _run_fd_oracle_test(
    *,
    repo: pathlib.Path,
    output: pathlib.Path,
    runtime: dict[str, pathlib.Path],
    run_id: str,
    receipt_id: str,
    snapshot_sha256: str,
    publication_capability: _RunPublicationCapability,
) -> dict[str, Any]:
    label = f"{run_id}-directional-fd-oracle"
    stdout_path = output / f"{label}.stdout.log"
    stderr_path = output / f"{label}.stderr.log"
    timing_path = output / f"{label}.parent-timing.json"
    for path in (stdout_path, stderr_path, timing_path):
        path.unlink(missing_ok=True)
    test_path = (repo / "python" / "tests" / "test_adjoint_default_material_grid.py").resolve()
    command = [str(runtime["python"]), str(test_path), "-v"]
    environment = _child_environment(
        runtime,
        lane=CPU_ORACLE_LANE,
        output=output,
        pycache_namespace=run_id,
    )
    started_at = _utc_now()
    try:
        returncode, stdout, stderr, process_seconds = _run_command(
            command,
            cwd=repo,
            environment=environment,
            timeout_seconds=PROFILE["timeout_seconds"],
        )
    except CommandTimeout as error:
        atomic_write_text(stdout_path, error.stdout)
        atomic_write_text(stderr_path, error.stderr)
        raise RuntimeError("directional finite-difference oracle timed out") from error
    completed_at = _utc_now()
    ledger_key = f"fd:{label}"
    publication_capability.record_measurement(
        ledger_key,
        {
            "role": "finite-difference-oracle",
            "run_id": run_id,
            "output": str(output.resolve()),
            "label": label,
            "process_seconds": process_seconds,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "command_sha256": canonical_sha256(command),
            "environment_sha256": canonical_sha256(
                dict(sorted(environment.items()))
            ),
            "artifact_paths": {
                "stdout_log": str(stdout_path.resolve()),
                "stderr_log": str(stderr_path.resolve()),
                "timing_file": str(timing_path.resolve()),
            },
        },
    )
    atomic_write_text(stdout_path, stdout)
    atomic_write_text(stderr_path, stderr)
    if returncode != 0:
        raise RuntimeError(
            f"directional finite-difference oracle exited {returncode}; see {stderr_path}"
        )
    timing_record = {
        "schema_version": 1,
        "run_id": run_id,
        "label": label,
        "process_seconds": process_seconds,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "command_sha256": canonical_sha256(command),
        "environment_sha256": canonical_sha256(dict(sorted(environment.items()))),
    }
    atomic_write_json(timing_path, timing_record)
    direction_records = _parse_fd_oracle_output(stdout, stderr)
    oracle = {
        "pass": True,
        "kind": "installed-multi-direction-and-component-finite-difference",
        "receipt_id": receipt_id,
        "source_snapshot_sha256": snapshot_sha256,
        "test_file": str(test_path),
        "test_sha256": sha256_file(test_path),
        "methods": list(FD_TEST_METHODS),
        "directions": len(FD_DIRECTIONS),
        "direction_records": direction_records,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "process_seconds": process_seconds,
        "command": command,
        "environment": dict(sorted(environment.items())),
        "stdout_log": str(stdout_path.resolve()),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_log": str(stderr_path.resolve()),
        "stderr_sha256": sha256_file(stderr_path),
        "timing_file": str(timing_path.resolve()),
        "timing_sha256": sha256_file(timing_path),
    }
    validate_fd_oracle_summary(oracle)
    publication_capability.record_artifacts(
        ledger_key,
        {
            "role": "finite-difference-oracle",
            "label": label,
            "artifact_sha256": {
                "stdout_log": oracle["stdout_sha256"],
                "stderr_log": oracle["stderr_sha256"],
                "timing_file": oracle["timing_sha256"],
            },
        },
    )
    return oracle


def _parse_fd_oracle_output(stdout: str, stderr: str) -> list[dict[str, Any]]:
    combined = stdout + "\n" + stderr
    missing_methods = [name for name in FD_TEST_METHODS if name not in combined]
    if missing_methods or re.search(r"Ran\s+3\s+tests?\s+in", combined) is None:
        raise RuntimeError("directional finite-difference oracle test inventory is incomplete")
    if re.search(r"(?m)^OK\s*$", combined) is None:
        raise RuntimeError("directional finite-difference oracle did not report OK")
    records: list[dict[str, Any]] = []
    for line in combined.splitlines():
        if not line.startswith(FD_DIRECTION_PREFIX):
            continue
        try:
            record = json.loads(line[len(FD_DIRECTION_PREFIX) :])
        except json.JSONDecodeError as error:
            raise RuntimeError("finite-difference direction evidence is malformed") from error
        records.append(record)
    return records


def validate_fd_oracle_summary(fd_oracle: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(fd_oracle, dict)
        or fd_oracle.get("pass") is not True
        or fd_oracle.get("kind")
        != "installed-multi-direction-and-component-finite-difference"
        or fd_oracle.get("directions") != len(FD_DIRECTIONS)
    ):
        raise RuntimeError("directional finite-difference oracle is absent or incomplete")
    records = fd_oracle.get("direction_records")
    if not isinstance(records, list) or len(records) != len(FD_DIRECTIONS):
        raise RuntimeError("finite-difference direction inventory is incomplete")
    names: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "name",
            "finite_difference",
            "adjoint_projection",
            "rtol",
            "atol",
            "pass",
        }:
            raise RuntimeError("finite-difference direction record schema is invalid")
        name = record.get("name")
        finite_difference = record.get("finite_difference")
        adjoint_projection = record.get("adjoint_projection")
        if (
            not isinstance(name, str)
            or isinstance(finite_difference, bool)
            or not isinstance(finite_difference, (int, float))
            or isinstance(adjoint_projection, bool)
            or not isinstance(adjoint_projection, (int, float))
            or not math.isfinite(float(finite_difference))
            or not math.isfinite(float(adjoint_projection))
            or abs(float(adjoint_projection)) <= 5.0e-3
            or record.get("rtol") != 8.0e-2
            or record.get("atol") != 2.0e-5
            or record.get("pass") is not True
            or not math.isclose(
                float(adjoint_projection),
                float(finite_difference),
                rel_tol=8.0e-2,
                abs_tol=2.0e-5,
            )
        ):
            raise RuntimeError("finite-difference direction record did not pass the fixed gate")
        names.append(name)
    if tuple(names) != FD_DIRECTIONS:
        raise RuntimeError("finite-difference direction names/order changed")
    return {"pass": True, "directions": names}


def _expected_sample_command(
    *,
    repo: pathlib.Path,
    runtime: dict[str, pathlib.Path],
    receipt_id: str,
    snapshot_sha256: str,
    producer_sha256: str,
    run_id: str,
    nonce: str,
    lane: str,
    sample_kind: str,
    iteration: int,
    result_path: pathlib.Path,
    ranks: int,
    capture_field_times: bool,
) -> list[str]:
    _require_lane(lane)
    if ranks != _lane_ranks(lane):
        raise RuntimeError(f"{lane} rank count differs from the fixed lane policy")
    producer_command = [
        str(runtime["python"]),
        str(repo / "scripts" / "benchmark-adjoint.py"),
        "--resolution",
        str(PROFILE["resolution"]),
        "--run-time",
        str(PROFILE["run_time"]),
        "--cell-size",
        str(PROFILE["cell_size"]),
        "--design-resolution",
        str(PROFILE["design_resolution"]),
        "--result-file",
        str(result_path),
        "--qualification-profile",
        PROFILE_ID,
        "--run-nonce",
        nonce,
        "--qualification-run-id",
        run_id,
        "--build-receipt-id",
        receipt_id,
        "--source-snapshot-sha256",
        snapshot_sha256,
        "--producer-sha256",
        producer_sha256,
        "--sample-kind",
        sample_kind,
        "--sample-lane",
        lane,
        "--sample-iteration",
        str(iteration),
    ]
    if capture_field_times:
        producer_command.append("--capture-field-times")
    launcher = [str(runtime["mpiexec"])]
    if lane == CPU_BENCHMARK_LANE:
        launcher.extend(
            (
                "--nooversubscribe",
                "--report-bindings",
                "--map-by",
                f"ppr:{PROFILE['cpu_ranks']}:node:PE={PROFILE['cpu_omp_threads']}",
                "--bind-to",
                "core",
            )
        )
    return [*launcher, "-n", str(ranks), *producer_command]


def _expected_producer_command(
    **kwargs: Any,
) -> list[str]:
    command = _expected_sample_command(**kwargs)
    python_path = str(kwargs["runtime"]["python"])
    try:
        start = command.index(python_path)
    except ValueError as error:
        raise RuntimeError("sample launch command lacks the receipt Python") from error
    return command[start:]


def _run_raw_sample(
    *,
    repo: pathlib.Path,
    output: pathlib.Path,
    runtime: dict[str, pathlib.Path],
    receipt: dict[str, Any],
    snapshot_sha256: str,
    producer_sha256: str,
    run_id: str,
    lane: str,
    sample_kind: str,
    iteration: int,
    publication_capability: _RunPublicationCapability,
    completion_policy: str = "waitsome",
    artifact_label: str | None = None,
    hash_cache: Any | None = None,
    capture_field_times: bool = False,
) -> dict[str, Any]:
    if completion_policy not in ("waitsome", "waitall"):
        raise RuntimeError("unsupported MPI completion policy")
    ranks = _lane_ranks(lane)
    label = artifact_label or f"{run_id}-{lane}-{sample_kind}-{iteration}"
    result_path = output / f"{label}.json"
    stdout_path = output / f"{label}.stdout.log"
    stderr_path = output / f"{label}.stderr.log"
    timing_path = output / f"{label}.parent-timing.json"
    for path in (result_path, stdout_path, stderr_path, timing_path):
        path.unlink(missing_ok=True)
    nonce = uuid.uuid4().hex
    command = _expected_sample_command(
        repo=repo,
        runtime=runtime,
        receipt_id=receipt["receipt_id"],
        snapshot_sha256=snapshot_sha256,
        producer_sha256=producer_sha256,
        run_id=run_id,
        nonce=nonce,
        lane=lane,
        sample_kind=sample_kind,
        iteration=iteration,
        result_path=result_path,
        ranks=ranks,
        capture_field_times=capture_field_times,
    )
    environment = _child_environment(
        runtime, lane=lane, output=output, pycache_namespace=run_id
    )
    environment["MEEP_GPU_MPI_COMPLETION"] = completion_policy
    contention_tracker = _ACTIVE_HOST_CONTENTION_AUDIT
    contention_token = (
        contention_tracker.begin(
            label=label,
            lane=lane,
            sample_kind=sample_kind,
            iteration=iteration,
        )
        if isinstance(contention_tracker, _HostContentionAudit)
        else None
    )
    try:
        returncode, stdout, stderr, process_seconds = _run_command(
            command,
            cwd=repo,
            environment=environment,
            timeout_seconds=PROFILE["timeout_seconds"],
        )
    except CommandTimeout as error:
        atomic_write_text(stdout_path, error.stdout)
        atomic_write_text(stderr_path, error.stderr)
        raise RuntimeError(f"{label} exceeded the fixed MPI timeout") from error
    finally:
        if contention_token is not None:
            contention_tracker.end(contention_token)
    ledger_key = f"sample:{label}"
    publication_capability.record_measurement(
        ledger_key,
        {
            "role": "sample",
            "run_id": run_id,
            "output": str(output.resolve()),
            "label": label,
            "lane": lane,
            "sample_kind": sample_kind,
            "iteration": iteration,
            "nonce": nonce,
            "process_seconds": process_seconds,
            "command_sha256": canonical_sha256(command),
            "environment_sha256": canonical_sha256(
                dict(sorted(environment.items()))
            ),
            "artifact_paths": {
                "result_file": str(result_path.resolve()),
                "stdout_log": str(stdout_path.resolve()),
                "stderr_log": str(stderr_path.resolve()),
                "timing_file": str(timing_path.resolve()),
            },
        },
    )
    atomic_write_text(stdout_path, stdout)
    atomic_write_text(stderr_path, stderr)
    if returncode != 0:
        raise RuntimeError(f"{label} exited {returncode}; see {stderr_path}")
    timing_record = {
        "schema_version": 1,
        "run_id": run_id,
        "label": label,
        "lane": lane,
        "sample_kind": sample_kind,
        "iteration": iteration,
        "nonce": nonce,
        "process_seconds": process_seconds,
        "command_sha256": canonical_sha256(command),
    }
    atomic_write_json(timing_path, timing_record)
    pointer = _extract_pointer(stdout)
    if pathlib.Path(str(pointer.get("result_file", ""))).resolve() != result_path.resolve():
        raise RuntimeError(f"{label} returned the wrong result path")
    if not result_path.is_file() or pointer.get("sha256") != sha256_file(result_path):
        raise RuntimeError(f"{label} result digest is absent or invalid")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} raw result") from error
    COMPARATOR.validate_qualification_binding(
        result,
        expected_nonce=nonce,
        expected_run_id=run_id,
        expected_receipt_id=receipt["receipt_id"],
        expected_source_snapshot_sha256=snapshot_sha256,
        expected_producer_sha256=producer_sha256,
        expected_sample_kind=sample_kind,
        expected_sample_lane=lane,
        expected_sample_iteration=iteration,
        label=label,
        hash_cache=hash_cache,
    )
    producer_record = result["producer"]
    producer_command = producer_record.get("command")
    expected_producer = _expected_producer_command(
        repo=repo,
        runtime=runtime,
        receipt_id=receipt["receipt_id"],
        snapshot_sha256=snapshot_sha256,
        producer_sha256=producer_sha256,
        run_id=run_id,
        nonce=nonce,
        lane=lane,
        sample_kind=sample_kind,
        iteration=iteration,
        result_path=result_path,
        ranks=ranks,
        capture_field_times=capture_field_times,
    )
    if (
        not isinstance(producer_command, list)
        or len(producer_command) < 2
        or pathlib.Path(str(producer_command[0])).resolve() != runtime["python"].resolve()
        or producer_command != expected_producer
    ):
        raise RuntimeError(f"{label} raw producer command differs from the launched command")
    process_environment_gate = _validate_actual_process_environments(
        result, environment, label=label
    )
    _validate_lane_process_environment(
        result, environment, lane=lane, label=label
    )
    lazy_import_gate = _validate_lazy_import_contract(result, label=label)
    material_gradient_gate = _extract_material_gradient_stats(stdout, lane=lane)
    _fixed_workload_gate(result, lane=lane)
    if lane == CPU_ORACLE_LANE:
        validation = validate_cpu_oracle(result)
    elif lane == CPU_BENCHMARK_LANE:
        validation = validate_cpu_benchmark_record(result, label=label)
    elif lane in CUDA_LANES:
        validation = COMPARATOR.validate_distributed_record(
            result,
            expected_world_size=ranks,
            label=label,
            hash_cache=hash_cache,
        )
        validation["rank_driver"] = _validate_cuda_rank_driver_evidence(
            result, validation, label=label
        )
    receipt_gate = _validate_receipt_rank_runtimes(
        receipt, validation, hash_cache=hash_cache
    )
    if lane != CPU_ORACLE_LANE and process_seconds < _finite_positive(
        validation.get("workload_wall_seconds"), f"{label}.validated workload wall"
    ):
        raise RuntimeError(f"{label} parent process time is shorter than workload time")
    sample = {
        "lane": lane,
        "sample_kind": sample_kind,
        "iteration": iteration,
        "completion_policy": completion_policy,
        "capture_field_times": capture_field_times,
        "artifact_label": label,
        "ranks": ranks,
        "nonce": nonce,
        "process_seconds": process_seconds,
        "result": result,
        "validation": validation,
        "receipt_gate": receipt_gate,
        "process_environment_gate": process_environment_gate,
        "lazy_import_gate": lazy_import_gate,
        "material_gradient_gate": material_gradient_gate,
        "result_file": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "stdout_log": str(stdout_path.resolve()),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_log": str(stderr_path.resolve()),
        "stderr_sha256": sha256_file(stderr_path),
        "timing_file": str(timing_path.resolve()),
        "timing_sha256": sha256_file(timing_path),
        "command": command,
        "environment": dict(sorted(environment.items())),
    }
    publication_capability.record_artifacts(
        ledger_key,
        {
            "role": "sample",
            "label": label,
            "artifact_sha256": {
                "result_file": sample["result_sha256"],
                "stdout_log": sample["stdout_sha256"],
                "stderr_log": sample["stderr_sha256"],
                "timing_file": sample["timing_sha256"],
            },
        },
    )
    return sample


def _same_workload(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "resolution",
        "run_time",
        "cell_size",
        "design_resolution",
        "design_shape",
        "design_variables",
        "yee_cells",
        "weights_sha256",
        "direction_sha256",
    )
    return all(
        left.get("workload", {}).get(name) == right.get("workload", {}).get(name)
        for name in keys
    )


def validate_sample_matrix(samples: list[dict[str, Any]]) -> dict[str, Any]:
    expected: list[tuple[str, str, int]] = [(CPU_ORACLE_LANE, "measured", 0)]
    for iteration in range(PROFILE["warmups"]):
        expected.extend((lane, "warmup", iteration) for lane in MEASURED_LANES)
    for iteration in range(PROFILE["repeats"]):
        order = MEASURED_ORDERS[iteration % len(MEASURED_ORDERS)]
        expected.extend((lane, "measured", iteration) for lane in order)
    actual = [
        (sample.get("lane"), sample.get("sample_kind"), sample.get("iteration"))
        for sample in samples
    ]
    if actual != expected:
        raise RuntimeError("qualification sample matrix is incomplete or not interleaved")
    nonces = [sample.get("nonce") for sample in samples]
    if any(not isinstance(nonce, str) or not nonce for nonce in nonces):
        raise RuntimeError("qualification sample nonce is absent")
    if len(set(nonces)) != len(nonces):
        raise RuntimeError("qualification sample nonce was reused")
    return {
        "pass": True,
        "expected": [list(item) for item in expected],
        "actual": [list(item) for item in actual],
    }


def _phase_dft_gate(
    cpu: dict[str, Any], single: dict[str, Any], multi: dict[str, Any]
) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    overall = True
    for phase_name in ("forward", "adjoint"):
        cpu_dft = [
            rank["phases"][phase_name]["dft"]
            for rank in cpu["validation"]["rank_coverage"]
        ]
        single_dft = single["validation"]["rank_coverage"][0]["phases"][phase_name][
            "dft"
        ]
        multi_dft = [
            rank["phases"][phase_name]["dft"]
            for rank in multi["validation"]["rank_coverage"]
        ]
        expected_calls = PROFILE[f"{phase_name}_dft_calls"]
        expected_points = PROFILE[f"{phase_name}_dft_points"]
        cpu_points = sum(item["points"] for item in cpu_dft)
        multi_points = sum(item["points"] for item in multi_dft)
        cpu_point_ratio = cpu_points / expected_points
        point_ratio = multi_points / expected_points
        calls_match = (
            single_dft["calls"] == expected_calls
            and all(0 < item["calls"] <= expected_calls for item in cpu_dft)
            and max(item["calls"] for item in cpu_dft) == expected_calls
            and all(0 < item["calls"] <= expected_calls for item in multi_dft)
            and max(item["calls"] for item in multi_dft) == expected_calls
        )
        passed = (
            calls_match
            and single_dft["points"] == expected_points
            and cpu_point_ratio >= 1.0 / PROFILE["maximum_dft_point_ratio"]
            and cpu_point_ratio <= PROFILE["maximum_dft_point_ratio"]
            and point_ratio >= 1.0 / PROFILE["maximum_dft_point_ratio"]
            and point_ratio <= PROFILE["maximum_dft_point_ratio"]
        )
        phases[phase_name] = {
            "pass": passed,
            "cpu": cpu_dft,
            "single": single_dft,
            "multi": multi_dft,
            "expected_single_points": expected_points,
            "cpu_point_sum": cpu_points,
            "cpu_over_expected_single_point_ratio": cpu_point_ratio,
            "multi_point_sum": multi_points,
            "multi_over_expected_single_point_ratio": point_ratio,
        }
        overall = overall and passed
    return {"pass": overall, "phases": phases}


def _measured_samples_by_lane(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    measured = {
        lane: sorted(
            (
                sample
                for sample in samples
                if sample.get("lane") == lane and sample.get("sample_kind") == "measured"
            ),
            key=lambda sample: sample.get("iteration"),
        )
        for lane in MEASURED_LANES
    }
    if any(len(lane_samples) != PROFILE["repeats"] for lane_samples in measured.values()):
        raise RuntimeError("measured sample counts do not match the fixed profile")
    return measured


def _timing_summary(
    measured: dict[str, list[dict[str, Any]]], field: str
) -> dict[str, Any]:
    lane_values = {
        lane: [
            float(sample["validation"][field])
            if field != "process_seconds"
            else float(sample["process_seconds"])
            for sample in lane_samples
        ]
        for lane, lane_samples in measured.items()
    }
    lanes: dict[str, Any] = {}
    for lane, values in lane_values.items():
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise RuntimeError(f"{lane} timing vector contains invalid values")
        mean = statistics.mean(values)
        lanes[lane] = {
            "seconds": values,
            "median_seconds": statistics.median(values),
            "cv": statistics.pstdev(values) / mean,
        }
    pair_specs = (
        ("cpu_over_cuda_single", CPU_BENCHMARK_LANE, "cuda-single"),
        ("cpu_over_cuda_multi", CPU_BENCHMARK_LANE, "cuda-multi"),
        ("cuda_single_over_cuda_multi", "cuda-single", "cuda-multi"),
    )
    pairs = {
        name: {
            "numerator_lane": numerator,
            "denominator_lane": denominator,
            "median_speedup": lanes[numerator]["median_seconds"]
            / lanes[denominator]["median_seconds"],
            "conservative_speedup": min(lane_values[numerator])
            / max(lane_values[denominator]),
        }
        for name, numerator, denominator in pair_specs
    }
    return {"lanes": lanes, "comparisons": pairs}


def recompute_performance(samples: list[dict[str, Any]]) -> dict[str, Any]:
    measured = _measured_samples_by_lane(samples)
    workload_wall = _timing_summary(measured, "workload_wall_seconds")
    fresh_process_wall = _timing_summary(measured, "process_seconds")
    headline_pairs = workload_wall["comparisons"]
    timing_stable = all(
        lane["cv"] <= PROFILE["maximum_timing_cv"]
        for lane in workload_wall["lanes"].values()
    )
    cpu_gpu_pass = all(
        headline_pairs[name]["median_speedup"]
        >= PROFILE["minimum_cpu_gpu_median_speedup"]
        and headline_pairs[name]["conservative_speedup"]
        >= PROFILE["minimum_cpu_gpu_conservative_speedup"]
        for name in ("cpu_over_cuda_single", "cpu_over_cuda_multi")
    )
    multi_gpu_scaling = headline_pairs["cuda_single_over_cuda_multi"]
    multi_gpu_pass = (
        multi_gpu_scaling["median_speedup"] >= PROFILE["minimum_median_speedup"]
        and multi_gpu_scaling["conservative_speedup"]
        >= PROFILE["minimum_conservative_speedup"]
    )
    process_pairs = fresh_process_wall["comparisons"]
    process_cpu_gpu_pass = all(
        process_pairs[name]["median_speedup"]
        >= PROFILE["minimum_cpu_gpu_median_speedup"]
        and process_pairs[name]["conservative_speedup"]
        >= PROFILE["minimum_cpu_gpu_conservative_speedup"]
        for name in ("cpu_over_cuda_single", "cpu_over_cuda_multi")
    )
    process_multi_gpu = process_pairs["cuda_single_over_cuda_multi"]
    process_multi_gpu_pass = (
        process_multi_gpu["median_speedup"] >= PROFILE["minimum_median_speedup"]
        and process_multi_gpu["conservative_speedup"]
        >= PROFILE["minimum_conservative_speedup"]
    )
    return {
        "pass": (
            cpu_gpu_pass
            and multi_gpu_pass
            and process_cpu_gpu_pass
            and process_multi_gpu_pass
            and timing_stable
        ),
        "headline_metric": "barrier-delimited-max-rank-workload-wall-seconds",
        "workload_wall": workload_wall,
        "fresh_process_wall": fresh_process_wall,
        "minimum_median_speedup": PROFILE["minimum_median_speedup"],
        "minimum_conservative_speedup": PROFILE["minimum_conservative_speedup"],
        "minimum_cpu_gpu_median_speedup": PROFILE["minimum_cpu_gpu_median_speedup"],
        "minimum_cpu_gpu_conservative_speedup": PROFILE["minimum_cpu_gpu_conservative_speedup"],
        "maximum_timing_cv": PROFILE["maximum_timing_cv"],
    }


def compare_measured_samples(
    samples: list[dict[str, Any]], *, fd_oracle: dict[str, Any]
) -> dict[str, Any]:
    _assert_fixed_profile()
    validate_sample_matrix(samples)
    validate_fd_oracle_summary(fd_oracle)
    oracle = samples[0]
    measured = _measured_samples_by_lane(samples)
    cpus = measured[CPU_BENCHMARK_LANE]
    singles = measured["cuda-single"]
    multis = measured["cuda-multi"]
    lazy_import_contract_pass = all(
        sample.get("lazy_import_gate", {}).get("pass") is True
        for sample in samples
    )

    canonical_devices = multis[0]["validation"]["device_identifiers"]
    if len(canonical_devices) != PROFILE["multi_ranks"]:
        raise RuntimeError("canonical multi-GPU mapping is incomplete")
    device_mapping_pass = True
    inventory_digests_by_lane: dict[str, list[str]] = {}
    process_environment_digests_by_lane: dict[str, list[str]] = {}
    mapped_environment_digests_by_lane: dict[str, list[str]] = {}
    driver_mapping_by_lane: dict[str, list[dict[str, Any]]] = {}
    mapped_environment_stability_pass = True
    for sample in samples:
        if sample["lane"] not in CUDA_LANES:
            continue
        validation = sample["validation"]
        expected_devices = (
            canonical_devices if sample["lane"] == "cuda-multi" else canonical_devices[:1]
        )
        device_mapping_pass = device_mapping_pass and (
            validation["device_identifiers"] == expected_devices
        )
        inventory_digests = [
            rank.get("device_inventory_sha256")
            for rank in validation["rank_coverage"]
        ]
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in inventory_digests
        ):
            raise RuntimeError("validated CUDA inventory digest is absent")
        environment_digests = [
            rank.get("process_environment_sha256")
            for rank in validation["rank_coverage"]
        ]
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in environment_digests
        ):
            raise RuntimeError("validated rank process-environment digest is absent")
        rank_runtimes = validation.get("rank_runtimes")
        if not isinstance(rank_runtimes, list) or len(rank_runtimes) != len(
            validation["rank_coverage"]
        ):
            raise RuntimeError("validated rank runtime-closure matrix is absent")
        mapped_environment_digests: list[str] = []
        for runtime in rank_runtimes:
            records = (
                runtime.get("loaded_libraries", {}).get("environment")
                if isinstance(runtime, dict)
                else None
            )
            if not isinstance(records, list) or not records:
                raise RuntimeError("complete mapped-environment evidence is absent")
            projection = sorted(
                (record.get("path"), record.get("sha256"))
                for record in records
                if isinstance(record, dict)
            )
            if (
                len(projection) != len(records)
                or len(set(projection)) != len(projection)
                or any(
                    not isinstance(path, str)
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    for path, digest in projection
                )
            ):
                raise RuntimeError("mapped-environment evidence is malformed")
            mapped_environment_digests.append(canonical_sha256(projection))
        lane = sample["lane"]
        previous_inventories = inventory_digests_by_lane.setdefault(
            lane, inventory_digests
        )
        previous_environments = process_environment_digests_by_lane.setdefault(
            lane, environment_digests
        )
        previous_mapped_environment = mapped_environment_digests_by_lane.setdefault(
            lane, mapped_environment_digests
        )
        driver_mapping = validation.get("rank_driver")
        if not isinstance(driver_mapping, list) or len(driver_mapping) != len(
            validation["rank_coverage"]
        ):
            raise RuntimeError("validated rank-local NVIDIA driver mapping is absent")
        previous_driver_mapping = driver_mapping_by_lane.setdefault(
            lane, driver_mapping
        )
        if (
            inventory_digests != previous_inventories
            or environment_digests != previous_environments
            or driver_mapping != previous_driver_mapping
        ):
            device_mapping_pass = False
        if mapped_environment_digests != previous_mapped_environment:
            mapped_environment_stability_pass = False
    if not device_mapping_pass:
        raise RuntimeError("physical GPU mapping or visible-device inventory changed")
    if not mapped_environment_stability_pass:
        raise RuntimeError(
            "mapped environment dependency closure changed within a lane/rank"
        )

    cpu_affinity_projection = [sample["validation"]["affinity"] for sample in cpus]
    cpu_affinity_stability_pass = all(
        affinity == cpu_affinity_projection[0]
        for affinity in cpu_affinity_projection[1:]
    )
    if not cpu_affinity_stability_pass:
        raise RuntimeError("CPU core placement changed between measured samples")

    multi_transports = {
        coverage["transport"]
        for sample in samples
        if sample["lane"] == "cuda-multi"
        for coverage in sample["validation"]["rank_coverage"]
    }
    if len(multi_transports) != 1 or "none" in multi_transports:
        raise RuntimeError("MPI GPU-halo transport changed between ranks or repetitions")
    multi_transport = next(iter(multi_transports))

    comparisons: list[dict[str, Any]] = []
    dft_comparisons: list[dict[str, Any]] = []
    all_workloads = True
    all_objectives = True
    all_gradients = True
    all_dft = True
    oracle_result = oracle["result"]
    expected_count = int(oracle_result["workload"]["design_variables"])
    for cpu, single, multi in zip(cpus, singles, multis):
        if len({cpu["iteration"], single["iteration"], multi["iteration"]}) != 1:
            raise RuntimeError("CPU/single/multi measured iterations are mispaired")
        all_workloads = all_workloads and _same_workload(oracle_result, cpu["result"])
        all_workloads = all_workloads and _same_workload(cpu["result"], single["result"])
        all_workloads = all_workloads and _same_workload(single["result"], multi["result"])
        for label, candidate in (
            (CPU_BENCHMARK_LANE, cpu),
            ("cuda-single", single),
            ("cuda-multi", multi),
        ):
            objective_error = abs(
                oracle["validation"]["objective"]
                - candidate["validation"]["objective"]
            )
            objective_pass = objective_error <= PROFILE["objective_atol"]
            gradient = compare_gradient_vectors(
                oracle_result,
                candidate["result"],
                atol=PROFILE["gradient_atol"],
                rtol=PROFILE["gradient_rtol"],
                expected_count=expected_count,
            )
            comparisons.append(
                {
                    "iteration": cpu["iteration"],
                    "reference": CPU_ORACLE_LANE,
                    "candidate": label,
                    "objective_absolute_error": objective_error,
                    "objective_pass": objective_pass,
                    "gradient": gradient,
                }
            )
            all_objectives = all_objectives and objective_pass
            all_gradients = all_gradients and gradient["pass"]
        for label, candidate in (("cuda-single", single), ("cuda-multi", multi)):
            objective_error = abs(
                cpu["validation"]["objective"] - candidate["validation"]["objective"]
            )
            objective_pass = objective_error <= PROFILE["objective_atol"]
            gradient = compare_gradient_vectors(
                cpu["result"],
                candidate["result"],
                atol=PROFILE["gradient_atol"],
                rtol=PROFILE["gradient_rtol"],
                expected_count=expected_count,
            )
            comparisons.append(
                {
                    "iteration": cpu["iteration"],
                    "reference": CPU_BENCHMARK_LANE,
                    "candidate": label,
                    "objective_absolute_error": objective_error,
                    "objective_pass": objective_pass,
                    "gradient": gradient,
                }
            )
            all_objectives = all_objectives and objective_pass
            all_gradients = all_gradients and gradient["pass"]
        dft = _phase_dft_gate(cpu, single, multi)
        dft["iteration"] = cpu["iteration"]
        dft_comparisons.append(dft)
        all_dft = all_dft and dft["pass"]

    performance = recompute_performance(samples)
    gates = {
        "sample_matrix": True,
        "fixed_workload": all_workloads,
        "cpu_legacy_nonzero_full_gradient_oracle": True,
        "multi_direction_and_component_finite_difference_oracle": True,
        "objective": all_objectives,
        "gradient": all_gradients,
        "strict_phase_cuda_coverage": True,
        "strict_cpu_baseline_coverage": True,
        "stable_cpu_core_placement": cpu_affinity_stability_pass,
        "stable_physical_device_mapping": device_mapping_pass,
        "stable_mapped_environment_closure": mapped_environment_stability_pass,
        "stable_mpi_gpu_halo_transport": True,
        "absolute_dft_decimation": all_dft,
        "lazy_optional_imports": lazy_import_contract_pass,
        "performance": performance["pass"],
        "receipt_and_runtime_closure": True,
    }
    gates["pass"] = all(gates.values())
    return {
        "gate": gates,
        "performance": performance,
        "oracle": {
            "production_cpu_legacy": {
                "objective": oracle["validation"]["objective"],
                "gradient_sha256": oracle["result"]["result"]["gradient_sha256"],
                "material_gradient": oracle["material_gradient_gate"],
            },
            "finite_difference": fd_oracle,
        },
        "comparisons": comparisons,
        "dft": dft_comparisons,
        "devices": {
            "identifiers": canonical_devices,
            "inventory_sha256_by_lane_and_rank": inventory_digests_by_lane,
            "process_environment_sha256_by_lane_and_rank": process_environment_digests_by_lane,
            "mapped_environment_sha256_by_lane_and_rank": mapped_environment_digests_by_lane,
            "rank_local_driver_by_lane_and_rank": driver_mapping_by_lane,
            "transport": multi_transport,
        },
    }


def validate_driver_device_binding(
    driver_probe: dict[str, Any],
    selected_identifiers: list[str],
    topology_probe: dict[str, Any] | None = None,
    *,
    repo: pathlib.Path,
) -> dict[str, Any]:
    trusted_executable = _resolve_trusted_nvidia_smi()
    executable_records = [driver_probe.get("executable")]
    if topology_probe is not None:
        executable_records.append(topology_probe.get("executable"))
    expected_executable_record = file_record(trusted_executable, repo)
    if any(record != expected_executable_record for record in executable_records):
        raise RuntimeError("NVIDIA probe executable path/hash is absent or changed")
    if (
        driver_probe.get("available") is not True
        or driver_probe.get("exit_code") != 0
        or not isinstance(driver_probe.get("stdout"), str)
    ):
        raise RuntimeError("NVIDIA driver/device qualification probe failed")
    inventory_identifiers: set[str] = set()
    inventory_rows: list[dict[str, str]] = []
    for line in driver_probe["stdout"].splitlines():
        columns = [value.strip() for value in line.split(",")]
        if len(columns) != 6:
            raise RuntimeError("NVIDIA driver inventory row is malformed")
        raw_uuid = columns[1]
        normalized = re.sub(r"[^0-9a-f]", "", raw_uuid.lower().removeprefix("gpu-"))
        if re.fullmatch(r"[0-9a-f]{32}", normalized) is None:
            raise RuntimeError(f"NVIDIA driver UUID is unsupported: {raw_uuid}")
        if normalized in inventory_identifiers:
            raise RuntimeError("NVIDIA driver inventory contains a duplicate UUID")
        inventory_identifiers.add(normalized)
        inventory_rows.append(
            {
                "index": columns[0],
                "uuid": raw_uuid,
                "normalized_uuid": normalized,
                "name": columns[2],
                "driver_version": columns[3],
                "compute_capability": columns[4],
                "memory_total": columns[5],
            }
        )
    if not inventory_rows:
        raise RuntimeError("NVIDIA driver inventory is empty")
    topology_identifiers: set[str] = set()
    if topology_probe is not None:
        if (
            topology_probe.get("available") is not True
            or topology_probe.get("exit_code") != 0
            or not isinstance(topology_probe.get("stdout"), str)
        ):
            raise RuntimeError("NVIDIA topology/MIG qualification probe failed")
        for raw_uuid in re.findall(
            r"UUID:\s*((?:GPU|MIG)-[0-9A-Fa-f-]+)",
            topology_probe["stdout"],
        ):
            normalized = re.sub(
                r"[^0-9a-f]", "", raw_uuid.lower().split("-", 1)[1]
            )
            if re.fullmatch(r"[0-9a-f]{32}", normalized):
                topology_identifiers.add(normalized)
    selected = set(selected_identifiers)
    if len(selected) != len(selected_identifiers) or not selected.issubset(
        inventory_identifiers | topology_identifiers
    ):
        raise RuntimeError("CUDA runtime UUIDs do not match the NVIDIA driver inventory")
    return {
        "pass": True,
        "executable": expected_executable_record,
        "selected_identifiers": selected_identifiers,
        "inventory": inventory_rows,
        "topology_identifiers": sorted(topology_identifiers),
    }


def _resolve_trusted_nvidia_smi() -> pathlib.Path:
    for directory in TRUSTED_NVIDIA_SMI_DIRECTORIES:
        candidate = directory / "nvidia-smi"
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        try:
            resolved = candidate.resolve(strict=True)
            protected_paths = (directory.resolve(strict=True), resolved.parent, resolved)
            if any(
                path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                for path in protected_paths
            ):
                continue
        except OSError:
            continue
        return resolved
    raise RuntimeError("nvidia-smi is absent from the fixed protected system paths")


def _nvidia_smi_probe(arguments: list[str], repo: pathlib.Path) -> dict[str, Any]:
    executable = _resolve_trusted_nvidia_smi()
    probe = command_probe([str(executable), *arguments], repo)
    probe["executable"] = file_record(executable, repo)
    return probe


def _canonical_output_directory(output: pathlib.Path) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(os.fspath(output)))
    try:
        resolved = output.resolve(strict=True)
        metadata = output.lstat()
    except OSError as error:
        raise RuntimeError("qualification output directory is absent") from error
    if (
        lexical != resolved
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise RuntimeError(
            "qualification output must be a canonical non-symlink directory"
        )
    return resolved


def _require_running_state(output: pathlib.Path, run_id: str) -> None:
    state_path = output / "state.json"
    try:
        metadata = state_path.lstat()
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("current qualification RUNNING state is absent") from error
    expected = {"schema_version": 1, "state": "RUNNING", "run_id": run_id}
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or state != expected
    ):
        raise RuntimeError("current qualification RUNNING state is invalid")


def _require_owned_regular_file(
    value: Any,
    *,
    output: pathlib.Path,
    expected_name: str,
    label: str,
) -> pathlib.Path:
    expected = output / expected_name
    if not isinstance(value, str) or value != str(expected):
        raise RuntimeError(f"{label} is not the exact current-run artifact path")
    path = pathlib.Path(value)
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} current-run artifact is absent") from error
    if (
        resolved != expected
        or resolved.parent != output
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise RuntimeError(f"{label} is not an owned regular artifact")
    return resolved


def prepare_output(output: pathlib.Path, run_id: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    output = _canonical_output_directory(output)
    for name in FINAL_ARTIFACTS:
        (output / name).unlink(missing_ok=True)
    atomic_write_json(
        output / "state.json",
        {"schema_version": 1, "state": "RUNNING", "run_id": run_id},
    )


def valid_build_receipt_report_record(record: Any) -> bool:
    expected_keys = {
        "path",
        "sha256",
        "receipt_id",
        "build_input_id",
        "artifact_set_id",
        "qualification_contract",
    }
    return bool(
        isinstance(record, dict)
        and set(record) == expected_keys
        and isinstance(record.get("path"), str)
        and record["path"]
        and all(
            isinstance(record.get(name), str) and record[name]
            for name in ("receipt_id", "build_input_id", "artifact_set_id")
        )
        and isinstance(record.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None
        and record.get("qualification_contract") == BUILD_QUALIFICATION_CONTRACT
    )


def _reconstruct_unqualified_samples(
    *,
    report_samples: list[dict[str, Any]],
    receipt_record: dict[str, Any],
    source_snapshot_record: dict[str, Any],
    run_id: str,
    repo: pathlib.Path,
    output: pathlib.Path,
) -> list[dict[str, Any]]:
    output = _canonical_output_directory(output)
    receipt_path = pathlib.Path(receipt_record["path"]).resolve()
    if sha256_file(receipt_path) != receipt_record["sha256"]:
        raise RuntimeError("unqualified build receipt digest changed")
    receipt = verify_build_receipt(receipt_path, repo)
    if (
        receipt.get("receipt_id") != receipt_record["receipt_id"]
        or receipt.get("source_end") != source_snapshot_record
    ):
        raise RuntimeError("unqualified build/source receipt binding changed")
    runtime = _receipt_runtime(receipt, repo)
    producer_sha256 = sha256_file(repo / "scripts" / "benchmark-adjoint.py")
    reconstructed_samples: list[dict[str, Any]] = []
    for summary in report_samples:
        sample = {
            name: value
            for name, value in summary.items()
            if name not in {"raw_result_schema", "raw_gradient_sha256"}
        }
        artifact_label = sample["artifact_label"]
        paths: dict[str, pathlib.Path] = {}
        for path_name, digest_name, suffix in (
            ("result_file", "result_sha256", ".json"),
            ("stdout_log", "stdout_sha256", ".stdout.log"),
            ("stderr_log", "stderr_sha256", ".stderr.log"),
            ("timing_file", "timing_sha256", ".parent-timing.json"),
        ):
            path = _require_owned_regular_file(
                sample[path_name],
                output=output,
                expected_name=f"{artifact_label}{suffix}",
                label=f"unqualified {path_name}",
            )
            if sha256_file(path) != sample[digest_name]:
                raise RuntimeError(f"unqualified {path_name} digest changed")
            paths[path_name] = path
        result_path = paths["result_file"]
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        sample["result"] = raw
        if (
            set(sample) != EXPECTED_SAMPLE_KEYS
            or raw.get("schema_version") != summary.get("raw_result_schema")
            or canonical_float64_sha256(raw["result"]["gradient"])
            != summary.get("raw_gradient_sha256")
        ):
            raise RuntimeError("unqualified raw sample schema/digest disagrees")
        lane = sample["lane"]
        label = f"{lane}-{sample['sample_kind']}-{sample['iteration']}"
        expected_command = _expected_sample_command(
            repo=repo,
            runtime=runtime,
            receipt_id=receipt["receipt_id"],
            snapshot_sha256=source_snapshot_record["sha256"],
            producer_sha256=producer_sha256,
            run_id=run_id,
            nonce=sample["nonce"],
            lane=lane,
            sample_kind=sample["sample_kind"],
            iteration=sample["iteration"],
            result_path=result_path,
            ranks=sample["ranks"],
            capture_field_times=False,
        )
        expected_environment = _child_environment(
            runtime,
            lane=lane,
            output=result_path.parent,
            pycache_namespace=run_id,
            prepare_filesystem=False,
        )
        expected_environment["MEEP_GPU_MPI_COMPLETION"] = "waitsome"
        expected_environment = dict(sorted(expected_environment.items()))
        if (
            sample["command"] != expected_command
            or sample["environment"] != expected_environment
        ):
            raise RuntimeError(f"{label} unqualified launch binding changed")
        timing = json.loads(paths["timing_file"].read_text(encoding="utf-8"))
        if timing != {
            "schema_version": 1,
            "run_id": run_id,
            "label": sample["artifact_label"],
            "lane": lane,
            "sample_kind": sample["sample_kind"],
            "iteration": sample["iteration"],
            "nonce": sample["nonce"],
            "process_seconds": sample["process_seconds"],
            "command_sha256": canonical_sha256(expected_command),
        }:
            raise RuntimeError(f"{label} unqualified parent timing changed")
        stdout = paths["stdout_log"].read_text(encoding="utf-8")
        paths["stderr_log"].read_text(encoding="utf-8")
        pointer = _extract_pointer(stdout)
        if (
            pathlib.Path(str(pointer.get("result_file", ""))).resolve()
            != result_path
            or pointer.get("sha256") != sample["result_sha256"]
        ):
            raise RuntimeError(f"{label} unqualified raw pointer changed")
        COMPARATOR.validate_qualification_binding(
            raw,
            expected_nonce=sample["nonce"],
            expected_run_id=run_id,
            expected_receipt_id=receipt["receipt_id"],
            expected_source_snapshot_sha256=source_snapshot_record["sha256"],
            expected_producer_sha256=producer_sha256,
            expected_sample_kind=sample["sample_kind"],
            expected_sample_lane=lane,
            expected_sample_iteration=sample["iteration"],
            label=label,
        )
        if raw.get("producer", {}).get("command") != _expected_producer_command(
            repo=repo,
            runtime=runtime,
            receipt_id=receipt["receipt_id"],
            snapshot_sha256=source_snapshot_record["sha256"],
            producer_sha256=producer_sha256,
            run_id=run_id,
            nonce=sample["nonce"],
            lane=lane,
            sample_kind=sample["sample_kind"],
            iteration=sample["iteration"],
            result_path=result_path,
            ranks=sample["ranks"],
            capture_field_times=False,
        ):
            raise RuntimeError(f"{label} unqualified producer command changed")
        _fixed_workload_gate(raw, lane=lane)
        lazy = _validate_lazy_import_contract(raw, label=label)
        if lane == CPU_ORACLE_LANE:
            validation = validate_cpu_oracle(raw)
        elif lane == CPU_BENCHMARK_LANE:
            validation = validate_cpu_benchmark_record(raw, label=label)
        else:
            validation = COMPARATOR.validate_distributed_record(
                raw, expected_world_size=sample["ranks"], label=label
            )
            validation["rank_driver"] = _validate_cuda_rank_driver_evidence(
                raw, validation, label=label
            )
        receipt_gate = _validate_receipt_rank_runtimes(receipt, validation)
        if lane != CPU_ORACLE_LANE and sample["process_seconds"] < _finite_positive(
            validation.get("workload_wall_seconds"),
            f"{label}.validated workload wall",
        ):
            raise RuntimeError(
                f"{label} unqualified parent process time is shorter than workload"
            )
        process_environment = _validate_actual_process_environments(
            raw, expected_environment, label=label
        )
        _validate_lane_process_environment(
            raw, expected_environment, lane=lane, label=label
        )
        material_gradient = _extract_material_gradient_stats(stdout, lane=lane)
        if (
            sample["validation"] != validation
            or sample["receipt_gate"] != receipt_gate
            or sample["process_environment_gate"] != process_environment
            or sample["lazy_import_gate"] != lazy
            or sample["material_gradient_gate"] != material_gradient
        ):
            raise RuntimeError(f"{label} unqualified derived evidence changed")
        reconstructed_samples.append(sample)
    return reconstructed_samples


def _validate_unqualified_fd_semantics(
    *,
    fd_oracle: dict[str, Any],
    receipt_record: dict[str, Any],
    source_snapshot_record: dict[str, Any],
    run_id: str,
    output: pathlib.Path,
    expected_test: pathlib.Path,
    paths: dict[str, pathlib.Path],
) -> None:
    repo = _SCRIPT_DIRECTORY.parent
    receipt_path = pathlib.Path(receipt_record["path"]).resolve()
    receipt_payload = verify_build_receipt(receipt_path, repo)
    receipt_contract = validate_build_receipt_contract(receipt_payload, repo)
    if (
        build_receipt_report_record(receipt_path, receipt_payload, receipt_contract)
        != receipt_record
        or receipt_payload.get("source_end") != source_snapshot_record
    ):
        raise RuntimeError("finite-difference receipt closure changed")
    runtime = _receipt_runtime(receipt_payload, repo)
    command = [str(runtime["python"]), str(expected_test), "-v"]
    environment = dict(
        sorted(
            _child_environment(
                runtime,
                lane=CPU_ORACLE_LANE,
                output=output,
                pycache_namespace=run_id,
                prepare_filesystem=False,
            ).items()
        )
    )
    timing = json.loads(paths["timing_file"].read_text(encoding="utf-8"))
    if (
        fd_oracle.get("command") != command
        or fd_oracle.get("environment") != environment
        or timing
        != {
            "schema_version": 1,
            "run_id": run_id,
            "label": f"{run_id}-directional-fd-oracle",
            "process_seconds": fd_oracle.get("process_seconds"),
            "started_at_utc": fd_oracle.get("started_at_utc"),
            "completed_at_utc": fd_oracle.get("completed_at_utc"),
            "command_sha256": canonical_sha256(command),
            "environment_sha256": canonical_sha256(environment),
        }
        or fd_oracle.get("direction_records")
        != _parse_fd_oracle_output(
            paths["stdout_log"].read_text(encoding="utf-8"),
            paths["stderr_log"].read_text(encoding="utf-8"),
        )
    ):
        raise RuntimeError("finite-difference derived evidence changed")


def validate_unqualified_report(
    run_id: str,
    report: Any,
    *,
    output: pathlib.Path,
) -> dict[str, Any]:
    expected_sample_count = 1 + len(MEASURED_LANES) * (
        PROFILE["warmups"] + PROFILE["repeats"]
    )
    expected_top_level = {
        "schema_version",
        "state",
        "evidence_class",
        "run_id",
        "profile",
        "build_receipt",
        "finite_difference_oracle",
        "host_audit",
        "integrity_reverification",
        "sample_matrix",
        "samples",
        "comparison",
        "gate",
    }
    expected_gate_names = {
        "sample_matrix",
        "fixed_workload",
        "cpu_legacy_nonzero_full_gradient_oracle",
        "multi_direction_and_component_finite_difference_oracle",
        "objective",
        "gradient",
        "strict_phase_cuda_coverage",
        "strict_cpu_baseline_coverage",
        "stable_cpu_core_placement",
        "stable_physical_device_mapping",
        "stable_mapped_environment_closure",
        "stable_mpi_gpu_halo_transport",
        "absolute_dft_decimation",
        "lazy_optional_imports",
        "performance",
        "receipt_and_runtime_closure",
        "host_topology_power_load",
        "pass",
    }
    if not isinstance(report, dict):
        raise RuntimeError("unqualified MPI evidence is not a JSON object")
    comparison = report.get("comparison")
    gate = report.get("gate")
    samples = report.get("samples")
    receipt = report.get("build_receipt")
    integrity = report.get("integrity_reverification")
    fd_oracle = report.get("finite_difference_oracle")
    if (
        set(report) != expected_top_level
        or report.get("schema_version") != 1
        or report.get("state") != "UNQUALIFIED"
        or report.get("evidence_class") != "failed-release-qualification"
        or report.get("run_id") != run_id
        or report.get("profile") != dict(PROFILE)
        or not valid_build_receipt_report_record(receipt)
        or not isinstance(comparison, dict)
        or set(comparison)
        != {
            "gate",
            "performance",
            "oracle",
            "comparisons",
            "dft",
            "devices",
            "host",
        }
        or not isinstance(gate, dict)
        or set(gate) != expected_gate_names
        or any(not isinstance(value, bool) for value in gate.values())
        or comparison.get("gate") != gate
        or gate.get("pass") is not False
        or gate["pass"]
        != all(value for name, value in gate.items() if name != "pass")
        or not isinstance(comparison.get("performance"), dict)
        or comparison["performance"].get("pass") is not gate["performance"]
        or not isinstance(comparison.get("oracle"), dict)
        or not isinstance(comparison["oracle"].get("production_cpu_legacy"), dict)
        or comparison["oracle"].get("finite_difference") != fd_oracle
        or not isinstance(comparison.get("comparisons"), list)
        or not isinstance(comparison.get("dft"), list)
        or not isinstance(comparison.get("devices"), dict)
        or not isinstance(integrity, dict)
        or set(integrity)
        != {
            "pass",
            "verified_at_utc",
            "build_receipt",
            "receipt_id",
            "receipt_sha256",
            "source_snapshot",
            "contract",
            "samples",
            "finite_difference_logs",
            "controller_ledger",
            "output_inventory",
        }
        or integrity.get("pass") is not True
        or not isinstance(integrity.get("verified_at_utc"), str)
        or not integrity["verified_at_utc"]
        or integrity.get("receipt_id") != receipt.get("receipt_id")
        or integrity.get("receipt_sha256") != receipt.get("sha256")
        or integrity.get("build_receipt") != receipt
        or not isinstance(integrity.get("source_snapshot"), dict)
        or not isinstance(integrity.get("contract"), dict)
        or integrity["contract"].get("pass") is not True
        or integrity["contract"].get("contract")
        != receipt.get("qualification_contract")
        or not isinstance(integrity.get("samples"), list)
        or len(integrity["samples"]) != expected_sample_count
        or not isinstance(integrity.get("finite_difference_logs"), dict)
        or not isinstance(integrity.get("controller_ledger"), dict)
        or set(integrity["controller_ledger"]) != {"pass", "entries", "sha256"}
        or integrity["controller_ledger"].get("pass") is not True
        or integrity["controller_ledger"].get("entries") != expected_sample_count + 1
        or not isinstance(integrity.get("output_inventory"), dict)
        or set(integrity["output_inventory"])
        != {"pass", "epoch", "manifest", "sha256"}
        or integrity["output_inventory"].get("pass") is not True
        or integrity["output_inventory"].get("epoch")
        != "prepublication-running"
        or not isinstance(samples, list)
        or len(samples) != expected_sample_count
        or any(
            not isinstance(sample, dict)
            or not isinstance(sample.get("nonce"), str)
            or not sample["nonce"]
            or sample.get("raw_result_schema") != 4
            or not isinstance(sample.get("raw_gradient_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", sample["raw_gradient_sha256"])
            is None
            or isinstance(sample.get("process_seconds"), bool)
            or not isinstance(sample.get("process_seconds"), (int, float))
            or not math.isfinite(sample["process_seconds"])
            or sample["process_seconds"] <= 0
            for sample in samples
        )
    ):
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")
    try:
        recomputed_matrix = validate_sample_matrix(samples)
        validate_fd_oracle_summary(fd_oracle)
    except RuntimeError as error:
        raise RuntimeError(
            "refusing to write incomplete unqualified MPI evidence"
        ) from error
    if canonical_sha256(report.get("sample_matrix")) != canonical_sha256(
        recomputed_matrix
    ):
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")

    try:
        expected_host = validate_host_audit(
            report.get("host_audit"),
            samples=samples,
            repo=_SCRIPT_DIRECTORY.parent,
            verify_live=False,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "refusing to write incomplete unqualified MPI evidence"
        ) from error
    if comparison.get("host") != expected_host:
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")

    def valid_sha256(value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    source_snapshot = integrity["source_snapshot"]
    if not valid_sha256(source_snapshot.get("sha256")):
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")
    required_fd_fields = {
        "receipt_id",
        "source_snapshot_sha256",
        "test_file",
        "test_sha256",
        "methods",
        "stdout_log",
        "stdout_sha256",
        "stderr_log",
        "stderr_sha256",
        "timing_file",
        "timing_sha256",
    }
    if (
        not required_fd_fields.issubset(fd_oracle)
        or fd_oracle["receipt_id"] != receipt["receipt_id"]
        or fd_oracle["source_snapshot_sha256"] != source_snapshot["sha256"]
        or fd_oracle["methods"] != list(FD_TEST_METHODS)
        or not isinstance(fd_oracle["test_file"], str)
        or not fd_oracle["test_file"]
        or not valid_sha256(fd_oracle["test_sha256"])
    ):
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")
    expected_fd_test = (
        _SCRIPT_DIRECTORY.parent
        / "python"
        / "tests"
        / "test_adjoint_default_material_grid.py"
    ).resolve()
    try:
        fd_test = pathlib.Path(fd_oracle["test_file"]).resolve(strict=True)
        if (
            fd_test != expected_fd_test
            or fd_test.is_symlink()
            or not fd_test.is_file()
            or sha256_file(fd_test) != fd_oracle["test_sha256"]
        ):
            raise RuntimeError("finite-difference test changed")
        fd_label = f"{run_id}-directional-fd-oracle"
        fd_paths: dict[str, pathlib.Path] = {}
        for path_name, digest_name, suffix in (
            ("stdout_log", "stdout_sha256", ".stdout.log"),
            ("stderr_log", "stderr_sha256", ".stderr.log"),
            ("timing_file", "timing_sha256", ".parent-timing.json"),
        ):
            path = _require_owned_regular_file(
                fd_oracle[path_name],
                output=output,
                expected_name=f"{fd_label}{suffix}",
                label=f"unqualified finite-difference {path_name}",
            )
            if sha256_file(path) != fd_oracle[digest_name]:
                raise RuntimeError("finite-difference artifact digest changed")
            path.read_text(encoding="utf-8")
            fd_paths[path_name] = path
        _validate_unqualified_fd_semantics(
            fd_oracle=fd_oracle,
            receipt_record=receipt,
            source_snapshot_record=source_snapshot,
            run_id=run_id,
            output=output,
            expected_test=expected_fd_test,
            paths=fd_paths,
        )
    except (OSError, RuntimeError, UnicodeDecodeError) as error:
        raise RuntimeError(
            "refusing to write incomplete unqualified MPI evidence"
        ) from error

    required_sample_fields = (EXPECTED_SAMPLE_KEYS - {"result"}) | {
        "raw_result_schema",
        "raw_gradient_sha256",
    }
    file_fields = (
        ("result_file", "result_sha256"),
        ("stdout_log", "stdout_sha256"),
        ("stderr_log", "stderr_sha256"),
        ("timing_file", "timing_sha256"),
    )
    for sample, verified in zip(samples, integrity["samples"]):
        if (
            set(sample) != required_sample_fields
            or not isinstance(sample["validation"], dict)
            or not isinstance(sample["receipt_gate"], dict)
            or sample["receipt_gate"].get("receipt_id") != receipt["receipt_id"]
            or not isinstance(sample["process_environment_gate"], dict)
            or not isinstance(sample["material_gradient_gate"], dict)
            or not isinstance(sample["command"], list)
            or not sample["command"]
            or not isinstance(sample["environment"], dict)
            or not isinstance(verified, dict)
            or set(verified)
            != {
                "label",
                "nonce",
                "process_seconds",
                "summary_sha256",
                "files",
            }
            or verified["label"]
            != f"{sample['lane']}-{sample['sample_kind']}-{sample['iteration']}"
            or verified["nonce"] != sample["nonce"]
            or verified["process_seconds"] != sample["process_seconds"]
            or verified["summary_sha256"] != canonical_sha256(sample)
            or not isinstance(verified["files"], dict)
            or set(verified["files"]) != {name for name, _ in file_fields}
        ):
            raise RuntimeError("refusing to write incomplete unqualified MPI evidence")
        for path_name, digest_name in file_fields:
            record = verified["files"].get(path_name)
            if (
                not isinstance(sample[path_name], str)
                or not sample[path_name]
                or not valid_sha256(sample[digest_name])
                or not isinstance(record, dict)
                or set(record) != {"path", "sha256"}
                or record["path"] != sample[path_name]
                or record["sha256"] != sample[digest_name]
            ):
                raise RuntimeError(
                    "refusing to write incomplete unqualified MPI evidence"
                )

    fd_logs = integrity["finite_difference_logs"]
    if set(fd_logs) != {"stdout_log", "stderr_log", "timing_file"}:
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")
    for path_name, digest_name in (
        ("stdout_log", "stdout_sha256"),
        ("stderr_log", "stderr_sha256"),
        ("timing_file", "timing_sha256"),
    ):
        record = fd_logs.get(path_name)
        if (
            not isinstance(fd_oracle[path_name], str)
            or not fd_oracle[path_name]
            or not valid_sha256(fd_oracle[digest_name])
            or not isinstance(record, dict)
            or set(record) != {"path", "sha256"}
            or record["path"] != fd_oracle[path_name]
            or record["sha256"] != fd_oracle[digest_name]
        ):
            raise RuntimeError("refusing to write incomplete unqualified MPI evidence")

    controller_ledger = integrity["controller_ledger"]
    if not valid_sha256(controller_ledger.get("sha256")):
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")
    output_inventory = integrity["output_inventory"]
    manifest = output_inventory.get("manifest")
    if (
        output_inventory.get("epoch") != "prepublication-running"
        or not valid_sha256(output_inventory.get("sha256"))
        or not isinstance(manifest, dict)
        or set(manifest) != {"files", "runtime_directories"}
        or canonical_sha256(manifest) != output_inventory["sha256"]
        or not isinstance(manifest.get("files"), list)
        or not isinstance(manifest.get("runtime_directories"), list)
        or any(
            name not in {"cache", "home", "matplotlib", "pycache", "tmp"}
            for name in manifest["runtime_directories"]
        )
        or len(set(manifest["runtime_directories"]))
        != len(manifest["runtime_directories"])
    ):
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")
    expected_inventory_paths = {
        record["path"]
        for verified in integrity["samples"]
        for record in verified["files"].values()
    } | {record["path"] for record in fd_logs.values()}
    artifact_parents = {str(pathlib.Path(path).parent) for path in expected_inventory_paths}
    if len(artifact_parents) != 1:
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")
    expected_inventory_paths.add(
        str(pathlib.Path(next(iter(expected_inventory_paths))).parent / "state.json")
    )
    inventory_paths: set[str] = set()
    for record in manifest["files"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256"}
            or not isinstance(record.get("path"), str)
            or not record["path"]
            or not valid_sha256(record.get("sha256"))
            or record["path"] in inventory_paths
        ):
            raise RuntimeError("refusing to write incomplete unqualified MPI evidence")
        inventory_paths.add(record["path"])
    if inventory_paths != expected_inventory_paths:
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")

    performance = comparison["performance"]
    try:
        recomputed_performance = recompute_performance(samples)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "refusing to write incomplete unqualified MPI evidence"
        ) from error
    if (
        performance != recomputed_performance
        or performance["pass"] is not gate["performance"]
    ):
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")

    try:
        reconstructed_samples = _reconstruct_unqualified_samples(
            report_samples=samples,
            receipt_record=receipt,
            source_snapshot_record=source_snapshot,
            run_id=run_id,
            repo=_SCRIPT_DIRECTORY.parent,
            output=output,
        )
        recomputed_comparison = compare_measured_samples(
            reconstructed_samples, fd_oracle=fd_oracle
        )
        bind_host_audit(
            recomputed_comparison,
            report.get("host_audit"),
            samples=reconstructed_samples,
            repo=_SCRIPT_DIRECTORY.parent,
            verify_live=False,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "refusing to write incomplete unqualified MPI evidence"
        ) from error
    if comparison != recomputed_comparison:
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")

    production_oracle = comparison["oracle"]["production_cpu_legacy"]
    devices = comparison["devices"]
    comparison_inventory = [
        (item.get("iteration"), item.get("reference"), item.get("candidate"))
        for item in comparison["comparisons"]
        if isinstance(item, dict)
    ]
    expected_comparison_inventory = [
        (iteration, reference, candidate)
        for iteration in range(PROFILE["repeats"])
        for reference, candidate in (
            (CPU_ORACLE_LANE, CPU_BENCHMARK_LANE),
            (CPU_ORACLE_LANE, "cuda-single"),
            (CPU_ORACLE_LANE, "cuda-multi"),
            (CPU_BENCHMARK_LANE, "cuda-single"),
            (CPU_BENCHMARK_LANE, "cuda-multi"),
        )
    ]
    dft_inventory = [
        item.get("iteration")
        for item in comparison["dft"]
        if isinstance(item, dict)
    ]
    if (
        set(production_oracle)
        != {"objective", "gradient_sha256", "material_gradient"}
        or isinstance(production_oracle["objective"], bool)
        or not isinstance(production_oracle["objective"], (int, float))
        or not math.isfinite(production_oracle["objective"])
        or not valid_sha256(production_oracle["gradient_sha256"])
        or not isinstance(production_oracle["material_gradient"], dict)
        or len(comparison["comparisons"]) != 5 * PROFILE["repeats"]
        or any(
            not isinstance(item, dict)
            or set(item)
            != {
                "iteration",
                "reference",
                "candidate",
                "objective_absolute_error",
                "objective_pass",
                "gradient",
            }
            for item in comparison["comparisons"]
        )
        or comparison_inventory != expected_comparison_inventory
        or len(comparison["dft"]) != PROFILE["repeats"]
        or any(
            not isinstance(item, dict)
            or set(item) != {"pass", "phases", "iteration"}
            or not isinstance(item["pass"], bool)
            or not isinstance(item["phases"], dict)
            for item in comparison["dft"]
        )
        or dft_inventory != list(range(PROFILE["repeats"]))
        or gate["absolute_dft_decimation"]
        is not all(item["pass"] for item in comparison["dft"])
        or set(devices)
        != {
            "identifiers",
            "inventory_sha256_by_lane_and_rank",
            "process_environment_sha256_by_lane_and_rank",
            "mapped_environment_sha256_by_lane_and_rank",
            "rank_local_driver_by_lane_and_rank",
            "transport",
        }
        or not isinstance(devices["identifiers"], list)
        or len(devices["identifiers"]) != PROFILE["multi_ranks"]
        or not isinstance(devices["transport"], str)
        or not devices["transport"]
        or devices["transport"] == "none"
        or any(
            not isinstance(devices[name], dict) or not devices[name]
            for name in (
                "inventory_sha256_by_lane_and_rank",
                "process_environment_sha256_by_lane_and_rank",
                "mapped_environment_sha256_by_lane_and_rank",
                "rank_local_driver_by_lane_and_rank",
            )
        )
        or any(
            set(devices[name]) != set(CUDA_LANES)
            for name in (
                "inventory_sha256_by_lane_and_rank",
                "process_environment_sha256_by_lane_and_rank",
                "mapped_environment_sha256_by_lane_and_rank",
                "rank_local_driver_by_lane_and_rank",
            )
        )
    ):
        raise RuntimeError("refusing to write incomplete unqualified MPI evidence")
    return report


def write_unqualified_report(
    output: pathlib.Path,
    run_id: str,
    report: dict[str, Any],
) -> dict[str, str]:
    validate_unqualified_report(
        run_id,
        report,
        output=output,
    )
    path = output / UNQUALIFIED_REPORT_NAME
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    payload_bytes = payload.encode("utf-8")
    expected_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    atomic_write_text(path, payload)
    if path.read_bytes() != payload_bytes:
        raise RuntimeError("unqualified MPI evidence changed during publication")
    return {"path": str(path.resolve()), "sha256": expected_sha256}


def build_receipt_report_record(
    receipt_path: pathlib.Path,
    receipt: dict[str, Any],
    qualification_contract_validation: dict[str, Any],
) -> dict[str, str]:
    """Return the stable receipt identity embedded in benchmark reports.

    The detailed validation result belongs to the independently recomputed
    integrity closure.  Keeping it out of this identity record prevents the
    contract name from silently changing type between producer and validator.
    """

    if (
        not isinstance(qualification_contract_validation, dict)
        or qualification_contract_validation.get("pass") is not True
        or qualification_contract_validation.get("contract")
        != BUILD_QUALIFICATION_CONTRACT
    ):
        raise RuntimeError("build qualification-contract validation is incomplete")
    required_receipt_fields = ("receipt_id", "build_input_id", "artifact_set_id")
    if any(
        not isinstance(receipt.get(name), str) or not receipt[name]
        for name in required_receipt_fields
    ):
        raise RuntimeError("build receipt identity is incomplete")
    return {
        "path": str(receipt_path.resolve()),
        "sha256": sha256_file(receipt_path),
        "receipt_id": receipt["receipt_id"],
        "build_input_id": receipt["build_input_id"],
        "artifact_set_id": receipt["artifact_set_id"],
        "qualification_contract": BUILD_QUALIFICATION_CONTRACT,
    }


class QualificationGateFailure(RuntimeError):
    def __init__(self, message: str, unqualified_report: dict[str, str]):
        super().__init__(message)
        self.unqualified_report = dict(unqualified_report)


def mark_failed(output: pathlib.Path, run_id: str, error: BaseException) -> None:
    global _ACTIVE_HOST_CONTENTION_AUDIT
    _ACTIVE_HOST_CONTENTION_AUDIT = None
    for name in ("report.json", "report.md", "COMPLETE"):
        (output / name).unlink(missing_ok=True)
    failure = {
        "schema_version": 1,
        "state": "FAILED",
        "run_id": run_id,
        "profile": dict(PROFILE),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }
    unqualified_path = output / UNQUALIFIED_REPORT_NAME
    if isinstance(error, QualificationGateFailure) or unqualified_path.exists():
        try:
            if not isinstance(error, QualificationGateFailure):
                raise RuntimeError("original unqualified evidence digest is unavailable")
            if not unqualified_path.is_file():
                raise RuntimeError("published unqualified evidence file is missing")
            expected_record = error.unqualified_report
            payload = unqualified_path.read_bytes()
            actual_record = {
                "path": str(unqualified_path.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            if expected_record != actual_record:
                raise RuntimeError(
                    "unqualified evidence changed after publication: "
                    f"expected {expected_record!r}, observed {actual_record!r}"
                )
            unqualified = json.loads(payload)
            validate_unqualified_report(run_id, unqualified, output=output)
            failure["unqualified_report"] = actual_record
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RuntimeError,
        ) as unqualified_error:
            failure["unqualified_report_rejected"] = str(unqualified_error)
    atomic_write_json(output / "FAILED.json", failure)
    atomic_write_json(output / "state.json", failure)


def summarize_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized_samples = []
    for sample in samples:
        summarized = {name: value for name, value in sample.items() if name != "result"}
        summarized["raw_result_schema"] = sample["result"].get("schema_version")
        summarized["raw_gradient_sha256"] = canonical_float64_sha256(
            sample["result"]["result"]["gradient"]
        )
        summarized_samples.append(summarized)
    return summarized_samples


def _validate_controller_ledger(
    capability: _RunPublicationCapability,
    *,
    output: pathlib.Path,
    run_id: str,
    samples: list[dict[str, Any]],
    fd_oracle: dict[str, Any],
) -> dict[str, Any]:
    ledger = capability.snapshot(output, run_id)
    expected = _expected_controller_ledger(
        output=output,
        run_id=run_id,
        samples=samples,
        fd_oracle=fd_oracle,
    )
    if ledger != expected:
        raise RuntimeError("controller-owned publication ledger disagrees")
    return {
        "pass": True,
        "entries": len(expected["measurements"]),
        "sha256": canonical_sha256(expected),
    }


def _expected_controller_ledger(
    *,
    output: pathlib.Path,
    run_id: str,
    samples: list[dict[str, Any]],
    fd_oracle: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the controller ledger for independent consumers."""
    measurements: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != EXPECTED_SAMPLE_KEYS:
            raise RuntimeError("qualification sample exact schema disagrees")
        label = sample["artifact_label"]
        key = f"sample:{label}"
        measurements[key] = {
            "role": "sample",
            "run_id": run_id,
            "output": str(output),
            "label": label,
            "lane": sample["lane"],
            "sample_kind": sample["sample_kind"],
            "iteration": sample["iteration"],
            "nonce": sample["nonce"],
            "process_seconds": sample["process_seconds"],
            "command_sha256": canonical_sha256(sample["command"]),
            "environment_sha256": canonical_sha256(sample["environment"]),
            "artifact_paths": {
                name: sample[name]
                for name in (
                    "result_file",
                    "stdout_log",
                    "stderr_log",
                    "timing_file",
                )
            },
        }
        artifacts[key] = {
            "role": "sample",
            "label": label,
            "artifact_sha256": {
                "result_file": sample["result_sha256"],
                "stdout_log": sample["stdout_sha256"],
                "stderr_log": sample["stderr_sha256"],
                "timing_file": sample["timing_sha256"],
            },
        }
    fd_label = f"{run_id}-directional-fd-oracle"
    fd_key = f"fd:{fd_label}"
    measurements[fd_key] = {
        "role": "finite-difference-oracle",
        "run_id": run_id,
        "output": str(output),
        "label": fd_label,
        "process_seconds": fd_oracle.get("process_seconds"),
        "started_at_utc": fd_oracle.get("started_at_utc"),
        "completed_at_utc": fd_oracle.get("completed_at_utc"),
        "command_sha256": canonical_sha256(fd_oracle.get("command")),
        "environment_sha256": canonical_sha256(fd_oracle.get("environment")),
        "artifact_paths": {
            name: fd_oracle.get(name)
            for name in ("stdout_log", "stderr_log", "timing_file")
        },
    }
    artifacts[fd_key] = {
        "role": "finite-difference-oracle",
        "label": fd_label,
        "artifact_sha256": {
            "stdout_log": fd_oracle.get("stdout_sha256"),
            "stderr_log": fd_oracle.get("stderr_sha256"),
            "timing_file": fd_oracle.get("timing_sha256"),
        },
    }
    return {"measurements": measurements, "artifacts": artifacts}


def _validate_output_inventory(
    output: pathlib.Path, owned_paths: set[pathlib.Path]
) -> dict[str, Any]:
    allowed_runtime_directories = {"cache", "home", "matplotlib", "pycache", "tmp"}
    state_path = output / "state.json"
    expected_files = set(owned_paths) | {state_path}
    observed_files: set[pathlib.Path] = set()
    observed_directories: list[str] = []
    for path in output.iterdir():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("qualification output inventory contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            observed_files.add(path.resolve())
        elif stat.S_ISDIR(metadata.st_mode) and path.name in allowed_runtime_directories:
            observed_directories.append(path.name)
        else:
            raise RuntimeError("qualification output inventory contains an unbound entry")
    if observed_files != expected_files:
        raise RuntimeError("qualification output inventory contains unbound artifacts")
    manifest = {
        "files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(observed_files, key=str)
        ],
        "runtime_directories": sorted(observed_directories),
    }
    return {
        "pass": True,
        "epoch": "prepublication-running",
        "manifest": manifest,
        "sha256": canonical_sha256(manifest),
    }


def _build_terminal_output_inventory(
    output: pathlib.Path,
    prepublication_inventory: dict[str, Any],
    *,
    expected_run_id: str,
) -> dict[str, Any]:
    """Seal the completed topology without trying to hash the marker itself.

    ``COMPLETE`` and ``state.json`` contain the same terminal marker bytes, so
    hashing either from inside that marker would be recursive.  The marker
    instead hashes every other final file and declares the exact two equal
    marker filenames plus the exact top-level topology.  A consumer can then
    replay every byte and the marker-equality contract without ambiguity.
    """

    output = _canonical_output_directory(output)
    if (
        not isinstance(prepublication_inventory, dict)
        or set(prepublication_inventory) != {"pass", "epoch", "manifest", "sha256"}
        or prepublication_inventory.get("pass") is not True
        or prepublication_inventory.get("epoch") != "prepublication-running"
        or not isinstance(prepublication_inventory.get("manifest"), dict)
        or canonical_sha256(prepublication_inventory["manifest"])
        != prepublication_inventory.get("sha256")
    ):
        raise RuntimeError("prepublication output inventory is invalid")
    pre_manifest = prepublication_inventory["manifest"]
    allowed_runtime_directories = {"cache", "home", "matplotlib", "pycache", "tmp"}
    if (
        set(pre_manifest) != {"files", "runtime_directories"}
        or not isinstance(pre_manifest.get("files"), list)
        or not isinstance(pre_manifest.get("runtime_directories"), list)
        or any(
            not isinstance(name, str) or name not in allowed_runtime_directories
            for name in pre_manifest.get("runtime_directories", [])
        )
        or pre_manifest.get("runtime_directories")
        != sorted(set(pre_manifest.get("runtime_directories", [])))
    ):
        raise RuntimeError("prepublication output inventory schema is invalid")

    running_state = output / "state.json"
    immutable_paths: set[pathlib.Path] = set()
    prepublication_names: list[str] = []
    saw_running_state = False
    for record in pre_manifest["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise RuntimeError("prepublication output file record is invalid")
        if (
            not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise RuntimeError("prepublication output file record is invalid")
        path = pathlib.Path(record["path"])
        if (
            path.parent != output
            or path.name in prepublication_names
            or path.name in {"report.json", "report.md", "COMPLETE"}
        ):
            raise RuntimeError("prepublication output file ownership is invalid")
        prepublication_names.append(path.name)
        if path == running_state:
            expected_running_bytes = (
                json.dumps(
                    {
                        "schema_version": 1,
                        "state": "RUNNING",
                        "run_id": expected_run_id,
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            if hashlib.sha256(expected_running_bytes).hexdigest() != record["sha256"]:
                raise RuntimeError("prepublication RUNNING state binding is invalid")
            saw_running_state = True
            continue
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RuntimeError("prepublication immutable artifact is absent") from error
        if (
            path.resolve(strict=True) != path
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or sha256_file(path) != record["sha256"]
            or path in immutable_paths
        ):
            raise RuntimeError("prepublication immutable artifact changed")
        immutable_paths.add(path)
    if not saw_running_state:
        raise RuntimeError("prepublication RUNNING state is absent")
    if prepublication_names != sorted(prepublication_names):
        raise RuntimeError("prepublication output file records are not canonical")

    report_path = output / "report.json"
    markdown_path = output / "report.md"
    hashed_paths = immutable_paths | {report_path, markdown_path}
    observed_files: set[pathlib.Path] = set()
    observed_directories: set[str] = set()
    for path in output.iterdir():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("terminal output inventory contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            observed_files.add(path.resolve())
        elif stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(path.name)
        else:
            raise RuntimeError("terminal output inventory contains a special entry")
    if observed_files != hashed_paths | {running_state}:
        raise RuntimeError("pre-marker terminal topology contains unbound files")
    if observed_directories != set(pre_manifest["runtime_directories"]):
        raise RuntimeError("pre-marker terminal runtime directories changed")
    report_sha256 = sha256_file(report_path)
    report_markdown_sha256 = sha256_file(markdown_path)
    finalizing = json.loads(running_state.read_text(encoding="utf-8"))
    expected_finalizing = {
        "schema_version": 1,
        "state": "FINALIZING",
        "run_id": expected_run_id,
        "profile_id": PROFILE_ID,
        "report_sha256": report_sha256,
        "report_markdown_sha256": report_markdown_sha256,
    }
    if finalizing != expected_finalizing:
        raise RuntimeError("terminal inventory requires FINALIZING state")

    hashed_files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(hashed_paths, key=lambda item: item.name)
    ]
    marker_files = ["COMPLETE", "state.json"]
    manifest = {
        "schema_version": 1,
        "epoch": "terminal-complete",
        "hashed_files": hashed_files,
        "equal_marker_files": marker_files,
        "runtime_directories": sorted(observed_directories),
        "top_level_files": sorted(
            [record["path"] for record in hashed_files] + marker_files
        ),
    }
    return {
        "pass": True,
        "manifest": manifest,
        "sha256": canonical_sha256(manifest),
    }


def verify_terminal_publication(
    output: pathlib.Path,
    *,
    expected_prepublication_inventory: dict[str, Any],
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Replay a terminal marker and its exact cross-epoch output closure."""

    output = _canonical_output_directory(output)
    marker_paths = [output / "COMPLETE", output / "state.json"]
    marker_bytes: list[bytes] = []
    for path in marker_paths:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("terminal marker is not a regular file")
        marker_bytes.append(path.read_bytes())
    if marker_bytes[0] != marker_bytes[1]:
        raise RuntimeError("COMPLETE and state.json terminal markers differ")
    try:
        marker = json.loads(marker_bytes[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("terminal marker is not valid JSON") from error
    expected_marker_keys = {
        "schema_version",
        "state",
        "run_id",
        "profile_id",
        "report",
        "report_sha256",
        "report_markdown",
        "report_markdown_sha256",
        "prepublication_output_inventory_sha256",
        "terminal_output_inventory",
    }
    if (
        not isinstance(marker, dict)
        or set(marker) != expected_marker_keys
        or marker.get("schema_version") != 2
        or marker.get("state") != "COMPLETE"
        or marker.get("profile_id") != PROFILE_ID
        or not isinstance(marker.get("run_id"), str)
        or not marker["run_id"]
        or not isinstance(marker.get("prepublication_output_inventory_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", marker["prepublication_output_inventory_sha256"]
        )
        is None
        or (expected_run_id is not None and marker.get("run_id") != expected_run_id)
    ):
        raise RuntimeError("terminal marker schema or binding is invalid")
    terminal = marker.get("terminal_output_inventory")
    if (
        not isinstance(terminal, dict)
        or set(terminal) != {"pass", "manifest", "sha256"}
        or terminal.get("pass") is not True
        or not isinstance(terminal.get("manifest"), dict)
        or canonical_sha256(terminal["manifest"]) != terminal.get("sha256")
    ):
        raise RuntimeError("terminal output inventory is invalid")
    manifest = terminal["manifest"]
    if (
        set(manifest)
        != {
            "schema_version",
            "epoch",
            "hashed_files",
            "equal_marker_files",
            "runtime_directories",
            "top_level_files",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("epoch") != "terminal-complete"
        or manifest.get("equal_marker_files") != ["COMPLETE", "state.json"]
        or not isinstance(manifest.get("hashed_files"), list)
        or not isinstance(manifest.get("runtime_directories"), list)
        or not isinstance(manifest.get("top_level_files"), list)
        or any(
            not isinstance(name, str)
            or name not in {"cache", "home", "matplotlib", "pycache", "tmp"}
            for name in manifest.get("runtime_directories", [])
        )
        or manifest.get("runtime_directories")
        != sorted(set(manifest.get("runtime_directories", [])))
        or any(
            not isinstance(name, str)
            or pathlib.Path(name).name != name
            for name in manifest.get("top_level_files", [])
        )
    ):
        raise RuntimeError("terminal output manifest schema is invalid")

    if (
        not isinstance(expected_prepublication_inventory, dict)
        or set(expected_prepublication_inventory)
        != {"pass", "epoch", "manifest", "sha256"}
        or expected_prepublication_inventory.get("pass") is not True
        or expected_prepublication_inventory.get("epoch")
        != "prepublication-running"
        or not isinstance(expected_prepublication_inventory.get("manifest"), dict)
        or canonical_sha256(expected_prepublication_inventory["manifest"])
        != expected_prepublication_inventory.get("sha256")
        or marker.get("prepublication_output_inventory_sha256")
        != expected_prepublication_inventory.get("sha256")
    ):
        raise RuntimeError("terminal prepublication inventory binding is invalid")
    pre_manifest = expected_prepublication_inventory["manifest"]
    if (
        set(pre_manifest) != {"files", "runtime_directories"}
        or not isinstance(pre_manifest.get("files"), list)
        or not isinstance(pre_manifest.get("runtime_directories"), list)
        or pre_manifest.get("runtime_directories")
        != sorted(set(pre_manifest.get("runtime_directories", [])))
        or any(
            not isinstance(name, str)
            or name not in {"cache", "home", "matplotlib", "pycache", "tmp"}
            for name in pre_manifest.get("runtime_directories", [])
        )
        or manifest["runtime_directories"] != pre_manifest["runtime_directories"]
    ):
        raise RuntimeError("terminal prepublication manifest schema is invalid")
    expected_running_bytes = (
        json.dumps(
            {
                "schema_version": 1,
                "state": "RUNNING",
                "run_id": marker["run_id"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    expected_running_sha256 = hashlib.sha256(expected_running_bytes).hexdigest()
    immutable_sha256: dict[str, str] = {}
    prepublication_names: list[str] = []
    saw_running_state = False
    for record in pre_manifest["files"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256"}
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise RuntimeError("terminal prepublication file record is invalid")
        path = pathlib.Path(record["path"])
        if (
            not path.is_absolute()
            or path.parent != output
            or record["path"] != str(output / path.name)
            or path.name in prepublication_names
            or path.name in {"report.json", "report.md", "COMPLETE"}
        ):
            raise RuntimeError("terminal prepublication file ownership is invalid")
        prepublication_names.append(path.name)
        if path.name == "state.json":
            if record["sha256"] != expected_running_sha256:
                raise RuntimeError("terminal prepublication RUNNING state is invalid")
            saw_running_state = True
        else:
            immutable_sha256[path.name] = record["sha256"]
    if (
        not saw_running_state
        or prepublication_names != sorted(prepublication_names)
    ):
        raise RuntimeError("terminal prepublication file records are not canonical")

    hashed_names: set[str] = set()
    hashed_sha256: dict[str, str] = {}
    for record in manifest["hashed_files"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size_bytes", "sha256"}
            or not isinstance(record.get("path"), str)
            or pathlib.Path(record["path"]).name != record["path"]
            or record["path"] in {"COMPLETE", "state.json"}
            or record["path"] in hashed_names
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] < 0
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise RuntimeError("terminal hashed-file record is invalid")
        path = output / record["path"]
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError("terminal hashed file changed")
        hashed_names.add(record["path"])
        hashed_sha256[record["path"]] = record["sha256"]
    if [record["path"] for record in manifest["hashed_files"]] != sorted(
        hashed_names
    ):
        raise RuntimeError("terminal hashed-file records are not canonical")
    expected_hashed_names = set(immutable_sha256) | {"report.json", "report.md"}
    if hashed_names != expected_hashed_names:
        raise RuntimeError("terminal hashed-file set is not the exact cross-epoch closure")
    if any(
        hashed_sha256.get(name) != digest
        for name, digest in immutable_sha256.items()
    ):
        raise RuntimeError("terminal immutable digest changed across publication epochs")
    expected_files = sorted(hashed_names | {"COMPLETE", "state.json"})
    if manifest["top_level_files"] != expected_files:
        raise RuntimeError("terminal top-level file manifest is inconsistent")

    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in output.iterdir():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("terminal output contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            observed_files.add(path.name)
        elif stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(path.name)
        else:
            raise RuntimeError("terminal output contains a special entry")
    if observed_files != set(expected_files):
        raise RuntimeError("terminal output contains unbound files")
    if observed_directories != set(manifest["runtime_directories"]):
        raise RuntimeError("terminal output runtime directories changed")

    report_path = output / "report.json"
    markdown_path = output / "report.md"
    if (
        marker.get("report") != str(report_path)
        or marker.get("report_markdown") != str(markdown_path)
        or marker.get("report_sha256") != sha256_file(report_path)
        or marker.get("report_markdown_sha256") != sha256_file(markdown_path)
    ):
        raise RuntimeError("terminal report binding is invalid")
    return marker


def reverify_final_closure(
    *,
    output: pathlib.Path,
    expected_run_id: str,
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    expected_receipt_id: str,
    expected_snapshot: dict[str, Any],
    expected_producer_sha256: str,
    samples: list[dict[str, Any]],
    fd_oracle: dict[str, Any],
    publication_capability: _RunPublicationCapability,
) -> dict[str, Any]:
    output = _canonical_output_directory(output)
    _require_running_state(output, expected_run_id)
    if any(
        not isinstance(sample, dict) or set(sample) != EXPECTED_SAMPLE_KEYS
        for sample in samples
    ):
        raise RuntimeError("qualification sample exact schema disagrees")
    validate_sample_matrix(samples)
    hash_cache = StatHashCache()
    receipt = verify_build_receipt(
        receipt_path, repo, hash_cache=hash_cache
    )
    if receipt.get("receipt_id") != expected_receipt_id:
        raise RuntimeError("build receipt changed during qualification")
    contract = validate_build_receipt_contract(
        receipt, repo, hash_cache=hash_cache
    )
    if (
        source_snapshot(repo, hash_cache=hash_cache) != expected_snapshot
        or receipt.get("source_end") != expected_snapshot
    ):
        raise RuntimeError("source changed before final qualification publication")
    if (
        sha256_file(repo / "scripts" / "benchmark-adjoint.py")
        != expected_producer_sha256
    ):
        raise RuntimeError("raw benchmark producer changed during qualification")
    runtime = _receipt_runtime(receipt, repo)

    checked_samples: list[dict[str, Any]] = []
    owned_paths: set[pathlib.Path] = set()
    artifact_labels: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise RuntimeError("qualification sample is not an object")
        lane = sample.get("lane")
        sample_kind = sample.get("sample_kind")
        iteration = sample.get("iteration")
        label = f"{lane}-{sample_kind}-{iteration}"
        artifact_label = f"{expected_run_id}-{lane}-{sample_kind}-{iteration}"
        ranks = _lane_ranks(lane)
        if (
            sample.get("artifact_label") != artifact_label
            or artifact_label in artifact_labels
            or sample.get("ranks") != ranks
            or sample.get("completion_policy") != "waitsome"
            or sample.get("capture_field_times") is not False
            or not isinstance(sample.get("nonce"), str)
            or not sample["nonce"]
            or isinstance(sample.get("process_seconds"), bool)
            or not isinstance(sample.get("process_seconds"), (int, float))
            or not math.isfinite(sample["process_seconds"])
            or sample["process_seconds"] <= 0
        ):
            raise RuntimeError(f"{label} current-run sample identity disagrees")
        artifact_labels.add(artifact_label)
        expected_names = {
            "result_file": f"{artifact_label}.json",
            "stdout_log": f"{artifact_label}.stdout.log",
            "stderr_log": f"{artifact_label}.stderr.log",
            "timing_file": f"{artifact_label}.parent-timing.json",
        }
        checked_files: dict[str, dict[str, str]] = {}
        for path_name, digest_name in (
            ("result_file", "result_sha256"),
            ("stdout_log", "stdout_sha256"),
            ("stderr_log", "stderr_sha256"),
            ("timing_file", "timing_sha256"),
        ):
            path = _require_owned_regular_file(
                sample.get(path_name),
                output=output,
                expected_name=expected_names[path_name],
                label=f"{label} {path_name}",
            )
            expected_digest = sample.get(digest_name)
            if (
                path in owned_paths
                or not isinstance(expected_digest, str)
                or sha256_file(path) != expected_digest
            ):
                raise RuntimeError(f"{label} {path_name} changed before publication")
            owned_paths.add(path)
            checked_files[path_name] = {
                "path": str(path),
                "sha256": expected_digest,
            }
        expected_command = _expected_sample_command(
            repo=repo,
            runtime=runtime,
            receipt_id=expected_receipt_id,
            snapshot_sha256=expected_snapshot["sha256"],
            producer_sha256=expected_producer_sha256,
            run_id=expected_run_id,
            nonce=sample["nonce"],
            lane=lane,
            sample_kind=sample_kind,
            iteration=iteration,
            result_path=pathlib.Path(sample["result_file"]),
            ranks=ranks,
            capture_field_times=False,
        )
        expected_environment = _child_environment(
            runtime,
            lane=lane,
            output=output,
            pycache_namespace=expected_run_id,
        )
        expected_environment["MEEP_GPU_MPI_COMPLETION"] = "waitsome"
        expected_environment = dict(sorted(expected_environment.items()))
        if (
            sample.get("command") != expected_command
            or sample.get("environment") != expected_environment
        ):
            raise RuntimeError(f"{label} launch command or environment disagrees")
        timing_record = json.loads(
            pathlib.Path(sample["timing_file"]).read_text(encoding="utf-8")
        )
        expected_timing_keys = {
            "schema_version",
            "run_id",
            "label",
            "lane",
            "sample_kind",
            "iteration",
            "nonce",
            "process_seconds",
            "command_sha256",
        }
        if (
            not isinstance(timing_record, dict)
            or set(timing_record) != expected_timing_keys
            or timing_record.get("schema_version") != 1
            or timing_record.get("run_id") != expected_run_id
            or timing_record.get("label") != artifact_label
            or timing_record.get("lane") != lane
            or timing_record.get("sample_kind") != sample_kind
            or timing_record.get("iteration") != iteration
            or timing_record.get("nonce") != sample["nonce"]
            or timing_record.get("process_seconds") != sample["process_seconds"]
            or timing_record.get("command_sha256")
            != canonical_sha256(expected_command)
        ):
            raise RuntimeError(f"{label} parent timing attestation disagrees")
        raw = json.loads(
            pathlib.Path(sample["result_file"]).read_text(encoding="utf-8")
        )
        if canonical_sha256(raw) != canonical_sha256(sample["result"]):
            raise RuntimeError(f"{label} in-memory/raw evidence disagrees")
        COMPARATOR.validate_qualification_binding(
            raw,
            expected_nonce=sample["nonce"],
            expected_run_id=expected_run_id,
            expected_receipt_id=expected_receipt_id,
            expected_source_snapshot_sha256=expected_snapshot["sha256"],
            expected_producer_sha256=expected_producer_sha256,
            expected_sample_kind=sample_kind,
            expected_sample_lane=lane,
            expected_sample_iteration=iteration,
            label=label,
            hash_cache=hash_cache,
        )
        producer_command = raw.get("producer", {}).get("command")
        expected_producer_command = _expected_producer_command(
            repo=repo,
            runtime=runtime,
            receipt_id=expected_receipt_id,
            snapshot_sha256=expected_snapshot["sha256"],
            producer_sha256=expected_producer_sha256,
            run_id=expected_run_id,
            nonce=sample["nonce"],
            lane=lane,
            sample_kind=sample_kind,
            iteration=iteration,
            result_path=pathlib.Path(sample["result_file"]),
            ranks=ranks,
            capture_field_times=False,
        )
        if producer_command != expected_producer_command:
            raise RuntimeError(f"{label} raw producer command differs from launch")
        stdout = pathlib.Path(sample["stdout_log"]).read_text(encoding="utf-8")
        pointer = _extract_pointer(stdout)
        if (
            pathlib.Path(str(pointer.get("result_file", ""))).resolve()
            != pathlib.Path(sample["result_file"])
            or pointer.get("sha256") != sample["result_sha256"]
        ):
            raise RuntimeError(f"{label} raw result pointer disagrees")
        _fixed_workload_gate(raw, lane=lane)
        lazy_import_gate = _validate_lazy_import_contract(raw, label=label)
        if lane == CPU_ORACLE_LANE:
            validation = validate_cpu_oracle(raw)
        elif lane == CPU_BENCHMARK_LANE:
            validation = validate_cpu_benchmark_record(raw, label=label)
        elif lane in CUDA_LANES:
            validation = COMPARATOR.validate_distributed_record(
                raw,
                expected_world_size=ranks,
                label=label,
                hash_cache=hash_cache,
            )
            validation["rank_driver"] = _validate_cuda_rank_driver_evidence(
                raw, validation, label=label
            )
        receipt_gate = _validate_receipt_rank_runtimes(
            receipt, validation, hash_cache=hash_cache
        )
        material_gradient_gate = _extract_material_gradient_stats(stdout, lane=lane)
        process_environment_gate = _validate_actual_process_environments(
            raw, expected_environment, label=label
        )
        _validate_lane_process_environment(
            raw, expected_environment, lane=lane, label=label
        )
        if (
            (
                lane != CPU_ORACLE_LANE
                and sample["process_seconds"]
                < _finite_positive(
                    validation.get("workload_wall_seconds"),
                    f"{label}.validated workload wall",
                )
            )
            or
            sample.get("validation") != validation
            or sample.get("receipt_gate") != receipt_gate
            or sample.get("process_environment_gate") != process_environment_gate
            or sample.get("lazy_import_gate") != lazy_import_gate
            or sample.get("material_gradient_gate") != material_gradient_gate
        ):
            raise RuntimeError(f"{label} derived validation evidence disagrees")
        summarized = summarize_samples([sample])[0]
        checked_samples.append(
            {
                "label": label,
                "nonce": sample["nonce"],
                "process_seconds": sample["process_seconds"],
                "summary_sha256": canonical_sha256(summarized),
                "files": checked_files,
            }
        )

    if not isinstance(fd_oracle, dict):
        raise RuntimeError("finite-difference oracle is absent")
    expected_fd_keys = {
        "pass",
        "kind",
        "receipt_id",
        "source_snapshot_sha256",
        "test_file",
        "test_sha256",
        "methods",
        "directions",
        "direction_records",
        "started_at_utc",
        "completed_at_utc",
        "process_seconds",
        "command",
        "environment",
        "stdout_log",
        "stdout_sha256",
        "stderr_log",
        "stderr_sha256",
        "timing_file",
        "timing_sha256",
    }
    if (
        set(fd_oracle) != expected_fd_keys
        or isinstance(fd_oracle.get("process_seconds"), bool)
        or not isinstance(fd_oracle.get("process_seconds"), (int, float))
        or not math.isfinite(fd_oracle["process_seconds"])
        or fd_oracle["process_seconds"] <= 0
        or not isinstance(fd_oracle.get("started_at_utc"), str)
        or not fd_oracle["started_at_utc"].endswith("Z")
        or not isinstance(fd_oracle.get("completed_at_utc"), str)
        or not fd_oracle["completed_at_utc"].endswith("Z")
    ):
        raise RuntimeError("finite-difference oracle schema is invalid")
    validate_fd_oracle_summary(fd_oracle)
    if (
        fd_oracle.get("receipt_id") != expected_receipt_id
        or fd_oracle.get("source_snapshot_sha256") != expected_snapshot["sha256"]
    ):
        raise RuntimeError("finite-difference oracle binding changed")
    expected_test_path = (
        repo / "python" / "tests" / "test_adjoint_default_material_grid.py"
    ).resolve()
    if (
        fd_oracle.get("test_file") != str(expected_test_path)
        or sha256_file(expected_test_path) != fd_oracle.get("test_sha256")
        or fd_oracle.get("methods") != list(FD_TEST_METHODS)
    ):
        raise RuntimeError("finite-difference oracle source changed")
    checked_fd_logs: dict[str, dict[str, str]] = {}
    fd_label = f"{expected_run_id}-directional-fd-oracle"
    for path_name, digest_name in (
        ("stdout_log", "stdout_sha256"),
        ("stderr_log", "stderr_sha256"),
        ("timing_file", "timing_sha256"),
    ):
        suffix = {
            "stdout_log": ".stdout.log",
            "stderr_log": ".stderr.log",
            "timing_file": ".parent-timing.json",
        }[path_name]
        path = _require_owned_regular_file(
            fd_oracle.get(path_name),
            output=output,
            expected_name=f"{fd_label}{suffix}",
            label=f"finite-difference oracle {path_name}",
        )
        if (
            path in owned_paths
            or sha256_file(path) != fd_oracle.get(digest_name)
        ):
            raise RuntimeError(f"finite-difference oracle {path_name} changed")
        owned_paths.add(path)
        checked_fd_logs[path_name] = {
            "path": str(path),
            "sha256": fd_oracle[digest_name],
        }
    fd_command = [str(runtime["python"]), str(expected_test_path), "-v"]
    fd_environment = dict(sorted(_child_environment(
        runtime,
        lane=CPU_ORACLE_LANE,
        output=output,
        pycache_namespace=expected_run_id,
    ).items()))
    fd_timing = json.loads(
        pathlib.Path(fd_oracle["timing_file"]).read_text(encoding="utf-8")
    )
    expected_fd_timing = {
        "schema_version": 1,
        "run_id": expected_run_id,
        "label": fd_label,
        "process_seconds": fd_oracle.get("process_seconds"),
        "started_at_utc": fd_oracle.get("started_at_utc"),
        "completed_at_utc": fd_oracle.get("completed_at_utc"),
        "command_sha256": canonical_sha256(fd_command),
        "environment_sha256": canonical_sha256(fd_environment),
    }
    fd_stdout = pathlib.Path(fd_oracle["stdout_log"]).read_text(encoding="utf-8")
    fd_stderr = pathlib.Path(fd_oracle["stderr_log"]).read_text(encoding="utf-8")
    if (
        fd_oracle.get("command") != fd_command
        or fd_oracle.get("environment") != fd_environment
        or fd_timing != expected_fd_timing
        or fd_oracle.get("direction_records")
        != _parse_fd_oracle_output(fd_stdout, fd_stderr)
    ):
        raise RuntimeError("finite-difference oracle derived evidence disagrees")
    controller_ledger = _validate_controller_ledger(
        publication_capability,
        output=output,
        run_id=expected_run_id,
        samples=samples,
        fd_oracle=fd_oracle,
    )
    output_inventory = _validate_output_inventory(output, owned_paths)
    receipt_record = build_receipt_report_record(receipt_path, receipt, contract)
    return {
        "pass": True,
        "verified_at_utc": _utc_now(),
        "build_receipt": receipt_record,
        "receipt_id": expected_receipt_id,
        "receipt_sha256": sha256_file(receipt_path),
        "source_snapshot": expected_snapshot,
        "contract": contract,
        "samples": checked_samples,
        "finite_difference_logs": checked_fd_logs,
        "controller_ledger": controller_ledger,
        "output_inventory": output_inventory,
    }


def render_complete_markdown(
    comparison: dict[str, Any], receipt_id: str
) -> str:
    performance = comparison["performance"]
    workload = performance["workload_wall"]
    process_performance = performance["fresh_process_wall"]
    lanes = workload["lanes"]
    speedups = workload["comparisons"]
    process_speedups = process_performance["comparisons"]
    return (
        "# gpmeep fixed MPI adjoint qualification\n\n"
        "- State: **COMPLETE**\n"
        f"- Profile: `{PROFILE_ID}`\n"
        "- Oracles: full-vector CPU legacy plus installed five-direction/component FD\n"
        f"- Samples: 1 CPU legacy oracle + {PROFILE['warmups']} warmup and "
        f"{PROFILE['repeats']} measured samples for CPU 2x4, 1 GPU, and 2 GPU\n"
        f"- Headline metric: `{performance['headline_metric']}`\n"
        f"- CPU 2x4 / 1-GPU / 2-GPU workload-wall medians: "
        f"`{lanes[CPU_BENCHMARK_LANE]['median_seconds']:.6f}` / "
        f"`{lanes['cuda-single']['median_seconds']:.6f}` / "
        f"`{lanes['cuda-multi']['median_seconds']:.6f}` s\n"
        f"- CPU-to-1GPU median/conservative speedup: "
        f"`{speedups['cpu_over_cuda_single']['median_speedup']:.6f}x` / "
        f"`{speedups['cpu_over_cuda_single']['conservative_speedup']:.6f}x`\n"
        f"- CPU-to-2GPU median/conservative speedup: "
        f"`{speedups['cpu_over_cuda_multi']['median_speedup']:.6f}x` / "
        f"`{speedups['cpu_over_cuda_multi']['conservative_speedup']:.6f}x`\n"
        f"- 1GPU-to-2GPU median/conservative scaling: "
        f"`{speedups['cuda_single_over_cuda_multi']['median_speedup']:.6f}x` / "
        f"`{speedups['cuda_single_over_cuda_multi']['conservative_speedup']:.6f}x`\n"
        f"- Fresh-process CPU-to-2GPU median/conservative speedup: "
        f"`{process_speedups['cpu_over_cuda_multi']['median_speedup']:.6f}x` / "
        f"`{process_speedups['cpu_over_cuda_multi']['conservative_speedup']:.6f}x`\n"
        f"- Receipt: `{receipt_id}`\n"
    )


def _validate_integrity_epoch(
    recorded: Any, fresh: dict[str, Any]
) -> None:
    if (
        not isinstance(recorded, dict)
        or set(recorded) != set(fresh)
        or not isinstance(recorded.get("verified_at_utc"), str)
        or not recorded["verified_at_utc"]
        or {
            name: value
            for name, value in recorded.items()
            if name != "verified_at_utc"
        }
        != {
            name: value
            for name, value in fresh.items()
            if name != "verified_at_utc"
        }
    ):
        raise RuntimeError("first integrity epoch is not bound to verified inputs")


def _validate_complete_postprocessor(
    postprocessor: Any,
    *,
    repo: pathlib.Path,
    expected_snapshot: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "git_head",
        "git_status_porcelain",
        "source_snapshot",
        "source_sha256",
        "python",
        "platform",
        "nvidia_driver",
        "nvidia_topology",
        "driver_device_binding",
        "command",
    }
    executable = pathlib.Path(sys.executable).resolve()
    expected_python = {
        "path": str(executable),
        "sha256": sha256_file(executable),
        "version": sys.version,
    }
    expected_command = [
        str(executable),
        str(pathlib.Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    if (
        not isinstance(postprocessor, dict)
        or set(postprocessor) != expected_keys
        or postprocessor.get("git_head")
        != git_output(repo, "rev-parse", "HEAD").strip()
        or postprocessor.get("git_status_porcelain")
        != git_output(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        or postprocessor.get("source_snapshot") != expected_snapshot
        or postprocessor.get("source_sha256")
        != {name: sha256_file(repo / name) for name in RUNNER_SOURCES}
        or postprocessor.get("python") != expected_python
        or postprocessor.get("platform") != platform.platform()
        or postprocessor.get("command") != expected_command
    ):
        raise RuntimeError("final postprocessor evidence is not source-bound")
    driver_binding = validate_driver_device_binding(
        postprocessor.get("nvidia_driver", {}),
        comparison.get("devices", {}).get("identifiers", [])[:1],
        postprocessor.get("nvidia_topology"),
        repo=repo,
    )
    driver_binding["rank_local"] = comparison.get("devices", {}).get(
        "rank_local_driver_by_lane_and_rank"
    )
    if postprocessor.get("driver_device_binding") != driver_binding:
        raise RuntimeError("final postprocessor driver binding disagrees")
    return driver_binding


def validate_complete_report(
    run_id: str,
    report: Any,
    *,
    repo: pathlib.Path,
    expected_build_receipt_record: dict[str, str],
    expected_snapshot: dict[str, Any],
    samples: list[dict[str, Any]],
    fd_oracle: dict[str, Any],
    fresh_integrity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_top_level = {
        "schema_version",
        "state",
        "run_id",
        "profile",
        "build_receipt",
        "finite_difference_oracle",
        "host_audit",
        "integrity_reverification",
        "sample_matrix",
        "samples",
        "comparison",
        "gate",
        "postprocessor",
    }
    if (
        not isinstance(report, dict)
        or set(report) != expected_top_level
        or report.get("schema_version") != 1
        or report.get("state") != "COMPLETE"
        or report.get("run_id") != run_id
        or report.get("profile") != dict(PROFILE)
        or not valid_build_receipt_report_record(report.get("build_receipt"))
        or report.get("build_receipt") != expected_build_receipt_record
        or report.get("finite_difference_oracle") != fd_oracle
        or report.get("samples") != summarize_samples(samples)
    ):
        raise RuntimeError("final COMPLETE report schema or inputs disagree")
    _validate_integrity_epoch(
        report.get("integrity_reverification"), fresh_integrity
    )
    recomputed_sample_matrix = validate_sample_matrix(samples)
    recomputed_comparison = compare_measured_samples(samples, fd_oracle=fd_oracle)
    bind_host_audit(
        recomputed_comparison,
        report.get("host_audit"),
        samples=samples,
        repo=repo,
        verify_live=True,
    )
    final_driver_reverification = _validate_complete_postprocessor(
        report.get("postprocessor"),
        repo=repo,
        expected_snapshot=expected_snapshot,
        comparison=recomputed_comparison,
    )
    recomputed_comparison["gate"]["driver_runtime_device_binding"] = True
    recomputed_comparison["gate"]["pass"] = all(
        value
        for name, value in recomputed_comparison["gate"].items()
        if name != "pass"
    )
    if (
        report.get("sample_matrix") != recomputed_sample_matrix
        or report.get("comparison") != recomputed_comparison
        or report.get("gate") != recomputed_comparison.get("gate")
        or any(value is not True for value in report["gate"].values())
    ):
        raise RuntimeError("refusing to publish an unqualified MPI run as COMPLETE")
    return recomputed_comparison, final_driver_reverification


def publish_complete(
    output: pathlib.Path,
    run_id: str,
    report: dict[str, Any],
    markdown: str,
    *,
    capability: _RunPublicationCapability,
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    expected_receipt_id: str,
    expected_build_receipt_record: dict[str, str],
    expected_snapshot: dict[str, Any],
    expected_producer_sha256: str,
    samples: list[dict[str, Any]],
    fd_oracle: dict[str, Any],
) -> None:
    _assert_fixed_profile()
    output = _canonical_output_directory(output)
    for name in ("report.json", "report.md", "COMPLETE"):
        (output / name).unlink(missing_ok=True)
    try:
        if not isinstance(capability, _RunPublicationCapability):
            raise RuntimeError("MPI qualification publisher capability is absent")
        capability.snapshot(output, run_id, consume=True)
        _require_running_state(output, run_id)
        final_publish_reverification = reverify_final_closure(
            output=output,
            expected_run_id=run_id,
            repo=repo,
            receipt_path=receipt_path,
            expected_receipt_id=expected_receipt_id,
            expected_snapshot=expected_snapshot,
            expected_producer_sha256=expected_producer_sha256,
            samples=samples,
            fd_oracle=fd_oracle,
            publication_capability=capability,
        )
        if (
            not isinstance(final_publish_reverification, dict)
            or final_publish_reverification.get("pass") is not True
            or not valid_build_receipt_report_record(
                final_publish_reverification.get("build_receipt")
            )
            or final_publish_reverification.get("build_receipt")
            != expected_build_receipt_record
            or final_publish_reverification.get("receipt_id") != expected_receipt_id
            or final_publish_reverification.get("receipt_sha256")
            != expected_build_receipt_record.get("sha256")
            or final_publish_reverification.get("source_snapshot")
            != expected_snapshot
        ):
            raise RuntimeError("final build receipt reverification is not bound")
        comparison, final_driver_reverification = validate_complete_report(
            run_id,
            report,
            repo=repo,
            expected_build_receipt_record=final_publish_reverification[
                "build_receipt"
            ],
            expected_snapshot=expected_snapshot,
            samples=samples,
            fd_oracle=fd_oracle,
            fresh_integrity=final_publish_reverification,
        )
        expected_markdown = render_complete_markdown(
            comparison, expected_receipt_id
        )
        if markdown != expected_markdown:
            raise RuntimeError("final Markdown report is not derived from verified inputs")
        report["final_publish_reverification"] = final_publish_reverification
        report["final_publish_reverification"]["nvidia_driver"] = (
            final_driver_reverification
        )
        markdown_path = output / "report.md"
        report_path = output / "report.json"
        atomic_write_text(markdown_path, markdown)
        report["report_markdown"] = {
            "path": str(markdown_path.resolve()),
            "sha256": sha256_file(markdown_path),
        }
        atomic_write_json(report_path, report)
        published_report = json.loads(report_path.read_text(encoding="utf-8"))
        if published_report != report:
            raise RuntimeError("published COMPLETE report changed during write-back")
        report_sha256 = sha256_file(report_path)
        report_markdown_sha256 = sha256_file(markdown_path)
        atomic_write_json(
            output / "state.json",
            {
                "schema_version": 1,
                "state": "FINALIZING",
                "run_id": run_id,
                "profile_id": PROFILE_ID,
                "report_sha256": report_sha256,
                "report_markdown_sha256": report_markdown_sha256,
            },
        )
        prepublication_inventory = final_publish_reverification[
            "output_inventory"
        ]
        terminal_inventory = _build_terminal_output_inventory(
            output,
            prepublication_inventory,
            expected_run_id=run_id,
        )
        marker = {
            "schema_version": 2,
            "state": "COMPLETE",
            "run_id": run_id,
            "profile_id": PROFILE_ID,
            "report": str(report_path.resolve()),
            "report_sha256": report_sha256,
            "report_markdown": str(markdown_path.resolve()),
            "report_markdown_sha256": report_markdown_sha256,
            "prepublication_output_inventory_sha256": (
                prepublication_inventory["sha256"]
            ),
            "terminal_output_inventory": terminal_inventory,
        }
        atomic_write_json(output / "COMPLETE", marker)
        atomic_write_json(output / "state.json", marker)
        verified_marker = verify_terminal_publication(
            output,
            expected_prepublication_inventory=prepublication_inventory,
            expected_run_id=run_id,
        )
        if verified_marker != marker:
            raise RuntimeError("published terminal marker changed during replay")
    except BaseException as error:
        for name in ("report.json", "report.md", "COMPLETE"):
            (output / name).unlink(missing_ok=True)
        failure_state = {
            "schema_version": 1,
            "state": "FAILED",
            "run_id": run_id,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            atomic_write_json(output / "state.json", failure_state)
        except BaseException:
            (output / "state.json").unlink(missing_ok=True)
        raise


def execute(args: argparse.Namespace, repo: pathlib.Path, run_id: str) -> int:
    _assert_fixed_profile()
    hash_cache = StatHashCache()
    snapshot_before = source_snapshot(repo, hash_cache=hash_cache)
    receipt_path = args.build_receipt.resolve()
    receipt = verify_build_receipt(
        receipt_path, repo, hash_cache=hash_cache
    )
    if receipt.get("build_kind") != "cuda-mpi-python-fp32":
        raise RuntimeError("qualification requires a CUDA+MPI+Python FP32 receipt")
    receipt_contract = validate_build_receipt_contract(
        receipt, repo, hash_cache=hash_cache
    )
    receipt_report_record = build_receipt_report_record(
        receipt_path, receipt, receipt_contract
    )
    if receipt.get("source_end") != snapshot_before:
        raise RuntimeError("build receipt does not match the current source snapshot")
    runtime = _receipt_runtime(receipt, repo)
    host_audit_start = capture_host_audit_start(repo)
    producer = repo / "scripts" / "benchmark-adjoint.py"
    producer_sha256 = sha256_file(producer)
    samples: list[dict[str, Any]] = []
    publication_capability = _RunPublicationCapability(args.output, run_id)

    fd_oracle = _run_fd_oracle_test(
        repo=repo,
        output=args.output,
        runtime=runtime,
        run_id=run_id,
        receipt_id=receipt["receipt_id"],
        snapshot_sha256=snapshot_before["sha256"],
        publication_capability=publication_capability,
    )

    samples.append(
        _run_raw_sample(
            repo=repo,
            output=args.output,
            runtime=runtime,
            receipt=receipt,
            snapshot_sha256=snapshot_before["sha256"],
            producer_sha256=producer_sha256,
            run_id=run_id,
            lane=CPU_ORACLE_LANE,
            sample_kind="measured",
            iteration=0,
            hash_cache=hash_cache,
            publication_capability=publication_capability,
        )
    )
    for iteration in range(PROFILE["warmups"]):
        for lane in MEASURED_LANES:
            samples.append(
                _run_raw_sample(
                    repo=repo,
                    output=args.output,
                    runtime=runtime,
                    receipt=receipt,
                    snapshot_sha256=snapshot_before["sha256"],
                    producer_sha256=producer_sha256,
                    run_id=run_id,
                    lane=lane,
                    sample_kind="warmup",
                    iteration=iteration,
                    hash_cache=hash_cache,
                    publication_capability=publication_capability,
                )
            )
    for iteration in range(PROFILE["repeats"]):
        lanes = MEASURED_ORDERS[iteration % len(MEASURED_ORDERS)]
        for lane in lanes:
            samples.append(
                _run_raw_sample(
                    repo=repo,
                    output=args.output,
                    runtime=runtime,
                    receipt=receipt,
                    snapshot_sha256=snapshot_before["sha256"],
                    producer_sha256=producer_sha256,
                    run_id=run_id,
                    lane=lane,
                    sample_kind="measured",
                    iteration=iteration,
                    hash_cache=hash_cache,
                    publication_capability=publication_capability,
                )
            )

    host_audit = complete_host_audit(host_audit_start, samples, repo=repo)
    publication_capability.seal()
    sample_matrix = validate_sample_matrix(samples)
    comparison = compare_measured_samples(samples, fd_oracle=fd_oracle)
    bind_host_audit(
        comparison,
        host_audit,
        samples=samples,
        repo=repo,
        verify_live=True,
    )
    snapshot_after = source_snapshot(repo, hash_cache=hash_cache)
    if (
        snapshot_after != snapshot_before
        or snapshot_after != receipt.get("source_end")
    ):
        raise RuntimeError("source changed during qualification")
    if any(
        sample["receipt_gate"]["receipt_id"] != receipt["receipt_id"]
        for sample in samples
    ):
        raise RuntimeError("sample receipt bindings disagree")
    cuda_environments = [
        sample["environment"]
        for sample in samples
        if sample["lane"] in CUDA_LANES
    ]
    if any(
        environment != cuda_environments[0]
        for environment in cuda_environments[1:]
    ):
        raise RuntimeError("CUDA qualification environment changed between samples")

    integrity_reverification = reverify_final_closure(
        output=args.output,
        expected_run_id=run_id,
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=receipt["receipt_id"],
        expected_snapshot=snapshot_before,
        expected_producer_sha256=producer_sha256,
        samples=samples,
        fd_oracle=fd_oracle,
        publication_capability=publication_capability,
    )
    summarized_samples = summarize_samples(samples)
    if not comparison["gate"]["pass"]:
        unqualified_report = {
            "schema_version": 1,
            "state": "UNQUALIFIED",
            "evidence_class": "failed-release-qualification",
            "run_id": run_id,
            "profile": dict(PROFILE),
            "build_receipt": dict(receipt_report_record),
            "finite_difference_oracle": fd_oracle,
            "host_audit": host_audit,
            "integrity_reverification": integrity_reverification,
            "sample_matrix": sample_matrix,
            "samples": summarized_samples,
            "comparison": comparison,
            "gate": comparison["gate"],
        }
        unqualified_record = write_unqualified_report(
            args.output, run_id, unqualified_report
        )
        failed = [
            name
            for name, passed in comparison["gate"].items()
            if name != "pass" and not passed
        ]
        raise QualificationGateFailure(
            "MPI adjoint qualification gates failed: " + ", ".join(failed),
            unqualified_record,
        )

    driver_probe = _nvidia_smi_probe(
        [
            "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
        ],
        repo,
    )
    topology_probe = _nvidia_smi_probe(["-L"], repo)
    driver_device_binding = validate_driver_device_binding(
        driver_probe,
        comparison["devices"]["identifiers"][:1],
        topology_probe,
        repo=repo,
    )
    driver_device_binding["rank_local"] = comparison["devices"][
        "rank_local_driver_by_lane_and_rank"
    ]
    comparison["gate"]["driver_runtime_device_binding"] = True
    comparison["gate"]["pass"] = all(
        value for name, value in comparison["gate"].items() if name != "pass"
    )
    postprocessor = {
        "git_head": git_output(repo, "rev-parse", "HEAD").strip(),
        "git_status_porcelain": git_output(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines(),
        "source_snapshot": snapshot_after,
        "source_sha256": {
            name: sha256_file(repo / name) for name in RUNNER_SOURCES
        },
        "python": {
            "path": str(pathlib.Path(sys.executable).resolve()),
            "sha256": sha256_file(pathlib.Path(sys.executable).resolve()),
            "version": sys.version,
        },
        "platform": platform.platform(),
        "nvidia_driver": driver_probe,
        "nvidia_topology": topology_probe,
        "driver_device_binding": driver_device_binding,
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
        "profile": dict(PROFILE),
        "build_receipt": dict(receipt_report_record),
        "finite_difference_oracle": fd_oracle,
        "host_audit": host_audit,
        "integrity_reverification": integrity_reverification,
        "sample_matrix": sample_matrix,
        "samples": summarized_samples,
        "comparison": comparison,
        "gate": comparison["gate"],
        "postprocessor": postprocessor,
    }
    markdown = render_complete_markdown(comparison, receipt["receipt_id"])
    publish_complete(
        args.output,
        run_id,
        report,
        markdown,
        capability=publication_capability,
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=receipt["receipt_id"],
        expected_build_receipt_record=receipt_report_record,
        expected_snapshot=snapshot_before,
        expected_producer_sha256=producer_sha256,
        samples=samples,
        fd_oracle=fd_oracle,
    )
    print(markdown, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = pathlib.Path(__file__).resolve().parent.parent
    run_id = uuid.uuid4().hex
    args.output = pathlib.Path(os.path.abspath(os.fspath(args.output)))
    prepare_output(args.output, run_id)
    try:
        return execute(args, repo, run_id)
    except BaseException as error:
        mark_failed(args.output, run_id, error)
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
