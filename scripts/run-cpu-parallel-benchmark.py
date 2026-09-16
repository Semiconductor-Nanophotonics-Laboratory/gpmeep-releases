#!/usr/bin/env python3
"""Run the topology-derived, receipt-bound gpmeep CPU tuning matrix.

The runner measures MPI/OpenMP CPU configurations on the same installed FP32
CUDA-capable Meep build used by the GPU qualification.  It deliberately keeps
the one-core result as a diagnostic reference and selects the fastest
correctness-passing parallel configuration by median end-to-end workload wall
time.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import signal
import socket
import stat
import statistics
import subprocess
import sys
import time
import traceback
import types
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable


def _load_source_module(name: str, path: pathlib.Path) -> types.ModuleType:
    """Load audited local tooling from source bytes, never adjacent bytecode."""

    resolved = path.resolve()
    module = types.ModuleType(name)
    module.__file__ = str(resolved)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(resolved.read_bytes(), str(resolved), "exec"), module.__dict__)
    return module


_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
_EVIDENCE = _load_source_module(
    "gpmeep_cpu_parallel_evidence",
    _SCRIPT_DIRECTORY / "gpmeep_benchmark_evidence.py",
)
_PROVENANCE = _load_source_module(
    "gpmeep_cpu_parallel_provenance", _SCRIPT_DIRECTORY / "gpmeep_provenance.py"
)
_MPI_RUNNER = _load_source_module(
    "gpmeep_cpu_parallel_mpi_runner",
    _SCRIPT_DIRECTORY / "run-mpi-adjoint-benchmark.py",
)
_COMPARATOR = _MPI_RUNNER.COMPARATOR
_TOPOLOGY_POLICY = _load_source_module(
    "gpmeep_cpu_topology_policy",
    _SCRIPT_DIRECTORY / "gpmeep_cpu_topology_policy.py",
)

atomic_write_json = _EVIDENCE.atomic_write_json
atomic_write_text = _EVIDENCE.atomic_write_text
canonical_float64_sha256 = _EVIDENCE.canonical_float64_sha256
compare_gradient_vectors = _EVIDENCE.compare_gradient_vectors
sha256_file = _EVIDENCE.sha256_file
validate_gradient_record = _EVIDENCE.validate_gradient_record
canonical_sha256 = _PROVENANCE.canonical_sha256
file_record = _PROVENANCE.file_record
git_output = _PROVENANCE.git_output
source_snapshot = _PROVENANCE.source_snapshot
StatHashCache = _PROVENANCE.StatHashCache
verify_build_receipt = _PROVENANCE.verify_build_receipt


PROFILE_ID = "m8.7-cpu-parallel-tuning-v2"
# Raw producer contract remains v1; v2 identifies the runner matrix/evidence
# policy and is independently bound in the run anchor.
PRODUCER_QUALIFICATION_PROFILE_ID = "m8.7-cpu-parallel-tuning-v1"
PREFIX = "gpmeep-adjoint-benchmark:"
FINAL_ARTIFACTS = ("report.json", "report.md", "COMPLETE", "FAILED.json")
RECOVERABLE_PUBLICATION_ARTIFACTS = (
    "report.md",
    "report.json",
    "PUBLISHING.json",
    "COMPLETE",
)
CHECKPOINT_DIRECTORY = "checkpoint"
CHECKPOINT_ANCHOR = "run.json"
CHECKPOINT_ENTRIES_DIRECTORY = "entries"
CHECKPOINT_RESUMES_DIRECTORY = "resumes"
CHECKPOINT_FAILURES_DIRECTORY = "failures"
CHECKPOINT_RECOVERIES_DIRECTORY = "recoveries"
CHECKPOINT_RECOVERY_INTENTS_DIRECTORY = "recovery-intents"
CHECKPOINT_ORPHANS_DIRECTORY = "orphans"
CHECKPOINT_SCHEMA_VERSION = 1
# The legacy schedule remains fixed until its checkpoint/recovery format is
# migrated to the topology-derived v2 matrix.  It is interim-only (see execute).
REFERENCE_SCHEDULE_INDEX = 8
RUN_LOCK_FILE = ".cpu-parallel-benchmark.lock"
RUNNER_SOURCES = (
    "scripts/benchmark-adjoint.py",
    "scripts/compare-mpi-adjoint-benchmarks.py",
    "scripts/gpmeep_benchmark_evidence.py",
    "scripts/gpmeep_cpu_topology_policy.py",
    "scripts/gpmeep_provenance.py",
    "scripts/run-cpu-parallel-benchmark.py",
    "scripts/run-mpi-adjoint-benchmark.py",
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
PHASES = ("update_design", "forward", "adjoint", "gradient")
CPU_COUNTERS = (
    ("dispatch", "cpu_curl_calls"),
    ("field_updates", "cpu_update_eh_calls"),
    ("sources", "cpu_source_calls"),
    ("boundaries", "cpu_boundary_calls"),
    ("dfts", "cpu_dft_calls"),
)
FINAL_REQUIRED_PER_RANK_CPU_COUNTERS = (
    "cpu_curl_calls",
    "cpu_update_eh_calls",
    "cpu_boundary_calls",
)
FINAL_REQUIRED_AGGREGATE_CPU_COUNTERS = (
    "cpu_source_calls",
    "cpu_dft_calls",
)
CUDA_COUNTERS = (
    ("dispatch", "cuda_curl_calls"),
    ("field_updates", "cuda_update_eh_calls"),
    ("sources", "cuda_source_calls"),
    ("boundaries", "cuda_boundary_calls"),
    ("dfts", "cuda_dft_calls"),
)
PHASE_REQUIRED_PER_RANK_CPU_COUNTERS = ("curl", "update_eh", "boundary")
PHASE_REQUIRED_AGGREGATE_CPU_COUNTERS = ("source", "dft")
STATISTICS_COUNTER_SCHEMA = types.MappingProxyType(
    {
        "boundaries": (
            "cpu_boundary_calls",
            "cpu_boundary_points",
            "cuda_boundary_calls",
            "cuda_boundary_points",
        ),
        "boundary_eh_overlap": (
            "boundary_eh_overlap_checks",
            "boundary_eh_overlap_eligible",
            "boundary_eh_overlap_launched_e",
            "boundary_eh_overlap_launched_h",
            "boundary_eh_overlap_rejected",
            "boundary_eh_overlap_skipped_cold_topology",
            "boundary_eh_overlap_skipped_disabled",
            "boundary_eh_overlap_skipped_no_remote",
            "boundary_eh_overlap_skipped_unsupported_schedule",
        ),
        "dfts": (
            "cpu_dft_calls",
            "cpu_dft_points",
            "cuda_dft_calls",
            "cuda_dft_points",
        ),
        "ldos": (
            "cpu_ldos_reduction_calls",
            "cpu_ldos_source_points",
            "cuda_ldos_reduction_calls",
            "cuda_ldos_submitted_profiles",
            "cuda_ldos_source_points",
            "cuda_ldos_descriptor_uploads",
            "cuda_ldos_kernel_launches",
            "cuda_ldos_result_device_to_host_bytes",
            "ldos_full_field_device_to_host_bytes_avoided",
        ),
        "dispatch": (
            "cpu_curl_calls",
            "cpu_curl_points",
            "cuda_curl_calls",
            "cuda_curl_points",
            "device_to_host_bytes",
            "host_to_device_bytes",
        ),
        "field_updates": (
            "cpu_update_eh_calls",
            "cpu_update_eh_points",
            "cuda_update_eh_calls",
            "cuda_update_eh_points",
        ),
        "halo_curl_overlap": (
            "halo_curl_overlap_checks",
            "halo_curl_overlap_eligible",
            "halo_curl_overlap_full_points",
            "halo_curl_overlap_interior_points",
            "halo_curl_overlap_launches",
            "halo_curl_overlap_rejected_feature",
            "halo_curl_overlap_rejected_small",
            "halo_curl_overlap_shell_points",
            "halo_curl_overlap_skipped_cold_topology",
            "halo_curl_overlap_skipped_disabled",
            "halo_curl_overlap_skipped_no_remote",
            "halo_curl_overlap_skipped_unsupported_schedule",
        ),
        "multi_gpu": (
            "cuda_aware_bytes",
            "mpi_messages",
            "mpi_scalars",
            "pinned_device_to_host_bytes",
            "pinned_host_to_device_bytes",
            "pinned_staging_bytes",
        ),
        "mpi_completion": (
            "mpi_waitall_executions",
            "mpi_waitsome_executions",
        ),
        "polarizations": (
            "cpu_polarization_calls",
            "cpu_polarization_points",
            "cuda_polarization_calls",
            "cuda_polarization_points",
        ),
        "resident": (
            "device_buffer_allocations",
            "device_buffer_reuses",
            "device_to_host_bytes_avoided",
            "host_to_device_bytes_avoided",
            "live_resident_device_buffers",
        ),
        "runtime": (
            "runtime_availability_probes",
            "runtime_device_enumerations",
            "runtime_device_selections",
        ),
        "sources": (
            "cpu_source_calls",
            "cpu_source_points",
            "cuda_source_calls",
            "cuda_source_points",
        ),
        "tile_coalescing": (
            "tile_coalesced_curl_chunk_phases",
            "tile_coalesced_curl_input_tiles",
            "tile_coalesced_update_eh_chunk_phases",
            "tile_coalesced_update_eh_input_tiles",
        ),
    }
)
CPU_DISABLED_OVERLAP_COUNTERS = types.MappingProxyType(
    {
        "boundary_eh_overlap": (
            "boundary_eh_overlap_checks",
            "boundary_eh_overlap_skipped_disabled",
        ),
        "halo_curl_overlap": (
            "halo_curl_overlap_checks",
            "halo_curl_overlap_skipped_disabled",
        ),
    }
)
PHASE_COUNTER_STATISTICS_FIELDS = types.MappingProxyType(
    {
        "curl": ("dispatch", "curl"),
        "update_eh": ("field_updates", "update_eh"),
        "polarization": ("polarizations", "polarization"),
        "source": ("sources", "source"),
        "boundary": ("boundaries", "boundary"),
        "dft": ("dfts", "dft"),
    }
)
ALLOWED_LAUNCHER_INJECTED_INFLUENCE_VARIABLES = {
    "CUDA_VISIBLE_DEVICES",
    "OMPI_MCA_PREFIXES",
    "OMPI_MCA_cpu_type",
    "OMPI_MCA_initial_wdir",
    "OMPI_MCA_num_procs",
}
LAUNCHER_CONSUMED_ENVIRONMENT_KEYS = {"PRTE_MCA_mca_base_param_files"}
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
    "FONTCONFIG_",
    "OMP_",
    "OPENBLAS_",
    "MKL_",
    "NUMEXPR_",
    "BLIS_",
    "VECLIB_",
    "UCX_",
    "NCCL_",
    "HDF5_",
    "PYTHON",
    "LD_",
    "OMPI_MCA_",
    "PMIX_MCA_",
    "PRTE_MCA_",
)
_PUBLISH_CAPABILITY = object()


class AlreadyComplete(RuntimeError):
    """Raised when explicit resume finds an already closed publication."""


@dataclass(frozen=True, slots=True)
class CpuConfig:
    config_id: str
    ranks: int
    omp_threads: int
    binding: str
    lane: str
    physical_budget: int
    full_physical: bool
    canonical_full_physical: bool
    rank_core_plan: list[list[list[int]]]
    rank_logical_plan: list[list[int]]
    rank_numa_plan: list[int]

    @property
    def workers(self) -> int:
        return self.ranks * self.omp_threads


def _legacy_interim_config(
    config_id: str,
    ranks: int,
    omp_threads: int,
    binding: str,
    lane: str,
    *,
    canonical: bool = False,
) -> CpuConfig:
    if binding == "hwthread":
        # Match Open MPI/hwloc topology traversal for this explicitly bounded
        # 8C/16T host: both hardware threads of a core are ranked together.
        # The Linux CPU IDs are 0..7 for the first siblings and 8..15 for the
        # second siblings, hence the rank order is 0,8,1,9,... rather than a
        # numeric 0..15 ordering.
        core_plan = [[[0, 0, rank // 2]] for rank in range(ranks)]
        logical_plan = [[rank // 2 + (rank % 2) * 8] for rank in range(ranks)]
    else:
        core_plan = [
            [[0, 0, core] for core in range(rank * omp_threads, (rank + 1) * omp_threads)]
            for rank in range(ranks)
        ]
        logical_plan = [
            sorted(cpu for core in group for cpu in (core[2], core[2] + 8))
            for group in core_plan
        ]
    physical_budget = 1 if lane == "diagnostic" else min(8, ranks * omp_threads)
    return CpuConfig(
        config_id=config_id,
        ranks=ranks,
        omp_threads=omp_threads,
        binding=binding,
        lane=lane,
        physical_budget=physical_budget,
        full_physical=lane == "physical" and physical_budget == 8,
        canonical_full_physical=canonical,
        rank_core_plan=core_plan,
        rank_logical_plan=logical_plan,
        rank_numa_plan=[0] * ranks,
    )


CONFIGS: tuple[CpuConfig, ...] = (
    _legacy_interim_config("1x1", 1, 1, "core", "diagnostic"),
    _legacy_interim_config("2x1", 2, 1, "core", "physical"),
    _legacy_interim_config("4x1", 4, 1, "core", "physical"),
    _legacy_interim_config("8x1", 8, 1, "core", "physical"),
    _legacy_interim_config("16x1", 16, 1, "hwthread", "smt"),
    _legacy_interim_config("1x8", 1, 8, "core", "physical", canonical=True),
    _legacy_interim_config("2x4", 2, 4, "core", "physical"),
    _legacy_interim_config("4x2", 4, 2, "core", "physical"),
)
TOPOLOGY_MATRIX: dict[str, Any] | None = None
PROFILE = types.MappingProxyType(
    {
        "profile_id": PROFILE_ID,
        "producer_qualification_profile_id": PRODUCER_QUALIFICATION_PROFILE_ID,
        "resolution": 64,
        "run_time": 40.0,
        "cell_size": 16.0,
        "design_resolution": 20,
        "design_shape": [41, 41],
        "design_variables": 1681,
        "yee_cells": 1048576,
        "weights_sha256": "441c6df996857239098db8694ec724f6e3ec2312042fa2a667c650d3c8c58ada",
        "direction_sha256": "581786e703e71a2548224e070d2b9da0bab5de69476f9be5357441175323fb1a",
        "material_gradient_path": "analytic-required",
        "forward_timesteps": 6614,
        "forward_end_time": 51.671875,
        "adjoint_timesteps": 19840,
        "adjoint_end_time": 155.0,
        "topology_policy_schema_version": _TOPOLOGY_POLICY.POLICY_SCHEMA_VERSION,
        "minimum_physical_cores": 2,
        "diagnostic_lane_selectable": False,
        "smt_lane_competes_with_physical": False,
        "warmups": 1,
        "repeats": 5,
        "rotation_stride": 3,
        "objective_atol": 1.0e-8,
        "gradient_atol": 2.0e-7,
        "gradient_rtol": 5.0e-5,
        "maximum_workload_cv": 0.05,
        "maximum_elapsed_cv": 0.05,
        "timeout_seconds": 900,
        "timeout_kill_grace_seconds": 5,
    }
)
# Updated deliberately whenever the immutable fixed profile changes.
PROFILE_CANONICAL_SHA256 = (
    "ef4ba6911b7a2f01ecd5c9647d8008a25c25409f4a97fae4558c4442ec4e240b"
)


def _assert_fixed_profile() -> None:
    if (
        not isinstance(PROFILE, types.MappingProxyType)
        or canonical_sha256(dict(PROFILE)) != PROFILE_CANONICAL_SHA256
    ):
        raise RuntimeError("fixed CPU parallel tuning profile was modified")


def activate_topology_policy(topology: dict[str, Any]) -> dict[str, Any]:
    """Activate the deterministic matrix bound to one normalized topology."""

    global CONFIGS, TOPOLOGY_MATRIX, REFERENCE_SCHEDULE_INDEX
    matrix = _TOPOLOGY_POLICY.generate_policy(topology)
    configs = tuple(CpuConfig(**record) for record in matrix["configs"])
    if len(configs) < 2 or configs[0].lane != "diagnostic":
        raise RuntimeError("CPU topology policy did not produce the required lanes")
    CONFIGS = configs
    TOPOLOGY_MATRIX = matrix
    reference = (matrix["diagnostic_config_id"], "measured", 0)
    REFERENCE_SCHEDULE_INDEX = sample_schedule().index(reference)
    return matrix


def _require_topology_matrix() -> dict[str, Any]:
    if TOPOLOGY_MATRIX is None or not CONFIGS or REFERENCE_SCHEDULE_INDEX is None:
        raise RuntimeError("CPU topology policy has not been activated")
    return TOPOLOGY_MATRIX


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _absolute_lexical(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(
    path: pathlib.Path, *, require_leaf: bool, label: str
) -> pathlib.Path:
    absolute = _absolute_lexical(path)
    current = pathlib.Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, part in enumerate(parts):
        current = current / part
        is_leaf = index == len(parts) - 1
        try:
            status = current.lstat()
        except FileNotFoundError:
            if require_leaf or not is_leaf:
                raise RuntimeError(f"{label} path component is absent: {current}")
            break
        if stat.S_ISLNK(status.st_mode):
            raise RuntimeError(f"{label} path contains a symlink: {current}")
        if not is_leaf and not stat.S_ISDIR(status.st_mode):
            raise RuntimeError(f"{label} ancestor is not a directory: {current}")
    return absolute


def _trusted_output_path(path: pathlib.Path, *, create: bool) -> pathlib.Path:
    absolute = _absolute_lexical(path)
    if create:
        existing = absolute
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        _assert_no_symlink_components(existing, require_leaf=True, label="output")
        absolute.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(absolute, require_leaf=True, label="output")
    status = absolute.lstat()
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError("CPU benchmark output is not a directory")
    return absolute


def _require_within_output(path: pathlib.Path, output: pathlib.Path, *, label: str) -> pathlib.Path:
    absolute = _assert_no_symlink_components(path, require_leaf=True, label=label)
    trusted_output = _absolute_lexical(output)
    try:
        absolute.relative_to(trusted_output)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes the expected output directory") from error
    return absolute


def _regular_evidence_record(
    path: pathlib.Path,
    *,
    output: pathlib.Path | None = None,
    expected_parent: pathlib.Path | None = None,
    label: str = "evidence",
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    absolute = _assert_no_symlink_components(path, require_leaf=True, label=label)
    if output is not None:
        _require_within_output(absolute, output, label=label)
    if expected_parent is not None and absolute.parent != _absolute_lexical(expected_parent):
        raise RuntimeError(f"{label} is outside its exact expected parent")
    status = absolute.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise RuntimeError(f"{label} is not a private regular single-link file")
    fingerprint = StatHashCache._fingerprint(status)
    cache = hash_cache if hash_cache is not None else StatHashCache()
    digest = cache.digest(absolute)
    if not cache._path_matches_fingerprint(absolute, fingerprint):
        raise RuntimeError(f"{label} changed while hashing")
    return {
        "path": str(absolute),
        "size_bytes": status.st_size,
        "sha256": digest,
    }


def _rename_noreplace(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Atomically rename without ever replacing an existing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise RuntimeError(f"immutable destination already exists: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _link_anonymous_file_noreplace(
    descriptor: int, parent: pathlib.Path, name: str
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    directory_descriptor = os.open(
        parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        result = linkat(
            descriptor,
            b"",
            directory_descriptor,
            os.fsencode(name),
            0x1000,  # AT_EMPTY_PATH
        )
    finally:
        os.close(directory_descriptor)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise RuntimeError(f"immutable destination already exists: {parent / name}")
    if error_number in (errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM):
        raise RuntimeError(
            "safe benchmark-evidence publication requires Linux "
            "O_TMPFILE+linkat(AT_EMPTY_PATH); filesystem/kernel refused it; "
            "no unsafe fallback is allowed"
        )
    raise OSError(error_number, os.strerror(error_number), str(parent / name))


def _evidence_publication_capability() -> dict[str, Any]:
    return {
        "scope": "benchmark-evidence-tooling-only; gpmeep runtime/package portability is unaffected",
        "platform": sys.platform,
        "primitive": "Linux O_TMPFILE + linkat(AT_EMPTY_PATH)",
        "atomic_visibility": True,
        "no_clobber": True,
        "single_link_at_publication": True,
        "unsafe_fallback_allowed": False,
    }


def _atomic_create_json(path: pathlib.Path, value: Any) -> None:
    """Durably publish one immutable JSON file without replacing an old file."""

    path = pathlib.Path(path)
    existing_parent = path.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    _assert_no_symlink_components(
        existing_parent, require_leaf=True, label="checkpoint parent"
    )
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(
        path.parent, require_leaf=True, label="checkpoint parent"
    )
    if not parent_existed:
        _fsync_directory(path.parent)
        _fsync_directory(path.parent.parent)
    if not hasattr(os, "O_TMPFILE"):
        raise RuntimeError("Linux O_TMPFILE is required for durable publication")
    try:
        descriptor = os.open(path.parent, os.O_RDWR | os.O_TMPFILE, 0o600)
    except OSError as error:
        if error.errno in (errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM):
            raise RuntimeError(
                "safe benchmark-evidence publication requires Linux O_TMPFILE; "
                "filesystem/kernel refused it; no unsafe fallback is allowed"
            ) from error
        raise
    try:
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        _link_anonymous_file_noreplace(descriptor, path.parent, path.name)
        _fsync_directory(path.parent)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def output_lock(output: pathlib.Path):
    """Hold a nonblocking, process-lifetime lock for one evidence directory."""

    output = _trusted_output_path(output, create=True)
    lock_path = output / RUN_LOCK_FILE
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("CPU benchmark lock requires O_NOFOLLOW support")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(f"cannot safely open CPU benchmark lock {lock_path}") from error
    try:
        path_status = lock_path.lstat()
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(path_status.st_mode)
            or not stat.S_ISREG(descriptor_status.st_mode)
            or path_status.st_nlink != 1
            or descriptor_status.st_nlink != 1
            or (path_status.st_dev, path_status.st_ino)
            != (descriptor_status.st_dev, descriptor_status.st_ino)
            or descriptor_status.st_mode & 0o077
        ):
            raise RuntimeError("CPU benchmark lock is not a private single-link file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another CPU benchmark writer holds {lock_path}"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _boot_identity() -> dict[str, Any]:
    path = pathlib.Path("/proc/sys/kernel/random/boot_id")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError("Linux boot identity is unavailable") from error
    text = payload.decode("ascii", errors="strict").strip().lower()
    if not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        text,
    ):
        raise RuntimeError("Linux boot identity is malformed")
    return {
        "path": str(path),
        "boot_id": text,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _host_identity() -> dict[str, str]:
    hostname = socket.gethostname()
    nodename = os.uname().nodename
    if not hostname or not nodename:
        raise RuntimeError("host identity is unavailable")
    return {"hostname": hostname, "uname_nodename": nodename}


def _runner_provenance(
    repo: pathlib.Path, *, hash_cache: StatHashCache | None = None
) -> dict[str, Any]:
    cache = hash_cache if hash_cache is not None else StatHashCache()
    sources = {name: cache.digest(repo / name) for name in RUNNER_SOURCES}
    python = pathlib.Path(sys.executable).resolve()
    record = {
        "sources": sources,
        "python": {
            "path": str(python),
            "size_bytes": python.stat().st_size,
            "sha256": cache.digest(python),
            "version": sys.version,
        },
    }
    return {"record": record, "sha256": canonical_sha256(record)}


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise RuntimeError(f"{label} must be finite and positive")
    return converted


def _validate_process_wall_seconds(
    process_seconds: Any, validation: dict[str, Any], *, label: str
) -> float:
    process_wall = _finite_positive(process_seconds, f"{label}.process_seconds")
    workload_wall = _finite_positive(
        validation.get("workload_wall_seconds"), f"{label}.workload_wall_seconds"
    )
    if process_wall < workload_wall:
        raise RuntimeError(f"{label} process wall time is shorter than workload wall time")
    return process_wall


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the exact trusted prefix in an existing output directory",
    )
    parser.add_argument(
        "--host-specific-interim",
        action="store_true",
        help=(
            "explicitly allow the legacy 1-socket/8-core interim matrix; this "
            "cannot produce a portable/final CPU-vs-GPU headline"
        ),
    )
    return parser.parse_args(argv)


def parse_lscpu_extended(value: Any, allowed_cpus: set[int]) -> dict[str, Any]:
    """Normalize online lscpu rows after intersecting the process cpuset."""

    rows = value.get("cpus") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("lscpu extended topology is absent")
    normalized: list[dict[str, int]] = []
    seen: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("online") is not True:
            continue
        fields: dict[str, int] = {}
        for name in ("cpu", "core", "socket", "node"):
            raw = row.get(name)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise RuntimeError(f"lscpu row {index} has invalid {name}")
            fields[name] = raw
        if fields["cpu"] in seen:
            raise RuntimeError("lscpu topology contains a duplicate logical CPU")
        seen.add(fields["cpu"])
        if fields["cpu"] in allowed_cpus:
            normalized.append(fields)
    if {row["cpu"] for row in normalized} != allowed_cpus:
        raise RuntimeError("process affinity contains CPUs absent from online lscpu topology")
    sockets = {row["socket"] for row in normalized}
    nodes = {row["node"] for row in normalized}
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for row in normalized:
        grouped.setdefault(
            (row["socket"], row["node"], row["core"]), []
        ).append(row["cpu"])
    if len(grouped) < PROFILE["minimum_physical_cores"]:
        raise RuntimeError(
            "normal CPU benchmark publication requires at least two allowed physical cores"
        )
    logical = sorted(normalized, key=lambda row: row["cpu"])
    cores = [
        {
            "socket": socket,
            "node": node,
            "core": core,
            "logical_cpus": sorted(cpus),
        }
        for (socket, node, core), cpus in sorted(grouped.items())
    ]
    sibling_counts = sorted({len(core["logical_cpus"]) for core in cores})
    return {
        "available_logical_cpus": sorted(allowed_cpus),
        "logical_cpu_count": len(logical),
        "physical_core_count": len(cores),
        "threads_per_core": (
            sibling_counts[0] if len(sibling_counts) == 1 else None
        ),
        "allowed_threads_per_core": sibling_counts,
        "sockets": sorted(sockets),
        "numa_nodes": sorted(nodes),
        "logical_cpus": logical,
        "cores": cores,
    }


def _validate_legacy_interim_topology(topology: dict[str, Any]) -> None:
    """Fail early if the opt-in legacy matrix is used on another CPU layout."""

    expected_cores = [
        {
            "socket": 0,
            "node": 0,
            "core": core,
            "logical_cpus": [core, core + 8],
        }
        for core in range(8)
    ]
    if (
        topology.get("available_logical_cpus") != list(range(16))
        or topology.get("cores") != expected_cores
    ):
        raise RuntimeError(
            "--host-specific-interim is restricted to its recorded 1S/1NUMA/"
            "8C/16T CPU layout; portable topology-derived execution is not yet enabled"
        )


def _trusted_lscpu() -> pathlib.Path:
    for candidate in (pathlib.Path("/usr/bin/lscpu"), pathlib.Path("/bin/lscpu")):
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        resolved = candidate.resolve(strict=True)
        if any(
            path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            for path in (resolved.parent, resolved)
        ):
            continue
        return resolved
    raise RuntimeError("protected lscpu executable is unavailable")


def _probe(command: list[str], *, timeout: int = 30) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd="/",
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr": completed.stderr,
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
    }


CPU_FREQUENCY_STATE_NAMES = (
    "scaling_governor",
    "scaling_driver",
    "energy_performance_preference",
    "scaling_cur_freq",
    "scaling_min_freq",
    "scaling_max_freq",
    "cpuinfo_min_freq",
    "cpuinfo_max_freq",
)
POWER_CONFIGURATION_NAMES = {
    "scaling_governor",
    "scaling_driver",
    "energy_performance_preference",
    "scaling_min_freq",
    "scaling_max_freq",
    "cpuinfo_min_freq",
    "cpuinfo_max_freq",
}


def _read_system_text_record(path: pathlib.Path) -> dict[str, Any]:
    """Capture one read-only sysfs/procfs value with exact byte identity."""

    path = path.absolute()
    if path != pathlib.Path("/proc/loadavg") and pathlib.Path("/sys") not in path.parents:
        raise RuntimeError(f"system-state path is outside protected roots: {path}")
    try:
        resolved = path.resolve(strict=True)
        payload = path.read_bytes()
    except FileNotFoundError:
        return {"path": str(path), "status": "absent"}
    except OSError as error:
        return {
            "path": str(path),
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "status": "ok",
        "text": payload.decode("utf-8", errors="strict"),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def capture_power_load_state(logical_cpus: list[int]) -> dict[str, Any]:
    if (
        not logical_cpus
        or logical_cpus != sorted(set(logical_cpus))
        or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in logical_cpus)
    ):
        raise RuntimeError("power-state CPU list is invalid")
    global_records = {
        "intel_pstate_status": _read_system_text_record(
            pathlib.Path("/sys/devices/system/cpu/intel_pstate/status")
        ),
        "intel_pstate_no_turbo": _read_system_text_record(
            pathlib.Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
        ),
        "cpufreq_boost": _read_system_text_record(
            pathlib.Path("/sys/devices/system/cpu/cpufreq/boost")
        ),
    }
    cpu_records = {
        str(cpu): {
            "cpufreq": {
                name: _read_system_text_record(
                    pathlib.Path(
                        f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}"
                    )
                )
                for name in CPU_FREQUENCY_STATE_NAMES
            },
            "thermal_throttle": {
                name: _read_system_text_record(
                    pathlib.Path(
                        f"/sys/devices/system/cpu/cpu{cpu}/thermal_throttle/{name}"
                    )
                )
                for name in ("core_throttle_count", "package_throttle_count")
            },
        }
        for cpu in logical_cpus
    }
    loadavg = _read_system_text_record(pathlib.Path("/proc/loadavg"))
    if loadavg.get("status") != "ok":
        raise RuntimeError("/proc/loadavg evidence is unavailable")
    columns = loadavg["text"].split()
    try:
        loads = [float(columns[index]) for index in range(3)]
    except (IndexError, TypeError, ValueError) as error:
        raise RuntimeError("/proc/loadavg evidence is malformed") from error
    if any(not math.isfinite(value) or value < 0 for value in loads):
        raise RuntimeError("/proc/loadavg contains an invalid load value")
    loadavg["load_1m"] = loads[0]
    loadavg["load_5m"] = loads[1]
    loadavg["load_15m"] = loads[2]
    return {
        "schema_version": 1,
        "captured_at_utc": _utc_now(),
        "logical_cpus": logical_cpus,
        "global": global_records,
        "per_cpu": cpu_records,
        "loadavg": loadavg,
    }


def _power_configuration_projection(state: dict[str, Any]) -> dict[str, Any]:
    global_records = state.get("global")
    per_cpu = state.get("per_cpu")
    if not isinstance(global_records, dict) or not isinstance(per_cpu, dict):
        raise RuntimeError("power-state evidence is incomplete")
    return {
        "logical_cpus": state.get("logical_cpus"),
        "global": global_records,
        "per_cpu": {
            cpu: {
                name: record
                for name, record in records.get("cpufreq", {}).items()
                if name in POWER_CONFIGURATION_NAMES
            }
            for cpu, records in per_cpu.items()
            if isinstance(records, dict)
        },
    }


def validate_power_load_comparability(
    start: dict[str, Any], end: dict[str, Any]
) -> dict[str, Any]:
    if _power_configuration_projection(start) != _power_configuration_projection(end):
        raise RuntimeError("CPU governor/driver/EPP/frequency limits changed during tuning")
    throttle_deltas: dict[str, dict[str, int | None]] = {}
    for cpu, start_records in start.get("per_cpu", {}).items():
        end_records = end.get("per_cpu", {}).get(cpu, {})
        throttle_deltas[cpu] = {}
        for name, start_record in start_records.get("thermal_throttle", {}).items():
            end_record = end_records.get("thermal_throttle", {}).get(name)
            if (
                isinstance(end_record, dict)
                and start_record.get("status") == end_record.get("status") == "absent"
            ):
                throttle_deltas[cpu][name] = None
                continue
            if (
                start_record.get("status") != "ok"
                or not isinstance(end_record, dict)
                or end_record.get("status") != "ok"
            ):
                raise RuntimeError("CPU thermal-throttle counter evidence changed availability")
            try:
                before = int(start_record["text"].strip())
                after = int(end_record["text"].strip())
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("CPU thermal-throttle counter is malformed") from error
            delta = after - before
            if delta != 0:
                raise RuntimeError("CPU thermal-throttle counter increased during tuning")
            throttle_deltas[cpu][name] = delta
    start_load = start.get("loadavg", {})
    end_load = end.get("loadavg", {})
    if any(
        isinstance(record.get(name), bool)
        or not isinstance(record.get(name), (int, float))
        for record in (start_load, end_load)
        for name in ("load_1m", "load_5m", "load_15m")
    ):
        raise RuntimeError("CPU tuning load-average comparison is incomplete")
    return {
        "pass": True,
        "host_specific": True,
        "scope": "results are specific to the recorded CPU model, topology, power policy, and system load",
        "configuration_unchanged": True,
        "thermal_throttle_counter_deltas": throttle_deltas,
        "start_load": {
            name: start_load[name] for name in ("load_1m", "load_5m", "load_15m")
        },
        "end_load": {
            name: end_load[name] for name in ("load_1m", "load_5m", "load_15m")
        },
    }


def _capture_topology_record(repo: pathlib.Path) -> dict[str, Any]:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("CPU tuning requires sched_getaffinity support")
    allowed = {int(cpu) for cpu in os.sched_getaffinity(0)}
    if not allowed:
        raise RuntimeError("CPU tuning parent has an empty CPU affinity")
    lscpu = _trusted_lscpu()
    extended = _probe(
        [str(lscpu), "--json", "--extended=CPU,CORE,SOCKET,NODE,ONLINE"]
    )
    summary = _probe([str(lscpu), "--json"])
    if extended["exit_code"] != 0 or summary["exit_code"] != 0:
        raise RuntimeError("lscpu topology probe failed")
    try:
        parsed_extended = json.loads(extended["stdout"])
        parsed_summary = json.loads(summary["stdout"])
    except json.JSONDecodeError as error:
        raise RuntimeError("lscpu returned malformed JSON") from error
    topology = parse_lscpu_extended(parsed_extended, allowed)
    summary_rows = parsed_summary.get("lscpu") if isinstance(parsed_summary, dict) else None
    if not isinstance(summary_rows, list):
        raise RuntimeError("lscpu summary is absent")
    summary_fields = {
        str(row["field"]).rstrip(":"): row.get("data")
        for row in summary_rows
        if isinstance(row, dict) and isinstance(row.get("field"), str)
    }
    return {
        "schema_version": 1,
        "captured_at_utc": _utc_now(),
        "parent_pid": os.getpid(),
        "parent_affinity": sorted(allowed),
        "normalized": topology,
        "power_load_state_start": capture_power_load_state(
            topology["available_logical_cpus"]
        ),
        "summary": summary_fields,
        "lscpu_executable": file_record(lscpu, repo),
        "probes": {"extended": extended, "summary": summary},
    }


def capture_topology(repo: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    record = _capture_topology_record(repo)
    path = output / "topology.json"
    atomic_write_json(path, record)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "record": record}


def _topology_identity(record: dict[str, Any]) -> dict[str, Any]:
    summary = record.get("summary")
    stable_summary_names = (
        "Architecture",
        "CPU(s)",
        "On-line CPU(s) list",
        "Model name",
        "Socket(s)",
        "Core(s) per socket",
        "Thread(s) per core",
        "NUMA node(s)",
    )
    return {
        "parent_affinity": record.get("parent_affinity"),
        "normalized": record.get("normalized"),
        "summary": {
            name: summary.get(name)
            for name in stable_summary_names
        }
        if isinstance(summary, dict)
        else None,
        "lscpu_executable": record.get("lscpu_executable"),
    }


def _checkpoint_root(output: pathlib.Path) -> pathlib.Path:
    return output / CHECKPOINT_DIRECTORY


def _anchor_path(output: pathlib.Path) -> pathlib.Path:
    return _checkpoint_root(output) / CHECKPOINT_ANCHOR


def _entry_path(output: pathlib.Path, index: int) -> pathlib.Path:
    return _checkpoint_root(output) / CHECKPOINT_ENTRIES_DIRECTORY / f"{index:06d}.json"


def _resume_path(output: pathlib.Path, index: int) -> pathlib.Path:
    return _checkpoint_root(output) / CHECKPOINT_RESUMES_DIRECTORY / f"{index:06d}.json"


def _failure_path(output: pathlib.Path, index: int) -> pathlib.Path:
    if index == 0:
        return output / "FAILED.json"
    return (
        _checkpoint_root(output)
        / CHECKPOINT_FAILURES_DIRECTORY
        / f"{index:06d}.json"
    )


def _recovery_path(output: pathlib.Path, index: int) -> pathlib.Path:
    return _checkpoint_root(output) / CHECKPOINT_RECOVERIES_DIRECTORY / f"{index:06d}.json"


def _recovery_intent_path(output: pathlib.Path, index: int) -> pathlib.Path:
    return (
        _checkpoint_root(output)
        / CHECKPOINT_RECOVERY_INTENTS_DIRECTORY
        / f"{index:06d}.json"
    )


def _orphan_path(output: pathlib.Path, index: int) -> pathlib.Path:
    return _checkpoint_root(output) / CHECKPOINT_ORPHANS_DIRECTORY / f"{index:06d}"


def _read_json_object(path: pathlib.Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _resolved_file_record(
    path: pathlib.Path, *, hash_cache: StatHashCache | None = None
) -> dict[str, Any]:
    return _regular_evidence_record(
        path, label="checkpoint artifact", hash_cache=hash_cache
    )


def _matrix_binding(topology_record: dict[str, Any]) -> dict[str, Any]:
    normalized = topology_record.get("normalized")
    record = {
        "schema_version": 2,
        "execution_scope": "host-specific-interim",
        "portable_final": False,
        "topology_sha256": canonical_sha256(normalized),
        "configs": [asdict(config) for config in CONFIGS],
        "schedule": [list(row) for row in sample_schedule()],
    }
    return {"record": record, "sha256": canonical_sha256(record)}


def _anchor_payload(
    *,
    repo: pathlib.Path,
    output: pathlib.Path,
    run_id: str,
    receipt_path: pathlib.Path,
    receipt: dict[str, Any],
    snapshot: dict[str, Any],
    producer_sha256: str,
    topology_evidence: dict[str, Any],
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    producer_path = (repo / "scripts" / "benchmark-adjoint.py").resolve()
    cache = hash_cache if hash_cache is not None else StatHashCache()
    runner = _runner_provenance(repo, hash_cache=cache)
    topology_record = topology_evidence["record"]
    power_policy = _power_configuration_projection(
        topology_record["power_load_state_start"]
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "gpmeep-cpu-parallel-run-anchor",
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "profile": dict(PROFILE),
        "profile_sha256": canonical_sha256(dict(PROFILE)),
        "schedule": [list(row) for row in sample_schedule()],
        "schedule_sha256": canonical_sha256(sample_schedule()),
        "matrix": _matrix_binding(topology_record),
        "repo": str(repo.resolve()),
        "output": str(output.resolve()),
        "receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": cache.digest(receipt_path),
            "receipt_id": receipt["receipt_id"],
            "build_input_id": receipt["build_input_id"],
            "artifact_set_id": receipt["artifact_set_id"],
        },
        "source_snapshot": snapshot,
        "producer": {
            "path": str(producer_path),
            "size_bytes": producer_path.stat().st_size,
            "sha256": producer_sha256,
        },
        "runner_provenance": runner,
        "evidence_publication_capability": _evidence_publication_capability(),
        "topology": {
            "path": topology_evidence["path"],
            "sha256": topology_evidence["sha256"],
            "identity_sha256": canonical_sha256(_topology_identity(topology_record)),
        },
        "boot": _boot_identity(),
        "host": _host_identity(),
        "parent_affinity": topology_record["parent_affinity"],
        "power_policy": {
            "record": power_policy,
            "sha256": canonical_sha256(power_policy),
        },
    }


def _publish_run_anchor(output: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = _anchor_path(output)
    _atomic_create_json(path, payload)
    file = _regular_evidence_record(
        path, output=output, expected_parent=_checkpoint_root(output), label="run anchor"
    )
    return {"path": file["path"], "sha256": file["sha256"], "record": payload}


def _verify_anchor_evidence_on_disk(
    output: pathlib.Path,
    anchor: dict[str, Any],
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    expected_path = _anchor_path(output)
    file = _regular_evidence_record(
        expected_path,
        output=output,
        expected_parent=_checkpoint_root(output),
        label="run anchor",
        hash_cache=hash_cache,
    )
    if (
        anchor.get("path") != file["path"]
        or anchor.get("sha256") != file["sha256"]
    ):
        raise RuntimeError("in-memory/on-disk run anchor binding changed")
    record = _read_json_object(expected_path, label="CPU checkpoint run anchor")
    if anchor.get("record") != record:
        raise RuntimeError("in-memory/on-disk run anchor JSON changed")
    return {"path": file["path"], "sha256": file["sha256"], "record": record}


def _validate_run_anchor(
    *,
    repo: pathlib.Path,
    output: pathlib.Path,
    receipt_path: pathlib.Path,
    receipt: dict[str, Any],
    snapshot: dict[str, Any],
    producer_sha256: str,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    cache = hash_cache if hash_cache is not None else StatHashCache()
    path = _anchor_path(output)
    anchor_file = _regular_evidence_record(
        path,
        output=output,
        expected_parent=_checkpoint_root(output),
        label="resume run anchor",
        hash_cache=cache,
    )
    record = _read_json_object(path, label="CPU checkpoint run anchor")
    if (
        record.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or record.get("kind") != "gpmeep-cpu-parallel-run-anchor"
        or not re.fullmatch(r"[0-9a-f]{32}", str(record.get("run_id", "")))
        or record.get("profile") != dict(PROFILE)
        or record.get("profile_sha256") != canonical_sha256(dict(PROFILE))
        or record.get("schedule") != [list(row) for row in sample_schedule()]
        or record.get("schedule_sha256") != canonical_sha256(sample_schedule())
        or record.get("repo") != str(repo.resolve())
        or record.get("output") != str(output.resolve())
        or record.get("evidence_publication_capability")
        != _evidence_publication_capability()
    ):
        raise RuntimeError("resume run anchor identity/profile/output changed")
    expected_receipt = {
        "path": str(receipt_path.resolve()),
        "sha256": cache.digest(receipt_path),
        "receipt_id": receipt["receipt_id"],
        "build_input_id": receipt["build_input_id"],
        "artifact_set_id": receipt["artifact_set_id"],
    }
    producer_path = (repo / "scripts" / "benchmark-adjoint.py").resolve()
    expected_producer = {
        "path": str(producer_path),
        "size_bytes": producer_path.stat().st_size,
        "sha256": producer_sha256,
    }
    if (
        record.get("receipt") != expected_receipt
        or record.get("source_snapshot") != snapshot
        or record.get("producer") != expected_producer
        or record.get("runner_provenance")
        != _runner_provenance(repo, hash_cache=cache)
        or record.get("boot") != _boot_identity()
        or record.get("host") != _host_identity()
    ):
        raise RuntimeError("resume receipt/source/producer/runner/boot/host changed")
    topology_binding = record.get("topology")
    topology_recorded_path = (
        pathlib.Path(str(topology_binding.get("path", "")))
        if isinstance(topology_binding, dict)
        else pathlib.Path()
    )
    topology_path = _absolute_lexical(topology_recorded_path)
    topology_file = _regular_evidence_record(
        topology_path,
        output=output,
        expected_parent=output,
        label="CPU topology evidence",
        hash_cache=cache,
    )
    if (
        topology_path != _absolute_lexical(output / "topology.json")
        or topology_binding.get("sha256") != topology_file["sha256"]
    ):
        raise RuntimeError("resume topology evidence changed")
    topology_record = _read_json_object(topology_path, label="CPU topology evidence")
    if record.get("matrix") != _matrix_binding(topology_record):
        raise RuntimeError("resume CPU matrix/topology hash binding changed")
    if topology_binding.get("identity_sha256") != canonical_sha256(
        _topology_identity(topology_record)
    ):
        raise RuntimeError("resume topology identity binding changed")
    live = _capture_topology_record(repo)
    if (
        _topology_identity(live) != _topology_identity(topology_record)
        or record.get("parent_affinity") != live.get("parent_affinity")
    ):
        raise RuntimeError("resume host topology/affinity changed")
    live_policy = _power_configuration_projection(live["power_load_state_start"])
    if record.get("power_policy") != {
        "record": live_policy,
        "sha256": canonical_sha256(live_policy),
    }:
        raise RuntimeError("resume CPU power policy changed")
    continuity = validate_power_load_comparability(
        topology_record["power_load_state_start"],
        live["power_load_state_start"],
    )
    return {
        "path": anchor_file["path"],
        "sha256": anchor_file["sha256"],
        "record": record,
        "topology_evidence": {
            "path": str(topology_path),
            "sha256": topology_binding["sha256"],
            "record": topology_record,
        },
        "resume_live_state": live["power_load_state_start"],
        "resume_power_continuity": continuity,
    }


def _config_by_id(config_id: str) -> CpuConfig:
    matches = [config for config in CONFIGS if config.config_id == config_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown CPU configuration {config_id!r}")
    return matches[0]


def binding_arguments(config: CpuConfig) -> list[str]:
    if len(set(config.rank_numa_plan)) > 1:
        raise RuntimeError(
            "portable multi-NUMA launcher placement is not implemented; refusing "
            "to launch a topology plan that cannot yet be realized exactly"
        )
    if config.binding == "core" and config.omp_threads == 1:
        return [
            "--map-by",
            "core",
            "--bind-to",
            "core",
        ]
    if config.binding == "core":
        return [
            "--map-by",
            f"ppr:{config.ranks}:node:PE={config.omp_threads}",
            "--bind-to",
            "core",
        ]
    if config.binding == "hwthread" and config.omp_threads == 1:
        return [
            "--use-hwthread-cpus",
            "--map-by",
            "hwthread",
            "--bind-to",
            "hwthread",
        ]
    raise RuntimeError(f"unsupported CPU binding for {config.config_id}")


def sample_schedule() -> list[tuple[str, str, int]]:
    schedule: list[tuple[str, str, int]] = []
    for iteration in range(PROFILE["warmups"]):
        schedule.extend((config.config_id, "warmup", iteration) for config in CONFIGS)
    for iteration in range(PROFILE["repeats"]):
        offset = (iteration * PROFILE["rotation_stride"]) % len(CONFIGS)
        rotated = CONFIGS[offset:] + CONFIGS[:offset]
        schedule.extend(
            (config.config_id, "measured", iteration) for config in rotated
        )
    return schedule


def validate_sample_prefix(
    samples: list[dict[str, Any]], *, require_complete: bool = False
) -> dict[str, Any]:
    schedule = sample_schedule()
    if len(samples) > len(schedule):
        raise RuntimeError("CPU tuning sample prefix is longer than the fixed schedule")
    expected = schedule if require_complete else schedule[: len(samples)]
    actual = [
        (sample.get("config_id"), sample.get("sample_kind"), sample.get("iteration"))
        for sample in samples
    ]
    if actual != expected:
        raise RuntimeError(
            "CPU tuning samples are not an exact deterministically rotated prefix"
        )
    nonces = [sample.get("nonce") for sample in samples]
    if any(not isinstance(nonce, str) or not nonce for nonce in nonces):
        raise RuntimeError("CPU tuning sample nonce is absent")
    if len(set(nonces)) != len(nonces):
        raise RuntimeError("CPU tuning sample nonce was reused")
    signatures_by_world_size: dict[str, dict[str, Any]] = {}
    for sample in samples:
        config = _config_by_id(str(sample.get("config_id")))
        signature = sample.get("validation", {}).get("aggregate_counter_signature")
        if not isinstance(signature, dict) or not isinstance(signature.get("record"), dict):
            raise RuntimeError("CPU tuning aggregate counter signature is absent")
        record = signature["record"]
        if (
            record.get("world_size") != config.ranks
            or signature.get("sha256") != canonical_sha256(record)
        ):
            raise RuntimeError("CPU tuning aggregate counter signature is invalid")
        previous = signatures_by_world_size.setdefault(str(config.ranks), signature)
        if signature != previous:
            raise RuntimeError(
                f"CPU tuning aggregate counters changed for world size {config.ranks}"
            )
    return {
        "pass": True,
        "complete": len(samples) == len(schedule),
        "sample_count": len(samples),
        "expected": [list(row) for row in expected],
        "actual": [list(row) for row in actual],
        "aggregate_counter_signatures_by_world_size": signatures_by_world_size,
    }


def validate_sample_matrix(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) != len(sample_schedule()):
        raise RuntimeError("CPU tuning sample matrix is incomplete")
    return validate_sample_prefix(samples, require_complete=True)


def _qualification_fontconfig(
    receipt: dict[str, Any],
    repo: pathlib.Path,
    runtime: dict[str, pathlib.Path],
    *,
    hash_cache: StatHashCache | None = None,
) -> pathlib.Path:
    records = receipt.get("configuration_files")
    record = records.get("qualification_fontconfig") if isinstance(records, dict) else None
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
    ):
        raise RuntimeError("receipt-bound qualification Fontconfig record is absent")
    recorded_path = _MPI_RUNNER._resolve_recorded_path(record["path"], repo).resolve()
    path = runtime.get("qualification_fontconfig")
    if not isinstance(path, pathlib.Path) or path.resolve() != recorded_path:
        raise RuntimeError("runtime/receipt qualification Fontconfig paths disagree")
    path = path.resolve()
    size = record.get("size_bytes")
    if (
        not path.is_file()
        or (
            hash_cache.digest(path)
            if hash_cache is not None
            else sha256_file(path)
        )
        != record["sha256"]
        or (
            size is not None
            and (
                isinstance(size, bool)
                or not isinstance(size, int)
                or path.stat().st_size != size
            )
        )
    ):
        raise RuntimeError("receipt-bound qualification Fontconfig file changed")
    return path


def _child_environment(
    runtime: dict[str, pathlib.Path],
    *,
    config: CpuConfig,
    output: pathlib.Path,
    run_id: str,
    sample_kind: str,
    iteration: int,
    nonce: str,
    prepare_xdg_cache: bool,
) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise RuntimeError("CPU tuning run ID is invalid")
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise RuntimeError("CPU tuning sample nonce is invalid")
    if (
        sample_kind not in {"warmup", "measured"}
        or isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration < 0
    ):
        raise RuntimeError("CPU tuning sample cache identity is invalid")
    output = _trusted_output_path(output, create=False)

    def runtime_directory(*parts: str, label: str) -> pathlib.Path:
        directory = _trusted_output_path(output.joinpath(*parts), create=True)
        return _require_within_output(directory, output, label=label)

    environment_prefix = runtime["python"].parent.parent
    temporary = runtime_directory("tmp", label="CPU tuning temporary directory")
    matplotlib = runtime_directory(
        "matplotlib", label="CPU tuning Matplotlib directory"
    )
    pycache = runtime_directory(
        "pycache", run_id, label="CPU tuning Python cache directory"
    )
    cache_parent = runtime_directory(
        "cache", run_id, label="CPU tuning XDG cache parent"
    )
    cache_name = f"{config.config_id}-{sample_kind}-{iteration}-{nonce}"
    cache_candidate = cache_parent / cache_name
    if prepare_xdg_cache:
        if cache_candidate.exists() or cache_candidate.is_symlink():
            _assert_no_symlink_components(
                cache_candidate,
                require_leaf=True,
                label="CPU tuning XDG cache",
            )
            raise RuntimeError("CPU tuning per-sample XDG cache is not fresh")
        xdg_cache = runtime_directory(
            "cache", run_id, cache_name, label="CPU tuning XDG cache"
        )
    else:
        xdg_cache = _require_within_output(
            _trusted_output_path(cache_candidate, create=False),
            output,
            label="CPU tuning XDG cache",
        )
    qualification_home_candidate = output / "home" / run_id
    qualification_home_exists = (
        qualification_home_candidate.exists()
        or qualification_home_candidate.is_symlink()
    )
    qualification_home = runtime_directory(
        "home", run_id, label="CPU tuning HOME"
    )
    openmpi_params = (
        pathlib.Path(__file__).resolve().parent.parent
        / "environment"
        / "openmpi-qualification-mca-params.conf"
    ).resolve()
    if any(path.is_file() for path in pycache.rglob("*")):
        raise RuntimeError("CPU tuning Python bytecode cache is contaminated")
    marker = qualification_home / ".gpmeep-empty-home"
    expected_marker = "gpmeep isolated CPU tuning home\n"
    if qualification_home_exists:
        if (
            not qualification_home.is_dir()
            or sorted(path.name for path in qualification_home.iterdir()) != [marker.name]
            or marker.read_text(encoding="utf-8") != expected_marker
        ):
            raise RuntimeError("isolated CPU tuning HOME is contaminated")
    else:
        marker.write_text(expected_marker, encoding="utf-8")
    if not openmpi_params.is_file():
        raise RuntimeError("fixed Open MPI qualification parameter file is absent")
    fontconfig_file = runtime.get("qualification_fontconfig")
    if not isinstance(fontconfig_file, pathlib.Path):
        raise RuntimeError("receipt runtime qualification Fontconfig path is absent")
    fontconfig_file = fontconfig_file.resolve()
    if not fontconfig_file.is_file():
        raise RuntimeError("receipt-bound qualification Fontconfig file is absent")
    for override_name in (
        "openmpi-mca-params-override.conf",
        "pmix-mca-params-override.conf",
        "prte-mca-params-override.conf",
    ):
        if (environment_prefix / "etc" / override_name).exists():
            raise RuntimeError(f"MPI override parameter file is not allowed: {override_name}")
    environment = {
        "PYTHONPATH": os.pathsep.join(
            (str(pathlib.Path(__file__).resolve().parent), str(runtime["package_parent"]))
        ),
        "PATH": os.pathsep.join((str(environment_prefix / "bin"), "/usr/bin", "/bin")),
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
        "OMP_NUM_THREADS": str(config.omp_threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "close",
        "OMPI_MCA_mca_base_param_files": str(openmpi_params),
        "OMPI_MCA_mca_base_component_path": str(environment_prefix / "lib" / "openmpi"),
        "PMIX_MCA_mca_base_component_path": str(environment_prefix / "lib" / "pmix"),
        "PMIX_MCA_mca_base_param_files": str(openmpi_params),
        "PRTE_MCA_mca_base_param_files": str(openmpi_params),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(matplotlib),
        "CUDA_CACHE_DISABLE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "FONTCONFIG_FILE": str(fontconfig_file),
        "MEEP_GPU_BACKEND": "cpu",
        "MEEP_GPU_MPI_TRANSPORT": "auto",
        "MEEP_MATERIAL_GRADIENT_PATH": PROFILE["material_gradient_path"],
        "MEEP_MATERIAL_GRADIENT_STATS": "1",
    }
    if any(not isinstance(name, str) or not isinstance(value, str) for name, value in environment.items()):
        raise RuntimeError("fixed CPU child environment is malformed")
    return environment


def producer_arguments(
    *,
    repo: pathlib.Path,
    runtime: dict[str, pathlib.Path],
    result_path: pathlib.Path,
    receipt_id: str,
    snapshot_sha256: str,
    producer_sha256: str,
    nonce: str,
    run_id: str,
    sample_kind: str,
    iteration: int,
) -> list[str]:
    return [
        str(runtime["python"]),
        str((repo / "scripts" / "benchmark-adjoint.py").resolve()),
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
        PRODUCER_QUALIFICATION_PROFILE_ID,
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
        "cpu-parallel",
        "--sample-iteration",
        str(iteration),
    ]


def launch_arguments(
    *,
    runtime: dict[str, pathlib.Path],
    config: CpuConfig,
    producer_argv: list[str],
) -> list[str]:
    return [
        str(runtime["mpiexec"]),
        "--nooversubscribe",
        "--report-bindings",
        *binding_arguments(config),
        "-n",
        str(config.ranks),
        *producer_argv,
    ]


class CommandTimeout(RuntimeError):
    def __init__(self, message: str, stdout: str, stderr: str):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class ResumableRunFailure(RuntimeError):
    """A cleaned-up interruption that may safely resume the fixed schedule."""


class CorrectnessGateFailure(RuntimeError):
    """A receipt-invalidating objective/full-gradient qualification failure."""

    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


class NonResumableRun(RuntimeError):
    """Raised when an immutable terminal failure forbids resume."""


def _signal_process_group(process: subprocess.Popen[str], signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        # A session's process group can outlive its mpiexec leader, so signal
        # the recorded PGID even when poll() says the leader exited.  ESRCH is
        # the only safe indication that there is no remaining group to kill;
        # communicate() below still performs the authoritative leader reap.
        pass


def _terminate_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Terminate the isolated process group and synchronously reap its leader."""

    _signal_process_group(process, signal.SIGTERM)
    try:
        return process.communicate(timeout=PROFILE["timeout_kill_grace_seconds"])
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # A second interrupt must not abandon mpiexec or its descendants.  Move
        # directly to SIGKILL, then defer the interrupt until cleanup is done.
        _signal_process_group(process, signal.SIGKILL)
        while True:
            try:
                return process.communicate()
            except KeyboardInterrupt:
                continue


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
        stdout, stderr = _terminate_process_group(process)
        raise CommandTimeout(
            f"process group exceeded the fixed {timeout_seconds}s timeout",
            stdout or "",
            stderr or "",
        ) from error
    except BaseException:
        # KeyboardInterrupt/SystemExit are classified as safely resumable only
        # after the new-session process group has been terminated and reaped.
        _terminate_process_group(process)
        raise
    return process.returncode, stdout, stderr, time.perf_counter() - started


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


def validate_qualification_binding(
    result: dict[str, Any],
    *,
    nonce: str,
    run_id: str,
    receipt_id: str,
    snapshot_sha256: str,
    producer_sha256: str,
    sample_kind: str,
    iteration: int,
    label: str,
) -> dict[str, Any]:
    expected = {
        "mode": "qualification",
        "profile": PRODUCER_QUALIFICATION_PROFILE_ID,
        "run_nonce": nonce,
        "run_id": run_id,
        "build_receipt_id": receipt_id,
        "source_snapshot_sha256": snapshot_sha256,
        "producer_sha256": producer_sha256,
        "sample_kind": sample_kind,
        "sample_lane": "cpu-parallel",
        "sample_iteration": iteration,
    }
    if result.get("qualification") != expected:
        raise RuntimeError(f"{label} qualification binding is stale or forged")
    producer = result.get("producer")
    if not isinstance(producer, dict) or producer.get("sha256") != producer_sha256:
        raise RuntimeError(f"{label} producer digest does not match its binding")
    path = pathlib.Path(str(producer.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != producer_sha256:
        raise RuntimeError(f"{label} producer changed after measurement")
    for timestamp in ("started_at_utc", "completed_at_utc"):
        value = producer.get(timestamp)
        if not isinstance(value, str) or not value.endswith("Z"):
            raise RuntimeError(f"{label} {timestamp} is absent")
    return expected


def _validate_process_environments(
    result: dict[str, Any], launched: dict[str, str], *, label: str
) -> dict[str, Any]:
    producer_environment = result.get("producer", {}).get("environment")
    ranks = result.get("distributed", {}).get("ranks")
    if not isinstance(producer_environment, dict) or not isinstance(ranks, list):
        raise RuntimeError(f"{label} actual process environment evidence is absent")
    digests: list[str] = []
    for rank_index, rank in enumerate(ranks):
        environment = rank.get("process_environment") if isinstance(rank, dict) else None
        if not isinstance(environment, dict) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in environment.items()
        ):
            raise RuntimeError(f"{label}.rank{rank_index} process environment is invalid")
        for name, value in launched.items():
            if name in LAUNCHER_CONSUMED_ENVIRONMENT_KEYS:
                if name in environment and environment[name] != value:
                    raise RuntimeError(f"{label}.rank{rank_index} changed {name}")
            elif environment.get(name) != value:
                raise RuntimeError(f"{label}.rank{rank_index} changed fixed launch environment")
        if environment.get("PMIX_MCA_mca_base_param_files") != launched.get(
            "OMPI_MCA_mca_base_param_files"
        ):
            raise RuntimeError(f"{label}.rank{rank_index} changed fixed PMIx parameters")
        forbidden = sorted(FORBIDDEN_INFLUENCE_VARIABLES & set(environment))
        if forbidden:
            raise RuntimeError(
                f"{label}.rank{rank_index} inherited forbidden controls: "
                + ", ".join(forbidden)
            )
        unexpected = sorted(
            name
            for name in environment
            if name not in launched
            and name not in ALLOWED_LAUNCHER_INJECTED_INFLUENCE_VARIABLES
            and name.startswith(INFLUENTIAL_ENVIRONMENT_PREFIXES)
        )
        if unexpected:
            raise RuntimeError(
                f"{label}.rank{rank_index} gained unlaunched controls: "
                + ", ".join(unexpected)
            )
        digests.append(canonical_sha256(environment))
    if not ranks or producer_environment != ranks[0].get("process_environment"):
        raise RuntimeError(f"{label} producer/rank-zero environments disagree")
    pycache = pathlib.Path(launched["PYTHONPYCACHEPREFIX"])
    if any(path.is_file() for path in pycache.rglob("*")):
        raise RuntimeError(f"{label} Python bytecode isolation is contaminated")
    home = pathlib.Path(launched["HOME"])
    marker = home / ".gpmeep-empty-home"
    if (
        sorted(path.name for path in home.iterdir()) != [marker.name]
        or marker.read_text(encoding="utf-8") != "gpmeep isolated CPU tuning home\n"
    ):
        raise RuntimeError(f"{label} isolated CPU tuning HOME was modified")
    return {"pass": True, "rank_environment_sha256": digests}


def _counter(statistics_record: Any, group: str, name: str, label: str) -> int:
    value = (
        statistics_record.get(group, {}).get(name)
        if isinstance(statistics_record, dict)
        else None
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} counter {group}.{name} is invalid")
    return value


def _validate_statistics_record(record: Any, *, label: str) -> dict[str, int]:
    if not isinstance(record, dict) or set(record) != set(STATISTICS_COUNTER_SCHEMA):
        raise RuntimeError(f"{label} statistics schema is incomplete")
    flattened: dict[str, int] = {}
    for group, fields in STATISTICS_COUNTER_SCHEMA.items():
        values = record.get(group)
        if not isinstance(values, dict) or set(values) != set(fields):
            raise RuntimeError(f"{label}.{group} statistics schema is incomplete")
        for field in fields:
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"{label}.{group}.{field} is invalid")
            if (
                field.startswith("cuda_")
                or field
                in (
                    "device_to_host_bytes",
                    "host_to_device_bytes",
                    "ldos_full_field_device_to_host_bytes_avoided",
                )
                or group
                in (
                    "multi_gpu",
                    "mpi_completion",
                    "resident",
                    "runtime",
                    "tile_coalescing",
                )
                or field.startswith("cpu_polarization_")
            ) and value != 0:
                raise RuntimeError(f"{label}.{group}.{field} contains GPU/dispersive work")
            flattened[f"{group}.{field}"] = value
    for group, (checks_field, disabled_field) in CPU_DISABLED_OVERLAP_COUNTERS.items():
        values = record[group]
        if values[checks_field] != values[disabled_field] or any(
            value != 0
            for field, value in values.items()
            if field not in (checks_field, disabled_field)
        ):
            raise RuntimeError(f"{label}.{group} contains GPU overlap work")
    return flattened


def _validate_phase_counter_view(
    counters: Any,
    statistics: dict[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, dict[str, int]]]:
    if not isinstance(counters, dict) or set(counters) != {"cpu", "cuda"}:
        raise RuntimeError(f"{label} backend counter schema is incomplete")
    normalized: dict[str, dict[str, dict[str, int]]] = {}
    for backend in ("cpu", "cuda"):
        view = counters.get(backend)
        if not isinstance(view, dict) or set(view) != set(PHASE_COUNTER_STATISTICS_FIELDS):
            raise RuntimeError(f"{label}.{backend} backend counter schema is incomplete")
        normalized[backend] = {}
        for name, (group, stem) in PHASE_COUNTER_STATISTICS_FIELDS.items():
            values = view.get(name)
            if not isinstance(values, dict) or set(values) != {"calls", "points"}:
                raise RuntimeError(f"{label}.{backend}.{name} counter schema is incomplete")
            calls = values.get("calls")
            points = values.get("points")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (calls, points)
            ):
                raise RuntimeError(f"{label}.{backend}.{name} calls/points are invalid")
            if (calls == 0) != (points == 0):
                raise RuntimeError(f"{label}.{backend}.{name} calls/points disagree")
            expected_calls = statistics[group][f"{backend}_{stem}_calls"]
            expected_points = statistics[group][f"{backend}_{stem}_points"]
            if calls != expected_calls or points != expected_points:
                raise RuntimeError(f"{label}.{backend}.{name} view/statistics disagree")
            normalized[backend][name] = {"calls": calls, "points": points}
    return normalized


def _validate_cpu_counter_accounting(rank: dict[str, Any], *, label: str) -> dict[str, Any]:
    final_statistics = rank.get("final_statistics")
    final_flat = _validate_statistics_record(final_statistics, label=f"{label}.final")
    phases = rank.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise RuntimeError(f"{label} phase counter accounting is incomplete")
    phase_flat: dict[str, dict[str, int]] = {}
    phase_views: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for phase_name in PHASES:
        phase = phases.get(phase_name)
        if not isinstance(phase, dict):
            raise RuntimeError(f"{label}.{phase_name} phase record is invalid")
        statistics = phase.get("statistics_delta")
        phase_flat[phase_name] = _validate_statistics_record(
            statistics, label=f"{label}.{phase_name}"
        )
        phase_views[phase_name] = _validate_phase_counter_view(
            phase.get("backend_counters"),
            statistics,
            label=f"{label}.{phase_name}",
        )
        if phase_name in ("forward", "adjoint"):
            expected_timesteps = PROFILE[f"{phase_name}_timesteps"]
            expected_end_time = PROFILE[f"{phase_name}_end_time"]
            if (
                phase.get("timesteps") != expected_timesteps
                or phase.get("end_time") != expected_end_time
            ):
                raise RuntimeError(
                    f"{label}.{phase_name} differs from the fixed FDTD workload"
                )
            expected_overlap_checks = {
                "boundary_eh_overlap.boundary_eh_overlap_checks": 2
                * expected_timesteps,
                "boundary_eh_overlap.boundary_eh_overlap_skipped_disabled": 2
                * expected_timesteps,
                "halo_curl_overlap.halo_curl_overlap_checks": expected_timesteps,
                "halo_curl_overlap.halo_curl_overlap_skipped_disabled": expected_timesteps,
            }
            if any(
                phase_flat[phase_name][path] != expected
                for path, expected in expected_overlap_checks.items()
            ):
                raise RuntimeError(
                    f"{label}.{phase_name} overlap checks do not match fixed timesteps"
                )
        elif any(phase_flat[phase_name].values()):
            raise RuntimeError(f"{label}.{phase_name} unexpectedly recorded FDTD work")
    summed = {
        path: sum(phase_flat[phase][path] for phase in PHASES) for path in final_flat
    }
    if summed != final_flat:
        raise RuntimeError(f"{label} final statistics differ from exact phase sum")
    return {"final": final_flat, "phases": phase_flat, "views": phase_views}


def _validate_cpu_backend_and_counters(rank: dict[str, Any], *, label: str) -> dict[str, Any]:
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
    accounting = _validate_cpu_counter_accounting(rank, label=label)
    statistics_record = rank.get("final_statistics")
    cpu = {name: _counter(statistics_record, group, name, label) for group, name in CPU_COUNTERS}
    cuda = {name: _counter(statistics_record, group, name, label) for group, name in CUDA_COUNTERS}
    if any(cpu[name] <= 0 for name in FINAL_REQUIRED_PER_RANK_CPU_COUNTERS) or any(
        value != 0 for value in cuda.values()
    ):
        raise RuntimeError(f"{label} CPU coverage is absent or contains CUDA work")
    phases = rank["phases"]
    phase_cpu: dict[str, dict[str, dict[str, int]]] = {}
    for phase_name in ("forward", "adjoint"):
        counters = accounting["views"][phase_name]
        cpu_phase = counters["cpu"]
        cuda_phase = counters["cuda"]
        for name in PHASE_REQUIRED_PER_RANK_CPU_COUNTERS:
            values = cpu_phase.get(name)
            if (
                not isinstance(values, dict)
                or not isinstance(values.get("calls"), int)
                or not isinstance(values.get("points"), int)
                or values["calls"] <= 0
                or values["points"] <= 0
            ):
                raise RuntimeError(f"{label}.{phase_name} CPU {name} coverage is absent")
        sparse_phase: dict[str, dict[str, int]] = {}
        for name in PHASE_REQUIRED_AGGREGATE_CPU_COUNTERS:
            values = cpu_phase.get(name)
            if not isinstance(values, dict) or not all(
                isinstance(values.get(field), int) and values[field] >= 0
                for field in ("calls", "points")
            ):
                raise RuntimeError(f"{label}.{phase_name} CPU {name} evidence is invalid")
            if (values["calls"] == 0) != (values["points"] == 0):
                raise RuntimeError(f"{label}.{phase_name} CPU {name} calls/points disagree")
            sparse_phase[name] = {
                "calls": values["calls"],
                "points": values["points"],
            }
        phase_cpu[phase_name] = sparse_phase
        for name, values in cuda_phase.items():
            if (
                not isinstance(values, dict)
                or values.get("calls") != 0
                or values.get("points") != 0
            ):
                raise RuntimeError(f"{label}.{phase_name} contains CUDA {name} work")
    return {
        "cpu_calls": cpu,
        "cuda_calls": cuda,
        "phase_cpu": phase_cpu,
        "accounting": accounting,
    }


def validate_affinity(
    result: dict[str, Any], config: CpuConfig, topology: dict[str, Any], *, label: str
) -> dict[str, Any]:
    ranks = result.get("distributed", {}).get("ranks")
    if not isinstance(ranks, list) or len(ranks) != config.ranks:
        raise RuntimeError(f"{label} rank matrix is absent")
    logical_rows = topology.get("logical_cpus")
    cores = topology.get("cores")
    if not isinstance(logical_rows, list) or not isinstance(cores, list):
        raise RuntimeError(f"{label} topology mapping is absent")
    identity = {
        row["cpu"]: (row["socket"], row["node"], row["core"])
        for row in logical_rows
        if isinstance(row, dict)
    }
    core_cpus = {
        (core["socket"], core["node"], core["core"]): set(core["logical_cpus"])
        for core in cores
        if isinstance(core, dict)
    }
    if (
        len(identity) != topology.get("logical_cpu_count")
        or len(core_cpus) != topology.get("physical_core_count")
    ):
        raise RuntimeError(f"{label} topology is incomplete")
    hostnames: set[str] = set()
    rank_records: list[dict[str, Any]] = []
    used_logical: set[int] = set()
    used_cores: set[tuple[int, int]] = set()
    for rank_index, rank in enumerate(ranks):
        affinity = rank.get("cpu_affinity") if isinstance(rank, dict) else None
        completion_affinity = (
            rank.get("completion_thread_cpu_affinity")
            if isinstance(rank, dict)
            else None
        )
        thread_affinities = (
            rank.get("process_thread_cpu_affinities")
            if isinstance(rank, dict)
            else None
        )
        hostname = rank.get("hostname") if isinstance(rank, dict) else None
        if (
            not isinstance(affinity, list)
            or not affinity
            or any(isinstance(cpu, bool) or not isinstance(cpu, int) for cpu in affinity)
            or affinity != sorted(set(affinity))
            or any(cpu not in identity for cpu in affinity)
            or not isinstance(hostname, str)
            or not hostname
        ):
            raise RuntimeError(f"{label}.rank{rank_index} affinity/hostname is invalid")
        if (
            not isinstance(completion_affinity, list)
            or not completion_affinity
            or completion_affinity != sorted(set(completion_affinity))
            or not isinstance(thread_affinities, dict)
            or set(thread_affinities)
            != {"thread_count", "logical_cpu_union", "threads"}
        ):
            raise RuntimeError(f"{label}.rank{rank_index} completion/thread affinity is invalid")
        thread_records = thread_affinities["threads"]
        if (
            isinstance(thread_affinities["thread_count"], bool)
            or not isinstance(thread_affinities["thread_count"], int)
            or thread_affinities["thread_count"] <= 0
            or not isinstance(thread_records, list)
            or thread_affinities["thread_count"] != len(thread_records)
        ):
            raise RuntimeError(f"{label}.rank{rank_index} thread-affinity count is invalid")
        thread_ids: set[int] = set()
        thread_union: set[int] = set()
        observed_thread_placements: list[set[int]] = []
        for thread in thread_records:
            thread_id = thread.get("thread_id") if isinstance(thread, dict) else None
            logical_cpus = thread.get("logical_cpus") if isinstance(thread, dict) else None
            if (
                isinstance(thread_id, bool)
                or not isinstance(thread_id, int)
                or thread_id <= 0
                or thread_id in thread_ids
                or not isinstance(logical_cpus, list)
                or not logical_cpus
                or logical_cpus != sorted(set(logical_cpus))
                or any(cpu not in identity for cpu in logical_cpus)
            ):
                raise RuntimeError(f"{label}.rank{rank_index} thread-affinity row is invalid")
            thread_ids.add(thread_id)
            placement = set(logical_cpus)
            observed_thread_placements.append(placement)
            thread_union.update(placement)
        if (
            thread_affinities["logical_cpu_union"] != sorted(thread_union)
            or not set(affinity).issubset(thread_union)
            or not set(completion_affinity).issubset(thread_union)
        ):
            raise RuntimeError(f"{label}.rank{rank_index} thread-affinity union is incomplete")
        rank_cores = {identity[cpu] for cpu in thread_union}
        expected_rank_cores = {
            tuple(core) for core in config.rank_core_plan[rank_index]
        }
        expected_rank_logical = set(config.rank_logical_plan[rank_index])
        expected_rank_node = config.rank_numa_plan[rank_index]
        if (
            rank_cores != expected_rank_cores
            or thread_union != expected_rank_logical
            or {core[1] for core in rank_cores} != {expected_rank_node}
        ):
            raise RuntimeError(
                f"{label}.rank{rank_index} differs from the explicit core/NUMA plan"
            )
        if config.binding == "core":
            expected_logical = set().union(*(core_cpus[core] for core in rank_cores))
            if len(rank_cores) != config.omp_threads or thread_union != expected_logical:
                raise RuntimeError(f"{label}.rank{rank_index} core binding is incomplete")
            expected_worker_placements = [core_cpus[core] for core in sorted(rank_cores)]
            unexpected_thread_placements = [
                sorted(observed)
                for observed in observed_thread_placements
                if not any(observed == expected for expected in expected_worker_placements)
            ]
            missing_worker_placements = [
                sorted(expected)
                for expected in expected_worker_placements
                if not any(observed == expected for observed in observed_thread_placements)
            ]
            if missing_worker_placements or unexpected_thread_placements:
                raise RuntimeError(
                    f"{label}.rank{rank_index} thread rows are not exact OpenMP "
                    "physical-core places"
                )
            if used_cores & rank_cores:
                raise RuntimeError(f"{label} ranks overlap physical cores")
        else:
            if len(thread_union) != config.omp_threads:
                raise RuntimeError(f"{label}.rank{rank_index} hwthread binding is incomplete")
            if any(
                observed != thread_union for observed in observed_thread_placements
            ):
                raise RuntimeError(
                    f"{label}.rank{rank_index} thread rows are not the exact bound hwthread"
                )
        if used_logical & thread_union:
            raise RuntimeError(f"{label} ranks overlap logical CPUs")
        used_logical.update(thread_union)
        used_cores.update(rank_cores)
        hostnames.add(hostname)
        rank_records.append(
            {
                "world_rank": rank_index,
                "hostname": hostname,
                "launch_thread_logical_cpus": affinity,
                "completion_thread_logical_cpus": completion_affinity,
                "thread_count": thread_affinities["thread_count"],
                "thread_logical_cpu_union": sorted(thread_union),
                "exact_worker_placements": [
                    sorted(placement)
                    for placement in observed_thread_placements
                    if placement in core_cpus.values()
                    or (config.binding == "hwthread" and placement == thread_union)
                ],
                "physical_cores": [list(core) for core in sorted(rank_cores)],
            }
        )
    if len(hostnames) != 1:
        raise RuntimeError(f"{label} unexpectedly crossed hosts")
    if config.full_physical and used_cores != set(core_cpus):
        raise RuntimeError(f"{label} full-physical endpoint did not cover every core")
    if config.binding == "hwthread" and used_logical != set(identity):
        raise RuntimeError(f"{label} SMT endpoint did not cover every logical CPU")
    producer_affinity = result.get("producer", {}).get("cpu_affinity")
    if (
        producer_affinity != ranks[0].get("cpu_affinity")
        or result.get("producer", {}).get("completion_thread_cpu_affinity")
        != ranks[0].get("completion_thread_cpu_affinity")
        or result.get("producer", {}).get("process_thread_cpu_affinities")
        != ranks[0].get("process_thread_cpu_affinities")
    ):
        raise RuntimeError(f"{label} producer/rank-zero affinities disagree")
    return {
        "pass": True,
        "binding": config.binding,
        "hostname": next(iter(hostnames)),
        "rank_affinity": rank_records,
        "used_logical_cpus": sorted(used_logical),
        "used_physical_cores": [list(core) for core in sorted(used_cores)],
        "explicit_placement_plan_match": True,
    }


def _fixed_workload_gate(result: dict[str, Any], *, label: str) -> None:
    workload = result.get("workload")
    if not isinstance(workload, dict):
        raise RuntimeError(f"{label} workload evidence is absent")
    for name in WORKLOAD_KEYS:
        expected = PROFILE[name]
        if workload.get(name) != expected:
            raise RuntimeError(f"{label} changed fixed workload {name}")
    for phase_name in ("forward", "adjoint"):
        phase = workload.get("phases", {}).get(phase_name, {})
        for suffix in ("timesteps", "end_time"):
            if phase.get(suffix) != PROFILE[f"{phase_name}_{suffix}"]:
                raise RuntimeError(f"{label} changed fixed {phase_name} {suffix}")


def validate_cpu_record(
    result: dict[str, Any],
    config: CpuConfig,
    topology: dict[str, Any],
    *,
    label: str,
    runtime_validator: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if result.get("schema_version") != 4:
        raise RuntimeError(f"{label} benchmark schema is unsupported")
    _COMPARATOR.validate_gradient_summaries(result, label)
    gradient = validate_gradient_record(result)
    if not gradient or max(abs(value) for value in gradient) <= 1.0e-12:
        raise RuntimeError(f"{label} gradient is empty or all zero")
    objective = result.get("result", {}).get("objective")
    if isinstance(objective, bool) or not isinstance(objective, (int, float)) or not math.isfinite(float(objective)):
        raise RuntimeError(f"{label} objective is invalid")
    _fixed_workload_gate(result, label=label)
    distributed = result.get("distributed")
    if not isinstance(distributed, dict) or distributed.get("world_size") != config.ranks:
        raise RuntimeError(f"{label} world size differs from requested ranks")
    if distributed.get("timing_semantics") != "barrier-delimited maximum rank elapsed per phase":
        raise RuntimeError(f"{label} timing semantics are invalid")
    gate = distributed.get("gate")
    if not isinstance(gate, dict) or any(
        gate.get(name) is not True
        for name in ("pass", "objective", "gradient", "strict_cuda_coverage")
    ):
        raise RuntimeError(f"{label} producer distributed gate did not pass")
    ranks = distributed.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != config.ranks:
        raise RuntimeError(f"{label} rank evidence is incomplete")
    if [rank.get("world_rank") for rank in ranks] != list(range(config.ranks)):
        raise RuntimeError(f"{label} rank evidence is duplicate or out of order")
    validator = runtime_validator or _COMPARATOR.validate_runtime_closure
    runtime = validator(result, label)
    rank_runtimes: list[dict[str, Any]] = []
    coverages: list[dict[str, Any]] = []
    for rank_index, rank in enumerate(ranks):
        rank_label = f"{label}.rank{rank_index}"
        rank_runtime = validator(rank, rank_label)
        if rank_index == 0:
            if rank_runtime != runtime:
                raise RuntimeError(f"{rank_label} runtime differs from rank zero")
        elif _COMPARATOR.runtime_environment_independent_projection(
            rank_runtime
        ) != _COMPARATOR.runtime_environment_independent_projection(runtime):
            raise RuntimeError(f"{rank_label} loaded different controlled runtimes")
        rank_runtimes.append(rank_runtime)
        rank_gradient = rank.get("gradient")
        if (
            not isinstance(rank_gradient, list)
            or canonical_float64_sha256(rank_gradient)
            != result.get("result", {}).get("gradient_sha256")
            or float(rank.get("objective")) != float(objective)
        ):
            raise RuntimeError(f"{rank_label} objective/full gradient differs")
        coverages.append(_validate_cpu_backend_and_counters(rank, label=rank_label))
    for name in FINAL_REQUIRED_AGGREGATE_CPU_COUNTERS:
        if sum(coverage["cpu_calls"][name] for coverage in coverages) <= 0:
            raise RuntimeError(f"{label} has no aggregate CPU {name} coverage")
    for phase_name in ("forward", "adjoint"):
        for name in PHASE_REQUIRED_AGGREGATE_CPU_COUNTERS:
            calls = sum(
                coverage["phase_cpu"][phase_name][name]["calls"]
                for coverage in coverages
            )
            points = sum(
                coverage["phase_cpu"][phase_name][name]["points"]
                for coverage in coverages
            )
            if calls <= 0 or points <= 0:
                raise RuntimeError(
                    f"{label}.{phase_name} has no aggregate CPU {name} coverage"
                )
    workload = result.get("workload")
    if not isinstance(workload, dict) or workload.get("final_statistics") != ranks[0].get(
        "final_statistics"
    ):
        raise RuntimeError(f"{label} top-level statistics are not rank-zero statistics")
    top_phases = workload.get("phases")
    if not isinstance(top_phases, dict):
        raise RuntimeError(f"{label} top-level phase evidence is absent")
    for phase_name in PHASES:
        top_phase = top_phases.get(phase_name)
        rank_zero_phase = ranks[0].get("phases", {}).get(phase_name)
        if not isinstance(top_phase, dict) or not isinstance(rank_zero_phase, dict):
            raise RuntimeError(f"{label}.{phase_name} top/rank-zero phase evidence is absent")
        for field in ("statistics_delta", "backend_counters"):
            if top_phase.get(field) != rank_zero_phase.get(field):
                raise RuntimeError(
                    f"{label}.{phase_name} top/rank-zero {field} differs"
                )
    aggregate_final = {
        path: sum(coverage["accounting"]["final"][path] for coverage in coverages)
        for path in coverages[0]["accounting"]["final"]
    }
    aggregate_phases = {
        phase_name: {
            path: sum(
                coverage["accounting"]["phases"][phase_name][path]
                for coverage in coverages
            )
            for path in coverages[0]["accounting"]["phases"][phase_name]
        }
        for phase_name in PHASES
    }
    aggregate_counter_signature_record = {
        "world_size": config.ranks,
        "final": aggregate_final,
        "phases": aggregate_phases,
    }
    aggregate_counter_signature = {
        "record": aggregate_counter_signature_record,
        "sha256": canonical_sha256(aggregate_counter_signature_record),
    }
    affinity = validate_affinity(result, config, topology, label=label)
    timing = result.get("timing")
    if not isinstance(timing, dict):
        raise RuntimeError(f"{label} timing evidence is absent")
    rank_wall = [
        _finite_positive(rank.get("timing", {}).get("workload_wall_seconds"), f"{label}.rank{index}.workload_wall")
        for index, rank in enumerate(ranks)
    ]
    workload_wall = _finite_positive(timing.get("workload_wall_seconds"), f"{label}.workload_wall")
    if workload_wall != max(rank_wall):
        raise RuntimeError(f"{label} workload wall time is not exact max-rank time")
    rank_zero_wall = _finite_positive(
        timing.get("rank_local_workload_wall_seconds"), f"{label}.rank_local_workload_wall"
    )
    if rank_zero_wall != rank_wall[0]:
        raise RuntimeError(f"{label} rank-zero workload wall time is inconsistent")
    phase_maxima: dict[str, float] = {}
    rank_phase_seconds: dict[str, list[float]] = {}
    for phase in PHASES:
        field = f"{phase}_seconds"
        rank_values: list[float] = []
        for index, rank in enumerate(ranks):
            rank_value = _finite_positive(
                rank.get("timing", {}).get(field),
                f"{label}.rank{index}.{field}",
            )
            phase_value = _finite_positive(
                rank.get("phases", {}).get(phase, {}).get("elapsed_seconds"),
                f"{label}.rank{index}.phases.{phase}.elapsed_seconds",
            )
            if rank_value != phase_value:
                raise RuntimeError(
                    f"{label}.rank{index} {phase} timing differs from phase evidence"
                )
            rank_values.append(rank_value)
        maximum = max(rank_values)
        recorded = _finite_positive(timing.get(field), f"{label}.{field}")
        if recorded != maximum:
            raise RuntimeError(f"{label} {field} is not exact max-rank time")
        top_phase = _finite_positive(
            result.get("workload", {})
            .get("phases", {})
            .get(phase, {})
            .get("elapsed_seconds"),
            f"{label}.workload.phases.{phase}.elapsed_seconds",
        )
        if top_phase != recorded:
            raise RuntimeError(f"{label} top {phase} timing differs from phase evidence")
        phase_maxima[field] = maximum
        rank_phase_seconds[phase] = rank_values
    rank_elapsed: list[float] = []
    for index, rank in enumerate(ranks):
        rank_value = _finite_positive(
            rank.get("timing", {}).get("elapsed_seconds"),
            f"{label}.rank{index}.elapsed_seconds",
        )
        expected = sum(rank_phase_seconds[phase][index] for phase in PHASES)
        if not math.isclose(rank_value, expected, rel_tol=1e-14, abs_tol=1e-14):
            raise RuntimeError(f"{label}.rank{index} elapsed time differs from its phases")
        if rank_wall[index] < rank_value:
            raise RuntimeError(
                f"{label}.rank{index} workload wall time is shorter than phase elapsed time"
            )
        rank_elapsed.append(rank_value)
    elapsed = _finite_positive(timing.get("elapsed_seconds"), f"{label}.elapsed")
    if not math.isclose(elapsed, sum(phase_maxima.values()), rel_tol=1e-14, abs_tol=1e-14):
        raise RuntimeError(f"{label} elapsed time is not the sum of phase maxima")
    rank_zero_elapsed = _finite_positive(
        timing.get("rank_local_elapsed_seconds"),
        f"{label}.rank_local_elapsed_seconds",
    )
    if rank_zero_elapsed != rank_elapsed[0]:
        raise RuntimeError(f"{label} rank-zero elapsed time is inconsistent")
    top_backend = result.get("backend")
    if not isinstance(top_backend, dict) or top_backend != ranks[0].get("backend"):
        raise RuntimeError(f"{label} top/rank-zero backend evidence differs")
    return {
        "pass": True,
        "runtime": runtime,
        "rank_runtimes": rank_runtimes,
        "objective": float(objective),
        "gradient_sha256": result["result"]["gradient_sha256"],
        "coverage": coverages,
        "aggregate_counter_signature": aggregate_counter_signature,
        "affinity": affinity,
        "workload_wall_seconds": workload_wall,
        "rank_workload_wall_seconds": rank_wall,
        "elapsed_seconds": elapsed,
        "rank_elapsed_seconds": rank_elapsed,
        "phase_seconds": {
            phase: phase_maxima[f"{phase}_seconds"] for phase in PHASES
        },
    }


def compare_to_reference(
    reference: dict[str, Any], candidate: dict[str, Any], *, label: str
) -> dict[str, Any]:
    _fixed_workload_gate(reference, label="reference")
    _fixed_workload_gate(candidate, label=label)
    if any(
        reference.get("workload", {}).get(name) != candidate.get("workload", {}).get(name)
        for name in WORKLOAD_KEYS
    ):
        raise RuntimeError(f"{label} workload differs from reference")
    objective_reference = float(reference["result"]["objective"])
    objective_candidate = float(candidate["result"]["objective"])
    objective_error = abs(objective_reference - objective_candidate)
    gradient = compare_gradient_vectors(
        reference,
        candidate,
        atol=PROFILE["gradient_atol"],
        rtol=PROFILE["gradient_rtol"],
        expected_count=PROFILE["design_variables"],
    )
    result = {
        "pass": objective_error <= PROFILE["objective_atol"] and gradient["pass"],
        "objective": {
            "pass": objective_error <= PROFILE["objective_atol"],
            "reference": objective_reference,
            "candidate": objective_candidate,
            "absolute_error": objective_error,
            "atol": PROFILE["objective_atol"],
        },
        "gradient": gradient,
        "workload": {"pass": True, "keys": list(WORKLOAD_KEYS)},
    }
    if not result["pass"]:
        raise CorrectnessGateFailure(
            f"{label} objective/full-gradient correctness gate failed", result
        )
    return result


def _timing_summary(values: list[float], *, label: str) -> dict[str, Any]:
    checked = [_finite_positive(value, label) for value in values]
    if not checked:
        raise RuntimeError(f"{label} timing sample set is empty")
    mean = statistics.fmean(checked)
    standard_deviation = statistics.stdev(checked) if len(checked) > 1 else 0.0
    return {
        "samples_seconds": checked,
        "median_seconds": statistics.median(checked),
        "mean_seconds": mean,
        "standard_deviation_seconds": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean,
        "minimum_seconds": min(checked),
        "maximum_seconds": max(checked),
    }


def select_fastest(samples: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [sample for sample in samples if sample.get("sample_kind") == "measured"]
    by_config: dict[str, list[dict[str, Any]]] = {config.config_id: [] for config in CONFIGS}
    for sample in measured:
        config_id = sample.get("config_id")
        if config_id not in by_config:
            raise RuntimeError("measured sample has an unknown CPU configuration")
        by_config[config_id].append(sample)
    summaries: list[dict[str, Any]] = []
    eligible_summaries: list[dict[str, Any]] = []
    unstable_configs: list[str] = []
    for order, config in enumerate(CONFIGS):
        values = by_config[config.config_id]
        if len(values) != PROFILE["repeats"] or sorted(
            sample.get("iteration") for sample in values
        ) != list(range(PROFILE["repeats"])):
            raise RuntimeError(f"{config.config_id} measured repetition matrix is incomplete")
        timings = [
            _finite_positive(
                sample.get("validation", {}).get("workload_wall_seconds"),
                f"{config.config_id}.workload_wall",
            )
            for sample in values
        ]
        elapsed_timings = [
            _finite_positive(
                sample.get("validation", {}).get("elapsed_seconds"),
                f"{config.config_id}.elapsed_seconds",
            )
            for sample in values
        ]
        phase_timings = {
            phase: [
                _finite_positive(
                    sample.get("validation", {})
                    .get("phase_seconds", {})
                    .get(phase),
                    f"{config.config_id}.{phase}_seconds",
                )
                for sample in values
            ]
            for phase in PHASES
        }
        workload_statistics = _timing_summary(
            timings, label=f"{config.config_id}.workload_wall"
        )
        elapsed_statistics = _timing_summary(
            elapsed_timings, label=f"{config.config_id}.elapsed"
        )
        gate_names = (
            "validation",
            "correctness",
            "receipt_gate",
            "process_environment_gate",
            "material_gradient_gate",
        )
        failed_gates = sorted(
            {
                gate_name
                for sample in values
                for gate_name in gate_names
                if sample.get(gate_name, {}).get("pass") is not True
            }
        )
        if (
            workload_statistics["coefficient_of_variation"]
            > PROFILE["maximum_workload_cv"]
        ):
            failed_gates.append("workload_wall_cv")
        if (
            elapsed_statistics["coefficient_of_variation"]
            > PROFILE["maximum_elapsed_cv"]
        ):
            failed_gates.append("elapsed_cv")
        failed_gates = sorted(set(failed_gates))
        if any(name in failed_gates for name in ("workload_wall_cv", "elapsed_cv")):
            unstable_configs.append(config.config_id)
        selection_eligible = not failed_gates and config.lane == "physical"
        summary = {
            **asdict(config),
            "order": order,
            "eligible": not failed_gates,
            "selection_eligible": selection_eligible,
            "failed_gates": failed_gates,
            "workload_wall_seconds": timings,
            "workload_wall_median_seconds": statistics.median(timings),
            "workload_wall_min_seconds": min(timings),
            "workload_wall_max_seconds": max(timings),
            "workload_wall_statistics": workload_statistics,
            "elapsed_statistics": elapsed_statistics,
            "phase_statistics": {
                phase: _timing_summary(
                    phase_timings[phase], label=f"{config.config_id}.{phase}"
                )
                for phase in PHASES
            },
        }
        summaries.append(summary)
        if selection_eligible:
            eligible_summaries.append(summary)
    if unstable_configs:
        raise RuntimeError(
            "CPU timing variability exceeds the fixed CV threshold: "
            + ", ".join(unstable_configs)
        )
    if not eligible_summaries:
        raise RuntimeError("no correctness-passing CPU configuration is eligible")
    winner = min(
        eligible_summaries,
        key=lambda item: (
            item["workload_wall_median_seconds"],
            item["elapsed_statistics"]["median_seconds"],
            item["order"],
        ),
    )
    reference = next(item for item in summaries if item["lane"] == "diagnostic")
    canonical = [
        item
        for item in summaries
        if item["lane"] == "physical" and item["canonical_full_physical"] is True
    ]
    if len(canonical) != 1 or canonical[0]["eligible"] is not True:
        raise RuntimeError(
            "canonical full-physical CPU configuration is absent or failed qualification"
        )
    smt = [
        item for item in summaries if item["lane"] == "smt" and item["eligible"] is True
    ]
    fastest_smt = min(
        smt,
        key=lambda item: (
            item["workload_wall_median_seconds"],
            item["elapsed_statistics"]["median_seconds"],
            item["order"],
        ),
    ) if smt else None
    return {
        "pass": True,
        "criterion": (
            "minimum correctness-passing median timing.workload_wall_seconds; "
            "median timing.elapsed_seconds then fixed profile order break exact ties"
        ),
        "selected_config_id": winner["config_id"],
        "selected": winner,
        "fastest_passing_physical": winner,
        "canonical_full_physical": canonical[0],
        "fastest_passing_smt": fastest_smt,
        "single_core_reference": reference,
        "single_core_reference_is_diagnostic_only": True,
        "headline_lane": "physical",
        "speedup_over_1x1": (
            reference["workload_wall_median_seconds"]
            / winner["workload_wall_median_seconds"]
        ),
        "configurations": summaries,
    }


def _markdown_timing_table(selection: dict[str, Any]) -> str:
    rows = [
        "| Config | MPI | OMP | Selectable | Workload median / CV | Elapsed median / CV | Update median / CV | Forward median / CV | Adjoint median / CV | Gradient median / CV |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]

    def render(statistic: dict[str, Any]) -> str:
        return (
            f"{statistic['median_seconds']:.6f}s / "
            f"{statistic['coefficient_of_variation']:.4f}"
        )

    for config in selection["configurations"]:
        phases = config["phase_statistics"]
        rows.append(
            "| "
            + " | ".join(
                (
                    config["config_id"],
                    str(config["ranks"]),
                    str(config["omp_threads"]),
                    "yes" if config["selection_eligible"] else "no",
                    render(config["workload_wall_statistics"]),
                    render(config["elapsed_statistics"]),
                    render(phases["update_design"]),
                    render(phases["forward"]),
                    render(phases["adjoint"]),
                    render(phases["gradient"]),
                )
            )
            + " |"
        )
    return "\n".join(rows) + "\n"


def _material_gradient_gate(stdout: str, *, label: str) -> dict[str, Any]:
    return _MPI_RUNNER._extract_material_gradient_stats(stdout, lane=label)


def _sample_artifact_paths(
    output: pathlib.Path,
    run_id: str,
    config_id: str,
    sample_kind: str,
    iteration: int,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    label = f"{run_id}-{config_id}-{sample_kind}-{iteration}"
    return (
        output / f"{label}.json",
        output / f"{label}.stdout.log",
        output / f"{label}.stderr.log",
    )


def _is_scheduled_sample_artifact_name(name: str) -> bool:
    """Recognize artifacts from the active matrix without a config-ID regex."""

    if len(name) <= 33 or name[32] != "-" or not re.fullmatch(r"[0-9a-f]{32}", name[:32]):
        return False
    suffixes: set[str] = set()
    sentinel = "0" * 32
    for config_id, sample_kind, iteration in sample_schedule():
        suffixes.update(
            path.name[32:]
            for path in _sample_artifact_paths(
                pathlib.Path("."), sentinel, config_id, sample_kind, iteration
            )
        )
    return name[32:] in suffixes


def _validate_sample_artifacts(
    *,
    repo: pathlib.Path,
    output: pathlib.Path,
    runtime: dict[str, pathlib.Path],
    receipt: dict[str, Any],
    snapshot_sha256: str,
    producer_sha256: str,
    topology: dict[str, Any],
    run_id: str,
    config: CpuConfig,
    sample_kind: str,
    iteration: int,
    nonce: str,
    process_seconds: Any,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    label = f"{run_id}-{config.config_id}-{sample_kind}-{iteration}"
    result_path, stdout_path, stderr_path = _sample_artifact_paths(
        output, run_id, config.config_id, sample_kind, iteration
    )
    for path in (result_path, stdout_path, stderr_path):
        if path.parent.resolve() != output.resolve() or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{label} raw triad is incomplete or not regular")
    stdout = stdout_path.read_text(encoding="utf-8")
    pointer = _extract_pointer(stdout)
    if pathlib.Path(str(pointer.get("result_file", ""))).resolve() != result_path.resolve():
        raise RuntimeError(f"{label} returned the wrong result path")
    cache = hash_cache if hash_cache is not None else StatHashCache()
    if pointer.get("sha256") != cache.digest(result_path):
        raise RuntimeError(f"{label} result digest is absent or invalid")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} raw result") from error
    producer_argv = producer_arguments(
        repo=repo,
        runtime=runtime,
        result_path=result_path,
        receipt_id=receipt["receipt_id"],
        snapshot_sha256=snapshot_sha256,
        producer_sha256=producer_sha256,
        nonce=nonce,
        run_id=run_id,
        sample_kind=sample_kind,
        iteration=iteration,
    )
    validate_qualification_binding(
        result,
        nonce=nonce,
        run_id=run_id,
        receipt_id=receipt["receipt_id"],
        snapshot_sha256=snapshot_sha256,
        producer_sha256=producer_sha256,
        sample_kind=sample_kind,
        iteration=iteration,
        label=label,
    )
    if result.get("producer", {}).get("command") != producer_argv:
        raise RuntimeError(f"{label} producer command differs from launched argv")
    validation = validate_cpu_record(result, config, topology, label=label)
    checked_process_seconds = _validate_process_wall_seconds(
        process_seconds, validation, label=label
    )
    receipt_gate = _MPI_RUNNER._validate_receipt_rank_runtimes(
        receipt, validation, hash_cache=cache
    )
    receipt_gate["pass"] = True
    environment = _child_environment(
        runtime,
        config=config,
        output=output,
        run_id=run_id,
        sample_kind=sample_kind,
        iteration=iteration,
        nonce=nonce,
        prepare_xdg_cache=False,
    )
    environment_gate = _validate_process_environments(result, environment, label=label)
    material_gradient_gate = _material_gradient_gate(stdout, label="cpu-parallel")
    command = launch_arguments(
        runtime=runtime, config=config, producer_argv=producer_argv
    )
    return {
        "config_id": config.config_id,
        "config": asdict(config),
        "sample_kind": sample_kind,
        "iteration": iteration,
        "nonce": nonce,
        "process_seconds": checked_process_seconds,
        "validation": validation,
        "receipt_gate": receipt_gate,
        "process_environment_gate": environment_gate,
        "material_gradient_gate": material_gradient_gate,
        "result": result,
        "result_file": str(result_path.resolve()),
        "result_sha256": cache.digest(result_path),
        "stdout_log": str(stdout_path.resolve()),
        "stdout_sha256": cache.digest(stdout_path),
        "stderr_log": str(stderr_path.resolve()),
        "stderr_sha256": cache.digest(stderr_path),
        "command": command,
        "producer_command": producer_argv,
        "environment": dict(sorted(environment.items())),
    }


def _run_raw_sample(
    *,
    repo: pathlib.Path,
    output: pathlib.Path,
    runtime: dict[str, pathlib.Path],
    receipt: dict[str, Any],
    snapshot_sha256: str,
    producer_sha256: str,
    topology: dict[str, Any],
    run_id: str,
    config: CpuConfig,
    sample_kind: str,
    iteration: int,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    label = f"{run_id}-{config.config_id}-{sample_kind}-{iteration}"
    result_path, stdout_path, stderr_path = _sample_artifact_paths(
        output, run_id, config.config_id, sample_kind, iteration
    )
    for path in (result_path, stdout_path, stderr_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to overwrite existing sample artifact: {path}")
    nonce = uuid.uuid4().hex
    producer_argv = producer_arguments(
        repo=repo,
        runtime=runtime,
        result_path=result_path,
        receipt_id=receipt["receipt_id"],
        snapshot_sha256=snapshot_sha256,
        producer_sha256=producer_sha256,
        nonce=nonce,
        run_id=run_id,
        sample_kind=sample_kind,
        iteration=iteration,
    )
    command = launch_arguments(runtime=runtime, config=config, producer_argv=producer_argv)
    environment = _child_environment(
        runtime,
        config=config,
        output=output,
        run_id=run_id,
        sample_kind=sample_kind,
        iteration=iteration,
        nonce=nonce,
        prepare_xdg_cache=True,
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
        raise ResumableRunFailure(
            f"{label} exceeded the fixed CPU benchmark timeout"
        ) from error
    atomic_write_text(stdout_path, stdout)
    atomic_write_text(stderr_path, stderr)
    if returncode != 0:
        raise RuntimeError(f"{label} exited {returncode}; see {stderr_path}")
    return _validate_sample_artifacts(
        repo=repo,
        output=output,
        runtime=runtime,
        receipt=receipt,
        snapshot_sha256=snapshot_sha256,
        producer_sha256=producer_sha256,
        topology=topology,
        run_id=run_id,
        config=config,
        sample_kind=sample_kind,
        iteration=iteration,
        nonce=nonce,
        process_seconds=process_seconds,
        hash_cache=hash_cache,
    )


def _gate_checkpoint_record(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("pass") is not True:
        raise RuntimeError(f"cannot checkpoint failed or absent {label} gate")
    return {
        "pass": True,
        "payload": payload,
        "sha256": canonical_sha256(payload),
    }


def _checkpoint_provenance(anchor: dict[str, Any]) -> dict[str, Any]:
    record = anchor["record"]
    return {
        "anchor_sha256": anchor["sha256"],
        "receipt_id": record["receipt"]["receipt_id"],
        "receipt_sha256": record["receipt"]["sha256"],
        "source_snapshot_sha256": record["source_snapshot"]["sha256"],
        "producer_sha256": record["producer"]["sha256"],
        "runner_provenance_sha256": record["runner_provenance"]["sha256"],
        "topology_sha256": record["topology"]["sha256"],
        "matrix_sha256": record["matrix"]["sha256"],
        "boot_id": record["boot"]["boot_id"],
    }


def _make_checkpoint_entry(
    *,
    output: pathlib.Path,
    anchor: dict[str, Any],
    index: int,
    previous_sha256: str | None,
    sample: dict[str, Any],
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    schedule = sample_schedule()
    if index < 0 or index >= len(schedule):
        raise RuntimeError("checkpoint sample index is outside the fixed schedule")
    config_id, sample_kind, iteration = schedule[index]
    if (
        sample.get("config_id") != config_id
        or sample.get("sample_kind") != sample_kind
        or sample.get("iteration") != iteration
        or sample.get("config") != asdict(_config_by_id(config_id))
    ):
        raise RuntimeError("checkpoint sample is not the exact next fixed-schedule item")
    nonce = sample.get("nonce")
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise RuntimeError("checkpoint sample nonce is invalid")
    process_seconds = _finite_positive(
        sample.get("process_seconds"), "checkpoint.process_seconds"
    )
    _validate_process_wall_seconds(
        process_seconds, sample.get("validation", {}), label="checkpoint"
    )
    gates = {
        name: _gate_checkpoint_record(sample.get(name), label=name)
        for name in (
            "validation",
            "receipt_gate",
            "process_environment_gate",
            "material_gradient_gate",
        )
    }
    correctness = sample.get("correctness")
    if index < REFERENCE_SCHEDULE_INDEX:
        if correctness is not None:
            raise RuntimeError("pre-reference checkpoint correctness must be deferred")
        gates["correctness"] = {
            "status": "deferred",
            "reference_schedule_index": REFERENCE_SCHEDULE_INDEX,
        }
    else:
        gates["correctness"] = {
            "status": "complete",
            **_gate_checkpoint_record(correctness, label="correctness"),
        }
    result_path, stdout_path, stderr_path = _sample_artifact_paths(
        output, anchor["record"]["run_id"], config_id, sample_kind, iteration
    )
    files = {
        "result": _resolved_file_record(result_path, hash_cache=hash_cache),
        "stdout": _resolved_file_record(stdout_path, hash_cache=hash_cache),
        "stderr": _resolved_file_record(stderr_path, hash_cache=hash_cache),
    }
    expected_sample_paths = {
        "result": sample.get("result_file"),
        "stdout": sample.get("stdout_log"),
        "stderr": sample.get("stderr_log"),
    }
    expected_sample_hashes = {
        "result": sample.get("result_sha256"),
        "stdout": sample.get("stdout_sha256"),
        "stderr": sample.get("stderr_sha256"),
    }
    if any(
        expected_sample_paths[name] != files[name]["path"]
        or expected_sample_hashes[name] != files[name]["sha256"]
        for name in files
    ):
        raise RuntimeError("checkpoint sample/raw triad records disagree")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "gpmeep-cpu-parallel-sample-checkpoint",
        "committed_at_utc": _utc_now(),
        "run_id": anchor["record"]["run_id"],
        "index": index,
        "previous_entry_sha256": previous_sha256,
        "sample": {
            "config_id": config_id,
            "config": asdict(_config_by_id(config_id)),
            "sample_kind": sample_kind,
            "iteration": iteration,
            "nonce": nonce,
            "process_seconds": process_seconds,
        },
        "files": files,
        "provenance": _checkpoint_provenance(anchor),
        "gates": gates,
    }


def _publish_checkpoint_entry(
    output: pathlib.Path,
    entry: dict[str, Any],
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    index = entry.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise RuntimeError("checkpoint entry index is invalid")
    path = _entry_path(output, index)
    _atomic_create_json(path, entry)
    file = _regular_evidence_record(
        path,
        output=output,
        expected_parent=_checkpoint_root(output) / CHECKPOINT_ENTRIES_DIRECTORY,
        label="checkpoint entry",
        hash_cache=hash_cache,
    )
    return {"path": file["path"], "sha256": file["sha256"], "record": entry}


def _enumerate_indexed_json(
    directory: pathlib.Path,
    *,
    label: str,
    hash_cache: StatHashCache | None = None,
) -> list[pathlib.Path]:
    if not directory.exists():
        return []
    _assert_no_symlink_components(directory, require_leaf=True, label=label)
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError(f"{label} directory is invalid")
    paths = sorted(directory.iterdir())
    expected_names = [f"{index:06d}.json" for index in range(len(paths))]
    if [path.name for path in paths] != expected_names:
        raise RuntimeError(f"{label} contains a gap, duplicate, temporary, or invalid entry")
    for path in paths:
        _regular_evidence_record(
            path,
            expected_parent=directory,
            label=label,
            hash_cache=hash_cache,
        )
    return paths


def _inspect_checkpoint_chain(
    output: pathlib.Path,
    anchor: dict[str, Any],
    *,
    hash_cache: StatHashCache | None = None,
) -> list[dict[str, Any]]:
    _verify_anchor_evidence_on_disk(output, anchor, hash_cache=hash_cache)
    paths = _enumerate_indexed_json(
        _checkpoint_root(output) / CHECKPOINT_ENTRIES_DIRECTORY,
        label="checkpoint entries",
        hash_cache=hash_cache,
    )
    if len(paths) > len(sample_schedule()):
        raise RuntimeError("checkpoint journal exceeds the fixed schedule")
    inspected: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    nonces: set[str] = set()
    for index, path in enumerate(paths):
        entry = _read_json_object(path, label=f"checkpoint entry {index}")
        config_id, sample_kind, iteration = sample_schedule()[index]
        sample_binding = entry.get("sample")
        if (
            entry.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or entry.get("kind") != "gpmeep-cpu-parallel-sample-checkpoint"
            or entry.get("run_id") != anchor["record"]["run_id"]
            or entry.get("index") != index
            or entry.get("previous_entry_sha256") != previous_sha256
            or not isinstance(entry.get("committed_at_utc"), str)
            or not entry["committed_at_utc"].endswith("Z")
            or not isinstance(sample_binding, dict)
            or sample_binding.get("config_id") != config_id
            or sample_binding.get("config") != asdict(_config_by_id(config_id))
            or sample_binding.get("sample_kind") != sample_kind
            or sample_binding.get("iteration") != iteration
            or entry.get("provenance") != _checkpoint_provenance(anchor)
        ):
            raise RuntimeError("checkpoint entry identity/order/hash-chain changed")
        nonce = sample_binding.get("nonce")
        if (
            not isinstance(nonce, str)
            or not re.fullmatch(r"[0-9a-f]{32}", nonce)
            or nonce in nonces
        ):
            raise RuntimeError("checkpoint nonce is invalid or duplicate")
        nonces.add(nonce)
        file = _regular_evidence_record(
            path,
            output=output,
            expected_parent=_checkpoint_root(output) / CHECKPOINT_ENTRIES_DIRECTORY,
            label="checkpoint entry",
            hash_cache=hash_cache,
        )
        inspected.append({"path": path, "file": file, "record": entry})
        previous_sha256 = file["sha256"]
    return inspected


def _gate_record_matches(stored: Any, payload: dict[str, Any], *, label: str) -> None:
    expected = _gate_checkpoint_record(payload, label=label)
    if stored != expected:
        raise RuntimeError(f"checkpoint {label} gate differs from raw recomputation")


def _scan_sample_artifacts(
    output: pathlib.Path, run_id: str, committed_count: int
) -> None:
    schedule = sample_schedule()
    expected_all: set[pathlib.Path] = set()
    for index, (config_id, sample_kind, iteration) in enumerate(schedule):
        paths = _sample_artifact_paths(
            output, run_id, config_id, sample_kind, iteration
        )
        expected_all.update(path.resolve() for path in paths)
        existence = [path.exists() or path.is_symlink() for path in paths]
        if index < committed_count and existence != [True, True, True]:
            raise RuntimeError("committed checkpoint raw triad is partial")
        if index >= committed_count and any(existence):
            raise RuntimeError("future/uncheckpointed raw sample artifact is present")
    for path in output.iterdir():
        if _is_scheduled_sample_artifact_name(path.name) and path.resolve() not in expected_all:
            raise RuntimeError("stale or foreign raw sample artifact is present")


def _load_checkpoint_entries(
    *,
    repo: pathlib.Path,
    output: pathlib.Path,
    runtime: dict[str, pathlib.Path],
    receipt: dict[str, Any],
    anchor: dict[str, Any],
    hash_cache: StatHashCache | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = hash_cache if hash_cache is not None else StatHashCache()
    inspected_entries = _inspect_checkpoint_chain(
        output, anchor, hash_cache=cache
    )
    paths = [item["path"] for item in inspected_entries]
    _scan_sample_artifacts(output, anchor["record"]["run_id"], len(paths))
    samples: list[dict[str, Any]] = []
    entry_records: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    nonces: set[str] = set()
    reference: dict[str, Any] | None = None
    for index, path in enumerate(paths):
        entry = inspected_entries[index]["record"]
        config_id, sample_kind, iteration = sample_schedule()[index]
        sample_binding = entry.get("sample")
        if (
            entry.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or entry.get("kind") != "gpmeep-cpu-parallel-sample-checkpoint"
            or entry.get("run_id") != anchor["record"]["run_id"]
            or entry.get("index") != index
            or entry.get("previous_entry_sha256") != previous_sha256
            or not isinstance(entry.get("committed_at_utc"), str)
            or not entry["committed_at_utc"].endswith("Z")
            or not isinstance(sample_binding, dict)
            or sample_binding.get("config_id") != config_id
            or sample_binding.get("config") != asdict(_config_by_id(config_id))
            or sample_binding.get("sample_kind") != sample_kind
            or sample_binding.get("iteration") != iteration
            or entry.get("provenance") != _checkpoint_provenance(anchor)
        ):
            raise RuntimeError("checkpoint entry identity/order/hash-chain changed")
        nonce = sample_binding.get("nonce")
        if (
            not isinstance(nonce, str)
            or not re.fullmatch(r"[0-9a-f]{32}", nonce)
            or nonce in nonces
        ):
            raise RuntimeError("checkpoint nonce is invalid or duplicate")
        nonces.add(nonce)
        process_seconds = _finite_positive(
            sample_binding.get("process_seconds"),
            f"checkpoint[{index}].process_seconds",
        )
        expected_paths = _sample_artifact_paths(
            output,
            anchor["record"]["run_id"],
            config_id,
            sample_kind,
            iteration,
        )
        stored_files = entry.get("files")
        if not isinstance(stored_files, dict) or set(stored_files) != {
            "result",
            "stdout",
            "stderr",
        }:
            raise RuntimeError("checkpoint raw triad manifest is malformed")
        for name, artifact_path in zip(
            ("result", "stdout", "stderr"), expected_paths
        ):
            if stored_files[name] != _resolved_file_record(
                artifact_path, hash_cache=cache
            ):
                raise RuntimeError("checkpoint raw triad changed")
        sample = _validate_sample_artifacts(
            repo=repo,
            output=output,
            runtime=runtime,
            receipt=receipt,
            snapshot_sha256=anchor["record"]["source_snapshot"]["sha256"],
            producer_sha256=anchor["record"]["producer"]["sha256"],
            topology=anchor["topology_evidence"]["record"]["normalized"],
            run_id=anchor["record"]["run_id"],
            config=_config_by_id(config_id),
            sample_kind=sample_kind,
            iteration=iteration,
            nonce=nonce,
            process_seconds=process_seconds,
            hash_cache=cache,
        )
        gates = entry.get("gates")
        if not isinstance(gates, dict):
            raise RuntimeError("checkpoint gate manifest is absent")
        for gate_name in (
            "validation",
            "receipt_gate",
            "process_environment_gate",
            "material_gradient_gate",
        ):
            _gate_record_matches(gates.get(gate_name), sample[gate_name], label=gate_name)
        if index == REFERENCE_SCHEDULE_INDEX:
            reference = sample["result"]
        if index < REFERENCE_SCHEDULE_INDEX:
            if gates.get("correctness") != {
                "status": "deferred",
                "reference_schedule_index": REFERENCE_SCHEDULE_INDEX,
            }:
                raise RuntimeError("pre-reference checkpoint correctness is not deferred")
        else:
            if reference is None:
                raise RuntimeError("checkpoint correctness reference is absent")
            sample["correctness"] = compare_to_reference(
                reference,
                sample["result"],
                label=f"{config_id}-{sample_kind}-{iteration}",
            )
            expected_correctness = {
                "status": "complete",
                **_gate_checkpoint_record(sample["correctness"], label="correctness"),
            }
            if gates.get("correctness") != expected_correctness:
                raise RuntimeError("checkpoint correctness differs from raw recomputation")
        samples.append(sample)
        current_sha256 = inspected_entries[index]["file"]["sha256"]
        entry_records.append(
            {"index": index, "path": str(path.resolve()), "sha256": current_sha256}
        )
        previous_sha256 = current_sha256
    if reference is not None:
        for index, sample in enumerate(samples[:REFERENCE_SCHEDULE_INDEX]):
            sample["correctness"] = compare_to_reference(
                reference,
                sample["result"],
                label=f"{sample['config_id']}-{sample['sample_kind']}-{sample['iteration']}",
            )
    validate_sample_prefix(samples)
    closure_record = {
        "entry_count": len(entry_records),
        "expected_entry_count": len(sample_schedule()),
        "complete": len(entry_records) == len(sample_schedule()),
        "head_sha256": previous_sha256,
        "entries": entry_records,
    }
    return samples, {
        "pass": True,
        "record": closure_record,
        "sha256": canonical_sha256(closure_record),
    }


def _structural_checkpoint_closure(
    output: pathlib.Path,
    anchor: dict[str, Any],
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    inspected = _inspect_checkpoint_chain(
        output, anchor, hash_cache=hash_cache
    )
    entries = [
        {
            "index": index,
            "path": item["file"]["path"],
            "sha256": item["file"]["sha256"],
        }
        for index, item in enumerate(inspected)
    ]
    record = {
        "entry_count": len(entries),
        "expected_entry_count": len(sample_schedule()),
        "complete": len(entries) == len(sample_schedule()),
        "head_sha256": entries[-1]["sha256"] if entries else None,
        "entries": entries,
    }
    return {"pass": True, "record": record, "sha256": canonical_sha256(record)}


def _enumerate_failure_paths(
    output: pathlib.Path, *, hash_cache: StatHashCache | None = None
) -> list[pathlib.Path]:
    failed_path = output / "FAILED.json"
    failure_directory = _checkpoint_root(output) / CHECKPOINT_FAILURES_DIRECTORY
    if not (failed_path.exists() or failed_path.is_symlink()):
        if failure_directory.exists() or failure_directory.is_symlink():
            paths = _enumerate_indexed_json(
                failure_directory,
                label="failure events",
                hash_cache=hash_cache,
            )
            if paths:
                raise RuntimeError("failure journal has no immutable FAILED genesis")
        return []
    _regular_evidence_record(
        failed_path,
        output=output,
        expected_parent=output,
        label="immutable FAILED genesis",
        hash_cache=hash_cache,
    )
    later: list[pathlib.Path] = []
    if failure_directory.exists() or failure_directory.is_symlink():
        _assert_no_symlink_components(
            failure_directory, require_leaf=True, label="failure events"
        )
        if not failure_directory.is_dir() or failure_directory.is_symlink():
            raise RuntimeError("failure events directory is invalid")
        later = sorted(failure_directory.iterdir())
        expected = [f"{index:06d}.json" for index in range(1, len(later) + 1)]
        if [path.name for path in later] != expected:
            raise RuntimeError("failure events contain a gap or invalid entry")
        for path in later:
            _regular_evidence_record(
                path,
                output=output,
                expected_parent=failure_directory,
                label="failure event",
                hash_cache=hash_cache,
            )
    return [failed_path, *later]


def _load_failure_events(
    output: pathlib.Path,
    anchor: dict[str, Any],
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    paths = _enumerate_failure_paths(output, hash_cache=hash_cache)
    structural = _structural_checkpoint_closure(
        output, anchor, hash_cache=hash_cache
    )
    current_entries = structural["record"]["entries"]
    records: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    terminal_seen = False
    for index, path in enumerate(paths):
        if terminal_seen:
            raise RuntimeError("failure event exists after a terminal failure")
        event = _read_json_object(path, label=f"failure event {index}")
        disposition = event.get("disposition")
        checkpoint = event.get("checkpoint")
        checkpoint_record = checkpoint.get("record") if isinstance(checkpoint, dict) else None
        checkpoint_count = (
            checkpoint_record.get("entry_count")
            if isinstance(checkpoint_record, dict)
            else None
        )
        if (
            isinstance(checkpoint_count, bool)
            or not isinstance(checkpoint_count, int)
            or checkpoint_count < 0
            or checkpoint_count > len(current_entries)
        ):
            raise RuntimeError("failure event checkpoint count is invalid")
        prefix_entries = current_entries[:checkpoint_count]
        expected_checkpoint_record = {
            "entry_count": checkpoint_count,
            "expected_entry_count": len(sample_schedule()),
            "complete": checkpoint_count == len(sample_schedule()),
            "head_sha256": prefix_entries[-1]["sha256"] if prefix_entries else None,
            "entries": prefix_entries,
        }
        expected_checkpoint = {
            "pass": True,
            "record": expected_checkpoint_record,
            "sha256": canonical_sha256(expected_checkpoint_record),
        }
        error_record = event.get("error")
        superseded = event.get("superseded_state")
        if superseded is not None:
            superseded_file = superseded.get("file") if isinstance(superseded, dict) else None
            superseded_payload = (
                superseded.get("payload") if isinstance(superseded, dict) else None
            )
            encoded = (
                json.dumps(
                    superseded_payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            if (
                not isinstance(superseded_file, dict)
                or superseded_file.get("path") != str((output / "state.json").resolve())
                or superseded_file.get("size_bytes") != len(encoded)
                or superseded_file.get("sha256")
                != hashlib.sha256(encoded).hexdigest()
            ):
                raise RuntimeError("failure event superseded-state evidence changed")
        if (
            event.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or event.get("kind") != "gpmeep-cpu-parallel-failure-event"
            or event.get("index") != index
            or event.get("previous_failure_sha256") != previous_sha256
            or event.get("run_id") != anchor["record"]["run_id"]
            or event.get("anchor_sha256") != anchor["sha256"]
            or event.get("receipt") != anchor["record"]["receipt"]
            or event.get("provenance") != _checkpoint_provenance(anchor)
            or event.get("checkpoint") != expected_checkpoint
            or not isinstance(event.get("recorded_at_utc"), str)
            or not event["recorded_at_utc"].endswith("Z")
            or event.get("classification")
            not in {
                "resumable-interruption",
                "receipt-invalid-correctness",
                "nonresumable-run-failure",
            }
            or not isinstance(disposition, dict)
            or set(disposition) != {"resumable", "receipt_valid", "terminal"}
            or disposition.get("terminal") is not (not disposition.get("resumable"))
            or not isinstance(error_record, dict)
            or set(error_record) != {"type", "message", "traceback", "details"}
            or not isinstance(error_record.get("type"), str)
            or not isinstance(error_record.get("message"), str)
            or not isinstance(error_record.get("traceback"), str)
        ):
            raise RuntimeError("failure event identity/hash-chain changed")
        if event["classification"] == "receipt-invalid-correctness":
            if disposition != {
                "resumable": False,
                "receipt_valid": False,
                "terminal": True,
            } or not isinstance(error_record.get("details"), dict):
                raise RuntimeError("correctness failure disposition/details changed")
        elif event["classification"] == "resumable-interruption":
            if disposition != {
                "resumable": True,
                "receipt_valid": True,
                "terminal": False,
            }:
                raise RuntimeError("resumable failure disposition changed")
        elif disposition != {
            "resumable": False,
            "receipt_valid": None,
            "terminal": True,
        }:
            raise RuntimeError("nonresumable failure disposition changed")
        file = _regular_evidence_record(
            path,
            output=output,
            expected_parent=(
                output
                if index == 0
                else _checkpoint_root(output) / CHECKPOINT_FAILURES_DIRECTORY
            ),
            label="failure event",
            hash_cache=hash_cache,
        )
        records.append({"index": index, **file})
        previous_sha256 = file["sha256"]
        terminal_seen = disposition["terminal"]
    record = {
        "event_count": len(records),
        "head_sha256": previous_sha256,
        "terminal": terminal_seen,
        "events": records,
    }
    return {"pass": True, "record": record, "sha256": canonical_sha256(record)}


def _failure_classification(error: BaseException) -> tuple[str, dict[str, Any]]:
    if isinstance(error, CorrectnessGateFailure):
        return "receipt-invalid-correctness", {
            "resumable": False,
            "receipt_valid": False,
            "terminal": True,
        }
    if isinstance(error, (KeyboardInterrupt, ResumableRunFailure)):
        return "resumable-interruption", {
            "resumable": True,
            "receipt_valid": True,
            "terminal": False,
        }
    return "nonresumable-run-failure", {
        "resumable": False,
        "receipt_valid": None,
        "terminal": True,
    }


def _publish_failure_event(
    output: pathlib.Path,
    anchor: dict[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    existing = _load_failure_events(output, anchor)
    if existing["record"]["terminal"]:
        raise RuntimeError("cannot append a failure after a terminal failure")
    checkpoint = _structural_checkpoint_closure(output, anchor)
    state_path = output / "state.json"
    superseded_state: dict[str, Any] | None = None
    if state_path.exists() or state_path.is_symlink():
        state_file = _regular_evidence_record(
            state_path,
            output=output,
            expected_parent=output,
            label="superseded state evidence",
        )
        superseded_state = {
            "file": state_file,
            "payload": _read_json_object(
                state_path, label="superseded state evidence"
            ),
        }
    classification, disposition = _failure_classification(error)
    index = existing["record"]["event_count"]
    event = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "gpmeep-cpu-parallel-failure-event",
        "recorded_at_utc": _utc_now(),
        "index": index,
        "previous_failure_sha256": existing["record"]["head_sha256"],
        "run_id": anchor["record"]["run_id"],
        "anchor_sha256": anchor["sha256"],
        "receipt": anchor["record"]["receipt"],
        "checkpoint": checkpoint,
        "classification": classification,
        "disposition": disposition,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "details": getattr(error, "details", None),
        },
        "superseded_state": superseded_state,
        "provenance": _checkpoint_provenance(anchor),
    }
    path = _failure_path(output, index)
    _atomic_create_json(path, event)
    closure = _load_failure_events(output, anchor)
    if closure["record"]["event_count"] != index + 1:
        raise RuntimeError("failure event publication did not extend the journal")
    return {"event": closure["record"]["events"][-1], "closure": closure}


def _load_resume_events(
    output: pathlib.Path,
    anchor: dict[str, Any],
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    cache = hash_cache if hash_cache is not None else StatHashCache()
    paths = _enumerate_indexed_json(
        _checkpoint_root(output) / CHECKPOINT_RESUMES_DIRECTORY,
        label="resume events",
        hash_cache=cache,
    )
    records: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    entry_paths = _enumerate_indexed_json(
        _checkpoint_root(output) / CHECKPOINT_ENTRIES_DIRECTORY,
        label="checkpoint entries",
        hash_cache=cache,
    )
    entry_hashes = [cache.digest(path) for path in entry_paths]
    failure_closure = _load_failure_events(
        output, anchor, hash_cache=cache
    )
    failure_events = failure_closure["record"]["events"]
    for index, path in enumerate(paths):
        event = _read_json_object(path, label=f"resume event {index}")
        checkpoint_count = event.get("checkpoint_entry_count")
        if (
            isinstance(checkpoint_count, bool)
            or not isinstance(checkpoint_count, int)
            or checkpoint_count < 0
            or checkpoint_count > len(entry_paths)
        ):
            raise RuntimeError("resume event checkpoint count is invalid")
        expected_checkpoint_head = (
            entry_hashes[checkpoint_count - 1] if checkpoint_count else None
        )
        expected_next = (
            list(sample_schedule()[checkpoint_count])
            if checkpoint_count < len(sample_schedule())
            else None
        )
        stored_failure_closure = event.get("failure_closure")
        stored_failure_record = (
            stored_failure_closure.get("record")
            if isinstance(stored_failure_closure, dict)
            else None
        )
        failure_count = (
            stored_failure_record.get("event_count")
            if isinstance(stored_failure_record, dict)
            else None
        )
        if (
            isinstance(failure_count, bool)
            or not isinstance(failure_count, int)
            or failure_count < 0
            or failure_count > len(failure_events)
        ):
            raise RuntimeError("resume event failure count is invalid")
        failure_prefix = failure_events[:failure_count]
        expected_failure_record = {
            "event_count": failure_count,
            "head_sha256": (
                failure_prefix[-1]["sha256"] if failure_prefix else None
            ),
            "terminal": False,
            "events": failure_prefix,
        }
        expected_failure_closure = {
            "pass": True,
            "record": expected_failure_record,
            "sha256": canonical_sha256(expected_failure_record),
        }
        continuity = event.get("power_continuity")
        load_state = event.get("load_state")
        if (
            event.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or event.get("kind") != "gpmeep-cpu-parallel-resume-event"
            or event.get("index") != index
            or event.get("run_id") != anchor["record"]["run_id"]
            or event.get("anchor_sha256") != anchor["sha256"]
            or event.get("previous_event_sha256") != previous_sha256
            or event.get("boot") != anchor["record"]["boot"]
            or event.get("host") != anchor["record"]["host"]
            or event.get("topology_identity_sha256")
            != anchor["record"]["topology"]["identity_sha256"]
            or event.get("parent_affinity") != anchor["record"]["parent_affinity"]
            or event.get("power_policy") != anchor["record"]["power_policy"]
            or event.get("provenance") != _checkpoint_provenance(anchor)
            or event.get("checkpoint_head_sha256") != expected_checkpoint_head
            or event.get("next_schedule_item") != expected_next
            or stored_failure_closure != expected_failure_closure
            or not isinstance(event.get("recorded_at_utc"), str)
            or not event["recorded_at_utc"].endswith("Z")
            or not isinstance(continuity, dict)
            or continuity.get("pass") is not True
            or not isinstance(load_state, dict)
            or any(
                isinstance(load_state.get(name), bool)
                or not isinstance(load_state.get(name), (int, float))
                or not math.isfinite(float(load_state[name]))
                for name in ("load_1m", "load_5m", "load_15m")
            )
        ):
            raise RuntimeError("resume event identity/hash-chain changed")
        current_sha256 = cache.digest(path)
        records.append(
            {"index": index, "path": str(path.resolve()), "sha256": current_sha256}
        )
        previous_sha256 = current_sha256
    record = {
        "event_count": len(records),
        "head_sha256": previous_sha256,
        "events": records,
    }
    return {"pass": True, "record": record, "sha256": canonical_sha256(record)}


def _publish_resume_event(
    output: pathlib.Path,
    anchor: dict[str, Any],
    journal_closure: dict[str, Any],
) -> dict[str, Any]:
    existing = _load_resume_events(output, anchor)
    failure_closure = _load_failure_events(output, anchor)
    if failure_closure["record"]["terminal"]:
        raise NonResumableRun("terminal failure forbids resume-event publication")
    index = existing["record"]["event_count"]
    event = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "gpmeep-cpu-parallel-resume-event",
        "recorded_at_utc": _utc_now(),
        "index": index,
        "previous_event_sha256": existing["record"]["head_sha256"],
        "run_id": anchor["record"]["run_id"],
        "anchor_sha256": anchor["sha256"],
        "boot": anchor["record"]["boot"],
        "host": anchor["record"]["host"],
        "topology_identity_sha256": anchor["record"]["topology"]["identity_sha256"],
        "parent_affinity": anchor["record"]["parent_affinity"],
        "power_policy": anchor["record"]["power_policy"],
        "power_continuity": anchor["resume_power_continuity"],
        "load_state": anchor["resume_live_state"]["loadavg"],
        "checkpoint_entry_count": journal_closure["record"]["entry_count"],
        "checkpoint_head_sha256": journal_closure["record"]["head_sha256"],
        "failure_closure": failure_closure,
        "next_schedule_item": (
            list(sample_schedule()[journal_closure["record"]["entry_count"]])
            if journal_closure["record"]["entry_count"] < len(sample_schedule())
            else None
        ),
        "provenance": _checkpoint_provenance(anchor),
    }
    path = _resume_path(output, index)
    _atomic_create_json(path, event)
    file = _regular_evidence_record(
        path,
        output=output,
        expected_parent=_checkpoint_root(output) / CHECKPOINT_RESUMES_DIRECTORY,
        label="resume event",
    )
    return {"path": file["path"], "sha256": file["sha256"], "record": event}


def _enumerate_orphan_directories(output: pathlib.Path) -> list[pathlib.Path]:
    root = _checkpoint_root(output) / CHECKPOINT_ORPHANS_DIRECTORY
    if not root.exists():
        return []
    _assert_no_symlink_components(root, require_leaf=True, label="orphan root")
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("orphan root is invalid")
    paths = sorted(root.iterdir())
    if [path.name for path in paths] != [f"{index:06d}" for index in range(len(paths))]:
        raise RuntimeError("orphan recovery directories contain a gap or foreign entry")
    for path in paths:
        _assert_no_symlink_components(path, require_leaf=True, label="orphan directory")
        status = path.lstat()
        if not stat.S_ISDIR(status.st_mode) or status.st_mode & 0o077:
            raise RuntimeError("orphan recovery directory is not private")
    return paths


def _validate_recovery_manifest_artifacts(
    output: pathlib.Path, anchor: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    manifest_index = manifest.get("index")
    if (
        isinstance(manifest_index, bool)
        or not isinstance(manifest_index, int)
        or manifest_index < 0
    ):
        raise RuntimeError("recovery manifest index is invalid")
    expected_orphan = _absolute_lexical(_orphan_path(output, manifest_index))
    output = _absolute_lexical(output)
    count = manifest.get("committed_checkpoint_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or count > len(sample_schedule())
    ):
        raise RuntimeError("recovery manifest checkpoint count is invalid")
    allowed: dict[pathlib.Path, str] = {}
    if count < len(sample_schedule()):
        config_id, sample_kind, iteration = sample_schedule()[count]
        allowed.update(
            {
                _absolute_lexical(path): "interrupted-next-sample"
                for path in _sample_artifact_paths(
                    output,
                    anchor["record"]["run_id"],
                    config_id,
                    sample_kind,
                    iteration,
                )
            }
        )
    if count == len(sample_schedule()):
        allowed.update(
            {
                _absolute_lexical(output / name): "partial-publication"
                for name in RECOVERABLE_PUBLICATION_ARTIFACTS
            }
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("recovery manifest has no artifacts")
    seen: set[pathlib.Path] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError("recovery artifact record is malformed")
        original = _absolute_lexical(pathlib.Path(str(artifact.get("original_path", ""))))
        destination = _absolute_lexical(
            pathlib.Path(str(artifact.get("orphan_path", "")))
        )
        expected_destination = expected_orphan / original.name
        if (
            original in seen
            or original.parent != output
            or destination != expected_destination
            or destination.parent != expected_orphan
            or destination.name != original.name
            or allowed.get(original) != artifact.get("reason")
            or artifact.get("type") != "regular"
            or isinstance(artifact.get("size_bytes"), bool)
            or not isinstance(artifact.get("size_bytes"), int)
            or artifact["size_bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", "")))
        ):
            raise RuntimeError("recovery artifact is outside the exact recoverable set")
        _assert_no_symlink_components(
            original.parent, require_leaf=True, label="recovery original parent"
        )
        existing_destination_parent = destination.parent
        while (
            not existing_destination_parent.exists()
            and existing_destination_parent != existing_destination_parent.parent
        ):
            existing_destination_parent = existing_destination_parent.parent
        _assert_no_symlink_components(
            existing_destination_parent,
            require_leaf=True,
            label="recovery destination ancestor",
        )
        seen.add(original)
    return artifacts


def _validate_recovery_intent(
    output: pathlib.Path,
    anchor: dict[str, Any],
    intent: dict[str, Any],
    *,
    index: int,
    previous_sha256: str | None,
    expected_kind: str = "gpmeep-cpu-parallel-recovery-intent",
) -> list[dict[str, Any]]:
    if (
        intent.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or intent.get("kind") != expected_kind
        or intent.get("index") != index
        or intent.get("previous_event_sha256") != previous_sha256
        or intent.get("run_id") != anchor["record"]["run_id"]
        or intent.get("anchor_sha256") != anchor["sha256"]
        or intent.get("provenance") != _checkpoint_provenance(anchor)
        or not isinstance(intent.get("created_at_utc"), str)
        or not intent["created_at_utc"].endswith("Z")
    ):
        raise RuntimeError("recovery intent identity/provenance/hash-chain changed")
    return _validate_recovery_manifest_artifacts(output, anchor, intent)


def _load_recovery_events(
    output: pathlib.Path,
    anchor: dict[str, Any],
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    event_paths = _enumerate_indexed_json(
        _checkpoint_root(output) / CHECKPOINT_RECOVERIES_DIRECTORY,
        label="recovery events",
        hash_cache=hash_cache,
    )
    intent_paths = _enumerate_indexed_json(
        _checkpoint_root(output) / CHECKPOINT_RECOVERY_INTENTS_DIRECTORY,
        label="recovery intents",
        hash_cache=hash_cache,
    )
    orphan_paths = _enumerate_orphan_directories(output)
    if len(intent_paths) not in (len(event_paths), len(event_paths) + 1):
        raise RuntimeError("recovery intent/event counts disagree")
    if not (len(event_paths) <= len(orphan_paths) <= len(intent_paths)):
        raise RuntimeError("recovery intent/orphan/event counts disagree")
    records: list[dict[str, Any]] = []
    intent_records: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    for index, intent_path in enumerate(intent_paths):
        intent_file = _regular_evidence_record(
            intent_path,
            output=output,
            expected_parent=_checkpoint_root(output)
            / CHECKPOINT_RECOVERY_INTENTS_DIRECTORY,
            label="recovery intent",
            hash_cache=hash_cache,
        )
        intent = _read_json_object(intent_path, label=f"recovery intent {index}")
        artifacts = _validate_recovery_intent(
            output,
            anchor,
            intent,
            index=index,
            previous_sha256=previous_sha256,
        )
        intent_records.append(
            {"index": index, "path": intent_file["path"], "sha256": intent_file["sha256"]}
        )
        if index >= len(event_paths):
            if index < len(orphan_paths):
                orphan = orphan_paths[index]
                children = {child.name for child in orphan.iterdir()}
                if children:
                    manifest_path = orphan / "manifest.json"
                    if "manifest.json" not in children:
                        raise RuntimeError("pending orphan has artifacts without a manifest")
                    manifest_file = _regular_evidence_record(
                        manifest_path,
                        output=output,
                        expected_parent=orphan,
                        label="pending recovery manifest",
                        hash_cache=hash_cache,
                    )
                    manifest = _read_json_object(
                        manifest_path, label="pending recovery manifest"
                    )
                    if (
                        manifest.get("intent") != intent_file
                        or manifest.get("artifacts") != artifacts
                    ):
                        raise RuntimeError("pending recovery manifest differs from intent")
                    _validate_recovery_intent(
                        output,
                        anchor,
                        manifest,
                        index=index,
                        previous_sha256=previous_sha256,
                        expected_kind="gpmeep-cpu-parallel-recovery-manifest",
                    )
                    allowed = {
                        "manifest.json",
                        *(
                            pathlib.Path(artifact["orphan_path"]).name
                            for artifact in artifacts
                        ),
                    }
                    if not children.issubset(allowed):
                        raise RuntimeError("pending orphan contains an unrecorded artifact")
                    for artifact in artifacts:
                        original = _absolute_lexical(
                            pathlib.Path(artifact["original_path"])
                        )
                        destination = _absolute_lexical(
                            pathlib.Path(artifact["orphan_path"])
                        )
                        original_present = original.exists() or original.is_symlink()
                        destination_present = (
                            destination.exists() or destination.is_symlink()
                        )
                        if original_present == destination_present:
                            raise RuntimeError(
                                "pending recovery artifact must exist at exactly one path"
                            )
                        current_path = destination if destination_present else original
                        current_parent = orphan if destination_present else output
                        current = _regular_evidence_record(
                            current_path,
                            output=output,
                            expected_parent=current_parent,
                            label="pending recovery artifact",
                            hash_cache=hash_cache,
                        )
                        if current != {
                            "path": str(current_path),
                            "size_bytes": artifact["size_bytes"],
                            "sha256": artifact["sha256"],
                        }:
                            raise RuntimeError("pending recovery artifact changed")
            continue
        event_path = event_paths[index]
        event = _read_json_object(event_path, label=f"recovery event {index}")
        orphan = orphan_paths[index]
        manifest_path = orphan / "manifest.json"
        manifest_file = _regular_evidence_record(
            manifest_path,
            output=output,
            expected_parent=orphan,
            label="recovery manifest",
            hash_cache=hash_cache,
        )
        manifest = _read_json_object(manifest_path, label="recovery manifest")
        if (
            event.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or event.get("kind") != "gpmeep-cpu-parallel-recovery-event"
            or event.get("index") != index
            or event.get("run_id") != anchor["record"]["run_id"]
            or event.get("anchor_sha256") != anchor["sha256"]
            or event.get("previous_event_sha256") != previous_sha256
            or event.get("intent") != intent_file
            or event.get("manifest") != manifest_file
            or event.get("artifacts") != artifacts
            or manifest.get("intent") != intent_file
            or manifest.get("artifacts") != artifacts
        ):
            raise RuntimeError(
                "recovery event/intent/manifest closure or hash-chain changed"
            )
        _validate_recovery_intent(
            output,
            anchor,
            manifest,
            index=index,
            previous_sha256=previous_sha256,
            expected_kind="gpmeep-cpu-parallel-recovery-manifest",
        )
        allowed = {manifest_path.name}
        for artifact in artifacts:
            original = _absolute_lexical(pathlib.Path(artifact["original_path"]))
            destination = _absolute_lexical(pathlib.Path(artifact["orphan_path"]))
            if original.exists() or original.is_symlink():
                raise RuntimeError("completed recovery left its original artifact")
            destination_file = _regular_evidence_record(
                destination,
                output=output,
                expected_parent=orphan,
                label="quarantined recovery artifact",
                hash_cache=hash_cache,
            )
            if destination_file != {
                "path": str(destination),
                "size_bytes": artifact["size_bytes"],
                "sha256": artifact["sha256"],
            }:
                raise RuntimeError("quarantined recovery artifact changed")
            allowed.add(destination.name)
        if {child.name for child in orphan.iterdir()} != allowed:
            raise RuntimeError("orphan directory contains an unrecorded artifact")
        event_file = _regular_evidence_record(
            event_path,
            output=output,
            expected_parent=_checkpoint_root(output) / CHECKPOINT_RECOVERIES_DIRECTORY,
            label="recovery event",
            hash_cache=hash_cache,
        )
        records.append(
            {"index": index, "path": event_file["path"], "sha256": event_file["sha256"]}
        )
        previous_sha256 = event_file["sha256"]
    record = {
        "event_count": len(records),
        "intent_count": len(intent_records),
        "head_sha256": previous_sha256,
        "events": records,
        "intents": intent_records,
        "pending_intent_count": len(intent_paths) - len(event_paths),
        "pending_orphan_count": len(orphan_paths) - len(event_paths),
    }
    return {"pass": True, "record": record, "sha256": canonical_sha256(record)}


def _publication_recovery_context(
    output: pathlib.Path, anchor: dict[str, Any]
) -> dict[str, Any] | None:
    """Authenticate the narrow COMPLETE-state publication-recovery window."""

    state_path = output / "state.json"
    state_file = _regular_evidence_record(
        state_path,
        output=output,
        expected_parent=output,
        label="publication-recovery state",
    )
    state = _read_json_object(state_path, label="publication-recovery state")
    if state.get("state") != "COMPLETE":
        return None

    failures = _load_failure_events(output, anchor)
    if failures["record"]["terminal"]:
        raise NonResumableRun(
            "terminal failure forbids COMPLETE-state publication recovery"
        )
    recovery = _load_recovery_events(output, anchor)
    intents = recovery["record"]["intents"]
    if not intents:
        return None
    intent_reference = intents[-1]
    intent_path = pathlib.Path(intent_reference["path"])
    intent = _read_json_object(intent_path, label="publication recovery intent")
    index = intent["index"]
    if (
        index != len(intents) - 1
        or recovery["record"]["event_count"] not in (index, index + 1)
        or intent.get("committed_checkpoint_count") != len(sample_schedule())
    ):
        raise RuntimeError("COMPLETE-state recovery is not the latest full-run transaction")

    artifacts = intent.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("COMPLETE-state recovery artifacts are malformed")
    by_name: dict[str, dict[str, Any]] = {}
    current_paths: dict[str, pathlib.Path] = {}
    expected_orphan = _orphan_path(output, index)
    for artifact in artifacts:
        original = _absolute_lexical(pathlib.Path(str(artifact.get("original_path", ""))))
        destination = _absolute_lexical(pathlib.Path(str(artifact.get("orphan_path", ""))))
        name = original.name
        if (
            name in by_name
            or name not in RECOVERABLE_PUBLICATION_ARTIFACTS
            or original != _absolute_lexical(output / name)
            or destination != _absolute_lexical(expected_orphan / name)
            or artifact.get("reason") != "partial-publication"
        ):
            raise RuntimeError("COMPLETE-state recovery is not the exact publication quartet")
        original_present = original.exists() or original.is_symlink()
        destination_present = destination.exists() or destination.is_symlink()
        if original_present == destination_present:
            raise RuntimeError(
                "publication-recovery artifact must exist at exactly one bound path"
            )
        current = destination if destination_present else original
        current_file = _regular_evidence_record(
            current,
            output=output,
            expected_parent=expected_orphan if destination_present else output,
            label="publication-recovery artifact",
        )
        if (
            current_file["size_bytes"] != artifact.get("size_bytes")
            or current_file["sha256"] != artifact.get("sha256")
        ):
            raise RuntimeError("publication-recovery artifact changed after its intent")
        by_name[name] = artifact
        current_paths[name] = current
    if set(by_name) != set(RECOVERABLE_PUBLICATION_ARTIFACTS):
        raise RuntimeError("COMPLETE-state recovery is not the exact publication quartet")

    complete_path = current_paths["COMPLETE"]
    complete_file = by_name["COMPLETE"]
    if (
        state_file["size_bytes"] != complete_file["size_bytes"]
        or state_file["sha256"] != complete_file["sha256"]
        or state_path.read_bytes() != complete_path.read_bytes()
    ):
        raise RuntimeError("COMPLETE state is not byte-identical to the recovery marker")
    marker = _read_json_object(complete_path, label="recovered COMPLETE marker")
    if marker != state:
        raise RuntimeError("COMPLETE state payload differs from the recovery marker")

    marker_fields = {
        "schema_version",
        "state",
        "run_id",
        "profile_id",
        "report",
        "report_sha256",
        "report_markdown",
        "report_markdown_sha256",
        "run_anchor",
        "run_anchor_sha256",
        "checkpoint_entry_count",
        "checkpoint_head_sha256",
        "checkpoint_closure_sha256",
        "resume_event_count",
        "resume_closure_sha256",
        "failure_event_count",
        "failure_closure_sha256",
        "recovery_event_count",
        "recovery_closure_sha256",
        "publishing_marker",
    }
    if set(marker) != marker_fields:
        raise RuntimeError("recovered COMPLETE marker schema changed")

    checkpoint = _structural_checkpoint_closure(output, anchor)
    if checkpoint["record"]["complete"] is not True:
        raise RuntimeError("COMPLETE-state recovery has an incomplete checkpoint journal")
    resumes = _load_resume_events(output, anchor)
    marker_resume_count = marker.get("resume_event_count")
    marker_failure_count = marker.get("failure_event_count")
    for label, count, available in (
        ("resume", marker_resume_count, resumes["record"]["event_count"]),
        ("failure", marker_failure_count, failures["record"]["event_count"]),
    ):
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or count > available
        ):
            raise RuntimeError(f"recovered COMPLETE has an invalid {label} prefix")
    prior_resume_events = resumes["record"]["events"][:marker_resume_count]
    prior_resume_record = {
        "event_count": marker_resume_count,
        "head_sha256": (
            prior_resume_events[-1]["sha256"] if prior_resume_events else None
        ),
        "events": prior_resume_events,
    }
    prior_resumes = {
        "pass": True,
        "record": prior_resume_record,
        "sha256": canonical_sha256(prior_resume_record),
    }
    prior_failure_events = failures["record"]["events"][:marker_failure_count]
    prior_failure_record = {
        "event_count": marker_failure_count,
        "head_sha256": (
            prior_failure_events[-1]["sha256"] if prior_failure_events else None
        ),
        "terminal": False,
        "events": prior_failure_events,
    }
    prior_failures = {
        "pass": True,
        "record": prior_failure_record,
        "sha256": canonical_sha256(prior_failure_record),
    }
    prior_events = recovery["record"]["events"][:index]
    prior_intents = recovery["record"]["intents"][:index]
    if len(prior_events) != index or len(prior_intents) != index:
        raise RuntimeError("publication recovery does not extend a closed recovery prefix")
    prior_recovery_record = {
        "event_count": index,
        "intent_count": index,
        "head_sha256": intent.get("previous_event_sha256"),
        "events": prior_events,
        "intents": prior_intents,
        "pending_intent_count": 0,
        "pending_orphan_count": 0,
    }
    prior_recovery = {
        "pass": True,
        "record": prior_recovery_record,
        "sha256": canonical_sha256(prior_recovery_record),
    }
    expected_fields = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": anchor["record"]["run_id"],
        "profile_id": PROFILE_ID,
        "report": str(_absolute_lexical(output / "report.json")),
        "report_sha256": by_name["report.json"]["sha256"],
        "report_markdown": str(_absolute_lexical(output / "report.md")),
        "report_markdown_sha256": by_name["report.md"]["sha256"],
        "run_anchor": anchor["path"],
        "run_anchor_sha256": anchor["sha256"],
        "checkpoint_entry_count": checkpoint["record"]["entry_count"],
        "checkpoint_head_sha256": checkpoint["record"]["head_sha256"],
        "checkpoint_closure_sha256": checkpoint["sha256"],
        "resume_event_count": prior_resumes["record"]["event_count"],
        "resume_closure_sha256": prior_resumes["sha256"],
        "failure_event_count": prior_failures["record"]["event_count"],
        "failure_closure_sha256": prior_failures["sha256"],
        "recovery_event_count": prior_recovery["record"]["event_count"],
        "recovery_closure_sha256": prior_recovery["sha256"],
        "publishing_marker": {
            "path": str(_absolute_lexical(output / "PUBLISHING.json")),
            "size_bytes": by_name["PUBLISHING.json"]["size_bytes"],
            "sha256": by_name["PUBLISHING.json"]["sha256"],
        },
    }
    if any(marker.get(name) != value for name, value in expected_fields.items()):
        raise RuntimeError("recovered COMPLETE marker does not close live run evidence")
    publishing = _read_json_object(
        current_paths["PUBLISHING.json"], label="recovered PUBLISHING marker"
    )
    expected_publishing = {
        name: value for name, value in marker.items() if name != "publishing_marker"
    }
    expected_publishing["state"] = "PUBLISHING"
    if publishing != expected_publishing:
        raise RuntimeError("recovered COMPLETE does not close PUBLISHING evidence")
    return {
        "state": state,
        "intent": intent,
        "recovery_closure": recovery,
        "prior_resume_closure": prior_resumes,
        "prior_failure_closure": prior_failures,
        "prior_recovery_closure": prior_recovery,
    }


def _recovery_candidates(
    output: pathlib.Path,
    anchor: dict[str, Any],
    committed_count: int,
    *,
    allow_complete_publication_triad: bool,
) -> list[dict[str, Any]]:
    run_id = anchor["record"]["run_id"]
    schedule = sample_schedule()
    candidates: list[dict[str, Any]] = []
    expected_all: set[pathlib.Path] = set()
    for index, (config_id, sample_kind, iteration) in enumerate(schedule):
        paths = _sample_artifact_paths(output, run_id, config_id, sample_kind, iteration)
        expected_all.update(_absolute_lexical(path) for path in paths)
        if index <= committed_count:
            continue
        if any(path.exists() or path.is_symlink() for path in paths):
            raise RuntimeError("future sample beyond the exact next schedule item is present")
    if committed_count < len(schedule):
        config_id, sample_kind, iteration = schedule[committed_count]
        for path in _sample_artifact_paths(
            output, run_id, config_id, sample_kind, iteration
        ):
            if path.exists() or path.is_symlink():
                record = _regular_evidence_record(
                    path,
                    output=output,
                    expected_parent=output,
                    label="interrupted next-sample artifact",
                )
                candidates.append(
                    {
                        "original_path": record["path"],
                        "orphan_path": "",
                        "size_bytes": record["size_bytes"],
                        "sha256": record["sha256"],
                        "type": "regular",
                        "reason": "interrupted-next-sample",
                    }
                )
    for path in output.iterdir():
        if _is_scheduled_sample_artifact_name(path.name) and _absolute_lexical(path) not in expected_all:
            raise RuntimeError("foreign raw sample artifact is present")
    publication_paths = [output / name for name in RECOVERABLE_PUBLICATION_ARTIFACTS]
    present_publication = [path for path in publication_paths if path.exists() or path.is_symlink()]
    if present_publication:
        if committed_count != len(schedule):
            raise RuntimeError("publication artifact exists before the checkpoint journal is complete")
        if (
            len(present_publication) == len(publication_paths)
            and not allow_complete_publication_triad
        ):
            raise RuntimeError("complete publication triad requires authoritative verification")
        for path in present_publication:
            record = _regular_evidence_record(
                path,
                output=output,
                expected_parent=output,
                label="partial publication artifact",
            )
            candidates.append(
                {
                    "original_path": record["path"],
                    "orphan_path": "",
                    "size_bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                    "type": "regular",
                    "reason": "partial-publication",
                }
            )
    return candidates


def _complete_pending_recovery(
    output: pathlib.Path,
    anchor: dict[str, Any],
    orphan: pathlib.Path,
    manifest: dict[str, Any],
    *,
    fault_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    artifacts = _validate_recovery_manifest_artifacts(output, anchor, manifest)
    index = manifest["index"]
    _validate_recovery_intent(
        output,
        anchor,
        manifest,
        index=index,
        previous_sha256=manifest.get("previous_event_sha256"),
        expected_kind="gpmeep-cpu-parallel-recovery-manifest",
    )
    intent_path = _recovery_intent_path(output, index)
    intent_file = _regular_evidence_record(
        intent_path,
        output=output,
        expected_parent=_checkpoint_root(output)
        / CHECKPOINT_RECOVERY_INTENTS_DIRECTORY,
        label="recovery intent",
    )
    intent = _read_json_object(intent_path, label="recovery intent")
    intent_artifacts = _validate_recovery_intent(
        output,
        anchor,
        intent,
        index=index,
        previous_sha256=manifest.get("previous_event_sha256"),
    )
    if manifest.get("intent") != intent_file or artifacts != intent_artifacts:
        raise RuntimeError("recovery manifest differs from durable pending intent")
    expected_orphan = _absolute_lexical(_orphan_path(output, index))
    if _absolute_lexical(orphan) != expected_orphan:
        raise RuntimeError("passed recovery orphan is not the exact manifest orphan")
    _assert_no_symlink_components(
        orphan, require_leaf=True, label="recovery orphan directory"
    )
    orphan_status = orphan.lstat()
    if not stat.S_ISDIR(orphan_status.st_mode) or orphan_status.st_mode & 0o077:
        raise RuntimeError("recovery orphan directory is not private")
    for artifact_index, artifact in enumerate(artifacts):
        _validate_recovery_manifest_artifacts(output, anchor, manifest)
        original = _absolute_lexical(pathlib.Path(artifact["original_path"]))
        destination = _absolute_lexical(pathlib.Path(artifact["orphan_path"]))
        original_present = original.exists() or original.is_symlink()
        destination_present = destination.exists() or destination.is_symlink()
        if original_present and destination_present:
            raise RuntimeError("recovery artifact exists at both original and orphan paths")
        if original_present:
            current = _regular_evidence_record(
                original,
                output=output,
                expected_parent=output,
                label="recovery source",
            )
            if current["size_bytes"] != artifact["size_bytes"] or current["sha256"] != artifact["sha256"]:
                raise RuntimeError("recovery source changed before quarantine")
            if destination.exists() or destination.is_symlink():
                raise RuntimeError("recovery destination appeared before no-clobber move")
            _rename_noreplace(original, destination)
            _fsync_directory(output)
            _fsync_directory(orphan)
            if fault_injector is not None:
                fault_injector(f"after-artifact-move:{artifact_index}")
        elif not destination_present:
            raise RuntimeError("recovery artifact disappeared before quarantine")
        destination_file = _regular_evidence_record(
            destination,
            output=output,
            expected_parent=orphan,
            label="recovery orphan",
        )
        if destination_file["size_bytes"] != artifact["size_bytes"] or destination_file["sha256"] != artifact["sha256"]:
            raise RuntimeError("recovery orphan differs from its manifest")
    manifest_path = orphan / "manifest.json"
    manifest_file = _regular_evidence_record(
        manifest_path,
        output=output,
        expected_parent=orphan,
        label="recovery manifest",
    )
    event = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "gpmeep-cpu-parallel-recovery-event",
        "completed_at_utc": _utc_now(),
        "index": index,
        "previous_event_sha256": manifest["previous_event_sha256"],
        "run_id": anchor["record"]["run_id"],
        "anchor_sha256": anchor["sha256"],
        "intent": intent_file,
        "manifest": manifest_file,
        "artifacts": manifest["artifacts"],
    }
    path = _recovery_path(output, index)
    if fault_injector is not None:
        fault_injector("before-event")
    _atomic_create_json(path, event)
    if fault_injector is not None:
        fault_injector("after-event")
    return {"path": str(_absolute_lexical(path)), "sha256": sha256_file(path), "record": event}


def _recover_interrupted_evidence(
    output: pathlib.Path,
    anchor: dict[str, Any],
    committed_count: int,
    *,
    allow_complete_publication_triad: bool = False,
    fault_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    existing = _load_recovery_events(output, anchor)
    recovery_index = existing["record"]["event_count"]
    if existing["record"]["pending_intent_count"]:
        intent_path = _recovery_intent_path(output, recovery_index)
        intent = _read_json_object(intent_path, label="pending recovery intent")
        candidates = _validate_recovery_intent(
            output,
            anchor,
            intent,
            index=recovery_index,
            previous_sha256=existing["record"]["head_sha256"],
        )
        if intent.get("committed_checkpoint_count") != committed_count:
            raise RuntimeError("pending recovery intent has a stale checkpoint count")
        # Still reject any newly introduced future/foreign evidence. Remaining
        # original candidates may be a strict subset after a partial move.
        _recovery_candidates(
            output,
            anchor,
            committed_count,
            allow_complete_publication_triad=allow_complete_publication_triad,
        )
    else:
        candidates = _recovery_candidates(
            output,
            anchor,
            committed_count,
            allow_complete_publication_triad=allow_complete_publication_triad,
        )
        if not candidates:
            return existing
        expected_orphan = _orphan_path(output, recovery_index)
        for artifact in candidates:
            artifact["orphan_path"] = str(
                _absolute_lexical(
                    expected_orphan / pathlib.Path(artifact["original_path"]).name
                )
            )
        intent = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": "gpmeep-cpu-parallel-recovery-intent",
            "created_at_utc": _utc_now(),
            "index": recovery_index,
            "previous_event_sha256": existing["record"]["head_sha256"],
            "run_id": anchor["record"]["run_id"],
            "anchor_sha256": anchor["sha256"],
            "committed_checkpoint_count": committed_count,
            "provenance": _checkpoint_provenance(anchor),
            "artifacts": candidates,
        }
        _validate_recovery_intent(
            output,
            anchor,
            intent,
            index=recovery_index,
            previous_sha256=existing["record"]["head_sha256"],
        )
        if fault_injector is not None:
            fault_injector("before-intent")
        _atomic_create_json(_recovery_intent_path(output, recovery_index), intent)
        if fault_injector is not None:
            fault_injector("after-intent")
        existing = _load_recovery_events(output, anchor)
    orphan = _orphan_path(output, recovery_index)
    _assert_no_symlink_components(
        _checkpoint_root(output), require_leaf=True, label="checkpoint root"
    )
    orphan.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(orphan.parent, require_leaf=True, label="orphan root")
    if not orphan.exists():
        orphan.mkdir(mode=0o700)
        os.chmod(orphan, 0o700)
        _fsync_directory(orphan)
        _fsync_directory(orphan.parent)
        if fault_injector is not None:
            fault_injector("after-orphan-mkdir")
    else:
        _assert_no_symlink_components(
            orphan, require_leaf=True, label="pending orphan"
        )
        orphan_status = orphan.lstat()
        if not stat.S_ISDIR(orphan_status.st_mode) or orphan_status.st_mode & 0o077:
            raise RuntimeError("pending orphan is not a private directory")
    intent_path = _recovery_intent_path(output, recovery_index)
    intent_file = _regular_evidence_record(
        intent_path,
        output=output,
        expected_parent=_checkpoint_root(output)
        / CHECKPOINT_RECOVERY_INTENTS_DIRECTORY,
        label="recovery intent",
    )
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "gpmeep-cpu-parallel-recovery-manifest",
        "created_at_utc": intent["created_at_utc"],
        "index": recovery_index,
        "previous_event_sha256": existing["record"]["head_sha256"],
        "run_id": anchor["record"]["run_id"],
        "anchor_sha256": anchor["sha256"],
        "committed_checkpoint_count": committed_count,
        "provenance": _checkpoint_provenance(anchor),
        "intent": intent_file,
        "artifacts": candidates,
    }
    manifest_path = orphan / "manifest.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        if any(orphan.iterdir()):
            raise RuntimeError("pending orphan is nonempty before manifest publication")
        _atomic_create_json(manifest_path, manifest)
        if fault_injector is not None:
            fault_injector("after-manifest")
    else:
        stored_manifest = _read_json_object(
            manifest_path, label="pending recovery manifest"
        )
        if stored_manifest != manifest:
            raise RuntimeError("pending recovery manifest changed")
    _complete_pending_recovery(
        output,
        anchor,
        orphan,
        manifest,
        fault_injector=fault_injector,
    )
    return _load_recovery_events(output, anchor)


def reverify_raw_files(
    samples: list[dict[str, Any]], *, hash_cache: StatHashCache | None = None
) -> list[dict[str, Any]]:
    cache = hash_cache if hash_cache is not None else StatHashCache()
    checked: list[dict[str, Any]] = []
    for sample in samples:
        label = f"{sample.get('config_id')}-{sample.get('sample_kind')}-{sample.get('iteration')}"
        files: dict[str, dict[str, str]] = {}
        for path_name, digest_name in (
            ("result_file", "result_sha256"),
            ("stdout_log", "stdout_sha256"),
            ("stderr_log", "stderr_sha256"),
        ):
            path = pathlib.Path(str(sample.get(path_name, "")))
            digest = sample.get(digest_name)
            try:
                record = _regular_evidence_record(
                    path, label=path_name, hash_cache=cache
                )
            except RuntimeError as error:
                raise RuntimeError(f"{label} {path_name} changed before publication") from error
            if not isinstance(digest, str) or record["sha256"] != digest:
                raise RuntimeError(f"{label} {path_name} changed before publication")
            files[path_name] = {"path": record["path"], "sha256": digest}
        raw = json.loads(pathlib.Path(sample["result_file"]).read_text(encoding="utf-8"))
        if canonical_sha256(raw) != canonical_sha256(sample.get("result")):
            raise RuntimeError(f"{label} in-memory/raw evidence disagrees")
        checked.append({"label": label, "nonce": sample.get("nonce"), "files": files})
    return checked


def _assert_stored_gates_match_raw_recomputation(
    sample: dict[str, Any], recomputed: dict[str, Any], *, label: str
) -> None:
    for gate_name in (
        "validation",
        "correctness",
        "receipt_gate",
        "process_environment_gate",
        "material_gradient_gate",
    ):
        if sample.get(gate_name) != recomputed.get(gate_name):
            raise RuntimeError(
                f"{label} stored {gate_name} differs from raw recomputation"
            )


def _require_exact_recomputed_launch_environment(
    stored: Any, expected: dict[str, str], *, label: str
) -> dict[str, str]:
    canonical_expected = dict(sorted(expected.items()))
    if stored != canonical_expected:
        raise RuntimeError(
            f"{label} stored launch environment differs from fixed recomputation"
        )
    return canonical_expected


def reverify_final_closure(
    *,
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    expected_receipt_id: str,
    expected_snapshot: dict[str, Any],
    expected_producer_sha256: str,
    topology_evidence: dict[str, Any],
    samples: list[dict[str, Any]],
    expected_output: pathlib.Path,
    expected_run_id: str,
    anchor_evidence: dict[str, Any],
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    cache = hash_cache if hash_cache is not None else StatHashCache()
    expected_output = expected_output.resolve()
    if not expected_run_id or not re.fullmatch(r"[0-9a-f]{32}", expected_run_id):
        raise RuntimeError("CPU tuning run id is invalid")
    verified_anchor = _verify_anchor_evidence_on_disk(
        expected_output, anchor_evidence, hash_cache=cache
    )
    if verified_anchor["record"].get("run_id") != expected_run_id:
        raise RuntimeError("CPU tuning run anchor has the wrong run id")
    receipt = verify_build_receipt(receipt_path, repo, hash_cache=cache)
    if receipt.get("receipt_id") != expected_receipt_id:
        raise RuntimeError("build receipt changed during CPU tuning")
    contract = _MPI_RUNNER.validate_build_receipt_contract(
        receipt, repo, hash_cache=cache
    )
    if (
        source_snapshot(repo, hash_cache=cache) != expected_snapshot
        or receipt.get("source_end") != expected_snapshot
    ):
        raise RuntimeError("source changed before CPU tuning publication")
    producer = repo / "scripts" / "benchmark-adjoint.py"
    if cache.digest(producer) != expected_producer_sha256:
        raise RuntimeError("CPU benchmark producer changed during tuning")
    topology_path = pathlib.Path(topology_evidence["path"])
    topology_file = _regular_evidence_record(
        topology_path,
        output=expected_output,
        expected_parent=expected_output,
        label="CPU topology evidence",
        hash_cache=cache,
    )
    if (
        topology_file["path"] != str(_absolute_lexical(expected_output / "topology.json"))
        or topology_file["sha256"] != topology_evidence["sha256"]
    ):
        raise RuntimeError("CPU topology evidence changed during tuning")
    if json.loads(topology_path.read_text(encoding="utf-8")) != topology_evidence["record"]:
        raise RuntimeError("CPU topology raw/in-memory evidence disagrees")
    runtime = _MPI_RUNNER._receipt_runtime(receipt, repo)
    fontconfig_file = _qualification_fontconfig(
        receipt, repo, runtime, hash_cache=cache
    )
    power_load_state_end = capture_power_load_state(
        topology_evidence["record"]["normalized"]["available_logical_cpus"]
    )
    power_load_comparability = validate_power_load_comparability(
        topology_evidence["record"]["power_load_state_start"],
        power_load_state_end,
    )
    checked = reverify_raw_files(samples, hash_cache=cache)
    raw_records = [
        json.loads(pathlib.Path(sample["result_file"]).read_text(encoding="utf-8"))
        for sample in samples
    ]
    reference = next(
        raw
        for sample, raw in zip(samples, raw_records)
        if sample["config_id"] == "1x1"
        and sample["sample_kind"] == "measured"
        and sample["iteration"] == 0
    )
    recomputed_samples: list[dict[str, Any]] = []
    for sample, raw in zip(samples, raw_records):
        label = f"{sample['config_id']}-{sample['sample_kind']}-{sample['iteration']}"
        config = _config_by_id(sample["config_id"])
        result_path = pathlib.Path(sample["result_file"]).resolve()
        if result_path.parent != expected_output:
            raise RuntimeError(f"{label} raw result is outside the expected output")
        validate_qualification_binding(
            raw,
            nonce=sample["nonce"],
            run_id=expected_run_id,
            receipt_id=expected_receipt_id,
            snapshot_sha256=expected_snapshot["sha256"],
            producer_sha256=expected_producer_sha256,
            sample_kind=sample["sample_kind"],
            iteration=sample["iteration"],
            label=label,
        )
        expected_producer_command = producer_arguments(
            repo=repo,
            runtime=runtime,
            result_path=result_path,
            receipt_id=expected_receipt_id,
            snapshot_sha256=expected_snapshot["sha256"],
            producer_sha256=expected_producer_sha256,
            nonce=sample["nonce"],
            run_id=expected_run_id,
            sample_kind=sample["sample_kind"],
            iteration=sample["iteration"],
        )
        expected_launch_command = launch_arguments(
            runtime=runtime,
            config=config,
            producer_argv=expected_producer_command,
        )
        if (
            raw.get("producer", {}).get("command") != expected_producer_command
            or sample.get("producer_command") != expected_producer_command
            or sample.get("command") != expected_launch_command
        ):
            raise RuntimeError(f"{label} stored/launched command evidence differs")
        expected_environment = _child_environment(
            runtime,
            config=config,
            output=expected_output,
            run_id=expected_run_id,
            sample_kind=sample["sample_kind"],
            iteration=sample["iteration"],
            nonce=sample["nonce"],
            prepare_xdg_cache=False,
        )
        expected_environment = _require_exact_recomputed_launch_environment(
            sample.get("environment"), expected_environment, label=label
        )
        validation = validate_cpu_record(
            raw, config, topology_evidence["record"]["normalized"], label=label
        )
        _validate_process_wall_seconds(
            sample.get("process_seconds"), validation, label=label
        )
        receipt_gate = _MPI_RUNNER._validate_receipt_rank_runtimes(
            receipt, validation, hash_cache=cache
        )
        receipt_gate["pass"] = True
        environment_gate = _validate_process_environments(
            raw, expected_environment, label=label
        )
        if expected_environment.get("FONTCONFIG_FILE") != str(fontconfig_file):
            raise RuntimeError(f"{label} changed receipt-bound FONTCONFIG_FILE")
        stdout = pathlib.Path(sample["stdout_log"]).read_text(encoding="utf-8")
        material_gradient_gate = _material_gradient_gate(
            stdout, label="cpu-parallel"
        )
        correctness = compare_to_reference(reference, raw, label=label)
        recomputed = {
            "config_id": sample["config_id"],
            "sample_kind": sample["sample_kind"],
            "iteration": sample["iteration"],
            "nonce": sample["nonce"],
            "validation": validation,
            "correctness": correctness,
            "receipt_gate": receipt_gate,
            "process_environment_gate": environment_gate,
            "material_gradient_gate": material_gradient_gate,
        }
        _assert_stored_gates_match_raw_recomputation(
            sample, recomputed, label=label
        )
        recomputed_samples.append(recomputed)
    recomputed_matrix = validate_sample_matrix(recomputed_samples)
    recomputed_selection = select_fastest(recomputed_samples)
    checkpoint_samples, checkpoint_closure = _load_checkpoint_entries(
        repo=repo,
        output=expected_output,
        runtime=runtime,
        receipt=receipt,
        anchor=anchor_evidence,
        hash_cache=cache,
    )
    if checkpoint_closure["record"]["complete"] is not True:
        raise RuntimeError("CPU checkpoint journal is incomplete at publication")
    if len(checkpoint_samples) != len(samples):
        raise RuntimeError("CPU checkpoint/sample count differs at publication")
    for index, (stored, checkpointed) in enumerate(zip(samples, checkpoint_samples)):
        for name in (
            "config_id",
            "config",
            "sample_kind",
            "iteration",
            "nonce",
            "process_seconds",
            "result_file",
            "result_sha256",
            "stdout_log",
            "stdout_sha256",
            "stderr_log",
            "stderr_sha256",
            "command",
            "producer_command",
            "environment",
        ):
            if stored.get(name) != checkpointed.get(name):
                raise RuntimeError(
                    f"CPU checkpoint sample {index} changed field {name}"
                )
        if canonical_sha256(stored.get("result")) != canonical_sha256(
            checkpointed.get("result")
        ):
            raise RuntimeError(f"CPU checkpoint sample {index} raw result changed")
        _assert_stored_gates_match_raw_recomputation(
            stored, checkpointed, label=f"checkpoint[{index}]"
        )
    resume_closure = _load_resume_events(
        expected_output, anchor_evidence, hash_cache=cache
    )
    failure_closure = _load_failure_events(
        expected_output, anchor_evidence, hash_cache=cache
    )
    if failure_closure["record"]["terminal"]:
        raise RuntimeError("terminal failure journal cannot produce a COMPLETE report")
    recovery_closure = _load_recovery_events(
        expected_output, anchor_evidence, hash_cache=cache
    )
    if (
        recovery_closure["record"]["pending_orphan_count"] != 0
        or recovery_closure["record"]["pending_intent_count"] != 0
    ):
        raise RuntimeError("CPU recovery journal has an incomplete transaction")
    _verify_anchor_evidence_on_disk(
        expected_output, anchor_evidence, hash_cache=cache
    )
    return {
        "pass": True,
        "verified_at_utc": _utc_now(),
        "receipt_id": expected_receipt_id,
        "receipt_sha256": cache.digest(receipt_path),
        "source_snapshot": expected_snapshot,
        "contract": contract,
        "topology": {"path": str(topology_path), "sha256": topology_evidence["sha256"]},
        "samples": checked,
        "sample_matrix": recomputed_matrix,
        "selection": recomputed_selection,
        "checkpoint_closure": checkpoint_closure,
        "resume_closure": resume_closure,
        "failure_closure": failure_closure,
        "recovery_closure": recovery_closure,
        "power_load_state_end": power_load_state_end,
        "power_load_comparability": power_load_comparability,
    }


def _require_fresh_output(output: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name for path in output.iterdir() if path.name != RUN_LOCK_FILE
    )
    if unexpected:
        raise RuntimeError(
            "new CPU benchmark output is not empty; use a fresh directory or "
            "explicit --resume for a trusted checkpoint"
        )


def _reject_stale_publication_artifacts(output: pathlib.Path) -> None:
    present = [
        name
        for name in RECOVERABLE_PUBLICATION_ARTIFACTS
        if (output / name).exists() or (output / name).is_symlink()
    ]
    if present:
        raise RuntimeError(
            "resume output contains stale or already-published final artifacts: "
            + ", ".join(present)
        )


def _validate_resume_state(
    output: pathlib.Path,
    anchor: dict[str, Any],
    *,
    publication_recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_path = output / "state.json"
    _regular_evidence_record(
        state_path, output=output, expected_parent=output, label="resume state marker"
    )
    state = _read_json_object(state_path, label="CPU benchmark state")
    complete_recovery = (
        state.get("state") == "COMPLETE"
        and isinstance(publication_recovery, dict)
        and publication_recovery.get("state") == state
    )
    anchor_field = "run_anchor_sha256" if complete_recovery else "anchor_sha256"
    if (
        state.get("run_id") != anchor["record"]["run_id"]
        or state.get(anchor_field) != anchor["sha256"]
    ):
        raise RuntimeError("resume state marker has the wrong run identity")
    if state.get("state") not in ("FAILED", "RUNNING") and not complete_recovery:
        raise RuntimeError("resume state is neither trusted RUNNING nor FAILED")
    failure_closure = _load_failure_events(output, anchor)
    if failure_closure["record"]["terminal"]:
        raise NonResumableRun(
            "immutable terminal failure forbids resume for this receipt-bound run"
        )
    latest = (
        failure_closure["record"]["events"][-1]
        if failure_closure["record"]["events"]
        else None
    )
    if latest is not None:
        event = _read_json_object(
            pathlib.Path(latest["path"]), label="latest failure event"
        )
        structural = _structural_checkpoint_closure(output, anchor)
        if event.get("checkpoint") != structural:
            raise RuntimeError("latest failure does not close the current checkpoint prefix")
        if state.get("state") == "FAILED" and (
            state.get("failure_event") != latest
            or state.get("failure_closure") != failure_closure
        ):
            raise RuntimeError("FAILED state does not bind the latest failure event")
    elif state.get("state") == "FAILED":
        raise RuntimeError("FAILED state has no immutable failure event")
    return {
        "state": state,
        "failed": latest,
        "failure_closure": failure_closure,
        "publication_recovery": publication_recovery if complete_recovery else None,
    }


def _write_running_state(
    output: pathlib.Path,
    anchor: dict[str, Any],
    journal_closure: dict[str, Any],
    *,
    resumed: bool,
    prior_failure: dict[str, Any] | None = None,
) -> None:
    failure_closure = _load_failure_events(output, anchor)
    if failure_closure["record"]["terminal"]:
        raise NonResumableRun("terminal failure forbids RUNNING state publication")
    latest_failure = (
        failure_closure["record"]["events"][-1]
        if failure_closure["record"]["events"]
        else None
    )
    if prior_failure != latest_failure:
        raise RuntimeError("RUNNING state prior-failure binding is stale")
    atomic_write_json(
        output / "state.json",
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state": "RUNNING",
            "run_id": anchor["record"]["run_id"],
            "anchor_sha256": anchor["sha256"],
            "checkpoint_entry_count": journal_closure["record"]["entry_count"],
            "checkpoint_head_sha256": journal_closure["record"]["head_sha256"],
            "resumed": resumed,
            "prior_failure": latest_failure,
            "failure_closure": failure_closure,
        },
    )


def mark_failed(
    output: pathlib.Path,
    run_id: str,
    error: BaseException,
    *,
    anchor: dict[str, Any] | None = None,
    journal_closure: dict[str, Any] | None = None,
) -> None:
    if not isinstance(anchor, dict):
        raise RuntimeError("cannot journal a failure without a trusted run anchor")
    if run_id != anchor["record"]["run_id"]:
        raise RuntimeError("failure run identity differs from its anchor")
    published = _publish_failure_event(output, anchor, error)
    structural = _structural_checkpoint_closure(output, anchor)
    if journal_closure is not None and journal_closure != structural:
        raise RuntimeError("failure context has a stale checkpoint closure")
    failure = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state": "FAILED",
        "run_id": run_id,
        "profile": dict(PROFILE),
        "anchor_sha256": anchor["sha256"],
        "checkpoint": structural,
        "failure_event": published["event"],
        "failure_closure": published["closure"],
    }
    atomic_write_json(output / "state.json", failure)


def validate_publication_report(
    report: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    authoritative_recomputation: dict[str, Any],
) -> dict[str, Any]:
    """Require report summaries to equal raw-derived authoritative recomputation."""

    if authoritative_recomputation.get("pass") is not True:
        raise RuntimeError("authoritative raw recomputation did not pass")
    recomputed_matrix = authoritative_recomputation.get("sample_matrix")
    recomputed_selection = authoritative_recomputation.get("selection")
    required_sample_gates = (
        "validation",
        "correctness",
        "receipt_gate",
        "process_environment_gate",
        "material_gradient_gate",
    )
    if any(
        sample.get(gate_name, {}).get("pass") is not True
        for sample in samples
        for gate_name in required_sample_gates
    ):
        raise RuntimeError("CPU tuning report contains a failed sample gate")
    expected_gate = {
        "pass": True,
        "receipt_and_source": True,
        "topology": True,
        "sample_matrix": True,
        "world_backend_counters_affinity": True,
        "objective_and_full_gradient": True,
        "raw_evidence_integrity": True,
        "durable_checkpoint_journal": True,
        "durable_recovery_journal": True,
        "selection": True,
    }
    integrity = report.get("integrity_reverification")
    checkpoint_closure = authoritative_recomputation.get("checkpoint_closure")
    resume_closure = authoritative_recomputation.get("resume_closure")
    failure_closure = authoritative_recomputation.get("failure_closure")
    recovery_closure = authoritative_recomputation.get("recovery_closure")
    if (
        report.get("state") != "COMPLETE"
        or report.get("profile") != dict(PROFILE)
        or report.get("sample_matrix") != recomputed_matrix
        or report.get("selection") != recomputed_selection
        or report.get("checkpoint_closure") != checkpoint_closure
        or report.get("resume_closure") != resume_closure
        or report.get("failure_closure") != failure_closure
        or report.get("recovery_closure") != recovery_closure
        or report.get("gate") != expected_gate
        or not isinstance(integrity, dict)
        or integrity.get("pass") is not True
        or integrity.get("sample_matrix") != recomputed_matrix
        or integrity.get("selection") != recomputed_selection
        or integrity.get("checkpoint_closure") != checkpoint_closure
        or integrity.get("resume_closure") != resume_closure
        or integrity.get("failure_closure") != failure_closure
        or integrity.get("recovery_closure") != recovery_closure
    ):
        raise RuntimeError("CPU tuning report failed publication recomputation")
    return {
        "pass": True,
        "sample_matrix": recomputed_matrix,
        "selection": recomputed_selection,
        "checkpoint_closure": checkpoint_closure,
        "resume_closure": resume_closure,
        "failure_closure": failure_closure,
        "recovery_closure": recovery_closure,
        "gate": expected_gate,
    }


def _validate_complete_marker(
    output: pathlib.Path,
    expected: dict[str, Any],
    *,
    require_state: bool,
    hash_cache: StatHashCache | None = None,
) -> None:
    marker_path = output / "COMPLETE"
    _regular_evidence_record(
        marker_path,
        output=output,
        expected_parent=output,
        label="CPU COMPLETE marker",
        hash_cache=hash_cache,
    )
    marker = _read_json_object(marker_path, label="CPU COMPLETE marker")
    if marker != expected:
        raise RuntimeError("CPU COMPLETE marker differs from publication closure")
    report_path = pathlib.Path(str(marker.get("report", "")))
    markdown_path = pathlib.Path(str(marker.get("report_markdown", "")))
    if (
        _absolute_lexical(report_path) != _absolute_lexical(output / "report.json")
        or _absolute_lexical(markdown_path) != _absolute_lexical(output / "report.md")
    ):
        raise RuntimeError("CPU COMPLETE marker report closure is invalid")
    report_file = _regular_evidence_record(
        report_path,
        output=output,
        expected_parent=output,
        label="CPU report",
        hash_cache=hash_cache,
    )
    markdown_file = _regular_evidence_record(
        markdown_path,
        output=output,
        expected_parent=output,
        label="CPU report markdown",
        hash_cache=hash_cache,
    )
    if (
        marker.get("report_sha256") != report_file["sha256"]
        or marker.get("report_markdown_sha256") != markdown_file["sha256"]
    ):
        raise RuntimeError("CPU COMPLETE marker report closure is invalid")
    if require_state:
        _regular_evidence_record(
            output / "state.json",
            output=output,
            expected_parent=output,
            label="CPU complete state",
            hash_cache=hash_cache,
        )
        state = _read_json_object(output / "state.json", label="CPU complete state")
        if state != expected:
            raise RuntimeError("CPU complete state differs from COMPLETE marker")


def _summarize_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized_samples: list[dict[str, Any]] = []
    for sample in samples:
        result = sample.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("cannot summarize a CPU sample without its raw result")
        result_payload = result.get("result")
        if not isinstance(result_payload, dict):
            raise RuntimeError("cannot summarize a CPU sample with a malformed result")
        gradient = result_payload.get("gradient")
        summarized = {name: value for name, value in sample.items() if name != "result"}
        summarized["raw_result_schema"] = result.get("schema_version")
        summarized["raw_gradient_sha256"] = canonical_float64_sha256(gradient)
        summarized_samples.append(summarized)
    return summarized_samples


def _verify_report_failure_consistency(
    output: pathlib.Path,
    report: dict[str, Any],
    anchor: dict[str, Any],
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any] | None:
    continuity = report.get("execution_continuity")
    if not isinstance(continuity, dict):
        raise RuntimeError("CPU report execution-continuity evidence is absent")
    closure = _load_failure_events(
        output, anchor, hash_cache=hash_cache
    )
    latest = closure["record"]["events"][-1] if closure["record"]["events"] else None
    if closure["record"]["terminal"]:
        raise RuntimeError("CPU COMPLETE report has a terminal failure history")
    if (
        continuity.get("failure_closure") != closure
        or report.get("failure_closure") != closure
        or continuity.get("prior_failure") != latest
    ):
        raise RuntimeError("CPU report contradicts the immutable failure journal")
    return latest


def _verify_published_complete(
    *,
    repo: pathlib.Path,
    output: pathlib.Path,
    receipt_path: pathlib.Path,
    receipt: dict[str, Any],
    anchor_evidence: dict[str, Any],
    require_complete_state: bool = True,
    marker_name: str = "COMPLETE",
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    cache = hash_cache if hash_cache is not None else StatHashCache()
    if marker_name not in ("COMPLETE", "PUBLISHING.json"):
        raise RuntimeError("unsupported publication marker")
    anchor = _verify_anchor_evidence_on_disk(
        output, anchor_evidence, hash_cache=cache
    )
    topology_path = pathlib.Path(anchor["record"]["topology"]["path"])
    topology_file = _regular_evidence_record(
        topology_path,
        output=output,
        expected_parent=output,
        label="published topology evidence",
        hash_cache=cache,
    )
    if topology_file["sha256"] != anchor["record"]["topology"]["sha256"]:
        raise RuntimeError("published topology evidence changed")
    topology_record = _read_json_object(topology_path, label="published topology evidence")
    topology_evidence = {
        "path": topology_file["path"],
        "sha256": topology_file["sha256"],
        "record": topology_record,
    }
    runtime = _MPI_RUNNER._receipt_runtime(receipt, repo)
    samples, checkpoint_closure = _load_checkpoint_entries(
        repo=repo,
        output=output,
        runtime=runtime,
        receipt=receipt,
        anchor=anchor_evidence,
        hash_cache=cache,
    )
    if checkpoint_closure["record"]["complete"] is not True:
        raise RuntimeError("published COMPLETE has an incomplete checkpoint journal")
    report_file = _regular_evidence_record(
        output / "report.json",
        output=output,
        expected_parent=output,
        label="published report",
        hash_cache=cache,
    )
    markdown_file = _regular_evidence_record(
        output / "report.md",
        output=output,
        expected_parent=output,
        label="published report markdown",
        hash_cache=cache,
    )
    report = _read_json_object(output / "report.json", label="published report")
    if report.get("samples") != _summarize_samples(samples):
        raise RuntimeError("published report samples differ from all raw/log evidence")
    authoritative = reverify_final_closure(
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=receipt["receipt_id"],
        expected_snapshot=anchor["record"]["source_snapshot"],
        expected_producer_sha256=anchor["record"]["producer"]["sha256"],
        topology_evidence=topology_evidence,
        samples=samples,
        expected_output=output,
        expected_run_id=anchor["record"]["run_id"],
        anchor_evidence=anchor_evidence,
        hash_cache=cache,
    )
    validate_publication_report(
        report, samples, authoritative_recomputation=authoritative
    )
    final_publish = report.get("final_publish_reverification")
    if (
        not isinstance(final_publish, dict)
        or final_publish.get("pass") is not True
        or final_publish.get("checkpoint_closure") != authoritative["checkpoint_closure"]
        or final_publish.get("resume_closure") != authoritative["resume_closure"]
        or final_publish.get("failure_closure") != authoritative["failure_closure"]
        or final_publish.get("recovery_closure") != authoritative["recovery_closure"]
    ):
        raise RuntimeError("published report lacks its final closure")
    markdown_binding = report.get("report_markdown")
    if markdown_binding != {
        "path": markdown_file["path"],
        "sha256": markdown_file["sha256"],
    }:
        raise RuntimeError("published report markdown binding changed")
    _verify_report_failure_consistency(
        output, report, anchor_evidence, hash_cache=cache
    )
    marker_path = output / marker_name
    _regular_evidence_record(
        marker_path,
        output=output,
        expected_parent=output,
        label="CPU publication marker",
        hash_cache=cache,
    )
    marker = _read_json_object(marker_path, label="CPU publication marker")
    expected_marker_fields = {
        "run_anchor": anchor["path"],
        "run_anchor_sha256": anchor["sha256"],
        "checkpoint_entry_count": authoritative["checkpoint_closure"]["record"]["entry_count"],
        "checkpoint_head_sha256": authoritative["checkpoint_closure"]["record"]["head_sha256"],
        "checkpoint_closure_sha256": authoritative["checkpoint_closure"]["sha256"],
        "resume_event_count": authoritative["resume_closure"]["record"]["event_count"],
        "resume_closure_sha256": authoritative["resume_closure"]["sha256"],
        "failure_event_count": authoritative["failure_closure"]["record"]["event_count"],
        "failure_closure_sha256": authoritative["failure_closure"]["sha256"],
        "recovery_event_count": authoritative["recovery_closure"]["record"]["event_count"],
        "recovery_closure_sha256": authoritative["recovery_closure"]["sha256"],
        "report": report_file["path"],
        "report_sha256": report_file["sha256"],
        "report_markdown": markdown_file["path"],
        "report_markdown_sha256": markdown_file["sha256"],
    }
    if any(marker.get(name) != value for name, value in expected_marker_fields.items()):
        raise RuntimeError("COMPLETE marker contradicts the authoritative publication closure")
    if marker_name == "COMPLETE":
        publishing_file = _regular_evidence_record(
            output / "PUBLISHING.json",
            output=output,
            expected_parent=output,
            label="PUBLISHING marker",
            hash_cache=cache,
        )
        if marker.get("publishing_marker") != publishing_file:
            raise RuntimeError("COMPLETE marker does not close PUBLISHING evidence")
        _validate_complete_marker(
            output,
            marker,
            require_state=require_complete_state,
            hash_cache=cache,
        )
    else:
        if marker.get("state") != "PUBLISHING":
            raise RuntimeError("provisional publication marker has the wrong state")
    return {
        "pass": True,
        "report": report,
        "authoritative_reverification": authoritative,
        "marker": marker,
    }


def publish_complete(
    output: pathlib.Path,
    run_id: str,
    report: dict[str, Any],
    markdown: str,
    *,
    capability: object,
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    expected_receipt_id: str,
    expected_snapshot: dict[str, Any],
    expected_producer_sha256: str,
    topology_evidence: dict[str, Any],
    samples: list[dict[str, Any]],
    anchor_evidence: dict[str, Any],
    hash_cache: StatHashCache | None = None,
) -> None:
    _assert_fixed_profile()
    cache = hash_cache if hash_cache is not None else StatHashCache()
    if capability is not _PUBLISH_CAPABILITY:
        raise RuntimeError("CPU tuning publisher capability is absent")
    final_reverification = reverify_final_closure(
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=expected_receipt_id,
        expected_snapshot=expected_snapshot,
        expected_producer_sha256=expected_producer_sha256,
        topology_evidence=topology_evidence,
        samples=samples,
        expected_output=output,
        expected_run_id=run_id,
        anchor_evidence=anchor_evidence,
        hash_cache=cache,
    )
    validate_publication_report(
        report,
        samples,
        authoritative_recomputation=final_reverification,
    )
    if report.get("run_id") != run_id:
        raise RuntimeError("refusing to publish an incomplete CPU tuning run")
    report["final_publish_reverification"] = final_reverification
    markdown_path = output / "report.md"
    report_path = output / "report.json"
    atomic_write_text(markdown_path, markdown)
    report["report_markdown"] = {
        "path": str(markdown_path.resolve()),
        "sha256": cache.digest(markdown_path),
    }
    atomic_write_json(report_path, report)
    marker_payload = {
        "schema_version": 1,
        "state": "PUBLISHING",
        "run_id": run_id,
        "profile_id": PROFILE_ID,
        "report": str(report_path.resolve()),
        "report_sha256": cache.digest(report_path),
        "report_markdown": str(markdown_path.resolve()),
        "report_markdown_sha256": cache.digest(markdown_path),
        "run_anchor": anchor_evidence["path"],
        "run_anchor_sha256": anchor_evidence["sha256"],
        "checkpoint_entry_count": final_reverification["checkpoint_closure"]["record"]["entry_count"],
        "checkpoint_head_sha256": final_reverification["checkpoint_closure"]["record"]["head_sha256"],
        "checkpoint_closure_sha256": final_reverification["checkpoint_closure"]["sha256"],
        "resume_event_count": final_reverification["resume_closure"]["record"]["event_count"],
        "resume_closure_sha256": final_reverification["resume_closure"]["sha256"],
        "failure_event_count": final_reverification["failure_closure"]["record"]["event_count"],
        "failure_closure_sha256": final_reverification["failure_closure"]["sha256"],
        "recovery_event_count": final_reverification["recovery_closure"]["record"]["event_count"],
        "recovery_closure_sha256": final_reverification["recovery_closure"]["sha256"],
    }
    _atomic_create_json(output / "PUBLISHING.json", marker_payload)
    verified_receipt = verify_build_receipt(
        receipt_path, repo, hash_cache=cache
    )
    _verify_published_complete(
        repo=repo,
        output=output,
        receipt_path=receipt_path,
        receipt=verified_receipt,
        anchor_evidence=anchor_evidence,
        require_complete_state=False,
        marker_name="PUBLISHING.json",
        hash_cache=cache,
    )
    publishing_file = _regular_evidence_record(
        output / "PUBLISHING.json",
        output=output,
        expected_parent=output,
        label="PUBLISHING marker",
        hash_cache=cache,
    )
    marker = {
        **marker_payload,
        "state": "COMPLETE",
        "publishing_marker": publishing_file,
    }
    _atomic_create_json(output / "COMPLETE", marker)
    _validate_complete_marker(
        output, marker, require_state=False, hash_cache=cache
    )
    atomic_write_json(output / "state.json", marker)
    _validate_complete_marker(
        output, marker, require_state=True, hash_cache=cache
    )
    try:
        _verify_published_complete(
            repo=repo,
            output=output,
            receipt_path=receipt_path,
            receipt=verified_receipt,
            anchor_evidence=anchor_evidence,
            hash_cache=cache,
        )
    except Exception:
        # A pre-COMPLETE deep verification already passed.  Independently
        # recompute the authoritative closure without calling the failed
        # post-verifier (guards verifier implementation regressions).
        try:
            fallback = reverify_final_closure(
                repo=repo,
                receipt_path=receipt_path,
                expected_receipt_id=expected_receipt_id,
                expected_snapshot=expected_snapshot,
                expected_producer_sha256=expected_producer_sha256,
                topology_evidence=topology_evidence,
                samples=samples,
                expected_output=output,
                expected_run_id=run_id,
                anchor_evidence=anchor_evidence,
                hash_cache=cache,
            )
            on_disk_report = _read_json_object(
                output / "report.json", label="post-publication report"
            )
            if on_disk_report.get("samples") != _summarize_samples(samples):
                raise RuntimeError("post-publication report samples changed")
            validate_publication_report(
                on_disk_report,
                samples,
                authoritative_recomputation=fallback,
            )
            _verify_report_failure_consistency(
                output,
                on_disk_report,
                anchor_evidence,
                hash_cache=cache,
            )
            _verify_anchor_evidence_on_disk(
                output, anchor_evidence, hash_cache=cache
            )
            _validate_complete_marker(
                output, marker, require_state=True, hash_cache=cache
            )
            current_publishing_file = _regular_evidence_record(
                output / "PUBLISHING.json",
                output=output,
                expected_parent=output,
                label="post-publication PUBLISHING marker",
                hash_cache=cache,
            )
            current_publishing_payload = _read_json_object(
                output / "PUBLISHING.json",
                label="post-publication PUBLISHING marker",
            )
            if (
                marker.get("publishing_marker") != current_publishing_file
                or current_publishing_file != publishing_file
                or current_publishing_payload != marker_payload
            ):
                raise RuntimeError("COMPLETE/PUBLISHING closure changed")
        except Exception as fallback_error:
            _recover_interrupted_evidence(
                output,
                anchor_evidence,
                len(sample_schedule()),
                allow_complete_publication_triad=True,
            )
            raise RuntimeError(
                "post-publication verification and independent fallback failed; "
                "final markers were quarantined"
            ) from fallback_error
        # The independently recomputed closure is authoritative.  Keep the
        # valid COMPLETE/state and do not let main append a contradictory
        # FAILED marker solely because the primary post-verifier regressed.


def _empty_journal_closure() -> dict[str, Any]:
    record = {
        "entry_count": 0,
        "expected_entry_count": len(sample_schedule()),
        "complete": False,
        "head_sha256": None,
        "entries": [],
    }
    return {"pass": True, "record": record, "sha256": canonical_sha256(record)}


def _extend_journal_closure(
    closure: dict[str, Any], entry_evidence: dict[str, Any]
) -> dict[str, Any]:
    old = closure["record"]
    index = old["entry_count"]
    if entry_evidence["record"].get("index") != index:
        raise RuntimeError("checkpoint journal extension is out of order")
    entries = [*old["entries"], {
        "index": index,
        "path": entry_evidence["path"],
        "sha256": entry_evidence["sha256"],
    }]
    record = {
        "entry_count": len(entries),
        "expected_entry_count": len(sample_schedule()),
        "complete": len(entries) == len(sample_schedule()),
        "head_sha256": entry_evidence["sha256"],
        "entries": entries,
    }
    return {"pass": True, "record": record, "sha256": canonical_sha256(record)}


def execute(
    args: argparse.Namespace,
    repo: pathlib.Path,
    failure_context: dict[str, Any] | None = None,
) -> int:
    _assert_fixed_profile()
    hash_cache = StatHashCache()
    if getattr(args, "host_specific_interim", False) is not True:
        raise RuntimeError(
            "portable/final CPU publication is disabled until the topology-derived "
            "v2 schedule and checkpoint/recovery migration is complete; use "
            "--host-specific-interim only for the explicitly non-portable legacy run"
        )
    snapshot_before = source_snapshot(repo, hash_cache=hash_cache)
    receipt_path = args.build_receipt.resolve()
    receipt = verify_build_receipt(
        receipt_path, repo, hash_cache=hash_cache
    )
    if receipt.get("build_kind") != "cuda-mpi-python-fp32":
        raise RuntimeError("CPU tuning requires the CUDA+MPI+Python FP32 receipt")
    receipt_contract = _MPI_RUNNER.validate_build_receipt_contract(
        receipt, repo, hash_cache=hash_cache
    )
    if receipt.get("source_end") != snapshot_before:
        raise RuntimeError("build receipt does not match the current source snapshot")
    runtime = _MPI_RUNNER._receipt_runtime(receipt, repo)
    _qualification_fontconfig(receipt, repo, runtime, hash_cache=hash_cache)
    producer_path = repo / "scripts" / "benchmark-adjoint.py"
    producer_sha256 = hash_cache.digest(producer_path)
    prior_failure: dict[str, Any] | None = None
    if args.resume:
        anchor_evidence = _validate_run_anchor(
            repo=repo,
            output=args.output,
            receipt_path=receipt_path,
            receipt=receipt,
            snapshot=snapshot_before,
            producer_sha256=producer_sha256,
            hash_cache=hash_cache,
        )
        _validate_legacy_interim_topology(
            anchor_evidence["topology_evidence"]["record"]["normalized"]
        )
        publication_recovery = _publication_recovery_context(
            args.output, anchor_evidence
        )
        publication_present = [
            (args.output / name).exists() or (args.output / name).is_symlink()
            for name in RECOVERABLE_PUBLICATION_ARTIFACTS
        ]
        recover_complete_publication_triad = publication_recovery is not None
        if all(publication_present) and publication_recovery is None:
            try:
                _verify_published_complete(
                    repo=repo,
                    output=args.output,
                    receipt_path=receipt_path,
                    receipt=receipt,
                    anchor_evidence=anchor_evidence,
                    hash_cache=hash_cache,
                )
            except RuntimeError:
                _verify_published_complete(
                    repo=repo,
                    output=args.output,
                    receipt_path=receipt_path,
                    receipt=receipt,
                    anchor_evidence=anchor_evidence,
                    require_complete_state=False,
                    hash_cache=hash_cache,
                )
                recover_complete_publication_triad = True
            else:
                raise AlreadyComplete(
                    "explicit --resume found an already valid COMPLETE publication"
                )
        state = _validate_resume_state(
            args.output,
            anchor_evidence,
            publication_recovery=publication_recovery,
        )
        prior_failure = state["failed"]
        inspected_prefix = _inspect_checkpoint_chain(
            args.output, anchor_evidence, hash_cache=hash_cache
        )
        recovered = _recover_interrupted_evidence(
            args.output,
            anchor_evidence,
            len(inspected_prefix),
            allow_complete_publication_triad=recover_complete_publication_triad,
        )
        if publication_recovery is not None:
            if (
                recovered["record"]["pending_intent_count"] != 0
                or recovered["record"]["pending_orphan_count"] != 0
                or any(
                    (args.output / name).exists()
                    or (args.output / name).is_symlink()
                    for name in RECOVERABLE_PUBLICATION_ARTIFACTS
                )
            ):
                raise RuntimeError("COMPLETE-state publication recovery did not close")
        topology_evidence = anchor_evidence["topology_evidence"]
        run_id = anchor_evidence["record"]["run_id"]
        journal_closure = _structural_checkpoint_closure(
            args.output, anchor_evidence, hash_cache=hash_cache
        )
        if failure_context is not None:
            failure_context.update(
                {
                    "initialized": True,
                    "run_id": run_id,
                    "anchor": anchor_evidence,
                    "journal_closure": journal_closure,
                }
            )
        samples, journal_closure = _load_checkpoint_entries(
            repo=repo,
            output=args.output,
            runtime=runtime,
            receipt=receipt,
            anchor=anchor_evidence,
            hash_cache=hash_cache,
        )
        if failure_context is not None:
            failure_context["journal_closure"] = journal_closure
        _publish_resume_event(args.output, anchor_evidence, journal_closure)
        _write_running_state(
            args.output,
            anchor_evidence,
            journal_closure,
            resumed=True,
            prior_failure=prior_failure,
        )
    else:
        _require_fresh_output(args.output)
        run_id = uuid.uuid4().hex
        topology_evidence = capture_topology(repo, args.output)
        _validate_legacy_interim_topology(topology_evidence["record"]["normalized"])
        anchor_payload = _anchor_payload(
            repo=repo,
            output=args.output,
            run_id=run_id,
            receipt_path=receipt_path,
            receipt=receipt,
            snapshot=snapshot_before,
            producer_sha256=producer_sha256,
            topology_evidence=topology_evidence,
            hash_cache=hash_cache,
        )
        anchor_evidence = _publish_run_anchor(args.output, anchor_payload)
        anchor_evidence["topology_evidence"] = topology_evidence
        samples = []
        journal_closure = _empty_journal_closure()
        _write_running_state(
            args.output,
            anchor_evidence,
            journal_closure,
            resumed=False,
        )
    if failure_context is not None:
        failure_context.update(
            {
                "initialized": True,
                "run_id": run_id,
                "anchor": anchor_evidence,
                "journal_closure": journal_closure,
            }
        )
    reference: dict[str, Any] | None = (
        samples[REFERENCE_SCHEDULE_INDEX]["result"]
        if len(samples) > REFERENCE_SCHEDULE_INDEX
        else None
    )
    for index in range(len(samples), len(sample_schedule())):
        config_id, sample_kind, iteration = sample_schedule()[index]
        config = _config_by_id(config_id)
        sample = _run_raw_sample(
            repo=repo,
            output=args.output,
            runtime=runtime,
            receipt=receipt,
            snapshot_sha256=snapshot_before["sha256"],
            producer_sha256=producer_sha256,
            topology=topology_evidence["record"]["normalized"],
            run_id=run_id,
            config=config,
            sample_kind=sample_kind,
            iteration=iteration,
            hash_cache=hash_cache,
        )
        if index == REFERENCE_SCHEDULE_INDEX:
            reference = sample["result"]
        if reference is not None:
            sample["correctness"] = compare_to_reference(
                reference, sample["result"], label=f"{config_id}-{sample_kind}-{iteration}"
            )
        checkpoint_entry = _make_checkpoint_entry(
            output=args.output,
            anchor=anchor_evidence,
            index=index,
            previous_sha256=journal_closure["record"]["head_sha256"],
            sample=sample,
            hash_cache=hash_cache,
        )
        entry_evidence = _publish_checkpoint_entry(
            args.output, checkpoint_entry, hash_cache=hash_cache
        )
        journal_closure = _extend_journal_closure(journal_closure, entry_evidence)
        samples.append(sample)
        if failure_context is not None:
            failure_context["journal_closure"] = journal_closure
        _write_running_state(
            args.output,
            anchor_evidence,
            journal_closure,
            resumed=args.resume,
            prior_failure=prior_failure,
        )
    if reference is None:
        raise RuntimeError("measured 1x1 correctness reference is absent")
    # Warmups precede the measured reference; validate them now as well.
    for sample in samples:
        if "correctness" not in sample:
            sample["correctness"] = compare_to_reference(
                reference,
                sample["result"],
                label=f"{sample['config_id']}-{sample['sample_kind']}-{sample['iteration']}",
            )
    matrix = validate_sample_matrix(samples)
    selection = select_fastest(samples)
    snapshot_after = source_snapshot(repo, hash_cache=hash_cache)
    if snapshot_after != snapshot_before or snapshot_after != receipt.get("source_end"):
        raise RuntimeError("source changed during CPU tuning")
    integrity = reverify_final_closure(
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=receipt["receipt_id"],
        expected_snapshot=snapshot_before,
        expected_producer_sha256=producer_sha256,
        topology_evidence=topology_evidence,
        samples=samples,
        expected_output=args.output,
        expected_run_id=run_id,
        anchor_evidence=anchor_evidence,
        hash_cache=hash_cache,
    )
    summarized_samples: list[dict[str, Any]] = []
    for sample in samples:
        summarized = {name: value for name, value in sample.items() if name != "result"}
        summarized["raw_result_schema"] = sample["result"].get("schema_version")
        summarized["raw_gradient_sha256"] = canonical_float64_sha256(
            sample["result"]["result"]["gradient"]
        )
        summarized_samples.append(summarized)
    gate = {
        "pass": True,
        "receipt_and_source": True,
        "topology": True,
        "sample_matrix": True,
        "world_backend_counters_affinity": True,
        "objective_and_full_gradient": True,
        "raw_evidence_integrity": True,
        "durable_checkpoint_journal": True,
        "durable_recovery_journal": True,
        "selection": True,
    }
    start_power = topology_evidence["record"]["power_load_state_start"]
    cpu_zero_power = start_power["per_cpu"]["0"]["cpufreq"]
    power_policy = {
        name: record.get("text", "").strip()
        if record.get("status") == "ok"
        else record.get("status")
        for name, record in cpu_zero_power.items()
        if name in POWER_CONFIGURATION_NAMES
    }
    host_power_scope = {
        "host_specific": True,
        "cpu_model": topology_evidence["record"]["summary"].get("Model name"),
        "power_policy_cpu0": power_policy,
        "power_load_comparability": integrity["power_load_comparability"],
    }
    report = {
        "schema_version": 1,
        "state": "COMPLETE",
        "publication_scope": "host-specific-interim-not-portable-final",
        "run_id": run_id,
        "profile": dict(PROFILE),
        "build_receipt": {
            "path": str(receipt_path),
            "sha256": hash_cache.digest(receipt_path),
            "receipt_id": receipt["receipt_id"],
            "build_input_id": receipt["build_input_id"],
            "artifact_set_id": receipt["artifact_set_id"],
            "qualification_contract": receipt_contract,
        },
        "topology": topology_evidence,
        "matrix_binding": anchor_evidence["record"]["matrix"],
        "host_power_scope": host_power_scope,
        "sample_matrix": matrix,
        "samples": summarized_samples,
        "selection": selection,
        "checkpoint_closure": integrity["checkpoint_closure"],
        "resume_closure": integrity["resume_closure"],
        "failure_closure": integrity["failure_closure"],
        "recovery_closure": integrity["recovery_closure"],
        "execution_continuity": {
            "same_boot_resume_only": True,
            "resume_event_count": integrity["resume_closure"]["record"]["event_count"],
            "failure_event_count": integrity["failure_closure"]["record"]["event_count"],
            "recovery_event_count": integrity["recovery_closure"]["record"]["event_count"],
            "fixed_schedule_continued_without_replaying_committed_samples": True,
            "journal_io_excluded_from_sample_process_seconds": True,
            "prior_failure": prior_failure,
            "failure_closure": integrity["failure_closure"],
        },
        "integrity_reverification": integrity,
        "gate": gate,
        "postprocessor": {
            "git_head": git_output(repo, "rev-parse", "HEAD").strip(),
            "git_status_porcelain": git_output(
                repo, "status", "--porcelain=v1", "--untracked-files=all"
            ).splitlines(),
            "source_snapshot": snapshot_after,
            "source_sha256": {
                name: hash_cache.digest(repo / name) for name in RUNNER_SOURCES
            },
            "python": {
                "path": str(pathlib.Path(sys.executable).resolve()),
                "sha256": hash_cache.digest(pathlib.Path(sys.executable).resolve()),
                "version": sys.version,
            },
            "command": [
                str(pathlib.Path(sys.executable).resolve()),
                str(pathlib.Path(__file__).resolve()),
                *sys.argv[1:],
            ],
        },
    }
    selected = selection["fastest_passing_physical"]
    canonical_physical = selection["canonical_full_physical"]
    power_comparison = integrity["power_load_comparability"]
    markdown = (
        "# gpmeep host-specific interim CPU parallel tuning\n\n"
        "- State: **COMPLETE**\n"
        "- Publication scope: **INTERIM / NOT A PORTABLE FINAL CPU-vs-GPU HEADLINE**\n"
        f"- Profile: `{PROFILE_ID}`\n"
        f"- Matrix: `{', '.join(config.config_id for config in CONFIGS)}` "
        f"with {PROFILE['warmups']} warmup and {PROFILE['repeats']} measured runs each\n"
        f"- Fastest passing physical configuration: `{selection['selected_config_id']}`\n"
        f"- Fastest physical median workload wall time: `{selected['workload_wall_median_seconds']:.6f}` s\n"
        f"- Canonical full-physical configuration: `{canonical_physical['config_id']}` "
        f"at `{canonical_physical['workload_wall_median_seconds']:.6f}` s\n"
        "- `1x1` is diagnostic-only and SMT is a separate non-headline lane.\n"
        f"- CPU model: `{host_power_scope['cpu_model']}`\n"
        f"- Power policy (CPU 0): `{json.dumps(power_policy, sort_keys=True)}`\n"
        f"- Start/end 1-minute load: `{power_comparison['start_load']['load_1m']:.3f}` / "
        f"`{power_comparison['end_load']['load_1m']:.3f}`\n"
        "- Scope: **host/power-policy/load-specific**; thermal throttle counters did not increase.\n"
        f"- Receipt: `{receipt['receipt_id']}`\n"
        f"- Durable checkpoint entries/head: `{integrity['checkpoint_closure']['record']['entry_count']}` / "
        f"`{integrity['checkpoint_closure']['record']['head_sha256']}`\n"
        f"- Same-boot resume events: `{integrity['resume_closure']['record']['event_count']}`\n"
        f"- Interruption recovery events: `{integrity['recovery_closure']['record']['event_count']}`\n"
        "\n## Per-configuration timing\n\n"
        + _markdown_timing_table(selection)
    )
    publish_complete(
        args.output,
        run_id,
        report,
        markdown,
        capability=_PUBLISH_CAPABILITY,
        repo=repo,
        receipt_path=receipt_path,
        expected_receipt_id=receipt["receipt_id"],
        expected_snapshot=snapshot_before,
        expected_producer_sha256=producer_sha256,
        topology_evidence=topology_evidence,
        samples=samples,
        anchor_evidence=anchor_evidence,
        hash_cache=hash_cache,
    )
    print(markdown, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = pathlib.Path(__file__).resolve().parent.parent
    failure_context: dict[str, Any] = {}
    try:
        args.output = _trusted_output_path(args.output, create=True)
        with output_lock(args.output):
            try:
                return execute(args, repo, failure_context)
            except BaseException as error:
                if failure_context.get("initialized") is True:
                    mark_failed(
                        args.output,
                        failure_context["run_id"],
                        error,
                        anchor=failure_context["anchor"],
                        journal_closure=failure_context["journal_closure"],
                    )
                print(f"error: {error}", file=sys.stderr)
                return 1
    except BaseException as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
