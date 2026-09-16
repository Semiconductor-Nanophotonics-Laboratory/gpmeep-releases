#!/usr/bin/env python3
"""Receipt-bound 1/2-GPU A/B qualification for fused DFT monitor batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import statistics
import subprocess
import sys
import time
import traceback
import types
from typing import Any


SCRIPTS = pathlib.Path(__file__).resolve().parent
LOADED_MODULE_SOURCES: dict[str, tuple[pathlib.Path, bytes, str]] = {}


def load_source_module(name: str, filename: str) -> types.ModuleType:
    path = SCRIPTS / filename
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__gpmeep_loaded_source_sha256__ = digest
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    LOADED_MODULE_SOURCES[filename] = (path, source, digest)
    return module


QUALIFICATION = load_source_module(
    "gpmeep_qualification_contract_for_provenance",
    "gpmeep_qualification_contract.py",
)
PROVENANCE = load_source_module("gpmeep_provenance", "gpmeep_provenance.py")
PROVENANCE._load_qualification_contract_from_source = lambda: QUALIFICATION
BASE = load_source_module(
    "gpmeep_dft_phase_benchmark_base", "benchmark-dft-phase-sharing.py"
)


INNER_BOOTSTRAP = r'''
import hashlib
import os
import stat
import sys

path = os.path.realpath(sys.argv[1])
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(path, flags)
try:
    before = os.fstat(fd)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
            before.st_size <= 0 or before.st_size > 4 * 1024 * 1024):
        raise RuntimeError("runner source is not one bounded regular file")
    chunks = []
    remaining = before.st_size
    while remaining:
        block = os.read(fd, min(1024 * 1024, remaining))
        if not block:
            raise RuntimeError("short read while snapshotting runner source")
        chunks.append(block)
        remaining -= len(block)
    if os.read(fd, 1):
        raise RuntimeError("runner source grew during snapshot")
    after = os.fstat(fd)
finally:
    os.close(fd)
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_nlink, value.st_size, value.st_mode,
    value.st_mtime_ns, value.st_ctime_ns,
)
if identity(before) != identity(after):
    raise RuntimeError("runner source changed during snapshot")
source = b"".join(chunks)
digest = hashlib.sha256(source).hexdigest()
sys.argv = [path, *sys.argv[2:]]
namespace = {
    "__name__": "__main__",
    "__file__": path,
    "__package__": None,
    "__gpmeep_internal_snapshot__": True,
    "__gpmeep_loaded_source_bytes__": source,
    "__gpmeep_loaded_source_sha256__": digest,
}
exec(compile(source, path, "exec"), namespace)
'''


def reexec_immutable_source() -> None:
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            INNER_BOOTSTRAP,
            str(pathlib.Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        dict(os.environ),
    )

DFT_WORKLOAD_PREFIX = "gpmeep-dft-workload-v1:"
DFT_WARMUP_PLAN_PREFIX = BASE.DFT_WARMUP_PLAN_PREFIX
PROTOCOL_ID = "m27-dft-multi-monitor-authoritative-v1"
MANIFEST_NAME = "evidence-manifest.json"
TERMINAL_NAMES = frozenset({"COMPLETE.json", "FAILED.json"})
OBSERVABLES = BASE.OBSERVABLES
ALLOWED_PREFIXES = BASE.WORKER_GPMEEP_PREFIXES + (
    DFT_WORKLOAD_PREFIX,
)
MULTI_FIELDS = (
    "automatic_checks",
    "automatic_selected",
    "automatic_rejected",
    "forced_batches",
    "batched_updates",
    "unbatched_updates",
    "plan_uploads",
    "plan_reuses",
    "metadata_host_to_device_bytes",
)
DFT_FIELDS = (
    "batch_calls",
    "submitted_updates",
    "phase_preparation_launches",
    "phase_reuses",
    "update_kernel_launches",
    "maximum_batch_size",
)
DFT_WARMUP_PLAN_FIELDS = (
    "metadata_host_to_device_bytes",
    "plan_reuses",
    "plan_uploads",
)
EAGER_FIELDS = (
    "receive_requests",
    "receive_start_calls",
    "send_requests",
    "send_start_calls",
)
PINGPONG_FIELDS = (
    "secondary_selections",
    "selections",
    "warmup_secondary_allocations",
)
GRAPH_FIELDS = ("creations", "launches")
DEVICE_FIELDS = {
    "rank",
    "ordinal",
    "uuid",
    "name",
    "compute_major",
    "compute_minor",
    "compatible",
}
BENCHMARK_FIELDS = {
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
PHASE_CALLS = {
    "curl",
    "update_eh",
    "source",
    "boundary",
    "dft",
    "polarization",
}
WORKLOADS = (
    {
        "name": "launch-dominated",
        "pixels": 32,
        "warmup_steps": 40,
        "steps": 2400,
        "monitors": 64,
        "frequencies": 2,
        "pairs": 8,
        "minimum_measured_seconds": 2.0,
        "minimum_median_speedup_percent": 10.0,
        "maximum_t90_upper_time_ratio": 0.95,
    },
    {
        "name": "frequency-heavy",
        "pixels": 96,
        "warmup_steps": 12,
        "steps": 120,
        "monitors": 8,
        "frequencies": 32,
        "pairs": 6,
        "minimum_measured_seconds": 2.0,
        "minimum_median_speedup_percent": 3.0,
        "maximum_t90_upper_time_ratio": 1.02,
    },
)


def filesystem_identity(
    path: pathlib.Path, root: pathlib.Path, *, include_sha256: bool = True
) -> dict[str, Any]:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(f"sealed artifact is not one regular link: {path}")
    result: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "link_count": metadata.st_nlink,
        "size_bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }
    if include_sha256:
        result["sha256"] = BASE.sha256_file(path)
    return result


def verify_execution_handle(binding: dict[str, Any], descriptor: int) -> None:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        block = os.pread(
            descriptor, min(1024 * 1024, before.st_size - offset), offset
        )
        if not block:
            break
        digest.update(block)
        offset += len(block)
    after = os.fstat(descriptor)
    fields = (
        "st_dev", "st_ino", "st_nlink", "st_size", "st_mode",
        "st_mtime_ns", "st_ctime_ns",
    )
    actual = {
        "execution_path": f"/proc/{os.getpid()}/fd/{descriptor}",
        "device": after.st_dev,
        "inode": after.st_ino,
        "link_count": after.st_nlink,
        "size_bytes": after.st_size,
        "mode": stat.S_IMODE(after.st_mode),
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "sha256": digest.hexdigest(),
    }
    if offset != after.st_size or any(
        getattr(before, field) != getattr(after, field) for field in fields
    ):
        raise RuntimeError("sealed execution image changed while hashing")
    if any(binding.get(key) != value for key, value in actual.items()):
        raise RuntimeError(f"sealed execution image changed: {binding.get('label')}")


def create_execution_handle(
    snapshot: pathlib.Path, root: pathlib.Path, label: str
) -> tuple[int, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(snapshot, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"invalid execution snapshot: {snapshot}")
        binding = {
            "label": label,
            "snapshot": filesystem_identity(snapshot, root),
            "execution_model": "persistent-read-only-no-follow-proc-fd",
            "execution_path": f"/proc/{os.getpid()}/fd/{descriptor}",
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "link_count": metadata.st_nlink,
            "size_bytes": metadata.st_size,
            "mode": stat.S_IMODE(metadata.st_mode),
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
            "sha256": BASE.sha256_file(snapshot),
        }
        verify_execution_handle(binding, descriptor)
        return descriptor, binding
    except BaseException:
        os.close(descriptor)
        raise


def verify_execution_handles(args: argparse.Namespace) -> None:
    if set(args.execution_bindings) != set(args.execution_descriptors):
        raise RuntimeError("sealed execution handle set is incomplete")
    for label in sorted(args.execution_bindings):
        verify_execution_handle(
            args.execution_bindings[label], args.execution_descriptors[label]
        )


def write_evidence_manifest(result_dir: pathlib.Path) -> pathlib.Path:
    path = result_dir / MANIFEST_NAME
    excluded = {MANIFEST_NAME, *TERMINAL_NAMES}
    files = sorted(
        item for item in result_dir.rglob("*")
        if item.is_file()
        and item.name not in excluded
        and "home" not in item.relative_to(result_dir).parts
    )
    BASE.atomic_json(
        path,
        {"schema_version": 1,
         "files": [BASE.file_record(item, result_dir) for item in files]},
    )
    return path


def terminal_present(result_dir: pathlib.Path) -> bool:
    return any((result_dir / name).exists() for name in TERMINAL_NAMES)


def finalize_success(
    result_dir: pathlib.Path, receipt_id: str
) -> None:
    if terminal_present(result_dir):
        raise RuntimeError("refusing to create a second terminal marker")
    manifest = write_evidence_manifest(result_dir)
    BASE.atomic_json(
        result_dir / "COMPLETE.json",
        {
            "schema_version": 1,
            "state": "COMPLETE",
            "receipt_id": receipt_id,
            "manifest": BASE.file_record(manifest, result_dir),
            "summary": BASE.file_record(result_dir / "summary.json", result_dir),
            "report": BASE.file_record(result_dir / "report.md", result_dir),
        },
    )


def finalize_failure(result_dir: pathlib.Path, payload: dict[str, Any]) -> None:
    if terminal_present(result_dir):
        return
    terminal = dict(payload)
    try:
        manifest = write_evidence_manifest(result_dir)
        terminal["manifest"] = BASE.file_record(manifest, result_dir)
    except Exception as error:
        terminal["manifest_error"] = f"{type(error).__name__}: {error}"
    BASE.atomic_json(result_dir / "FAILED.json", terminal)


def reject_unknown_markers(output: str) -> None:
    for line in output.splitlines():
        start = 0
        while True:
            position = line.find("gpmeep-", start)
            if position < 0:
                break
            if not any(
                line.startswith(prefix, position) for prefix in ALLOWED_PREFIXES
            ):
                raise RuntimeError(
                    f"unknown gpmeep worker marker: {line[:160]}"
                )
            start = position + len("gpmeep-")


def exact_nonnegative_integers(
    record: dict[str, Any], fields: tuple[str, ...], label: str
) -> None:
    if set(record) != set(fields) or any(
        type(record.get(field)) is not int or record[field] < 0
        for field in fields
    ):
        raise RuntimeError(f"{label} has an invalid exact integer schema")


def physical_observables(record: dict[str, Any]) -> dict[str, Any]:
    result = {name: record.get(name) for name in OBSERVABLES}
    if any(value is None for value in result.values()):
        raise RuntimeError("benchmark record lacks a physical observable")
    for name in (
        "cut_probe_l2",
        "ez_l2",
        "ez_weighted_checksum",
        "energy",
        "dft_norm",
        "remote_cut_l2",
    ):
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise RuntimeError(f"physical observable {name!r} is non-finite")
    if not (
        float(result["cut_probe_l2"]) > 0.0
        and float(result["ez_l2"]) > 0.0
        and float(result["ez_weighted_checksum"]) > 0.0
        and float(result["energy"]) > 0.0
        and float(result["dft_norm"]) > 0.0
    ):
        raise RuntimeError("benchmark produced a zero physical signal")
    return result


def finite_number(value: Any, *, positive: bool = False) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and (not positive or float(value) > 0.0)
    )


def validate_benchmark_contract(
    benchmark: dict[str, Any], ranks: int, workload: dict[str, Any]
) -> None:
    expected_cells = workload["pixels"] ** 3
    expected_mcells = (
        expected_cells * workload["steps"] / float(benchmark.get("seconds", 0.0))
        / 1.0e6
        if finite_number(benchmark.get("seconds"), positive=True)
        else math.nan
    )
    if (
        set(benchmark) != BENCHMARK_FIELDS
        or type(benchmark.get("schema_version")) is not int
        or benchmark["schema_version"] != 4
        or type(benchmark.get("mpi_ranks")) is not int
        or benchmark["mpi_ranks"] != ranks
        or type(benchmark.get("pixels")) is not int
        or benchmark["pixels"] != workload["pixels"]
        or type(benchmark.get("cells")) is not int
        or benchmark["cells"] != expected_cells
        or type(benchmark.get("warmup_steps")) is not int
        or benchmark["warmup_steps"] != workload["warmup_steps"]
        or type(benchmark.get("steps")) is not int
        or benchmark["steps"] != workload["steps"]
        or type(benchmark.get("loop_tile_base_db")) is not int
        or benchmark["loop_tile_base_db"] != 128
        or type(benchmark.get("bfast")) is not bool
        or benchmark["bfast"]
        or type(benchmark.get("source_enabled")) is not bool
        or not benchmark["source_enabled"]
        or type(benchmark.get("overlap_material")) is not bool
        or benchmark["overlap_material"]
        or not finite_number(benchmark.get("seconds"), positive=True)
        or float(benchmark["seconds"]) < workload["minimum_measured_seconds"]
        or not finite_number(benchmark.get("mcells_per_second"), positive=True)
        or not math.isclose(
            float(benchmark["mcells_per_second"]),
            expected_mcells,
            rel_tol=2.0e-15,
            abs_tol=0.0,
        )
        or benchmark.get("completion_policy") != "waitsome"
        or benchmark.get("initial_condition") != "trigonometric-v1"
        or type(benchmark.get("cpu_calls")) is not int
        or benchmark["cpu_calls"] != 0
        or type(benchmark.get("cuda_calls")) is not int
        or benchmark["cuda_calls"] <= 0
    ):
        raise RuntimeError("benchmark record differs from the fixed workload")
    counter_fields = (
        "h2d_bytes",
        "d2h_bytes",
        "mpi_messages",
        "mpi_scalars",
        "cuda_aware_bytes",
        "pinned_bytes",
        "remote_cut_scalars",
    )
    if any(
        type(benchmark.get(name)) is not int or benchmark[name] < 0
        for name in counter_fields
    ):
        raise RuntimeError("benchmark byte/message accounting is invalid")
    probes = benchmark.get("cut_probe_values")
    if (
        not isinstance(probes, list)
        or len(probes) != 18
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not finite_number(value) for value in pair)
            for pair in probes
        )
    ):
        raise RuntimeError("benchmark cut-probe vector is invalid")


def validate_sample(
    condition: str,
    topology: dict[str, Any],
    workload: dict[str, Any],
    output: str,
) -> dict[str, Any]:
    if condition not in {"forced", "disabled", "automatic"}:
        raise RuntimeError("unknown DFT batching condition")
    reject_unknown_markers(output)
    benchmark = BASE.exactly_one(output, BASE.BENCHMARK_PREFIX, "benchmark")
    initialization = BASE.validate_initialization_markers(output, benchmark)
    dft = BASE.exactly_one(output, BASE.DFT_PREFIX, "DFT phase sharing")
    multi = BASE.exactly_one(
        output, BASE.DFT_MULTI_PREFIX, "multi-monitor DFT batching"
    )
    warmup_plan = BASE.exactly_one(
        output, DFT_WARMUP_PLAN_PREFIX, "DFT warmup plan"
    )
    eager = BASE.exactly_one(output, BASE.EAGER_PREFIX, "eager MPI")
    pingpong = BASE.exactly_one(
        output, BASE.PINGPONG_PREFIX, "receive ping-pong"
    )
    graph = BASE.exactly_one(output, BASE.GRAPH_PREFIX, "boundary graph")
    identity = BASE.exactly_one(
        output, DFT_WORKLOAD_PREFIX, "DFT workload identity"
    )
    exact_nonnegative_integers(dft, DFT_FIELDS, "DFT record")
    exact_nonnegative_integers(
        multi, MULTI_FIELDS, "multi-monitor DFT record"
    )
    exact_nonnegative_integers(
        warmup_plan, DFT_WARMUP_PLAN_FIELDS, "DFT warmup-plan record"
    )
    exact_nonnegative_integers(eager, EAGER_FIELDS, "eager MPI record")
    exact_nonnegative_integers(
        pingpong, PINGPONG_FIELDS, "receive ping-pong record"
    )
    exact_nonnegative_integers(graph, GRAPH_FIELDS, "boundary graph record")
    if set(identity) != {"components", "frequencies", "monitors"} or any(
        type(identity.get(name)) is not int or identity[name] <= 0
        for name in identity
    ):
        raise RuntimeError("DFT workload identity has an invalid exact schema")
    expected_identity = {
        "components": 2,
        "frequencies": workload["frequencies"],
        "monitors": workload["monitors"],
    }
    if identity != expected_identity:
        raise RuntimeError("DFT workload identity differs from the protocol")

    ranks = topology["ranks"]
    devices = BASE.records(output, BASE.DEVICE_PREFIX)
    if len(devices) != ranks or any(
        not isinstance(record, dict) or set(record) != DEVICE_FIELDS
        for record in devices
    ):
        raise RuntimeError("CUDA device records have an invalid exact schema")
    if (
        {record["rank"] for record in devices} != set(range(ranks))
        or {record["ordinal"] for record in devices} != set(range(ranks))
        or len({record["uuid"] for record in devices}) != ranks
        or any(
            type(record["rank"]) is not int
            or type(record["ordinal"]) is not int
            or type(record["compute_major"]) is not int
            or type(record["compute_minor"]) is not int
            or type(record["compatible"]) is not bool
            or not record["compatible"]
            or not isinstance(record["uuid"], str)
            or not record["uuid"]
            or not isinstance(record["name"], str)
            or not record["name"]
            for record in devices
        )
    ):
        raise RuntimeError("sample did not prove the requested CUDA topology")
    processes = BASE.records(output, BASE.PROCESS_PREFIX)
    if (
        len(processes) != ranks
        or any(
            not isinstance(record, dict)
            or set(record) != {"pid", "rank"}
            or type(record["pid"]) is not int
            or record["pid"] <= 0
            or type(record["rank"]) is not int
            for record in processes
        )
        or {record["rank"] for record in processes} != set(range(ranks))
        or len({record["pid"] for record in processes}) != ranks
    ):
        raise RuntimeError("sample did not prove distinct rank process IDs")

    validate_benchmark_contract(benchmark, ranks, workload)
    phase_calls = benchmark["phase_calls"]
    if (
        not isinstance(phase_calls, dict)
        or set(phase_calls) != PHASE_CALLS
        or any(
            not isinstance(record, dict)
            or set(record) != {"cpu_calls", "cuda_calls", "expectation"}
            or type(record["cpu_calls"]) is not int
            or record["cpu_calls"] < 0
            or type(record["cuda_calls"]) is not int
            or record["cuda_calls"] < 0
            or record["expectation"]
            not in {"cuda_required", "not_applicable"}
            for record in phase_calls.values()
        )
        or sum(record["cpu_calls"] for record in phase_calls.values())
        != benchmark["cpu_calls"]
        or sum(record["cuda_calls"] for record in phase_calls.values())
        != benchmark["cuda_calls"]
        or any(record["cpu_calls"] != 0 for record in phase_calls.values())
        or any(
            phase_calls[name]["cuda_calls"] <= 0
            or phase_calls[name]["expectation"] != "cuda_required"
            for name in {"curl", "update_eh", "source", "boundary", "dft"}
        )
        or phase_calls["polarization"]
        != {"cpu_calls": 0, "cuda_calls": 0, "expectation": "not_applicable"}
    ):
        raise RuntimeError("benchmark phase-call coverage is inconsistent")
    if ranks == 2 and (
        benchmark.get("requested_transport") != "cuda-aware"
        or benchmark.get("selected_transport") != "cuda-aware"
        or benchmark.get("pinned_bytes") != 0
        or benchmark.get("cuda_aware_bytes", 0) <= 0
        or benchmark.get("mpi_messages", 0) <= 0
        or benchmark.get("mpi_scalars", 0) <= 0
        or benchmark.get("remote_cut_scalars", 0) <= 0
        or not finite_number(benchmark.get("remote_cut_l2"), positive=True)
    ):
        raise RuntimeError("two-GPU sample did not use exclusive CUDA-aware MPI")
    if ranks == 2 and (
        min(eager.values()) <= 0
        or min(pingpong.values()) <= 0
        or graph["creations"] != 0
        or graph["launches"] <= 0
    ):
        raise RuntimeError("two-GPU sample lacks required communication overlap")
    if ranks == 1 and (
        benchmark.get("requested_transport") != "auto"
        or benchmark.get("selected_transport") != "none"
        or benchmark.get("pinned_bytes") != 0
        or benchmark.get("cuda_aware_bytes") != 0
        or benchmark.get("mpi_messages") != 0
        or benchmark.get("mpi_scalars") != 0
        or benchmark.get("remote_cut_scalars") != 0
        or benchmark.get("remote_cut_l2") != 0.0
    ):
        raise RuntimeError("one-GPU sample unexpectedly reported MPI transport")
    observables = physical_observables(benchmark)

    if (
        dft["batch_calls"] < workload["steps"]
        or dft["submitted_updates"]
        < dft["batch_calls"] * workload["monitors"]
        or dft["maximum_batch_size"] < workload["monitors"]
        or dft["phase_preparation_launches"] + dft["phase_reuses"]
        != dft["submitted_updates"]
        or dft["phase_reuses"] <= 0
    ):
        raise RuntimeError("workload did not create reusable multi-monitor DFT work")
    if (
        multi["automatic_selected"] + multi["automatic_rejected"]
        != multi["automatic_checks"]
        or multi["batched_updates"] + multi["unbatched_updates"]
        != dft["submitted_updates"]
        or multi["plan_uploads"] + multi["plan_reuses"]
        != multi["automatic_selected"] + multi["forced_batches"]
        or (multi["plan_uploads"] == 0)
        != (multi["metadata_host_to_device_bytes"] == 0)
        or dft["update_kernel_launches"]
        != multi["automatic_selected"]
        + multi["forced_batches"]
        + multi["unbatched_updates"]
    ):
        raise RuntimeError("multi-monitor DFT accounting is inconsistent")
    if condition == "forced":
        if not (
            multi["automatic_checks"] == 0
            and multi["automatic_selected"] == 0
            and multi["automatic_rejected"] == 0
            and multi["forced_batches"] == dft["batch_calls"]
            and multi["batched_updates"] == dft["submitted_updates"]
            and multi["unbatched_updates"] == 0
            and multi["plan_uploads"] == 0
            and multi["plan_reuses"] == dft["batch_calls"]
            and multi["metadata_host_to_device_bytes"] == 0
            and dft["update_kernel_launches"] < dft["submitted_updates"]
            and warmup_plan["plan_uploads"] > 0
            and warmup_plan["plan_reuses"] > 0
            and warmup_plan["metadata_host_to_device_bytes"] > 0
        ):
            raise RuntimeError("forced sample did not reuse a fused DFT plan")
    elif condition == "disabled":
        if not (
            multi["automatic_checks"] == 0
            and multi["automatic_selected"] == 0
            and multi["automatic_rejected"] == 0
            and multi["forced_batches"] == 0
            and multi["batched_updates"] == 0
            and multi["unbatched_updates"] == dft["submitted_updates"]
            and multi["plan_uploads"] == 0
            and multi["plan_reuses"] == 0
            and dft["update_kernel_launches"] == dft["submitted_updates"]
            and all(value == 0 for value in warmup_plan.values())
        ):
            raise RuntimeError("disabled sample reported fused DFT execution")
    else:
        if not (
            multi["automatic_checks"] == dft["batch_calls"]
            and multi["automatic_selected"] == 0
            and multi["automatic_rejected"] == multi["automatic_checks"]
            and multi["forced_batches"] == 0
            and multi["batched_updates"] == 0
            and multi["unbatched_updates"] == dft["submitted_updates"]
            and multi["plan_uploads"] == 0
            and multi["plan_reuses"] == 0
            and dft["update_kernel_launches"] == dft["submitted_updates"]
            and all(value == 0 for value in warmup_plan.values())
        ):
            raise RuntimeError("automatic sample did not fail closed")
    return {
        "benchmark": benchmark,
        "dft": dft,
        "dft_multi": multi,
        "dft_warmup_plan": warmup_plan,
        "communication": {
            "eager": eager,
            "pingpong": pingpong,
            "graph": graph,
        },
        "devices": devices,
        "processes": processes,
        "identity": identity,
        "initialization": initialization,
        "observables": observables,
    }


def condition_environment(condition: str) -> dict[str, str]:
    if condition == "forced":
        return {
            "MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH": "1",
            "MEEP_GPU_EXPECT_MULTI_MONITOR_DFT_BATCH": "1",
        }
    if condition == "disabled":
        return {
            "MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH": "1",
            "MEEP_GPU_EXPECT_NO_MULTI_MONITOR_DFT_BATCH": "1",
        }
    if condition == "automatic":
        return {"MEEP_GPU_EXPECT_NO_MULTI_MONITOR_DFT_BATCH": "1"}
    raise RuntimeError("unknown DFT batching condition")


def run_sample(
    args: argparse.Namespace,
    topology: dict[str, Any],
    workload: dict[str, Any],
    condition: str,
    ordinal: int,
    base_environment: dict[str, str],
) -> dict[str, Any]:
    stem = (
        f"{topology['name']}-{workload['name']}-"
        f"{ordinal:03d}-{condition}"
    )
    before = BASE.gpu_snapshot()
    BASE.require_idle(before)
    environment = dict(base_environment)
    environment.update(
        {
            "MEEP_GPU_MULTI_PIXELS": str(workload["pixels"]),
            "MEEP_GPU_MULTI_WARMUP_STEPS": str(workload["warmup_steps"]),
            "MEEP_GPU_MULTI_STEPS": str(workload["steps"]),
            "MEEP_GPU_MULTI_DFT_MONITORS": str(workload["monitors"]),
            "MEEP_GPU_MULTI_DFT_FREQUENCIES": str(workload["frequencies"]),
            "MEEP_GPU_EMIT_DFT_WORKLOAD": "1",
            "MEEP_GPU_EMIT_PROCESS_ID": "1",
            "MEEP_GPU_EXPECT_DFT_PHASE_SHARING": "1",
            "CUDA_VISIBLE_DEVICES": topology["visible_devices"],
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
    )
    if topology["ranks"] == 2:
        environment.update(
            {
                "MEEP_GPU_MPI_TRANSPORT": "cuda-aware",
                "MEEP_GPU_EXPECT_BOUNDARY_PHASE_GRAPH": "1",
                "MEEP_GPU_EXPECT_EAGER_MPI": "1",
                "MEEP_GPU_EXPECT_RECEIVE_PINGPONG": "1",
            }
        )
    else:
        environment["MEEP_GPU_MPI_TRANSPORT"] = "auto"
    environment.update(condition_environment(condition))
    argv = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=10s",
        f"{args.timeout}s",
        str(args.execution_mpiexec),
        "--host",
        f"localhost:{topology['ranks']}",
        "--bind-to",
        "none",
        "-np",
        str(topology["ranks"]),
        "/usr/bin/env",
        f"LD_PRELOAD={args.execution_library}",
        str(args.execution_executable),
    ]
    verify_execution_handles(args)
    started = BASE.utc_now()
    wall_start = time.perf_counter()
    result = subprocess.run(
        argv,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_seconds = time.perf_counter() - wall_start
    verify_execution_handles(args)
    raw_path = args.result_dir / f"{stem}.log"
    BASE.atomic_text(raw_path, result.stdout)
    after: dict[str, str] | None = None
    parsed: dict[str, Any] | None = None
    error: str | None = None
    try:
        after = BASE.gpu_snapshot()
        BASE.require_idle(after)
        if result.returncode:
            raise RuntimeError(f"sample exited with status {result.returncode}")
        parsed = validate_sample(condition, topology, workload, result.stdout)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    metadata = {
        "schema_version": 1,
        "status": "PASS" if error is None else "FAIL",
        "topology": topology["name"],
        "workload": workload["name"],
        "condition": condition,
        "ordinal": ordinal,
        "started_at_utc": started,
        "finished_at_utc": BASE.utc_now(),
        "command": argv,
        "persistent_execution_images": args.execution_bindings,
        "selected_environment": dict(sorted(environment.items())),
        "external_process_wall_seconds": wall_seconds,
        "returncode": result.returncode,
        "before": before,
        "after": after,
        "parsed": parsed,
        "raw_log": raw_path.name,
        "raw_log_sha256": BASE.sha256_file(raw_path),
        "error": error,
    }
    BASE.atomic_json(args.result_dir / f"{stem}.json", metadata)
    if error is not None:
        raise RuntimeError(f"invalid sample {stem}: {error}; see {raw_path}")
    return metadata


def summarize_pairs(
    topology: dict[str, Any],
    workload: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    forced_times = [float(pair["forced"]["parsed"]["benchmark"]["seconds"])
                    for pair in pairs]
    disabled_times = [float(pair["disabled"]["parsed"]["benchmark"]["seconds"])
                      for pair in pairs]
    ratios = [forced / disabled for forced, disabled
              in zip(forced_times, disabled_times)]
    speedups = [(disabled / forced - 1.0) * 100.0
                for forced, disabled in zip(forced_times, disabled_times)]
    interval = BASE.paired_log_interval(ratios)
    median_speedup = statistics.median(speedups)
    gates = [
        {
            "name": "minimum_median_paired_speedup_percent",
            "threshold": workload["minimum_median_speedup_percent"],
            "observed": median_speedup,
            "pass": median_speedup
            >= workload["minimum_median_speedup_percent"],
        }
    ]
    if workload["maximum_t90_upper_time_ratio"] is not None:
        gates.append(
            {
                "name": "maximum_t90_upper_forced_over_disabled_ratio",
                "threshold": workload["maximum_t90_upper_time_ratio"],
                "observed": interval["t90_upper_time_ratio"],
                "pass": interval["t90_upper_time_ratio"]
                <= workload["maximum_t90_upper_time_ratio"],
            }
        )
    return {
        "topology": topology["name"],
        "workload": workload["name"],
        "pair_speedup_percent": speedups,
        "median_paired_speedup_percent": median_speedup,
        "forced_seconds": forced_times,
        "disabled_seconds": disabled_times,
        "geometric_mean_forced_over_disabled_time_ratio": interval[
            "geometric_time_ratio_on_over_off"
        ],
        "t90_lower_forced_over_disabled_time_ratio": interval[
            "t90_lower_time_ratio"
        ],
        "t90_upper_forced_over_disabled_time_ratio": interval[
            "t90_upper_time_ratio"
        ],
        "ratio_of_median_times_forced_over_disabled": (
            statistics.median(forced_times)
            / statistics.median(disabled_times)
        ),
        "gates": gates,
        "pass": all(gate["pass"] for gate in gates),
    }


def update_device_catalog(
    catalog: dict[int, dict[str, Any]], devices: list[dict[str, Any]]
) -> None:
    for record in devices:
        ordinal = record["ordinal"]
        identity = {
            key: record[key]
            for key in (
                "uuid", "name", "compute_major", "compute_minor", "compatible"
            )
        }
        previous = catalog.setdefault(ordinal, identity)
        if previous != identity:
            raise RuntimeError(
                "CUDA device identity changed across authoritative samples"
            )


def execute_samples(
    args: argparse.Namespace,
    topologies: tuple[dict[str, Any], ...],
    base_environment: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    device_catalog: dict[int, dict[str, Any]] = {}
    ordinal = 0
    for topology in topologies:
        for workload in WORKLOADS:
            reference: dict[str, Any] | None = None
            pair_records: list[dict[str, Any]] = []
            for pair_index in range(1, workload["pairs"] + 1):
                order = (
                    ("forced", "disabled")
                    if pair_index % 2
                    else ("disabled", "forced")
                )
                pair: dict[str, Any] = {}
                for condition in order:
                    ordinal += 1
                    sample = run_sample(
                        args,
                        topology,
                        workload,
                        condition,
                        ordinal,
                        base_environment,
                    )
                    samples.append(sample)
                    update_device_catalog(
                        device_catalog, sample["parsed"]["devices"]
                    )
                    pair[condition] = sample
                    current = sample["parsed"]["observables"]
                    if reference is None:
                        reference = current
                    elif current != reference:
                        raise RuntimeError(
                            f"{topology['name']}/{workload['name']} changed "
                            "a physical observable"
                        )
                pair_records.append(pair)
            ordinal += 1
            automatic = run_sample(
                args,
                topology,
                workload,
                "automatic",
                ordinal,
                base_environment,
            )
            samples.append(automatic)
            update_device_catalog(
                device_catalog, automatic["parsed"]["devices"]
            )
            if automatic["parsed"]["observables"] != reference:
                raise RuntimeError(
                    f"{topology['name']}/{workload['name']} automatic "
                    "control changed a physical observable"
                )
            summary = summarize_pairs(topology, workload, pair_records)
            summary["automatic_control"] = {
                "ordinal": automatic["ordinal"],
                "counters": automatic["parsed"]["dft_multi"],
                "observables_exact": True,
            }
            summaries.append(summary)
    if set(device_catalog) != {0, 1}:
        raise RuntimeError("authoritative matrix did not retain both GPU identities")
    return samples, summaries


def stop_process_monitor(
    process: subprocess.Popen[bytes],
    stream: Any,
    path: pathlib.Path,
    expected_pid_to_gpu: dict[int, int],
) -> dict[str, Any]:
    alive_before_stop = process.poll() is None
    if alive_before_stop:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()
    pid_to_gpu: dict[int, int] = {}
    unrelated: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 3:
            unrelated.append(line)
            continue
        if columns[1] == "-":
            continue
        try:
            gpu = int(columns[0])
            pid = int(columns[1])
        except ValueError:
            unrelated.append(line)
            continue
        if (
            "gpu-mpi-per" not in columns[-1]
            or expected_pid_to_gpu.get(pid) != gpu
        ):
            unrelated.append(line)
            continue
        if pid in pid_to_gpu and pid_to_gpu[pid] != gpu:
            unrelated.append(line)
            continue
        pid_to_gpu[pid] = gpu
    expected_per_gpu = {
        gpu: sum(expected == gpu for expected in expected_pid_to_gpu.values())
        for gpu in sorted(set(expected_pid_to_gpu.values()))
    }
    observed_per_gpu = {
        gpu: sum(observed == gpu for observed in pid_to_gpu.values())
        for gpu in sorted(set(pid_to_gpu.values()))
    }
    record = {
        "schema_version": 1,
        "alive_before_runner_stop": alive_before_stop,
        "monitor_returncode": process.returncode,
        "expected_pids_per_gpu": {
            str(key): value for key, value in sorted(expected_per_gpu.items())
        },
        "expected_pid_to_gpu": {
            str(key): value for key, value in sorted(expected_pid_to_gpu.items())
        },
        "observed_pids_per_gpu": {
            str(key): value for key, value in sorted(observed_per_gpu.items())
        },
        "observed_benchmark_pids": sorted(pid_to_gpu),
        "unrelated_process_rows": unrelated,
        "raw": path.name,
        "raw_sha256": BASE.sha256_file(path),
    }
    record["pass"] = (
        alive_before_stop
        and process.returncode in {0, -15}
        and not unrelated
        and pid_to_gpu == expected_pid_to_gpu
        and observed_per_gpu == expected_per_gpu
    )
    return record


def expected_pid_gpu_map(
    samples: list[dict[str, Any]], gpu_index_by_uuid: dict[str, int]
) -> dict[int, int]:
    result: dict[int, int] = {}
    for sample in samples:
        parsed = sample.get("parsed")
        if not isinstance(parsed, dict):
            continue
        devices = parsed.get("devices")
        processes = parsed.get("processes")
        if not isinstance(devices, list) or not isinstance(processes, list):
            raise RuntimeError("sample lacks process/device evidence")
        devices_by_rank: dict[int, int] = {}
        for record in devices:
            uuid = record["uuid"]
            if uuid not in gpu_index_by_uuid:
                raise RuntimeError("worker CUDA UUID is absent from nvidia-smi")
            devices_by_rank[record["rank"]] = gpu_index_by_uuid[uuid]
        for process_record in processes:
            pid = process_record["pid"]
            rank = process_record["rank"]
            if pid in result:
                raise RuntimeError(
                    "worker PID was reused across authoritative DFT samples"
                )
            if rank not in devices_by_rank:
                raise RuntimeError("worker process has no CUDA device record")
            result[pid] = devices_by_rank[rank]
    return result


def physical_gpu_index_by_uuid() -> dict[str, int]:
    output = BASE.command_output(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    result: dict[str, int] = {}
    indices: set[int] = set()
    for line in output.splitlines():
        columns = [value.strip() for value in line.split(",")]
        if len(columns) != 2:
            raise RuntimeError("nvidia-smi GPU UUID map is malformed")
        try:
            index = int(columns[0])
        except ValueError as error:
            raise RuntimeError("nvidia-smi GPU index is malformed") from error
        uuid = columns[1]
        if index < 0 or not uuid or uuid in result or index in indices:
            raise RuntimeError("nvidia-smi GPU UUID map is not one-to-one")
        result[uuid] = index
        indices.add(index)
    if len(result) < 2:
        raise RuntimeError("M27 qualification requires at least two GPUs")
    return result


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# gpmeep fused multi-monitor DFT qualification",
        "",
        f"- State: **{summary['state']}**",
        f"- Receipt: `{summary['receipt_id']}`",
        f"- Samples: {summary['sample_count']}",
        f"- Physical observables exact: {summary['all_observables_exact']}",
        "",
        "| Topology | Workload | Median speedup | t90 upper forced/disabled | Pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in summary["results"]:
        lines.append(
            "| {topology} | {workload} | {speedup:.3f}% | {upper:.6f} | {passed} |".format(
                topology=record["topology"],
                workload=record["workload"],
                speedup=record["median_paired_speedup_percent"],
                upper=record[
                    "t90_upper_forced_over_disabled_time_ratio"
                ],
                passed="PASS" if record["pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "Automatic mode remained fail-closed in every topology/workload control.",
            "",
        ]
    )
    return "\n".join(lines)


def sealed_execution_preflight(
    args: argparse.Namespace, base_environment: dict[str, str]
) -> dict[str, Any]:
    verify_execution_handles(args)
    mpiexec_command = [
        "/usr/bin/env",
        "-i",
        *(f"{key}={base_environment[key]}" for key in sorted(base_environment)),
        str(args.execution_mpiexec),
        "--version",
    ]
    mpiexec = subprocess.run(
        mpiexec_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )
    verify_execution_handles(args)
    mpiexec_log = args.result_dir / "sealed-mpiexec-preflight.log"
    BASE.atomic_text(mpiexec_log, mpiexec.stdout)
    if mpiexec.returncode != 0 or "Open MPI" not in mpiexec.stdout:
        raise RuntimeError("persistent-FD mpiexec preflight failed")

    loader_environment = dict(base_environment)
    loader_environment["LD_PRELOAD"] = str(args.execution_library)
    loader_environment["LD_TRACE_LOADED_OBJECTS"] = "1"
    loader_command = [
        "/usr/bin/env",
        "-i",
        *(f"{key}={loader_environment[key]}"
          for key in sorted(loader_environment)),
        str(args.execution_executable),
    ]
    loader = subprocess.run(
        loader_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )
    verify_execution_handles(args)
    loader_log = args.result_dir / "sealed-worker-loader-preflight.log"
    BASE.atomic_text(loader_log, loader.stdout)
    if (
        loader.returncode != 0
        or str(args.execution_library) not in loader.stdout
        or str(args.library) in loader.stdout
    ):
        raise RuntimeError("worker loader did not bind the sealed libmeep image")
    record = {
        "schema_version": 1,
        "persistent_images": args.execution_bindings,
        "checks": {
            "mpiexec": {
                "command": mpiexec_command,
                "returncode": mpiexec.returncode,
                "log": BASE.file_record(mpiexec_log, args.result_dir),
            },
            "worker_loader": {
                "command": loader_command,
                "returncode": loader.returncode,
                "expected_library_image": str(args.execution_library),
                "log": BASE.file_record(loader_log, args.result_dir),
            },
        },
    }
    BASE.atomic_json(args.result_dir / "sealed-execution-preflight.json", record)
    return record


def run_authoritative(args: argparse.Namespace) -> int:
    repo = SCRIPTS.parent
    prefix = args.mpiexec.parent.parent
    receipt, args.executable, args.library, ldd_output = (
        BASE.resolve_receipt_artifacts(
            repo,
            args.build_receipt,
            args.expected_receipt_id,
            args.mpiexec,
        )
    )
    topologies = (
        {"name": "1gpu", "ranks": 1, "visible_devices": "0"},
        {"name": "2gpu", "ranks": 2, "visible_devices": "0,1"},
    )
    mca_path = repo / "environment/openmpi-qualification-mca-params.conf"
    artifact_records = {
        "executable": BASE.file_record(args.executable, repo),
        "library": BASE.file_record(args.library, repo),
        "build_receipt": BASE.file_record(args.build_receipt, repo),
        "mpiexec": BASE.file_record(args.mpiexec, repo),
        "mca": BASE.file_record(mca_path, repo),
    }
    base_environment = {
        "HOME": str(args.result_dir / "home"),
        "PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": f"{args.library.parent}:{prefix / 'lib'}",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MEEP_GPU_BACKEND": "cuda",
        "MEEP_GPU_STRICT": "1",
        "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
        "MEEP_GPU_MULTI_INITIAL_CONDITION": "trigonometric-v1",
        "MEEP_GPU_MULTI_SOURCE_PROFILE": "single-ez-v1",
        "MEEP_GPU_MULTI_LOOP_TILE_BASE_DB": "128",
        "MEEP_GPU_MULTI_BFAST": "0",
        "MEEP_GPU_MULTI_DISABLE_SOURCE": "0",
        "MEEP_GPU_MULTI_OVERLAP_MATERIAL": "0",
        "MEEP_GPU_MPI_COMPLETION": "waitsome",
        "OMP_NUM_THREADS": "1",
        "OPAL_PREFIX": str(prefix),
        "OMPI_MCA_mca_base_param_files": str(
            repo / "environment/openmpi-qualification-mca-params.conf"
        ),
        "OMPI_MCA_mca_base_component_path": str(prefix / "lib/openmpi"),
        "PMIX_MCA_mca_base_param_files": str(
            repo / "environment/openmpi-qualification-mca-params.conf"
        ),
        "PMIX_MCA_mca_base_component_path": str(prefix / "lib/pmix"),
        "PRTE_MCA_mca_base_param_files": str(
            repo / "environment/openmpi-qualification-mca-params.conf"
        ),
    }
    pathlib.Path(base_environment["HOME"]).mkdir(exist_ok=True)

    archive = args.result_dir / "archive"
    archive.mkdir()
    loaded_runner_source = globals().get("__gpmeep_loaded_source_bytes__")
    loaded_runner_sha256 = globals().get("__gpmeep_loaded_source_sha256__")
    if (
        not globals().get("__gpmeep_internal_snapshot__")
        or not isinstance(loaded_runner_source, bytes)
        or not re.fullmatch(r"[0-9a-f]{64}", str(loaded_runner_sha256))
        or hashlib.sha256(loaded_runner_source).hexdigest()
        != loaded_runner_sha256
    ):
        raise RuntimeError(
            "authoritative DFT runner lacks an immutable source snapshot"
        )
    archive_sources: dict[str, tuple[pathlib.Path, bytes, str]] = {
        pathlib.Path(__file__).name: (
            pathlib.Path(__file__).resolve(),
            loaded_runner_source,
            str(loaded_runner_sha256),
        ),
        **LOADED_MODULE_SOURCES,
    }
    archived: dict[str, Any] = {}
    for name, (_source_path, source_bytes, source_sha256) in archive_sources.items():
        destination = archive / name
        BASE.atomic_bytes(destination, source_bytes)
        archived[name] = {
            **BASE.file_record(destination, args.result_dir),
            "loaded_source_sha256": source_sha256,
        }
    execution_paths: dict[str, pathlib.Path] = {}
    artifact_copies = {
        "receipt": (args.build_receipt, archive / "build-provenance.json", 0o444),
        "executable": (args.executable, archive / args.executable.name, 0o555),
        "library": (args.library, archive / args.library.name, 0o444),
        "mpiexec": (args.mpiexec, archive / "receipt-mpiexec", 0o555),
        "mca": (mca_path, archive / mca_path.name, 0o444),
    }
    for label, (source, destination, mode) in artifact_copies.items():
        BASE.atomic_bytes(destination, source.read_bytes())
        os.chmod(destination, mode)
        archived[label] = BASE.file_record(destination, args.result_dir)
        if label != "receipt":
            execution_paths[label] = destination
    args.execution_descriptors = {}
    args.execution_bindings = {}
    for label in ("executable", "library", "mpiexec", "mca"):
        descriptor, binding = create_execution_handle(
            execution_paths[label], args.result_dir, label
        )
        args.execution_descriptors[label] = descriptor
        args.execution_bindings[label] = binding
    args.execution_executable = pathlib.Path(
        args.execution_bindings["executable"]["execution_path"]
    )
    args.execution_library = pathlib.Path(
        args.execution_bindings["library"]["execution_path"]
    )
    args.execution_mpiexec = pathlib.Path(
        args.execution_bindings["mpiexec"]["execution_path"]
    )
    args.execution_mca = pathlib.Path(
        args.execution_bindings["mca"]["execution_path"]
    )
    base_environment["LD_LIBRARY_PATH"] = f"{archive}:{prefix / 'lib'}"
    for key in (
        "OMPI_MCA_mca_base_param_files",
        "PMIX_MCA_mca_base_param_files",
        "PRTE_MCA_mca_base_param_files",
    ):
        base_environment[key] = str(args.execution_mca)
    verify_execution_handles(args)
    invocations_by_topology = {
        topology["name"]: sum(2 * workload["pairs"] + 1 for workload in WORKLOADS)
        for topology in topologies
    }
    expected_process_count = sum(
        invocations_by_topology[topology["name"]] * topology["ranks"]
        for topology in topologies
    )
    gpu_index_by_uuid = physical_gpu_index_by_uuid()
    protocol = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": BASE.utc_now(),
        "receipt_id": receipt["receipt_id"],
        "artifact_set_id": receipt.get("artifact_set_id"),
        "topologies": topologies,
        "workloads": WORKLOADS,
        "conditions": ["forced", "disabled", "automatic-control"],
        "pair_order": "odd:forced-disabled;even:disabled-forced",
        "all_physical_observables_exact": True,
        "expected_processes": {
            "invocations_by_topology": invocations_by_topology,
            "total_rank_processes": expected_process_count,
            "physical_gpu_index_by_uuid": gpu_index_by_uuid,
        },
        "artifacts": artifact_records,
        "archived": archived,
        "immutable_controller_execution": {
            "runner_loaded_source_sha256": loaded_runner_sha256,
            "runner": "open-fstat-read-fstat-compile-exec-exact-bytes",
            "local_modules": "read-once-compile-exec-and-archive-exact-bytes",
        },
        "sealed_execution": {
            "policy": "persistent-read-only-fd-images-verified-every-sample",
            "persistent_images": args.execution_bindings,
        },
        "ldd": ldd_output,
    }
    BASE.atomic_json(args.result_dir / "protocol.json", protocol)
    BASE.atomic_json(
        args.result_dir / "run-state-start.json",
        {
            "schema_version": 1,
            "state": "RUNNING",
            "started_at_utc": BASE.utc_now(),
            "receipt_id": receipt["receipt_id"],
        },
    )
    sealed_execution_preflight(args, base_environment)
    inventory = {
        "schema_version": 1,
        "started_at_utc": BASE.utc_now(),
        "receipt_id": receipt["receipt_id"],
        "gpu": BASE.gpu_snapshot(),
        "lscpu": BASE.command_output(["/usr/bin/lscpu"]),
        "topologies": topologies,
        "workloads": WORKLOADS,
        "physical_gpu_index_by_uuid": gpu_index_by_uuid,
        "sealed_execution_preflight": BASE.file_record(
            args.result_dir / "sealed-execution-preflight.json",
            args.result_dir,
        ),
    }
    BASE.require_idle(inventory["gpu"])
    BASE.atomic_json(args.result_dir / "inventory.json", inventory)

    monitor_path = args.result_dir / "continuous-gpu-process-monitor.log"
    monitor_process: subprocess.Popen[bytes] | None = None
    monitor_stream: Any = None
    monitor_stopped = False
    try:
        monitor_process, monitor_stream = BASE.start_process_monitor(monitor_path)
        samples, results = execute_samples(args, topologies, base_environment)
        expected_pid_to_gpu = expected_pid_gpu_map(
            samples, gpu_index_by_uuid
        )
        if len(expected_pid_to_gpu) != expected_process_count:
            raise RuntimeError("authoritative sample PID count is incomplete")
        monitor = stop_process_monitor(
            monitor_process, monitor_stream, monitor_path, expected_pid_to_gpu
        )
        monitor_stopped = True
        BASE.atomic_json(
            args.result_dir / "continuous-gpu-process-monitor.json", monitor
        )
        if not monitor["pass"]:
            raise RuntimeError(
                "continuous GPU process monitor did not prove the exact 1/2-GPU run set"
            )
        receipt_after, executable_after, library_after, ldd_after = (
            BASE.resolve_receipt_artifacts(
                repo,
                args.build_receipt,
                args.expected_receipt_id,
                args.mpiexec,
            )
        )
        verify_execution_handles(args)
        if (
            receipt_after["receipt_id"] != receipt["receipt_id"]
            or executable_after != args.executable
            or library_after != args.library
            or BASE.normalized_ldd_dependencies(ldd_after)
            != BASE.normalized_ldd_dependencies(ldd_output)
            or any(
                BASE.sha256_file(archive / name) != source_sha256
                or BASE.sha256_file(source_path) != source_sha256
                for name, (
                    source_path,
                    _source_bytes,
                    source_sha256,
                ) in archive_sources.items()
            )
            or any(
                BASE.sha256_file(source) != BASE.sha256_file(destination)
                for source, destination, _mode in artifact_copies.values()
            )
            or any(
                filesystem_identity(execution_paths[label], args.result_dir)
                != args.execution_bindings[label]["snapshot"]
                for label in execution_paths
            )
        ):
            raise RuntimeError("receipt or archived benchmark artifact changed")
        passed = all(record["pass"] for record in results)
        summary = {
            "schema_version": 1,
            "state": "COMPLETE" if passed else "FAILED_PERFORMANCE_GATE",
            "finished_at_utc": BASE.utc_now(),
            "receipt_id": receipt["receipt_id"],
            "artifact_set_id": receipt.get("artifact_set_id"),
            "sample_count": len(samples),
            "all_observables_exact": True,
            "automatic_fail_closed": True,
            "continuous_contention_check_pass": True,
            "results": results,
            "artifacts": artifact_records,
        }
        BASE.atomic_json(args.result_dir / "summary.json", summary)
        BASE.atomic_text(args.result_dir / "report.md", markdown_report(summary))
        BASE.atomic_json(
            args.result_dir / "run-state-final.json",
            {
                "schema_version": 1,
                "state": summary["state"],
                "finished_at_utc": summary["finished_at_utc"],
                "receipt_id": receipt["receipt_id"],
            },
        )
        if not passed:
            finalize_failure(
                args.result_dir,
                {
                    "schema_version": 1,
                    "state": "FAILED_GATES",
                    "finished_at_utc": summary["finished_at_utc"],
                    "receipt_id": receipt["receipt_id"],
                    "summary": BASE.file_record(
                        args.result_dir / "summary.json", args.result_dir
                    ),
                    "report": BASE.file_record(
                        args.result_dir / "report.md", args.result_dir
                    ),
                },
            )
            return 2
        finalize_success(args.result_dir, receipt["receipt_id"])
        return 0
    finally:
        if monitor_process is not None and not monitor_stopped:
            try:
                monitor = stop_process_monitor(
                    monitor_process,
                    monitor_stream,
                    monitor_path,
                    expected_pid_gpu_map(
                        locals().get("samples", []), gpu_index_by_uuid
                    ),
                )
                BASE.atomic_json(
                    args.result_dir / "continuous-gpu-process-monitor-error.json",
                    monitor,
                )
            except Exception as monitor_error:
                BASE.atomic_json(
                    args.result_dir / "continuous-gpu-process-monitor-stop-error.json",
                    {"error": f"{type(monitor_error).__name__}: {monitor_error}"},
                )


def parse_args() -> argparse.Namespace:
    repo = SCRIPTS.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--build-receipt",
        type=pathlib.Path,
        default=repo / "build/meep-cuda-mpi-python-fp32/build-provenance.json",
    )
    parser.add_argument("--expected-receipt-id", required=True)
    parser.add_argument(
        "--mpiexec",
        type=pathlib.Path,
        default=repo / ".envs/meep-gpu-cuda-mpi/bin/mpiexec",
    )
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir = args.result_dir.resolve()
    args.build_receipt = args.build_receipt.resolve()
    args.mpiexec = args.mpiexec.resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_receipt_id):
        raise RuntimeError("expected receipt ID must be a lowercase SHA-256")
    if args.timeout <= 0:
        raise RuntimeError("timeout is invalid")
    if args.result_dir.exists() and any(args.result_dir.iterdir()):
        raise RuntimeError("authoritative result directory must be absent or empty")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.execution_descriptors = {}
    try:
        for path in (args.build_receipt, args.mpiexec):
            if not path.is_file():
                raise RuntimeError(f"required benchmark artifact is absent: {path}")
        return run_authoritative(args)
    except Exception as error:
        finalize_failure(
            args.result_dir,
            {
                "schema_version": 1,
                "state": "FAILED_PREFLIGHT_OR_EXECUTION",
                "finished_at_utc": BASE.utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        for descriptor in args.execution_descriptors.values():
            os.close(descriptor)


if __name__ == "__main__":
    if not globals().get("__gpmeep_internal_snapshot__"):
        reexec_immutable_source()
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
