#!/usr/bin/env python3
"""Sealed A/B/C benchmark for automatic CUDA curl/E/H phase batching.

The three conditions differ only in the curl and update-E/H phase policy:
automatic leaves both controls unset, disabled uses both opt-outs, and forced
uses both exact opt-ins.  Source batching is held forced in every condition.
"""

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


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


LOADED_MODULE_SOURCES: dict[str, tuple[pathlib.Path, bytes, str]] = {}


def load_source_module(name: str, filename: str) -> types.ModuleType:
    path = SCRIPT_DIR / filename
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


# Load the transitive local dependencies from one captured byte string each.
# In particular, MGB must not import mutable path-selected helper bytes.
EVIDENCE = load_source_module(
    "gpmeep_benchmark_evidence", "gpmeep_benchmark_evidence.py"
)
QUALIFICATION = load_source_module(
    "gpmeep_qualification_contract_for_provenance",
    "gpmeep_qualification_contract.py",
)
PROVENANCE = load_source_module("gpmeep_provenance", "gpmeep_provenance.py")
PROVENANCE._load_qualification_contract_from_source = lambda: QUALIFICATION
MGB = load_source_module("multi_gpu_benchmark", "multi_gpu_benchmark.py")
BASE = load_source_module(
    "gpmeep_phase_batch_evidence_base", "benchmark-dft-phase-sharing.py"
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
        raise RuntimeError("runner source is not one bounded unlinked regular file")
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
POLICY_PREFIX = "gpmeep-phase-batch-policy-v1:"
PROCESS_PREFIX = "gpmeep-multi-process-v1:"
CONDITIONS = ("automatic", "disabled", "forced")
CONDITION_ORDERS = (
    ("automatic", "disabled", "forced"),
    ("disabled", "forced", "automatic"),
    ("forced", "automatic", "disabled"),
)
POLICY_KEYS = {
    "curl_automatic_checks",
    "curl_automatic_selected",
    "curl_automatic_rejected",
    "curl_forced_batches",
    "curl_batched_operations",
    "curl_unbatched_operations",
    "update_eh_automatic_checks",
    "update_eh_automatic_selected",
    "update_eh_automatic_rejected",
    "update_eh_forced_batches",
    "update_eh_batched_operations",
    "update_eh_unbatched_operations",
}
OBSERVABLES = BASE.OBSERVABLES
SMALL_MINIMUM_DISABLED_OVER_AUTOMATIC = 1.15
LARGE_MAXIMUM_AUTOMATIC_OVER_DISABLED = 1.02
LARGE_MAXIMUM_AUTOMATIC_OVER_FORCED = 1.02
MANIFEST_NAME = "evidence-manifest.json"
TERMINAL_NAMES = frozenset({"COMPLETE.json", "FAILED.json"})


def filesystem_identity(
    path: pathlib.Path,
    root: pathlib.Path,
    *,
    require_regular: bool,
    include_sha256: bool,
) -> dict[str, Any]:
    metadata = path.stat(follow_symlinks=False)
    if require_regular and (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
    ):
        raise RuntimeError(
            f"sealed execution artifact is not one regular link: {path}"
        )
    if not require_regular and not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"sealed execution directory is not a directory: {path}")
    record: dict[str, Any] = {
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
        record["sha256"] = BASE.sha256_file(path)
    return record


def execution_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "result_directory": filesystem_identity(
            args.result_dir,
            args.result_dir.parent,
            require_regular=False,
            include_sha256=False,
        ),
        "archive_directory": filesystem_identity(
            args.execution_snapshot_executable.parent,
            args.result_dir,
            require_regular=False,
            include_sha256=False,
        ),
        "executable": filesystem_identity(
            args.execution_snapshot_executable,
            args.result_dir,
            require_regular=True,
            include_sha256=True,
        ),
        "library": filesystem_identity(
            args.execution_snapshot_library,
            args.result_dir,
            require_regular=True,
            include_sha256=True,
        ),
        "mpiexec": filesystem_identity(
            args.execution_snapshot_mpiexec,
            args.result_dir,
            require_regular=True,
            include_sha256=True,
        ),
        "mca": filesystem_identity(
            args.execution_snapshot_mca,
            args.result_dir,
            require_regular=True,
            include_sha256=True,
        ),
    }


def stable_execution_artifacts(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in identity.items()
        if key != "result_directory"
    }


def create_execution_handle(
    snapshot: pathlib.Path,
    root: pathlib.Path,
    label: str,
) -> tuple[int, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(snapshot, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(
                f"execution snapshot is not one regular link: {snapshot}"
            )
        digest = hashlib.sha256()
        offset = 0
        while offset < metadata.st_size:
            block = os.pread(
                descriptor,
                min(1024 * 1024, metadata.st_size - offset),
                offset,
            )
            if not block:
                break
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
        fields = (
            "st_dev",
            "st_ino",
            "st_nlink",
            "st_size",
            "st_mode",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if offset != after.st_size or any(
            getattr(metadata, key) != getattr(after, key) for key in fields
        ):
            raise RuntimeError(
                f"execution handle changed while hashing: {snapshot}"
            )
        snapshot_record = filesystem_identity(
            snapshot,
            root,
            require_regular=True,
            include_sha256=True,
        )
        if (
            snapshot_record["device"] != after.st_dev
            or snapshot_record["inode"] != after.st_ino
            or snapshot_record["sha256"] != digest.hexdigest()
        ):
            raise RuntimeError(
                f"open execution image differs from archived path: {snapshot}"
            )
        binding = {
            "label": label,
            "snapshot": snapshot_record,
            "execution_model": "persistent-read-only-no-follow-proc-fd",
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
        verify_execution_handle(binding, descriptor)
        return descriptor, binding
    except BaseException:
        os.close(descriptor)
        raise


def verify_execution_handle(binding: dict[str, Any], descriptor: int) -> None:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        block = os.pread(
            descriptor,
            min(1024 * 1024, before.st_size - offset),
            offset,
        )
        if not block:
            break
        digest.update(block)
        offset += len(block)
    after = os.fstat(descriptor)
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
        getattr(before, key) != getattr(after, key)
        for key in (
            "st_dev",
            "st_ino",
            "st_nlink",
            "st_size",
            "st_mode",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    ):
        raise RuntimeError(
            f"execution image changed while hashing: {binding.get('label')}"
        )
    if any(binding.get(key) != value for key, value in actual.items()):
        raise RuntimeError(f"execution image changed: {binding.get('label')}")


def verify_execution_handles(args: argparse.Namespace) -> None:
    if set(args.execution_bindings) != set(args.execution_descriptors):
        raise RuntimeError("sealed execution handle set is incomplete")
    for name in sorted(args.execution_bindings):
        verify_execution_handle(
            args.execution_bindings[name], args.execution_descriptors[name]
        )


def write_evidence_manifest(result_dir: pathlib.Path) -> pathlib.Path:
    manifest_path = result_dir / MANIFEST_NAME
    excluded = {MANIFEST_NAME, *TERMINAL_NAMES}
    files = sorted(
        path
        for path in result_dir.rglob("*")
        if path.is_file()
        and path.name not in excluded
        and "home" not in path.relative_to(result_dir).parts
    )
    BASE.atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "files": [BASE.file_record(path, result_dir) for path in files],
        },
    )
    return manifest_path


def terminal_present(result_dir: pathlib.Path) -> bool:
    return any((result_dir / name).exists() for name in TERMINAL_NAMES)


def finalize_success(
    result_dir: pathlib.Path,
    receipt_id: str,
) -> None:
    if terminal_present(result_dir):
        raise RuntimeError("refusing to create a second terminal evidence marker")
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


def print_summary_best_effort(summary: dict[str, Any]) -> None:
    try:
        print(json.dumps(summary, sort_keys=True))
    except (BrokenPipeError, OSError):
        pass


def workloads(repeats: int) -> list[dict[str, int]]:
    return [
        {"pixels": pixels, "warmup_steps": 12, "steps": 80, "repeats": repeats}
        for pixels in (128, 256)
    ]


def configurations(repeats: int) -> list[dict[str, int]]:
    return [
        {
            "pixels": workload["pixels"],
            "warmup_steps": workload["warmup_steps"],
            "steps": workload["steps"],
            "repeats": repeats,
            "ranks": ranks,
        }
        for workload in workloads(repeats)
        for ranks in (1, 2)
    ]


def condition_environment(
    base: dict[str, str], condition: str, configuration: dict[str, int]
) -> dict[str, str]:
    if condition not in CONDITIONS:
        raise RuntimeError(f"unknown phase-batch condition: {condition}")
    environment = dict(base)
    environment.update(
        {
            "MEEP_GPU_MULTI_PIXELS": str(configuration["pixels"]),
            "MEEP_GPU_MULTI_WARMUP_STEPS": str(
                configuration["warmup_steps"]
            ),
            "MEEP_GPU_MULTI_STEPS": str(configuration["steps"]),
        }
    )
    controls = {
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL",
        "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
        "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH",
    }
    for name in controls:
        environment.pop(name, None)
    if condition == "disabled":
        environment["MEEP_GPU_DISABLE_PHASE_BATCHED_CURL"] = "1"
        environment["MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH"] = "1"
    elif condition == "forced":
        environment["MEEP_GPU_ENABLE_PHASE_BATCHED_CURL"] = "1"
        environment["MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH"] = "1"
    return environment


def exact_policy_marker(output: str) -> dict[str, int]:
    value = BASE.exactly_one(output, POLICY_PREFIX, "phase-batch policy")
    if set(value) != POLICY_KEYS or any(
        type(value[key]) is not int or value[key] < 0 for key in POLICY_KEYS
    ):
        raise RuntimeError("phase-batch policy marker has an invalid exact schema")
    return value


def validate_policy(
    condition: str, pixels: int, ranks: int, policy: dict[str, int]
) -> None:
    for prefix in ("curl", "update_eh"):
        checks = policy[f"{prefix}_automatic_checks"]
        selected = policy[f"{prefix}_automatic_selected"]
        rejected = policy[f"{prefix}_automatic_rejected"]
        forced = policy[f"{prefix}_forced_batches"]
        batched = policy[f"{prefix}_batched_operations"]
        unbatched = policy[f"{prefix}_unbatched_operations"]
        if condition == "automatic":
            if (
                checks <= 0
                or selected + rejected != checks
                or forced != 0
                or batched + unbatched <= 0
            ):
                raise RuntimeError(
                    f"automatic {prefix} policy accounting is inconsistent"
                )
            if pixels == 128 and (selected <= 0 or batched <= 0):
                raise RuntimeError(
                    f"automatic {prefix} policy selected no small-grid batches"
                )
        elif condition == "disabled":
            if any((checks, selected, rejected, forced, batched, unbatched)):
                raise RuntimeError(
                    f"disabled {prefix} policy reported collected phase work"
                )
        elif (
            checks != 0
            or selected != 0
            or rejected != 0
            or forced <= 0
            or batched <= 0
            or unbatched != 0
        ):
            raise RuntimeError(f"forced {prefix} policy accounting is inconsistent")
    if condition == "automatic" and pixels == 256 and ranks == 1:
        if (
            policy["curl_automatic_rejected"]
            + policy["update_eh_automatic_rejected"]
            <= 0
        ):
            raise RuntimeError(
                "one-GPU large-grid automatic policy proved no crossover rejection"
            )


def validate_devices(
    output: str, ranks: int
) -> list[dict[str, Any]]:
    devices = BASE.records(output, BASE.DEVICE_PREFIX)
    exact_keys = {
        "rank",
        "ordinal",
        "uuid",
        "name",
        "compute_major",
        "compute_minor",
        "compatible",
    }
    if len(devices) != ranks:
        raise RuntimeError("device marker count differs from MPI rank count")
    if any(
        not isinstance(item, dict)
        or set(item) != exact_keys
        or type(item["rank"]) is not int
        or type(item["ordinal"]) is not int
        or type(item["uuid"]) is not str
        or not item["uuid"]
        or type(item["name"]) is not str
        or not item["name"]
        or type(item["compute_major"]) is not int
        or type(item["compute_minor"]) is not int
        or item["compatible"] is not True
        for item in devices
    ):
        raise RuntimeError("device marker has an invalid exact schema")
    devices.sort(key=lambda item: item["rank"])
    if (
        {item["rank"] for item in devices} != set(range(ranks))
        or {item["ordinal"] for item in devices} != set(range(ranks))
        or len({item["uuid"] for item in devices}) != ranks
    ):
        raise RuntimeError("MPI ranks did not prove a one-rank-per-GPU mapping")
    return devices


def validate_processes(output: str, ranks: int) -> list[dict[str, int]]:
    processes = BASE.records(output, PROCESS_PREFIX)
    if (
        len(processes) != ranks
        or any(
            not isinstance(item, dict)
            or set(item) != {"rank", "pid"}
            or type(item["rank"]) is not int
            or type(item["pid"]) is not int
            or item["pid"] <= 1
            for item in processes
        )
    ):
        raise RuntimeError("process marker has an invalid exact schema")
    processes.sort(key=lambda item: item["rank"])
    if (
        {item["rank"] for item in processes} != set(range(ranks))
        or len({item["pid"] for item in processes}) != ranks
    ):
        raise RuntimeError("MPI ranks did not prove distinct worker processes")
    return processes


def validate_sample(
    condition: str, configuration: dict[str, int], output: str
) -> tuple[
    dict[str, Any],
    dict[str, int],
    list[dict[str, Any]],
    list[dict[str, int]],
]:
    BASE.reject_unknown_gpmeep_markers(output)
    benchmark = MGB.extract_record(
        output,
        expected_ranks=configuration["ranks"],
        expected_pixels=configuration["pixels"],
        expected_warmup_steps=configuration["warmup_steps"],
        expected_steps=configuration["steps"],
        expected_transport="pinned",
        expected_worker_profile={
            "loop_tile_base_db": 128,
            "bfast": False,
            "source_enabled": True,
            "overlap_material": False,
            "completion_policy": "waitsome",
            "initial_condition": "affine-v1",
        },
    )
    BASE.validate_initialization_markers(
        output,
        benchmark,
        expected_profile="affine-v1",
        expected_source_profile="dual-electric-v1",
    )
    ranks = configuration["ranks"]
    if (
        benchmark.get("schema_version") != 4
        or benchmark.get("mpi_ranks") != ranks
        or benchmark.get("pixels") != configuration["pixels"]
        or benchmark.get("warmup_steps") != configuration["warmup_steps"]
        or benchmark.get("steps") != configuration["steps"]
        or benchmark.get("initial_condition") != "affine-v1"
        or benchmark.get("requested_transport") != "pinned"
        or benchmark.get("selected_transport")
        != ("none" if ranks == 1 else "pinned")
        or benchmark.get("cpu_calls") != 0
        or type(benchmark.get("cuda_calls")) is not int
        or benchmark["cuda_calls"] <= 0
    ):
        raise RuntimeError("benchmark identity or exclusive CUDA dispatch is invalid")
    phases = benchmark.get("phase_calls")
    if not isinstance(phases, dict) or any(
        not isinstance(record, dict) or record.get("cpu_calls") != 0
        for record in phases.values()
    ):
        raise RuntimeError("benchmark phase record contains CPU field work")
    if ranks == 2 and (
        benchmark.get("pinned_bytes", 0) <= 0
        or benchmark.get("cuda_aware_bytes") != 0
        or benchmark.get("mpi_messages", 0) <= 0
        or benchmark.get("remote_cut_scalars", 0) <= 0
    ):
        raise RuntimeError("two-GPU sample did not prove pinned MPI field exchange")
    if ranks == 1 and (
        benchmark.get("pinned_bytes") != 0
        or benchmark.get("cuda_aware_bytes") != 0
        or benchmark.get("mpi_messages") != 0
        or benchmark.get("remote_cut_scalars") != 0
    ):
        raise RuntimeError("one-GPU sample unexpectedly used MPI field exchange")
    for key in OBSERVABLES:
        value = benchmark.get(key)
        if value is None:
            raise RuntimeError(f"benchmark lacks physical observable {key}")
    seconds = benchmark.get("seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("benchmark seconds is not finite and positive")
    policy = exact_policy_marker(output)
    validate_policy(condition, configuration["pixels"], ranks, policy)
    devices = validate_devices(output, ranks)
    processes = validate_processes(output, ranks)
    return benchmark, policy, devices, processes


def sample_command(
    args: argparse.Namespace,
    environment: dict[str, str],
    configuration: dict[str, int],
) -> list[str]:
    return [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=10s",
        f"{args.timeout}s",
        "/usr/bin/env",
        "-i",
        *(f"{key}={environment[key]}" for key in sorted(environment)),
        str(args.execution_mpiexec),
        "--host",
        f"localhost:{configuration['ranks']}",
        "--bind-to",
        "none",
        "-np",
        str(configuration["ranks"]),
        "/usr/bin/env",
        f"LD_PRELOAD={args.execution_library}",
        str(args.execution_executable),
    ]


def sealed_execution_preflight(
    args: argparse.Namespace,
    base_environment: dict[str, str],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    mpiexec_command = [
        "/usr/bin/env",
        "-i",
        *(f"{key}={base_environment[key]}" for key in sorted(base_environment)),
        str(args.execution_mpiexec),
        "--version",
    ]
    verify_execution_handles(args)
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
    checks["mpiexec"] = {
        "command": mpiexec_command,
        "returncode": mpiexec.returncode,
        "log": BASE.file_record(mpiexec_log, args.result_dir),
    }

    loader_environment = dict(base_environment)
    loader_environment["LD_PRELOAD"] = str(args.execution_library)
    loader_environment["LD_TRACE_LOADED_OBJECTS"] = "1"
    loader_command = [
        "/usr/bin/env",
        "-i",
        *(f"{key}={loader_environment[key]}" for key in sorted(loader_environment)),
        str(args.execution_executable),
    ]
    verify_execution_handles(args)
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
        raise RuntimeError(
            "worker loader did not bind libmeep to the persistent FD image"
        )
    checks["worker_loader"] = {
        "command": loader_command,
        "returncode": loader.returncode,
        "expected_library_image": str(args.execution_library),
        "log": BASE.file_record(loader_log, args.result_dir),
    }
    record = {
        "schema_version": 1,
        "persistent_images": args.execution_bindings,
        "checks": checks,
    }
    BASE.atomic_json(args.result_dir / "sealed-execution-preflight.json", record)
    return record


def run_sample(
    args: argparse.Namespace,
    base_environment: dict[str, str],
    configuration: dict[str, int],
    repeat: int,
    order_index: int,
    condition: str,
) -> dict[str, Any]:
    stem = (
        f"p{configuration['pixels']}-r{configuration['ranks']}-"
        f"repeat{repeat:02d}-{order_index}-{condition}"
    )
    before = BASE.gpu_snapshot()
    BASE.require_idle(before)
    environment = condition_environment(base_environment, condition, configuration)
    command = sample_command(args, environment, configuration)
    execution_before = execution_identity(args)
    if stable_execution_artifacts(execution_before) != args.execution_baseline:
        raise RuntimeError("sealed execution artifacts changed before sample")
    verify_execution_handles(args)
    started_at = BASE.utc_now()
    wall_start = time.perf_counter()
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_seconds = time.perf_counter() - wall_start
    verify_execution_handles(args)
    execution_after = execution_identity(args)
    if execution_after != execution_before:
        raise RuntimeError("sealed execution path changed while sample ran")
    raw_path = args.result_dir / f"{stem}.log"
    BASE.atomic_text(raw_path, result.stdout)
    after: dict[str, str] | None = None
    benchmark: dict[str, Any] | None = None
    policy: dict[str, int] | None = None
    devices: list[dict[str, Any]] = []
    processes: list[dict[str, int]] = []
    error: str | None = None
    try:
        after = BASE.gpu_snapshot()
        BASE.require_idle(after)
        if result.returncode:
            raise RuntimeError(f"sample exited with status {result.returncode}")
        benchmark, policy, devices, processes = validate_sample(
            condition, configuration, result.stdout
        )
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    sample = {
        "schema_version": 1,
        "status": "PASS" if error is None else "FAIL",
        "identity": {
            "pixels": configuration["pixels"],
            "ranks": configuration["ranks"],
            "warmup_steps": configuration["warmup_steps"],
            "steps": configuration["steps"],
            "repeat": repeat,
            "order_index": order_index,
            "condition": condition,
        },
        "started_at_utc": started_at,
        "finished_at_utc": BASE.utc_now(),
        "command": command,
        "execution_identity_before": execution_before,
        "execution_identity_after": execution_after,
        "persistent_execution_images": args.execution_bindings,
        "selected_environment": dict(sorted(environment.items())),
        "external_process_wall_seconds": wall_seconds,
        "returncode": result.returncode,
        "before": before,
        "after": after,
        "devices": devices,
        "processes": processes,
        "phase_batch_policy": policy,
        "result": benchmark,
        "raw_log": raw_path.name,
        "raw_log_sha256": BASE.sha256_file(raw_path),
        "error": error,
    }
    BASE.atomic_json(args.result_dir / f"{stem}.json", sample)
    if error is not None:
        raise RuntimeError(f"invalid sample {stem}: {error}; see {raw_path}")
    return sample


def physical_observables(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: sample["result"][key] for key in OBSERVABLES}


def summarize_configuration(
    configuration: dict[str, int], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    expected = configuration["repeats"] * len(CONDITIONS)
    if len(samples) != expected:
        raise RuntimeError("configuration sample count is incomplete")
    by_repeat: dict[int, dict[str, dict[str, Any]]] = {}
    reference_observables: dict[str, Any] | None = None
    reference_devices: dict[str, Any] | None = None
    for sample in samples:
        identity = sample["identity"]
        repeat = identity["repeat"]
        condition = identity["condition"]
        if condition in by_repeat.setdefault(repeat, {}):
            raise RuntimeError("configuration has a duplicate repeat condition")
        by_repeat[repeat][condition] = sample
        current = physical_observables(sample)
        if reference_observables is None:
            reference_observables = current
        elif current != reference_observables:
            raise RuntimeError("phase policy changed a physical observable")
        current_devices = {
            str(item["rank"]): {
                key: item[key]
                for key in (
                    "ordinal",
                    "uuid",
                    "name",
                    "compute_major",
                    "compute_minor",
                    "compatible",
                )
            }
            for item in sample["devices"]
        }
        if reference_devices is None:
            reference_devices = current_devices
        elif current_devices != reference_devices:
            raise RuntimeError("rank-to-GPU device identity changed across samples")
    if set(by_repeat) != set(range(1, configuration["repeats"] + 1)) or any(
        set(items) != set(CONDITIONS) for items in by_repeat.values()
    ):
        raise RuntimeError("configuration repeat matrix is incomplete")
    seconds = {
        condition: [
            float(by_repeat[repeat][condition]["result"]["seconds"])
            for repeat in sorted(by_repeat)
        ]
        for condition in CONDITIONS
    }
    medians = {
        condition: statistics.median(values)
        for condition, values in seconds.items()
    }
    paired_ratios = {
        "disabled_over_automatic": [
            seconds["disabled"][index] / seconds["automatic"][index]
            for index in range(configuration["repeats"])
        ],
        "forced_over_automatic": [
            seconds["forced"][index] / seconds["automatic"][index]
            for index in range(configuration["repeats"])
        ],
        "automatic_over_disabled": [
            seconds["automatic"][index] / seconds["disabled"][index]
            for index in range(configuration["repeats"])
        ],
        "automatic_over_forced": [
            seconds["automatic"][index] / seconds["forced"][index]
            for index in range(configuration["repeats"])
        ],
    }
    ratios = {
        name: statistics.median(values)
        for name, values in paired_ratios.items()
    }
    ratio_of_medians = {
        "disabled_over_automatic": medians["disabled"] / medians["automatic"],
        "forced_over_automatic": medians["forced"] / medians["automatic"],
        "automatic_over_disabled": medians["automatic"] / medians["disabled"],
        "automatic_over_forced": medians["automatic"] / medians["forced"],
    }
    if configuration["pixels"] == 128:
        gates = [
            {
                "name": "minimum_median_paired_disabled_over_automatic_time_ratio",
                "threshold": SMALL_MINIMUM_DISABLED_OVER_AUTOMATIC,
                "observed": ratios["disabled_over_automatic"],
                "pass": ratios["disabled_over_automatic"]
                >= SMALL_MINIMUM_DISABLED_OVER_AUTOMATIC,
            }
        ]
    else:
        gates = [
            {
                "name": "maximum_median_paired_automatic_over_disabled_time_ratio",
                "threshold": LARGE_MAXIMUM_AUTOMATIC_OVER_DISABLED,
                "observed": ratios["automatic_over_disabled"],
                "pass": ratios["automatic_over_disabled"]
                <= LARGE_MAXIMUM_AUTOMATIC_OVER_DISABLED,
            },
            {
                "name": "maximum_median_paired_automatic_over_forced_time_ratio",
                "threshold": LARGE_MAXIMUM_AUTOMATIC_OVER_FORCED,
                "observed": ratios["automatic_over_forced"],
                "pass": ratios["automatic_over_forced"]
                <= LARGE_MAXIMUM_AUTOMATIC_OVER_FORCED,
            },
        ]
    return {
        "pixels": configuration["pixels"],
        "ranks": configuration["ranks"],
        "repeats": configuration["repeats"],
        "seconds": seconds,
        "median_seconds": medians,
        "paired_time_ratios": paired_ratios,
        "median_paired_time_ratios": ratios,
        "ratio_of_median_times": ratio_of_medians,
        "physical_observables": reference_observables,
        "device_mapping": reference_devices,
        "gates": gates,
        "pass": all(gate["pass"] for gate in gates),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# M25 automatic CUDA phase-batch benchmark",
        "",
        f"- Result: **{'PASS' if summary['pass'] else 'FAIL'}**",
        f"- Samples: `{summary['sample_count']}`",
        f"- Build receipt: `{summary['receipt_id']}`",
        "- Numerical gate: every recorded observable is exactly identical within each configuration.",
        "",
        "| Grid | GPUs | automatic (s) | disabled (s) | forced (s) | disabled/auto | forced/auto | Result |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in summary["configurations"]:
        median = item["median_seconds"]
        ratios = item["median_paired_time_ratios"]
        lines.append(
            f"| {item['pixels']}³ | {item['ranks']} | "
            f"{median['automatic']:.9f} | {median['disabled']:.9f} | "
            f"{median['forced']:.9f} | "
            f"{ratios['disabled_over_automatic']:.4f}× | "
            f"{ratios['forced_over_automatic']:.4f}× | "
            f"{'PASS' if item['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "`disabled/auto > 1` means the automatic policy is faster than per-operation launches. All timings are the worker's synchronized maximum rank time.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_process_monitor(
    output: str,
    alive_before_stop: bool,
    monitor_returncode: int | None,
    expected_pid_to_gpu: dict[int, int],
) -> dict[str, Any]:
    benchmark_rows: list[str] = []
    unrelated_rows: list[str] = []
    observed_gpus: set[int] = set()
    observed_pid_to_gpu: dict[int, int] = {}
    for line in output.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 3 or columns[1] == "-":
            continue
        command = columns[-1]
        try:
            gpu_index = int(columns[0])
            pid = int(columns[1])
        except ValueError:
            unrelated_rows.append(line)
            continue
        if (
            "gpu-mpi-per" in command
            and expected_pid_to_gpu.get(pid) == gpu_index
        ):
            benchmark_rows.append(line)
            previous = observed_pid_to_gpu.setdefault(pid, gpu_index)
            if previous != gpu_index:
                unrelated_rows.append(line)
                continue
            observed_gpus.add(gpu_index)
        else:
            unrelated_rows.append(line)
    record = {
        "schema_version": 1,
        "alive_before_stop": alive_before_stop,
        "monitor_returncode": monitor_returncode,
        "observed_gpu_indices": sorted(observed_gpus),
        "expected_worker_pid_to_gpu": {
            str(pid): gpu
            for pid, gpu in sorted(expected_pid_to_gpu.items())
        },
        "observed_worker_pid_to_gpu": {
            str(pid): gpu
            for pid, gpu in sorted(observed_pid_to_gpu.items())
        },
        "unobserved_expected_worker_pids": sorted(
            set(expected_pid_to_gpu) - set(observed_pid_to_gpu)
        ),
        "benchmark_rows": benchmark_rows,
        "unrelated_process_rows": unrelated_rows,
    }
    record["pass"] = (
        alive_before_stop
        and monitor_returncode in {0, -15}
        and not unrelated_rows
        and observed_gpus == {0, 1}
        and bool(benchmark_rows)
        and set(observed_pid_to_gpu).issubset(expected_pid_to_gpu)
    )
    return record


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
    record = parse_process_monitor(
        path.read_text(encoding="utf-8", errors="replace"),
        alive_before_stop,
        process.returncode,
        expected_pid_to_gpu,
    )
    record.update(
        {
            "raw": path.name,
            "raw_sha256": BASE.sha256_file(path),
        }
    )
    return record


def parse_args() -> argparse.Namespace:
    repo = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
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
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def run_authoritative(args: argparse.Namespace) -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    prefix = args.mpiexec.parent.parent
    receipt, args.executable, args.library, ldd_output = BASE.resolve_receipt_artifacts(
        repo, args.build_receipt, args.expected_receipt_id, args.mpiexec
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
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "MEEP_GPU_BACKEND": "cuda",
        "MEEP_GPU_STRICT": "1",
        "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
        "MEEP_GPU_MPI_TRANSPORT": "pinned",
        "MEEP_GPU_MPI_COMPLETION": "waitsome",
        "MEEP_GPU_MULTI_INITIAL_CONDITION": "affine-v1",
        "MEEP_GPU_MULTI_SOURCE_PROFILE": "dual-electric-v1",
        "MEEP_GPU_MULTI_LOOP_TILE_BASE_DB": "128",
        "MEEP_GPU_MULTI_BFAST": "0",
        "MEEP_GPU_MULTI_DISABLE_SOURCE": "0",
        "MEEP_GPU_MULTI_OVERLAP_MATERIAL": "0",
        "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE": "1",
        "MEEP_GPU_EMIT_PROCESS_ID": "1",
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
    matrix = configurations(args.repeats)
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
            "authoritative benchmark was not loaded through its immutable source snapshot"
        )
    archive_sources: dict[str, tuple[pathlib.Path, bytes, str]] = {
        pathlib.Path(__file__).name: (
            pathlib.Path(__file__).resolve(),
            loaded_runner_source,
            loaded_runner_sha256,
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
    for label, source, name, mode in (
        ("build_receipt", args.build_receipt, "build-provenance.json", 0o444),
        ("executable", args.executable, args.executable.name, 0o555),
        ("library", args.library, args.library.name, 0o444),
        ("mpiexec", args.mpiexec, "receipt-mpiexec", 0o555),
        (
            "mca",
            mca_path,
            "openmpi-qualification-mca-params.conf",
            0o444,
        ),
    ):
        destination = archive / name
        BASE.atomic_bytes(destination, source.read_bytes())
        os.chmod(destination, mode)
        archived[label] = BASE.file_record(destination, args.result_dir)
        if label in {"executable", "library", "mpiexec", "mca"}:
            execution_paths[label] = destination
    args.execution_snapshot_executable = execution_paths["executable"]
    args.execution_snapshot_library = execution_paths["library"]
    args.execution_snapshot_mpiexec = execution_paths["mpiexec"]
    args.execution_snapshot_mca = execution_paths["mca"]
    args.execution_descriptors = {}
    args.execution_bindings = {}
    for label, snapshot in (
        ("executable", args.execution_snapshot_executable),
        ("library", args.execution_snapshot_library),
        ("mpiexec", args.execution_snapshot_mpiexec),
        ("mca", args.execution_snapshot_mca),
    ):
        descriptor, binding = create_execution_handle(
            snapshot, args.result_dir, label
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
    base_environment["LD_LIBRARY_PATH"] = (
        f"{archive}:{prefix / 'lib'}"
    )
    for key in (
        "OMPI_MCA_mca_base_param_files",
        "PMIX_MCA_mca_base_param_files",
        "PRTE_MCA_mca_base_param_files",
    ):
        base_environment[key] = str(args.execution_mca)
    initial_execution_identity = execution_identity(args)
    args.execution_baseline = stable_execution_artifacts(
        initial_execution_identity
    )
    protocol = {
        "schema_version": 1,
        "protocol_id": "m25-automatic-phase-batch-authoritative-v1",
        "created_at_utc": BASE.utc_now(),
        "receipt_id": receipt["receipt_id"],
        "artifact_set_id": receipt.get("artifact_set_id"),
        "matrix": matrix,
        "conditions": list(CONDITIONS),
        "condition_orders": [list(item) for item in CONDITION_ORDERS],
        "source_batching": "forced-identically-in-all-conditions",
        "gates": {
            "p128_minimum_median_paired_disabled_over_automatic": SMALL_MINIMUM_DISABLED_OVER_AUTOMATIC,
            "p256_maximum_median_paired_automatic_over_disabled": LARGE_MAXIMUM_AUTOMATIC_OVER_DISABLED,
            "p256_maximum_median_paired_automatic_over_forced": LARGE_MAXIMUM_AUTOMATIC_OVER_FORCED,
            "physical_observables": "exact-within-configuration",
            "cpu_field_calls": 0,
            "rank_to_gpu_mapping": "one-distinct-compatible-device-per-rank",
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
            "initial_identity": initial_execution_identity,
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
        "matrix": matrix,
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
        samples: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for configuration in matrix:
            selected: list[dict[str, Any]] = []
            for repeat in range(1, args.repeats + 1):
                order = CONDITION_ORDERS[(repeat - 1) % len(CONDITION_ORDERS)]
                for order_index, condition in enumerate(order, start=1):
                    sample = run_sample(
                        args,
                        base_environment,
                        configuration,
                        repeat,
                        order_index,
                        condition,
                    )
                    samples.append(sample)
                    selected.append(sample)
            summaries.append(summarize_configuration(configuration, selected))
        if monitor_process is None:
            raise RuntimeError("continuous GPU monitor did not start")
        expected_pid_to_gpu: dict[int, int] = {}
        for sample in samples:
            devices_by_rank = {
                item["rank"]: item["ordinal"] for item in sample["devices"]
            }
            for process_record in sample["processes"]:
                pid = process_record["pid"]
                gpu = devices_by_rank[process_record["rank"]]
                if pid in expected_pid_to_gpu:
                    raise RuntimeError(
                        "worker PID was reused across authoritative samples"
                    )
                expected_pid_to_gpu[pid] = gpu
        monitor = stop_process_monitor(
            monitor_process,
            monitor_stream,
            monitor_path,
            expected_pid_to_gpu,
        )
        monitor_stopped = True
        BASE.atomic_json(
            args.result_dir / "continuous-gpu-process-monitor.json", monitor
        )
        if not monitor["pass"]:
            raise RuntimeError(
                "continuous monitor did not prove exclusive benchmark GPU use"
            )
        receipt_after, executable_after, library_after, ldd_after = (
            BASE.resolve_receipt_artifacts(
                repo, args.build_receipt, args.expected_receipt_id, args.mpiexec
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
            or BASE.sha256_file(archive / "build-provenance.json")
            != BASE.sha256_file(args.build_receipt)
            or BASE.sha256_file(archive / args.executable.name)
            != BASE.sha256_file(args.executable)
            or BASE.sha256_file(archive / args.library.name)
            != BASE.sha256_file(args.library)
            or BASE.sha256_file(archive / "receipt-mpiexec")
            != BASE.sha256_file(args.mpiexec)
            or BASE.sha256_file(
                archive / "openmpi-qualification-mca-params.conf"
            )
            != BASE.sha256_file(mca_path)
            or stable_execution_artifacts(execution_identity(args))
            != args.execution_baseline
        ):
            raise RuntimeError("receipt, source, or measured artifact changed")
        summary = {
            "schema_version": 1,
            "finished_at_utc": BASE.utc_now(),
            "receipt_id": receipt["receipt_id"],
            "artifact_set_id": receipt.get("artifact_set_id"),
            "sample_count": len(samples),
            "all_observables_exact": True,
            "continuous_contention_check_pass": True,
            "configurations": summaries,
            "pass": all(item["pass"] for item in summaries),
        }
        BASE.atomic_json(args.result_dir / "summary.json", summary)
        BASE.atomic_text(args.result_dir / "report.md", render_markdown(summary))
        state = "COMPLETE" if summary["pass"] else "FAILED_GATES"
        BASE.atomic_json(
            args.result_dir / "run-state-final.json",
            {
                "schema_version": 1,
                "state": state,
                "finished_at_utc": BASE.utc_now(),
                "sample_count": len(samples),
                "receipt_id": receipt["receipt_id"],
            },
        )
        if summary["pass"]:
            finalize_success(args.result_dir, receipt["receipt_id"])
        else:
            finalize_failure(
                args.result_dir,
                {
                    "schema_version": 1,
                    "state": "FAILED_GATES",
                    "summary": BASE.file_record(
                        args.result_dir / "summary.json", args.result_dir
                    ),
                    "report": BASE.file_record(
                        args.result_dir / "report.md", args.result_dir
                    ),
                },
            )
        print_summary_best_effort(summary)
        return 0 if summary["pass"] else 2
    except Exception as error:
        if not monitor_stopped and monitor_process is not None:
            try:
                monitor = stop_process_monitor(
                    monitor_process,
                    monitor_stream,
                    monitor_path,
                    {
                        process_record["pid"]: {
                            item["rank"]: item["ordinal"]
                            for item in sample["devices"]
                        }[process_record["rank"]]
                        for sample in locals().get("samples", [])
                        for process_record in sample.get("processes", [])
                    },
                )
                BASE.atomic_json(
                    args.result_dir / "continuous-gpu-process-monitor.json",
                    monitor,
                )
            except Exception as monitor_error:
                BASE.atomic_json(
                    args.result_dir / "continuous-gpu-process-monitor-error.json",
                    {"error": f"{type(monitor_error).__name__}: {monitor_error}"},
                )
        finalize_failure(
            args.result_dir,
            {
                "schema_version": 1,
                "state": "FAILED_EXECUTION",
                "finished_at_utc": BASE.utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        for descriptor in args.execution_descriptors.values():
            os.close(descriptor)


def main() -> int:
    args = parse_args()
    args.result_dir = args.result_dir.resolve()
    args.build_receipt = args.build_receipt.resolve()
    args.mpiexec = args.mpiexec.resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_receipt_id):
        raise RuntimeError("expected receipt ID must be a lowercase SHA-256")
    if args.repeats < 3 or args.repeats % len(CONDITIONS) != 0:
        raise RuntimeError(
            "repeats must be a positive multiple of three for balanced order"
        )
    if args.timeout <= 0:
        raise RuntimeError("a positive timeout is required")
    if args.result_dir.exists() and any(args.result_dir.iterdir()):
        raise RuntimeError("authoritative result directory must be absent or empty")
    args.result_dir.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    if not globals().get("__gpmeep_internal_snapshot__"):
        reexec_immutable_source()
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
