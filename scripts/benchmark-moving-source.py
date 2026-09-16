#!/usr/bin/env python3
"""Benchmark a moving point source while retaining a numeric field oracle."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import tempfile
import time
from fractions import Fraction

# Capture the launcher-supplied cpuset before NumPy or Meep initializes an
# OpenMP runtime.  With OMP_PROC_BIND enabled, libgomp may narrow the main
# thread's affinity to one physical core during either import.
_LAUNCH_AFFINITY = frozenset(os.sched_getaffinity(0))

import numpy as np

import meep as mp

from gpmeep_provenance import (
    ProvenanceError,
    StatHashCache,
    verify_build_receipt,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_json_atomically(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=target.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    temporary_path.replace(target)
    return {
        "path": str(target.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def mapped_library(name_prefix):
    candidates = set()
    for line in Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith("/"):
            candidate = Path(columns[5])
            if candidate.name.startswith(name_prefix):
                candidates.add(candidate.resolve())
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one mapped {name_prefix}, found {len(candidates)}"
        )
    return next(iter(candidates))


def mapped_file_identity(path):
    resolved = Path(path).resolve()
    identities = set()
    deleted_mapping = False
    for line in Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) != 6 or not columns[5].startswith("/"):
            continue
        raw_path = columns[5]
        if raw_path.endswith(" (deleted)"):
            raw_path = raw_path[: -len(" (deleted)")]
            deleted = True
        else:
            deleted = False
        if Path(raw_path).resolve() != resolved:
            continue
        deleted_mapping = deleted_mapping or deleted
        major_text, minor_text = columns[3].split(":", 1)
        identities.add(
            (int(major_text, 16), int(minor_text, 16), int(columns[4]))
        )
    if deleted_mapping or len(identities) != 1:
        raise RuntimeError(
            f"loaded mapping identity is ambiguous or deleted: {resolved}"
        )
    stat = resolved.stat()
    current = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    if identities != {current}:
        raise RuntimeError(
            f"loaded mapping inode differs from current pathname: {resolved}"
        )
    return {
        "device_major": current[0],
        "device_minor": current[1],
        "inode": current[2],
    }


def task_affinities():
    affinities = []
    for task in sorted(Path("/proc/self/task").iterdir(), key=lambda p: int(p.name)):
        status = (task / "status").read_text(encoding="utf-8", errors="replace")
        allowed = next(
            line.split(":", 1)[1].strip()
            for line in status.splitlines()
            if line.startswith("Cpus_allowed_list:")
        )
        affinities.append({"tid": int(task.name), "cpus_allowed_list": allowed})
    return affinities


def cpu_topology(cpus):
    records = []
    for cpu in sorted(cpus):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        records.append(
            {
                "cpu": cpu,
                "core_id": int((topology / "core_id").read_text()),
                "physical_package_id": int(
                    (topology / "physical_package_id").read_text()
                ),
            }
        )
    return records


def cpu_identity():
    first_processor = Path("/proc/cpuinfo").read_text(
        encoding="utf-8", errors="replace"
    ).split("\n\n", 1)[0]
    values = {}
    for line in first_processor.splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            values[name.strip()] = value.strip()
    required = (
        "vendor_id",
        "cpu family",
        "model",
        "model name",
        "stepping",
        "microcode",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError("missing CPU identity fields: " + ", ".join(missing))
    return {name: values[name] for name in required}


def runtime_contract(args):
    receipt_path = os.environ.get("GPMEEP_VALIDATION_BUILD_RECEIPT_FILE")
    if not receipt_path:
        raise RuntimeError("GPMEEP_VALIDATION_BUILD_RECEIPT_FILE is required")
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    receipt_repo_value = receipt.get("repo")
    if not isinstance(receipt_repo_value, str):
        raise RuntimeError("benchmark build receipt has no repository root")
    receipt_repo = Path(receipt_repo_value).resolve()
    if receipt_repo != Path.cwd().resolve():
        raise RuntimeError("benchmark must run from the receipt repository root")
    try:
        receipt = verify_build_receipt(
            Path(receipt_path).resolve(),
            receipt_repo,
            verify_source=True,
            hash_cache=StatHashCache(),
        )
    except (OSError, ProvenanceError, RuntimeError) as error:
        raise RuntimeError(
            f"benchmark build receipt failed full verification: {error}"
        ) from error
    expected_receipt = os.environ.get("GPMEEP_VALIDATION_BUILD_RECEIPT_ID")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_receipt or ""):
        raise RuntimeError("a lowercase SHA-256 build receipt ID is required")
    if receipt.get("receipt_id") != expected_receipt:
        raise RuntimeError("benchmark build receipt ID does not match receipt file")
    expected_result = os.environ.get("GPMEEP_VALIDATION_RESULT_FILE")
    if Path(expected_result or "").resolve() != args.output.resolve():
        raise RuntimeError("benchmark result path does not match evidence contract")
    expected_complete = os.environ.get("GPMEEP_VALIDATION_COMPLETE_FILE")
    if not expected_complete:
        raise RuntimeError("GPMEEP_VALIDATION_COMPLETE_FILE is required")

    environment_names = (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "MPLCONFIGDIR",
        "MPLBACKEND",
        "FONTCONFIG_FILE",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "OMP_WAIT_POLICY",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MEEP_GPU_BACKEND",
        "MEEP_GPU_STRICT",
        "MEEP_GPU_AUTO_MIN_CELLS",
        "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS",
        "CUDA_CACHE_DISABLE",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "GPMEEP_VALIDATION_EXPECTED_BACKEND",
        "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND",
        "GPMEEP_VALIDATION_STRICT_CUDA",
        "GPMEEP_VALIDATION_RUN_NONCE",
        "GPMEEP_VALIDATION_BUILD_RECEIPT_ID",
        "GPMEEP_VALIDATION_BUILD_RECEIPT_FILE",
        "GPMEEP_VALIDATION_STATS_FILE",
        "GPMEEP_VALIDATION_RESULT_FILE",
        "GPMEEP_VALIDATION_COMPLETE_FILE",
        "GPMEEP_VALIDATION_EXECUTION_CLASS",
        "GPMEEP_VALIDATION_CONCURRENT_GPU_WORK",
    )
    current_affinity = os.sched_getaffinity(0)
    topology = cpu_topology(_LAUNCH_AFFINITY)
    physical_cores = {
        (entry["physical_package_id"], entry["core_id"]) for entry in topology
    }
    if len(physical_cores) != args.expected_physical_cores:
        raise RuntimeError(
            "process affinity exposes "
            f"{len(physical_cores)} physical cores, expected "
            f"{args.expected_physical_cores}"
        )
    expected_backend = os.environ.get("GPMEEP_VALIDATION_EXPECTED_BACKEND")
    if expected_backend not in ("cpu", "cuda"):
        raise RuntimeError("GPMEEP_VALIDATION_EXPECTED_BACKEND must be cpu or cuda")
    expected_requested_backend = os.environ.get(
        "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND", expected_backend
    )
    if expected_requested_backend not in ("cpu", "auto", "cuda"):
        raise RuntimeError(
            "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND must be cpu, auto, "
            "or cuda"
        )
    run_nonce = os.environ.get("GPMEEP_VALIDATION_RUN_NONCE")
    if not re.fullmatch(r"[0-9a-f]{32}", run_nonce or ""):
        raise RuntimeError("a 128-bit lowercase hexadecimal run nonce is required")
    execution_class = os.environ.get("GPMEEP_VALIDATION_EXECUTION_CLASS")
    if not execution_class:
        raise RuntimeError("GPMEEP_VALIDATION_EXECUTION_CLASS is required")
    if os.environ.get("GPMEEP_VALIDATION_CONCURRENT_GPU_WORK") != "0":
        raise RuntimeError("formal benchmark must declare no concurrent GPU work")
    if mp.count_processors() != 1:
        raise RuntimeError(
            "benchmark-moving-source.py is a singleton producer; fair CPU "
            "performance baselines must use the separate multi-rank runner"
        )
    # The core CPU FDTD stencil is serial within an MPI rank, so a singleton
    # CPU row always reserves one host thread and is explicitly non-release.
    # CUDA may use the same physical-core budget for auxiliary host work; that
    # does not misrepresent those threads as parallel CPU Yee workers.
    expected_omp_threads = (
        args.expected_physical_cores if expected_backend == "cuda" else 1
    )
    if os.environ.get("OMP_NUM_THREADS") != str(expected_omp_threads):
        raise RuntimeError(
            f"{expected_backend} benchmark requires OMP_NUM_THREADS="
            f"{expected_omp_threads}"
        )
    if os.environ.get("OMP_DYNAMIC", "").upper() != "FALSE":
        raise RuntimeError("formal benchmark requires OMP_DYNAMIC=FALSE")
    if os.environ.get("OMP_PROC_BIND", "").lower() not in ("close", "spread"):
        raise RuntimeError("formal benchmark requires OpenMP thread binding")
    if os.environ.get("OMP_PLACES", "").lower() != "cores":
        raise RuntimeError("formal benchmark requires OMP_PLACES=cores")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise RuntimeError("formal benchmark requires OPENBLAS_NUM_THREADS=1")
    fontconfig_file = os.environ.get("FONTCONFIG_FILE")
    if not fontconfig_file:
        raise RuntimeError("formal benchmark requires FONTCONFIG_FILE")
    extension = getattr(mp, "_meep")
    libmeep_path = mapped_library("libmeep.so")
    if mp.gpu.requested_backend != expected_requested_backend:
        raise RuntimeError("requested backend does not match benchmark contract")
    if expected_requested_backend != "auto" and mp.gpu.active_backend != expected_backend:
        raise RuntimeError("initial active backend does not match benchmark contract")
    if expected_requested_backend == "cuda" and (
        os.environ.get("GPMEEP_VALIDATION_STRICT_CUDA") != "1"
        or os.environ.get("MEEP_GPU_STRICT") != "1"
    ):
        raise RuntimeError("explicit CUDA benchmark requires both strict-mode markers")
    if expected_requested_backend != "cuda" and (
        os.environ.get("GPMEEP_VALIDATION_STRICT_CUDA") == "1"
        or os.environ.get("MEEP_GPU_STRICT") == "1"
    ):
        raise RuntimeError("strict-mode markers require an explicit CUDA request")
    if not mp.is_single_precision():
        raise RuntimeError("benchmark requires the FP32 gpmeep build")
    if not mp.with_mpi():
        raise RuntimeError("benchmark requires an MPI-enabled gpmeep build")
    contract = {
        "argv": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd().resolve()),
        "environment": {name: os.environ.get(name) for name in environment_names},
        "launch_affinity_before_openmp_imports": sorted(_LAUNCH_AFFINITY),
        "main_thread_affinity_after_openmp_imports": sorted(current_affinity),
        "physical_core_count": len(physical_cores),
        "expected_physical_core_count": args.expected_physical_cores,
        "mpi_rank_count": mp.count_processors(),
        "parallelism_contract": (
            "serial-single-rank-diagnostic"
            if expected_backend == "cpu"
            else "single-rank-gpu"
        ),
        "performance_claim_eligible": expected_backend != "cpu",
        "task_affinities_before": task_affinities(),
        "cpu_topology": topology,
        "cpu_identity": cpu_identity(),
        "fontconfig_file": file_record(fontconfig_file),
        "platform": list(platform.uname()),
        "python_executable": file_record(sys.executable),
        "meep_module": file_record(mp.__file__),
        "python_extension": file_record(extension.__file__),
        "python_extension_mapping": mapped_file_identity(extension.__file__),
        "libmeep": file_record(libmeep_path),
        "libmeep_mapping": mapped_file_identity(libmeep_path),
        "build_receipt": file_record(receipt_path),
        "verified_build_receipt": {
            "receipt_id": receipt["receipt_id"],
            "artifact_set_id": receipt["artifact_set_id"],
            "build_input_id": receipt["build_input_id"],
        },
        "requested_backend": mp.gpu.requested_backend,
        "active_backend": mp.gpu.active_backend,
        "backend_diagnostic": mp.gpu.backend_diagnostic,
        "selected_device": mp.gpu.selected_device,
        "expected_requested_backend": expected_requested_backend,
        "expected_final_backend": expected_backend,
        "single_precision": bool(mp.is_single_precision()),
        "with_mpi": bool(mp.with_mpi()),
    }
    return contract


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--expected-physical-cores", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def expected_workload_contract(resolution):
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
        raise ValueError("moving-source resolution must be a positive integer")
    sx = Fraction("40.0")
    sy = Fraction("40.0")
    speed = Fraction("0.7")
    courant = Fraction(1, 2)
    requested = sx / speed
    timestep = courant / resolution
    ratio = requested / timestep
    expected_steps = (ratio.numerator + ratio.denominator - 1) // ratio.denominator
    return {
        "sx": float(sx),
        "sy": float(sy),
        "dpml": 1.0,
        "source_speed": float(speed),
        "source_frequency": 1e-10,
        "courant": float(courant),
        "requested_until": float(requested),
        "expected_timesteps": expected_steps,
        "expected_meep_time": float(expected_steps * timestep),
        "expected_source_updates": expected_steps + 1,
        "expected_field_shape": [int(sx * resolution), int(sy * resolution)],
    }


def validate_metric(metric, contract):
    required = (
        "sum", "l1", "l2", "maximum_absolute", "meep_time"
    )
    if metric.get("shape") != contract["expected_field_shape"]:
        raise RuntimeError("moving-source field shape differs from the workload contract")
    if metric.get("source_updates") != contract["expected_source_updates"]:
        raise RuntimeError("moving-source update count differs from the workload contract")
    if not math.isclose(
        metric.get("meep_time", math.nan),
        contract["expected_meep_time"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("moving-source end time differs from the workload contract")
    if not all(
        isinstance(metric.get(name), (int, float))
        and not isinstance(metric.get(name), bool)
        and math.isfinite(float(metric[name]))
        for name in required
    ):
        raise RuntimeError("moving-source metric contains a non-finite value")
    if any(metric[name] < 0 for name in ("l1", "l2", "maximum_absolute")):
        raise RuntimeError("moving-source norm metric must be nonnegative")


def run_once(resolution):
    workload = expected_workload_contract(resolution)
    sx = workload["sx"]
    sy = workload["sy"]
    dpml = workload["dpml"]
    speed = workload["source_speed"]
    cell_size = mp.Vector3(sx, sy, 0)
    simulation = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        default_material=mp.Medium(index=1.5),
        symmetries=[mp.Mirror(direction=mp.Y)],
        boundary_layers=[mp.PML(thickness=dpml)],
    )
    source_updates = 0

    def move_source(active_simulation):
        nonlocal source_updates
        current_time = float(active_simulation.meep_time())
        current_x = -0.5 * sx + dpml + speed * current_time
        active_simulation.change_sources(
            [
                mp.Source(
                    mp.ContinuousSource(frequency=1e-10),
                    component=mp.Ex,
                    center=mp.Vector3(current_x),
                )
            ]
        )
        source_updates += 1

    started = time.perf_counter()
    simulation.run(move_source, until=sx / speed)
    solver_seconds = time.perf_counter() - started
    field = np.asarray(
        simulation.get_array(
            center=mp.Vector3(), size=cell_size, component=mp.Hz
        )
    )
    metric = {
        "shape": list(field.shape),
        "sum": float(np.sum(field, dtype=np.float64)),
        "l1": float(np.sum(np.abs(field), dtype=np.float64)),
        "l2": float(np.linalg.norm(field)),
        "maximum_absolute": float(np.max(np.abs(field))),
        "source_updates": source_updates,
        "meep_time": float(simulation.meep_time()),
    }
    validate_metric(metric, workload)
    owner_cuda = bool(simulation.fields.gpu_cuda_execution_selected())
    owner_backend = "cuda" if owner_cuda else "cpu"
    owner_diagnostic = simulation.fields.gpu_execution_diagnostic()
    expected_owner_marker = "CUDA" if owner_cuda else "selected CPU"
    if expected_owner_marker not in owner_diagnostic:
        raise RuntimeError(
            "fields owner query returned no completed backend decision: "
            + owner_diagnostic
        )
    if mp.gpu.active_backend != owner_backend:
        raise RuntimeError("fields owner and process active backends disagree")
    if mp.gpu.backend_diagnostic != owner_diagnostic:
        raise RuntimeError("fields owner and process backend diagnostics disagree")
    backend_decision = {
        "requested_backend": mp.gpu.requested_backend,
        "active_backend": owner_backend,
        "backend_diagnostic": owner_diagnostic,
        "process_active_backend": mp.gpu.active_backend,
        "process_backend_diagnostic": mp.gpu.backend_diagnostic,
        "selected_device": mp.gpu.selected_device,
    }
    simulation.reset_meep()
    del simulation
    gc.collect()
    return solver_seconds, metric, backend_decision


def main():
    args = parse_args()
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if args.repeat <= 0:
        raise ValueError("repeat must be positive")
    if args.expected_physical_cores <= 0:
        raise ValueError("expected physical core count must be positive")

    evidence_paths = {
        args.output.resolve(),
        Path(os.environ["GPMEEP_VALIDATION_STATS_FILE"]).resolve(),
        Path(os.environ["GPMEEP_VALIDATION_COMPLETE_FILE"]).resolve(),
    }
    preexisting = sorted(str(path) for path in evidence_paths if path.exists())
    if preexisting:
        raise RuntimeError(
            "formal benchmark evidence paths must be fresh: "
            + ", ".join(preexisting)
        )

    contract = runtime_contract(args)
    workload_contract = expected_workload_contract(args.resolution)
    expected_backend = os.environ["GPMEEP_VALIDATION_EXPECTED_BACKEND"]
    expected_requested_backend = os.environ.get(
        "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND", expected_backend
    )
    load_average_before = list(os.getloadavg())
    durations = []
    metrics = []
    backend_decisions = []
    statistics_by_repeat = []
    mp.gpu.reset_statistics()
    for _ in range(args.repeat):
        duration, metric, decision = run_once(args.resolution)
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("moving-source solver duration must be finite and positive")
        if decision["requested_backend"] != expected_requested_backend:
            raise RuntimeError("requested backend changed during benchmark")
        if decision["active_backend"] != expected_backend:
            raise RuntimeError(
                "workload selected "
                f"{decision['active_backend']}, expected {expected_backend}: "
                f"{decision['backend_diagnostic']}"
            )
        durations.append(duration)
        metrics.append(metric)
        backend_decisions.append(decision)
        statistics_by_repeat.append(mp.gpu.statistics())

    if any(metric != metrics[0] for metric in metrics[1:]):
        raise RuntimeError("moving-source numerical metric changed across repeats")

    decision_signatures = {
        (
            decision["requested_backend"],
            decision["active_backend"],
            decision["backend_diagnostic"],
            json.dumps(decision["selected_device"], sort_keys=True),
        )
        for decision in backend_decisions
    }
    if len(decision_signatures) != 1:
        raise RuntimeError("automatic backend decision changed across repeats")

    receipt_path = Path(
        os.environ["GPMEEP_VALIDATION_BUILD_RECEIPT_FILE"]
    ).resolve()
    try:
        ending_receipt = verify_build_receipt(
            receipt_path,
            Path.cwd().resolve(),
            verify_source=True,
            hash_cache=StatHashCache(),
        )
    except (OSError, ProvenanceError, RuntimeError) as error:
        raise RuntimeError(
            f"build receipt changed while benchmark was running: {error}"
        ) from error
    ending_identity = {
        "receipt_id": ending_receipt["receipt_id"],
        "artifact_set_id": ending_receipt["artifact_set_id"],
        "build_input_id": ending_receipt["build_input_id"],
    }
    if ending_identity != contract["verified_build_receipt"]:
        raise RuntimeError("build receipt identity changed during benchmark")
    contract["verified_build_receipt_after_runs"] = ending_identity

    contract["task_affinities_after"] = task_affinities()
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    contract["invocation_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()

    result = {
        "schema_version": 2,
        "backend": expected_backend,
        "requested_backend": expected_requested_backend,
        "parallelism_contract": contract["parallelism_contract"],
        "performance_claim_eligible": contract["performance_claim_eligible"],
        "resolution": args.resolution,
        "repeat": args.repeat,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "solver_seconds": durations,
        "solver_seconds_median": statistics.median(durations),
        "metrics": metrics,
        "backend_decisions": backend_decisions,
        "workload_contract": workload_contract,
        "statistics_by_repeat": statistics_by_repeat,
        "runtime_contract": contract,
        "load_average_before": load_average_before,
        "load_average_after": list(os.getloadavg()),
        "gpu_statistics": statistics_by_repeat[-1],
    }
    result_record = write_json_atomically(args.output, result)
    evidence_hook = sys.modules.get("sitecustomize")
    register = getattr(
        evidence_hook, "register_gpmeep_benchmark_evidence", None
    )
    if not callable(register):
        raise RuntimeError(
            "formal benchmark requires the gpmeep sitecustomize evidence hook"
        )
    register(contract["invocation_sha256"], result_record["sha256"])
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
