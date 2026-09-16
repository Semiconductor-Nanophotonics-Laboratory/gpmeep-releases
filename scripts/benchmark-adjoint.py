#!/usr/bin/env python3
"""Produce one complete forward/adjoint MaterialGrid benchmark record.

The fixed post-source run time prevents CPU/CUDA roundoff from changing the
number of timesteps.  Unlike a timing-only benchmark, this producer persists
the entire gradient and enough runtime/workload evidence for the parent gate
to prove that both backends performed the same calculation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import socket
import stat
import subprocess
import sys
import time
import types
from typing import Any, Callable

from autograd import numpy as npa
import numpy as np

import meep as mp
import meep.adjoint as mpa


_DEFERRED_IMPORT_ROOTS = ("jax", "matplotlib", "scipy")
_CORE_IMPORT_SURFACE = {
    "meep.Simulation": callable(getattr(mp, "Simulation", None)),
    "meep.adjoint.DesignRegion": callable(getattr(mpa, "DesignRegion", None)),
    "meep.adjoint.FourierFields": callable(getattr(mpa, "FourierFields", None)),
    "meep.adjoint.OptimizationProblem": callable(
        getattr(mpa, "OptimizationProblem", None)
    ),
}
_LAZY_IMPORT_CONTRACT = {
    "schema_version": 1,
    "capture_point": "immediately-after-meep-and-meep-adjoint-import",
    "deferred_module_roots": list(_DEFERRED_IMPORT_ROOTS),
    "deferred_module_loaded": {
        name: name in sys.modules for name in _DEFERRED_IMPORT_ROOTS
    },
    "core_surface": dict(sorted(_CORE_IMPORT_SURFACE.items())),
    "lazy_surface_discoverable": {
        "meep.Animate2D": "Animate2D" in dir(mp),
        "meep.plot2D": "plot2D" in dir(mp),
        "meep.adjoint.FilteredSource": "FilteredSource" in dir(mpa),
        "meep.adjoint.MeepJaxWrapper": "MeepJaxWrapper" in dir(mpa),
        "meep.adjoint.conic_filter": "conic_filter" in dir(mpa),
    },
}
_LAZY_IMPORT_CONTRACT["pass"] = bool(
    not any(_LAZY_IMPORT_CONTRACT["deferred_module_loaded"].values())
    and all(_LAZY_IMPORT_CONTRACT["core_surface"].values())
    and all(_LAZY_IMPORT_CONTRACT["lazy_surface_discoverable"].values())
)


def _load_source_module(name: str, path: pathlib.Path) -> types.ModuleType:
    """Load receipt-bound benchmark helpers from source rather than .pyc."""

    resolved = path.resolve()
    module = types.ModuleType(name)
    module.__file__ = str(resolved)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(resolved.read_bytes(), str(resolved), "exec"), module.__dict__)
    return module


_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
_EVIDENCE = _load_source_module(
    "gpmeep_benchmark_evidence", _SCRIPT_DIRECTORY / "gpmeep_benchmark_evidence.py"
)
_PROVENANCE = _load_source_module(
    "gpmeep_provenance", _SCRIPT_DIRECTORY / "gpmeep_provenance.py"
)
atomic_write_json = _EVIDENCE.atomic_write_json
canonical_float64_sha256 = _EVIDENCE.canonical_float64_sha256
loaded_library_records = _EVIDENCE.loaded_library_records
loaded_library_path = _EVIDENCE.loaded_library_path
loaded_mapped_file_records = _EVIDENCE.loaded_mapped_file_records
phase_counter_view = _EVIDENCE.phase_counter_view
sha256_file = _EVIDENCE.sha256_file
statistics_delta = _EVIDENCE.statistics_delta
tree_manifest = _PROVENANCE.tree_manifest


PREFIX = "gpmeep-adjoint-benchmark:"
QUALIFICATION_PROFILE = "m19-cpu-gpu-adjoint-release-v4"
CPU_PARALLEL_QUALIFICATION_PROFILE = "m8.7-cpu-parallel-tuning-v1"
QUALIFICATION_PROFILES = (
    QUALIFICATION_PROFILE,
    CPU_PARALLEL_QUALIFICATION_PROFILE,
)
EVIDENCE_ENVIRONMENT_KEYS = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "GPMEEP_VALIDATION_STRICT_CUDA",
    "MEEP_GPU_BACKEND",
    "MEEP_GPU_DEVICE",
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
)
TRUSTED_NVIDIA_SMI_DIRECTORIES = (
    pathlib.Path("/usr/bin"),
    pathlib.Path("/usr/sbin"),
    pathlib.Path("/bin"),
    pathlib.Path("/sbin"),
    pathlib.Path("/usr/local/cuda/bin"),
    pathlib.Path("/usr/lib/wsl/lib"),
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _process_cpu_affinity() -> list[int]:
    """Return the kernel-enforced logical-CPU set for this rank."""

    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("qualification requires sched_getaffinity support")
    affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    if not affinity:
        raise RuntimeError("process CPU affinity is empty")
    return affinity


def _process_thread_cpu_affinities() -> dict[str, Any]:
    """Capture every live thread's kernel affinity and their process-wide union."""

    task_directory = pathlib.Path("/proc/self/task")
    if not task_directory.is_dir():
        raise RuntimeError("qualification requires /proc/self/task")
    records = []
    union: set[int] = set()
    for task in sorted(task_directory.iterdir(), key=lambda path: int(path.name)):
        try:
            thread_id = int(task.name)
            logical_cpus = sorted(int(cpu) for cpu in os.sched_getaffinity(thread_id))
        except (OSError, ValueError, ProcessLookupError):
            continue
        if not logical_cpus:
            raise RuntimeError(f"thread {thread_id} has empty CPU affinity")
        union.update(logical_cpus)
        records.append({"thread_id": thread_id, "logical_cpus": logical_cpus})
    if not records or not union:
        raise RuntimeError("process thread-affinity evidence is empty")
    return {
        "thread_count": len(records),
        "logical_cpu_union": sorted(union),
        "threads": records,
    }


def _start_workload_timer() -> float:
    if mp.with_mpi():
        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()
    return time.perf_counter()


def _distributed_max(value: float) -> float:
    if not mp.with_mpi():
        return value
    from mpi4py import MPI

    return float(MPI.COMM_WORLD.allreduce(value, op=MPI.MAX))


def _rank_local_nvidia_driver_evidence() -> dict[str, Any]:
    executable: pathlib.Path | None = None
    for directory in TRUSTED_NVIDIA_SMI_DIRECTORIES:
        candidate = directory / "nvidia-smi"
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        try:
            resolved = candidate.resolve(strict=True)
            if any(
                path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                for path in (directory.resolve(strict=True), resolved.parent, resolved)
            ):
                continue
        except OSError:
            continue
        executable = resolved
        break
    if executable is None:
        raise RuntimeError("rank-local nvidia-smi is absent from protected system paths")

    def probe(arguments: list[str]) -> dict[str, Any]:
        process = subprocess.run(
            [str(executable), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        return {
            "command": [str(executable), *arguments],
            "exit_code": process.returncode,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        }

    return {
        "hostname": socket.gethostname(),
        "executable": {
            "path": str(executable),
            "size_bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
        "inventory": probe(
            [
                "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "topology": probe(["-L"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Yee-grid pixels per distance unit (default: 64)",
    )
    parser.add_argument(
        "--run-time",
        type=float,
        default=40,
        help="fixed post-source forward and adjoint run time (default: 40)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=8,
        help="square computational-cell side length (default: 8)",
    )
    parser.add_argument(
        "--design-resolution",
        type=int,
        default=20,
        help="MaterialGrid pixels per distance unit (default: 20)",
    )
    parser.add_argument(
        "--result-file",
        type=pathlib.Path,
        required=True,
        help="atomic JSON destination for the complete benchmark record",
    )
    parser.add_argument("--distributed-objective-atol", type=float, default=1e-8)
    parser.add_argument("--distributed-gradient-atol", type=float, default=2e-7)
    parser.add_argument("--distributed-gradient-rtol", type=float, default=5e-5)
    parser.add_argument(
        "--qualification-profile",
        choices=QUALIFICATION_PROFILES,
        help="enable receipt-bound raw qualification evidence",
    )
    parser.add_argument("--run-nonce")
    parser.add_argument("--qualification-run-id")
    parser.add_argument("--build-receipt-id")
    parser.add_argument("--source-snapshot-sha256")
    parser.add_argument("--producer-sha256")
    parser.add_argument(
        "--sample-kind", choices=("warmup", "measured"), default="measured"
    )
    parser.add_argument(
        "--sample-lane",
        choices=(
            "cpu-legacy-oracle",
            "cpu-2x4",
            "cpu-parallel",
            "cuda-single",
            "cuda-multi",
        ),
        default="cuda-single",
    )
    parser.add_argument("--sample-iteration", type=int, default=0)
    parser.add_argument(
        "--directional-fd-step",
        type=float,
        default=0.0,
        help="central finite-difference step; qualification uses this only for the CPU oracle",
    )
    parser.add_argument(
        "--capture-field-times",
        action="store_true",
        help="record per-rank Meep field timing sinks for performance diagnosis",
    )
    return parser.parse_args()


def _timed(function: Callable[[], Any]) -> tuple[Any, float, float]:
    world = None
    if mp.with_mpi():
        from mpi4py import MPI

        world = MPI.COMM_WORLD
        world.Barrier()
    started = time.perf_counter()
    value = function()
    local_elapsed = time.perf_counter() - started
    distributed_elapsed = local_elapsed
    if world is not None:
        distributed_elapsed = world.allreduce(local_elapsed, op=MPI.MAX)
    return value, distributed_elapsed, local_elapsed


def _phase_record(
    before: dict[str, Any],
    after: dict[str, Any],
    elapsed_seconds: float,
    *,
    timesteps: int | None = None,
    end_time: float | None = None,
) -> dict[str, Any]:
    delta = statistics_delta(before, after)
    result: dict[str, Any] = {
        "elapsed_seconds": elapsed_seconds,
        "statistics_delta": delta,
        "backend_counters": {
            "cpu": phase_counter_view(delta, "cpu"),
            "cuda": phase_counter_view(delta, "cuda"),
        },
    }
    if timesteps is not None:
        result["timesteps"] = int(timesteps)
    if end_time is not None:
        result["end_time"] = float(end_time)
    return result


def _field_timing_snapshot(simulation: Any) -> dict[str, list[float]]:
    sinks = (
        ("connecting_chunks", mp.Connecting),
        ("time_stepping", mp.Stepping),
        ("boundaries", mp.Boundaries),
        ("mpi_all_to_all", mp.MpiAllTime),
        ("mpi_one_to_one", mp.MpiOneTime),
        ("fourier_transform", mp.FourierTransforming),
        ("other", mp.Other),
        ("field_update_b", mp.FieldUpdateB),
        ("field_update_h", mp.FieldUpdateH),
        ("field_update_d", mp.FieldUpdateD),
        ("field_update_e", mp.FieldUpdateE),
        ("boundary_stepping_b", mp.BoundarySteppingB),
        ("boundary_stepping_wh", mp.BoundarySteppingWH),
        ("boundary_stepping_ph", mp.BoundarySteppingPH),
        ("boundary_stepping_h", mp.BoundarySteppingH),
        ("boundary_stepping_d", mp.BoundarySteppingD),
        ("boundary_stepping_we", mp.BoundarySteppingWE),
        ("boundary_stepping_pe", mp.BoundarySteppingPE),
        ("boundary_stepping_e", mp.BoundarySteppingE),
    )
    timing_data = simulation.get_timing_data()
    return {
        name: [float(value) for value in timing_data[sink]]
        for name, sink in sinks
    }


def _runtime_closure(mp_module: Any) -> dict[str, Any]:
    python_executable = pathlib.Path(sys.executable).resolve()
    meep_module = pathlib.Path(mp_module.__file__).resolve()
    package_root = meep_module.parent
    extension = pathlib.Path(mp_module._meep.__file__).resolve()
    libmeep = loaded_library_path("libmeep.so")
    package_manifest = tree_manifest(package_root, package_root.parent)
    environment_prefix = python_executable.parent.parent
    loaded_libraries = loaded_library_records(
        {
            "mpi": ("libmpi.so",),
            "cuda_driver": ("libcuda.so",),
            "cuda_runtime": ("libcudart.so",),
            "hdf5": ("libhdf5",),
        }
    )
    loaded_libraries["environment"] = loaded_mapped_file_records(
        environment_prefix
    )
    return {
        "python_executable": {
            "path": str(python_executable),
            "sha256": sha256_file(python_executable),
        },
        "meep_module": {
            "path": str(meep_module),
            "sha256": sha256_file(meep_module),
        },
        "python_package": {
            "root": str(package_root),
            "file_count": package_manifest["file_count"],
            "sha256": package_manifest["sha256"],
        },
        "extension": {
            "path": str(extension),
            "sha256": sha256_file(extension),
        },
        "libmeep": {
            "path": str(libmeep),
            "sha256": sha256_file(libmeep),
        },
        "loaded_libraries": loaded_libraries,
    }


def _qualification_binding(args: argparse.Namespace) -> dict[str, Any]:
    fields = {
        "profile": args.qualification_profile,
        "run_nonce": args.run_nonce,
        "run_id": args.qualification_run_id,
        "build_receipt_id": args.build_receipt_id,
        "source_snapshot_sha256": args.source_snapshot_sha256,
        "producer_sha256": args.producer_sha256,
    }
    qualification_requested = any(value is not None for value in fields.values())
    if not qualification_requested:
        return {"mode": "diagnostic"}
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise ValueError(
            "qualification evidence binding is incomplete: " + ", ".join(missing)
        )
    if args.qualification_profile not in QUALIFICATION_PROFILES:
        raise ValueError("unsupported qualification profile")
    allowed_lanes = (
        {"cpu-parallel"}
        if args.qualification_profile == CPU_PARALLEL_QUALIFICATION_PROFILE
        else {"cpu-legacy-oracle", "cpu-2x4", "cuda-single", "cuda-multi"}
    )
    if args.sample_lane not in allowed_lanes:
        raise ValueError("sample lane is incompatible with the qualification profile")
    if args.producer_sha256 != sha256_file(pathlib.Path(__file__).resolve()):
        raise RuntimeError("producer SHA-256 does not match the executing script")
    if args.sample_iteration < 0:
        raise ValueError("sample iteration must be nonnegative")
    return {
        "mode": "qualification",
        **fields,
        "sample_kind": args.sample_kind,
        "sample_lane": args.sample_lane,
        "sample_iteration": args.sample_iteration,
    }


def main() -> None:
    args = parse_args()
    started_at_utc = _utc_now()
    qualification = _qualification_binding(args)
    if min(
        args.resolution,
        args.run_time,
        args.cell_size,
        args.design_resolution,
    ) <= 0:
        raise ValueError("all workload dimensions and run times must be positive")
    if args.directional_fd_step < 0:
        raise ValueError("directional finite-difference step must be nonnegative")

    gpu = getattr(mp, "gpu", None)
    if gpu is None:
        raise RuntimeError("gpmeep GPU-control API is unavailable")

    # Capture the launcher's complete rank cpuset before the first OpenMP
    # parallel region may pin the calling thread to one OMP place.
    launch_cpu_affinity = _process_cpu_affinity()
    mp.verbosity(0)
    workload_started = _start_workload_timer()
    cell = mp.Vector3(args.cell_size, args.cell_size)
    pml = 1.0
    design_size = mp.Vector3(2.0, 2.0)
    design_nx = int(round(design_size.x * args.design_resolution)) + 1
    design_ny = int(round(design_size.y * args.design_resolution)) + 1

    low_index = mp.Medium(index=1.45)
    high_index = mp.Medium(index=3.4)
    design = mp.MaterialGrid(
        mp.Vector3(design_nx, design_ny),
        low_index,
        high_index,
        do_averaging=True,
    )
    design_region = mpa.DesignRegion(
        design,
        volume=mp.Volume(center=mp.Vector3(), size=design_size),
    )

    fcen = 1 / 1.55
    sources = [
        mp.Source(
            mp.GaussianSource(frequency=fcen, fwidth=0.30 * fcen),
            component=mp.Ez,
            center=mp.Vector3(-0.5 * args.cell_size + pml + 0.5),
            size=mp.Vector3(y=args.cell_size - 2 * pml),
        )
    ]
    geometry = [
        mp.Block(
            material=high_index,
            center=mp.Vector3(),
            size=mp.Vector3(mp.inf, 0.8),
        ),
        mp.Block(
            material=design,
            center=design_region.center,
            size=design_region.size,
        ),
    ]
    sim = mp.Simulation(
        resolution=args.resolution,
        cell_size=cell,
        boundary_layers=[mp.PML(pml)],
        sources=sources,
        geometry=geometry,
    )

    fourier_fields = mpa.FourierFields(
        sim,
        mp.Volume(
            center=mp.Vector3(0.5 * design_size.x + 0.5),
            size=mp.Vector3(0, 1.0),
        ),
        mp.Ez,
    )

    def objective(fields):
        return npa.mean(npa.abs(fields) ** 2)

    problem = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=objective,
        objective_arguments=[fourier_fields],
        design_regions=[design_region],
        frequencies=[fcen],
        minimum_run_time=args.run_time,
        maximum_run_time=args.run_time,
    )

    grid_x = np.linspace(-1.0, 1.0, design_nx)[:, None]
    grid_y = np.linspace(-1.0, 1.0, design_ny)[None, :]
    weights = (
        0.5
        + 0.20 * np.cos(np.pi * grid_x) * np.cos(2 * np.pi * grid_y)
        + 0.05 * np.sin(3 * np.pi * grid_x + np.pi * grid_y)
    )
    weights = np.clip(weights, 0.05, 0.95).astype(np.float64, copy=False)
    weights_vector = weights.reshape(-1)

    direction = np.sin(
        np.arange(weights.size, dtype=np.float64) * np.float64(0.61803398875)
    )
    direction /= np.linalg.norm(direction)

    gpu.reset_statistics()
    baseline_statistics = gpu.statistics()

    _, update_seconds, local_update_seconds = _timed(
        lambda: problem.update_design(rho_vector=[weights_vector])
    )
    after_update = gpu.statistics()
    _, forward_seconds, local_forward_seconds = _timed(problem.forward_run)
    forward_timesteps = int(sim.timestep())
    forward_end_time = float(sim.meep_time())
    after_forward = gpu.statistics()
    forward_field_times = (
        _field_timing_snapshot(sim) if args.capture_field_times else None
    )
    _, adjoint_seconds, local_adjoint_seconds = _timed(problem.adjoint_run)
    adjoint_timesteps = int(sim.timestep())
    adjoint_end_time = float(sim.meep_time())
    after_adjoint = gpu.statistics()
    adjoint_field_times = (
        _field_timing_snapshot(sim) if args.capture_field_times else None
    )
    _, gradient_seconds, local_gradient_seconds = _timed(
        problem.calculate_gradient
    )
    after_gradient = gpu.statistics()

    value = problem.f0
    gradient_array = np.asarray(problem.gradient, dtype=np.float64)
    gradient_shape = list(gradient_array.shape)
    gradient = gradient_array.reshape(-1)
    if gradient.size != weights.size:
        raise RuntimeError(
            f"gradient contains {gradient.size} values; expected {weights.size}"
        )
    if not np.all(np.isfinite(gradient)):
        raise RuntimeError("gradient contains NaN or infinity")
    objective_value = float(np.asarray(value).reshape(-1)[0])
    if not np.isfinite(objective_value):
        raise RuntimeError("objective contains NaN or infinity")
    local_workload_wall_seconds = time.perf_counter() - workload_started
    workload_wall_seconds = _distributed_max(local_workload_wall_seconds)

    directional_finite_difference = None
    if args.directional_fd_step > 0:
        step = float(args.directional_fd_step)
        plus_value, _ = problem(
            rho_vector=[weights_vector + step * direction], need_gradient=False
        )
        minus_value, _ = problem(
            rho_vector=[weights_vector - step * direction], need_gradient=False
        )
        plus_scalar = float(np.asarray(plus_value).reshape(-1)[0])
        minus_scalar = float(np.asarray(minus_value).reshape(-1)[0])
        derivative = (plus_scalar - minus_scalar) / (2.0 * step)
        if not all(np.isfinite(value) for value in (plus_scalar, minus_scalar, derivative)):
            raise RuntimeError("directional finite-difference result is non-finite")
        directional_finite_difference = {
            "step": step,
            "plus_objective": plus_scalar,
            "minus_objective": minus_scalar,
            "derivative": derivative,
            "adjoint_projection": float(np.dot(gradient, direction)),
        }

    # Resolve CUDA availability before capturing /proc/self/maps so the
    # runtime closure includes the actually loaded driver library even for
    # the CPU-oracle lane of this CUDA-qualified build.
    runtime_available = bool(gpu.runtime_available)
    runtime = _runtime_closure(mp)
    thread_cpu_affinities = _process_thread_cpu_affinities()
    rank_nvidia_driver = (
        _rank_local_nvidia_driver_evidence()
        if qualification.get("mode") == "qualification"
        else None
    )
    total_elapsed = (
        update_seconds + forward_seconds + adjoint_seconds + gradient_seconds
    )
    local_total_elapsed = (
        local_update_seconds
        + local_forward_seconds
        + local_adjoint_seconds
        + local_gradient_seconds
    )
    result = {
        "schema_version": 4,
        "qualification": qualification,
        "producer": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "command": [
                str(pathlib.Path(sys.executable).resolve()),
                str(pathlib.Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            "environment": {
                name: os.environ.get(name) for name in EVIDENCE_ENVIRONMENT_KEYS
            }
            if qualification.get("mode") != "qualification"
            else dict(sorted(os.environ.items())),
            "started_at_utc": started_at_utc,
            "hostname": socket.gethostname(),
            "cpu_affinity": launch_cpu_affinity,
            "completion_thread_cpu_affinity": _process_cpu_affinity(),
            "process_thread_cpu_affinities": thread_cpu_affinities,
            "lazy_import_contract": _LAZY_IMPORT_CONTRACT,
        },
        "backend": {
            "requested": gpu.requested_backend,
            "active": gpu.active_backend,
            "compiled": bool(gpu.compiled),
            "runtime_available": runtime_available,
            "runtime_diagnostic": gpu.runtime_diagnostic,
            "compiled_architectures": gpu.compiled_architectures,
            "selected_device": int(gpu.selected_device),
            "selected_device_identifier": gpu.selected_device_identifier,
            "devices": gpu.devices(),
            "single_precision": bool(mp.is_single_precision()),
            "strict_cuda_marker": os.environ.get(
                "GPMEEP_VALIDATION_STRICT_CUDA"
            )
            == "1",
        },
        "runtime": runtime,
        "workload": {
            "resolution": args.resolution,
            "run_time": args.run_time,
            "cell_size": args.cell_size,
            "design_resolution": args.design_resolution,
            "design_shape": [design_nx, design_ny],
            "design_variables": int(weights.size),
            "yee_cells": int(
                round(args.cell_size * args.resolution)
                * round(args.cell_size * args.resolution)
            ),
            "weights_sha256": canonical_float64_sha256(weights_vector),
            "direction_sha256": canonical_float64_sha256(direction),
            "material_gradient_path": os.environ.get(
                "MEEP_MATERIAL_GRADIENT_PATH", "auto"
            ),
            "phases": {
                "update_design": _phase_record(
                    baseline_statistics, after_update, update_seconds
                ),
                "forward": _phase_record(
                    after_update,
                    after_forward,
                    forward_seconds,
                    timesteps=forward_timesteps,
                    end_time=forward_end_time,
                ),
                "adjoint": _phase_record(
                    after_forward,
                    after_adjoint,
                    adjoint_seconds,
                    timesteps=adjoint_timesteps,
                    end_time=adjoint_end_time,
                ),
                "gradient": _phase_record(
                    after_adjoint, after_gradient, gradient_seconds
                ),
            },
            "final_statistics": after_gradient,
        },
        "result": {
            "objective": objective_value,
            "gradient": gradient.tolist(),
            "gradient_shape": gradient_shape,
            "gradient_count": int(gradient.size),
            "gradient_sha256": canonical_float64_sha256(gradient),
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient_projection": float(np.dot(gradient, direction)),
            "gradient_sum": float(np.sum(gradient)),
            "directional_finite_difference": directional_finite_difference,
        },
        "timing": {
            "workload_wall_seconds": workload_wall_seconds,
            "elapsed_seconds": total_elapsed,
            "update_design_seconds": update_seconds,
            "forward_seconds": forward_seconds,
            "adjoint_seconds": adjoint_seconds,
            "gradient_seconds": gradient_seconds,
            "rank_local_elapsed_seconds": local_total_elapsed,
            "rank_local_workload_wall_seconds": local_workload_wall_seconds,
        },
    }
    if args.capture_field_times:
        result["workload"]["field_timing"] = {
            "forward": forward_field_times,
            "adjoint": adjoint_field_times,
        }

    if mp.with_mpi():
        from mpi4py import MPI

        world = MPI.COMM_WORLD
        local_phases = {
            "update_design": _phase_record(
                baseline_statistics, after_update, local_update_seconds
            ),
            "forward": _phase_record(
                after_update,
                after_forward,
                local_forward_seconds,
                timesteps=forward_timesteps,
                end_time=forward_end_time,
            ),
            "adjoint": _phase_record(
                after_forward,
                after_adjoint,
                local_adjoint_seconds,
                timesteps=adjoint_timesteps,
                end_time=adjoint_end_time,
            ),
            "gradient": _phase_record(
                after_adjoint, after_gradient, local_gradient_seconds
            ),
        }
        local_rank_result = {
            "world_rank": int(world.Get_rank()),
            "backend": dict(result["backend"]),
            "runtime": dict(result["runtime"]),
            "active_backend": gpu.active_backend,
            "selected_device": int(gpu.selected_device),
            "selected_device_identifier": gpu.selected_device_identifier,
            "backend_diagnostic": gpu.backend_diagnostic,
            "nvidia_driver": rank_nvidia_driver,
            "process_environment": dict(sorted(os.environ.items())),
            "hostname": socket.gethostname(),
            "cpu_affinity": launch_cpu_affinity,
            "completion_thread_cpu_affinity": _process_cpu_affinity(),
            "process_thread_cpu_affinities": thread_cpu_affinities,
            "objective": objective_value,
            "gradient": gradient.tolist(),
            "gradient_sha256": canonical_float64_sha256(gradient),
            "phases": local_phases,
            "final_statistics": after_gradient,
            "timing": {
                "workload_wall_seconds": local_workload_wall_seconds,
                "elapsed_seconds": local_total_elapsed,
                "update_design_seconds": local_update_seconds,
                "forward_seconds": local_forward_seconds,
                "adjoint_seconds": local_adjoint_seconds,
                "gradient_seconds": local_gradient_seconds,
            },
        }
        rank_results = world.gather(local_rank_result, root=0)
        if world.Get_rank() != 0:
            return

        rank_results.sort(key=lambda item: item["world_rank"])
        reference_gradient = np.asarray(
            rank_results[0]["gradient"], dtype=np.float64
        )
        objective_reference = float(rank_results[0]["objective"])
        objective_max_abs_difference = max(
            abs(float(item["objective"]) - objective_reference)
            for item in rank_results
        )
        gradient_max_abs_difference = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(item["gradient"], dtype=np.float64)
                        - reference_gradient
                    )
                )
            )
            for item in rank_results
        )
        reference_norm = float(np.linalg.norm(reference_gradient))
        gradient_max_relative_l2_difference = max(
            float(
                np.linalg.norm(
                    np.asarray(item["gradient"], dtype=np.float64)
                    - reference_gradient
                )
                / max(reference_norm, np.finfo(np.float64).tiny)
            )
            for item in rank_results
        )
        device_identifiers = [
            item["selected_device_identifier"] for item in rank_results
        ]
        selected_devices_unique = (
            all(device_identifiers)
            and len(set(device_identifiers)) == len(device_identifiers)
        )
        gradient_gate = all(
            np.allclose(
                np.asarray(item["gradient"], dtype=np.float64),
                reference_gradient,
                atol=args.distributed_gradient_atol,
                rtol=args.distributed_gradient_rtol,
            )
            for item in rank_results
        )
        strict_cuda = result["backend"]["strict_cuda_marker"]
        cuda_coverage_gate = True
        if strict_cuda:
            for item in rank_results:
                statistics = item["final_statistics"]
                cpu_calls = sum(
                    statistics[category][field]
                    for category, field in (
                        ("dispatch", "cpu_curl_calls"),
                        ("field_updates", "cpu_update_eh_calls"),
                        ("polarizations", "cpu_polarization_calls"),
                        ("sources", "cpu_source_calls"),
                        ("boundaries", "cpu_boundary_calls"),
                        ("dfts", "cpu_dft_calls"),
                    )
                )
                cuda_coverage_gate = cuda_coverage_gate and (
                    item["active_backend"] == "cuda"
                    and cpu_calls == 0
                    and statistics["dispatch"]["cuda_curl_calls"] > 0
                    and statistics["field_updates"]["cuda_update_eh_calls"] > 0
                    and statistics["sources"]["cuda_source_calls"] > 0
                    and statistics["boundaries"]["cuda_boundary_calls"] > 0
                    and statistics["dfts"]["cuda_dft_calls"] > 0
                )
                if world.Get_size() > 1:
                    cuda_coverage_gate = cuda_coverage_gate and (
                        statistics["multi_gpu"]["mpi_messages"] > 0
                        and statistics["multi_gpu"]["mpi_scalars"] > 0
                        and statistics["multi_gpu"]["cuda_aware_bytes"]
                        + statistics["multi_gpu"]["pinned_staging_bytes"]
                        > 0
                    )
            cuda_coverage_gate = cuda_coverage_gate and selected_devices_unique
        objective_gate = (
            objective_max_abs_difference <= args.distributed_objective_atol
        )
        distributed_gate = objective_gate and gradient_gate and cuda_coverage_gate
        result["distributed"] = {
            "world_size": int(world.Get_size()),
            "timing_semantics": "barrier-delimited maximum rank elapsed per phase",
            "selected_devices_unique": selected_devices_unique,
            "objective_max_abs_difference": objective_max_abs_difference,
            "gradient_max_abs_difference": gradient_max_abs_difference,
            "gradient_max_relative_l2_difference": gradient_max_relative_l2_difference,
            "tolerances": {
                "objective_atol": args.distributed_objective_atol,
                "gradient_atol": args.distributed_gradient_atol,
                "gradient_rtol": args.distributed_gradient_rtol,
            },
            "gate": {
                "pass": distributed_gate,
                "objective": objective_gate,
                "gradient": gradient_gate,
                "strict_cuda_coverage": cuda_coverage_gate,
            },
            "ranks": rank_results,
        }
        if not distributed_gate:
            raise RuntimeError(
                "distributed adjoint rank-consistency or strict CUDA gate failed"
            )

    result["producer"]["completed_at_utc"] = _utc_now()
    atomic_write_json(args.result_file, result)
    digest = sha256_file(args.result_file)
    print(
        PREFIX
        + json.dumps(
            {
                "result_file": str(args.result_file.resolve()),
                "sha256": digest,
                "active_backend": result["backend"]["active"],
                "elapsed_seconds": total_elapsed,
            },
            sort_keys=True,
        )
    )


def _entrypoint() -> None:
    try:
        main()
    except BaseException:
        # A rank-local Python/C++ error before a later collective must not
        # strand peer ranks until the parent timeout.  Open MPI's Abort is the
        # release-path failure broadcast; the parent runner additionally owns
        # and terminates the complete process group on timeout.
        try:
            if mp.with_mpi():
                from mpi4py import MPI

                if MPI.COMM_WORLD.Get_size() > 1:
                    MPI.COMM_WORLD.Abort(1)
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    _entrypoint()
