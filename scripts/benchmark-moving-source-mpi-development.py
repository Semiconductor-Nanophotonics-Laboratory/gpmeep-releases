#!/usr/bin/env python3
"""Development-only fair MPI-CPU/CUDA moving-source benchmark.

The record is deliberately non-authoritative, but it is still fail-closed: it
binds the complete Git working source, loaded native artifacts, physical CPU
affinity, per-rank backend decisions, and native dispatch statistics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import tempfile
import time
from fractions import Fraction

MPI = None
np = None
mp = None
COMM = None
WORKLOAD = {
    "sx": 40.0,
    "sy": 40.0,
    "dpml": 1.0,
    "source_speed": 0.7,
    "source_frequency": 1e-10,
}
METRIC_SCALARS = ("sum", "l1", "l2", "maximum_absolute")


def expected_workload_contract(resolution):
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
        raise ValueError("moving-source resolution must be a positive integer")
    sx = Fraction(str(WORKLOAD["sx"]))
    sy = Fraction(str(WORKLOAD["sy"]))
    speed = Fraction(str(WORKLOAD["source_speed"]))
    courant = Fraction(1, 2)
    requested = sx / speed
    timestep = courant / resolution
    ratio = requested / timestep
    steps = (ratio.numerator + ratio.denominator - 1) // ratio.denominator
    return {
        **WORKLOAD,
        "courant": float(courant),
        "requested_until": float(requested),
        "expected_timesteps": steps,
        "expected_meep_time": float(steps * timestep),
        "expected_source_updates": steps + 1,
        "expected_field_shape": [int(sx * resolution), int(sy * resolution)],
    }


def validate_metric(metric, contract):
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
        for name in METRIC_SCALARS
    ):
        raise RuntimeError("moving-source run produced a non-finite metric")
    if any(metric[name] < 0 for name in ("l1", "l2", "maximum_absolute")):
        raise RuntimeError("moving-source norm metric must be nonnegative")


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--omp-threads-per-rank", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args():
    return argument_parser().parse_args()


def parse_output_path():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_known_args()[0].output


def load_benchmark_dependencies():
    global MPI, np, mp, COMM
    from mpi4py import MPI as loaded_mpi
    import numpy as loaded_numpy
    import meep as loaded_meep

    MPI = loaded_mpi
    np = loaded_numpy
    mp = loaded_meep
    COMM = MPI.COMM_WORLD


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def git_output(repo, *arguments):
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def repository_root():
    return Path(
        git_output(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip()
    ).resolve(strict=True)


def invalidate_output(path, communicator=None):
    """Remove an older success before any validation or benchmark work."""
    rank = communicator.rank if communicator is not None else 0
    error = None
    if rank == 0:
        try:
            Path(path).resolve().unlink(missing_ok=True)
        except OSError as exc:
            error = f"cannot invalidate previous output: {exc}"
    if communicator is not None:
        error = communicator.bcast(error, root=0)
        communicator.Barrier()
    if error is not None:
        raise RuntimeError(error)


def source_snapshot(repo):
    """Hash every tracked or non-ignored untracked working-tree entry."""
    root = Path(repo).resolve(strict=True)
    names = git_output(
        root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    ).split(b"\0")
    entries = []
    for encoded in sorted(name for name in names if name):
        relative = os.fsdecode(encoded)
        path = root / relative
        try:
            status = path.lstat()
        except FileNotFoundError:
            entries.append({"path": relative, "kind": "missing"})
            continue
        if path.is_symlink():
            target = os.readlink(path)
            entries.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "mode": status.st_mode & 0o7777,
                    "target": target,
                    "sha256": sha256_bytes(os.fsencode(target)),
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": status.st_mode & 0o7777,
                    "size_bytes": status.st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            raise RuntimeError(f"unsupported source-tree entry: {relative}")
    payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "head": git_output(root, "rev-parse", "HEAD").decode().strip(),
        "index_sha256": sha256_bytes(
            git_output(root, "diff", "--cached", "--binary", "--no-ext-diff")
        ),
        "entry_count": len(entries),
        "entries_sha256": sha256_bytes(payload),
        "entries": entries,
    }


def require_unchanged_source(start, end):
    if start != end:
        raise RuntimeError("repository source changed during benchmark execution")


def cpu_identity():
    first = Path("/proc/cpuinfo").read_text(
        encoding="utf-8", errors="replace"
    ).split("\n\n", 1)[0]
    values = {}
    for line in first.splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            values[name.strip()] = value.strip()
    return {
        name: values.get(name)
        for name in (
            "vendor_id",
            "cpu family",
            "model",
            "model name",
            "stepping",
            "microcode",
        )
    }


def physical_core_affinity(logical_cpus=None, sys_cpu_root=Path("/sys/devices/system/cpu")):
    logical = sorted(os.sched_getaffinity(0) if logical_cpus is None else logical_cpus)
    host = socket.gethostname()
    cores = set()
    for cpu in logical:
        topology = Path(sys_cpu_root) / f"cpu{cpu}" / "topology"
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
        cores.add((host, package, core))
    return [
        {"host": item[0], "package_id": item[1], "core_id": item[2]}
        for item in sorted(cores)
    ]


def mapped_libmeep_record(maps_path=Path("/proc/self/maps")):
    candidates = set()
    for line in Path(maps_path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "libmeep.so" in Path(fields[5]).name:
            mapped = Path(fields[5])
            if mapped.exists():
                candidates.add(mapped.resolve(strict=True))
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in sorted(candidates)) or "<none>"
        raise RuntimeError(f"expected exactly one mapped libmeep, found {rendered}")
    return file_record(next(iter(candidates)))


def runtime_identity():
    extension = mp._meep.__file__
    return {
        "python": file_record(os.sys.executable),
        "extension": file_record(extension),
        "mapped_libmeep": mapped_libmeep_record(),
        "platform": list(platform.uname()),
        "cpu_identity": cpu_identity(),
    }


def rank_execution_identity():
    return {
        "rank": COMM.rank,
        "physical_core_affinity": physical_core_affinity(),
        "selected_device": int(mp.gpu.selected_device),
        "selected_device_identifier": mp.gpu.selected_device_identifier,
    }


def atomic_json(path, value):
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
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


def run_once(resolution):
    workload = expected_workload_contract(resolution)
    sx = WORKLOAD["sx"]
    sy = WORKLOAD["sy"]
    dpml = WORKLOAD["dpml"]
    speed = WORKLOAD["source_speed"]
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
                    mp.ContinuousSource(frequency=WORKLOAD["source_frequency"]),
                    component=mp.Ex,
                    center=mp.Vector3(current_x),
                )
            ]
        )
        source_updates += 1

    COMM.Barrier()
    started = time.perf_counter()
    simulation.run(move_source, until=sx / speed)
    local_seconds = time.perf_counter() - started
    COMM.Barrier()
    global_seconds = COMM.allreduce(local_seconds, op=MPI.MAX)

    field = np.asarray(
        simulation.get_array(center=mp.Vector3(), size=cell_size, component=mp.Hz)
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
    owner = {
        "selected_backend": "cuda" if owner_cuda else "cpu",
        "diagnostic": simulation.fields.gpu_execution_diagnostic(),
        "process_backend": mp.gpu.active_backend,
    }
    local_record = {
        "rank": COMM.rank,
        "local_seconds": local_seconds,
        "physical_core_affinity": physical_core_affinity(),
        "metric": metric,
        "owner": owner,
        "statistics": mp.gpu.statistics(),
    }
    rank_records = COMM.gather(local_record, root=0)
    simulation.reset_meep()
    del simulation
    gc.collect()
    return global_seconds, metric, owner, rank_records


def main(arguments=None):
    if arguments is None:
        output_path = parse_output_path()
        invalidate_output(output_path)
        args = parse_args()
        if args.output.resolve() != output_path.resolve():
            raise RuntimeError("parsed output path changed during argument validation")
    else:
        args = arguments
        invalidate_output(args.output)
    try:
        load_benchmark_dependencies()
    except (ImportError, OSError) as exc:
        raise RuntimeError("benchmark requires mpi4py, numpy, and meep") from exc
    COMM.Barrier()

    repo = repository_root()
    output_path = args.output.resolve()
    try:
        output_path.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ValueError("development benchmark output must be outside the source tree")

    source_start = source_snapshot(repo) if COMM.rank == 0 else None
    source_start = COMM.bcast(source_start, root=0)
    COMM.Barrier()

    if args.resolution <= 0 or args.repeat <= 0 or args.omp_threads_per_rank <= 0:
        raise ValueError("resolution, repeat, and OMP threads per rank must be positive")
    if COMM.size != args.expected_ranks or mp.count_processors() != COMM.size:
        raise RuntimeError("MPI rank count does not match the benchmark contract")
    if os.environ.get("OMP_NUM_THREADS") != str(args.omp_threads_per_rank):
        raise RuntimeError("OMP_NUM_THREADS does not match the benchmark contract")
    if args.backend == "cpu" and args.omp_threads_per_rank != 1:
        raise RuntimeError("CPU FDTD baseline requires one real stencil worker per MPI rank")
    if not mp.is_single_precision() or not mp.with_mpi():
        raise RuntimeError("benchmark requires the FP32 MPI gpmeep build")
    runtime_start = runtime_identity()
    mp.gpu.reset_statistics()
    mp.gpu.set_backend(args.backend)

    durations = []
    metrics = []
    owners = []
    samples_by_rank = []
    for _ in range(args.repeat):
        duration, metric, owner, rank_records = run_once(args.resolution)
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("moving-source solver duration must be finite and positive")
        durations.append(duration)
        metrics.append(metric)
        owners.append(owner)
        if COMM.rank == 0:
            samples_by_rank.append(rank_records)

    statistics_by_rank = COMM.gather(mp.gpu.statistics(), root=0)
    execution_by_rank = COMM.gather(rank_execution_identity(), root=0)
    runtime_end = runtime_identity()
    runtime_stable = COMM.allreduce(runtime_start == runtime_end, op=MPI.LAND)
    if not runtime_stable:
        raise RuntimeError("Python/native runtime artifacts changed during benchmark execution")
    runtime_by_rank = COMM.gather(runtime_start, root=0)
    COMM.Barrier()
    source_end = source_snapshot(repo) if COMM.rank == 0 else None
    source_end = COMM.bcast(source_end, root=0)
    require_unchanged_source(source_start, source_end)
    if COMM.rank != 0:
        return

    if any(identity != runtime_by_rank[0] for identity in runtime_by_rank):
        raise RuntimeError("MPI ranks used different Python/native runtime artifacts")
    reference_metric = metrics[0]
    for repeat_index, (metric, owner, rank_records) in enumerate(
        zip(metrics, owners, samples_by_rank)
    ):
        if metric != reference_metric:
            raise RuntimeError("repeat changed deterministic moving-source metrics")
        if (
            owner["selected_backend"] != args.backend
            or owner["process_backend"] != args.backend
        ):
            raise RuntimeError("benchmark selected the wrong root backend")
        if len(rank_records) != COMM.size:
            raise RuntimeError("benchmark gathered an incomplete rank set")
        for rank_record in rank_records:
            if rank_record["metric"] != metric:
                raise RuntimeError(
                    f"rank {rank_record['rank']} observed different global work in repeat {repeat_index}"
                )
            rank_owner = rank_record["owner"]
            if (
                rank_owner["selected_backend"] != args.backend
                or rank_owner["process_backend"] != args.backend
            ):
                raise RuntimeError("one or more ranks selected the wrong backend")

    output = {
        "schema_version": 2,
        "development_only": True,
        "authoritative_release_evidence": False,
        "performance_claim_eligible": False,
        "reason": (
            "uncommitted source checkpoint; rerun from a sealed receipt for release claims"
        ),
        "backend": args.backend,
        "resolution": args.resolution,
        "repeat": args.repeat,
        "mpi_ranks": COMM.size,
        "omp_threads_per_rank": args.omp_threads_per_rank,
        "total_host_thread_budget": COMM.size * args.omp_threads_per_rank,
        "solver_seconds": durations,
        "solver_seconds_median": statistics.median(durations),
        "metrics": metrics,
        "global_work_totals": [
            {
                "field_sample_count": math.prod(metric["shape"]),
                "source_updates": metric["source_updates"],
                "meep_time": metric["meep_time"],
            }
            for metric in metrics
        ],
        "owners": owners,
        "rank_samples": samples_by_rank,
        "statistics_by_rank": statistics_by_rank,
        "execution_by_rank": execution_by_rank,
        "source_state": {"start": source_start, "end": source_end},
        "workload_contract": expected_workload_contract(args.resolution),
        "runtime": runtime_by_rank[0],
        "runtime_by_rank": runtime_by_rank,
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "gpu_mpi_transport": os.environ.get("MEEP_GPU_MPI_TRANSPORT", "auto"),
            "allow_oversubscribe": os.environ.get("MEEP_GPU_ALLOW_OVERSUBSCRIBE", "0"),
        },
    }
    atomic_json(args.output, output)
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
