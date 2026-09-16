#!/usr/bin/env python3
"""Run the byte-exact user TERS script with externally isolated rank I/O.

Every MPI rank executes the original byte stream in its own working directory.
The adapter observes ``Simulation.run`` but does not reset simulations, replace
I/O, freeze time, or suppress plotting/CSV work.  It reports FDTD-only timing
separately from the exact original-script execution interval.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import pathlib
import runpy
import shutil
import sys
import time
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME,
    TERS_SHA256,
    TERS_PERFORMANCE_ADAPTATION,
    TERS_PERFORMANCE_MEASUREMENT_MODE,
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    file_record,
    json_safe,
    phase_call_totals,
    phase_counter_view,
    physical_core_affinity,
    statistics_delta,
    validate_dft_coordinate_metadata,
    validate_phase_contract,
    verify_file,
    verify_build_receipt_collective,
)


SCHEMA = "gpmeep-ters-workload-v2"
PHASE_CONTRACT = [
    {
        "stage": stage,
        "required_per_rank": ["curl", "update_eh"],
        "required_aggregate": ["polarization", "source", "boundary", "dft"],
        "forbidden": [],
    }
    for stage in ("with-tip", "without-tip")
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--build-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--expected-backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument(
        "--performance-window-meep-time",
        type=float,
        default=None,
        help=(
            "replace each original decay stop with this fixed Meep-time window; "
            "this is an explicitly adapted performance sample, never exact evidence"
        ),
    )
    return parser.parse_args(argv)


def _validate_backend_record(record: dict[str, Any], expected_backend: str) -> None:
    if record["active_backend"] != expected_backend:
        raise WorkloadError(
            f"TERS selected {record['active_backend']!r}, expected {expected_backend!r}"
        )
    if record["timestep_delta"] <= 0 or record["wall_seconds"] <= 0:
        raise WorkloadError("TERS did not perform positive timed FDTD work")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.performance_window_meep_time is not None and (
        not math.isfinite(args.performance_window_meep_time)
        or not MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
        <= args.performance_window_meep_time
        <= 1.0
    ):
        raise WorkloadError(
            "TERS performance window must be in "
            f"[{MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME},1] Meep time"
        )
    source = verify_file(args.source, TERS_SHA256, "TERS source")
    output = args.output.resolve()

    import meep as mp
    import numpy as np
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    repo = SCRIPT_DIR.parents[1]
    build_receipt, runtime_artifacts = verify_build_receipt_collective(
        comm, rank, args.build_receipt, repo, mp
    )

    preparation_error = None
    if rank == 0:
        try:
            if output.exists() and any(output.iterdir()):
                raise WorkloadError(f"TERS output directory is not empty: {output}")
            work = output / "work"
            work.mkdir(parents=True, exist_ok=True)
            for worker_rank in range(size):
                rank_work = work / f"rank-{worker_rank:05d}"
                rank_work.mkdir()
                shutil.copyfile(source, rank_work / source.name)
        except Exception as exc:  # broadcast a deterministic collective failure
            preparation_error = f"{type(exc).__name__}: {exc}"
    preparation_error = comm.bcast(preparation_error, root=0)
    if preparation_error is not None:
        raise WorkloadError(preparation_error)
    comm.Barrier()

    work = output / "work" / f"rank-{rank:05d}"
    copied_source = verify_file(work / source.name, TERS_SHA256, "copied TERS source")
    records: list[dict[str, Any]] = []
    dft_monitors: dict[int, list[Any]] = {}
    dft_coordinates: list[dict[str, np.ndarray]] = []

    original_run = mp.Simulation.run
    original_add_dft_fields = mp.Simulation.add_dft_fields

    def audited_add_dft_fields(
        simulation: Any, *monitor_args: Any, **monitor_kwargs: Any
    ) -> Any:
        monitor = original_add_dft_fields(
            simulation, *monitor_args, **monitor_kwargs
        )
        dft_monitors.setdefault(id(simulation), []).append(monitor)
        return monitor

    def coordinate_evidence(values: dict[str, np.ndarray]) -> dict[str, Any]:
        axes: dict[str, Any] = {}
        for name, value in values.items():
            array = np.ascontiguousarray(value)
            digest = hashlib.sha256()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(memoryview(array).cast("B"))
            axes[name] = {
                "count": int(array.size),
                "dtype": str(array.dtype),
                "sha256": digest.hexdigest(),
            }
        return axes

    def audited_run(simulation: Any, *step_functions: Any, **kwargs: Any) -> Any:
        before_timestep = int(simulation.fields.t) if simulation.fields is not None else 0
        before_stats_full = json_safe(mp.gpu.statistics())
        comm.Barrier()
        started = time.perf_counter()
        run_kwargs = kwargs
        if args.performance_window_meep_time is not None:
            if step_functions:
                raise WorkloadError(
                    "TERS performance adaptation does not accept step functions"
                )
            run_kwargs = {"until": args.performance_window_meep_time}
        result = original_run(simulation, *step_functions, **run_kwargs)
        comm.Barrier()
        wall_seconds = time.perf_counter() - started
        record = None
        validation_error = None
        try:
            after_stats_full = json_safe(mp.gpu.statistics())
            delta = statistics_delta(before_stats_full, after_stats_full)
            monitors = dft_monitors.get(id(simulation), [])
            if len(monitors) != 1:
                raise WorkloadError(
                    f"TERS run has {len(monitors)} DFT field monitors, expected one"
                )
            metadata = simulation.get_array_metadata(dft_cell=monitors[0])
            if len(metadata) != 4:
                raise WorkloadError("TERS DFT coordinate metadata is incomplete")
            coordinate_values = {
                name: np.asarray(value).copy()
                for name, value in zip(("x", "y", "z", "w"), metadata)
            }
            validate_dft_coordinate_metadata(coordinate_values, np.isfinite)
            record = {
                "run_index": len(records),
                "timestep": int(simulation.timestep()),
                "timestep_delta": int(simulation.timestep()) - before_timestep,
                "meep_time": float(simulation.meep_time()),
                "wall_seconds": wall_seconds,
                "active_backend": str(mp.gpu.active_backend),
                "requested_backend": str(
                    getattr(mp.gpu, "requested_backend", mp.gpu.active_backend)
                ),
                "device_ordinal": int(mp.gpu.selected_device),
                "device_identifier": str(mp.gpu.selected_device_identifier)
                if args.expected_backend == "cuda"
                else None,
                "phase_calls": phase_call_totals(delta),
                "phase_counters": phase_counter_view(delta),
                "statistics_before": before_stats_full,
                "statistics_after": after_stats_full,
                "statistics_delta": delta,
                "dft_coordinates": coordinate_evidence(coordinate_values),
            }
            _validate_backend_record(record, args.expected_backend)
        except Exception as exc:
            validation_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        validation_errors = comm.allgather(validation_error)
        if any(error is not None for error in validation_errors):
            raise WorkloadError(
                "collective TERS run audit failed: "
                + "; ".join(error for error in validation_errors if error is not None)
            )
        assert record is not None
        records.append(record)
        dft_coordinates.append(coordinate_values)
        return result

    previous_cwd = pathlib.Path.cwd()
    previous_argv = sys.argv
    namespace: dict[str, Any] | None = None
    mp.Simulation.add_dft_fields = audited_add_dft_fields
    mp.Simulation.run = audited_run
    mp.gpu.reset_statistics()
    comm.Barrier()
    exact_started = time.perf_counter()
    try:
        os.chdir(work)
        sys.argv = [str(copied_source)]
        namespace = runpy.run_path(str(copied_source), run_name="__main__")
    finally:
        mp.Simulation.run = original_run
        mp.Simulation.add_dft_fields = original_add_dft_fields
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    comm.Barrier()
    exact_end_to_end_seconds = time.perf_counter() - exact_started

    if namespace is None or len(records) != 2 or len(dft_coordinates) != 2:
        raise WorkloadError(f"TERS expected two Simulation.run calls, got {len(records)}")
    result_candidates = sorted((work / "results").glob(f"{source.name}_*"))
    if len(result_candidates) != 1 or not result_candidates[0].is_dir():
        raise WorkloadError(
            f"rank {rank} TERS original result directory is not unique: "
            f"{[path.name for path in result_candidates]}"
        )
    result_root = result_candidates[0]
    rank_original_outputs = [
        file_record(path, output)
        for path in sorted(result_root.iterdir())
        if path.is_file()
    ]
    rank_original_names = {
        record["path"].rsplit("/", 1)[-1] for record in rank_original_outputs
    }
    required_rank_original_names = {
        f"{source.stem}_backup.py",
        "withtip_geo.png",
        "withtip_dft.csv",
        "withtip_dft.png",
        "withtip_dft.npy",
        "wotip_geo.png",
        "wotip_dft.csv",
        "wotip_dft.png",
        "wotip_dft.npy",
    }
    if rank_original_names != required_rank_original_names:
        raise WorkloadError(
            f"rank {rank} exact original output inventory is incomplete: "
            f"{sorted(rank_original_names)}"
        )
    backup_record = next(
        record
        for record in rank_original_outputs
        if record["path"].endswith(f"/{source.stem}_backup.py")
    )
    if backup_record["sha256"] != TERS_SHA256:
        raise WorkloadError(f"rank {rank} original backup is not byte-exact")
    local = {
        "rank": rank,
        "hostname": MPI.Get_processor_name(),
        "records": records,
        "exact_end_to_end_seconds": exact_end_to_end_seconds,
        "original_result_root": result_root.relative_to(output).as_posix(),
        "original_outputs": rank_original_outputs,
        "runtime_artifacts": runtime_artifacts,
        "physical_cores": physical_core_affinity(),
        "build_receipt_id": build_receipt["receipt_id"],
        "final_statistics": json_safe(mp.gpu.statistics()),
    }
    gathered = comm.gather(local, root=0)
    comm.Barrier()

    publication_error = None
    if rank == 0:
        try:
            validate_phase_contract(gathered, args.expected_backend, PHASE_CONTRACT)
            results = output / gathered[0]["original_result_root"]
            expected_arrays = [results / "withtip_dft.npy", results / "wotip_dft.npy"]
            if any(not path.is_file() for path in expected_arrays):
                raise WorkloadError("TERS did not retain both complex DFT arrays")
            arrays = []
            for path in expected_arrays:
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                if value.size == 0 or not np.issubdtype(value.dtype, np.complexfloating):
                    raise WorkloadError(f"invalid TERS complex DFT array: {path}")
                arrays.append(
                    {
                        **file_record(path, output),
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                )
            x = np.asarray(namespace.get("x"))
            y = np.asarray(namespace.get("y"))
            if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
                raise WorkloadError("TERS did not expose valid DFT coordinates")
            if not np.array_equal(x, dft_coordinates[1]["x"]) or not np.array_equal(
                y, dft_coordinates[1]["y"]
            ):
                raise WorkloadError(
                    "TERS final namespace coordinates differ from its second monitor"
                )
            coordinate_records = []
            evidence_directory = output / "evidence"
            evidence_directory.mkdir()
            for run_index, coordinate_values in enumerate(dft_coordinates):
                coordinate_path = evidence_directory / f"dft_coordinates_run{run_index}.npz"
                np.savez(coordinate_path, **coordinate_values)
                coordinate_records.append(
                    {
                        **file_record(coordinate_path, output),
                        "run_index": run_index,
                        "axes": coordinate_evidence(coordinate_values),
                    }
                )
            original_outputs = [
                file_record(path, output)
                for path in sorted(results.iterdir())
                if path.is_file()
            ]
            original_names = {record["path"].rsplit("/", 1)[-1] for record in original_outputs}
            required_original_names = {
                f"{source.stem}_backup.py",
                "withtip_geo.png",
                "withtip_dft.csv",
                "withtip_dft.png",
                "withtip_dft.npy",
                "wotip_geo.png",
                "wotip_dft.csv",
                "wotip_dft.png",
                "wotip_dft.npy",
            }
            if original_names != required_original_names:
                raise WorkloadError(
                    "TERS exact original output inventory is incomplete or unexpected: "
                    f"{sorted(original_names)}"
                )
            summary = {
                "schema": SCHEMA,
                "source": absolute_file_record(source, "TERS source"),
                "source_sha256": TERS_SHA256,
                "expected_backend": args.expected_backend,
                "single_precision": bool(mp.is_single_precision()),
                "meep_version": str(getattr(mp, "__version__", "unknown")),
                "meep_module": str(pathlib.Path(mp.__file__).resolve()),
                "build_receipt": build_receipt,
                "adapter": file_record(pathlib.Path(__file__), repo),
                "adapter_common": file_record(SCRIPT_DIR / "common.py", repo),
                "mpi_size": size,
                "rank_records": gathered,
                "arrays": arrays,
                "coordinates": coordinate_records,
                "phase_contract": PHASE_CONTRACT,
                "original_outputs": original_outputs,
                "presentation_suppressed": False,
                "measurement_mode": (
                    "exact"
                    if args.performance_window_meep_time is None
                    else TERS_PERFORMANCE_MEASUREMENT_MODE
                ),
                "performance_window_meep_time": args.performance_window_meep_time,
                "physics_overrides": (
                    []
                    if args.performance_window_meep_time is None
                    else [TERS_PERFORMANCE_ADAPTATION]
                ),
                "execution_policy_overrides": [
                    "one byte-exact source copy and private working directory per MPI rank",
                    "observer-only Simulation.run instrumentation",
                    *(
                        []
                        if args.performance_window_meep_time is None
                        else [
                            "fixed-window timing adaptation; outputs are not "
                            "exact-workload evidence"
                        ]
                    ),
                ],
                "timing_semantics": {
                    "fdtd_only": "sum of rank-synchronized Simulation.run wall intervals",
                    "exact_end_to_end": (
                        "maximum rank interval around source execution including plots, "
                        "CSV/NPY, and backup; natural Simulation lifetime only in exact mode"
                    ),
                    "adapter_process": (
                        "controller-observed wrapper/provenance/publication interval"
                    ),
                },
            }
            atomic_write_json(output / "summary.json", summary)
            atomic_write_json(
                output / "COMPLETE",
                {
                    "schema": "gpmeep-ters-workload-complete-v2",
                    "summary": file_record(output / "summary.json", output),
                },
            )
        except Exception as exc:
            publication_error = f"{type(exc).__name__}: {exc}"
    publication_error = comm.bcast(publication_error, root=0)
    if publication_error is not None:
        raise WorkloadError(f"TERS evidence publication failed: {publication_error}")
    comm.Barrier()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkloadError as error:
        print(f"TERS workload error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
