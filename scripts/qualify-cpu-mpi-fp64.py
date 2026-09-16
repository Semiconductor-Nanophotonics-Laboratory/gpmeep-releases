#!/usr/bin/env python3
"""Qualify an installed CPU-only FP64 MPI+Python gpmeep build."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import stat
import sys
from typing import Any


SCHEMA = "gpmeep-cpu-mpi-fp64-qualification-v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument(
        "--runtime-kind", required=True, choices=("in-place", "installed")
    )
    parser.add_argument("--expected-extension", required=True, type=pathlib.Path)
    parser.add_argument("--expected-libmeep", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.expected_size < 1:
        parser.error("--expected-size must be positive")
    return args


def loaded_libmeep_path(maps_payload: str) -> pathlib.Path:
    candidates: set[pathlib.Path] = set()
    for line in maps_payload.splitlines():
        columns = line.split(None, 5)
        if len(columns) != 6:
            continue
        raw_path = columns[5]
        if raw_path.endswith(" (deleted)"):
            raise RuntimeError("the loaded libmeep mapping has been deleted")
        path = pathlib.Path(raw_path)
        if path.is_absolute() and path.name.startswith("libmeep.so"):
            candidates.add(path.resolve(strict=True))
    if len(candidates) != 1:
        raise RuntimeError(
            "expected one loaded libmeep image, observed "
            f"{len(candidates)}: {sorted(map(str, candidates))}"
        )
    return next(iter(candidates))


def file_identity(path: pathlib.Path) -> dict[str, Any]:
    lexical = pathlib.Path(path)
    before = lexical.lstat()
    resolved = lexical.resolve(strict=True)
    if not stat.S_ISREG(before.st_mode) or resolved != lexical:
        raise RuntimeError(f"runtime artifact is not a canonical regular file: {lexical}")
    fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    digest = hashlib.sha256()
    with lexical.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    after = lexical.lstat()
    if fingerprint != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError(f"runtime artifact changed while hashing: {lexical}")
    return {
        "path": str(lexical),
        "size_bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def validate_runtime_artifacts(
    mp: Any,
    expected_extension: pathlib.Path,
    expected_libmeep: pathlib.Path,
) -> dict[str, Any]:
    extension = pathlib.Path(mp._meep.__file__).resolve(strict=True)
    libmeep = loaded_libmeep_path(
        pathlib.Path("/proc/self/maps").read_text(encoding="utf-8")
    )
    expected_extension = expected_extension.resolve(strict=True)
    expected_libmeep = expected_libmeep.resolve(strict=True)
    if extension != expected_extension:
        raise RuntimeError(
            f"loaded Python extension mismatch: {extension} vs {expected_extension}"
        )
    if libmeep != expected_libmeep:
        raise RuntimeError(f"loaded libmeep mismatch: {libmeep} vs {expected_libmeep}")
    return {
        "python": file_identity(pathlib.Path(sys.executable).resolve(strict=True)),
        "python_extension": file_identity(extension),
        "libmeep": file_identity(libmeep),
    }


def validate_runtime_identity(mp: Any, actual_size: int, expected_size: int) -> None:
    if actual_size != expected_size:
        raise RuntimeError(
            "mpi4py communicator size mismatch: "
            f"expected {expected_size}, observed {actual_size}"
        )
    if not bool(mp.with_mpi()):
        raise RuntimeError("Meep was not compiled with MPI")
    meep_size = mp.count_processors()
    if (
        not isinstance(meep_size, int)
        or isinstance(meep_size, bool)
        or meep_size != expected_size
        or meep_size != actual_size
    ):
        raise RuntimeError(
            "Meep communicator size mismatch: "
            f"expected {expected_size}, observed {meep_size!r}"
        )
    if bool(mp.is_single_precision()):
        raise RuntimeError("CPU reference build is not FP64")
    if bool(mp.gpu.compiled):
        raise RuntimeError("CPU reference build unexpectedly contains CUDA")
    if mp.gpu.requested_backend != "cpu" or mp.gpu.active_backend != "cpu":
        raise RuntimeError("CPU reference build did not select the CPU backend")


def validate_cpu_statistics(
    statistics: dict[str, Any], *, require_boundary: bool
) -> dict[str, int]:
    required = {
        "curl": ("dispatch", "cpu_curl_calls", "cpu_curl_points"),
        "update_eh": (
            "field_updates",
            "cpu_update_eh_calls",
            "cpu_update_eh_points",
        ),
        "source": ("sources", "cpu_source_calls", "cpu_source_points"),
        "dft": ("dfts", "cpu_dft_calls", "cpu_dft_points"),
    }
    selected: dict[str, int] = {}
    for label, (group, calls_key, points_key) in required.items():
        for suffix, key in (("calls", calls_key), ("points", points_key)):
            value = statistics.get(group, {}).get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RuntimeError(
                    f"CPU qualification has no positive {group}.{key}"
                )
            selected[f"{label}_{suffix}"] = value

    boundary_calls = statistics.get("boundaries", {}).get("cpu_boundary_calls")
    boundary_points = statistics.get("boundaries", {}).get("cpu_boundary_points")
    if (
        not isinstance(boundary_calls, int)
        or isinstance(boundary_calls, bool)
        or boundary_calls < 0
        or (require_boundary and boundary_calls <= 0)
        or not isinstance(boundary_points, int)
        or isinstance(boundary_points, bool)
        or boundary_points < 0
        or (require_boundary and boundary_points <= 0)
    ):
        raise RuntimeError("CPU qualification has invalid boundary dispatch")
    selected["boundary_calls"] = boundary_calls
    selected["boundary_points"] = boundary_points

    for group, values in statistics.items():
        if not isinstance(values, dict):
            raise RuntimeError(f"statistics group {group!r} is not an object")
        for key, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError(f"statistics value {group}.{key} is invalid")
            if key.startswith("cuda_") and value != 0:
                raise RuntimeError(
                    f"CPU-only qualification recorded CUDA work in {group}.{key}"
                )
    return selected


def collective_call(comm: Any, label: str, function: Any) -> Any:
    result = None
    error = None
    try:
        result = function()
    except Exception as exc:  # noqa: BLE001 - converted to collective failure
        error = f"{type(exc).__name__}: {exc}"
    errors = comm.allgather(error)
    failures = [f"rank {rank}: {value}" for rank, value in enumerate(errors) if value]
    if failures:
        raise RuntimeError(f"collective {label} failed: " + "; ".join(failures))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import meep as mp
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    runtime_artifacts = collective_call(
        comm,
        "runtime identity",
        lambda: (
            mp.gpu.set_backend("cpu"),
            validate_runtime_identity(mp, comm.Get_size(), args.expected_size),
            validate_runtime_artifacts(
                mp, args.expected_extension, args.expected_libmeep
            ),
        )[-1],
    )
    mp.gpu.reset_statistics()

    frequency = 0.2
    simulation = mp.Simulation(
        cell_size=mp.Vector3(8),
        boundary_layers=[mp.PML(1.0)],
        resolution=12,
        sources=[
            mp.Source(
                mp.GaussianSource(frequency, fwidth=0.15),
                component=mp.Ez,
                center=mp.Vector3(),
                size=mp.Vector3(6.0),
            )
        ],
    )
    monitor = simulation.add_dft_fields(
        [mp.Ez], frequency, 0.0, 1, center=mp.Vector3(), size=mp.Vector3(2)
    )
    simulation.run(until=8.0)

    energy = float(
        simulation.field_energy_in_box(box=simulation.fields.total_volume())
    )
    dft_norm = float(simulation.fields.dft_norm())
    if not math.isfinite(energy) or energy <= 0:
        raise RuntimeError("CPU FP64 qualification produced invalid field energy")
    if not math.isfinite(dft_norm) or dft_norm <= 0:
        raise RuntimeError("CPU FP64 qualification produced invalid DFT norm")
    if monitor is None:
        raise RuntimeError("CPU FP64 qualification did not create its DFT monitor")

    meep_time = float(simulation.meep_time())
    if not math.isfinite(meep_time) or not math.isclose(
        meep_time, 8.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"CPU FP64 qualification did not finish at Meep time 8.0: {meep_time}"
        )
    gathered = comm.allgather((energy, dft_norm, meep_time))
    reference = gathered[0]
    for rank, value in enumerate(gathered):
        for index, label in enumerate(("energy", "dft_norm")):
            if not math.isclose(value[index], reference[index], rel_tol=1e-12, abs_tol=1e-14):
                raise RuntimeError(
                    f"rank {rank} disagrees on global {label}: {value[index]} vs "
                    f"{reference[index]}"
                )
        if not math.isclose(
            value[2], reference[2], rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"rank {rank} disagrees on final Meep time")

    statistics = mp.gpu.statistics()
    selected_statistics = collective_call(
        comm,
        "CPU statistics",
        lambda: validate_cpu_statistics(
            statistics, require_boundary=args.expected_size > 1
        ),
    )
    rank_records = comm.gather(
        {
            "rank": comm.Get_rank(),
            "energy": energy,
            "dft_norm": dft_norm,
            "meep_time": meep_time,
            "statistics": selected_statistics,
            "runtime_artifacts": runtime_artifacts,
        },
        root=0,
    )
    if comm.Get_rank() == 0:
        print(
            "gpmeep-cpu-mpi-fp64-qualification:"
            + json.dumps(
                {
                    "schema": SCHEMA,
                    "mpi_size": comm.Get_size(),
                    "runtime_kind": args.runtime_kind,
                    "single_precision": False,
                    "cuda_compiled": False,
                    "ranks": rank_records,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
