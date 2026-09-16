#!/usr/bin/env python3
"""Receipt-bound two-rank Python FDTD probe used by the MPI controller.

The controller owns process isolation and evidence publication.  This worker
only runs one fixed workload and emits one gathered, versioned JSON record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import socket
import struct
import time

from mpi4py import MPI
import numpy as np
import meep as mp


PREFIX = "gpmeep-mpi-python-probe-v1:"
SCHEMA_VERSION = 1
EXPECTED_RANKS = 2
RESOLUTION = 20
CELL_SIZE = 4.0
PML_THICKNESS = 0.5
RUN_UNTIL = 8.0
EXPECTED_TIMESTEPS = 320


def file_record(path: str | pathlib.Path) -> dict[str, object]:
    resolved = pathlib.Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def mapped_libmeep() -> dict[str, object]:
    paths: set[pathlib.Path] = set()
    for line in pathlib.Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith("/"):
            path = pathlib.Path(columns[5])
            if path.name.startswith("libmeep.so") and path.is_file():
                paths.add(path.resolve(strict=True))
    if len(paths) != 1:
        raise RuntimeError(f"expected one mapped libmeep, found {len(paths)}")
    return file_record(next(iter(paths)))


def canonical_field_sha256(values: list[float]) -> str:
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("field evidence contains a non-finite value")
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def physical_affinity() -> list[dict[str, object]]:
    cores = set()
    host = socket.gethostname()
    for logical_cpu in os.sched_getaffinity(0):
        topology = pathlib.Path(
            f"/sys/devices/system/cpu/cpu{logical_cpu}/topology"
        )
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
        cores.add((host, package, core))
    return [
        {"host": host_name, "package_id": package, "core_id": core}
        for host_name, package, core in sorted(cores)
    ]


def main() -> None:
    communicator = MPI.COMM_WORLD
    if communicator.size != EXPECTED_RANKS or mp.count_processors() != EXPECTED_RANKS:
        raise RuntimeError("MPI Python probe requires exactly two aligned ranks")
    lane = os.environ.get("GPMEEP_MPI_PROBE_LANE", "")
    expected_backend = "cpu" if lane == "cpu-hidden" else "cuda"
    if lane not in {
        "cpu-hidden",
        "cuda-pinned-waitsome",
        "cuda-aware-waitall",
    }:
        raise RuntimeError(f"invalid probe lane {lane!r}")
    receipt_id = os.environ.get("GPMEEP_MPI_PROBE_RECEIPT_ID", "")
    nonce = os.environ.get("GPMEEP_MPI_PROBE_NONCE", "")
    if len(receipt_id) != 64 or len(nonce) != 32:
        raise RuntimeError("probe receipt ID or nonce is absent")
    if not mp.with_mpi() or not mp.is_single_precision() or not mp.gpu.compiled:
        raise RuntimeError("probe requires the FP32 CUDA+MPI Python build")

    # Capture import/initialization effects before resetting workload counters.
    # In particular, the CPU-hidden lane must prove that importing the Python
    # extension did not initialize, enumerate, or select through CUDA.
    startup_statistics = mp.gpu.statistics()
    mp.gpu.reset_statistics()
    mp.gpu.set_backend(expected_backend)
    simulation = mp.Simulation(
        cell_size=mp.Vector3(CELL_SIZE, CELL_SIZE),
        resolution=RESOLUTION,
        boundary_layers=[mp.PML(PML_THICKNESS)],
        sources=[
            mp.Source(
                mp.GaussianSource(0.3, fwidth=0.1),
                component=mp.Ez,
                center=mp.Vector3(-0.7),
            )
        ],
    )
    flux = simulation.add_flux(
        0.3,
        0.1,
        3,
        mp.FluxRegion(center=mp.Vector3(0.7), size=mp.Vector3(y=2.0)),
        decimation_factor=1,
    )
    communicator.Barrier()
    started = time.perf_counter()
    simulation.run(until=RUN_UNTIL)
    local_seconds = time.perf_counter() - started
    communicator.Barrier()

    field = np.asarray(
        simulation.get_array(
            center=mp.Vector3(),
            size=mp.Vector3(CELL_SIZE, CELL_SIZE),
            component=mp.Ez,
        ),
        dtype=np.float64,
    )
    values = [float(value) for value in field.ravel(order="C")]
    flux_values = [float(value) for value in mp.get_fluxes(flux)]
    if not all(math.isfinite(value) for value in flux_values):
        raise RuntimeError("flux evidence contains a non-finite value")
    metrics = {
        "timesteps": int(simulation.timestep()),
        "meep_time": float(simulation.meep_time()),
        "shape": list(field.shape),
        "field_values": values,
        "field_sha256": canonical_field_sha256(values),
        "field_sum": float(np.sum(field, dtype=np.float64)),
        "field_l1": float(np.sum(np.abs(field), dtype=np.float64)),
        "field_l2": float(np.linalg.norm(field)),
        "field_maximum_absolute": float(np.max(np.abs(field))),
        "field_weighted_checksum": float(
            np.dot(field.ravel(order="C"), np.arange(1, field.size + 1))
        ),
        "flux_values": flux_values,
    }
    if metrics["timesteps"] != EXPECTED_TIMESTEPS:
        raise RuntimeError(
            f"probe executed {metrics['timesteps']} timesteps, expected {EXPECTED_TIMESTEPS}"
        )

    statistics = mp.gpu.statistics()
    active_backend = mp.gpu.active_backend
    rank_record = {
        "rank": communicator.rank,
        "pid": os.getpid(),
        "physical_affinity": physical_affinity(),
        "active_backend": active_backend,
        "requested_backend": mp.gpu.requested_backend,
        "backend_diagnostic": mp.gpu.backend_diagnostic,
        "selected_device": int(mp.gpu.selected_device),
        "selected_device_identifier": mp.gpu.selected_device_identifier,
        "local_seconds": local_seconds,
        "statistics": statistics,
        "startup_statistics": startup_statistics,
        "metrics": metrics,
        "runtime": {
            "python": file_record(os.sys.executable),
            "extension": file_record(mp._meep.__file__),
            "libmeep": mapped_libmeep(),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "MEEP_GPU_BACKEND",
                "MEEP_GPU_STRICT",
                "MEEP_GPU_ALLOW_OVERSUBSCRIBE",
                "MEEP_GPU_MPI_TRANSPORT",
                "MEEP_GPU_MPI_COMPLETION",
                "OMP_NUM_THREADS",
            )
        },
    }
    if active_backend != expected_backend:
        raise RuntimeError(
            f"rank {communicator.rank} selected {active_backend}, expected {expected_backend}"
        )

    # Keep both CUDA contexts observable to the one-second NVIDIA process
    # monitor.  This is execution evidence, not part of solver timing.
    if expected_backend == "cuda":
        time.sleep(2.2)
    rank_records = communicator.gather(rank_record, root=0)
    communicator.Barrier()
    if communicator.rank != 0:
        return
    if any(record["metrics"] != metrics for record in rank_records):
        raise RuntimeError("MPI ranks observed different global numerical metrics")
    record = {
        "schema_version": SCHEMA_VERSION,
        "lane": lane,
        "receipt_id": receipt_id,
        "nonce": nonce,
        "mpi_ranks": communicator.size,
        "workload": {
            "resolution": RESOLUTION,
            "cell_size": [CELL_SIZE, CELL_SIZE, 0.0],
            "pml_thickness": PML_THICKNESS,
            "run_until": RUN_UNTIL,
            "expected_timesteps": EXPECTED_TIMESTEPS,
        },
        "metrics": metrics,
        "rank_records": rank_records,
    }
    print(PREFIX + json.dumps(record, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
