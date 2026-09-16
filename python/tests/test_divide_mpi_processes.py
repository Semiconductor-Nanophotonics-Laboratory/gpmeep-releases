import hashlib
import json
import os
import pathlib
import unittest

import meep as mp


_VALIDATION_NONCE = os.environ.get("GPMEEP_DIVIDE_VALIDATION_NONCE", "")
_STARTUP_GPU_STATISTICS = mp.gpu.statistics() if _VALIDATION_NONCE else None
_VALIDATION_PREFIX = "gpmeep-divide-mpi-validation-v1:"


@unittest.skipIf(mp.count_processors() < 2, "MPI specific test")
class TestDivideParallelProcesses(unittest.TestCase):
    def test_divide_parallel_processes(self):
        validation_backend = os.environ.get("GPMEEP_DIVIDE_VALIDATION_BACKEND", "")
        if _VALIDATION_NONCE:
            if validation_backend not in {"cpu", "cuda"}:
                self.fail("invalid divide validation backend")
            mp.gpu.reset_statistics()
        resolution = 20

        sxy = 4
        dpml = 1
        cell = mp.Vector3(sxy + 2 * dpml, sxy + 2 * dpml)

        pml_layers = [mp.PML(dpml)]

        n = mp.divide_parallel_processes(2)
        fcen = 1.0 / (n + 1)

        sources = [
            mp.Source(
                src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
                center=mp.Vector3(),
                component=mp.Ez,
            )
        ]

        symmetries = [mp.Mirror(mp.X), mp.Mirror(mp.Y)]

        self.sim = mp.Simulation(
            cell_size=cell,
            resolution=resolution,
            sources=sources,
            symmetries=symmetries,
            boundary_layers=pml_layers,
        )

        flux_box = self.sim.add_flux(
            fcen,
            0,
            1,
            mp.FluxRegion(mp.Vector3(y=0.5 * sxy), size=mp.Vector3(sxy)),
            mp.FluxRegion(mp.Vector3(y=-0.5 * sxy), size=mp.Vector3(sxy), weight=-1),
            mp.FluxRegion(mp.Vector3(0.5 * sxy), size=mp.Vector3(y=sxy)),
            mp.FluxRegion(mp.Vector3(-0.5 * sxy), size=mp.Vector3(y=sxy), weight=-1),
            decimation_factor=1,
        )

        self.sim.run(until_after_sources=30)

        tot_flux = mp.get_fluxes(flux_box)[0]

        tot_fluxes = mp.merge_subgroup_data(tot_flux)
        fcens = mp.merge_subgroup_data(fcen)

        self.assertEqual(fcens[0], 1)
        self.assertEqual(fcens[1], 0.5)
        places = 4 if mp.is_single_precision() else 7
        self.assertAlmostEqual(tot_fluxes[0], 9.8628728533, places=places)
        self.assertAlmostEqual(tot_fluxes[1], 19.6537275387, places=places)
        if _VALIDATION_NONCE:
            from mpi4py import MPI

            record = {
                "schema_version": 1,
                "nonce": _VALIDATION_NONCE,
                "world_rank": MPI.COMM_WORLD.rank,
                "pid": os.getpid(),
                "expected_backend": validation_backend,
                "active_backend": mp.gpu.active_backend,
                "requested_backend": mp.gpu.requested_backend,
                "selected_device": int(mp.gpu.selected_device),
                "selected_device_identifier": mp.gpu.selected_device_identifier,
                "startup_statistics": _STARTUP_GPU_STATISTICS,
                "statistics": mp.gpu.statistics(),
            }
            records = MPI.COMM_WORLD.gather(record, root=0)
            if MPI.COMM_WORLD.rank == 0:
                telemetry_path = pathlib.Path(
                    os.environ.get("GPMEEP_DIVIDE_TELEMETRY_PATH", "")
                )
                if not telemetry_path.is_absolute() or not telemetry_path.parent.is_dir():
                    self.fail("invalid divide validation telemetry path")
                payload = (
                    json.dumps(
                        {"records": records}, sort_keys=True, allow_nan=False
                    )
                    + "\n"
                ).encode("utf-8")
                # The controller reads only after a successful mpiexec exit,
                # so exclusive direct creation is both fail-closed and avoids
                # replacing any path introduced after controller preflight.
                with telemetry_path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                directory_fd = os.open(telemetry_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                print(
                    _VALIDATION_PREFIX
                    + json.dumps(
                        {
                            "schema_version": 1,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        },
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
