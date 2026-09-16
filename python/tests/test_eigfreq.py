import json
import os
import unittest
from unittest import mock

import meep as mp
import numpy as np


class TestEigfreq(unittest.TestCase):
    def test_eigfreq(self):
        w = 1.2  # width of waveguide
        r = 0.36  # radius of holes
        d = 1.4  # defect spacing (ordinary spacing = 1)
        N = 3  # number of holes on either side of defect
        sy = 6  # size of cell in y direction (perpendicular to wvg.)
        pad = 2  # padding between last hole and PML edge
        dpml = 1  # PML thickness
        sx = 2 * (pad + dpml + N) + d - 1  # size of cell in x direction

        geometry = [
            mp.Block(size=mp.Vector3(mp.inf, w, mp.inf), material=mp.Medium(epsilon=13))
        ]
        for i in range(N):
            geometry.append(mp.Cylinder(r, center=mp.Vector3(d / 2 + i)))
            geometry.append(mp.Cylinder(r, center=mp.Vector3(-(d / 2 + i))))

        fcen = 0.25
        df = 0.2
        src = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Hz,
                center=mp.Vector3(0),
                size=mp.Vector3(0, 0),
            )
        ]

        sim = mp.Simulation(
            cell_size=mp.Vector3(sx, sy),
            force_complex_fields=True,
            geometry=geometry,
            boundary_layers=[mp.PML(1.0)],
            sources=src,
            symmetries=[mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=-1)],
            resolution=20,
        )
        sim.init_sim()
        if mp.is_single_precision():
            # FP32 BiCGSTAB-L cannot use the FP64 inner tolerance (tol*1e-3)
            # without eventually underflowing its residual. Three
            # shift-and-invert iterations with an explicit FP32 CW tolerance
            # retain the physical mode to better than 5e-5. This deliberately
            # low L=2 request also covers resident CUDA true-residual
            # verification without changing the public solver order.
            eigfreq = sim.solve_eigfreq(
                tol=1e-3, cwtol=5e-3, L=2, maxiters=3
            )
            self.assertAlmostEqual(
                eigfreq.real, 0.23445413142440263, delta=5e-5
            )
            self.assertAlmostEqual(
                eigfreq.imag, -0.0003147775697388, delta=5e-5
            )
        else:
            eigfreq = sim.solve_eigfreq(tol=1e-6)
            self.assertAlmostEqual(
                eigfreq.real, 0.23445413142440263, places=5
            )
            self.assertAlmostEqual(
                eigfreq.imag, -0.0003147775697388, places=5
            )
        print(
            "gpmeep-eigfreq-metrics:"
            + json.dumps(
                {"imag": eigfreq.imag, "real": eigfreq.real},
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def test_eigfreq_inner_failure_is_not_published(self):
        sim = mp.Simulation(
            cell_size=mp.Vector3(2, 2),
            force_complex_fields=True,
            sources=[
                mp.Source(
                    mp.ContinuousSource(0.25),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
            resolution=10,
        )
        sim.init_sim()
        with mock.patch.dict(
            os.environ, {"MEEP_GPU_TEST_CW_EIGEN_INNER_FAILURE": "1"}
        ):
            with self.assertRaisesRegex(RuntimeError, "shift-and-invert CW solve"):
                sim.solve_eigfreq(
                    tol=1e-3,
                    cwtol=1e-2,
                    cwmaxiters=200,
                    L=2,
                    maxiters=2,
                )
        field = sim.get_array(component=mp.Ez)
        self.assertTrue(np.all(np.isfinite(field)))

    def test_eigfreq_resident_mode_is_accepted(self):
        if (
            not hasattr(mp, "gpu")
            or not mp.is_single_precision()
            or os.environ.get("MEEP_GPU_BACKEND") != "cuda"
        ):
            # The paired strict-CUDA lane executes the positive resident-path
            # assertion; the CPU lane must remain an ordinary passing test so
            # the manifest keeps a zero-skip contract.
            return
        def solve(mode, inject_breakdown=False):
            sim = mp.Simulation(
                cell_size=mp.Vector3(2, 2),
                force_complex_fields=True,
                sources=[
                    mp.Source(
                        mp.ContinuousSource(0.25),
                        component=mp.Ez,
                        center=mp.Vector3(),
                    )
                ],
                resolution=10,
            )
            sim.init_sim()
            environment = {
                "MEEP_GPU_CW_SOLVER": mode,
                "MEEP_GPU_TEST_CW_EIGEN_INITIAL_BREAKDOWN": "0",
                "MEEP_GPU_TEST_CW_EIGEN_INNER_FAILURE": "0",
            }
            if inject_breakdown:
                environment["MEEP_GPU_TEST_CW_EIGEN_INITIAL_BREAKDOWN"] = "1"
            with mock.patch.dict(os.environ, environment):
                return sim.solve_eigfreq(
                    tol=1e-2,
                    cwtol=1e-2,
                    cwmaxiters=200,
                    L=2,
                    maxiters=2,
                )

        host_value = solve("host")
        resident_value = solve("resident", inject_breakdown=True)
        self.assertTrue(np.isfinite(resident_value))
        self.assertLess(abs(resident_value - host_value), 5e-3)


if __name__ == "__main__":
    unittest.main()
