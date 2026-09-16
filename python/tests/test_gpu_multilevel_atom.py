import math
import unittest

import meep as mp
import numpy as np


class TestGpuMultilevelAtom(unittest.TestCase):
    def test_reduced_fp32_oracle_and_cuda_dispatch(self):
        transition = mp.Transition(
            1,
            2,
            pumping_rate=0.0051,
            frequency=40 / (2 * math.pi),
            gamma=8 / (2 * math.pi),
            sigma_diag=mp.Vector3(80, 0, 0),
        )
        decay = mp.Transition(2, 1, transition_rate=0.005)
        atom = mp.MultilevelAtom(
            sigma=1,
            transitions=[transition, decay],
            initial_populations=[28],
        )
        medium = mp.Medium(index=1.5, E_susceptibilities=[atom])
        simulation = mp.Simulation(
            cell_size=mp.Vector3(z=3),
            dimensions=1,
            resolution=32,
            boundary_layers=[mp.PML(1, side=mp.High)],
            geometry=[
                mp.Block(
                    center=mp.Vector3(z=-1),
                    size=mp.Vector3(mp.inf, mp.inf, 1),
                    material=medium,
                )
            ],
        )

        simulation.init_sim()
        simulation.initialize_field(
            mp.Ex, lambda point: 1 if point.z == -1 else 0
        )
        mp.gpu.reset_statistics()
        simulation.run(until=12)

        values = np.asarray(
            [
                simulation.get_field_point(
                    mp.Ex, mp.Vector3(z=position)
                ).real
                for position in (-1.25, -1.0, -0.75, -0.25, 0.5)
            ]
        )
        self.assertTrue(np.all(np.isfinite(values)))
        np.testing.assert_allclose(
            values,
            np.asarray(
                [
                    3.02422723e-06,
                    1.40880991e-04,
                    8.15244057e-05,
                    -1.06990963e-04,
                    3.47592199e-04,
                ]
            ),
            rtol=3e-5,
            atol=2e-8,
        )
        print(
            "multilevel-fp32-oracle:",
            ",".join(f"{value:.9g}" for value in values),
        )

        statistics = mp.gpu.statistics()["polarizations"]
        if mp.gpu.active_backend == "cuda":
            self.assertGreater(statistics["cuda_polarization_calls"], 0)
            self.assertEqual(statistics["cpu_polarization_calls"], 0)
        else:
            self.assertGreater(statistics["cpu_polarization_calls"], 0)
            self.assertEqual(statistics["cuda_polarization_calls"], 0)


if __name__ == "__main__":
    unittest.main()
