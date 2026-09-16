import unittest

import numpy as np

import meep as mp


class TestGenericExampleOracleState(unittest.TestCase):
    """The generic post-run probes must not perturb a later time step."""

    def _trajectory(self, probe_between_runs):
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.5, 1.5),
            resolution=12,
            boundary_layers=[mp.PML(0.25)],
            sources=[
                mp.Source(
                    mp.GaussianSource(0.35, fwidth=0.2),
                    component=mp.Ez,
                    center=mp.Vector3(-0.2),
                )
            ],
        )
        monitor = simulation.add_dft_fields(
            [mp.Ez],
            0.35,
            0,
            1,
            center=mp.Vector3(),
            size=mp.Vector3(0.75, 0.5),
            decimation_factor=1,
        )
        simulation.run(until=2)
        timestep_before_probe = simulation.timestep()
        if probe_between_runs:
            energy = simulation.field_energy_in_box(
                box=simulation.fields.total_volume()
            )
            dft_norm = simulation.fields.dft_norm()
            self.assertTrue(np.isfinite(energy))
            self.assertTrue(np.isfinite(dft_norm))
        self.assertEqual(simulation.timestep(), timestep_before_probe)
        simulation.run(until=2)
        fields = np.asarray(
            simulation.get_array(
                center=mp.Vector3(),
                size=mp.Vector3(1.0, 0.75),
                component=mp.Ez,
            )
        ).copy()
        dft_fields = np.asarray(
            simulation.get_dft_array(monitor, mp.Ez, 0)
        ).copy()
        final_energy = simulation.field_energy_in_box(
            box=simulation.fields.total_volume()
        )
        final_timestep = simulation.timestep()
        simulation.reset_meep()
        return fields, dft_fields, final_energy, final_timestep

    def test_post_run_probes_preserve_later_trajectory(self):
        control = self._trajectory(False)
        probed = self._trajectory(True)
        self.assertEqual(control[3], probed[3])
        np.testing.assert_allclose(control[0], probed[0], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(control[1], probed[1], rtol=2e-6, atol=2e-7)
        self.assertAlmostEqual(control[2], probed[2], delta=2e-7)


if __name__ == "__main__":
    unittest.main()
