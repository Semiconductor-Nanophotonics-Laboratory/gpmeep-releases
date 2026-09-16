import math
import unittest

import numpy as np

from meep.mie_theory import (
    mie_differential_cross_section,
    mie_scattering_cross_section,
    mie_scattering_efficiency,
)


class TestMieTheory(unittest.TestCase):
    def test_rayleigh_limit_and_frozen_series_values(self):
        refractive_index = 2.0
        size_parameters = np.array([0.01, 2.0, 4.0, 10.0])
        efficiencies = mie_scattering_efficiency(
            refractive_index, size_parameters
        )
        rayleigh = (
            8
            / 3
            * size_parameters[0] ** 4
            * abs(
                (refractive_index**2 - 1)
                / (refractive_index**2 + 2)
            )
            ** 2
        )
        self.assertLess(abs(efficiencies[0] - rayleigh) / rayleigh, 5e-5)
        np.testing.assert_allclose(
            efficiencies[1:],
            [4.77043984, 1.74766238, 2.04365094],
            rtol=2e-9,
            atol=1e-10,
        )

    def test_cross_section_and_scalar_vector_contract(self):
        radius = 1.25
        size_parameters = np.array([0.3, 1.5, 7.0])
        efficiencies = mie_scattering_efficiency(1.7 + 0.03j, size_parameters)
        cross_sections = mie_scattering_cross_section(
            1.7 + 0.03j, size_parameters, radius
        )
        np.testing.assert_allclose(
            cross_sections / (math.pi * radius**2), efficiencies
        )
        self.assertIsInstance(mie_scattering_efficiency(2.0, 1.0), float)
        self.assertEqual(efficiencies.shape, size_parameters.shape)
        self.assertTrue(np.all(np.isfinite(efficiencies)))
        self.assertTrue(np.all(efficiencies > 0))

    def test_differential_pattern_integrates_to_total_cross_section(self):
        radius = 0.7
        size_parameter = 3.4
        angles = np.linspace(0, math.pi, 20001)
        differential = mie_differential_cross_section(
            2.0, size_parameter, radius, angles
        )
        integrated = 2 * math.pi * np.trapezoid(
            differential * np.sin(angles), angles
        )
        expected = mie_scattering_cross_section(
            2.0, size_parameter, radius
        )
        self.assertLess(abs(integrated - expected) / expected, 1e-7)
        self.assertGreater(float(np.ptp(differential)), 0.1 * np.max(differential))
        self.assertIsInstance(
            mie_differential_cross_section(2.0, size_parameter, radius, 0.5),
            float,
        )

    def test_invalid_inputs_fail_closed(self):
        for size_parameter in (0, -1, float("nan"), float("inf")):
            with self.subTest(size_parameter=size_parameter):
                with self.assertRaises(ValueError):
                    mie_scattering_efficiency(2.0, size_parameter)
        for refractive_index in (0, complex(float("nan"), 0)):
            with self.subTest(refractive_index=refractive_index):
                with self.assertRaises(ValueError):
                    mie_scattering_efficiency(refractive_index, 1.0)
        with self.assertRaises(ValueError):
            mie_scattering_cross_section(2.0, 1.0, 0)
        with self.assertRaises(ValueError):
            mie_differential_cross_section(2.0, 1.0, 1.0, [-0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
