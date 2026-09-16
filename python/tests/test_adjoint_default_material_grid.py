import json
import os
import subprocess
import sys
import tempfile
import unittest

from autograd import numpy as npa
import numpy as np

import meep as mp
import meep.adjoint as mpa


class TestAdjointDefaultMaterialGrid(unittest.TestCase):
    def _solve(self, gradient_path, *, beta=0, weights=None, need_gradient=True):
        previous_path = os.environ.get("MEEP_MATERIAL_GRADIENT_PATH")
        os.environ["MEEP_MATERIAL_GRADIENT_PATH"] = gradient_path
        try:
            mp.verbosity(0)
            cell = mp.Vector3(5, 5)
            shape = (5, 5)
            if weights is None:
                weights = np.linspace(
                    0.2, 0.8, np.prod(shape), dtype=np.float64
                )
            else:
                weights = np.asarray(weights, dtype=np.float64)
            center = mp.Vector3(0.4, -0.3)
            material_grid = mp.MaterialGrid(
                mp.Vector3(*shape),
                mp.Medium(epsilon=2.0),
                mp.Medium(epsilon=6.0),
                weights=weights,
                do_averaging=True,
                beta=beta,
            )
            design_region = mpa.DesignRegion(
                material_grid,
                volume=mp.Volume(center=center, size=cell),
            )
            simulation = mp.Simulation(
                cell_size=cell,
                geometry_center=center,
                resolution=10,
                default_material=material_grid,
                boundary_layers=[mp.PML(1)],
                sources=[
                    mp.Source(
                        mp.GaussianSource(
                            frequency=1 / 1.55,
                            fwidth=0.05 / 1.55,
                        ),
                        component=mp.Ez,
                        center=center + mp.Vector3(-1.5),
                    )
                ],
            )
            fields = mpa.FourierFields(
                simulation,
                mp.Volume(
                    center=center + mp.Vector3(1.25),
                    size=mp.Vector3(0.25, 1),
                ),
                mp.Ez,
            )

            def objective(values):
                return npa.abs(values[:, 1, 4]) ** 2

            problem = mpa.OptimizationProblem(
                simulation=simulation,
                objective_functions=objective,
                objective_arguments=[fields],
                design_regions=[design_region],
                frequencies=[1 / 1.55],
            )
            value, gradient = problem([weights], need_gradient=need_gradient)
            objective_value = float(np.asarray(value).reshape(-1)[0])
            if not need_gradient:
                return objective_value, None
            return objective_value, np.asarray(gradient, dtype=np.float64).reshape(
                -1
            )
        finally:
            if previous_path is None:
                os.environ.pop("MEEP_MATERIAL_GRADIENT_PATH", None)
            else:
                os.environ["MEEP_MATERIAL_GRADIENT_PATH"] = previous_path

    def test_default_material_grid_legacy_matches_analytic(self):
        legacy_value, legacy_gradient = self._solve("legacy")
        analytic_value, analytic_gradient = self._solve("analytic-required")

        self.assertTrue(np.isfinite(legacy_value))
        self.assertTrue(np.all(np.isfinite(legacy_gradient)))
        self.assertEqual(legacy_gradient.size, 25)
        self.assertEqual(legacy_value, analytic_value)
        np.testing.assert_allclose(
            analytic_gradient,
            legacy_gradient,
            rtol=2e-5,
            atol=2e-8,
        )

        weights = np.linspace(0.2, 0.8, analytic_gradient.size)
        step = 2e-3
        indices = np.arange(analytic_gradient.size, dtype=np.float64)
        directions = {
            "cosine": np.cos(indices),
            "quasiperiodic-sine": np.sin(indices * np.float64(0.61803398875)),
        }
        # Use three separated interior components. Boundary/corner weights can
        # have a zero central-FD response in this compact PML fixture even
        # though the discrete adjoint carries interpolation-edge roundoff.
        for component in (6, analytic_gradient.size // 2, 18):
            direction = np.zeros(analytic_gradient.size, dtype=np.float64)
            direction[component] = 1.0
            directions[f"component-{component}"] = direction

        for name, raw_direction in directions.items():
            with self.subTest(finite_difference_direction=name):
                direction = raw_direction / np.linalg.norm(raw_direction)
                plus_value, _ = self._solve(
                    "analytic-required",
                    weights=weights + step * direction,
                    need_gradient=False,
                )
                minus_value, _ = self._solve(
                    "analytic-required",
                    weights=weights - step * direction,
                    need_gradient=False,
                )
                finite_difference = (plus_value - minus_value) / (2 * step)
                adjoint_projection = float(np.dot(analytic_gradient, direction))
                self.assertGreater(abs(adjoint_projection), 5e-3)
                np.testing.assert_allclose(
                    adjoint_projection,
                    finite_difference,
                    rtol=8e-2,
                    atol=2e-5,
                )
                print(
                    "gpmeep-fd-direction:"
                    + json.dumps(
                        {
                            "name": name,
                            "finite_difference": float(finite_difference),
                            "adjoint_projection": adjoint_projection,
                            "rtol": 8e-2,
                            "atol": 2e-5,
                            "pass": True,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    def test_projected_grid_auto_uses_legacy_semantics(self):
        legacy_value, legacy_gradient = self._solve("legacy", beta=4)
        auto_value, auto_gradient = self._solve("auto", beta=4)

        self.assertEqual(legacy_value, auto_value)
        np.testing.assert_array_equal(auto_gradient, legacy_gradient)

    def test_analytic_required_rejects_every_unsupported_class(self):
        child = r"""
import json
import os
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

from autograd import numpy as npa
import numpy as np
import meep as mp
import meep.adjoint as mpa

case = os.environ["GPMEEP_REQUIRED_REJECTION_CASE"]
os.environ["MEEP_MATERIAL_GRADIENT_PATH"] = (
    "invalid-value" if case == "invalid-environment" else "analytic-required"
)
mp.verbosity(0)
shape = (3, 3, 3) if case == "3d" else (3, 3)
grid_size = mp.Vector3(*shape)
medium_1 = mp.Medium(epsilon=2)
medium_2 = mp.Medium(epsilon=5)
if case == "offdiagonal":
    medium_2 = mp.Medium(epsilon_diag=mp.Vector3(5, 5, 5), epsilon_offdiag=mp.Vector3(0.1))
if case == "dispersive":
    medium_2 = mp.Medium(
        epsilon=5,
        E_susceptibilities=[mp.LorentzianSusceptibility(frequency=1.5, gamma=0.1, sigma=0.2)],
    )
if case == "conductivity":
    medium_2 = mp.Medium(epsilon=5, D_conductivity=0.2)
material_grid = mp.MaterialGrid(
    grid_size,
    medium_1,
    medium_2,
    weights=np.linspace(0.25, 0.75, int(np.prod(shape))),
    beta=4 if case == "beta" else 0,
    grid_type={
        "u-mean": "U_MEAN",
        "u-min": "U_MIN",
        "u-prod": "U_PROD",
    }.get(case, "U_DEFAULT"),
)
cell = mp.Vector3(1.5, 1.5, 1.5 if case == "3d" else 0)
component = mp.Ex if case == "non-z" else mp.Ez
simulation = mp.Simulation(
    cell_size=cell,
    resolution=6,
    default_material=material_grid,
    sources=[
        mp.Source(
            mp.GaussianSource(1.0, fwidth=0.5),
            component=component,
            center=mp.Vector3(),
        )
    ],
)
design_region = mpa.DesignRegion(material_grid, volume=mp.Volume(size=cell))
fields = mpa.FourierFields(simulation, mp.Volume(size=cell), component)
problem = mpa.OptimizationProblem(
    simulation=simulation,
    objective_functions=lambda values: npa.mean(npa.abs(values) ** 2),
    objective_arguments=[fields],
    design_regions=[design_region],
    frequencies=[1.0],
    minimum_run_time=0.5,
    maximum_run_time=0.5,
)
problem([material_grid.weights], need_gradient=True)
"""
        cases = (
            "invalid-environment",
            "beta",
            "3d",
            "non-z",
            "dispersive",
            "conductivity",
            "offdiagonal",
            "u-mean",
            "u-min",
            "u-prod",
        )
        with tempfile.TemporaryDirectory() as temporary:
            for case in cases:
                with self.subTest(case=case):
                    env = os.environ.copy()
                    env["GPMEEP_REQUIRED_REJECTION_CASE"] = case
                    proc = subprocess.run(
                        [sys.executable, "-c", child],
                        cwd=temporary,
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=120,
                        check=False,
                    )
                    self.assertNotEqual(
                        proc.returncode,
                        0,
                        msg=f"{case} unexpectedly succeeded\n{proc.stdout}\n{proc.stderr}",
                    )
                    output = proc.stdout + proc.stderr
                    if case == "invalid-environment":
                        self.assertIn("must be legacy, auto, or analytic-required", output)
                    else:
                        self.assertIn("analytic-required MaterialGrid gradient", output)


if __name__ == "__main__":
    unittest.main()
