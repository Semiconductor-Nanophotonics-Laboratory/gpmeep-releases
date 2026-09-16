#!/usr/bin/env python3
"""Unit tests for the bounded point-dipole cylindrical validation profile."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[1]
EXAMPLE = REPO_ROOT / "python/examples/point_dipole_cyl.py"
sys.path.insert(0, str(SCRIPT_DIR))

import run_point_dipole_cyl_validation as launcher


def load_example_module():
    spec = importlib.util.spec_from_file_location("point_dipole_cyl_test_target", EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load point_dipole_cyl.py")
    module = importlib.util.module_from_spec(spec)
    # These are pure metric/default-contract tests.  Loading the example must
    # remain part of the documented harness self-test that requires neither a
    # built Meep package nor a GPU.
    with mock.patch.dict(sys.modules, {"meep": types.ModuleType("meep")}):
        spec.loader.exec_module(module)
    return module


class PointDipoleCylMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = load_example_module()

    def test_synthetic_symmetry_tail_and_contrast(self):
        radiated = np.asarray(
            [
                [1.00, 1.0001, 1.0001002],
                [1.80, 2.00, 2.0000002],
                [1.00, 1.0001, 1.0001002],
            ]
        )
        source = 2.0 * radiated
        ldos = 3.0 * radiated
        metrics = self.target.point_dipole_validation_metrics(
            radiated, source, ldos
        )
        np.testing.assert_array_equal(
            metrics["pm_symmetry_relative_errors"], np.zeros((4, 3))
        )
        self.assertEqual(metrics["tail_relative_changes"].shape, (4, 3))
        self.assertEqual(metrics["tail_contraction_ratios"].shape, (4, 3))
        self.assertGreater(float(np.max(metrics["m0_contrast"])), 0.4)
        np.testing.assert_allclose(metrics["efficiencies"], 0.5)
        self.target.validate_point_dipole_metrics(metrics)

    def test_symmetry_failure_is_rejected(self):
        radiated = np.asarray(
            [[1.0, 1.1, 1.11], [1.8, 2.0, 2.01], [1.0, 1.1, 1.11]]
        )
        metrics = self.target.point_dipole_validation_metrics(
            radiated, 2 * radiated, 3 * radiated
        )
        metrics["pm_symmetry_relative_errors"][0, 0] = 0.01
        with self.assertRaisesRegex(RuntimeError, "violate symmetry"):
            self.target.validate_point_dipole_metrics(metrics)

    def test_large_absolute_tail_correction_is_rejected(self):
        signal = np.asarray(
            [
                [1.0, 1.0001, 1.0001002],
                [1.8, 2.0, 2.0000002],
                [1.0, 1.0001, 1.0001002],
            ]
        )
        metrics = self.target.point_dipole_validation_metrics(
            signal, 2 * signal, 3 * signal
        )
        metrics["tail_signed_corrections"][0, 0] = 3e-6
        with self.assertRaisesRegex(RuntimeError, "absolute late correction"):
            self.target.validate_point_dipole_metrics(metrics)

    def test_weak_tail_contraction_is_rejected(self):
        signal = np.asarray(
            [
                [1.0, 1.0001, 1.0001002],
                [1.8, 2.0, 2.0000002],
                [1.0, 1.0001, 1.0001002],
            ]
        )
        metrics = self.target.point_dipole_validation_metrics(
            signal, 2 * signal, 3 * signal
        )
        metrics["tail_contraction_ratios"][0, 0] = 0.02
        with self.assertRaisesRegex(RuntimeError, "did not contract"):
            self.target.validate_point_dipole_metrics(metrics)

    def test_nonpositive_signal_is_rejected(self):
        signal = np.ones((3, 3))
        signal[1, 1] = 0
        with self.assertRaisesRegex(RuntimeError, "must be positive"):
            self.target.point_dipole_validation_metrics(
                signal, np.ones((3, 3)), np.ones((3, 3))
            )

    def test_wrong_shape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "has shape"):
            self.target.point_dipole_validation_metrics(
                np.ones((3, 2)), np.ones((3, 3)), np.ones((3, 3))
            )

    def test_published_function_retains_original_profile(self):
        fixture = {
            "radiated_fluxes": np.asarray([2.0]),
            "source_fluxes": np.asarray([4.0]),
        }
        with mock.patch.object(
            self.target, "_dipole_in_slab_profile", return_value=fixture
        ) as run_profile:
            self.assertEqual(self.target.dipole_in_slab(0.5, 3.5, 2), (2.0, 4.0))
        run_profile.assert_called_once_with(
            0.5,
            3.5,
            2,
            resolution_um=50,
            pml_um=1.0,
            padding_um=1.0,
            radius_um=20.0,
            source_fwidth_fraction=0.05,
            checkpoint_after_sources=None,
            flux_decay_threshold=1e-4,
            emit_flux=True,
        )


class PointDipoleCylLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = load_example_module()

    def test_oracle_contract_is_exact_and_complete(self):
        argv = launcher.build_oracle_argv(EXAMPLE)
        self.assertEqual(argv.count("--result-vector"), len(launcher.RESULT_SHAPES))
        self.assertEqual(
            argv.count("--expected-run-timestep-delta-range"),
            len(launcher.EXPECTED_TIMESTEP_DELTAS),
        )
        self.assertIn("--expected-run-count", argv)
        self.assertIn("9", argv)
        self.assertEqual(argv[-3:], [str(EXAMPLE.resolve()), "--", "--validation"])

    def test_launcher_constants_match_target_profile(self):
        self.assertEqual(self.target.VALIDATION_M_VALUES, (-1, 0, 1))
        self.assertEqual(
            self.target.VALIDATION_AFTER_SOURCES, (10.0, 30.0, 50.0)
        )
        self.assertEqual(launcher.EXPECTED_TIMESTEP_DELTAS, (1920, 640, 640) * 3)
        self.assertEqual(launcher.EXPECTED_FINAL_TIMESTEP, 3200)
        self.assertEqual(
            launcher.MAXIMUM_ABS["point_dipole_pm_symmetry_relative_errors"],
            self.target.VALIDATION_PM_SYMMETRY_LIMIT,
        )
        self.assertEqual(
            launcher.MAXIMUM_ABS["point_dipole_tail_relative_changes"],
            self.target.VALIDATION_TAIL_RELATIVE_LIMIT,
        )
        self.assertEqual(
            launcher.MAXIMUM_ABS["point_dipole_tail_signed_corrections"],
            self.target.VALIDATION_TAIL_CORRECTION_ABSOLUTE_LIMIT,
        )
        self.assertEqual(
            launcher.MAXIMUM_ABS["point_dipole_tail_contraction_ratios"],
            self.target.VALIDATION_TAIL_CONTRACTION_LIMIT,
        )


if __name__ == "__main__":
    unittest.main()
