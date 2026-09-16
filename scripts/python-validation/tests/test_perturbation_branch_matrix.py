"""Self-tests for the perturbation polarization branch wrapper."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np


WRAPPER_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "perturbation_branch_matrix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_perturbation_branch_matrix_test", WRAPPER_PATH
)
assert SPEC and SPEC.loader
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


class PerturbationBranchMatrixTests(unittest.TestCase):
    @staticmethod
    def fake_run_path(_path: str, *, run_name: str):
        assert run_name == "__main__"
        perpendicular = "-perpendicular" in sys.argv
        scale = 2.0 if perpendicular else 1.0
        return {
            "perturbation_frequencies": np.asarray([0.2, 0.199]),
            "perturbation_derivatives": np.asarray([-0.1 * scale, -0.11 * scale]),
            "perturbation_integrals": np.asarray([-0.5 * scale, 3.0]),
            "perturbation_relative_error": np.asarray([1 / 11]),
            "perturbation_mode_decays": np.asarray([-1e-4, -1.1e-4]),
            "perturbation_mode_q": np.asarray([1000.0, 904.5454545454545]),
            "perturbation_mode_amplitudes": np.asarray([1 + 2j, 2 - 1j]),
            "perturbation_mode_errors": np.asarray([1e-8, 2e-8]),
            "perturbation_polarization": np.asarray([int(perpendicular)]),
        }

    def test_both_polarizations_are_pinned_and_argv_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "perturbation.py"
            example.touch()
            args = WRAPPER.parse_args([str(example), "--resolution", "20"])
            previous_argv = sys.argv
            result = WRAPPER.run_branch_matrix(
                args, run_path=self.fake_run_path
            )
        self.assertIs(sys.argv, previous_argv)
        np.testing.assert_array_equal(result["branch_polarizations"], [0, 1])
        self.assertEqual(result["branch_perturbation_frequencies"].shape, (2, 2))
        self.assertEqual(result["branch_perturbation_derivatives"].shape, (2, 2))
        self.assertEqual(result["branch_perturbation_integrals"].shape, (2, 2))
        self.assertEqual(
            result["branch_perturbation_relative_errors"].shape, (2,)
        )
        self.assertEqual(result["branch_harminv_decays"].shape, (2, 2))
        self.assertEqual(result["branch_harminv_q"].shape, (2, 2))
        self.assertEqual(result["branch_harminv_amplitudes"].shape, (2, 2))
        self.assertEqual(result["branch_harminv_errors"].shape, (2, 2))

    def test_bad_finite_difference_closure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "perturbation.py"
            example.touch()
            args = WRAPPER.parse_args([str(example), "--resolution", "20"])

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                namespace["perturbation_relative_error"] = np.asarray([0.5])
                return namespace

            with self.assertRaisesRegex(RuntimeError, "finite-difference closure"):
                WRAPPER.run_branch_matrix(args, run_path=bad_run_path)

    def test_bad_harminv_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "perturbation.py"
            example.touch()
            args = WRAPPER.parse_args([str(example), "--resolution", "20"])

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                namespace["perturbation_mode_errors"] = np.asarray([1e-8, 2e-3])
                return namespace

            with self.assertRaisesRegex(RuntimeError, "inaccurate Harminv mode"):
                WRAPPER.run_branch_matrix(args, run_path=bad_run_path)


if __name__ == "__main__":
    unittest.main()
