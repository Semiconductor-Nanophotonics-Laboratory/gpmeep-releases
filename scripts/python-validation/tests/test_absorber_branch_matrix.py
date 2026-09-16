"""Self-tests for the Absorber/PML branch-matrix wrapper."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np


WRAPPER_PATH = pathlib.Path(__file__).resolve().parents[1] / "absorber_branch_matrix.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_absorber_branch_matrix_test", WRAPPER_PATH)
assert SPEC and SPEC.loader
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


class AbsorberBranchMatrixTests(unittest.TestCase):
    @staticmethod
    def fake_run_path(_path: str, *, run_name: str):
        assert run_name == "__main__"
        use_pml = "-pml" in sys.argv
        times = np.linspace(0.0, 120.0, 2401)
        base = np.exp(-((times - 55.0) / 12.0) ** 2) * np.exp(1j * times)
        fields = base * (1.0 + (1e-4 if use_pml else 0.0))
        return {
            "absorber_result": {
                "branch": int(use_pml),
                "stop_time": 120.0,
                "timestep": 9600,
                "peak": float(np.max(np.abs(fields))),
                "tail_amplitude_ratio": 1e-3,
                "tail_power_ratio": 1e-6,
                "parameters": np.asarray([40.0, 20.0, 1e-5, 1.2, 0.1]),
                "times": times,
                "fields": fields,
            }
        }

    def make_args(self, example):
        return WRAPPER.parse_args([str(example), "--", "--validation"])

    def test_both_branches_and_argv_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "absorber.py"
            example.touch()
            previous_argv = sys.argv
            result = WRAPPER.run_branch_matrix(
                self.make_args(example), run_path=self.fake_run_path
            )
        self.assertIs(sys.argv, previous_argv)
        np.testing.assert_array_equal(result["absorber_branch_codes"], [0, 1])
        self.assertEqual(result["absorber_trace_fields"].shape, (2, 4096))
        self.assertLess(result["absorber_branch_relative_error"][0], 0.02)

    def test_bad_decay_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "absorber.py"
            example.touch()

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                namespace["absorber_result"]["tail_power_ratio"] = 1e-3
                return namespace

            with self.assertRaisesRegex(RuntimeError, "decay contract"):
                WRAPPER.run_branch_matrix(
                    self.make_args(example), run_path=bad_run_path
                )

    def test_aliased_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "absorber.py"
            example.touch()

            def sparse_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                indices = np.arange(0, 2401, 40)
                namespace["absorber_result"]["times"] = namespace["absorber_result"][
                    "times"
                ][indices]
                namespace["absorber_result"]["fields"] = namespace["absorber_result"][
                    "fields"
                ][indices]
                return namespace

            with self.assertRaisesRegex(RuntimeError, "undersamples"):
                WRAPPER.run_branch_matrix(
                    self.make_args(example), run_path=sparse_run_path
                )

    def test_nonfinite_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "absorber.py"
            example.touch()

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                namespace["absorber_result"]["fields"][10] = np.nan
                return namespace

            with self.assertRaisesRegex(RuntimeError, "invalid fields"):
                WRAPPER.run_branch_matrix(
                    self.make_args(example), run_path=bad_run_path
                )


if __name__ == "__main__":
    unittest.main()
