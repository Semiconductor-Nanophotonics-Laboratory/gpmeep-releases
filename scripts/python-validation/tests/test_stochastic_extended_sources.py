"""Self-tests for line-basis and reciprocity validation wrappers."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


WRAPPER_DIR = pathlib.Path(__file__).resolve().parents[1]


def load_wrapper(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, WRAPPER_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LINE = load_wrapper(
    "stochastic_line_basis_matrix.py", "gpmeep_stochastic_line_basis_test"
)
RECIPROCITY = load_wrapper(
    "stochastic_reciprocity_matrix.py", "gpmeep_stochastic_reciprocity_test"
)
sys.modules["stochastic_reciprocity_matrix"] = RECIPROCITY
RESOLUTION_PROBE = load_wrapper(
    "stochastic_reciprocity_resolution_probe.py",
    "gpmeep_stochastic_reciprocity_resolution_probe_test",
)
EXAMPLE_ORACLE = load_wrapper(
    "run_example_oracle.py", "gpmeep_stochastic_resolution_oracle_test"
)
sys.modules["run_example_oracle"] = EXAMPLE_ORACLE
RESOLUTION_LAUNCHER = load_wrapper(
    "run_stochastic_reciprocity_resolution90.py",
    "gpmeep_stochastic_reciprocity_resolution_launcher_test",
)
TEMPORAL_PROBE = load_wrapper(
    "stochastic_reciprocity_temporal_extension_probe.py",
    "gpmeep_stochastic_reciprocity_temporal_extension_probe_test",
)
TEMPORAL_LAUNCHER = load_wrapper(
    "run_stochastic_reciprocity_temporal512.py",
    "gpmeep_stochastic_reciprocity_temporal_extension_launcher_test",
)
sys.modules["run_stochastic_reciprocity_temporal512"] = TEMPORAL_LAUNCHER
STABILITY_POINT = load_wrapper(
    "stochastic_reciprocity_stability_point.py",
    "gpmeep_stochastic_reciprocity_stability_point_test",
)
STABILITY_LAUNCHER = load_wrapper(
    "run_stochastic_reciprocity_stability_point.py",
    "gpmeep_stochastic_reciprocity_stability_launcher_test",
)
EXECUTION_IDENTITY = load_wrapper(
    "gpmeep_execution_identity.py",
    "gpmeep_execution_identity_test",
)
sys.modules["gpmeep_execution_identity"] = EXECUTION_IDENTITY
STABILITY_TRACE = load_wrapper(
    "stochastic_reciprocity_stability_trace.py",
    "gpmeep_stochastic_reciprocity_stability_trace_test",
)
sys.modules["stochastic_reciprocity_stability_trace"] = STABILITY_TRACE
TRACE_NEUTRALITY = load_wrapper(
    "stochastic_reciprocity_trace_neutrality.py",
    "gpmeep_stochastic_reciprocity_trace_neutrality_test",
)


class StochasticLineBasisTests(unittest.TestCase):
    @staticmethod
    def fake_run_path(_path: str, *, run_name: str):
        assert run_name == "__main__"
        textured = "-textured" in sys.argv
        frequencies = int(sys.argv[sys.argv.index("-nf") + 1])
        namespace = {
            "np": np,
            "sx": 1.5,
            "textured_scale": 1.7 if textured else 1.0,
            "frequency_grid": np.linspace(0.9, 1.1, frequencies),
        }
        exec(
            """
def src_amp_func(n):
    def amplitude(point):
        if n == 0:
            return 1 / np.sqrt(sx)
        return np.sqrt(2 / sx) * np.cos(n * np.pi * (point.x + 0.5 * sx) / sx)
    return amplitude

def compute_flux(_method, n):
    class Point:
        def __init__(self, x):
            self.x = x
    points = [
        Point(-0.43 * sx),
        Point(-0.31 * sx),
        Point(-0.07 * sx),
        Point(0.19 * sx),
        Point(0.41 * sx),
    ]
    amplitude = src_amp_func(n)
    quadratic = np.asarray([amplitude(point) ** 2 for point in points])
    if n == 14:
        quadratic = np.zeros_like(quadratic)
    return frequency_grid, textured_scale * quadratic * np.linspace(1.0, 2.0, frequency_grid.size)
""",
            namespace,
        )
        return namespace

    def make_args(self, example: pathlib.Path):
        return LINE.parse_args(
            [str(example), "--resolution", "50", "--frequencies", "5"]
        )

    def test_both_geometries_close_under_unitary_basis_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "line.py"
            example.touch()
            previous_argv = sys.argv
            result = LINE.run_basis_matrix(
                self.make_args(example), run_path=self.fake_run_path
            )
        self.assertIs(sys.argv, previous_argv)
        self.assertEqual(result["line_flat_basis_flux"].shape, (5, 2))
        self.assertEqual(result["line_flat_basis_flux_full"].shape, (5, 15))
        self.assertEqual(result["line_full_basis_gram"].shape, (15, 15))
        self.assertEqual(
            result["line_full_basis_active_mode_mask"].shape, (2, 15)
        )
        self.assertEqual(
            result["line_full_basis_active_mode_count"].shape, (2,)
        )
        self.assertTrue(
            np.all(result["line_full_basis_active_mode_count"] >= (9, 13))
        )
        active_mode_mask = result["line_full_basis_active_mode_mask"].astype(bool)
        self.assertTrue(np.all(~active_mode_mask[:, 14]))
        self.assertEqual(
            result["line_full_basis_pairwise_relative_l2"].shape, (2, 15, 15)
        )
        self.assertEqual(
            result["line_full_basis_min_pairwise_relative_l2"].shape, (2,)
        )
        self.assertGreaterEqual(
            np.min(result["line_full_basis_min_pairwise_relative_l2"]), 0.05
        )
        self.assertTrue(
            np.all(
                np.diagonal(
                    result["line_full_basis_pairwise_relative_l2"],
                    axis1=1,
                    axis2=2,
                )
                == 0
            )
        )
        active_pair_mask = (
            active_mode_mask[:, :, np.newaxis]
            & active_mode_mask[:, np.newaxis, :]
            & ~np.eye(15, dtype=bool)[np.newaxis, :, :]
        )
        self.assertTrue(
            np.all(
                result["line_full_basis_pairwise_relative_l2"][~active_pair_mask]
                == 0
            )
        )
        self.assertEqual(result["line_textured_rotated_flux"].shape, (5, 2))
        self.assertEqual(result["line_basis_callback_counts"].shape, (2, 15))
        self.assertTrue(np.all(result["line_basis_callback_counts"] > 0))
        self.assertEqual(result["line_flat_scaled_high_mode_flux"].shape, (5, 13))
        self.assertEqual(
            result["line_scaled_high_mode_callback_counts"].shape, (2, 13)
        )
        self.assertTrue(np.all(result["line_scaled_high_mode_callback_counts"] > 0))
        self.assertEqual(
            result["line_scaled_high_mode_residual"].shape, (2, 5, 13)
        )
        self.assertLess(
            np.max(result["line_scaled_high_mode_relative_l2"]), 1e-12
        )
        self.assertEqual(result["line_unitary_closure_residual"].shape, (2, 5))
        self.assertLess(np.max(result["line_unitary_closure_relative_l2"]), 1e-12)
        self.assertLess(result["line_m12_m15_convergence_relative_l2"][0], 1e-12)
        self.assertGreater(result["line_texture_relative_difference"][0], 0.1)

    def test_nonfinite_line_spectrum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "line.py"
            example.touch()

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                original = namespace["compute_flux"]

                def compute_flux(method, index):
                    frequencies, spectrum = original(method, index)
                    spectrum = np.asarray(spectrum).copy()
                    if index == 3:
                        spectrum[0] = np.nan
                    return frequencies, spectrum

                namespace["compute_flux"] = compute_flux
                return namespace

            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                LINE.run_basis_matrix(
                    self.make_args(example), run_path=bad_run_path
                )

    def test_ignored_amp_func_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "line.py"
            example.touch()

            def ignored_run_path(_path: str, *, run_name: str):
                assert run_name == "__main__"
                frequencies = int(sys.argv[sys.argv.index("-nf") + 1])
                namespace = {
                    "np": np,
                    "sx": 1.5,
                    "frequency_grid": np.linspace(0.9, 1.1, frequencies),
                }
                exec(
                    """
def src_amp_func(_index):
    return lambda _point: 1.0

def compute_flux(_method, index):
    # Deliberately never retrieves or invokes src_amp_func.
    return frequency_grid, (1.0 + index) * np.linspace(1.0, 2.0, frequency_grid.size)
""",
                    namespace,
                )
                return namespace

            with self.assertRaisesRegex(RuntimeError, "amp_func was not evaluated"):
                LINE.run_basis_matrix(
                    self.make_args(example), run_path=ignored_run_path
                )

    def test_source_factory_is_restored_after_basis_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "line.py"
            example.touch()
            previous_argv = sys.argv
            try:
                sys.argv = [str(example), "-nf", "5"]
                namespace = self.fake_run_path(
                    str(example), run_name="__main__"
                )
            finally:
                sys.argv = previous_argv
        original_factory = namespace["src_amp_func"]
        namespace["failure_index"] = 4
        exec(
            """
original_compute_flux = compute_flux
def failing_compute_flux(method, index):
    if index == failure_index:
        raise RuntimeError("synthetic basis failure")
    return original_compute_flux(method, index)
compute_flux = failing_compute_flux
""",
            namespace,
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic basis failure"):
            LINE._basis_spectra(namespace, 5)
        self.assertIs(
            namespace["compute_flux"].__globals__["src_amp_func"],
            original_factory,
        )

    def test_invoked_but_ignored_high_mode_amplitudes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "line.py"
            example.touch()
            loaded_namespaces = []

            def ignored_high_mode_run_path(_path: str, *, run_name: str):
                assert run_name == "__main__"
                textured = "-textured" in sys.argv
                frequencies = int(sys.argv[sys.argv.index("-nf") + 1])
                namespace = {
                    "np": np,
                    "sx": 1.5,
                    "textured_scale": 1.7 if textured else 1.0,
                    "frequency_grid": np.linspace(0.9, 1.1, frequencies),
                }
                exec(
                    """
def src_amp_func(n):
    def amplitude(point):
        if n == 0:
            return 1 / np.sqrt(sx)
        return np.sqrt(2 / sx) * np.cos(n * np.pi * (point.x + 0.5 * sx) / sx)
    return amplitude

def compute_flux(_method, n):
    class Point:
        def __init__(self, x):
            self.x = x
    points = [Point(-0.37 * sx), Point(-0.11 * sx), Point(0.29 * sx)]
    weights = [0.7, 1.3, 0.9]
    amplitude = src_amp_func(n)
    sampled = [amplitude(point) for point in points]
    if n >= 2:
        # The callback is invoked, but every returned high-mode amplitude is
        # deliberately discarded and replaced by the mode-0 amplitude.
        sampled = [1 / np.sqrt(sx)] * len(points)
    quadratic = sum(w * value**2 for w, value in zip(weights, sampled))
    return frequency_grid, textured_scale * quadratic * np.linspace(1.0, 2.0, frequency_grid.size)
""",
                    namespace,
                )
                loaded_namespaces.append(namespace)
                return namespace

            with self.assertRaisesRegex(RuntimeError, "scaled line-source high mode"):
                LINE.run_basis_matrix(
                    self.make_args(example), run_path=ignored_high_mode_run_path
                )
            self.assertEqual(len(loaded_namespaces), 2)
            for namespace in loaded_namespaces:
                self.assertIs(
                    namespace["compute_flux"].__globals__["src_amp_func"],
                    namespace["src_amp_func"],
                )

    def test_scaled_but_collapsed_high_mode_identities_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "line.py"
            example.touch()

            def collapsed_high_mode_run_path(_path: str, *, run_name: str):
                assert run_name == "__main__"
                textured = "-textured" in sys.argv
                frequencies = int(sys.argv[sys.argv.index("-nf") + 1])
                namespace = {
                    "np": np,
                    "sx": 1.5,
                    "textured_scale": 1.7 if textured else 1.0,
                    "frequency_grid": np.linspace(0.9, 1.1, frequencies),
                }
                exec(
                    """
def src_amp_func(n):
    def amplitude(point):
        if n == 0:
            return 1 / np.sqrt(sx)
        return np.sqrt(2 / sx) * np.cos(n * np.pi * (point.x + 0.5 * sx) / sx)
    return amplitude

canonical_src_amp_func = src_amp_func

def compute_flux(_method, n):
    class Point:
        def __init__(self, x):
            self.x = x
    points = [
        Point(-0.43 * sx),
        Point(-0.31 * sx),
        Point(-0.07 * sx),
        Point(0.19 * sx),
        Point(0.41 * sx),
    ]
    amplitude = src_amp_func(n)
    sampled = np.asarray([amplitude(point) for point in points])
    if n >= 2:
        canonical = canonical_src_amp_func(n)
        canonical_sampled = np.asarray([canonical(point) for point in points])
        consumed_scale = np.linalg.norm(sampled) / np.linalg.norm(canonical_sampled)
        mode_zero = canonical_src_amp_func(0)
        sampled = consumed_scale * np.asarray([mode_zero(point) for point in points])
    quadratic = sampled**2
    return frequency_grid, textured_scale * quadratic * np.linspace(1.0, 2.0, frequency_grid.size)
""",
                    namespace,
                )
                return namespace

            with self.assertRaisesRegex(RuntimeError, "pairwise separation"):
                LINE.run_basis_matrix(
                    self.make_args(example), run_path=collapsed_high_mode_run_path
                )

    def test_zero_high_modes_are_rejected_by_active_mode_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "line.py"
            example.touch()

            def zero_high_mode_run_path(_path: str, *, run_name: str):
                assert run_name == "__main__"
                textured = "-textured" in sys.argv
                frequencies = int(sys.argv[sys.argv.index("-nf") + 1])
                namespace = {
                    "np": np,
                    "sx": 1.5,
                    "textured_scale": 1.7 if textured else 1.0,
                    "frequency_grid": np.linspace(0.9, 1.1, frequencies),
                }
                exec(
                    """
def src_amp_func(n):
    def amplitude(point):
        if n == 0:
            return 1 / np.sqrt(sx)
        return np.sqrt(2 / sx) * np.cos(n * np.pi * (point.x + 0.5 * sx) / sx)
    return amplitude

def compute_flux(_method, n):
    class Point:
        def __init__(self, x):
            self.x = x
    points = [
        Point(-0.43 * sx),
        Point(-0.31 * sx),
        Point(-0.07 * sx),
        Point(0.19 * sx),
        Point(0.41 * sx),
    ]
    amplitude = src_amp_func(n)
    sampled = np.asarray([amplitude(point) for point in points])
    if n >= 2:
        sampled = np.zeros_like(sampled)
    quadratic = sampled**2
    return frequency_grid, textured_scale * quadratic * np.linspace(1.0, 2.0, frequency_grid.size)
""",
                    namespace,
                )
                return namespace

            with self.assertRaisesRegex(RuntimeError, "active mode count"):
                LINE.run_basis_matrix(
                    self.make_args(example), run_path=zero_high_mode_run_path
                )


class StochasticReciprocityTests(unittest.TestCase):
    @staticmethod
    def fake_run_path(_path: str, *, run_name: str):
        assert run_name == "gpmeep_reciprocity_target"
        namespace = {"np": np, "sx": 1.0, "resolution": 99, "nfreq": 99, "ndipole": 99}
        exec(
            """
def runtime_factor(runtime):
    # Model a small, monotone finite-runtime error that cancels only after the
    # complete textured/flat ratios are formed at a common convergence level.
    return 1.0 - 0.16 / runtime

def forward(n, runtime, textured):
    frequencies = np.linspace(0.9, 1.1, nfreq)
    if not textured:
        factor = 1.0
    else:
        mirror_index = min(n, ndipole - n)
        factor = 2.0 + mirror_index
    return frequencies, runtime_factor(runtime) * factor * np.linspace(1.0, 1.5, nfreq)

def backward(runtime, textured, return_metadata=False):
    frequencies = np.linspace(0.9, 1.1, nfreq)
    unique_indices, mirror_weights = reciprocity_module.mirror_quadrature(ndipole)
    texture_ratio = sum(
        weight * (2.0 + index)
        for index, weight in zip(unique_indices, mirror_weights)
    ) / ndipole
    spectrum = runtime_factor(runtime) * (texture_ratio if textured else 1.0) * np.linspace(1.0, 1.5, nfreq)
    spacing = sx / ndipole
    x = -0.5 * sx - spacing + spacing * np.arange(ndipole + 3)
    weights = np.zeros(ndipole + 3)
    weights[1] = 0.5 * spacing
    weights[-2] = 0.5 * spacing
    weights[2:-2] = spacing
    raw_power = np.repeat((spectrum / sx)[:, np.newaxis], ndipole + 3, axis=1)
    # Zero-weight ghosts can contain arbitrary finite values without changing
    # the cubature integral.
    raw_power[:, 0] = 1e12
    raw_power[:, -1] = 2e12
    if return_metadata:
        return frequencies, spectrum, {
            "x": x,
            "weights": weights,
            "raw_power": raw_power,
        }
    return frequencies, spectrum
""",
            namespace,
        )
        namespace["reciprocity_module"] = RECIPROCITY
        return namespace

    def make_args(self, example: pathlib.Path):
        return RECIPROCITY.parse_args(
            [str(example), "--resolution", "4", "--frequencies", "3"]
        )

    def test_mirror_quadrature_and_reciprocity_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            result = RECIPROCITY.run_reciprocity_matrix(
                self.make_args(example), run_path=self.fake_run_path
            )
        np.testing.assert_array_equal(result["reciprocity_textured_weights"], [1, 2, 1])
        self.assertEqual(result["reciprocity_forward_textured_unique_flux"].shape, (3, 3))
        self.assertEqual(
            result["reciprocity_forward_textured_unique_doubled_flux"].shape,
            (3, 3),
        )
        self.assertEqual(result["reciprocity_backward_metadata_x"].shape, (4, 7))
        self.assertEqual(
            result["reciprocity_backward_metadata_raw_power"].shape,
            (4, 3, 7),
        )
        np.testing.assert_array_equal(
            result["reciprocity_backward_metadata_effective_count"],
            [4, 4, 4, 4],
        )
        self.assertLess(result["reciprocity_flat_translation_relative_l2"][0], 1e-12)
        self.assertLess(result["reciprocity_textured_reflection_relative_l2"][0], 1e-12)
        self.assertLess(np.max(result["reciprocity_convergence_relative_l2"]), 0.1)
        self.assertGreater(np.max(result["reciprocity_convergence_relative_l2"]), 0.0)
        self.assertLess(
            np.max(result["reciprocity_convergence_pointwise_relative_error"]),
            0.1,
        )
        self.assertLess(
            np.max(
                result[
                    "reciprocity_convergence_scale_aware_pointwise_relative_error"
                ]
            ),
            0.05,
        )
        self.assertLess(
            result["reciprocity_forward_normalized_convergence_relative_l2"][0],
            0.02,
        )
        self.assertLess(
            np.max(
                result[
                    "reciprocity_forward_normalized_convergence_pointwise_relative_error"
                ]
            ),
            0.05,
        )
        self.assertLess(
            np.max(
                result[
                    "reciprocity_forward_normalized_convergence_scale_aware_pointwise_relative_error"
                ]
            ),
            0.05,
        )
        self.assertLess(
            result["reciprocity_backward_normalized_convergence_relative_l2"][0],
            0.02,
        )
        np.testing.assert_array_equal(
            result["reciprocity_backward_metadata_ghost_weight_max_abs"],
            np.zeros(4),
        )
        np.testing.assert_array_equal(
            result["reciprocity_backward_metadata_weighted_ghost_contribution"],
            np.zeros((4, 3)),
        )
        np.testing.assert_allclose(
            result["reciprocity_backward_expected_recomputed_flux"],
            result["reciprocity_backward_recomputed_flux"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertLess(result["reciprocity_relative_l2"][0], 1e-12)
        self.assertLess(result["reciprocity_max_relative_error"][0], 1e-12)
        self.assertLess(
            result["reciprocity_max_scale_aware_pointwise_relative_error"][0],
            1e-12,
        )
        np.testing.assert_array_equal(
            result["reciprocity_convergence_runtimes"], [32, 128, 64, 256]
        )

    def test_scale_aware_pointwise_allows_weak_bin_pure_drift(self) -> None:
        reference = np.asarray([1e-9, 1.0, 1.0, 1.0])
        residual = np.asarray([1e-9, 0.0, 0.0, 0.0])
        pure, scale_aware = RECIPROCITY._pointwise_relative_errors(
            residual, reference
        )
        self.assertEqual(pure[0], 1.0)
        self.assertLess(np.max(scale_aware), 0.05)
        RECIPROCITY._gate_scale_aware_pointwise(scale_aware, "weak-bin control")

    def test_scale_aware_pointwise_rejects_strong_bin_drift(self) -> None:
        reference = np.asarray([1e-9, 1.0, 1.0, 1.0])
        residual = np.asarray([0.0, 0.06, 0.0, 0.0])
        pure, scale_aware = RECIPROCITY._pointwise_relative_errors(
            residual, reference
        )
        self.assertGreater(np.max(pure), 0.05)
        self.assertGreater(np.max(scale_aware), 0.05)
        with self.assertRaisesRegex(RuntimeError, "scale-aware pointwise"):
            RECIPROCITY._gate_scale_aware_pointwise(
                scale_aware, "strong-bin control"
            )

    def test_bounded_resolution_reciprocity_closure_accepts_four_percent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()

            def shifted_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                original = namespace["backward"]

                def backward(runtime, textured, return_metadata=False):
                    frequencies, spectrum, metadata = original(
                        runtime, textured, return_metadata=True
                    )
                    if textured:
                        spectrum = np.asarray(spectrum) * 1.04
                        metadata = dict(metadata)
                        metadata["raw_power"] = (
                            np.asarray(metadata["raw_power"]) * 1.04
                        )
                    if return_metadata:
                        return frequencies, spectrum, metadata
                    return frequencies, spectrum

                namespace["backward"] = backward
                return namespace

            result = RECIPROCITY.run_reciprocity_matrix(
                self.make_args(example), run_path=shifted_run_path
            )

        self.assertGreater(result["reciprocity_relative_l2"][0], 0.03)
        self.assertLess(
            result["reciprocity_relative_l2"][0],
            RECIPROCITY.RECIPROCITY_CLOSURE_RELATIVE_L2_LIMIT,
        )
        self.assertLess(
            result["reciprocity_max_scale_aware_pointwise_relative_error"][0],
            0.05,
        )

    def test_bounded_resolution_reciprocity_closure_rejects_six_percent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()

            def shifted_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                original = namespace["backward"]

                def backward(runtime, textured, return_metadata=False):
                    frequencies, spectrum, metadata = original(
                        runtime, textured, return_metadata=True
                    )
                    if textured:
                        spectrum = np.asarray(spectrum) * 1.06
                        metadata = dict(metadata)
                        metadata["raw_power"] = (
                            np.asarray(metadata["raw_power"]) * 1.06
                        )
                    if return_metadata:
                        return frequencies, spectrum, metadata
                    return frequencies, spectrum

                namespace["backward"] = backward
                return namespace

            with self.assertRaisesRegex(RuntimeError, "does not close"):
                RECIPROCITY.run_reciprocity_matrix(
                    self.make_args(example), run_path=shifted_run_path
                )


    def test_near_zero_ghost_weight_with_large_power_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                original = namespace["backward"]

                def backward(runtime, textured, return_metadata=False):
                    frequencies, spectrum, metadata = original(
                        runtime, textured, return_metadata=True
                    )
                    metadata = dict(metadata)
                    metadata["weights"] = np.asarray(metadata["weights"]).copy()
                    metadata["weights"][0] = 1e-8
                    if return_metadata:
                        return frequencies, spectrum, metadata
                    return frequencies, spectrum

                namespace["backward"] = backward
                return namespace

            with self.assertRaisesRegex(RuntimeError, "ghost cubature weights"):
                RECIPROCITY.run_reciprocity_matrix(
                    self.make_args(example), run_path=bad_run_path
                )

    def test_machine_scale_ghost_weight_with_large_power_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                original = namespace["backward"]

                def backward(runtime, textured, return_metadata=False):
                    frequencies, spectrum, metadata = original(
                        runtime, textured, return_metadata=True
                    )
                    metadata = dict(metadata)
                    metadata["weights"] = np.asarray(metadata["weights"]).copy()
                    # This is below the explicit machine-scale weight ceiling,
                    # but its product with the deliberately huge ghost power is
                    # physically non-negligible and must fail the second gate.
                    metadata["weights"][0] = 5e-15
                    if return_metadata:
                        return frequencies, spectrum, metadata
                    return frequencies, spectrum

                namespace["backward"] = backward
                return namespace

            with self.assertRaisesRegex(RuntimeError, "weighted ghost contribution"):
                RECIPROCITY.run_reciprocity_matrix(
                    self.make_args(example), run_path=bad_run_path
                )

    def test_reported_flux_must_match_canonical_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                original = namespace["backward"]

                def backward(runtime, textured, return_metadata=False):
                    frequencies, _spectrum, metadata = original(
                        runtime, textured, return_metadata=True
                    )
                    metadata = dict(metadata)
                    metadata["weights"] = np.asarray(metadata["weights"]).copy()
                    metadata["weights"][2] += 1e-7
                    spectrum = np.sum(
                        np.asarray(metadata["raw_power"])
                        * metadata["weights"][np.newaxis, :],
                        axis=1,
                    )
                    if return_metadata:
                        return frequencies, spectrum, metadata
                    return frequencies, spectrum

                namespace["backward"] = backward
                return namespace

            with self.assertRaisesRegex(RuntimeError, "canonical DFT cubature"):
                RECIPROCITY.run_reciprocity_matrix(
                    self.make_args(example), run_path=bad_run_path
                )

    def test_normalized_ratio_runtime_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            args = self.make_args(example)

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                original = namespace["forward"]

                def forward(index, runtime, textured):
                    frequencies, spectrum = original(index, runtime, textured)
                    if textured and runtime == 2 * args.textured_runtime:
                        spectrum = np.asarray(spectrum) * 1.03
                    return frequencies, spectrum

                namespace["forward"] = forward
                return namespace

            with self.assertRaisesRegex(
                RuntimeError, "forward aggregate normalized textured/flat ratio"
            ):
                RECIPROCITY.run_reciprocity_matrix(
                    args, run_path=bad_run_path
                )

    def test_odd_and_even_periodic_mirror_weights(self) -> None:
        even_indices, even_weights = RECIPROCITY.mirror_quadrature(4)
        odd_indices, odd_weights = RECIPROCITY.mirror_quadrature(5)
        np.testing.assert_array_equal(even_indices, [0, 1, 2])
        np.testing.assert_array_equal(even_weights, [1, 2, 1])
        np.testing.assert_array_equal(odd_indices, [0, 1, 2])
        np.testing.assert_array_equal(odd_weights, [1, 2, 2])
        self.assertEqual(np.sum(even_weights), 4)
        self.assertEqual(np.sum(odd_weights), 5)

    def test_broken_backward_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()

            def bad_run_path(path: str, *, run_name: str):
                namespace = self.fake_run_path(path, run_name=run_name)
                original = namespace["backward"]

                def backward(runtime, textured, return_metadata=False):
                    frequencies, spectrum, metadata = original(
                        runtime, textured, return_metadata=True
                    )
                    if textured:
                        scale = np.asarray([1.0, 2.0, 4.0])
                        spectrum = spectrum * scale
                        metadata = dict(metadata)
                        metadata["raw_power"] = (
                            metadata["raw_power"] * scale[:, np.newaxis]
                        )
                    if return_metadata:
                        return frequencies, spectrum, metadata
                    return frequencies, spectrum

                namespace["backward"] = backward
                return namespace

            with self.assertRaisesRegex(RuntimeError, "reciprocity"):
                RECIPROCITY.run_reciprocity_matrix(
                    self.make_args(example), run_path=bad_run_path
                )

    def test_target_globals_are_restored_after_failure(self) -> None:
        captured = {}

        def failing_run_path(_path: str, *, run_name: str):
            namespace = self.fake_run_path(_path, run_name=run_name)
            captured.update(namespace)
            exec(
                """
def forward(_n, _runtime, _textured):
    raise RuntimeError("synthetic forward failure")
""",
                namespace,
            )
            captured["forward"] = namespace["forward"]
            captured["backward"] = namespace["backward"]
            return namespace

        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
                RECIPROCITY.run_reciprocity_matrix(
                    self.make_args(example), run_path=failing_run_path
                )
        for function in (captured["forward"], captured["backward"]):
            self.assertEqual(function.__globals__["resolution"], 99)
            self.assertEqual(function.__globals__["nfreq"], 99)
            self.assertEqual(function.__globals__["ndipole"], 99)


class StochasticReciprocityResolutionProbeTests(unittest.TestCase):
    @staticmethod
    def make_run_path(
        *,
        closure_shift: float = 0.0,
        metadata_fault: str | None = None,
        calls: list[tuple[str, int, int, bool]] | None = None,
    ):
        recorded_calls = calls if calls is not None else []

        def fake_run_path(_path: str, *, run_name: str):
            assert run_name == "gpmeep_reciprocity_target"
            namespace = {
                "np": np,
                "sx": 1.0,
                "resolution": 99,
                "nfreq": 99,
                "ndipole": 99,
                "closure_shift": closure_shift,
                "metadata_fault": metadata_fault,
                "recorded_calls": recorded_calls,
            }
            exec(
                """
def runtime_factor(runtime):
    return 1.0 - 0.16 / runtime

def forward(n, runtime, textured):
    recorded_calls.append(("forward", n, runtime, textured))
    frequencies = np.linspace(0.9, 1.1, nfreq)
    ratio = 2.0 if textured else 1.0
    spectrum = runtime_factor(runtime) * ratio * np.linspace(1.0, 1.5, nfreq)
    return frequencies, spectrum

def backward(runtime, textured, return_metadata=False):
    recorded_calls.append(("backward", -1, runtime, textured))
    frequencies = np.linspace(0.9, 1.1, nfreq)
    ratio = 2.0 * (1.0 + closure_shift) if textured else 1.0
    local_power = runtime_factor(runtime) * ratio * np.linspace(1.0, 1.5, nfreq)
    spacing = sx / ndipole
    x = -0.5 * sx - spacing + spacing * np.arange(ndipole + 3)
    weights = np.zeros(ndipole + 3)
    weights[1] = 0.5 * spacing
    weights[-2] = 0.5 * spacing
    weights[2:-2] = spacing
    raw_power = np.repeat(local_power[:, np.newaxis], ndipole + 3, axis=1)
    raw_power[:, 0] = 1e12
    raw_power[:, -1] = 2e12
    if metadata_fault == "position":
        x[2] += 1e-3
    elif metadata_fault == "ghost":
        weights[0] = 1e-8
    elif metadata_fault == "canonical":
        weights[2] += 1e-7
    spectrum = np.sum(raw_power * weights[np.newaxis, :], axis=1)
    if return_metadata:
        return frequencies, spectrum, {
            "x": x,
            "weights": weights,
            "raw_power": raw_power,
        }
    return frequencies, spectrum
""",
                namespace,
            )
            return namespace

        return fake_run_path

    @staticmethod
    def make_args(
        example: pathlib.Path,
        *,
        reference: float = 0.01,
        position: float = -0.3,
    ):
        return RESOLUTION_PROBE.parse_args(
            [
                str(example),
                "--resolution",
                "5",
                "--frequencies",
                "3",
                "--position",
                str(position),
                "--reference-res50-relative-l2",
                str(reference),
            ]
        )

    def test_exact_eight_run_local_reciprocity_contract(self) -> None:
        calls: list[tuple[str, int, int, bool]] = []
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            result = RESOLUTION_PROBE.run_resolution_probe(
                self.make_args(example),
                run_path=self.make_run_path(calls=calls),
            )

        self.assertEqual(
            calls,
            [
                ("forward", 1, 32, False),
                ("forward", 1, 128, True),
                ("backward", -1, 32, False),
                ("backward", -1, 128, True),
                ("forward", 1, 64, False),
                ("forward", 1, 256, True),
                ("backward", -1, 64, False),
                ("backward", -1, 256, True),
            ],
        )
        np.testing.assert_array_equal(
            result["reciprocity_resolution_probe_resolution"], [5]
        )
        np.testing.assert_array_equal(
            result["reciprocity_resolution_probe_dipole_count"], [5]
        )
        np.testing.assert_array_equal(
            result["reciprocity_resolution_probe_source_index"], [1]
        )
        np.testing.assert_array_equal(
            result["reciprocity_resolution_probe_metadata_index"],
            [2, 2, 2, 2],
        )
        self.assertEqual(
            result["reciprocity_resolution_probe_forward_flat_flux"].shape,
            (2, 3),
        )
        self.assertEqual(
            result[
                "reciprocity_resolution_probe_backward_textured_local_power"
            ].shape,
            (2, 3),
        )
        self.assertEqual(
            result["reciprocity_resolution_probe_raw_time_pointwise"].shape,
            (4, 3),
        )
        self.assertEqual(
            result[
                "reciprocity_resolution_probe_normalized_time_scale_aware"
            ].shape,
            (2, 3),
        )
        np.testing.assert_allclose(
            result["reciprocity_resolution_probe_metadata_position_error"],
            np.zeros(4),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_array_equal(
            result["reciprocity_resolution_probe_metadata_ghost_fraction"],
            np.zeros((4, 3)),
        )
        self.assertEqual(
            result[
                "reciprocity_resolution_probe_doubled_closure_relative_l2"
            ][0],
            0.0,
        )

    def test_two_percent_closure_and_point_seven_refinement_pass(self) -> None:
        closure_shift = 0.02
        expected_closure = closure_shift / (1.0 + closure_shift)
        reference = expected_closure / 0.7
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            result = RESOLUTION_PROBE.run_resolution_probe(
                self.make_args(example, reference=reference),
                run_path=self.make_run_path(closure_shift=closure_shift),
            )

        self.assertAlmostEqual(
            result[
                "reciprocity_resolution_probe_doubled_closure_relative_l2"
            ][0],
            expected_closure,
        )
        self.assertAlmostEqual(
            result["reciprocity_resolution_probe_refinement_ratio"][0],
            0.7,
        )

    def test_closure_and_refinement_fail_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            with self.assertRaisesRegex(RuntimeError, "3% gate"):
                RESOLUTION_PROBE.run_resolution_probe(
                    self.make_args(example, reference=0.1),
                    run_path=self.make_run_path(closure_shift=0.04),
                )
            with self.assertRaisesRegex(RuntimeError, "did not improve"):
                RESOLUTION_PROBE.run_resolution_probe(
                    self.make_args(example, reference=0.021),
                    run_path=self.make_run_path(closure_shift=0.02),
                )

    def test_failure_emits_complete_diagnostics_before_gate(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "3% gate"):
                    RESOLUTION_PROBE.run_resolution_probe(
                        self.make_args(example, reference=0.1),
                        run_path=self.make_run_path(closure_shift=0.04),
                    )

        lines = [
            line
            for line in output.getvalue().splitlines()
            if line.startswith(RESOLUTION_PROBE.DIAGNOSTICS_PREFIX)
        ]
        self.assertEqual(len(lines), 1)
        diagnostics = json.loads(
            lines[0][len(RESOLUTION_PROBE.DIAGNOSTICS_PREFIX) :]
        )
        self.assertEqual(
            diagnostics["schema"],
            "gpmeep-reciprocity-resolution-diagnostics-v1",
        )
        self.assertEqual(len(diagnostics["forward_flat"]), 2)
        self.assertEqual(len(diagnostics["forward_flat"][0]), 3)
        self.assertEqual(len(diagnostics["normalized_time_relative_l2"]), 2)
        self.assertGreater(diagnostics["doubled_closure_relative_l2"], 0.03)

    def test_bad_position_ghost_and_canonical_metadata_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            for fault, message in (
                ("position", "periodic x grid"),
                ("ghost", "ghost cubature weights"),
                ("canonical", "canonical DFT cubature"),
            ):
                with self.subTest(fault=fault):
                    with self.assertRaisesRegex(RuntimeError, message):
                        RESOLUTION_PROBE.run_resolution_probe(
                            self.make_args(example),
                            run_path=self.make_run_path(
                                metadata_fault=fault
                            ),
                        )

    def test_off_grid_position_is_rejected_before_simulation(self) -> None:
        calls: list[tuple[str, int, int, bool]] = []
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            with self.assertRaisesRegex(RuntimeError, "exactly shared"):
                RESOLUTION_PROBE.run_resolution_probe(
                    self.make_args(example, position=-0.2),
                    run_path=self.make_run_path(calls=calls),
                )
        self.assertEqual(calls, [])

    def test_even_pixel_period_is_rejected_before_simulation(self) -> None:
        calls: list[tuple[str, int, int, bool]] = []
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            args = RESOLUTION_PROBE.parse_args(
                [
                    str(example),
                    "--resolution",
                    "4",
                    "--frequencies",
                    "3",
                    "--position",
                    "-0.25",
                    "--reference-res50-relative-l2",
                    "0.01",
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "odd number of pixels"):
                RESOLUTION_PROBE.run_resolution_probe(
                    args,
                    run_path=self.make_run_path(calls=calls),
                )
        self.assertEqual(calls, [])

    def test_launcher_pins_exact_eight_run_oracle_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            argv = RESOLUTION_LAUNCHER.build_oracle_argv(
                example, 0.0335799799220571
            )
            args = EXAMPLE_ORACLE.parse_args(argv)

        self.assertEqual(args.expected_run_count, 8)
        self.assertEqual(RESOLUTION_LAUNCHER.RESOLUTION, 90)
        self.assertEqual(round(1.1 * RESOLUTION_LAUNCHER.RESOLUTION) % 2, 1)
        self.assertEqual(args.expected_final_timestep, 930600)
        self.assertIsNone(args.expected_total_timestep_delta)
        self.assertEqual(
            args.expected_run_timestep_delta_range,
            [(value, value) for value in RESOLUTION_LAUNCHER.EXPECTED_TIMESTEPS],
        )
        self.assertEqual(
            set(args.result_vector),
            set(RESOLUTION_LAUNCHER.RESULT_SHAPES),
        )
        self.assertEqual(
            dict(args.expected_result_shape),
            {
                name: tuple(int(item) for item in shape.split(","))
                for name, shape in RESOLUTION_LAUNCHER.RESULT_SHAPES.items()
            },
        )
        self.assertEqual(
            dict(args.max_result_abs), RESOLUTION_LAUNCHER.MAXIMUMS
        )
        self.assertNotIn(
            "reciprocity_resolution_probe_raw_time_pointwise",
            dict(args.max_result_abs),
        )
        self.assertNotIn(
            "reciprocity_resolution_probe_doubled_closure_pointwise",
            dict(args.max_result_abs),
        )
        self.assertEqual(
            args.example.name, "stochastic_reciprocity_resolution_probe.py"
        )
        self.assertIn("--reference-res50-relative-l2", args.example_args)

    def test_launcher_recomputes_reference_and_rejects_tampering(self) -> None:
        frequencies = np.linspace(0.9, 1.1, 4)
        forward_flat = np.asarray([1.0, 1.1, 1.2, 1.3])
        forward = np.asarray([2.0, 2.1, 2.2, 2.3])
        backward = np.asarray([2.02, 2.08, 2.24, 2.31])
        forward_unique = np.ones((4, 28))
        forward_unique[:, 15] = forward * forward_flat
        forward_summed = np.ones((2, 4))
        forward_summed[0] = 55 * forward_flat
        backward_raw = np.ones((4, 4, 58))
        backward_raw[2, :, 16] = 1.0
        backward_raw[3, :, 16] = backward
        spacing = 1.1 / 55
        metadata_x = np.repeat(
            (-0.55 - spacing + spacing * np.arange(58))[np.newaxis, :],
            4,
            axis=0,
        )

        def metric(array: np.ndarray) -> dict[str, object]:
            value = np.asarray(array, dtype=float)
            return {
                "shape": list(value.shape),
                "real": value.reshape(-1).tolist(),
            }

        payload = {
            "example_results": {
                "reciprocity_frequencies": metric(frequencies),
                "reciprocity_forward_textured_unique_doubled_flux": metric(
                    forward_unique
                ),
                "reciprocity_forward_doubled_summed_flux": metric(
                    forward_summed
                ),
                "reciprocity_backward_metadata_raw_power": metric(
                    backward_raw
                ),
                "reciprocity_backward_metadata_x": metric(metadata_x),
            }
        }
        residual = forward - backward
        relative_l2 = float(
            np.linalg.norm(residual) / np.linalg.norm(backward)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "cuda.log"
            source.write_text(
                RESOLUTION_LAUNCHER.METRICS_PREFIX
                + json.dumps(payload, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            reference = {
                "schema": RESOLUTION_LAUNCHER.REFERENCE_SCHEMA,
                "source": {
                    "path": str(source),
                    "sha256": RESOLUTION_LAUNCHER.sha256_file(source),
                    "backend": "cuda",
                },
                "resolution": 50,
                "position": -0.25,
                "dipole_count": 55,
                "source_index": 15,
                "metadata_index": 16,
                "forward_normalized_doubled": forward.tolist(),
                "backward_normalized_doubled": backward.tolist(),
                "residual": residual.tolist(),
                "relative_l2": relative_l2,
            }
            reference_path = root / "reference.json"
            reference_path.write_text(
                json.dumps(reference), encoding="utf-8"
            )

            actual, audit = RESOLUTION_LAUNCHER.load_reference(
                reference_path
            )
            self.assertAlmostEqual(actual, relative_l2)
            self.assertEqual(audit["source_sha256"], reference["source"]["sha256"])

            tampered_reference = dict(reference)
            tampered_reference["relative_l2"] = relative_l2 + 1e-3
            reference_path.write_text(
                json.dumps(tampered_reference), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "does not recompute"):
                RESOLUTION_LAUNCHER.load_reference(reference_path)

            reference_path.write_text(
                json.dumps(reference), encoding="utf-8"
            )
            source.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash does not match"):
                RESOLUTION_LAUNCHER.load_reference(reference_path)


class StochasticReciprocityTemporalExtensionTests(unittest.TestCase):
    @staticmethod
    def reference_diagnostics(*, failing_prior: bool = False):
        frequencies = np.linspace(0.9, 1.1, 4)
        forward_flat = np.asarray([1.0, 2.0, 3.0, 4.0])
        backward_flat = np.asarray([1.1, 2.2, 3.3, 4.4])
        prior_ratio = 1.8 if failing_prior else 1.99
        forward_textured = np.stack(
            (prior_ratio * forward_flat, prior_ratio * forward_flat)
        )
        backward_textured = np.stack(
            (prior_ratio * backward_flat, prior_ratio * backward_flat)
        )
        if failing_prior:
            forward_textured[1] = 2.0 * forward_flat
            backward_textured[1] = 2.0 * backward_flat
        forward_flat_rows = np.stack((forward_flat, forward_flat))
        backward_flat_rows = np.stack((backward_flat, backward_flat))
        raw_base = np.stack(
            (
                forward_flat_rows[0],
                forward_textured[0],
                backward_flat_rows[0],
                backward_textured[0],
            )
        )
        raw_doubled = np.stack(
            (
                forward_flat_rows[1],
                forward_textured[1],
                backward_flat_rows[1],
                backward_textured[1],
            )
        )
        raw_residual = raw_base - raw_doubled
        raw_relative_l2 = np.asarray(
            [
                np.linalg.norm(raw_residual[row])
                / np.linalg.norm(raw_doubled[row])
                for row in range(4)
            ]
        )
        _, raw_scale_aware = RECIPROCITY._pointwise_relative_errors(
            raw_residual, raw_doubled, axis=1
        )
        forward_normalized = forward_textured / forward_flat_rows
        backward_normalized = backward_textured / backward_flat_rows
        normalized_residual = np.stack(
            (
                forward_normalized[0] - forward_normalized[1],
                backward_normalized[0] - backward_normalized[1],
            )
        )
        normalized_reference = np.stack(
            (forward_normalized[1], backward_normalized[1])
        )
        normalized_relative_l2 = np.asarray(
            [
                np.linalg.norm(normalized_residual[row])
                / np.linalg.norm(normalized_reference[row])
                for row in range(2)
            ]
        )
        _, normalized_scale_aware = RECIPROCITY._pointwise_relative_errors(
            normalized_residual, normalized_reference, axis=1
        )
        closure = forward_normalized[1] - backward_normalized[1]
        closure_l2 = np.linalg.norm(closure) / np.linalg.norm(
            backward_normalized[1]
        )
        _, closure_scale_aware = RECIPROCITY._pointwise_relative_errors(
            closure, backward_normalized[1]
        )
        return {
            "schema": "gpmeep-reciprocity-resolution-diagnostics-v1",
            "resolution": 90,
            "dipole_count": 99,
            "position": -0.25,
            "source_index": 27,
            "frequencies": frequencies.tolist(),
            "forward_flat": forward_flat_rows.tolist(),
            "forward_textured": forward_textured.tolist(),
            "backward_flat": backward_flat_rows.tolist(),
            "backward_textured": backward_textured.tolist(),
            "raw_time_relative_l2": raw_relative_l2.tolist(),
            "raw_time_scale_aware_max": float(np.max(raw_scale_aware)),
            "forward_normalized": forward_normalized.tolist(),
            "backward_normalized": backward_normalized.tolist(),
            "normalized_time_relative_l2": normalized_relative_l2.tolist(),
            "normalized_time_scale_aware_max": float(
                np.max(normalized_scale_aware)
            ),
            "base_closure_relative_l2": 0.0,
            "doubled_closure_relative_l2": float(closure_l2),
            "doubled_closure_scale_aware_max": float(
                np.max(closure_scale_aware)
            ),
            "reference_res50_relative_l2": 0.04,
            "refinement_ratio": float(closure_l2 / 0.04),
        }

    @staticmethod
    def fake_run_path(calls):
        def run_path(_path: str, *, run_name: str):
            assert run_name == "gpmeep_reciprocity_target"
            namespace = {
                "np": np,
                "sx": 1.1,
                "sy": 9.2,
                "dAg": 0.5,
                "dsub": 5.0,
                "resolution": 999,
                "nfreq": 999,
                "ndipole": 999,
                "calls": calls,
            }
            exec(
                """
def forward(n, runtime, textured):
    calls.append(("forward", n, runtime, textured))
    frequencies = np.linspace(0.9, 1.1, nfreq)
    flat = np.asarray([1.0, 2.0, 3.0, 4.0])
    return frequencies, 2.0 * flat

def backward(runtime, textured, return_metadata=False):
    calls.append(("backward", -1, runtime, textured))
    frequencies = np.linspace(0.9, 1.1, nfreq)
    flat = np.asarray([1.1, 2.2, 3.3, 4.4])
    local_power = 2.0 * flat
    spacing = sx / ndipole
    x = -0.5 * sx - spacing + spacing * np.arange(ndipole + 3)
    weights = np.zeros(ndipole + 3)
    weights[1] = 0.5 * spacing
    weights[-2] = 0.5 * spacing
    weights[2:-2] = spacing
    raw_power = np.repeat(local_power[:, np.newaxis], ndipole + 3, axis=1)
    raw_power[:, 0] = 1e12
    raw_power[:, -1] = 2e12
    spectrum = np.sum(raw_power * weights[np.newaxis, :], axis=1)
    metadata = {
        "x": x,
        "y": np.asarray([-1.6]),
        "z": np.asarray([0.0]),
        "weights": weights,
        "raw_power": raw_power,
    }
    if return_metadata:
        return frequencies, spectrum, metadata
    return frequencies, spectrum
""",
                namespace,
            )
            return namespace

        return run_path

    def make_args(self, example: pathlib.Path, reference=None):
        payload = reference or self.reference_diagnostics()
        return TEMPORAL_PROBE.parse_args(
            [
                str(example),
                "--reference-diagnostics-json",
                json.dumps(payload, separators=(",", ":")),
            ]
        )

    def test_two_run_extension_contract_and_coordinate_metadata(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            result = TEMPORAL_PROBE.run_temporal_extension(
                self.make_args(example), run_path=self.fake_run_path(calls)
            )

        self.assertEqual(
            calls,
            [
                ("forward", 27, 512, True),
                ("backward", -1, 512, True),
            ],
        )
        np.testing.assert_array_equal(
            result["reciprocity_temporal_extension_source_index"], [27]
        )
        np.testing.assert_array_equal(
            result["reciprocity_temporal_extension_metadata_index"], [28]
        )
        np.testing.assert_allclose(
            result["reciprocity_temporal_extension_metadata_y_error"],
            [0.0],
            rtol=0.0,
            atol=1e-15,
        )
        self.assertLess(
            np.max(
                result[
                    "reciprocity_temporal_extension_normalized_time_relative_l2"
                ]
            ),
            0.02,
        )
        self.assertEqual(
            result["reciprocity_temporal_extension_closure_relative_l2"][0],
            0.0,
        )

    def test_wrong_y_metadata_is_rejected(self) -> None:
        calls = []

        def bad_run_path(path: str, *, run_name: str):
            namespace = self.fake_run_path(calls)(path, run_name=run_name)
            original = namespace["backward"]

            def backward(runtime, textured, return_metadata=False):
                frequencies, spectrum, metadata = original(
                    runtime, textured, return_metadata=True
                )
                metadata = dict(metadata)
                metadata["y"] = np.asarray([-1.5])
                if return_metadata:
                    return frequencies, spectrum, metadata
                return frequencies, spectrum

            namespace["backward"] = backward
            return namespace

        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            with self.assertRaisesRegex(RuntimeError, "y metadata"):
                TEMPORAL_PROBE.run_temporal_extension(
                    self.make_args(example), run_path=bad_run_path
                )

    def test_launcher_pins_exact_two_run_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            argv = TEMPORAL_LAUNCHER.build_oracle_argv(
                example, self.reference_diagnostics()
            )
            args = EXAMPLE_ORACLE.parse_args(argv)

        self.assertEqual(args.expected_run_count, 2)
        self.assertEqual(args.expected_final_timestep, 1_852_200)
        self.assertEqual(
            args.expected_run_timestep_delta_range,
            [(1_852_200, 1_852_200), (1_852_200, 1_852_200)],
        )
        self.assertEqual(set(args.result_vector), set(TEMPORAL_LAUNCHER.RESULT_SHAPES))
        self.assertEqual(
            dict(args.max_result_abs), TEMPORAL_LAUNCHER.MAXIMUMS
        )
        self.assertEqual(
            args.example.name,
            "stochastic_reciprocity_temporal_extension_probe.py",
        )

    def test_reference_loader_recomputes_failure_and_rejects_tampering(self) -> None:
        diagnostics = self.reference_diagnostics(failing_prior=True)
        required_inputs = {
            name: "a" * 64
            for name in (
                "python/examples/stochastic_emitter_reciprocity.py",
                "scripts/python-validation/stochastic_reciprocity_resolution_probe.py",
                "scripts/python-validation/run_stochastic_reciprocity_resolution90.py",
                "scripts/python-validation/run_example_oracle.py",
                "build/meep-cuda-mpi-python-fp32/src/.libs/libmeep.so.38.0.0",
                "build/meep-cuda-mpi-python-fp32/python/meep/_meep.so",
            )
        }
        freeze = {
            "schema": "gpmeep-development-source-freeze-v1",
            "git_head": "b65efaa749ec04d76f469db0a408c22e003468b3",
            "release_eligible": False,
            "inputs": required_inputs,
            "profile": {
                "resolution": 90,
                "period_pixels": 99,
                "position": -0.25,
                "source_index": 27,
                "expected_metadata_index": 28,
                "runs": 8,
                "expected_timestep_deltas": list(
                    TEMPORAL_LAUNCHER.V3_EXPECTED_TIMESTEPS
                ),
            },
        }
        command = (
            'Script started [COMMAND="CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 '
            "MEEP_GPU_BACKEND=cuda GPMEEP_VALIDATION_STRICT_CUDA=1 "
            'run_stochastic_reciprocity_resolution90.py"]'
        )
        run_lines = [
            f"run 0 finished at t = 1.0 ({value} timesteps)"
            for value in TEMPORAL_LAUNCHER.V3_EXPECTED_TIMESTEPS
        ]
        log_lines = [
            command,
            "Using MPI version 3.1, 1 processes",
            *run_lines,
            TEMPORAL_LAUNCHER.DIAGNOSTICS_PREFIX
            + json.dumps(diagnostics, separators=(",", ":")),
            "RuntimeError: resolution probe normalized ratios did not converge",
            'Script done [COMMAND_EXIT_CODE="1"]',
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            log_path = root / "reference.log"
            freeze_path = root / "freeze.json"
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            log_sha = TEMPORAL_LAUNCHER.sha256_file(log_path)
            freeze_sha = TEMPORAL_LAUNCHER.sha256_file(freeze_path)
            with mock.patch.object(
                TEMPORAL_LAUNCHER, "V3_LOG_SHA256", log_sha
            ), mock.patch.object(
                TEMPORAL_LAUNCHER, "V3_FREEZE_SHA256", freeze_sha
            ):
                loaded, audit = TEMPORAL_LAUNCHER.load_reference(
                    log_path, freeze_path
                )
                self.assertEqual(loaded, diagnostics)
                self.assertEqual(audit["runs"], 8)
                self.assertGreater(
                    audit[
                        "reference_failure_normalized_relative_l2_max"
                    ],
                    0.02,
                )
                log_path.write_text("tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "log hash"):
                    TEMPORAL_LAUNCHER.load_reference(log_path, freeze_path)


class StochasticReciprocityStabilityPointTests(unittest.TestCase):
    def test_forward_and_backward_single_run_contracts(self) -> None:
        reference = StochasticReciprocityTemporalExtensionTests.reference_diagnostics()
        for branch in ("forward", "backward"):
            calls = []
            with self.subTest(branch=branch), tempfile.TemporaryDirectory() as directory:
                example = pathlib.Path(directory) / "reciprocity.py"
                example.touch()
                args = STABILITY_POINT.parse_args(
                    [
                        str(example),
                        "--branch",
                        branch,
                        "--runtime",
                        "384",
                        "--reference-diagnostics-json",
                        json.dumps(reference, separators=(",", ":")),
                    ]
                )
                result = STABILITY_POINT.run_stability_point(
                    args,
                    run_path=(
                        StochasticReciprocityTemporalExtensionTests.fake_run_path(
                            calls
                        )
                    ),
                )

            expected = (
                [("forward", 27, 384, True)]
                if branch == "forward"
                else [("backward", -1, 384, True)]
            )
            self.assertEqual(calls, expected)
            self.assertEqual(
                result["reciprocity_stability_spectrum"].shape, (4,)
            )
            self.assertLess(
                result["reciprocity_stability_relative_l2"][0], 0.02
            )
            np.testing.assert_allclose(
                result[
                    "reciprocity_stability_metadata_coordinate_errors"
                ],
                np.zeros(3),
                rtol=0.0,
                atol=1e-15,
            )

    def test_launcher_timestep_and_result_contract(self) -> None:
        reference = StochasticReciprocityTemporalExtensionTests.reference_diagnostics()
        self.assertEqual(
            STABILITY_LAUNCHER.expected_timestep(384, 0.5), 1_391_400
        )
        self.assertEqual(
            STABILITY_LAUNCHER.expected_timestep(384, 0.4), 1_739_250
        )
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            argv = STABILITY_LAUNCHER.build_oracle_argv(
                example, "forward", 384, 0.5, reference
            )
            args = EXAMPLE_ORACLE.parse_args(argv)

        self.assertEqual(args.expected_run_count, 1)
        self.assertEqual(args.expected_final_timestep, 1_391_400)
        self.assertEqual(
            args.expected_run_timestep_delta_range,
            [(1_391_400, 1_391_400)],
        )
        self.assertEqual(set(args.result_vector), set(STABILITY_LAUNCHER.RESULT_SHAPES))


class StochasticReciprocityStabilityTraceTests(unittest.TestCase):
    @staticmethod
    def fake_run_path(state):
        class Vector3:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = float(x)
                self.y = float(y)
                self.z = float(z)

        class FakeGpu:
            active_backend = "cpu"
            requested_backend = "cpu"
            selected_device = -1
            selected_device_identifier = "none"

            def __init__(self):
                self.cpu_calls = 0

            def statistics(self):
                return {
                    "dispatch": {
                        "cpu_curl_calls": self.cpu_calls,
                        "cuda_curl_calls": 0,
                    }
                }

        gpu = FakeGpu()

        class FakeFields:
            def __init__(self, resolution, courant):
                self.t = 0
                self.dt = courant / resolution

            def last_source_time(self):
                return 50.0

            def dft_norm(self):
                return 1.0 + self.t / 1_000_000

            def gpu_execution_diagnostic(self):
                return "fake CPU trace"

        class FakeMode:
            def __init__(self, frequencies):
                self.alpha = np.zeros((1, frequencies, 2), dtype=complex)
                self.alpha[0, :, 0] = np.arange(1, frequencies + 1)

        class FakeMonitor:
            def __init__(self, simulation):
                self.simulation = simulation

        class FakeSimulation:
            def __init__(self, resolution, courant, frequencies):
                self.resolution = resolution
                self.courant = courant
                self.frequencies = frequencies
                self.fields = FakeFields(resolution, courant)
                self.monitor = FakeMonitor(self)
                self.run_arguments = []
                self.reset_count = 0
                self.init_count = 0
                self.get_array_allocation_count = 0
                self.get_array_reuse_count = 0
                self.signature_bias = 0.0
                state.setdefault("simulations", []).append(self)

            def init_sim(self):
                self.init_count += 1

            def run(self, *, until_after_sources):
                self.run_arguments.append(until_after_sources)
                target = int(
                    round(
                        (50.0 + until_after_sources)
                        * self.resolution
                        / self.courant
                    )
                )
                gpu.cpu_calls += target - self.fields.t
                self.fields.t = target

            def round_time(self):
                return self.fields.t * self.fields.dt

            def get_array_metadata(self, *, center, size):
                del center, size
                x = np.asarray([-0.4, 0.0, 0.4])
                y = np.asarray([-4.5, -2.0, 1.2, 2.5, 4.0])
                z = np.asarray([0.0])
                weights = np.ones((x.size, y.size))
                return x, y, z, weights

            def get_array(self, *, component, center, size, arr=None):
                del center, size
                if state.get("measurement_backend_call"):
                    gpu.cpu_calls += 1
                values = np.full(
                    (3, 5),
                    (component + 1) * (1.0 + self.fields.t / 1_000_000)
                    + self.signature_bias,
                    dtype=np.float32,
                )
                if arr is not None:
                    arr[...] = values
                    self.get_array_reuse_count += 1
                    if state.get("perturb_on_reuse"):
                        self.signature_bias = 0.125
                    return arr
                self.get_array_allocation_count += 1
                return values

            def get_dft_array(self, monitor, component, frequency_index):
                del monitor
                return np.full(
                    (3,),
                    (component + 1)
                    * (frequency_index + 1)
                    * (1.0 + self.fields.t / 1_000_000)
                    + self.signature_bias,
                    dtype=np.complex64,
                )

            def get_eigenmode_coefficients(self, monitor, bands, eig_parity):
                self.last_mode_arguments = (monitor, bands, eig_parity)
                return FakeMode(self.frequencies)

            def reset_meep(self):
                self.reset_count += 1

        class FakeMeep:
            Ez = 0
            Dz = 1
            Hx = 2
            Hy = 3
            ODD_Z = 2

            @staticmethod
            def get_fluxes(monitor):
                scale = 1.0 + monitor.simulation.fields.t / 1_000_000
                return scale * np.arange(1, monitor.simulation.frequencies + 1)

            @staticmethod
            def get_flux_freqs(monitor):
                return np.linspace(0.9, 1.1, monitor.simulation.frequencies)

            @staticmethod
            def is_single_precision():
                return True

            @staticmethod
            def am_master():
                return False

        FakeMeep.Vector3 = Vector3
        FakeMeep.gpu = gpu
        fake_mp = FakeMeep()
        state["mp"] = fake_mp
        namespace = {
            "FakeSimulation": FakeSimulation,
            "mp": fake_mp,
            "resolution": 200,
            "courant": 0.5,
            "nfreq": 100,
            "ndipole": 10,
            "sx": 1.1,
            "sy": 9.2,
            "df": 0.2,
            "dAg": 0.5,
            "dsub": 5.0,
            "dpml": 1.0,
            "hrod": 0.7,
            "wrod": 0.5,
        }
        exec(
            "def _forward_problem(n, is_textured):\n"
            "    assert n == 27 and is_textured\n"
            "    sim = FakeSimulation(resolution, courant, nfreq)\n"
            "    return sim, sim.monitor\n",
            namespace,
        )

        def run_path(_path, *, run_name):
            assert run_name in (
                "gpmeep_reciprocity_trace_target",
                "gpmeep_reciprocity_neutrality_target",
            )
            return namespace

        return run_path

    @staticmethod
    def fake_identity(_mp, label, simulation, _validation_sources):
        document = {
            "schema": "gpmeep-execution-identity-v2",
            "label": label,
            "records": [
                {
                    "rank": 0,
                    "world_size": 1,
                    "simulation": {
                        "initial_timestep": int(simulation.fields.t)
                    },
                }
            ],
        }
        document["identity_sha256"] = EXECUTION_IDENTITY.canonical_sha256(
            document
        )
        return document

    def test_segmented_trace_is_continuous_and_resets_once(self) -> None:
        state = {}
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            args = STABILITY_TRACE.parse_args([str(example)])
            result = STABILITY_TRACE.run_stability_trace(
                args,
                run_path=self.fake_run_path(state),
                identity_capture=self.fake_identity,
            )

        expected_timestep = np.asarray(
            [469800, 700200, 930600, 1161000, 1391400, 1621800, 1852200]
        )
        np.testing.assert_array_equal(
            result["reciprocity_trace_checkpoint_timestep"],
            expected_timestep,
        )
        np.testing.assert_array_equal(
            result["reciprocity_trace_timestep_delta"],
            np.diff(np.concatenate(([0], expected_timestep))),
        )
        self.assertEqual(result["reciprocity_trace_field_max"].shape, (7, 5, 5))
        self.assertEqual(
            result["reciprocity_trace_field_argmax_xy"].shape, (7, 5, 2)
        )
        self.assertFalse(
            np.any(result["reciprocity_trace_field_nonfinite_count"])
        )
        self.assertTrue(
            np.all(result["reciprocity_trace_run_phase_calls"][:, 0] > 0)
        )
        self.assertFalse(
            np.any(result["reciprocity_trace_run_phase_calls"][:, 1])
        )
        np.testing.assert_array_equal(
            result["reciprocity_trace_endpoint_spectrum"],
            np.asarray([1.0, 4.0, 9.0, 16.0]),
        )
        simulation = state["simulations"][0]
        self.assertEqual(len(state["simulations"]), 1)
        self.assertEqual(simulation.init_count, 1)
        self.assertEqual(simulation.reset_count, 1)
        self.assertEqual(simulation.get_array_allocation_count, 4)
        self.assertEqual(simulation.get_array_reuse_count, 24)
        np.testing.assert_array_equal(
            result["reciprocity_trace_field_buffer_allocations"], [4.0]
        )
        np.testing.assert_array_equal(
            result["reciprocity_trace_field_buffer_reuses"], [24.0]
        )
        np.testing.assert_array_equal(
            result["reciprocity_trace_region_weight"],
            [15.0, 3.0, 4.0, 3.0, 5.0],
        )
        np.testing.assert_allclose(
            simulation.run_arguments,
            np.asarray(STABILITY_TRACE.DEFAULT_CHECKPOINTS) * 20.0,
        )

    def test_checkpoint_contract_rejects_missing_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            with self.assertRaises(SystemExit):
                STABILITY_TRACE.parse_args(
                    [str(example), "--checkpoints", "128,256", "--runtime", "512"]
                )

    def test_grid_partition_rejects_negative_weights(self) -> None:
        state = {}
        run_path = self.fake_run_path(state)
        namespace = run_path("unused", run_name="gpmeep_reciprocity_trace_target")
        simulation, _monitor = namespace["_forward_problem"](27, True)
        original = simulation.get_array_metadata

        def negative_metadata(*, center, size):
            x, y, z, weights = original(center=center, size=size)
            weights = weights.copy()
            weights[0, 0] = -1.0
            return x, y, z, weights

        simulation.get_array_metadata = negative_metadata
        with self.assertRaisesRegex(RuntimeError, "metadata differ"):
            STABILITY_TRACE._grid_partition(
                simulation, state["mp"], namespace
            )

    def test_trace_rejects_measurement_backend_calls(self) -> None:
        state = {"measurement_backend_call": True}
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            args = STABILITY_TRACE.parse_args(
                [str(example), "--runtime", "2", "--checkpoints", "1,2"]
            )
            with self.assertRaisesRegex(RuntimeError, "getters invoked"):
                STABILITY_TRACE.run_stability_trace(
                    args,
                    run_path=self.fake_run_path(state),
                    identity_capture=self.fake_identity,
                )
        self.assertEqual(state["simulations"][0].reset_count, 1)

    def test_strict_trace_rejects_cpu_backend(self) -> None:
        state = {}
        with tempfile.TemporaryDirectory() as directory:
            example = pathlib.Path(directory) / "reciprocity.py"
            example.touch()
            args = STABILITY_TRACE.parse_args(
                [str(example), "--runtime", "2", "--checkpoints", "1,2"]
            )
            with mock.patch.dict(
                STABILITY_TRACE.os.environ,
                {"GPMEEP_VALIDATION_STRICT_CUDA": "1"},
            ), self.assertRaisesRegex(RuntimeError, "all-CUDA FP32"):
                STABILITY_TRACE.run_stability_trace(
                    args,
                    run_path=self.fake_run_path(state),
                    identity_capture=self.fake_identity,
                )
        self.assertEqual(state["simulations"][0].reset_count, 1)


class StochasticReciprocityTraceNeutralityTests(unittest.TestCase):
    def _args(self):
        directory = tempfile.TemporaryDirectory()
        example = pathlib.Path(directory.name) / "reciprocity.py"
        example.touch()
        return directory, TRACE_NEUTRALITY.parse_args([str(example)])

    def test_two_observations_reuse_buffers_and_remain_bitwise_neutral(self) -> None:
        state = {}
        directory, args = self._args()
        try:
            with mock.patch.dict(
                TRACE_NEUTRALITY.os.environ,
                {"GPMEEP_VALIDATION_STRICT_CUDA": "0"},
            ):
                result = TRACE_NEUTRALITY.run_neutrality_probe(
                    args,
                    run_path=(
                        StochasticReciprocityStabilityTraceTests.fake_run_path(
                            state
                        )
                    ),
                    identity_capture=(
                        StochasticReciprocityStabilityTraceTests.fake_identity
                    ),
                )
        finally:
            directory.cleanup()

        np.testing.assert_array_equal(
            result["reciprocity_trace_neutrality_bitwise_equal"], [1.0]
        )
        np.testing.assert_array_equal(
            result["reciprocity_trace_neutrality_observations"],
            [2.0, 1.0, 4.0, 4.0],
        )
        self.assertEqual(len(state["simulations"]), 2)
        uninterrupted, segmented = state["simulations"]
        self.assertEqual(uninterrupted.run_arguments, [60.0])
        self.assertEqual(segmented.run_arguments, [20.0, 40.0, 60.0])
        self.assertEqual(uninterrupted.init_count, 1)
        self.assertEqual(segmented.init_count, 1)
        self.assertEqual(uninterrupted.reset_count, 1)
        self.assertEqual(segmented.reset_count, 1)
        self.assertEqual(segmented.get_array_reuse_count, 4)

    def test_strict_cuda_rejects_cpu_fallback(self) -> None:
        state = {}
        directory, args = self._args()
        try:
            with mock.patch.dict(
                TRACE_NEUTRALITY.os.environ,
                {"GPMEEP_VALIDATION_STRICT_CUDA": "1"},
            ), self.assertRaisesRegex(RuntimeError, "all-CUDA FP32"):
                TRACE_NEUTRALITY.run_neutrality_probe(
                    args,
                    run_path=(
                        StochasticReciprocityStabilityTraceTests.fake_run_path(
                            state
                        )
                    ),
                    identity_capture=(
                        StochasticReciprocityStabilityTraceTests.fake_identity
                    ),
                )
        finally:
            directory.cleanup()

    def test_neutrality_detects_observer_perturbation(self) -> None:
        state = {"perturb_on_reuse": True}
        directory, args = self._args()
        try:
            with mock.patch.dict(
                TRACE_NEUTRALITY.os.environ,
                {"GPMEEP_VALIDATION_STRICT_CUDA": "0"},
            ), self.assertRaisesRegex(RuntimeError, "perturbed final state"):
                TRACE_NEUTRALITY.run_neutrality_probe(
                    args,
                    run_path=(
                        StochasticReciprocityStabilityTraceTests.fake_run_path(
                            state
                        )
                    ),
                    identity_capture=(
                        StochasticReciprocityStabilityTraceTests.fake_identity
                    ),
                )
        finally:
            directory.cleanup()

    def test_bytewise_comparison_rejects_signed_zero(self) -> None:
        left = {"signed_zero": np.asarray([0.0], dtype=np.float32)}
        right = {"signed_zero": np.asarray([-0.0], dtype=np.float32)}
        self.assertTrue(np.array_equal(left["signed_zero"], right["signed_zero"]))
        unequal, left_manifest, right_manifest = (
            TRACE_NEUTRALITY._bytewise_signature_comparison(left, right)
        )
        self.assertEqual(unequal, ["signed_zero"])
        self.assertNotEqual(
            left_manifest["arrays"]["signed_zero"]["sha256"],
            right_manifest["arrays"]["signed_zero"]["sha256"],
        )
        self.assertNotEqual(
            TRACE_NEUTRALITY._float64_bit_record(0.0)["hex"],
            TRACE_NEUTRALITY._float64_bit_record(-0.0)["hex"],
        )


class ExecutionIdentityTests(unittest.TestCase):
    def test_mapping_verification_ignores_unrelated_deleted_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = root / "libmeep.so.test"
            target.write_bytes(b"mapped-library")
            record = EXECUTION_IDENTITY.file_record(target)
            resolved = target.resolve()
            mappings = {
                resolved: {
                    (
                        record["device_major"],
                        record["device_minor"],
                        record["inode"],
                    )
                }
            }
            verified = EXECUTION_IDENTITY.mapped_file_record(
                target, mappings, {root / "unrelated-deleted"}
            )
            self.assertTrue(verified["mapped_identity_verified"])
            with self.assertRaisesRegex(RuntimeError, "mapping was deleted"):
                EXECUTION_IDENTITY.mapped_file_record(
                    target, mappings, {resolved}
                )

    def test_identity_emitter_rejects_tampering(self) -> None:
        class FakeMeep:
            @staticmethod
            def am_master():
                return False

        identity = {"schema": "test", "records": []}
        identity["identity_sha256"] = EXECUTION_IDENTITY.canonical_sha256(
            identity
        )
        EXECUTION_IDENTITY.emit_execution_identity(identity, FakeMeep())
        identity["records"].append({"rank": 0})
        with self.assertRaisesRegex(RuntimeError, "self-hash"):
            EXECUTION_IDENTITY.emit_execution_identity(identity, FakeMeep())


class TargetLifecycleContractTests(unittest.TestCase):
    def test_target_simulations_reset_in_finally(self) -> None:
        repo = WRAPPER_DIR.parents[1]
        targets = {
            repo / "python/examples/stochastic_emitter_line.py": {"compute_flux"},
            repo / "python/examples/stochastic_emitter_reciprocity.py": {
                "forward",
                "backward",
            },
        }
        for path, function_names in targets.items():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            functions = {
                node.name: node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for name in function_names:
                with self.subTest(path=path.name, function=name):
                    function = functions[name]
                    reset_finally = [
                        child
                        for child in ast.walk(function)
                        if isinstance(child, ast.Try)
                        and any(
                            isinstance(call, ast.Call)
                            and isinstance(call.func, ast.Attribute)
                            and call.func.attr == "reset_meep"
                            for final_node in child.finalbody
                            for call in ast.walk(final_node)
                        )
                    ]
                    self.assertEqual(len(reset_finally), 1)

    def test_forward_delegates_to_single_shared_problem_factory(self) -> None:
        target = (
            WRAPPER_DIR.parents[1]
            / "python/examples/stochastic_emitter_reciprocity.py"
        )
        tree = ast.parse(target.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("_forward_problem", functions)
        forward = functions["forward"]
        factory_calls = [
            call
            for call in ast.walk(forward)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_forward_problem"
        ]
        direct_simulations = [
            call
            for call in ast.walk(forward)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "Simulation"
        ]
        self.assertEqual(len(factory_calls), 1)
        self.assertFalse(direct_simulations)


if __name__ == "__main__":
    unittest.main()
