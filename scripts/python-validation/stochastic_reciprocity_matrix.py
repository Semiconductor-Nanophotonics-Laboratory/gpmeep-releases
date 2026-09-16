#!/usr/bin/env python3
"""Run a discrete-quadrature-aligned stochastic-emitter reciprocity oracle.

The backward calculation integrates the squared DFT field along the source
line. To make the forward and backward paths represent the same discrete
quadrature even at bounded resolution, this wrapper represents every x-grid
sample in the period. Translation invariance collapses the flat branch to one
weighted solve, while mirror pairs collapse the textured branch to one solve
per unique position. Normalizing textured by flat response cancels the common
source and modal normalization factors.
"""

from __future__ import annotations

import argparse
import pathlib
import runpy
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np


# The upstream tutorial notes that forward/backward equivalence is exact only
# for the continuum model and uses 200 pixels/um for close agreement.  This
# bounded oracle deliberately runs at 50 pixels/um; its independent runtime
# convergence gates separate temporal error from the remaining spatial
# discretization error.  Keep this limit aligned with the scale-aware
# pointwise closure gate below.
RECIPROCITY_CLOSURE_RELATIVE_L2_LIMIT = 0.05


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("--resolution", type=positive_int, default=50)
    parser.add_argument("--frequencies", type=positive_int, default=4)
    parser.add_argument("--flat-runtime", type=positive_int, default=32)
    parser.add_argument("--textured-runtime", type=positive_int, default=128)
    args = parser.parse_args(argv)
    if not args.example.is_file():
        raise RuntimeError(f"reciprocity example does not exist: {args.example}")
    return args


def _finite_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise RuntimeError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} contains non-finite values")
    return array


def _pointwise_relative_errors(
    residual: np.ndarray,
    reference: np.ndarray,
    *,
    axis: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transparent pure and scale-aware pointwise relative errors.

    The scale-aware denominator floors each reference bin at two percent of
    the reference vector's L2 norm. For a matrix, ``axis`` selects the vector
    dimension and the floor is computed independently for every remaining
    slice (the raw convergence matrix uses ``axis=1``).
    """
    residual_array = np.asarray(residual, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if residual_array.shape != reference_array.shape:
        raise RuntimeError("pointwise error operands have different shapes")
    pure = np.abs(residual_array) / np.maximum(
        np.abs(reference_array), np.finfo(float).tiny
    )
    reference_l2 = np.linalg.norm(reference_array, axis=axis, keepdims=True)
    denominator = np.maximum(np.abs(reference_array), 0.02 * reference_l2)
    scale_aware = np.abs(residual_array) / np.maximum(
        denominator, np.finfo(float).tiny
    )
    return pure, scale_aware


def _gate_scale_aware_pointwise(error: np.ndarray, name: str) -> None:
    maximum = float(np.max(np.asarray(error, dtype=float)))
    if maximum > 0.05:
        raise RuntimeError(
            f"{name} has a large scale-aware pointwise error: {maximum:.6g}"
        )


def mirror_quadrature(dipole_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the unique indices and multiplicities for a periodic mirror pair."""
    if dipole_count < 2:
        raise RuntimeError("validation needs at least two source-line samples")
    unique_indices = np.arange(dipole_count // 2 + 1, dtype=int)
    weights = np.full(unique_indices.size, 2.0)
    weights[0] = 1.0
    if dipole_count % 2 == 0:
        weights[-1] = 1.0
    if int(np.sum(weights)) != dipole_count:
        raise RuntimeError("reciprocity mirror-quadrature weights do not close")
    return unique_indices, weights


@contextmanager
def _patched_target_globals(
    functions: Sequence[Callable[..., Any]], updates: dict[str, Any]
):
    global_maps: list[dict[str, Any]] = []
    for function in functions:
        function_globals = getattr(function, "__globals__", None)
        if not isinstance(function_globals, dict):
            raise RuntimeError("target solver does not expose mutable Python globals")
        if not any(function_globals is item for item in global_maps):
            global_maps.append(function_globals)

    snapshots = [
        (
            function_globals,
            {
                name: (name in function_globals, function_globals.get(name))
                for name in updates
            },
        )
        for function_globals in global_maps
    ]
    try:
        for function_globals in global_maps:
            function_globals.update(updates)
        yield
    finally:
        for function_globals, snapshot in reversed(snapshots):
            for name, (was_present, value) in snapshot.items():
                if was_present:
                    function_globals[name] = value
                else:
                    function_globals.pop(name, None)


def _validate_backward_metadata(
    metadata: Any,
    name: str,
    frequencies: int,
    dipole_count: int,
    sx: float,
) -> dict[str, np.ndarray]:
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{name} did not return DFT metadata")

    metadata_count = dipole_count + 3
    x_metadata = _finite_array(
        metadata.get("x"), f"{name} x metadata", (metadata_count,)
    )
    metadata_weights = _finite_array(
        metadata.get("weights"), f"{name} cubature weights", (metadata_count,)
    )
    raw_power = _finite_array(
        metadata.get("raw_power"),
        f"{name} raw DFT power",
        (frequencies, metadata_count),
    )
    if np.min(raw_power) < 0:
        raise RuntimeError(f"{name} raw DFT power contains negative values")

    grid_spacing = sx / dipole_count
    expected_x = -0.5 * sx - grid_spacing + grid_spacing * np.arange(
        metadata_count
    )
    expected_weights = np.zeros(metadata_count, dtype=float)
    expected_weights[1] = 0.5 * grid_spacing
    expected_weights[-2] = 0.5 * grid_spacing
    expected_weights[2:-2] = grid_spacing
    x_error = x_metadata - expected_x
    weight_error = metadata_weights - expected_weights
    topology_atol = max(1e-12, 1e-6 * grid_spacing)
    ghost_indices = np.flatnonzero(expected_weights == 0.0)
    ghost_weight_atol = 32 * np.finfo(float).eps * max(grid_spacing, 1.0)
    if not np.allclose(x_metadata, expected_x, rtol=0.0, atol=topology_atol):
        raise RuntimeError(f"{name} does not have the required N+3 periodic x grid")
    if not np.allclose(
        metadata_weights, expected_weights, rtol=0.0, atol=topology_atol
    ):
        raise RuntimeError(
            f"{name} does not have zero ghosts and half-weight periodic endpoints"
        )
    ghost_weight_max_abs = np.asarray(
        [float(np.max(np.abs(metadata_weights[ghost_indices])))]
    )
    if ghost_weight_max_abs[0] > ghost_weight_atol:
        raise RuntimeError(
            f"{name} ghost cubature weights are not zero to machine precision"
        )
    if not np.all(np.diff(x_metadata) > 0):
        raise RuntimeError(f"{name} x metadata is not strictly increasing")
    if np.min(metadata_weights) < -topology_atol:
        raise RuntimeError(f"{name} cubature weights contain negative values")

    effective_count = np.asarray([np.sum(metadata_weights) / grid_spacing])
    if not np.allclose(
        effective_count, [dipole_count], rtol=0.0, atol=1e-5
    ):
        raise RuntimeError(f"{name} cubature weights have the wrong effective count")
    weighted_power = raw_power * metadata_weights[np.newaxis, :]
    recomputed_flux = np.sum(weighted_power, axis=1)
    expected_recomputed_flux = np.sum(
        raw_power * expected_weights[np.newaxis, :], axis=1
    )
    weighted_ghost_contribution = np.sum(
        np.abs(weighted_power[:, ghost_indices]), axis=1
    )
    weighted_ghost_fraction = weighted_ghost_contribution / np.maximum(
        np.abs(expected_recomputed_flux), np.finfo(float).tiny
    )
    if np.max(weighted_ghost_fraction) > 1e-12:
        raise RuntimeError(
            f"{name} weighted ghost contribution is not negligible"
        )
    return {
        "x": x_metadata,
        "weights": metadata_weights,
        "raw_power": raw_power,
        "expected_x": expected_x,
        "expected_weights": expected_weights,
        "x_error": x_error,
        "weight_error": weight_error,
        "effective_count": effective_count,
        "recomputed_flux": recomputed_flux,
        "expected_recomputed_flux": expected_recomputed_flux,
        "ghost_weight_max_abs": ghost_weight_max_abs,
        "weighted_ghost_contribution": weighted_ghost_contribution,
        "weighted_ghost_fraction": weighted_ghost_fraction,
    }


def _evaluate_reciprocity(
    args: argparse.Namespace,
    forward: Callable[..., Any],
    backward: Callable[..., Any],
    sx: float,
    dipole_count: int,
    unique_textured_indices: np.ndarray,
    textured_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    frequency_arrays: list[np.ndarray] = []

    def call_forward(index: int, runtime: int, textured: bool, name: str) -> np.ndarray:
        frequencies, spectrum = forward(index, runtime, textured)
        frequency_arrays.append(
            _finite_array(frequencies, f"{name} frequencies", (args.frequencies,))
        )
        spectrum_array = _finite_array(
            spectrum, f"{name} forward spectrum", (args.frequencies,)
        )
        if np.min(spectrum_array) < 0:
            raise RuntimeError(f"{name} forward spectrum contains negative power")
        return spectrum_array

    backward_metadata: list[dict[str, np.ndarray]] = []

    def call_backward(runtime: int, textured: bool, name: str) -> np.ndarray:
        result = backward(runtime, textured, return_metadata=True)
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(f"{name} did not return frequencies, flux, and metadata")
        frequencies, spectrum, metadata = result
        frequency_arrays.append(
            _finite_array(frequencies, f"{name} frequencies", (args.frequencies,))
        )
        spectrum_array = _finite_array(
            spectrum, f"{name} backward spectrum", (args.frequencies,)
        )
        validated_metadata = _validate_backward_metadata(
            metadata, name, args.frequencies, dipole_count, sx
        )
        recomputed_flux = validated_metadata["recomputed_flux"]
        scale = max(float(np.max(np.abs(recomputed_flux))), 1.0)
        if not np.allclose(
            spectrum_array, recomputed_flux, rtol=1e-12, atol=1e-12 * scale
        ):
            raise RuntimeError(f"{name} did not use its DFT cubature weights")
        expected_recomputed_flux = validated_metadata["expected_recomputed_flux"]
        expected_scale = max(
            float(np.max(np.abs(expected_recomputed_flux))), 1.0
        )
        if not np.allclose(
            spectrum_array,
            expected_recomputed_flux,
            rtol=5e-8,
            atol=1e-12 * expected_scale,
        ):
            raise RuntimeError(
                f"{name} does not match the canonical DFT cubature integral"
            )
        if np.min(spectrum_array) <= 0:
            raise RuntimeError(f"{name} backward spectrum is not positive")
        backward_metadata.append(validated_metadata)
        return spectrum_array

    forward_flat_flux = call_forward(0, args.flat_runtime, False, "flat reference")
    flat_translation_index = max(1, dipole_count // 3)
    forward_flat_translated_flux = call_forward(
        flat_translation_index,
        args.flat_runtime,
        False,
        "flat translated control",
    )

    forward_textured_unique_flux = np.empty(
        (args.frequencies, unique_textured_indices.size), dtype=float
    )
    for column, dipole_index in enumerate(unique_textured_indices):
        forward_textured_unique_flux[:, column] = call_forward(
            int(dipole_index),
            args.textured_runtime,
            True,
            f"textured source {int(dipole_index)}",
        )

    textured_reflection_index = max(1, dipole_count // 4)
    if 2 * textured_reflection_index == dipole_count:
        textured_reflection_index = 1
    textured_mirror_index = dipole_count - textured_reflection_index
    representative_column = int(
        np.flatnonzero(unique_textured_indices == textured_reflection_index)[0]
    )
    forward_textured_representative_flux = forward_textured_unique_flux[
        :, representative_column
    ]
    forward_textured_reflected_flux = call_forward(
        textured_mirror_index,
        args.textured_runtime,
        True,
        "textured reflected control",
    )

    backward_flat_flux = call_backward(
        args.flat_runtime, False, "flat backward reference"
    )
    backward_textured_flux = call_backward(
        args.textured_runtime, True, "textured backward reference"
    )

    forward_flat_doubled_flux = call_forward(
        0, 2 * args.flat_runtime, False, "flat forward doubled-runtime aggregate"
    )
    forward_textured_unique_doubled_flux = np.empty(
        (args.frequencies, unique_textured_indices.size), dtype=float
    )
    for column, dipole_index in enumerate(unique_textured_indices):
        forward_textured_unique_doubled_flux[:, column] = call_forward(
            int(dipole_index),
            2 * args.textured_runtime,
            True,
            f"textured doubled-runtime source {int(dipole_index)}",
        )
    forward_textured_doubled_flux = forward_textured_unique_doubled_flux[
        :, representative_column
    ]
    backward_flat_doubled_flux = call_backward(
        2 * args.flat_runtime, False, "flat backward doubled-runtime control"
    )
    backward_textured_doubled_flux = call_backward(
        2 * args.textured_runtime,
        True,
        "textured backward doubled-runtime control",
    )

    reciprocity_frequencies = frequency_arrays[0]
    for frequency_array in frequency_arrays[1:]:
        if not np.allclose(
            frequency_array, reciprocity_frequencies, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("reciprocity branches used different frequency grids")
    if not np.all(np.diff(reciprocity_frequencies) > 0):
        raise RuntimeError("reciprocity frequencies are not strictly increasing")

    flat_translation_flux = np.stack(
        (forward_flat_flux, forward_flat_translated_flux)
    )
    flat_translation_residual = (
        forward_flat_translated_flux - forward_flat_flux
    )
    flat_translation_relative_l2 = np.asarray(
        [
            np.linalg.norm(flat_translation_residual)
            / max(np.linalg.norm(forward_flat_flux), 1e-30)
        ]
    )
    if flat_translation_relative_l2[0] > 0.01:
        raise RuntimeError("flat translated dipoles are not equivalent")

    textured_reflection_flux = np.stack(
        (forward_textured_representative_flux, forward_textured_reflected_flux)
    )
    textured_reflection_residual = (
        forward_textured_reflected_flux - forward_textured_representative_flux
    )
    textured_reflection_relative_l2 = np.asarray(
        [
            np.linalg.norm(textured_reflection_residual)
            / max(np.linalg.norm(forward_textured_representative_flux), 1e-30)
        ]
    )
    if textured_reflection_relative_l2[0] > 0.01:
        raise RuntimeError("textured reflected dipoles are not equivalent")

    forward_base_summed_flux = np.stack(
        (
            dipole_count * forward_flat_flux,
            forward_textured_unique_flux @ textured_weights,
        )
    )
    forward_doubled_summed_flux = np.stack(
        (
            dipole_count * forward_flat_doubled_flux,
            forward_textured_unique_doubled_flux @ textured_weights,
        )
    )
    backward_base_flux = np.stack((backward_flat_flux, backward_textured_flux))
    backward_doubled_flux = np.stack(
        (backward_flat_doubled_flux, backward_textured_doubled_flux)
    )
    if (
        np.min(forward_base_summed_flux) <= 0
        or np.min(forward_doubled_summed_flux) <= 0
        or np.min(backward_base_flux) <= 0
        or np.min(backward_doubled_flux) <= 0
    ):
        raise RuntimeError("reciprocity path produced non-positive total power")

    convergence_base_flux = np.stack(
        (
            forward_flat_flux,
            forward_textured_representative_flux,
            backward_flat_flux,
            backward_textured_flux,
        )
    )
    convergence_doubled_flux = np.stack(
        (
            forward_flat_doubled_flux,
            forward_textured_doubled_flux,
            backward_flat_doubled_flux,
            backward_textured_doubled_flux,
        )
    )
    convergence_residual = convergence_base_flux - convergence_doubled_flux
    convergence_relative_l2 = np.asarray(
        [
            np.linalg.norm(convergence_residual[row])
            / max(np.linalg.norm(convergence_doubled_flux[row]), 1e-30)
            for row in range(convergence_residual.shape[0])
        ]
    )
    if np.max(convergence_relative_l2) > 0.1:
        raise RuntimeError(
            "representative reciprocity spectra did not converge when runtime doubled"
        )
    (
        convergence_pointwise_relative_error,
        convergence_scale_aware_pointwise_relative_error,
    ) = _pointwise_relative_errors(
        convergence_residual, convergence_doubled_flux, axis=1
    )
    _gate_scale_aware_pointwise(
        convergence_scale_aware_pointwise_relative_error,
        "representative reciprocity runtime convergence",
    )

    forward_base_normalized_flux = (
        forward_base_summed_flux[1] / forward_base_summed_flux[0]
    )
    forward_doubled_normalized_flux = (
        forward_doubled_summed_flux[1] / forward_doubled_summed_flux[0]
    )
    backward_base_normalized_flux = backward_base_flux[1] / backward_base_flux[0]
    backward_doubled_normalized_flux = (
        backward_doubled_flux[1] / backward_doubled_flux[0]
    )

    def normalized_convergence(
        base: np.ndarray, doubled: np.ndarray, name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        residual = base - doubled
        relative_l2 = np.asarray(
            [np.linalg.norm(residual) / max(np.linalg.norm(doubled), 1e-30)]
        )
        pointwise, scale_aware_pointwise = _pointwise_relative_errors(
            residual, doubled
        )
        if relative_l2[0] > 0.02:
            raise RuntimeError(
                f"{name} normalized textured/flat ratio did not converge: "
                f"relative L2 {relative_l2[0]:.6g}"
            )
        _gate_scale_aware_pointwise(
            scale_aware_pointwise,
            f"{name} normalized textured/flat ratio runtime drift",
        )
        return residual, relative_l2, pointwise, scale_aware_pointwise

    (
        forward_normalized_convergence_residual,
        forward_normalized_convergence_relative_l2,
        forward_normalized_convergence_pointwise_relative_error,
        forward_normalized_convergence_scale_aware_pointwise_relative_error,
    ) = normalized_convergence(
        forward_base_normalized_flux,
        forward_doubled_normalized_flux,
        "forward aggregate",
    )
    (
        backward_normalized_convergence_residual,
        backward_normalized_convergence_relative_l2,
        backward_normalized_convergence_pointwise_relative_error,
        backward_normalized_convergence_scale_aware_pointwise_relative_error,
    ) = normalized_convergence(
        backward_base_normalized_flux,
        backward_doubled_normalized_flux,
        "backward",
    )

    reciprocity_base_residual = (
        forward_base_normalized_flux - backward_base_normalized_flux
    )
    reciprocity_base_relative_l2 = np.asarray(
        [
            np.linalg.norm(reciprocity_base_residual)
            / max(np.linalg.norm(backward_base_normalized_flux), 1e-30)
        ]
    )
    reciprocity_base_pointwise_relative_error = np.abs(
        reciprocity_base_residual
    ) / np.maximum(np.abs(backward_base_normalized_flux), 1e-30)

    # The final closure is intentionally based only on the more-converged,
    # complete doubled-runtime forward and backward aggregates.
    forward_normalized_flux = forward_doubled_normalized_flux
    backward_normalized_flux = backward_doubled_normalized_flux
    reciprocity_residual = forward_normalized_flux - backward_normalized_flux
    reciprocity_relative_l2 = np.asarray(
        [
            np.linalg.norm(reciprocity_residual)
            / max(np.linalg.norm(backward_normalized_flux), 1e-30)
        ]
    )
    (
        reciprocity_pointwise_relative_error,
        reciprocity_scale_aware_pointwise_relative_error,
    ) = _pointwise_relative_errors(
        reciprocity_residual, backward_normalized_flux
    )
    reciprocity_max_relative_error = np.asarray(
        [np.max(reciprocity_pointwise_relative_error)]
    )
    reciprocity_max_scale_aware_pointwise_relative_error = np.asarray(
        [np.max(reciprocity_scale_aware_pointwise_relative_error)]
    )
    if (
        reciprocity_relative_l2[0]
        > RECIPROCITY_CLOSURE_RELATIVE_L2_LIMIT
    ):
        raise RuntimeError(
            "discrete forward/backward reciprocity does not close: relative L2 "
            f"{reciprocity_relative_l2[0]:.6g}"
        )
    _gate_scale_aware_pointwise(
        reciprocity_scale_aware_pointwise_relative_error,
        "discrete forward/backward reciprocity",
    )

    texture_response_l2 = np.asarray(
        [
            np.linalg.norm(forward_normalized_flux - 1.0),
            np.linalg.norm(backward_normalized_flux - 1.0),
        ]
    )
    if np.min(texture_response_l2) <= 1e-3:
        raise RuntimeError("textured geometry has no measurable reciprocity response")

    metadata_x = np.stack([item["x"] for item in backward_metadata])
    metadata_weights = np.stack([item["weights"] for item in backward_metadata])
    metadata_raw_power = np.stack([item["raw_power"] for item in backward_metadata])
    metadata_x_error = np.stack([item["x_error"] for item in backward_metadata])
    metadata_weight_error = np.stack(
        [item["weight_error"] for item in backward_metadata]
    )
    metadata_effective_count = np.concatenate(
        [item["effective_count"] for item in backward_metadata]
    )
    backward_recomputed_flux = np.stack(
        [item["recomputed_flux"] for item in backward_metadata]
    )
    backward_expected_recomputed_flux = np.stack(
        [item["expected_recomputed_flux"] for item in backward_metadata]
    )
    metadata_ghost_weight_max_abs = np.concatenate(
        [item["ghost_weight_max_abs"] for item in backward_metadata]
    )
    metadata_weighted_ghost_contribution = np.stack(
        [item["weighted_ghost_contribution"] for item in backward_metadata]
    )
    metadata_weighted_ghost_fraction = np.stack(
        [item["weighted_ghost_fraction"] for item in backward_metadata]
    )

    return {
        "reciprocity_frequencies": reciprocity_frequencies,
        "reciprocity_dipole_count": np.asarray([dipole_count], dtype=float),
        "reciprocity_textured_unique_indices": unique_textured_indices.astype(float),
        "reciprocity_textured_weights": textured_weights,
        "reciprocity_forward_flat_flux": forward_flat_flux,
        "reciprocity_forward_textured_unique_flux": forward_textured_unique_flux,
        "reciprocity_forward_textured_unique_doubled_flux": (
            forward_textured_unique_doubled_flux
        ),
        "reciprocity_forward_base_summed_flux": forward_base_summed_flux,
        "reciprocity_forward_doubled_summed_flux": forward_doubled_summed_flux,
        "reciprocity_forward_summed_flux": forward_doubled_summed_flux,
        "reciprocity_backward_base_flux": backward_base_flux,
        "reciprocity_backward_flux": backward_doubled_flux,
        "reciprocity_forward_base_normalized_flux": forward_base_normalized_flux,
        "reciprocity_forward_doubled_normalized_flux": (
            forward_doubled_normalized_flux
        ),
        "reciprocity_forward_normalized_flux": forward_normalized_flux,
        "reciprocity_backward_base_normalized_flux": backward_base_normalized_flux,
        "reciprocity_backward_doubled_normalized_flux": (
            backward_doubled_normalized_flux
        ),
        "reciprocity_backward_normalized_flux": backward_normalized_flux,
        "reciprocity_forward_normalized_convergence_residual": (
            forward_normalized_convergence_residual
        ),
        "reciprocity_forward_normalized_convergence_relative_l2": (
            forward_normalized_convergence_relative_l2
        ),
        "reciprocity_forward_normalized_convergence_pointwise_relative_error": (
            forward_normalized_convergence_pointwise_relative_error
        ),
        "reciprocity_forward_normalized_convergence_scale_aware_pointwise_relative_error": (
            forward_normalized_convergence_scale_aware_pointwise_relative_error
        ),
        "reciprocity_backward_normalized_convergence_residual": (
            backward_normalized_convergence_residual
        ),
        "reciprocity_backward_normalized_convergence_relative_l2": (
            backward_normalized_convergence_relative_l2
        ),
        "reciprocity_backward_normalized_convergence_pointwise_relative_error": (
            backward_normalized_convergence_pointwise_relative_error
        ),
        "reciprocity_backward_normalized_convergence_scale_aware_pointwise_relative_error": (
            backward_normalized_convergence_scale_aware_pointwise_relative_error
        ),
        "reciprocity_base_residual": reciprocity_base_residual,
        "reciprocity_base_pointwise_relative_error": (
            reciprocity_base_pointwise_relative_error
        ),
        "reciprocity_base_relative_l2": reciprocity_base_relative_l2,
        "reciprocity_residual": reciprocity_residual,
        "reciprocity_pointwise_relative_error": reciprocity_pointwise_relative_error,
        "reciprocity_scale_aware_pointwise_relative_error": (
            reciprocity_scale_aware_pointwise_relative_error
        ),
        "reciprocity_relative_l2": reciprocity_relative_l2,
        "reciprocity_max_relative_error": reciprocity_max_relative_error,
        "reciprocity_max_scale_aware_pointwise_relative_error": (
            reciprocity_max_scale_aware_pointwise_relative_error
        ),
        "reciprocity_texture_response_l2": texture_response_l2,
        "reciprocity_flat_translation_indices": np.asarray(
            [0, flat_translation_index], dtype=float
        ),
        "reciprocity_flat_translation_flux": flat_translation_flux,
        "reciprocity_flat_translation_residual": flat_translation_residual,
        "reciprocity_flat_translation_relative_l2": flat_translation_relative_l2,
        "reciprocity_textured_reflection_indices": np.asarray(
            [textured_reflection_index, textured_mirror_index], dtype=float
        ),
        "reciprocity_textured_reflection_flux": textured_reflection_flux,
        "reciprocity_textured_reflection_residual": textured_reflection_residual,
        "reciprocity_textured_reflection_relative_l2": (
            textured_reflection_relative_l2
        ),
        "reciprocity_convergence_base_flux": convergence_base_flux,
        "reciprocity_convergence_doubled_flux": convergence_doubled_flux,
        "reciprocity_convergence_residual": convergence_residual,
        "reciprocity_convergence_relative_l2": convergence_relative_l2,
        "reciprocity_convergence_pointwise_relative_error": (
            convergence_pointwise_relative_error
        ),
        "reciprocity_convergence_scale_aware_pointwise_relative_error": (
            convergence_scale_aware_pointwise_relative_error
        ),
        "reciprocity_convergence_runtimes": np.asarray(
            [
                args.flat_runtime,
                args.textured_runtime,
                2 * args.flat_runtime,
                2 * args.textured_runtime,
            ],
            dtype=float,
        ),
        "reciprocity_backward_doubled_flux": backward_doubled_flux,
        "reciprocity_backward_metadata_x": metadata_x,
        "reciprocity_backward_metadata_weights": metadata_weights,
        "reciprocity_backward_metadata_raw_power": metadata_raw_power,
        "reciprocity_backward_metadata_expected_x": backward_metadata[0][
            "expected_x"
        ],
        "reciprocity_backward_metadata_expected_weights": backward_metadata[0][
            "expected_weights"
        ],
        "reciprocity_backward_metadata_x_error": metadata_x_error,
        "reciprocity_backward_metadata_weight_error": metadata_weight_error,
        "reciprocity_backward_metadata_effective_count": metadata_effective_count,
        "reciprocity_backward_recomputed_flux": backward_recomputed_flux,
        "reciprocity_backward_expected_recomputed_flux": (
            backward_expected_recomputed_flux
        ),
        "reciprocity_backward_metadata_ghost_weight_max_abs": (
            metadata_ghost_weight_max_abs
        ),
        "reciprocity_backward_metadata_weighted_ghost_contribution": (
            metadata_weighted_ghost_contribution
        ),
        "reciprocity_backward_metadata_weighted_ghost_fraction": (
            metadata_weighted_ghost_fraction
        ),
    }


def run_reciprocity_matrix(
    args: argparse.Namespace,
    *,
    run_path: Callable[..., dict[str, Any]] = runpy.run_path,
) -> dict[str, np.ndarray]:
    namespace = run_path(str(args.example), run_name="gpmeep_reciprocity_target")
    forward = namespace.get("forward")
    backward = namespace.get("backward")
    if not callable(forward) or not callable(backward):
        raise RuntimeError("target does not expose forward and backward solvers")

    sx = float(namespace.get("sx", 0.0))
    if not np.isfinite(sx) or sx <= 0:
        raise RuntimeError("target period is not positive")
    grid_sample_count = sx * args.resolution
    dipole_count = int(round(grid_sample_count))
    if not np.isclose(grid_sample_count, dipole_count, rtol=0.0, atol=1e-9):
        raise RuntimeError("target period is not an integer number of grid samples")
    if dipole_count < 3:
        raise RuntimeError("validation needs at least three source-line samples")
    unique_textured_indices, textured_weights = mirror_quadrature(dipole_count)

    updates = {
        "resolution": args.resolution,
        "nfreq": args.frequencies,
        "ndipole": dipole_count,
    }
    with _patched_target_globals((forward, backward), updates):
        return _evaluate_reciprocity(
            args,
            forward,
            backward,
            sx,
            dipole_count,
            unique_textured_indices,
            textured_weights,
        )


def main(argv: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    return run_reciprocity_matrix(parse_args(argv))


if __name__ == "__main__":
    globals().update(main())
