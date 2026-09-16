#!/usr/bin/env python3
"""Validate the stochastic line-source cosine ensemble and its flux closure.

The target example exposes its basis factory and simulation helper. This
wrapper loads each geometry without executing the target's top-level sweep,
runs all 15 cosine modes, and then rotates the first two modes by 45 degrees.
Linearity and quadratic Poynting flux imply the per-frequency trace identity

``F(phi0) + F(phi1) == F((phi0+phi1)/sqrt(2)) + F((phi0-phi1)/sqrt(2))``.

The generic example oracle observes all 60 nested ``Simulation.run`` calls and
retains the complete modal spectra, actual amplitude-callback counts, ensemble
convergence observables, and independent closure observables.
"""

from __future__ import annotations

import argparse
import pathlib
import runpy
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


MODE_COUNT = 15
CONVERGENCE_MODE_COUNT = 12
MAX_CONVERGENCE_RELATIVE_L2 = 0.02
MIN_BASIS_RESPONSE_SEPARATION = 1e-3
MIN_FULL_BASIS_PAIRWISE_RELATIVE_L2 = 0.05
ACTIVE_MODE_RELATIVE_L2_THRESHOLD = 1e-5
MIN_FULL_BASIS_ACTIVE_MODE_COUNTS = (9, 13)
HIGH_MODE_AMPLITUDE_SCALE = 0.5
HIGH_MODE_FLUX_SCALE = HIGH_MODE_AMPLITUDE_SCALE**2
HIGH_MODE_SCALING_RELATIVE_L2_TOLERANCE = 0.01
HIGH_MODE_SCALING_ABSOLUTE_L2_TOLERANCE = 1e-7


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("--resolution", type=positive_int, default=50)
    parser.add_argument("--frequencies", type=positive_int, default=16)
    args = parser.parse_args(argv)
    if not args.example.is_file():
        raise RuntimeError(f"stochastic line-source example does not exist: {args.example}")
    return args


def _load_target(
    example: pathlib.Path,
    resolution: int,
    frequencies: int,
    textured: bool,
    run_path: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    target_argv = [
        str(example),
        "-res",
        str(resolution),
        "-nf",
        str(frequencies),
        "-nsrc",
        str(MODE_COUNT),
        "-method",
        "3",
        "--validation-define-only",
        "--no-save",
    ]
    if textured:
        target_argv.append("-textured")
    previous_argv = sys.argv
    try:
        sys.argv = target_argv
        return run_path(str(example), run_name="__main__")
    finally:
        sys.argv = previous_argv


def _validate_spectra(
    name: str,
    spectra: np.ndarray,
    expected_shape: tuple[int, ...],
    *,
    require_signal: bool = True,
) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=float)
    if spectra.shape != expected_shape:
        raise RuntimeError(
            f"{name} has shape {spectra.shape}, expected {expected_shape}"
        )
    if spectra.size == 0:
        raise RuntimeError(f"{name} is empty")
    if not np.all(np.isfinite(spectra)):
        raise RuntimeError(f"{name} contains non-finite flux")
    scale = float(np.max(np.abs(spectra)))
    if require_signal and scale <= 1e-12:
        raise RuntimeError(f"{name} contains only empty source spectra")
    if np.min(spectra) < -1e-5 * scale:
        raise RuntimeError(f"{name} has a materially negative outward flux")
    return spectra


def _target_functions(
    namespace: dict[str, Any],
) -> tuple[Callable[[int, int], Any], Callable[[int], Callable[[Any], float]]]:
    compute_flux = namespace.get("compute_flux")
    source_factory = namespace.get("src_amp_func")
    if not callable(compute_flux) or not callable(source_factory):
        raise RuntimeError("target does not expose compute_flux and src_amp_func")
    return compute_flux, source_factory


def _validate_frequency_vector(
    value: Any,
    frequencies: int,
    reference: np.ndarray | None,
    name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (frequencies,):
        raise RuntimeError(
            f"{name} has frequency shape {vector.shape}, expected {(frequencies,)}"
        )
    if not np.all(np.isfinite(vector)):
        raise RuntimeError(f"{name} contains non-finite frequencies")
    if reference is not None and not np.allclose(
        vector, reference, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(f"{name} used a different frequency grid")
    return vector


def _basis_spectra(
    namespace: dict[str, Any], frequencies: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Callable[[int], Callable[[Any], float]],
]:
    compute_flux, original_factory = _target_functions(namespace)
    callback_counts = np.zeros(MODE_COUNT, dtype=np.int64)

    def counting_factory(index: int):
        if index < 0 or index >= MODE_COUNT:
            raise RuntimeError(f"unexpected line-source basis index: {index}")
        original_amplitude = original_factory(index)

        def amplitude(point):
            callback_counts[index] += 1
            return original_amplitude(point)

        return amplitude

    function_globals = compute_flux.__globals__
    previous_factory = function_globals.get("src_amp_func")
    reference_frequencies = None
    columns = []
    function_globals["src_amp_func"] = counting_factory
    try:
        for index in range(MODE_COUNT):
            run_frequencies, flux = compute_flux(3, index)
            run_frequencies = _validate_frequency_vector(
                run_frequencies,
                frequencies,
                reference_frequencies,
                f"cosine mode {index}",
            )
            if reference_frequencies is None:
                reference_frequencies = run_frequencies
            columns.append(np.asarray(flux, dtype=float))
    finally:
        if previous_factory is None:
            function_globals.pop("src_amp_func", None)
        else:
            function_globals["src_amp_func"] = previous_factory

    if reference_frequencies is None:
        raise RuntimeError("line-source basis sweep did not run")
    spectra = _validate_spectra(
        "cosine-basis spectra",
        np.stack(columns, axis=1),
        (frequencies, MODE_COUNT),
    )
    if np.any(callback_counts <= 0):
        missing = np.flatnonzero(callback_counts <= 0).tolist()
        raise RuntimeError(
            "line-source amp_func was not evaluated for cosine modes " f"{missing}"
        )
    return reference_frequencies, spectra, callback_counts, original_factory


def _rotated_spectra(
    namespace: dict[str, Any],
    original_factory: Callable[[int], Callable[[Any], float]],
    reference_frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    compute_flux, _ = _target_functions(namespace)
    basis0 = original_factory(0)
    basis1 = original_factory(1)
    callback_counts = np.zeros(2, dtype=np.int64)

    def rotated_factory(index: int):
        if index not in (0, 1):
            raise RuntimeError(f"unexpected rotated line-source index: {index}")
        sign = 1.0 if index == 0 else -1.0

        def amplitude(point):
            callback_counts[index] += 1
            return (basis0(point) + sign * basis1(point)) / np.sqrt(2.0)

        return amplitude

    function_globals = compute_flux.__globals__
    previous_factory = function_globals.get("src_amp_func")
    function_globals["src_amp_func"] = rotated_factory
    try:
        rotated_columns = []
        for index in range(2):
            frequencies, flux = compute_flux(3, index)
            _validate_frequency_vector(
                frequencies,
                reference_frequencies.size,
                reference_frequencies,
                f"rotated mode {index}",
            )
            rotated_columns.append(np.asarray(flux, dtype=float))
    finally:
        if previous_factory is None:
            function_globals.pop("src_amp_func", None)
        else:
            function_globals["src_amp_func"] = previous_factory

    rotated_flux = _validate_spectra(
        "rotated-basis spectra",
        np.stack(rotated_columns, axis=1),
        (reference_frequencies.size, 2),
    )
    if np.any(callback_counts <= 0):
        missing = np.flatnonzero(callback_counts <= 0).tolist()
        raise RuntimeError(
            "line-source amp_func was not evaluated for rotated modes " f"{missing}"
        )
    return rotated_flux, callback_counts


def _scaled_high_mode_spectra(
    namespace: dict[str, Any],
    original_factory: Callable[[int], Callable[[Any], float]],
    reference_frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    compute_flux, _ = _target_functions(namespace)
    high_mode_indices = np.arange(2, MODE_COUNT, dtype=int)
    callback_counts = np.zeros(high_mode_indices.size, dtype=np.int64)

    def scaled_factory(index: int):
        if index < 2 or index >= MODE_COUNT:
            raise RuntimeError(f"unexpected scaled line-source index: {index}")
        original_amplitude = original_factory(index)
        count_index = index - 2

        def amplitude(point):
            callback_counts[count_index] += 1
            return HIGH_MODE_AMPLITUDE_SCALE * original_amplitude(point)

        return amplitude

    function_globals = compute_flux.__globals__
    previous_factory = function_globals.get("src_amp_func")
    scaled_columns = []
    function_globals["src_amp_func"] = scaled_factory
    try:
        for index in high_mode_indices:
            frequencies, flux = compute_flux(3, int(index))
            _validate_frequency_vector(
                frequencies,
                reference_frequencies.size,
                reference_frequencies,
                f"scaled cosine mode {int(index)}",
            )
            scaled_columns.append(np.asarray(flux, dtype=float))
    finally:
        if previous_factory is None:
            function_globals.pop("src_amp_func", None)
        else:
            function_globals["src_amp_func"] = previous_factory

    scaled_flux = _validate_spectra(
        "scaled high-mode spectra",
        np.stack(scaled_columns, axis=1),
        (reference_frequencies.size, high_mode_indices.size),
        require_signal=False,
    )
    if np.any(callback_counts <= 0):
        missing = high_mode_indices[callback_counts <= 0].tolist()
        raise RuntimeError(
            "line-source amp_func was not evaluated for scaled high modes "
            f"{missing}"
        )
    return scaled_flux, callback_counts


def _manual_trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    return float(
        np.sum(
            0.5
            * (values[:-1] + values[1:])
            * (coordinates[1:] - coordinates[:-1])
        )
    )


def run_basis_matrix(
    args: argparse.Namespace,
    *,
    run_path: Callable[..., dict[str, Any]] = runpy.run_path,
) -> dict[str, np.ndarray]:
    flat = _load_target(
        args.example, args.resolution, args.frequencies, False, run_path
    )
    if "fluxes" in flat:
        _validate_spectra(
            "unexpected precomputed flat spectra",
            flat["fluxes"],
            np.asarray(flat["fluxes"]).shape,
        )
    (
        flat_frequencies,
        flat_full_basis_flux,
        flat_basis_callback_counts,
        flat_factory,
    ) = _basis_spectra(flat, args.frequencies)
    flat_scaled_high_mode_flux, flat_scaled_high_mode_callback_counts = (
        _scaled_high_mode_spectra(flat, flat_factory, flat_frequencies)
    )
    flat_rotated_flux, flat_rotated_callback_counts = _rotated_spectra(
        flat, flat_factory, flat_frequencies
    )

    textured = _load_target(
        args.example, args.resolution, args.frequencies, True, run_path
    )
    if "fluxes" in textured:
        _validate_spectra(
            "unexpected precomputed textured spectra",
            textured["fluxes"],
            np.asarray(textured["fluxes"]).shape,
        )
    (
        textured_frequencies,
        textured_full_basis_flux,
        textured_basis_callback_counts,
        textured_factory,
    ) = _basis_spectra(textured, args.frequencies)
    textured_scaled_high_mode_flux, textured_scaled_high_mode_callback_counts = (
        _scaled_high_mode_spectra(textured, textured_factory, textured_frequencies)
    )
    textured_rotated_flux, textured_rotated_callback_counts = _rotated_spectra(
        textured, textured_factory, textured_frequencies
    )
    if not np.allclose(
        textured_frequencies, flat_frequencies, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("flat and textured branches used different frequency grids")
    if not np.all(np.diff(flat_frequencies) > 0):
        raise RuntimeError("line-source frequencies are not strictly increasing")

    flat_basis_flux = flat_full_basis_flux[:, :2]
    textured_basis_flux = textured_full_basis_flux[:, :2]
    basis_flux = np.stack((flat_basis_flux, textured_basis_flux))
    rotated_flux = np.stack((flat_rotated_flux, textured_rotated_flux))
    basis_trace = np.sum(basis_flux, axis=2)
    rotated_trace = np.sum(rotated_flux, axis=2)
    line_unitary_closure_residual = basis_trace - rotated_trace
    line_unitary_closure_relative_l2 = np.asarray(
        [
            np.linalg.norm(line_unitary_closure_residual)
            / max(np.linalg.norm(basis_trace), 1e-30),
            np.linalg.norm(line_unitary_closure_residual[0])
            / max(np.linalg.norm(basis_trace[0]), 1e-30),
            np.linalg.norm(line_unitary_closure_residual[1])
            / max(np.linalg.norm(basis_trace[1]), 1e-30),
        ]
    )
    if np.max(line_unitary_closure_relative_l2) > 2e-3:
        raise RuntimeError(
            "line-source unitary flux trace does not close: "
            f"{np.max(line_unitary_closure_relative_l2):.6g}"
        )

    sample_count = 4097
    sx = float(flat.get("sx", 0.0))
    if sx <= 0:
        raise RuntimeError("target line-source width is not positive")
    textured_sx = float(textured.get("sx", 0.0))
    if not np.isclose(textured_sx, sx, rtol=0.0, atol=1e-12):
        raise RuntimeError("flat and textured branches used different source widths")
    sample_x = np.linspace(-0.5 * sx, 0.5 * sx, sample_count)

    class Point:
        def __init__(self, x: float):
            self.x = x

    basis_functions = [flat_factory(index) for index in range(MODE_COUNT)]
    basis_values = np.asarray(
        [[basis(Point(x)) for x in sample_x] for basis in basis_functions], dtype=float
    )
    line_full_basis_gram = np.empty((MODE_COUNT, MODE_COUNT), dtype=float)
    for row in range(MODE_COUNT):
        for column in range(MODE_COUNT):
            line_full_basis_gram[row, column] = _manual_trapezoid(
                basis_values[row] * basis_values[column], sample_x
            )
    line_full_basis_gram_error = line_full_basis_gram - np.eye(MODE_COUNT)
    if np.max(np.abs(line_full_basis_gram_error)) > 1e-6:
        raise RuntimeError("line-source cosine functions are not orthonormal")
    line_basis_gram = line_full_basis_gram[:2, :2]
    line_basis_gram_error = line_full_basis_gram_error[:2, :2]

    line_basis_response_separation = np.asarray(
        [
            np.linalg.norm(spectra[:, 0] - spectra[:, 1])
            / max(
                np.linalg.norm(spectra[:, 0]),
                np.linalg.norm(spectra[:, 1]),
                1e-30,
            )
            for spectra in (flat_full_basis_flux, textured_full_basis_flux)
        ]
    )
    if np.min(line_basis_response_separation) <= MIN_BASIS_RESPONSE_SEPARATION:
        raise RuntimeError(
            "distinct line-source basis functions produced indistinguishable spectra"
        )

    all_basis_flux = np.stack((flat_full_basis_flux, textured_full_basis_flux))
    basis_flux_by_mode = np.transpose(all_basis_flux, (0, 2, 1))
    pairwise_residual_l2 = np.linalg.norm(
        basis_flux_by_mode[:, :, np.newaxis, :]
        - basis_flux_by_mode[:, np.newaxis, :, :],
        axis=3,
    )
    basis_flux_l2 = np.linalg.norm(basis_flux_by_mode, axis=2)
    active_mode_threshold = (
        ACTIVE_MODE_RELATIVE_L2_THRESHOLD
        * np.max(basis_flux_l2, axis=1, keepdims=True)
    )
    line_full_basis_active_mode_mask = basis_flux_l2 >= active_mode_threshold
    line_full_basis_active_mode_count = np.count_nonzero(
        line_full_basis_active_mode_mask, axis=1
    )
    minimum_active_mode_counts = np.asarray(MIN_FULL_BASIS_ACTIVE_MODE_COUNTS)
    if np.any(line_full_basis_active_mode_count < minimum_active_mode_counts):
        raise RuntimeError(
            "line-source full-basis active mode count is too small: "
            f"observed={line_full_basis_active_mode_count.tolist()}, "
            f"required={minimum_active_mode_counts.tolist()}"
        )
    pairwise_denominator = np.maximum(
        np.maximum(
            basis_flux_l2[:, :, np.newaxis],
            basis_flux_l2[:, np.newaxis, :],
        ),
        1e-30,
    )
    off_diagonal = ~np.eye(MODE_COUNT, dtype=bool)
    active_pair = (
        line_full_basis_active_mode_mask[:, :, np.newaxis]
        & line_full_basis_active_mode_mask[:, np.newaxis, :]
        & off_diagonal[np.newaxis, :, :]
    )
    raw_pairwise_relative_l2 = pairwise_residual_l2 / pairwise_denominator
    line_full_basis_pairwise_relative_l2 = np.where(
        active_pair, raw_pairwise_relative_l2, 0.0
    )
    line_full_basis_min_pairwise_relative_l2 = np.min(
        np.where(active_pair, raw_pairwise_relative_l2, np.inf), axis=(1, 2)
    )
    line_mode_sum_m12 = np.sum(
        all_basis_flux[:, :, :CONVERGENCE_MODE_COUNT], axis=2
    )
    line_mode_sum_m15 = np.sum(all_basis_flux, axis=2)
    if np.min(line_mode_sum_m12) <= 0 or np.min(line_mode_sum_m15) <= 0:
        raise RuntimeError("line-source modal ensemble produced non-positive flux")
    line_normalized_texture_over_flat_m12 = (
        line_mode_sum_m12[1] / line_mode_sum_m12[0]
    )
    line_normalized_texture_over_flat_m15 = (
        line_mode_sum_m15[1] / line_mode_sum_m15[0]
    )
    line_m12_m15_convergence_residual = (
        line_normalized_texture_over_flat_m15
        - line_normalized_texture_over_flat_m12
    )
    line_m12_m15_convergence_relative_l2 = np.asarray(
        [
            np.linalg.norm(line_m12_m15_convergence_residual)
            / max(np.linalg.norm(line_normalized_texture_over_flat_m15), 1e-30)
        ]
    )
    if line_m12_m15_convergence_relative_l2[0] > MAX_CONVERGENCE_RELATIVE_L2:
        raise RuntimeError(
            "line-source M=12 to M=15 ensemble did not converge: relative L2 "
            f"{line_m12_m15_convergence_relative_l2[0]:.6g}"
        )
    line_high_mode_tail_flux = np.sum(
        all_basis_flux[:, :, CONVERGENCE_MODE_COUNT:], axis=2
    )
    line_high_mode_tail_relative_l2 = np.asarray(
        [
            np.linalg.norm(line_high_mode_tail_flux[index])
            / max(np.linalg.norm(line_mode_sum_m15[index]), 1e-30)
            for index in range(2)
        ]
    )

    scaled_high_mode_flux = np.stack(
        (flat_scaled_high_mode_flux, textured_scaled_high_mode_flux)
    )
    scaled_high_mode_expected_flux = (
        HIGH_MODE_FLUX_SCALE * all_basis_flux[:, :, 2:]
    )
    line_scaled_high_mode_residual = (
        scaled_high_mode_flux - scaled_high_mode_expected_flux
    )
    line_scaled_high_mode_reference_l2 = np.linalg.norm(
        scaled_high_mode_expected_flux, axis=1
    )
    line_scaled_high_mode_residual_l2 = np.linalg.norm(
        line_scaled_high_mode_residual, axis=1
    )
    absolute_scale_floor = (
        HIGH_MODE_SCALING_ABSOLUTE_L2_TOLERANCE
        / HIGH_MODE_SCALING_RELATIVE_L2_TOLERANCE
    )
    line_scaled_high_mode_relative_l2 = line_scaled_high_mode_residual_l2 / np.maximum(
        line_scaled_high_mode_reference_l2, absolute_scale_floor
    )
    if (
        np.max(line_scaled_high_mode_relative_l2)
        > HIGH_MODE_SCALING_RELATIVE_L2_TOLERANCE
    ):
        geometry_index, mode_offset = np.unravel_index(
            np.argmax(line_scaled_high_mode_relative_l2),
            line_scaled_high_mode_relative_l2.shape,
        )
        raise RuntimeError(
            "scaled line-source high mode did not produce quadratic flux: "
            f"geometry={geometry_index}, mode={mode_offset + 2}, relative L2="
            f"{line_scaled_high_mode_relative_l2[geometry_index, mode_offset]:.6g}"
        )

    if (
        np.min(line_full_basis_min_pairwise_relative_l2)
        < MIN_FULL_BASIS_PAIRWISE_RELATIVE_L2
    ):
        masked_pairwise_relative_l2 = np.where(
            active_pair,
            raw_pairwise_relative_l2,
            np.inf,
        )
        geometry_index, first_mode, second_mode = np.unravel_index(
            np.argmin(masked_pairwise_relative_l2),
            masked_pairwise_relative_l2.shape,
        )
        raise RuntimeError(
            "line-source full-basis pairwise separation is too small: "
            f"geometry={geometry_index}, modes=({first_mode},{second_mode}), "
            "relative L2="
            f"{masked_pairwise_relative_l2[geometry_index, first_mode, second_mode]:.6g}"
        )

    line_texture_relative_difference = np.asarray(
        [
            np.linalg.norm(textured_full_basis_flux - flat_full_basis_flux)
            / max(np.linalg.norm(flat_full_basis_flux), 1e-30)
        ]
    )
    if line_texture_relative_difference[0] <= 1e-4:
        raise RuntimeError("textured geometry did not change the line-source spectra")

    return {
        "line_frequencies": flat_frequencies,
        "line_flat_basis_flux": flat_basis_flux,
        "line_flat_basis_flux_full": flat_full_basis_flux,
        "line_flat_rotated_flux": flat_rotated_flux,
        "line_textured_basis_flux": textured_basis_flux,
        "line_textured_basis_flux_full": textured_full_basis_flux,
        "line_textured_rotated_flux": textured_rotated_flux,
        "line_scaled_high_mode_indices": np.arange(2, MODE_COUNT, dtype=float),
        "line_flat_scaled_high_mode_flux": flat_scaled_high_mode_flux,
        "line_textured_scaled_high_mode_flux": textured_scaled_high_mode_flux,
        "line_scaled_high_mode_callback_counts": np.stack(
            (
                flat_scaled_high_mode_callback_counts,
                textured_scaled_high_mode_callback_counts,
            )
        ),
        "line_scaled_high_mode_residual": line_scaled_high_mode_residual,
        "line_scaled_high_mode_reference_l2": line_scaled_high_mode_reference_l2,
        "line_scaled_high_mode_residual_l2": line_scaled_high_mode_residual_l2,
        "line_scaled_high_mode_relative_l2": line_scaled_high_mode_relative_l2,
        "line_basis_callback_counts": np.stack(
            (flat_basis_callback_counts, textured_basis_callback_counts)
        ),
        "line_rotated_callback_counts": np.stack(
            (flat_rotated_callback_counts, textured_rotated_callback_counts)
        ),
        "line_basis_trace": basis_trace,
        "line_rotated_trace": rotated_trace,
        "line_unitary_closure_residual": line_unitary_closure_residual,
        "line_unitary_closure_relative_l2": line_unitary_closure_relative_l2,
        "line_basis_gram": line_basis_gram,
        "line_basis_gram_error": line_basis_gram_error,
        "line_full_basis_gram": line_full_basis_gram,
        "line_full_basis_gram_error": line_full_basis_gram_error,
        "line_basis_response_separation": line_basis_response_separation,
        "line_full_basis_active_mode_mask": (
            line_full_basis_active_mode_mask.astype(float)
        ),
        "line_full_basis_active_mode_count": line_full_basis_active_mode_count,
        "line_full_basis_pairwise_relative_l2": (
            line_full_basis_pairwise_relative_l2
        ),
        "line_full_basis_min_pairwise_relative_l2": (
            line_full_basis_min_pairwise_relative_l2
        ),
        "line_mode_sum_m12": line_mode_sum_m12,
        "line_mode_sum_m15": line_mode_sum_m15,
        "line_normalized_texture_over_flat_m12": line_normalized_texture_over_flat_m12,
        "line_normalized_texture_over_flat_m15": line_normalized_texture_over_flat_m15,
        "line_m12_m15_convergence_residual": line_m12_m15_convergence_residual,
        "line_m12_m15_convergence_relative_l2": line_m12_m15_convergence_relative_l2,
        "line_high_mode_tail_flux": line_high_mode_tail_flux,
        "line_high_mode_tail_relative_l2": line_high_mode_tail_relative_l2,
        "line_texture_relative_difference": line_texture_relative_difference,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    return run_basis_matrix(parse_args(argv))


if __name__ == "__main__":
    globals().update(main())
