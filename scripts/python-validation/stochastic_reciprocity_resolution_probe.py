#!/usr/bin/env python3
"""Probe spatial convergence of the stochastic-emitter reciprocity closure.

The bounded main oracle establishes complete branch coverage and temporal
convergence at resolution 50.  This independent eight-run profile evaluates
the same local forward/backward observable at an exactly shared grid point at
a finer resolution.  It uses the raw backward DFT power at that point rather
than the spatially integrated backward flux, and requires the closure error to
decrease relative to a hash-bound resolution-50 reference.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import runpy
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

import stochastic_reciprocity_matrix as reciprocity


RAW_TIME_RELATIVE_L2_LIMIT = 0.1
NORMALIZED_TIME_RELATIVE_L2_LIMIT = 0.02
REFINED_CLOSURE_RELATIVE_L2_LIMIT = 0.03
REFINEMENT_RATIO_LIMIT = 0.8
POSITION_ATOL = 1e-12
DIAGNOSTICS_PREFIX = "gpmeep-reciprocity-resolution-diagnostics:"


def positive_finite(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("--resolution", type=reciprocity.positive_int, default=90)
    parser.add_argument("--frequencies", type=reciprocity.positive_int, default=4)
    parser.add_argument("--position", type=float, default=-0.25)
    parser.add_argument("--flat-runtime", type=reciprocity.positive_int, default=32)
    parser.add_argument(
        "--textured-runtime", type=reciprocity.positive_int, default=128
    )
    parser.add_argument(
        "--reference-res50-relative-l2",
        type=positive_finite,
        required=True,
    )
    args = parser.parse_args(argv)
    if not args.example.is_file():
        raise RuntimeError(
            f"reciprocity example does not exist: {args.example}"
        )
    if not np.isfinite(args.position):
        raise RuntimeError("probe position must be finite")
    return args


def _relative_l2_rows(residual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.linalg.norm(residual[row])
            / max(np.linalg.norm(reference[row]), np.finfo(float).tiny)
            for row in range(residual.shape[0])
        ]
    )


def _run_probe(
    args: argparse.Namespace,
    forward: Callable[..., Any],
    backward: Callable[..., Any],
    sx: float,
    dipole_count: int,
    source_index: int,
) -> dict[str, np.ndarray]:
    frequency_arrays: list[np.ndarray] = []
    backward_metadata: list[dict[str, np.ndarray]] = []

    def call_forward(runtime: int, textured: bool, name: str) -> np.ndarray:
        frequencies, spectrum = forward(source_index, runtime, textured)
        frequency_arrays.append(
            reciprocity._finite_array(
                frequencies, f"{name} frequencies", (args.frequencies,)
            )
        )
        spectrum_array = reciprocity._finite_array(
            spectrum, f"{name} forward spectrum", (args.frequencies,)
        )
        if np.min(spectrum_array) <= 0:
            raise RuntimeError(f"{name} forward spectrum is not positive")
        return spectrum_array

    def call_backward(
        runtime: int, textured: bool, name: str
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        result = backward(runtime, textured, return_metadata=True)
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(
                f"{name} did not return frequencies, flux, and metadata"
            )
        frequencies, spectrum, metadata = result
        frequency_arrays.append(
            reciprocity._finite_array(
                frequencies, f"{name} frequencies", (args.frequencies,)
            )
        )
        spectrum_array = reciprocity._finite_array(
            spectrum, f"{name} backward spectrum", (args.frequencies,)
        )
        validated = reciprocity._validate_backward_metadata(
            metadata, name, args.frequencies, dipole_count, sx
        )
        recomputed = validated["recomputed_flux"]
        recompute_scale = max(float(np.max(np.abs(recomputed))), 1.0)
        if not np.allclose(
            spectrum_array,
            recomputed,
            rtol=1e-12,
            atol=1e-12 * recompute_scale,
        ):
            raise RuntimeError(f"{name} did not use its DFT cubature weights")
        canonical = validated["expected_recomputed_flux"]
        canonical_scale = max(float(np.max(np.abs(canonical))), 1.0)
        if not np.allclose(
            spectrum_array,
            canonical,
            rtol=5e-8,
            atol=1e-12 * canonical_scale,
        ):
            raise RuntimeError(
                f"{name} does not match the canonical DFT cubature integral"
            )

        matching_indices = np.flatnonzero(
            np.isclose(
                validated["x"],
                args.position,
                rtol=0.0,
                atol=POSITION_ATOL,
            )
        )
        if matching_indices.size != 1:
            raise RuntimeError(
                f"{name} metadata does not contain exactly one probe position"
            )
        metadata_index = int(matching_indices[0])
        expected_metadata_index = source_index + 1
        if metadata_index != expected_metadata_index:
            raise RuntimeError(
                f"{name} probe metadata index {metadata_index} does not match "
                f"source index {source_index}"
            )
        local_weight = float(validated["weights"][metadata_index])
        if not np.isfinite(local_weight) or local_weight <= 0:
            raise RuntimeError(f"{name} probe cubature weight is not positive")
        local_power = validated["raw_power"][:, metadata_index]
        if np.min(local_power) <= 0:
            raise RuntimeError(f"{name} local DFT power is not positive")

        return_recompute_relative_l2 = np.asarray(
            [
                np.linalg.norm(spectrum_array - recomputed)
                / max(np.linalg.norm(recomputed), np.finfo(float).tiny)
            ]
        )
        canonical_recompute_relative_l2 = np.asarray(
            [
                np.linalg.norm(spectrum_array - canonical)
                / max(np.linalg.norm(canonical), np.finfo(float).tiny)
            ]
        )
        validated = dict(validated)
        validated.update(
            {
                "metadata_index": np.asarray([metadata_index], dtype=float),
                "position_error": np.asarray(
                    [validated["x"][metadata_index] - args.position]
                ),
                "local_weight": np.asarray([local_weight]),
                "local_power": np.asarray(local_power, dtype=float),
                "return_recompute_relative_l2": (
                    return_recompute_relative_l2
                ),
                "canonical_recompute_relative_l2": (
                    canonical_recompute_relative_l2
                ),
            }
        )
        backward_metadata.append(validated)
        return local_power, validated

    # Keep this exact order aligned with the outer oracle's eight per-run
    # timestep contracts.  The final run is intentionally the longest one.
    forward_flat_base = call_forward(
        args.flat_runtime, False, "forward flat base"
    )
    forward_textured_base = call_forward(
        args.textured_runtime, True, "forward textured base"
    )
    backward_flat_base, _ = call_backward(
        args.flat_runtime, False, "backward flat base"
    )
    backward_textured_base, _ = call_backward(
        args.textured_runtime, True, "backward textured base"
    )
    forward_flat_doubled = call_forward(
        2 * args.flat_runtime, False, "forward flat doubled"
    )
    forward_textured_doubled = call_forward(
        2 * args.textured_runtime, True, "forward textured doubled"
    )
    backward_flat_doubled, _ = call_backward(
        2 * args.flat_runtime, False, "backward flat doubled"
    )
    backward_textured_doubled, _ = call_backward(
        2 * args.textured_runtime, True, "backward textured doubled"
    )

    frequencies = frequency_arrays[0]
    for frequency_array in frequency_arrays[1:]:
        if not np.allclose(
            frequency_array, frequencies, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("resolution probe branches used different frequencies")
    if not np.all(np.diff(frequencies) > 0):
        raise RuntimeError("resolution probe frequencies are not increasing")

    raw_base = np.stack(
        (
            forward_flat_base,
            forward_textured_base,
            backward_flat_base,
            backward_textured_base,
        )
    )
    raw_doubled = np.stack(
        (
            forward_flat_doubled,
            forward_textured_doubled,
            backward_flat_doubled,
            backward_textured_doubled,
        )
    )
    raw_residual = raw_base - raw_doubled
    raw_relative_l2 = _relative_l2_rows(raw_residual, raw_doubled)
    raw_pointwise, raw_scale_aware = reciprocity._pointwise_relative_errors(
        raw_residual, raw_doubled, axis=1
    )

    forward_normalized = np.stack(
        (
            forward_textured_base / forward_flat_base,
            forward_textured_doubled / forward_flat_doubled,
        )
    )
    backward_normalized = np.stack(
        (
            backward_textured_base / backward_flat_base,
            backward_textured_doubled / backward_flat_doubled,
        )
    )
    normalized_base = np.stack(
        (forward_normalized[0], backward_normalized[0])
    )
    normalized_doubled = np.stack(
        (forward_normalized[1], backward_normalized[1])
    )
    normalized_residual = normalized_base - normalized_doubled
    normalized_relative_l2 = _relative_l2_rows(
        normalized_residual, normalized_doubled
    )
    (
        normalized_pointwise,
        normalized_scale_aware,
    ) = reciprocity._pointwise_relative_errors(
        normalized_residual, normalized_doubled, axis=1
    )
    base_closure_residual = forward_normalized[0] - backward_normalized[0]
    base_closure_relative_l2 = np.asarray(
        [
            np.linalg.norm(base_closure_residual)
            / max(
                np.linalg.norm(backward_normalized[0]),
                np.finfo(float).tiny,
            )
        ]
    )
    doubled_closure_residual = (
        forward_normalized[1] - backward_normalized[1]
    )
    doubled_closure_relative_l2 = np.asarray(
        [
            np.linalg.norm(doubled_closure_residual)
            / max(
                np.linalg.norm(backward_normalized[1]),
                np.finfo(float).tiny,
            )
        ]
    )
    (
        doubled_closure_pointwise,
        doubled_closure_scale_aware,
    ) = reciprocity._pointwise_relative_errors(
        doubled_closure_residual, backward_normalized[1]
    )
    reference_res50_relative_l2 = np.asarray(
        [args.reference_res50_relative_l2]
    )
    refinement_ratio = (
        doubled_closure_relative_l2 / reference_res50_relative_l2
    )

    # Emit all small raw observables before applying any scientific gate.  A
    # failed long-running probe is still useful evidence and must identify the
    # exact branch and frequency which failed rather than discarding twelve
    # minutes of spectra behind a generic exception.
    diagnostics = {
        "schema": "gpmeep-reciprocity-resolution-diagnostics-v1",
        "resolution": args.resolution,
        "dipole_count": dipole_count,
        "position": args.position,
        "source_index": source_index,
        "frequencies": frequencies.tolist(),
        "forward_flat": np.stack(
            (forward_flat_base, forward_flat_doubled)
        ).tolist(),
        "forward_textured": np.stack(
            (forward_textured_base, forward_textured_doubled)
        ).tolist(),
        "backward_flat": np.stack(
            (backward_flat_base, backward_flat_doubled)
        ).tolist(),
        "backward_textured": np.stack(
            (backward_textured_base, backward_textured_doubled)
        ).tolist(),
        "raw_time_relative_l2": raw_relative_l2.tolist(),
        "raw_time_scale_aware_max": float(np.max(raw_scale_aware)),
        "forward_normalized": forward_normalized.tolist(),
        "backward_normalized": backward_normalized.tolist(),
        "normalized_time_relative_l2": normalized_relative_l2.tolist(),
        "normalized_time_scale_aware_max": float(
            np.max(normalized_scale_aware)
        ),
        "base_closure_relative_l2": float(base_closure_relative_l2[0]),
        "doubled_closure_relative_l2": float(
            doubled_closure_relative_l2[0]
        ),
        "doubled_closure_scale_aware_max": float(
            np.max(doubled_closure_scale_aware)
        ),
        "reference_res50_relative_l2": float(
            reference_res50_relative_l2[0]
        ),
        "refinement_ratio": float(refinement_ratio[0]),
    }
    print(
        DIAGNOSTICS_PREFIX
        + json.dumps(diagnostics, sort_keys=True, separators=(",", ":")),
        flush=True,
    )

    if np.max(raw_relative_l2) > RAW_TIME_RELATIVE_L2_LIMIT:
        raise RuntimeError(
            "resolution probe raw spectra did not converge: relative L2 "
            f"{raw_relative_l2.tolist()}"
        )
    reciprocity._gate_scale_aware_pointwise(
        raw_scale_aware, "resolution probe raw runtime convergence"
    )
    if np.max(normalized_relative_l2) > NORMALIZED_TIME_RELATIVE_L2_LIMIT:
        raise RuntimeError(
            "resolution probe normalized ratios did not converge: relative "
            f"L2 {normalized_relative_l2.tolist()}"
        )
    reciprocity._gate_scale_aware_pointwise(
        normalized_scale_aware,
        "resolution probe normalized runtime convergence",
    )
    if (
        doubled_closure_relative_l2[0]
        > REFINED_CLOSURE_RELATIVE_L2_LIMIT
    ):
        raise RuntimeError(
            f"resolution-{args.resolution} reciprocity closure "
            f"{doubled_closure_relative_l2[0]} does not meet the 3% gate"
        )
    reciprocity._gate_scale_aware_pointwise(
        doubled_closure_scale_aware,
        f"resolution-{args.resolution} reciprocity closure",
    )
    if refinement_ratio[0] > REFINEMENT_RATIO_LIMIT:
        raise RuntimeError(
            f"resolution-{args.resolution} reciprocity closure refinement "
            f"ratio {refinement_ratio[0]} did not improve enough over the "
            "resolution-50 reference"
        )

    return {
        "reciprocity_resolution_probe_resolution": np.asarray(
            [args.resolution], dtype=float
        ),
        "reciprocity_resolution_probe_dipole_count": np.asarray(
            [dipole_count], dtype=float
        ),
        "reciprocity_resolution_probe_position": np.asarray([args.position]),
        "reciprocity_resolution_probe_source_index": np.asarray(
            [source_index], dtype=float
        ),
        "reciprocity_resolution_probe_metadata_index": np.concatenate(
            [item["metadata_index"] for item in backward_metadata]
        ),
        "reciprocity_resolution_probe_frequencies": frequencies,
        "reciprocity_resolution_probe_forward_flat_flux": np.stack(
            (forward_flat_base, forward_flat_doubled)
        ),
        "reciprocity_resolution_probe_forward_textured_flux": np.stack(
            (forward_textured_base, forward_textured_doubled)
        ),
        "reciprocity_resolution_probe_backward_flat_local_power": np.stack(
            (backward_flat_base, backward_flat_doubled)
        ),
        "reciprocity_resolution_probe_backward_textured_local_power": np.stack(
            (backward_textured_base, backward_textured_doubled)
        ),
        "reciprocity_resolution_probe_forward_normalized": forward_normalized,
        "reciprocity_resolution_probe_backward_normalized": backward_normalized,
        "reciprocity_resolution_probe_raw_time_base": raw_base,
        "reciprocity_resolution_probe_raw_time_doubled": raw_doubled,
        "reciprocity_resolution_probe_raw_time_residual": raw_residual,
        "reciprocity_resolution_probe_raw_time_relative_l2": raw_relative_l2,
        "reciprocity_resolution_probe_raw_time_pointwise": raw_pointwise,
        "reciprocity_resolution_probe_raw_time_scale_aware": raw_scale_aware,
        "reciprocity_resolution_probe_normalized_time_residual": (
            normalized_residual
        ),
        "reciprocity_resolution_probe_normalized_time_relative_l2": (
            normalized_relative_l2
        ),
        "reciprocity_resolution_probe_normalized_time_pointwise": (
            normalized_pointwise
        ),
        "reciprocity_resolution_probe_normalized_time_scale_aware": (
            normalized_scale_aware
        ),
        "reciprocity_resolution_probe_base_closure_residual": (
            base_closure_residual
        ),
        "reciprocity_resolution_probe_base_closure_relative_l2": (
            base_closure_relative_l2
        ),
        "reciprocity_resolution_probe_doubled_closure_residual": (
            doubled_closure_residual
        ),
        "reciprocity_resolution_probe_doubled_closure_pointwise": (
            doubled_closure_pointwise
        ),
        "reciprocity_resolution_probe_doubled_closure_scale_aware": (
            doubled_closure_scale_aware
        ),
        "reciprocity_resolution_probe_doubled_closure_relative_l2": (
            doubled_closure_relative_l2
        ),
        "reciprocity_resolution_probe_reference_res50_relative_l2": (
            reference_res50_relative_l2
        ),
        "reciprocity_resolution_probe_refinement_ratio": refinement_ratio,
        "reciprocity_resolution_probe_metadata_position_error": np.concatenate(
            [item["position_error"] for item in backward_metadata]
        ),
        "reciprocity_resolution_probe_metadata_local_weight": np.concatenate(
            [item["local_weight"] for item in backward_metadata]
        ),
        "reciprocity_resolution_probe_metadata_return_recompute_relative_l2": (
            np.concatenate(
                [
                    item["return_recompute_relative_l2"]
                    for item in backward_metadata
                ]
            )
        ),
        "reciprocity_resolution_probe_metadata_canonical_recompute_relative_l2": (
            np.concatenate(
                [
                    item["canonical_recompute_relative_l2"]
                    for item in backward_metadata
                ]
            )
        ),
        "reciprocity_resolution_probe_metadata_ghost_weight_max_abs": (
            np.concatenate(
                [item["ghost_weight_max_abs"] for item in backward_metadata]
            )
        ),
        "reciprocity_resolution_probe_metadata_ghost_fraction": np.stack(
            [item["weighted_ghost_fraction"] for item in backward_metadata]
        ),
    }


def run_resolution_probe(
    args: argparse.Namespace,
    *,
    run_path: Callable[..., dict[str, Any]] = runpy.run_path,
) -> dict[str, np.ndarray]:
    namespace = run_path(
        str(args.example), run_name="gpmeep_reciprocity_target"
    )
    forward = namespace.get("forward")
    backward = namespace.get("backward")
    if not callable(forward) or not callable(backward):
        raise RuntimeError("target does not expose forward and backward solvers")

    sx = float(namespace.get("sx", 0.0))
    if not np.isfinite(sx) or sx <= 0:
        raise RuntimeError("target period is not positive")
    grid_sample_count = sx * args.resolution
    dipole_count = int(round(grid_sample_count))
    if not np.isclose(
        grid_sample_count, dipole_count, rtol=0.0, atol=1e-9
    ):
        raise RuntimeError("target period is not an integer number of samples")
    if dipole_count < 3:
        raise RuntimeError("resolution probe needs at least three samples")
    # The centered Ez DFT lattice is half a cell out of phase with the
    # periodic source lattice when the period contains an even number of
    # pixels.  Requiring an odd count makes the requested physical point a
    # real sample on both lattices instead of silently interpolating one side
    # of the reciprocity comparison.
    if dipole_count % 2 == 0:
        raise RuntimeError(
            "resolution probe period must contain an odd number of pixels "
            "so the source and centered DFT grids coincide"
        )

    grid_spacing = sx / dipole_count
    source_coordinate = (args.position + 0.5 * sx) / grid_spacing
    source_index = int(round(source_coordinate))
    if source_index <= 0 or source_index >= dipole_count:
        raise RuntimeError("probe position must be an interior periodic sample")
    source_x = sx * (-0.5 + source_index / dipole_count)
    if not np.isclose(source_x, args.position, rtol=0.0, atol=POSITION_ATOL):
        raise RuntimeError("probe position is not exactly shared by this grid")

    updates = {
        "resolution": args.resolution,
        "nfreq": args.frequencies,
        "ndipole": dipole_count,
    }
    with reciprocity._patched_target_globals((forward, backward), updates):
        return _run_probe(
            args, forward, backward, sx, dipole_count, source_index
        )


def main(argv: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    return run_resolution_probe(parse_args(argv))


if __name__ == "__main__":
    globals().update(main())
