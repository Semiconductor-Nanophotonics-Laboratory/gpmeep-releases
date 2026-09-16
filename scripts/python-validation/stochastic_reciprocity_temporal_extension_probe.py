#!/usr/bin/env python3
"""Extend the resolution-90 textured reciprocity runs from 256 to 512.

The full resolution probe already established that the flat 64-runtime
branches are converged.  Its textured 128/256 pair did not meet the temporal
gate at every frequency.  This bounded diagnostic reuses the hash-authenticated
64/256 observables and performs only the two missing textured 512-runtime
solves.  It applies the same strict temporal, pointwise, closure, and spatial
refinement gates before the expensive eight-run profile is repeated.
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


RESOLUTION = 90
FREQUENCIES = 4
POSITION = -0.25
TEXTURED_RUNTIME = 512
RAW_TIME_RELATIVE_L2_LIMIT = 0.1
NORMALIZED_TIME_RELATIVE_L2_LIMIT = 0.02
CLOSURE_RELATIVE_L2_LIMIT = 0.03
POINTWISE_LIMIT = 0.05
REFINEMENT_RATIO_LIMIT = 0.8
FLAT_CONDITION_RATIO_MINIMUM = 1e-5
POSITION_ATOL = 1e-12
DIAGNOSTICS_PREFIX = "gpmeep-reciprocity-temporal-extension-diagnostics:"


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("--resolution", type=_positive_int, default=RESOLUTION)
    parser.add_argument("--frequencies", type=_positive_int, default=FREQUENCIES)
    parser.add_argument("--position", type=float, default=POSITION)
    parser.add_argument(
        "--textured-runtime", type=_positive_int, default=TEXTURED_RUNTIME
    )
    parser.add_argument("--reference-diagnostics-json", required=True)
    args = parser.parse_args(argv)
    if not args.example.is_file():
        raise RuntimeError(f"reciprocity example does not exist: {args.example}")
    if not np.isfinite(args.position):
        raise RuntimeError("probe position must be finite")
    try:
        args.reference_diagnostics = json.loads(
            args.reference_diagnostics_json
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError("reference diagnostics JSON is invalid") from exc
    if not isinstance(args.reference_diagnostics, dict):
        raise RuntimeError("reference diagnostics must be a JSON object")
    return args


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"{name} is not finite")
    return result


def _relative_l2_rows(residual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.linalg.norm(residual[row])
            / max(np.linalg.norm(reference[row]), np.finfo(float).tiny)
            for row in range(residual.shape[0])
        ]
    )


def _symmetric_relative_error(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.maximum(
        np.abs(left) + np.abs(right), np.finfo(float).tiny
    )
    return 2.0 * np.abs(left - right) / denominator


def _reference_array(
    reference: dict[str, Any], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    return reciprocity._finite_array(reference.get(name), name, shape)


def _validate_reference(
    reference: dict[str, Any], args: argparse.Namespace
) -> dict[str, np.ndarray | float]:
    if reference.get("schema") != "gpmeep-reciprocity-resolution-diagnostics-v1":
        raise RuntimeError("unsupported temporal-extension reference schema")
    if (
        reference.get("resolution") != args.resolution
        or reference.get("dipole_count") != 99
        or reference.get("source_index") != 27
        or not np.isclose(
            float(reference.get("position", np.nan)),
            args.position,
            rtol=0.0,
            atol=POSITION_ATOL,
        )
    ):
        raise RuntimeError("temporal-extension reference has the wrong profile")

    frequencies = _reference_array(
        reference, "frequencies", (args.frequencies,)
    )
    forward_flat = _reference_array(reference, "forward_flat", (2, args.frequencies))
    forward_textured = _reference_array(
        reference, "forward_textured", (2, args.frequencies)
    )
    backward_flat = _reference_array(
        reference, "backward_flat", (2, args.frequencies)
    )
    backward_textured = _reference_array(
        reference, "backward_textured", (2, args.frequencies)
    )
    for name, value in (
        ("frequencies", frequencies),
        ("forward flat", forward_flat),
        ("forward textured", forward_textured),
        ("backward flat", backward_flat),
        ("backward textured", backward_textured),
    ):
        if np.min(value) <= 0:
            raise RuntimeError(f"temporal-extension reference {name} is not positive")
    if not np.all(np.diff(frequencies) > 0):
        raise RuntimeError("temporal-extension reference frequencies are not increasing")

    reference_res50 = _finite_scalar(
        reference.get("reference_res50_relative_l2"),
        "resolution-50 reference closure",
    )
    if reference_res50 <= 0:
        raise RuntimeError("resolution-50 reference closure is not positive")
    return {
        "frequencies": frequencies,
        "forward_flat": forward_flat[1],
        "forward_textured": forward_textured[1],
        "backward_flat": backward_flat[1],
        "backward_textured": backward_textured[1],
        "reference_res50": reference_res50,
    }


def _validate_line_coordinates(
    metadata: dict[str, Any], expected_y: float, name: str
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(metadata.get("y"), dtype=float).reshape(-1)
    z = np.asarray(metadata.get("z"), dtype=float).reshape(-1)
    if y.shape != (1,) or z.shape != (1,) or not (
        np.all(np.isfinite(y)) and np.all(np.isfinite(z))
    ):
        raise RuntimeError(f"{name} y/z metadata does not describe one finite line")
    if not np.allclose(y, [expected_y], rtol=0.0, atol=POSITION_ATOL):
        raise RuntimeError(f"{name} y metadata is not at the source line")
    if not np.allclose(z, [0.0], rtol=0.0, atol=POSITION_ATOL):
        raise RuntimeError(f"{name} z metadata is not at zero")
    return y, z


def _run_extension(
    args: argparse.Namespace,
    reference: dict[str, np.ndarray | float],
    forward: Callable[..., Any],
    backward: Callable[..., Any],
    sx: float,
    dipole_count: int,
    source_index: int,
    expected_y: float,
) -> dict[str, np.ndarray]:
    forward_frequencies, forward_spectrum = forward(
        source_index, args.textured_runtime, True
    )
    forward_frequencies = reciprocity._finite_array(
        forward_frequencies, "extended forward frequencies", (args.frequencies,)
    )
    forward_extended = reciprocity._finite_array(
        forward_spectrum, "extended forward textured spectrum", (args.frequencies,)
    )
    if np.min(forward_extended) <= 0:
        raise RuntimeError("extended forward textured spectrum is not positive")

    backward_result = backward(
        args.textured_runtime, True, return_metadata=True
    )
    if not isinstance(backward_result, tuple) or len(backward_result) != 3:
        raise RuntimeError(
            "extended backward run did not return frequencies, flux, and metadata"
        )
    backward_frequencies, backward_spectrum, metadata = backward_result
    backward_frequencies = reciprocity._finite_array(
        backward_frequencies, "extended backward frequencies", (args.frequencies,)
    )
    returned_backward = reciprocity._finite_array(
        backward_spectrum, "extended backward spectrum", (args.frequencies,)
    )
    if not isinstance(metadata, dict):
        raise RuntimeError("extended backward run did not return DFT metadata")
    y_metadata, z_metadata = _validate_line_coordinates(
        metadata, expected_y, "extended backward run"
    )
    validated = reciprocity._validate_backward_metadata(
        metadata,
        "extended backward run",
        args.frequencies,
        dipole_count,
        sx,
    )
    recomputed = validated["recomputed_flux"]
    canonical = validated["expected_recomputed_flux"]
    recompute_scale = max(float(np.max(np.abs(recomputed))), 1.0)
    canonical_scale = max(float(np.max(np.abs(canonical))), 1.0)
    if not np.allclose(
        returned_backward,
        recomputed,
        rtol=1e-12,
        atol=1e-12 * recompute_scale,
    ):
        raise RuntimeError("extended backward run did not use its DFT weights")
    if not np.allclose(
        returned_backward,
        canonical,
        rtol=5e-8,
        atol=1e-12 * canonical_scale,
    ):
        raise RuntimeError("extended backward run is not the canonical DFT integral")

    matching = np.flatnonzero(
        np.isclose(validated["x"], args.position, rtol=0.0, atol=POSITION_ATOL)
    )
    if matching.size != 1:
        raise RuntimeError("extended backward metadata lacks one probe position")
    metadata_index = int(matching[0])
    if metadata_index != source_index + 1:
        raise RuntimeError("extended backward probe metadata index is wrong")
    local_weight = float(validated["weights"][metadata_index])
    backward_extended = validated["raw_power"][:, metadata_index]
    if local_weight <= 0 or np.min(backward_extended) <= 0:
        raise RuntimeError("extended backward local observable is not positive")

    frequencies = np.asarray(reference["frequencies"], dtype=float)
    for name, value in (
        ("forward", forward_frequencies),
        ("backward", backward_frequencies),
    ):
        if not np.allclose(value, frequencies, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"extended {name} frequencies differ from reference")

    forward_flat = np.asarray(reference["forward_flat"], dtype=float)
    backward_flat = np.asarray(reference["backward_flat"], dtype=float)
    forward_prior = np.asarray(reference["forward_textured"], dtype=float)
    backward_prior = np.asarray(reference["backward_textured"], dtype=float)
    reference_res50 = float(reference["reference_res50"])

    flat_condition = np.stack(
        (
            np.abs(forward_flat) / np.max(np.abs(forward_flat)),
            np.abs(backward_flat) / np.max(np.abs(backward_flat)),
        )
    )
    if np.min(flat_condition) < FLAT_CONDITION_RATIO_MINIMUM:
        raise RuntimeError(
            "flat normalization denominator is too poorly conditioned: "
            f"minimum ratio {float(np.min(flat_condition))}"
        )

    raw_prior = np.stack((forward_prior, backward_prior))
    raw_extended = np.stack((forward_extended, backward_extended))
    raw_residual = raw_prior - raw_extended
    raw_relative_l2 = _relative_l2_rows(raw_residual, raw_extended)
    raw_pointwise, raw_scale_aware = reciprocity._pointwise_relative_errors(
        raw_residual, raw_extended, axis=1
    )

    normalized_prior = np.stack(
        (forward_prior / forward_flat, backward_prior / backward_flat)
    )
    normalized_extended = np.stack(
        (forward_extended / forward_flat, backward_extended / backward_flat)
    )
    normalized_residual = normalized_prior - normalized_extended
    normalized_relative_l2 = _relative_l2_rows(
        normalized_residual, normalized_extended
    )
    normalized_pointwise, normalized_scale_aware = (
        reciprocity._pointwise_relative_errors(
            normalized_residual, normalized_extended, axis=1
        )
    )
    normalized_symmetric = _symmetric_relative_error(
        normalized_prior, normalized_extended
    )

    closure_residual = normalized_extended[0] - normalized_extended[1]
    closure_relative_l2 = np.asarray(
        [
            np.linalg.norm(closure_residual)
            / max(np.linalg.norm(normalized_extended[1]), np.finfo(float).tiny)
        ]
    )
    closure_pointwise, closure_scale_aware = reciprocity._pointwise_relative_errors(
        closure_residual, normalized_extended[1]
    )
    closure_symmetric = _symmetric_relative_error(
        normalized_extended[0], normalized_extended[1]
    )
    refinement_ratio = closure_relative_l2 / reference_res50

    return_recompute_relative_l2 = np.asarray(
        [
            np.linalg.norm(returned_backward - recomputed)
            / max(np.linalg.norm(recomputed), np.finfo(float).tiny)
        ]
    )
    canonical_recompute_relative_l2 = np.asarray(
        [
            np.linalg.norm(returned_backward - canonical)
            / max(np.linalg.norm(canonical), np.finfo(float).tiny)
        ]
    )

    diagnostics = {
        "schema": "gpmeep-reciprocity-temporal-extension-diagnostics-v1",
        "resolution": args.resolution,
        "dipole_count": dipole_count,
        "position": args.position,
        "source_index": source_index,
        "metadata_index": metadata_index,
        "textured_runtimes": [args.textured_runtime // 2, args.textured_runtime],
        "frequencies": frequencies.tolist(),
        "raw_prior": raw_prior.tolist(),
        "raw_extended": raw_extended.tolist(),
        "raw_time_relative_l2": raw_relative_l2.tolist(),
        "raw_time_scale_aware_max": float(np.max(raw_scale_aware)),
        "normalized_prior": normalized_prior.tolist(),
        "normalized_extended": normalized_extended.tolist(),
        "normalized_time_relative_l2": normalized_relative_l2.tolist(),
        "normalized_time_symmetric_max": float(np.max(normalized_symmetric)),
        "normalized_time_scale_aware_max": float(
            np.max(normalized_scale_aware)
        ),
        "extended_closure_relative_l2": float(closure_relative_l2[0]),
        "extended_closure_symmetric_max": float(np.max(closure_symmetric)),
        "extended_closure_scale_aware_max": float(
            np.max(closure_scale_aware)
        ),
        "reference_res50_relative_l2": reference_res50,
        "refinement_ratio": float(refinement_ratio[0]),
        "flat_condition_ratio_minimum": float(np.min(flat_condition)),
        "expected_y": expected_y,
        "metadata_y": float(y_metadata[0]),
        "metadata_z": float(z_metadata[0]),
    }
    print(
        DIAGNOSTICS_PREFIX
        + json.dumps(
            diagnostics,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )

    if np.max(raw_relative_l2) > RAW_TIME_RELATIVE_L2_LIMIT:
        raise RuntimeError(
            "temporal extension raw spectra did not converge: relative L2 "
            f"{raw_relative_l2.tolist()}"
        )
    reciprocity._gate_scale_aware_pointwise(
        raw_scale_aware, "temporal extension raw spectra"
    )
    if np.max(normalized_relative_l2) > NORMALIZED_TIME_RELATIVE_L2_LIMIT:
        raise RuntimeError(
            "temporal extension normalized ratios did not converge: relative L2 "
            f"{normalized_relative_l2.tolist()}"
        )
    reciprocity._gate_scale_aware_pointwise(
        normalized_scale_aware, "temporal extension normalized ratios"
    )
    if np.max(normalized_symmetric) > POINTWISE_LIMIT:
        raise RuntimeError(
            "temporal extension normalized ratios fail the symmetric pointwise gate"
        )
    if closure_relative_l2[0] > CLOSURE_RELATIVE_L2_LIMIT:
        raise RuntimeError(
            f"resolution-{args.resolution} extended closure "
            f"{closure_relative_l2[0]} does not meet the 3% gate"
        )
    reciprocity._gate_scale_aware_pointwise(
        closure_scale_aware, f"resolution-{args.resolution} extended closure"
    )
    if np.max(closure_symmetric) > POINTWISE_LIMIT:
        raise RuntimeError("extended closure fails the symmetric pointwise gate")
    if refinement_ratio[0] > REFINEMENT_RATIO_LIMIT:
        raise RuntimeError(
            f"resolution-{args.resolution} refinement ratio "
            f"{refinement_ratio[0]} did not improve enough"
        )

    return {
        "reciprocity_temporal_extension_resolution": np.asarray(
            [args.resolution], dtype=float
        ),
        "reciprocity_temporal_extension_dipole_count": np.asarray(
            [dipole_count], dtype=float
        ),
        "reciprocity_temporal_extension_position": np.asarray([args.position]),
        "reciprocity_temporal_extension_source_index": np.asarray(
            [source_index], dtype=float
        ),
        "reciprocity_temporal_extension_metadata_index": np.asarray(
            [metadata_index], dtype=float
        ),
        "reciprocity_temporal_extension_frequencies": frequencies,
        "reciprocity_temporal_extension_reference_forward_flat": forward_flat,
        "reciprocity_temporal_extension_reference_backward_flat": backward_flat,
        "reciprocity_temporal_extension_reference_forward_textured": forward_prior,
        "reciprocity_temporal_extension_reference_backward_textured": backward_prior,
        "reciprocity_temporal_extension_forward_textured": forward_extended,
        "reciprocity_temporal_extension_backward_textured": backward_extended,
        "reciprocity_temporal_extension_raw_time_residual": raw_residual,
        "reciprocity_temporal_extension_raw_time_relative_l2": raw_relative_l2,
        "reciprocity_temporal_extension_raw_time_pointwise": raw_pointwise,
        "reciprocity_temporal_extension_raw_time_scale_aware": raw_scale_aware,
        "reciprocity_temporal_extension_reference_normalized": normalized_prior,
        "reciprocity_temporal_extension_extended_normalized": normalized_extended,
        "reciprocity_temporal_extension_normalized_time_residual": (
            normalized_residual
        ),
        "reciprocity_temporal_extension_normalized_time_relative_l2": (
            normalized_relative_l2
        ),
        "reciprocity_temporal_extension_normalized_time_pointwise": (
            normalized_pointwise
        ),
        "reciprocity_temporal_extension_normalized_time_scale_aware": (
            normalized_scale_aware
        ),
        "reciprocity_temporal_extension_normalized_time_symmetric": (
            normalized_symmetric
        ),
        "reciprocity_temporal_extension_closure_residual": closure_residual,
        "reciprocity_temporal_extension_closure_relative_l2": closure_relative_l2,
        "reciprocity_temporal_extension_closure_pointwise": closure_pointwise,
        "reciprocity_temporal_extension_closure_scale_aware": closure_scale_aware,
        "reciprocity_temporal_extension_closure_symmetric": closure_symmetric,
        "reciprocity_temporal_extension_reference_res50_relative_l2": np.asarray(
            [reference_res50]
        ),
        "reciprocity_temporal_extension_refinement_ratio": refinement_ratio,
        "reciprocity_temporal_extension_flat_condition_ratio": flat_condition,
        "reciprocity_temporal_extension_metadata_position_error": np.asarray(
            [validated["x"][metadata_index] - args.position]
        ),
        "reciprocity_temporal_extension_metadata_y_error": np.asarray(
            [y_metadata[0] - expected_y]
        ),
        "reciprocity_temporal_extension_metadata_z_error": z_metadata.copy(),
        "reciprocity_temporal_extension_metadata_local_weight": np.asarray(
            [local_weight]
        ),
        "reciprocity_temporal_extension_metadata_return_recompute_relative_l2": (
            return_recompute_relative_l2
        ),
        "reciprocity_temporal_extension_metadata_canonical_recompute_relative_l2": (
            canonical_recompute_relative_l2
        ),
        "reciprocity_temporal_extension_metadata_ghost_weight_max_abs": validated[
            "ghost_weight_max_abs"
        ],
        "reciprocity_temporal_extension_metadata_ghost_fraction": validated[
            "weighted_ghost_fraction"
        ],
    }


def run_temporal_extension(
    args: argparse.Namespace,
    *,
    run_path: Callable[..., dict[str, Any]] = runpy.run_path,
) -> dict[str, np.ndarray]:
    reference = _validate_reference(args.reference_diagnostics, args)
    namespace = run_path(str(args.example), run_name="gpmeep_reciprocity_target")
    forward = namespace.get("forward")
    backward = namespace.get("backward")
    if not callable(forward) or not callable(backward):
        raise RuntimeError("target does not expose forward and backward solvers")

    sx = _finite_scalar(namespace.get("sx"), "target period")
    sy = _finite_scalar(namespace.get("sy"), "target cell height")
    d_ag = _finite_scalar(namespace.get("dAg"), "target metal thickness")
    d_sub = _finite_scalar(namespace.get("dsub"), "target substrate thickness")
    if sx <= 0 or sy <= 0 or d_ag <= 0 or d_sub <= 0:
        raise RuntimeError("target geometry dimensions must be positive")
    grid_sample_count = sx * args.resolution
    dipole_count = int(round(grid_sample_count))
    if not np.isclose(grid_sample_count, dipole_count, rtol=0.0, atol=1e-9):
        raise RuntimeError("target period is not an integer number of samples")
    if dipole_count != 99 or dipole_count % 2 == 0:
        raise RuntimeError("temporal extension requires the odd 99-pixel period")
    grid_spacing = sx / dipole_count
    source_index = int(round((args.position + 0.5 * sx) / grid_spacing))
    source_x = sx * (-0.5 + source_index / dipole_count)
    if source_index != 27 or not np.isclose(
        source_x, args.position, rtol=0.0, atol=POSITION_ATOL
    ):
        raise RuntimeError("temporal extension position is not the shared grid point")
    expected_y = -0.5 * sy + d_ag + 0.5 * d_sub

    updates = {
        "resolution": args.resolution,
        "nfreq": args.frequencies,
        "ndipole": dipole_count,
    }
    with reciprocity._patched_target_globals((forward, backward), updates):
        return _run_extension(
            args,
            reference,
            forward,
            backward,
            sx,
            dipole_count,
            source_index,
            expected_y,
        )


def main(argv: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    return run_temporal_extension(parse_args(argv))


if __name__ == "__main__":
    globals().update(main())
