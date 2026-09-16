#!/usr/bin/env python3
"""Run one hash-comparable stochastic-reciprocity stability point."""

from __future__ import annotations

import argparse
import json
import pathlib
import runpy
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

import stochastic_reciprocity_matrix as reciprocity


POSITION_ATOL = 1e-12
DIAGNOSTICS_PREFIX = "gpmeep-reciprocity-stability-point:"


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_finite(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("--branch", choices=("forward", "backward"), required=True)
    parser.add_argument("--resolution", type=positive_int, default=90)
    parser.add_argument("--frequencies", type=positive_int, default=4)
    parser.add_argument("--runtime", type=positive_int, required=True)
    parser.add_argument("--courant", type=positive_finite, default=0.5)
    parser.add_argument("--position", type=float, default=-0.25)
    parser.add_argument("--reference-diagnostics-json", required=True)
    args = parser.parse_args(argv)
    if not args.example.is_file():
        raise RuntimeError(f"reciprocity example does not exist: {args.example}")
    if args.courant > 0.5:
        raise RuntimeError("stability probe Courant factor must not exceed 0.5")
    if not np.isfinite(args.position):
        raise RuntimeError("stability probe position must be finite")
    try:
        args.reference_diagnostics = json.loads(
            args.reference_diagnostics_json
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError("stability reference diagnostics JSON is invalid") from exc
    return args


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"{name} is not finite")
    return result


def run_stability_point(
    args: argparse.Namespace,
    *,
    run_path: Callable[..., dict[str, Any]] = runpy.run_path,
) -> dict[str, np.ndarray]:
    reference = args.reference_diagnostics
    if not isinstance(reference, dict) or (
        reference.get("schema")
        != "gpmeep-reciprocity-resolution-diagnostics-v1"
    ):
        raise RuntimeError("stability reference has the wrong schema")
    if (
        reference.get("resolution") != args.resolution
        or reference.get("dipole_count") != 99
        or reference.get("source_index") != 27
    ):
        raise RuntimeError("stability reference has the wrong profile")
    reference_frequencies = reciprocity._finite_array(
        reference.get("frequencies"),
        "stability reference frequencies",
        (args.frequencies,),
    )
    reference_name = f"{args.branch}_textured"
    reference_rows = reciprocity._finite_array(
        reference.get(reference_name),
        f"stability reference {reference_name}",
        (2, args.frequencies),
    )
    reference_spectrum = reference_rows[1]
    if np.min(reference_spectrum) <= 0:
        raise RuntimeError("stability reference spectrum is not positive")

    namespace = run_path(str(args.example), run_name="gpmeep_reciprocity_target")
    forward = namespace.get("forward")
    backward = namespace.get("backward")
    if not callable(forward) or not callable(backward):
        raise RuntimeError("target does not expose forward and backward solvers")
    sx = _finite_scalar(namespace.get("sx"), "target period")
    sy = _finite_scalar(namespace.get("sy"), "target height")
    d_ag = _finite_scalar(namespace.get("dAg"), "target metal thickness")
    d_sub = _finite_scalar(namespace.get("dsub"), "target substrate thickness")
    dipole_count = int(round(sx * args.resolution))
    if dipole_count != 99 or not np.isclose(
        sx * args.resolution, dipole_count, rtol=0.0, atol=1e-9
    ):
        raise RuntimeError("stability point requires the 99-pixel period")
    grid_spacing = sx / dipole_count
    source_index = int(round((args.position + 0.5 * sx) / grid_spacing))
    source_x = sx * (-0.5 + source_index / dipole_count)
    if source_index != 27 or not np.isclose(
        source_x, args.position, rtol=0.0, atol=POSITION_ATOL
    ):
        raise RuntimeError("stability point is not the shared grid point")

    updates = {
        "resolution": args.resolution,
        "nfreq": args.frequencies,
        "ndipole": dipole_count,
        "courant": args.courant,
    }
    metadata_index = source_index + 1
    metadata_errors = np.zeros(3)
    with reciprocity._patched_target_globals((forward, backward), updates):
        if args.branch == "forward":
            frequencies, spectrum = forward(source_index, args.runtime, True)
        else:
            result = backward(args.runtime, True, return_metadata=True)
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("backward stability point lacks metadata")
            frequencies, returned_spectrum, metadata = result
            if not isinstance(metadata, dict):
                raise RuntimeError("backward stability metadata is invalid")
            validated = reciprocity._validate_backward_metadata(
                metadata,
                "backward stability point",
                args.frequencies,
                dipole_count,
                sx,
            )
            matching = np.flatnonzero(
                np.isclose(
                    validated["x"],
                    args.position,
                    rtol=0.0,
                    atol=POSITION_ATOL,
                )
            )
            if matching.size != 1 or int(matching[0]) != metadata_index:
                raise RuntimeError("backward stability metadata index is wrong")
            returned = reciprocity._finite_array(
                returned_spectrum,
                "backward stability integrated spectrum",
                (args.frequencies,),
            )
            if not np.allclose(
                returned,
                validated["expected_recomputed_flux"],
                rtol=5e-8,
                atol=1e-12 * max(float(np.max(np.abs(returned))), 1.0),
            ):
                raise RuntimeError("backward stability cubature is not canonical")
            y = np.asarray(metadata.get("y"), dtype=float).reshape(-1)
            z = np.asarray(metadata.get("z"), dtype=float).reshape(-1)
            expected_y = -0.5 * sy + d_ag + 0.5 * d_sub
            if y.shape != (1,) or z.shape != (1,):
                raise RuntimeError("backward stability y/z metadata is invalid")
            metadata_errors = np.asarray(
                [
                    validated["x"][metadata_index] - args.position,
                    y[0] - expected_y,
                    z[0],
                ]
            )
            if np.max(np.abs(metadata_errors)) > POSITION_ATOL:
                raise RuntimeError("backward stability line coordinates are wrong")
            spectrum = validated["raw_power"][:, metadata_index]

    frequencies = reciprocity._finite_array(
        frequencies, "stability point frequencies", (args.frequencies,)
    )
    spectrum = reciprocity._finite_array(
        spectrum, "stability point spectrum", (args.frequencies,)
    )
    if np.min(spectrum) <= 0:
        raise RuntimeError("stability point spectrum is not positive")
    if not np.allclose(
        frequencies, reference_frequencies, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("stability point frequency grid differs from reference")
    residual = spectrum - reference_spectrum
    relative_l2 = np.asarray(
        [
            np.linalg.norm(residual)
            / max(np.linalg.norm(reference_spectrum), np.finfo(float).tiny)
        ]
    )
    symmetric = 2.0 * np.abs(residual) / np.maximum(
        np.abs(spectrum) + np.abs(reference_spectrum), np.finfo(float).tiny
    )
    diagnostics = {
        "schema": "gpmeep-reciprocity-stability-point-v1",
        "branch": args.branch,
        "resolution": args.resolution,
        "runtime": args.runtime,
        "courant": args.courant,
        "source_index": source_index,
        "metadata_index": metadata_index if args.branch == "backward" else None,
        "frequencies": frequencies.tolist(),
        "reference_runtime": 256,
        "reference_spectrum": reference_spectrum.tolist(),
        "spectrum": spectrum.tolist(),
        "relative_l2": float(relative_l2[0]),
        "symmetric_max": float(np.max(symmetric)),
        "metadata_coordinate_errors": metadata_errors.tolist(),
    }
    target_mp = namespace.get("mp")
    is_master = (
        bool(target_mp.am_master())
        if target_mp is not None and hasattr(target_mp, "am_master")
        else True
    )
    if is_master:
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
    return {
        "reciprocity_stability_resolution": np.asarray(
            [args.resolution], dtype=float
        ),
        "reciprocity_stability_runtime": np.asarray([args.runtime], dtype=float),
        "reciprocity_stability_courant": np.asarray([args.courant]),
        "reciprocity_stability_branch_code": np.asarray(
            [0.0 if args.branch == "forward" else 1.0]
        ),
        "reciprocity_stability_source_index": np.asarray(
            [source_index], dtype=float
        ),
        "reciprocity_stability_frequencies": frequencies,
        "reciprocity_stability_reference_spectrum": reference_spectrum,
        "reciprocity_stability_spectrum": spectrum,
        "reciprocity_stability_residual": residual,
        "reciprocity_stability_relative_l2": relative_l2,
        "reciprocity_stability_symmetric": symmetric,
        "reciprocity_stability_metadata_coordinate_errors": metadata_errors,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    return run_stability_point(parse_args(argv))


if __name__ == "__main__":
    globals().update(main())
