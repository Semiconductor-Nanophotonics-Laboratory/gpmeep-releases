#!/usr/bin/env python3
"""Launch one hash-bound stochastic-reciprocity stability diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from collections.abc import Sequence

import numpy as np

import run_example_oracle
import run_stochastic_reciprocity_temporal512 as temporal_reference


RESULT_SHAPES = {
    "reciprocity_stability_resolution": "1",
    "reciprocity_stability_runtime": "1",
    "reciprocity_stability_courant": "1",
    "reciprocity_stability_branch_code": "1",
    "reciprocity_stability_source_index": "1",
    "reciprocity_stability_frequencies": "4",
    "reciprocity_stability_reference_spectrum": "4",
    "reciprocity_stability_spectrum": "4",
    "reciprocity_stability_residual": "4",
    "reciprocity_stability_relative_l2": "1",
    "reciprocity_stability_symmetric": "4",
    "reciprocity_stability_metadata_coordinate_errors": "3",
}


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_courant(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0 or result > 0.5:
        raise argparse.ArgumentTypeError("must be finite and in (0, 0.5]")
    return result


def expected_timestep(runtime: int, courant: float) -> int:
    # GaussianSource(fwidth=0.2) ends at t=50; the requested post-source
    # duration is runtime * nfreq / df = runtime * 20 at resolution 90.
    exact = (50.0 + 20.0 * runtime) * 90.0 / courant
    rounded = int(round(exact))
    if not np.isclose(exact, rounded, rtol=0.0, atol=1e-9):
        raise RuntimeError("stability profile does not end on an exact timestep")
    return rounded


def build_oracle_argv(
    example: pathlib.Path,
    branch: str,
    runtime: int,
    courant: float,
    reference: dict,
) -> list[str]:
    wrapper = pathlib.Path(__file__).resolve().with_name(
        "stochastic_reciprocity_stability_point.py"
    )
    timestep = expected_timestep(runtime, courant)
    arguments = [
        "--expected-run-count",
        "1",
        "--expected-final-timestep",
        str(timestep),
        "--expected-run-timestep-delta-range",
        f"{timestep}:{timestep}",
        "--min-each-dft-norm",
        "1e-8",
    ]
    for name, shape in RESULT_SHAPES.items():
        arguments.extend(
            [
                "--result-vector",
                name,
                "--expected-result-shape",
                f"{name}={shape}",
            ]
        )
    for name in (
        "reciprocity_stability_frequencies",
        "reciprocity_stability_reference_spectrum",
        "reciprocity_stability_spectrum",
    ):
        arguments.extend(["--min-result-l2", f"{name}=1e-10"])
    arguments.extend(
        [
            "--max-result-abs",
            "reciprocity_stability_metadata_coordinate_errors=1e-12",
            str(wrapper),
            "--",
            str(example.resolve()),
            "--branch",
            branch,
            "--resolution",
            "90",
            "--frequencies",
            "4",
            "--runtime",
            str(runtime),
            "--courant",
            repr(courant),
            "--position",
            "-0.25",
            "--reference-diagnostics-json",
            json.dumps(
                reference,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )
    return arguments


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-log", required=True, type=pathlib.Path)
    parser.add_argument("--reference-source-freeze", required=True, type=pathlib.Path)
    parser.add_argument("--branch", choices=("forward", "backward"), required=True)
    parser.add_argument("--runtime", type=positive_int, required=True)
    parser.add_argument("--courant", type=positive_courant, default=0.5)
    parser.add_argument("example", type=pathlib.Path)
    args = parser.parse_args(argv)
    if not args.example.is_file() or args.example.suffix != ".py":
        parser.error(f"example is not a Python file: {args.example}")
    if not args.reference_log.is_file():
        parser.error(f"reference log is not a file: {args.reference_log}")
    if not args.reference_source_freeze.is_file():
        parser.error(
            f"reference source freeze is not a file: {args.reference_source_freeze}"
        )
    expected_timestep(args.runtime, args.courant)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reference, audit = temporal_reference.load_reference(
        args.reference_log, args.reference_source_freeze
    )
    profile = {
        "schema": "gpmeep-reciprocity-stability-reference-audit-v1",
        "reference": audit,
        "branch": args.branch,
        "runtime": args.runtime,
        "courant": args.courant,
        "expected_timestep": expected_timestep(args.runtime, args.courant),
    }
    if os.environ.get("OMPI_COMM_WORLD_RANK", "0") == "0":
        print(
            "gpmeep-reciprocity-stability-reference-audit:"
            + json.dumps(
                profile,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    return run_example_oracle.main(
        build_oracle_argv(
            args.example,
            args.branch,
            args.runtime,
            args.courant,
            reference,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
