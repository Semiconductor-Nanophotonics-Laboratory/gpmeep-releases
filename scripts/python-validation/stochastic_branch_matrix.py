#!/usr/bin/env python3
"""Exercise every stochastic-emitter algorithm/geometry branch.

The generic example oracle executes this wrapper and observes every nested
``Simulation.run`` call.  Each target-example execution gets a deterministic,
branch-specific seed, while this wrapper retains the complete spectra and
derived observables for CPU/CUDA comparison.
"""

from __future__ import annotations

import argparse
import pathlib
import runpy
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


BRANCHES = (
    (1, False),
    (1, True),
    (2, False),
    (2, True),
)


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("--resolution", type=positive_int, default=50)
    parser.add_argument("--trials", type=positive_int, default=2)
    parser.add_argument("--dipoles", type=positive_int, default=2)
    parser.add_argument("--frequencies", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args(argv)
    if args.trials != args.dipoles:
        raise RuntimeError(
            "branch-matrix trials and dipoles must match so every spectrum "
            "has one independently pinned rectangular shape"
        )
    if not args.example.is_file():
        raise RuntimeError(f"stochastic-emitter example does not exist: {args.example}")
    return args


def _require_array(
    namespace: dict[str, Any],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in namespace:
        raise RuntimeError(f"stochastic-emitter branch did not define {name!r}")
    value = np.asarray(namespace[name])
    if value.shape != shape:
        raise RuntimeError(
            f"stochastic-emitter {name} has shape {value.shape}, expected {shape}"
        )
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"stochastic-emitter {name} contains non-finite values")
    return value


def run_branch_matrix(
    args: argparse.Namespace,
    *,
    run_path: Callable[..., dict[str, Any]] = runpy.run_path,
) -> dict[str, np.ndarray]:
    branch_frequencies = []
    branch_fluxes = []
    branch_mean_fluxes = []
    branch_standard_deviation_fluxes = []
    branch_trial_flux_l2 = []
    branch_source_callback_counts = []
    branch_active_callback_counts = []
    branch_validation_seeds = []
    previous_argv = sys.argv
    try:
        for branch_index, (method, textured) in enumerate(BRANCHES):
            branch_seed = args.seed + branch_index * 100_000
            target_argv = [
                str(args.example),
                "-res",
                str(args.resolution),
                "-nr",
                str(args.trials),
                "-nd",
                str(args.dipoles),
                "-nf",
                str(args.frequencies),
                "-method",
                str(method),
                "-seed",
                str(branch_seed),
                "--no-save",
            ]
            if textured:
                target_argv.append("-textured")
            sys.argv = target_argv
            namespace = run_path(str(args.example), run_name="__main__")

            sample_count = args.trials if method == 1 else args.dipoles
            frequencies = _require_array(
                namespace, "freqs", (args.frequencies,)
            ).astype(float, copy=False)
            fluxes = _require_array(
                namespace, "fluxes", (args.frequencies, sample_count)
            ).astype(float, copy=False)
            mean_flux = _require_array(
                namespace, "mean_flux", (args.frequencies,)
            ).astype(float, copy=False)
            standard_deviation_flux = _require_array(
                namespace,
                "standard_deviation_flux",
                (args.frequencies,),
            ).astype(float, copy=False)
            trial_flux_l2 = _require_array(
                namespace, "trial_flux_l2", (sample_count,)
            ).astype(float, copy=False)
            callback_counts = _require_array(
                namespace,
                "source_callback_counts",
                (args.trials * args.dipoles if method == 1 else 0,),
            ).astype(np.int64, copy=False)
            validation_seed = int(namespace.get("validation_seed", -1))

            if not np.all(np.diff(frequencies) > 0):
                raise RuntimeError("stochastic-emitter frequencies are not increasing")
            if not np.any(np.abs(fluxes) > 0) or np.any(trial_flux_l2 <= 1e-10):
                raise RuntimeError("stochastic-emitter branch produced no spectrum")
            if not np.allclose(mean_flux, np.mean(fluxes, axis=1)):
                raise RuntimeError("stochastic-emitter mean does not close")
            if not np.allclose(
                standard_deviation_flux, np.std(fluxes, axis=1)
            ):
                raise RuntimeError("stochastic-emitter deviation does not close")
            if not np.allclose(trial_flux_l2, np.linalg.norm(fluxes, axis=0)):
                raise RuntimeError("stochastic-emitter trial norm does not close")
            if validation_seed != branch_seed:
                raise RuntimeError(
                    f"stochastic-emitter seed {validation_seed} != {branch_seed}"
                )
            if method == 1 and np.any(callback_counts <= 0):
                raise RuntimeError("CustomSource callbacks did not execute")
            if method == 2 and callback_counts.size:
                raise RuntimeError("GaussianSource branch unexpectedly used callbacks")

            padded_callbacks = np.zeros(args.trials * args.dipoles, dtype=np.int64)
            padded_callbacks[: callback_counts.size] = callback_counts
            branch_frequencies.append(frequencies)
            branch_fluxes.append(fluxes)
            branch_mean_fluxes.append(mean_flux)
            branch_standard_deviation_fluxes.append(standard_deviation_flux)
            branch_trial_flux_l2.append(trial_flux_l2)
            branch_source_callback_counts.append(padded_callbacks)
            branch_active_callback_counts.append(callback_counts.size)
            branch_validation_seeds.append(validation_seed)
    finally:
        sys.argv = previous_argv

    frequency_matrix = np.stack(branch_frequencies)
    if not np.allclose(frequency_matrix, frequency_matrix[0]):
        raise RuntimeError("stochastic-emitter branches used different frequency grids")
    flux_cube = np.stack(branch_fluxes)
    mean_matrix = np.stack(branch_mean_fluxes)
    texture_relative_differences = np.asarray(
        [
            np.linalg.norm(mean_matrix[textured] - mean_matrix[flat])
            / max(np.linalg.norm(mean_matrix[flat]), 1e-30)
            for flat, textured in ((0, 1), (2, 3))
        ]
    )
    method_relative_differences = np.asarray(
        [
            np.linalg.norm(mean_matrix[method2] - mean_matrix[method1])
            / max(np.linalg.norm(mean_matrix[method1]), 1e-30)
            for method1, method2 in ((0, 2), (1, 3))
        ]
    )
    if np.any(texture_relative_differences <= 1e-6):
        raise RuntimeError("textured geometry did not measurably change either method")
    if np.any(method_relative_differences <= 1e-6):
        raise RuntimeError("method 1 and method 2 produced indistinguishable spectra")

    return {
        "branch_codes": np.asarray([10, 11, 20, 21], dtype=np.int64),
        "branch_methods": np.asarray([1, 1, 2, 2], dtype=np.int64),
        "branch_textured": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "branch_frequencies": frequency_matrix,
        "branch_fluxes": flux_cube,
        "branch_mean_fluxes": mean_matrix,
        "branch_standard_deviation_fluxes": np.stack(
            branch_standard_deviation_fluxes
        ),
        "branch_trial_flux_l2": np.stack(branch_trial_flux_l2),
        "branch_source_callback_counts": np.stack(
            branch_source_callback_counts
        ),
        "branch_active_callback_counts": np.asarray(
            branch_active_callback_counts, dtype=np.int64
        ),
        "branch_validation_seeds": np.asarray(
            branch_validation_seeds, dtype=np.int64
        ),
        "texture_relative_differences": texture_relative_differences,
        "method_relative_differences": method_relative_differences,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    return run_branch_matrix(parse_args(argv))


if __name__ == "__main__":
    globals().update(main())
