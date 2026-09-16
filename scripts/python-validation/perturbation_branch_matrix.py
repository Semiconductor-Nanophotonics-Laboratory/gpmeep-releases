#!/usr/bin/env python3
"""Run both perturbation-theory polarization branches under one oracle."""

from __future__ import annotations

import argparse
import pathlib
import runpy
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    parser.add_argument("--resolution", type=positive_int, required=True)
    args = parser.parse_args(argv)
    if not args.example.is_file():
        raise RuntimeError(f"perturbation example does not exist: {args.example}")
    return args


def _array(
    namespace: dict[str, Any], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    if name not in namespace:
        raise RuntimeError(f"perturbation branch did not define {name!r}")
    value = np.asarray(namespace[name])
    if value.shape != shape:
        raise RuntimeError(f"perturbation {name} shape {value.shape} != {shape}")
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"perturbation {name} contains non-finite values")
    return value


def run_branch_matrix(
    args: argparse.Namespace,
    *,
    run_path: Callable[..., dict[str, Any]] = runpy.run_path,
) -> dict[str, np.ndarray]:
    frequencies = []
    derivatives = []
    integrals = []
    relative_errors = []
    mode_decays = []
    mode_q = []
    mode_amplitudes = []
    mode_errors = []
    polarizations = []
    previous_argv = sys.argv
    try:
        for perpendicular in (False, True):
            sys.argv = [str(args.example), "-res", str(args.resolution)]
            if perpendicular:
                sys.argv.append("-perpendicular")
            namespace = run_path(str(args.example), run_name="__main__")
            frequencies.append(_array(namespace, "perturbation_frequencies", (2,)))
            derivatives.append(_array(namespace, "perturbation_derivatives", (2,)))
            integrals.append(_array(namespace, "perturbation_integrals", (2,)))
            relative_errors.append(
                _array(namespace, "perturbation_relative_error", (1,))[0]
            )
            mode_decays.append(
                _array(namespace, "perturbation_mode_decays", (2,))
            )
            mode_q.append(_array(namespace, "perturbation_mode_q", (2,)))
            mode_amplitudes.append(
                _array(namespace, "perturbation_mode_amplitudes", (2,))
            )
            mode_errors.append(
                _array(namespace, "perturbation_mode_errors", (2,))
            )
            polarization = _array(namespace, "perturbation_polarization", (1,))[0]
            expected = 1 if perpendicular else 0
            if int(polarization) != expected:
                raise RuntimeError(
                    f"perturbation polarization marker {polarization} != {expected}"
                )
            polarizations.append(expected)
    finally:
        sys.argv = previous_argv

    frequency_matrix = np.stack(frequencies)
    derivative_matrix = np.stack(derivatives)
    integral_matrix = np.stack(integrals)
    error_vector = np.asarray(relative_errors, dtype=float)
    decay_matrix = np.stack(mode_decays)
    q_matrix = np.stack(mode_q)
    amplitude_matrix = np.stack(mode_amplitudes)
    mode_error_matrix = np.stack(mode_errors)
    if np.any(np.abs(derivative_matrix) <= 1e-10):
        raise RuntimeError("a perturbation branch produced a zero derivative")
    if np.any(np.real(integral_matrix[:, 1]) <= 0):
        raise RuntimeError("a perturbation branch produced non-positive energy")
    if np.any(error_vector > 0.35):
        raise RuntimeError(
            f"perturbation theory failed finite-difference closure: {error_vector}"
        )
    if np.any(decay_matrix >= 0) or np.any(q_matrix <= 0):
        raise RuntimeError("a perturbation branch produced a non-decaying mode")
    if np.any(np.abs(amplitude_matrix) <= 1e-12):
        raise RuntimeError("a perturbation branch produced a zero Harminv amplitude")
    if np.any(np.abs(mode_error_matrix) >= 1e-3):
        raise RuntimeError("a perturbation branch produced an inaccurate Harminv mode")
    q_from_decay = -frequency_matrix.real / (2 * decay_matrix)
    if not np.allclose(q_matrix, q_from_decay, rtol=1e-8, atol=1e-8):
        raise RuntimeError("a perturbation branch violated Harminv Q closure")
    return {
        "branch_polarizations": np.asarray(polarizations, dtype=np.int64),
        "branch_perturbation_frequencies": frequency_matrix,
        "branch_perturbation_derivatives": derivative_matrix,
        "branch_perturbation_integrals": integral_matrix,
        "branch_perturbation_relative_errors": error_vector,
        "branch_harminv_decays": decay_matrix,
        "branch_harminv_q": q_matrix,
        "branch_harminv_amplitudes": amplitude_matrix,
        "branch_harminv_errors": mode_error_matrix,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    return run_branch_matrix(parse_args(argv))


if __name__ == "__main__":
    globals().update(main())
