#!/usr/bin/env python3
"""Launch the bounded deterministic point-dipole cylindrical profile."""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

import run_example_oracle


EXPECTED_TIMESTEP_DELTAS = (1920, 640, 640) * 3
EXPECTED_FINAL_TIMESTEP = 3200

RESULT_SHAPES = {
    "point_dipole_m_values": "3",
    "point_dipole_parameters": "7",
    "point_dipole_checkpoint_after_sources": "3",
    "point_dipole_checkpoint_timesteps": "3,3",
    "point_dipole_checkpoint_meep_times": "3,3",
    "point_dipole_radiated_fluxes": "3,3",
    "point_dipole_source_fluxes": "3,3",
    "point_dipole_ldos_values": "3,3",
    "point_dipole_ldos_fdata": "3,3",
    "point_dipole_ldos_jdata": "3,3",
    "point_dipole_dft_norms": "3,3",
    "point_dipole_extraction_efficiencies": "3,3",
    "point_dipole_pm_symmetry_relative_errors": "4,3",
    "point_dipole_tail_signed_corrections": "4,3",
    "point_dipole_tail_relative_changes": "4,3",
    "point_dipole_tail_contraction_ratios": "4,3",
    "point_dipole_final_mode_ratios": "4,3",
    "point_dipole_m0_contrast": "4",
}

MINIMUM_L2 = {
    "point_dipole_m_values": 1.0,
    "point_dipole_parameters": 1.0,
    "point_dipole_checkpoint_after_sources": 1.0,
    "point_dipole_checkpoint_timesteps": 1.0,
    "point_dipole_checkpoint_meep_times": 1.0,
    "point_dipole_radiated_fluxes": 1e-8,
    "point_dipole_source_fluxes": 1e-8,
    "point_dipole_ldos_values": 1e-8,
    "point_dipole_ldos_fdata": 1e-8,
    "point_dipole_ldos_jdata": 1e-8,
    "point_dipole_dft_norms": 1e-8,
    "point_dipole_extraction_efficiencies": 1e-8,
    "point_dipole_final_mode_ratios": 1e-3,
    "point_dipole_m0_contrast": 0.02,
}

MAXIMUM_ABS = {
    "point_dipole_extraction_efficiencies": 1.05,
    "point_dipole_pm_symmetry_relative_errors": 2e-3,
    "point_dipole_tail_signed_corrections": 2e-6,
    "point_dipole_tail_relative_changes": 0.08,
    "point_dipole_tail_contraction_ratios": 0.01,
}


def build_oracle_argv(example: pathlib.Path) -> list[str]:
    """Returns the complete generic-oracle contract for this profile."""
    arguments = [
        "--expected-run-count",
        "9",
        "--expected-final-timestep",
        str(EXPECTED_FINAL_TIMESTEP),
        "--min-dft-norm",
        "1e-8",
        "--min-each-dft-norm",
        "1e-10",
    ]
    for timestep_delta in EXPECTED_TIMESTEP_DELTAS:
        arguments.extend(
            [
                "--expected-run-timestep-delta-range",
                f"{timestep_delta}:{timestep_delta}",
            ]
        )
    for name, shape in RESULT_SHAPES.items():
        arguments.extend(
            [
                "--result-vector",
                name,
                "--expected-result-shape",
                f"{name}={shape}",
            ]
        )
    for name, threshold in MINIMUM_L2.items():
        arguments.extend(["--min-result-l2", f"{name}={threshold}"])
    for name, threshold in MAXIMUM_ABS.items():
        arguments.extend(["--max-result-abs", f"{name}={threshold}"])
    arguments.extend([str(example.resolve()), "--", "--validation"])
    return arguments


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=pathlib.Path)
    args = parser.parse_args(argv)
    if not args.example.is_file() or args.example.name != "point_dipole_cyl.py":
        parser.error(f"not the point_dipole_cyl.py example: {args.example}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_example_oracle.main(build_oracle_argv(args.example))


if __name__ == "__main__":
    raise SystemExit(main())
