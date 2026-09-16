#!/usr/bin/env python3
"""Require exact physical observables for halo/curl enabled and disabled runs."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_boundary_overlap_comparator_base",
    SCRIPT_DIR / "compare-boundary-eh-overlap-observables.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load boundary-overlap comparator")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

HALO_PREFIX = "gpmeep-halo-curl-overlap-v1:"
PHYSICAL_KEYS = (
    "cut_probe_values",
    "cut_probe_l2",
    "ez_l2",
    "ez_weighted_checksum",
    "energy",
    "dft_norm",
    "remote_cut_scalars",
    "remote_cut_l2",
    # Splitting one curl into interior and shell phases intentionally creates
    # different CUDA plan/cache traffic.  H2D/D2H counters are performance
    # diagnostics, not physical observables, and therefore must not make an
    # otherwise exact enabled/disabled comparison fail.  MPI work remains an
    # invariant: overlap must not change the communicated domain.
    "mpi_messages",
    "mpi_scalars",
    "cuda_aware_bytes",
    "pinned_bytes",
)


def validate_counters(
    record: dict[str, Any], *, enabled: bool, expected_checks: int
) -> None:
    keys = (
        "checks",
        "eligible",
        "launches",
        "skipped_disabled",
        "skipped_unsupported_schedule",
        "skipped_no_remote",
        "skipped_cold_topology",
        "rejected_feature",
        "rejected_small",
        "full_points",
        "interior_points",
        "shell_points",
    )
    values = BASE.selected(record, keys)
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise ValueError("halo/curl counters must be nonnegative integers")
    classified = sum(
        values[key]
        for key in (
            "eligible",
            "skipped_disabled",
            "skipped_unsupported_schedule",
            "skipped_no_remote",
            "skipped_cold_topology",
            "rejected_feature",
            "rejected_small",
        )
    )
    if values["checks"] != expected_checks or classified != expected_checks:
        raise ValueError("halo/curl checks are not classified exactly once")
    if enabled:
        if (
            values["eligible"] != expected_checks
            or values["launches"] != expected_checks
            or values["interior_points"] + values["shell_points"]
            != values["full_points"]
            or min(
                values["full_points"],
                values["interior_points"],
                values["shell_points"],
            )
            <= 0
            or any(
                values[key] != 0
                for key in (
                    "skipped_disabled",
                    "skipped_unsupported_schedule",
                    "skipped_no_remote",
                    "skipped_cold_topology",
                    "rejected_feature",
                    "rejected_small",
                )
            )
        ):
            raise ValueError("enabled halo/curl dispatch proof is incomplete")
    elif not (
        values["eligible"] == 0
        and values["launches"] == 0
        and values["skipped_disabled"] == expected_checks
        and all(
            values[key] == 0
            for key in (
                "skipped_unsupported_schedule",
                "skipped_no_remote",
                "skipped_cold_topology",
                "rejected_feature",
                "rejected_small",
                "full_points",
                "interior_points",
                "shell_points",
            )
        )
    ):
        raise ValueError("disabled halo/curl path unexpectedly dispatched")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("enabled", type=pathlib.Path)
    parser.add_argument("disabled", type=pathlib.Path)
    args = parser.parse_args()
    enabled_result = BASE.one_record(args.enabled, BASE.RESULT_PREFIX)
    disabled_result = BASE.one_record(args.disabled, BASE.RESULT_PREFIX)
    BASE.validate_initialization_records(args.enabled, enabled_result)
    BASE.validate_initialization_records(args.disabled, disabled_result)
    BASE.validate_fixed_workload(enabled_result, "complete")
    BASE.validate_fixed_workload(disabled_result, "complete")
    identity_keys = BASE.IDENTITY_KEYS + ("loop_tile_base_db",)
    if BASE.selected(enabled_result, identity_keys) != BASE.selected(
        disabled_result, identity_keys
    ):
        raise ValueError("enabled/disabled workload identities differ")
    if BASE.selected(enabled_result, PHYSICAL_KEYS) != BASE.selected(
        disabled_result, PHYSICAL_KEYS
    ):
        raise ValueError("enabled/disabled physical observables differ")
    if enabled_result.get("cpu_calls") != 0 or disabled_result.get("cpu_calls") != 0:
        raise ValueError("halo/curl comparison recorded CPU fallback")
    expected_checks = enabled_result["steps"] * enabled_result["mpi_ranks"]
    validate_counters(
        BASE.one_record(args.enabled, HALO_PREFIX),
        enabled=True,
        expected_checks=expected_checks,
    )
    validate_counters(
        BASE.one_record(args.disabled, HALO_PREFIX),
        enabled=False,
        expected_checks=expected_checks,
    )
    print("PASS: halo/curl enabled and disabled physical observables are exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
