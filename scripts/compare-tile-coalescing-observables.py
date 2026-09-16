#!/usr/bin/env python3
"""Require exact physics for coalesced and legacy CUDA loop-tile runs."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_tile_coalescing_comparator_base",
    SCRIPT_DIR / "compare-boundary-eh-overlap-observables.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load boundary-overlap comparator")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

TILE_PREFIX = "gpmeep-tile-coalescing-v1:"
PHYSICAL_KEYS = (
    "cut_probe_values",
    "cut_probe_l2",
    "ez_l2",
    "ez_weighted_checksum",
    "energy",
    "dft_norm",
    "remote_cut_scalars",
    "remote_cut_l2",
    "mpi_messages",
    "mpi_scalars",
    "cuda_aware_bytes",
    "pinned_bytes",
)
TILE_KEYS = (
    "curl_chunk_phases",
    "curl_input_tiles",
    "update_eh_chunk_phases",
    "update_eh_input_tiles",
)


def validate_counters(record: dict[str, Any], *, enabled: bool) -> None:
    counters = BASE.selected(record, TILE_KEYS)
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError("tile-coalescing counters must be nonnegative integers")
    phase_tile_pairs = (
        (counters["curl_chunk_phases"], counters["curl_input_tiles"]),
        (
            counters["update_eh_chunk_phases"],
            counters["update_eh_input_tiles"],
        ),
    )
    if enabled:
        if phase_tile_pairs[0][0] <= 0:
            raise ValueError("enabled tile-coalescing curl proof is incomplete")
        for phases, input_tiles in phase_tile_pairs:
            if (phases == 0) != (input_tiles == 0) or (
                phases > 0 and input_tiles <= phases
            ):
                raise ValueError("enabled tile-coalescing dispatch proof is incomplete")
    elif any(counters.values()):
        raise ValueError("disabled tile-coalescing path unexpectedly dispatched")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("complete", "mixed"), default="mixed"
    )
    parser.add_argument("enabled", type=pathlib.Path)
    parser.add_argument("disabled", type=pathlib.Path)
    args = parser.parse_args()
    enabled_result = BASE.one_record(args.enabled, BASE.RESULT_PREFIX)
    disabled_result = BASE.one_record(args.disabled, BASE.RESULT_PREFIX)
    BASE.validate_initialization_records(args.enabled, enabled_result)
    BASE.validate_initialization_records(args.disabled, disabled_result)
    BASE.validate_fixed_workload(enabled_result, args.profile)
    BASE.validate_fixed_workload(disabled_result, args.profile)
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
        raise ValueError("tile-coalescing comparison recorded CPU fallback")
    validate_counters(BASE.one_record(args.enabled, TILE_PREFIX), enabled=True)
    validate_counters(BASE.one_record(args.disabled, TILE_PREFIX), enabled=False)
    print("PASS: tile-coalesced and legacy CUDA physical observables are exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
