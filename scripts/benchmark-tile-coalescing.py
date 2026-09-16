#!/usr/bin/env python3
"""Receipt-bound 2-GPU A/B for CUDA loop-tile coalescing.

The fixed profile compares the default coalesced CUDA execution with the
same-binary legacy per-CPU-tile schedule and a no-tile reference.  Every
sample must retain exact physical observables and exclusive CUDA-aware work.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import statistics
import sys
from typing import Any


SCRIPT = pathlib.Path(__file__).resolve()
BASE_PATH = SCRIPT.parent / "benchmark-boundary-eh-overlap.py"
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_tile_coalescing_evidence_base", BASE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load boundary-overlap evidence base")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

PROFILE_ID = "m8.13-cuda-tile-coalescing-ab-v1"
PROFILE = {
    "profile_id": PROFILE_ID,
    "mpi_ranks": 2,
    "pixels": 192,
    "warmup_steps": 12,
    "measured_steps": 80,
    "warmup_processes_per_condition": 1,
    "measured_pairs": 6,
    "transport": "cuda-aware",
    "completion_policy": "waitsome",
    "initial_condition": "trigonometric-v1",
    "source_enabled": False,
    "overlap_material": True,
    "loop_tile_base_db": 128,
    "reference_loop_tile_base_db": 0,
    "minimum_median_paired_speedup": 8.0,
    "minimum_faster_pairs": 6,
    "maximum_t90_upper_time_ratio": 0.20,
    "both_position_strata_must_be_positive": True,
    "maximum_reference_median_time_ratio": 1.10,
    "maximum_reference_t90_upper_time_ratio": 1.15,
    "maximum_reference_pair_time_ratio": 1.15,
    "minimum_reference_pairs_within_limit": 5,
}
PROFILE_SHA256 = "c2c9f384233b7e3cdbfd31186bd411b05e47927a7cc3bf217c067771dd914121"

TILE_PREFIX = "gpmeep-tile-coalescing-v1:"


def expected_schedule() -> list[tuple[str, int, int, str]]:
    schedule = [
        ("warmup", 0, 1, "enabled"),
        ("warmup", 0, 2, "disabled"),
        ("warmup", 0, 3, "reference"),
    ]
    for pair in range(1, PROFILE["measured_pairs"] + 1):
        order = (
            ("enabled", "disabled")
            if pair % 2
            else ("disabled", "enabled")
        )
        schedule.extend(
            ("measured", pair, position, condition)
            for position, condition in enumerate(order, start=1)
        )
        schedule.append(("reference", pair, 3, "reference"))
    return schedule


def validate_tile_coalescing(condition: str, record: dict[str, Any]) -> None:
    keys = (
        "curl_chunk_phases",
        "curl_input_tiles",
        "update_eh_chunk_phases",
        "update_eh_input_tiles",
    )
    if any(type(record.get(key)) is not int or record[key] < 0 for key in keys):
        raise RuntimeError("tile-coalescing counter record is incomplete")
    if condition == "enabled":
        phase_tile_pairs = (
            (
                record["curl_chunk_phases"],
                record["curl_input_tiles"],
            ),
            (
                record["update_eh_chunk_phases"],
                record["update_eh_input_tiles"],
            ),
        )
        if phase_tile_pairs[0][0] <= 0 or any(
            (phases == 0) != (input_tiles == 0)
            or (phases > 0 and input_tiles <= phases)
            for phases, input_tiles in phase_tile_pairs
        ):
            raise RuntimeError("default tile coalescing did not execute")
    elif condition in {"disabled", "reference"}:
        if any(record[key] != 0 for key in keys):
            raise RuntimeError("non-coalesced control unexpectedly coalesced tiles")
    else:
        raise RuntimeError(f"unknown tile-coalescing condition: {condition}")


def condition_environment(
    base_environment: dict[str, str], condition: str
) -> dict[str, str]:
    environment = dict(base_environment)
    for key in (
        "MEEP_GPU_DISABLE_TILE_COALESCING",
        "MEEP_GPU_EXPECT_TILE_COALESCING",
        "MEEP_GPU_EXPECT_NO_TILE_COALESCING",
    ):
        environment.pop(key, None)
    if condition == "enabled":
        environment["MEEP_GPU_EXPECT_TILE_COALESCING"] = "1"
    elif condition == "disabled":
        environment["MEEP_GPU_DISABLE_TILE_COALESCING"] = "1"
        environment["MEEP_GPU_EXPECT_NO_TILE_COALESCING"] = "1"
    elif condition == "reference":
        environment["MEEP_GPU_MULTI_LOOP_TILE_BASE_DB"] = str(
            PROFILE["reference_loop_tile_base_db"]
        )
        environment["MEEP_GPU_EXPECT_NO_TILE_COALESCING"] = "1"
    else:
        raise RuntimeError(f"unknown tile-coalescing condition: {condition}")
    return environment


def validate_sample(condition: str, output: str) -> dict[str, Any]:
    BASE.HELPER.reject_unknown_gpmeep_markers(output)
    benchmark = BASE.HELPER.exactly_one(
        output, BASE.BENCHMARK_PREFIX, "benchmark"
    )
    BASE.HELPER.validate_initialization_markers(
        output, benchmark, PROFILE["initial_condition"]
    )
    tile = BASE.HELPER.exactly_one(output, TILE_PREFIX, "tile coalescing")
    eager = BASE.HELPER.exactly_one(output, BASE.EAGER_PREFIX, "eager MPI")
    pingpong = BASE.HELPER.exactly_one(
        output, BASE.PINGPONG_PREFIX, "receive ping-pong"
    )
    graph = BASE.HELPER.exactly_one(
        output, BASE.GRAPH_PREFIX, "boundary graph"
    )
    dft = BASE.HELPER.exactly_one(output, BASE.DFT_PREFIX, "DFT batch")
    devices = BASE.HELPER.records(output, BASE.DEVICE_PREFIX)
    if (
        len(devices) != PROFILE["mpi_ranks"]
        or any(not isinstance(item, dict) for item in devices)
        or {item.get("rank") for item in devices} != {0, 1}
        or len({item.get("uuid") for item in devices}) != 2
        or any(not item.get("uuid") for item in devices)
    ):
        raise RuntimeError("sample did not prove two distinct physical GPUs")
    expected_tile = (
        PROFILE["reference_loop_tile_base_db"]
        if condition == "reference"
        else PROFILE["loop_tile_base_db"]
    )
    identity = {
        "schema_version": 4,
        "mpi_ranks": PROFILE["mpi_ranks"],
        "pixels": PROFILE["pixels"],
        "cells": PROFILE["pixels"] ** 3,
        "warmup_steps": PROFILE["warmup_steps"],
        "steps": PROFILE["measured_steps"],
        "loop_tile_base_db": expected_tile,
        "bfast": False,
        "source_enabled": PROFILE["source_enabled"],
        "overlap_material": PROFILE["overlap_material"],
        "requested_transport": PROFILE["transport"],
        "selected_transport": PROFILE["transport"],
        "completion_policy": PROFILE["completion_policy"],
        "initial_condition": PROFILE["initial_condition"],
    }
    if any(benchmark.get(key) != value for key, value in identity.items()):
        raise RuntimeError("benchmark workload identity differs from the profile")
    if (
        benchmark.get("cpu_calls") != 0
        or type(benchmark.get("cuda_calls")) is not int
        or benchmark["cuda_calls"] <= 0
        or benchmark.get("pinned_bytes") != 0
        or benchmark.get("cuda_aware_bytes", 0) <= 0
    ):
        raise RuntimeError("sample did not use exclusive CUDA-aware GPU work")
    if any(type(eager.get(key)) is not int or eager[key] <= 0 for key in eager):
        raise RuntimeError("sample did not prove eager MPI request posting")
    if (
        pingpong.get("secondary_selections", 0) <= 0
        or graph.get("launches", 0) <= 0
        or dft.get("submitted_updates", 0) <= 0
    ):
        raise RuntimeError("sample omitted a fixed CUDA scheduling path")
    validate_tile_coalescing(condition, tile)
    return {
        "benchmark": benchmark,
        "tile_coalescing": tile,
        "eager": eager,
        "pingpong": pingpong,
        "graph": graph,
        "dft": dft,
        "devices": sorted(devices, key=lambda item: item["rank"]),
    }


ORIGINAL_SUMMARIZE = BASE.summarize


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    device_mappings = []
    for sample in samples:
        devices = sample.get("records", {}).get("devices")
        if not isinstance(devices, list) or len(devices) != PROFILE["mpi_ranks"]:
            raise RuntimeError("sample omitted the physical GPU mapping")
        try:
            mapping = tuple((item["rank"], item["uuid"]) for item in devices)
        except (KeyError, TypeError) as exc:
            raise RuntimeError("sample has an invalid physical GPU mapping") from exc
        device_mappings.append(mapping)
    if not device_mappings or any(
        mapping != device_mappings[0] for mapping in device_mappings[1:]
    ):
        raise RuntimeError("rank-to-GPU UUID mapping changed between samples")
    summary = ORIGINAL_SUMMARIZE(samples)
    coalesced_over_reference: list[float] = []
    reference_pairs_within_limit = 0
    for pair in range(1, PROFILE["measured_pairs"] + 1):
        coalesced = next(
            sample
            for sample in samples
            if sample["kind"] == "measured"
            and sample["pair"] == pair
            and sample["condition"] == "enabled"
        )
        reference = next(
            sample
            for sample in samples
            if sample["kind"] == "reference" and sample["pair"] == pair
        )
        ratio = float(coalesced["records"]["benchmark"]["seconds"]) / float(
            reference["records"]["benchmark"]["seconds"]
        )
        if not math.isfinite(ratio) or ratio <= 0:
            raise RuntimeError("invalid coalesced/reference timing ratio")
        coalesced_over_reference.append(ratio)
        if ratio <= PROFILE["maximum_reference_pair_time_ratio"]:
            reference_pairs_within_limit += 1
    reference_interval = BASE.HELPER.paired_log_interval(
        coalesced_over_reference
    )
    reference_checks = {
        "reference_median_within_limit": statistics.median(
            coalesced_over_reference
        )
        <= PROFILE["maximum_reference_median_time_ratio"],
        "reference_t90_upper_within_limit": reference_interval[
            "t90_upper_time_ratio"
        ]
        <= PROFILE["maximum_reference_t90_upper_time_ratio"],
        "reference_pair_count_within_limit": reference_pairs_within_limit
        >= PROFILE["minimum_reference_pairs_within_limit"],
    }
    summary["promotion_checks"].update(reference_checks)
    summary["promote_default"] = all(summary["promotion_checks"].values())
    summary.update(
        rank_device_uuid_mapping=[
            {"rank": rank, "uuid": uuid}
            for rank, uuid in device_mappings[0]
        ],
        coalesced_over_reference_time_ratios=coalesced_over_reference,
        median_coalesced_over_reference_time_ratio=statistics.median(
            coalesced_over_reference
        ),
        reference_t90_lower_time_ratio=reference_interval[
            "t90_lower_time_ratio"
        ],
        reference_t90_upper_time_ratio=reference_interval[
            "t90_upper_time_ratio"
        ],
        reference_pairs_within_limit=reference_pairs_within_limit,
        decision=(
            "retain-gpu-tile-coalescing-default"
            if summary["promote_default"]
            else "disable-gpu-tile-coalescing-default"
        ),
    )
    return summary


def markdown_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# CUDA loop-tile coalescing paired A/B",
            "",
            f"- Decision: `{summary['decision']}`",
            "- Legacy/default median speedup: "
            f"`{summary['median_paired_speedup']:.6f}x`",
            "- Default/no-tile median time ratio: "
            f"`{summary['median_coalesced_over_reference_time_ratio']:.6f}`",
            f"- Faster legacy pairs: `{summary['faster_pairs']}/{summary['measured_pair_count']}`",
            f"- Exact physical observables: `{summary['all_observables_exact']}`",
            "",
        )
    )


BASE.__file__ = str(SCRIPT)
BASE.PROFILE_ID = PROFILE_ID
BASE.PROFILE = PROFILE
BASE.PROFILE_SHA256 = PROFILE_SHA256
BASE.ADDITIONAL_RUNNER_SOURCES = {
    "benchmark-boundary-eh-overlap.py": BASE_PATH,
}
BASE.OVERLAP_PREFIX = TILE_PREFIX
BASE.OBSERVABLE_KEYS = (
    "cut_probe_values",
    "cut_probe_l2",
    "ez_l2",
    "ez_weighted_checksum",
    "energy",
    "dft_norm",
    "remote_cut_scalars",
    "remote_cut_l2",
    "cpu_calls",
    "cuda_calls",
    "phase_calls",
    "h2d_bytes",
    "d2h_bytes",
    "mpi_messages",
    "mpi_scalars",
    "cuda_aware_bytes",
    "pinned_bytes",
)
BASE.expected_schedule = expected_schedule
BASE.validate_overlap = validate_tile_coalescing
BASE.validate_sample = validate_sample
BASE.condition_environment = condition_environment
BASE.summarize = summarize
BASE.markdown_report = markdown_report


if __name__ == "__main__":
    raise SystemExit(BASE.main())
