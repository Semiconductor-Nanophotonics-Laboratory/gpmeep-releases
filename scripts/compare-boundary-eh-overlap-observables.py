#!/usr/bin/env python3
"""Fail closed unless enabled/disabled MPI overlap runs are numerically identical."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any


RESULT_PREFIX = "gpmeep-multi-benchmark-v4:"
OVERLAP_PREFIX = "gpmeep-boundary-eh-overlap-v1:"
INITIAL_CONDITION_PREFIX = "gpmeep-initial-condition-v1:"
SOURCE_PROFILE_PREFIX = "gpmeep-source-profile-v1:"
INITIALIZATION_TIMING_PREFIX = "gpmeep-initialize-field-timing-v1:"
QUALIFICATION_PREFIX = "gpmeep-qualification:"
WORKER_GPMEEP_PREFIXES = (
    RESULT_PREFIX,
    "gpmeep-multi-device-v1:",
    "gpmeep-multi-process-v1:",
    SOURCE_PROFILE_PREFIX,
    INITIAL_CONDITION_PREFIX,
    INITIALIZATION_TIMING_PREFIX,
    "gpmeep-boundary-phase-graph-v1:",
    "gpmeep-eager-mpi-v1:",
    "gpmeep-receive-pingpong-v1:",
    OVERLAP_PREFIX,
    "gpmeep-halo-curl-overlap-v1:",
    "gpmeep-tile-coalescing-v1:",
    "gpmeep-phase-batch-policy-v1:",
    "gpmeep-curl-phase-replay-v1:",
    "gpmeep-boundary-descriptor-replay-v1:",
    "gpmeep-dft-phase-sharing-v1:",
    "gpmeep-dft-multi-monitor-batch-v1:",
    "gpmeep-dft-warmup-plan-v1:",
)

IDENTITY_KEYS = (
    "schema_version",
    "mpi_ranks",
    "pixels",
    "cells",
    "warmup_steps",
    "steps",
    "source_enabled",
    "overlap_material",
    "requested_transport",
    "selected_transport",
    "completion_policy",
    "initial_condition",
)
OBSERVABLE_KEYS = (
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

OVERLAP_KEYS = (
    "checks",
    "eligible_ranks",
    "eligible",
    "launched_h",
    "launched_e",
    "skipped_disabled",
    "skipped_unsupported_schedule",
    "skipped_no_remote",
    "skipped_cold_topology",
    "rejected",
)


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        parse_constant=reject_json_constant,
        object_pairs_hook=strict_json_object,
    )


def one_record(path: pathlib.Path, prefix: str) -> dict[str, Any]:
    matches = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith(prefix):
            value = strict_json_loads(line[len(prefix) :])
            if not isinstance(value, dict):
                raise ValueError(f"{path}: {prefix} payload is not an object")
            matches.append(value)
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected exactly one {prefix} record, found {len(matches)}"
        )
    return matches[0]


def reject_unknown_gpmeep_markers(path: pathlib.Path) -> None:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    expected_qualification = f"{QUALIFICATION_PREFIX}{path.name}:PASS"
    qualification = [line for line in lines if QUALIFICATION_PREFIX in line]
    nonempty = [line for line in lines if line]
    if (
        qualification != [expected_qualification]
        or not nonempty
        or nonempty[-1] != expected_qualification
    ):
        raise ValueError(
            f"{path}: expected one exact terminal PASS qualification annotation"
        )
    for line in lines:
        position = line.find("gpmeep-")
        if position < 0 or line == expected_qualification:
            continue
        if not any(
            line.startswith(prefix, position)
            for prefix in WORKER_GPMEEP_PREFIXES
        ):
            raise ValueError(
                f"{path}: unknown or mixed-version gpmeep marker"
            )


def validate_initialization_records(
    path: pathlib.Path, benchmark: dict[str, Any]
) -> None:
    reject_unknown_gpmeep_markers(path)
    source_profile = one_record(path, SOURCE_PROFILE_PREFIX)
    if source_profile != {"profile": "single-ez-v1"}:
        raise ValueError(
            "source-profile marker differs from the release profile"
        )
    initial_condition = one_record(path, INITIAL_CONDITION_PREFIX)
    if set(initial_condition) != {"profile"}:
        raise ValueError("initial-condition marker has an invalid exact schema")
    profile = initial_condition["profile"]
    if (
        type(profile) is not str
        or profile != "trigonometric-v1"
        or benchmark.get("initial_condition") != profile
    ):
        raise ValueError(
            "initial-condition marker, worker record, and release profile differ"
        )
    timing = one_record(path, INITIALIZATION_TIMING_PREFIX)
    if set(timing) != {"applications_per_rank", "max_seconds"}:
        raise ValueError(
            "initialize-field timing marker has an invalid exact schema"
        )
    applications = timing["applications_per_rank"]
    seconds = timing["max_seconds"]
    if type(applications) is not int or applications != 1:
        raise ValueError(
            "initialize-field timing marker must report one application per rank"
        )
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(float(seconds))
        or seconds <= 0
    ):
        raise ValueError(
            "initialize-field timing marker max_seconds must be finite and positive"
        )


def selected(record: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(f"record is missing required keys: {missing}")
    return {key: record[key] for key in keys}


def validate_fixed_workload(
    record: dict[str, Any], profile: str = "complete"
) -> None:
    identity = selected(record, IDENTITY_KEYS)
    for key in ("schema_version", "mpi_ranks", "pixels", "cells",
                "warmup_steps", "steps"):
        if type(identity[key]) is not int or identity[key] <= 0:
            raise ValueError(f"workload {key} must be a positive integer")
    if identity["schema_version"] != 4 or identity["mpi_ranks"] != 2:
        raise ValueError("overlap qualification requires schema v4 and two ranks")
    if identity["cells"] != identity["pixels"] ** 3:
        raise ValueError("workload cell count does not equal pixels cubed")
    if profile == "complete":
        if identity["source_enabled"] is not False:
            raise ValueError(
                "complete overlap qualification requires the source-free workload"
            )
        if identity["overlap_material"] is not True:
            raise ValueError(
                "complete overlap qualification requires diagonal epsilon and mu"
            )
    elif profile == "mixed":
        if identity["source_enabled"] is not True:
            raise ValueError("mixed overlap qualification requires the source workload")
        if identity["overlap_material"] is not False:
            raise ValueError("mixed overlap qualification requires the vacuum material")
    else:
        raise ValueError(f"unknown overlap qualification profile: {profile}")
    if identity["requested_transport"] != "cuda-aware" or identity[
        "selected_transport"
    ] != "cuda-aware":
        raise ValueError("exact overlap comparison requires CUDA-aware MPI")
    if identity["completion_policy"] not in ("waitsome", "waitall"):
        raise ValueError("overlap qualification has an invalid completion policy")
    if identity["initial_condition"] != "trigonometric-v1":
        raise ValueError(
            "overlap qualification requires trigonometric-v1 initial condition"
        )
    if record.get("cpu_calls") != 0:
        raise ValueError("overlap qualification recorded CPU fallback")


def validate_overlap_counters(
    record: dict[str, Any], *, enabled: bool, expected_checks: int,
    mixed: bool = False, expected_ranks: int = 2
) -> None:
    counters = selected(record, OVERLAP_KEYS)
    for key, value in counters.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"overlap counter {key} must be a nonnegative integer")
    classified = sum(
        counters[key]
        for key in (
            "eligible",
            "skipped_disabled",
            "skipped_unsupported_schedule",
            "skipped_no_remote",
            "skipped_cold_topology",
            "rejected",
        )
    )
    if counters["checks"] != expected_checks or classified != counters["checks"]:
        raise ValueError("overlap checks are not classified exactly once")
    if counters["launched_h"] + counters["launched_e"] != counters["eligible"]:
        raise ValueError("eligible overlap phases were not launched exactly once")
    if enabled and mixed:
        if not (
            0 < counters["eligible"] < expected_checks
            and 0 < counters["eligible_ranks"] < expected_ranks
            and counters["rejected"] > 0
            and counters["skipped_disabled"] == 0
            and counters["skipped_unsupported_schedule"] == 0
            and counters["skipped_no_remote"] == 0
            and counters["skipped_cold_topology"] == 0
        ):
            raise ValueError(
                "mixed run did not prove asymmetric overlap and fallback dispatch"
            )
    elif enabled:
        if not (
            counters["eligible"] == expected_checks
            and counters["eligible_ranks"] == expected_ranks
            and counters["launched_h"] == expected_checks // 2
            and counters["launched_e"] == expected_checks // 2
            and all(
                counters[key] == 0
                for key in (
                    "skipped_disabled",
                    "skipped_unsupported_schedule",
                    "skipped_no_remote",
                    "skipped_cold_topology",
                    "rejected",
                )
            )
        ):
            raise ValueError("enabled run did not prove complete H/E overlap dispatch")
    elif not (
        counters["eligible"] == 0
        and counters["eligible_ranks"] == 0
        and counters["launched_h"] == 0
        and counters["launched_e"] == 0
        and counters["skipped_disabled"] == expected_checks
        and counters["skipped_unsupported_schedule"] == 0
        and counters["skipped_no_remote"] == 0
        and counters["skipped_cold_topology"] == 0
        and counters["rejected"] == 0
    ):
        raise ValueError("disabled run did not prove zero overlap dispatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("complete", "mixed"), default="complete"
    )
    parser.add_argument("enabled", type=pathlib.Path)
    parser.add_argument("disabled", type=pathlib.Path)
    args = parser.parse_args()

    enabled_result = one_record(args.enabled, RESULT_PREFIX)
    disabled_result = one_record(args.disabled, RESULT_PREFIX)
    validate_initialization_records(args.enabled, enabled_result)
    validate_initialization_records(args.disabled, disabled_result)
    validate_fixed_workload(enabled_result, args.profile)
    validate_fixed_workload(disabled_result, args.profile)
    if selected(enabled_result, IDENTITY_KEYS) != selected(
        disabled_result, IDENTITY_KEYS
    ):
        raise ValueError("enabled/disabled workload identities differ")
    if selected(enabled_result, OBSERVABLE_KEYS) != selected(
        disabled_result, OBSERVABLE_KEYS
    ):
        raise ValueError("enabled/disabled numerical observables differ")

    enabled_overlap = one_record(args.enabled, OVERLAP_PREFIX)
    disabled_overlap = one_record(args.disabled, OVERLAP_PREFIX)
    expected_checks = (
        2 * enabled_result["steps"] * enabled_result["mpi_ranks"]
    )
    validate_overlap_counters(
        enabled_overlap, enabled=True, expected_checks=expected_checks,
        mixed=args.profile == "mixed",
        expected_ranks=enabled_result["mpi_ranks"],
    )
    validate_overlap_counters(
        disabled_overlap, enabled=False, expected_checks=expected_checks,
        expected_ranks=enabled_result["mpi_ranks"],
    )

    print(
        json.dumps(
            {
                "state": "PASS",
                "enabled": str(args.enabled.resolve()),
                "disabled": str(args.disabled.resolve()),
                "observables_exact": True,
                "profile": args.profile,
                "enabled_overlap": enabled_overlap,
                "disabled_overlap": disabled_overlap,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
