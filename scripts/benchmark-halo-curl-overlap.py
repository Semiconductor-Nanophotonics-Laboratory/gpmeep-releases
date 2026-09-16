#!/usr/bin/env python3
"""Receipt-bound paired A/B for final-H-halo/D-curl overlap.

This runner reuses the sealed scheduling, process-monitoring, receipt, and
artifact machinery of benchmark-boundary-eh-overlap.py while replacing the
feature-specific profile, counter contract, and environment projection.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any


SCRIPT = pathlib.Path(__file__).resolve()
BASE_PATH = SCRIPT.parent / "benchmark-boundary-eh-overlap.py"
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_halo_curl_overlap_evidence_base", BASE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load boundary-overlap evidence base")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

PROFILE_ID = "m8.12-halo-curl-overlap-ab-v1"
PROFILE = {
    "profile_id": PROFILE_ID,
    "mpi_ranks": 2,
    "pixels": 192,
    "warmup_steps": 12,
    "measured_steps": 400,
    "warmup_processes_per_condition": 1,
    "measured_pairs": 6,
    "transport": "cuda-aware",
    "completion_policy": "waitsome",
    "initial_condition": "trigonometric-v1",
    "source_enabled": False,
    "overlap_material": True,
    "loop_tile_base_db": 0,
    "minimum_median_paired_speedup": 1.03,
    "minimum_faster_pairs": 4,
    "maximum_t90_upper_time_ratio": 1.0,
    "both_position_strata_must_be_positive": True,
}
PROFILE_SHA256 = (
    "37b7064a71bee9280125fce5be75e07da83671fd7dc4275ee0c197071a332f4b"
)


def validate_overlap(condition: str, record: dict[str, Any]) -> None:
    expected_checks = PROFILE["measured_steps"] * PROFILE["mpi_ranks"]
    fields = (
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
    if any(
        type(record.get(key)) is not int or record[key] < 0 for key in fields
    ):
        raise RuntimeError("halo/curl overlap counter record is incomplete")
    classified = sum(
        record[key]
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
    if record["checks"] != expected_checks or classified != expected_checks:
        raise RuntimeError("halo/curl checks are not an exact phase partition")
    if condition == "enabled":
        if (
            record["eligible"] != expected_checks
            or record["launches"] != expected_checks
            or any(
                record[key] != 0
                for key in (
                    "skipped_disabled",
                    "skipped_unsupported_schedule",
                    "skipped_no_remote",
                    "skipped_cold_topology",
                    "rejected_feature",
                    "rejected_small",
                )
            )
            or min(
                record["full_points"],
                record["interior_points"],
                record["shell_points"],
            )
            <= 0
            or record["interior_points"] + record["shell_points"]
            != record["full_points"]
        ):
            raise RuntimeError("enabled halo/curl dispatch proof is incomplete")
    elif condition == "disabled":
        if (
            record["eligible"] != 0
            or record["launches"] != 0
            or record["skipped_disabled"] != expected_checks
            or any(
                record[key] != 0
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
            raise RuntimeError("disabled halo/curl path unexpectedly dispatched")
    else:
        raise RuntimeError(f"unknown halo/curl A/B condition: {condition}")


def condition_environment(
    base_environment: dict[str, str], condition: str
) -> dict[str, str]:
    environment = dict(base_environment)
    if condition == "enabled":
        environment["MEEP_GPU_ENABLE_HALO_CURL_OVERLAP"] = "1"
        environment["MEEP_GPU_EXPECT_HALO_CURL_OVERLAP"] = "1"
    elif condition == "disabled":
        environment["MEEP_GPU_DISABLE_HALO_CURL_OVERLAP"] = "1"
        environment["MEEP_GPU_EXPECT_NO_HALO_CURL_OVERLAP"] = "1"
    else:
        raise RuntimeError(f"unknown halo/curl A/B condition: {condition}")
    return environment


ORIGINAL_SUMMARIZE = BASE.summarize


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary = ORIGINAL_SUMMARIZE(samples)
    measured = [sample for sample in samples if sample["kind"] == "measured"]
    summary["cuda_transfer_diagnostics"] = {
        condition: {
            key: sorted(
                {
                    int(sample["records"]["benchmark"][key])
                    for sample in measured
                    if sample["condition"] == condition
                }
            )
            for key in ("h2d_bytes", "d2h_bytes")
        }
        for condition in ("enabled", "disabled")
    }
    promoted = bool(summary["promote_default"])
    summary["decision"] = (
        "enable-halo-curl-overlap-default"
        if promoted
        else "retain-halo-curl-overlap-opt-in"
    )
    return summary


def markdown_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# H halo / D curl overlap paired A/B",
            "",
            f"- Decision: `{summary['decision']}`",
            f"- Median paired speedup: `{summary['median_paired_speedup']:.6f}x`",
            f"- Faster pairs: `{summary['faster_pairs']}/{summary['measured_pair_count']}`",
            f"- 90% upper enabled/disabled time ratio: `{summary['t90_upper_time_ratio']:.6f}`",
            f"- Exact physical observables: `{summary['all_observables_exact']}`",
            "- CUDA transfer diagnostics: "
            f"`{summary['cuda_transfer_diagnostics']}`",
            "",
            "Per-pair speedups: "
            + ", ".join(
                f"{value:.6f}x" for value in summary["paired_speedups"]
            ),
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
BASE.OVERLAP_PREFIX = "gpmeep-halo-curl-overlap-v1:"
BASE.OBSERVABLE_KEYS = (
    "cut_probe_values",
    "cut_probe_l2",
    "ez_l2",
    "ez_weighted_checksum",
    "energy",
    "dft_norm",
    "remote_cut_scalars",
    "remote_cut_l2",
    # Interior/shell splitting changes CUDA plan/cache preparation traffic.
    # Keep transfer counters in every raw sample and summarize them above,
    # but do not misclassify them as physical observables.  Communication
    # work across MPI ranks must remain exact.
    "mpi_messages",
    "mpi_scalars",
    "cuda_aware_bytes",
    "pinned_bytes",
)
BASE.validate_overlap = validate_overlap
BASE.condition_environment = condition_environment
BASE.summarize = summarize
BASE.markdown_report = markdown_report


if __name__ == "__main__":
    raise SystemExit(BASE.main())
