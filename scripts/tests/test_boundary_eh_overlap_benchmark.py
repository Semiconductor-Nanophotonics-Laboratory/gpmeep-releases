from __future__ import annotations

import importlib.util
import json
import pathlib
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_boundary_eh_overlap_benchmark",
    SCRIPTS / "benchmark-boundary-eh-overlap.py",
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def overlap_record(condition: str) -> dict:
    checks = (
        2
        * BENCHMARK.PROFILE["measured_steps"]
        * BENCHMARK.PROFILE["mpi_ranks"]
    )
    common = {
        "checks": checks,
        "skipped_unsupported_schedule": 0,
        "skipped_no_remote": 0,
        "skipped_cold_topology": 0,
        "rejected": 0,
    }
    if condition == "enabled":
        return {
            **common,
            "eligible_ranks": 2,
            "eligible": checks,
            "launched_h": checks // 2,
            "launched_e": checks // 2,
            "skipped_disabled": 0,
        }
    return {
        **common,
        "eligible_ranks": 0,
        "eligible": 0,
        "launched_h": 0,
        "launched_e": 0,
        "skipped_disabled": checks,
    }


def observable_record(seconds: float) -> dict:
    return {
        "seconds": seconds,
        "cut_probe_values": [[0.1, 0.0]],
        "cut_probe_l2": 0.1,
        "ez_l2": 1.0,
        "ez_weighted_checksum": 2.0,
        "energy": 3.0,
        "dft_norm": 4.0,
        "remote_cut_scalars": 8,
        "remote_cut_l2": 5.0,
        "cpu_calls": 0,
        "cuda_calls": 100,
        "phase_calls": {"curl": {"cpu_calls": 0, "cuda_calls": 10}},
        "h2d_bytes": 32,
        "d2h_bytes": 16,
        "mpi_messages": 20,
        "mpi_scalars": 40,
        "cuda_aware_bytes": 160,
        "pinned_bytes": 0,
    }


def prefixed(prefix: str, value: dict) -> str:
    return prefix + json.dumps(value, separators=(",", ":")) + "\n"


def valid_worker_output(condition: str = "enabled") -> str:
    benchmark = {
        **observable_record(1.0),
        "schema_version": 4,
        "mpi_ranks": BENCHMARK.PROFILE["mpi_ranks"],
        "pixels": BENCHMARK.PROFILE["pixels"],
        "cells": BENCHMARK.PROFILE["pixels"] ** 3,
        "warmup_steps": BENCHMARK.PROFILE["warmup_steps"],
        "steps": BENCHMARK.PROFILE["measured_steps"],
        "source_enabled": BENCHMARK.PROFILE["source_enabled"],
        "overlap_material": BENCHMARK.PROFILE["overlap_material"],
        "requested_transport": BENCHMARK.PROFILE["transport"],
        "selected_transport": BENCHMARK.PROFILE["transport"],
        "completion_policy": BENCHMARK.PROFILE["completion_policy"],
        "initial_condition": BENCHMARK.PROFILE["initial_condition"],
    }
    markers = (
        (BENCHMARK.HELPER.SOURCE_PROFILE_PREFIX,
         {"profile": "single-ez-v1"}),
        (BENCHMARK.HELPER.INITIAL_CONDITION_PREFIX,
         {"profile": "trigonometric-v1"}),
        (BENCHMARK.HELPER.INITIALIZATION_TIMING_PREFIX,
         {"applications_per_rank": 1, "max_seconds": 0.25}),
        (BENCHMARK.OVERLAP_PREFIX, overlap_record(condition)),
        (BENCHMARK.EAGER_PREFIX, {"receive": 1, "send": 1}),
        (BENCHMARK.PINGPONG_PREFIX, {"secondary_selections": 1}),
        (BENCHMARK.GRAPH_PREFIX, {"launches": 1}),
        (BENCHMARK.DFT_PREFIX, {"submitted_updates": 1}),
        (BENCHMARK.BENCHMARK_PREFIX, benchmark),
    )
    devices = "".join(
        prefixed(
            BENCHMARK.DEVICE_PREFIX,
            {"rank": rank, "uuid": f"gpu-{rank}"},
        )
        for rank in (0, 1)
    )
    return devices + "".join(prefixed(prefix, value) for prefix, value in markers)


def synthetic_samples(speedup: float) -> list[dict]:
    samples = []
    for kind, pair, position, condition in BENCHMARK.expected_schedule():
        disabled_seconds = 10.0
        seconds = disabled_seconds / speedup if condition == "enabled" else 10.0
        samples.append(
            {
                "kind": kind,
                "pair": pair,
                "position": position,
                "condition": condition,
                "external_process_wall_seconds": (
                    20.0 / speedup if condition == "enabled" else 20.0
                ),
                "records": {
                    "benchmark": observable_record(seconds),
                    "overlap": overlap_record(condition),
                },
            }
        )
    return samples


class BoundaryEhOverlapBenchmarkTests(unittest.TestCase):
    def test_profile_hash_and_schedule_are_fixed(self) -> None:
        BENCHMARK.assert_fixed_profile()
        schedule = BENCHMARK.expected_schedule()
        self.assertEqual(len(schedule), 14)
        self.assertEqual(schedule[:2], [
            ("warmup", 0, 1, "enabled"),
            ("warmup", 0, 2, "disabled"),
        ])
        self.assertEqual(schedule[2:6], [
            ("measured", 1, 1, "enabled"),
            ("measured", 1, 2, "disabled"),
            ("measured", 2, 1, "disabled"),
            ("measured", 2, 2, "enabled"),
        ])

    def test_overlap_counter_partition_rejects_partial_launch(self) -> None:
        BENCHMARK.validate_overlap("enabled", overlap_record("enabled"))
        BENCHMARK.validate_overlap("disabled", overlap_record("disabled"))
        broken = overlap_record("enabled")
        broken["launched_e"] -= 1
        with self.assertRaisesRegex(RuntimeError, "exact phase partition"):
            BENCHMARK.validate_overlap("enabled", broken)

    def test_validate_sample_requires_initialization_markers(self) -> None:
        output = valid_worker_output()
        records = BENCHMARK.validate_sample("enabled", output)
        self.assertEqual(
            records["benchmark"]["initial_condition"], "trigonometric-v1"
        )
        missing = "".join(
            line + "\n"
            for line in output.splitlines()
            if not line.startswith(BENCHMARK.HELPER.INITIAL_CONDITION_PREFIX)
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            BENCHMARK.validate_sample("enabled", missing)
        missing_source_profile = "".join(
            line + "\n"
            for line in output.splitlines()
            if not line.startswith(BENCHMARK.HELPER.SOURCE_PROFILE_PREFIX)
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            BENCHMARK.validate_sample("enabled", missing_source_profile)

    def test_ab_environment_requires_explicit_opt_in(self) -> None:
        enabled = BENCHMARK.condition_environment({"BASE": "1"}, "enabled")
        self.assertEqual(enabled["MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP"], "1")
        self.assertNotIn("MEEP_GPU_DISABLE_BOUNDARY_EH_OVERLAP", enabled)
        disabled = BENCHMARK.condition_environment({"BASE": "1"}, "disabled")
        self.assertEqual(disabled["MEEP_GPU_DISABLE_BOUNDARY_EH_OVERLAP"], "1")
        self.assertNotIn("MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP", disabled)
        with self.assertRaisesRegex(RuntimeError, "unknown overlap A/B condition"):
            BENCHMARK.condition_environment({}, "typo")

    def test_summary_promotes_stable_material_improvement(self) -> None:
        summary = BENCHMARK.summarize(synthetic_samples(1.02))
        self.assertTrue(summary["promote_default"])
        self.assertEqual(
            summary["decision"], "retain-boundary-eh-overlap-default"
        )
        self.assertEqual(summary["faster_pairs"], 6)

    def test_summary_publishes_negative_decision_without_false_promotion(self) -> None:
        summary = BENCHMARK.summarize(synthetic_samples(0.99))
        self.assertFalse(summary["promote_default"])
        self.assertEqual(
            summary["decision"], "disable-boundary-eh-overlap-default"
        )


if __name__ == "__main__":
    unittest.main()
