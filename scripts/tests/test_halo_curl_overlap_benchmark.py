from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_halo_curl_overlap_benchmark_test",
    SCRIPTS / "benchmark-halo-curl-overlap.py",
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)

COMPARATOR_SPEC = importlib.util.spec_from_file_location(
    "gpmeep_halo_curl_overlap_comparator_test",
    SCRIPTS / "compare-halo-curl-overlap-observables.py",
)
assert COMPARATOR_SPEC is not None and COMPARATOR_SPEC.loader is not None
COMPARATOR = importlib.util.module_from_spec(COMPARATOR_SPEC)
sys.modules[COMPARATOR_SPEC.name] = COMPARATOR
COMPARATOR_SPEC.loader.exec_module(COMPARATOR)


def overlap_record(condition: str) -> dict:
    checks = (
        BENCHMARK.PROFILE["measured_steps"]
        * BENCHMARK.PROFILE["mpi_ranks"]
    )
    common = {
        "checks": checks,
        "skipped_unsupported_schedule": 0,
        "skipped_no_remote": 0,
        "skipped_cold_topology": 0,
        "rejected_feature": 0,
        "rejected_small": 0,
    }
    if condition == "enabled":
        return {
            **common,
            "eligible": checks,
            "launches": checks,
            "skipped_disabled": 0,
            "full_points": checks * 100,
            "interior_points": checks * 90,
            "shell_points": checks * 10,
        }
    return {
        **common,
        "eligible": 0,
        "launches": 0,
        "skipped_disabled": checks,
        "full_points": 0,
        "interior_points": 0,
        "shell_points": 0,
    }


def observable(seconds: float) -> dict:
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
    profile = BENCHMARK.PROFILE
    benchmark = {
        **observable(1.0),
        "schema_version": 4,
        "mpi_ranks": profile["mpi_ranks"],
        "pixels": profile["pixels"],
        "cells": profile["pixels"] ** 3,
        "warmup_steps": profile["warmup_steps"],
        "steps": profile["measured_steps"],
        "loop_tile_base_db": profile["loop_tile_base_db"],
        "source_enabled": profile["source_enabled"],
        "overlap_material": profile["overlap_material"],
        "requested_transport": profile["transport"],
        "selected_transport": profile["transport"],
        "completion_policy": profile["completion_policy"],
        "initial_condition": profile["initial_condition"],
        "cpu_calls": 0,
        "cuda_calls": 100,
    }
    markers = (
        (BENCHMARK.BASE.HELPER.SOURCE_PROFILE_PREFIX,
         {"profile": "single-ez-v1"}),
        (BENCHMARK.BASE.HELPER.INITIAL_CONDITION_PREFIX,
         {"profile": "trigonometric-v1"}),
        (BENCHMARK.BASE.HELPER.INITIALIZATION_TIMING_PREFIX,
         {"applications_per_rank": 1, "max_seconds": 0.25}),
        (BENCHMARK.BASE.OVERLAP_PREFIX, overlap_record(condition)),
        (BENCHMARK.BASE.EAGER_PREFIX, {"receive": 1, "send": 1}),
        (BENCHMARK.BASE.PINGPONG_PREFIX, {"secondary_selections": 1}),
        (BENCHMARK.BASE.GRAPH_PREFIX, {"launches": 1}),
        (BENCHMARK.BASE.DFT_PREFIX, {"submitted_updates": 1}),
        (BENCHMARK.BASE.BENCHMARK_PREFIX, benchmark),
    )
    devices = "".join(
        prefixed(
            BENCHMARK.BASE.DEVICE_PREFIX,
            {"rank": rank, "uuid": f"gpu-{rank}"},
        )
        for rank in (0, 1)
    )
    return devices + "".join(prefixed(prefix, value) for prefix, value in markers)


def samples(speedup: float) -> list[dict]:
    result = []
    for kind, pair, position, condition in BENCHMARK.BASE.expected_schedule():
        benchmark_observable = observable(
            10.0 / speedup if condition == "enabled" else 10.0
        )
        benchmark_observable["h2d_bytes"] = (
            265000 if condition == "enabled" else 135400
        )
        result.append(
            {
                "kind": kind,
                "pair": pair,
                "position": position,
                "condition": condition,
                "external_process_wall_seconds": (
                    20.0 / speedup if condition == "enabled" else 20.0
                ),
                "records": {
                    "benchmark": benchmark_observable,
                    "overlap": overlap_record(condition),
                },
            }
        )
    return result


class HaloCurlOverlapBenchmarkTests(unittest.TestCase):
    def test_inherited_validator_rejects_invalid_initialization_timing(self) -> None:
        output = valid_worker_output()
        records = BENCHMARK.BASE.validate_sample("enabled", output)
        self.assertEqual(
            records["benchmark"]["initial_condition"], "trigonometric-v1"
        )
        valid_timing = prefixed(
            BENCHMARK.BASE.HELPER.INITIALIZATION_TIMING_PREFIX,
            {"applications_per_rank": 1, "max_seconds": 0.25},
        )
        invalid_timing = prefixed(
            BENCHMARK.BASE.HELPER.INITIALIZATION_TIMING_PREFIX,
            {"applications_per_rank": True, "max_seconds": 0.25},
        )
        with self.assertRaisesRegex(RuntimeError, "one application"):
            BENCHMARK.BASE.validate_sample(
                "enabled", output.replace(valid_timing, invalid_timing)
            )

    def test_builder_warms_topology_before_feature_rejection_gate(self) -> None:
        builder = (SCRIPTS / "build-meep-cuda-mpi-python.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "MEEP_GPU_EXPECT_HALO_CURL_OVERLAP_REJECTED_FEATURE=1 \\\n"
            "  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \\\n",
            builder,
        )
        self.assertIn(
            "MEEP_GPU_EXPECT_HALO_CURL_OVERLAP_UNSUPPORTED_SCHEDULE=1 \\\n"
            "  MEEP_GPU_MULTI_PIXELS=16 MEEP_GPU_MULTI_WARMUP_STEPS=4 \\\n",
            builder,
        )

    def test_profile_and_environment_are_fixed(self) -> None:
        BENCHMARK.BASE.assert_fixed_profile()
        self.assertEqual(BENCHMARK.PROFILE["loop_tile_base_db"], 0)
        enabled = BENCHMARK.condition_environment({}, "enabled")
        self.assertEqual(enabled["MEEP_GPU_ENABLE_HALO_CURL_OVERLAP"], "1")
        self.assertNotIn("MEEP_GPU_DISABLE_HALO_CURL_OVERLAP", enabled)
        disabled = BENCHMARK.condition_environment({}, "disabled")
        self.assertEqual(disabled["MEEP_GPU_DISABLE_HALO_CURL_OVERLAP"], "1")
        self.assertNotIn("MEEP_GPU_ENABLE_HALO_CURL_OVERLAP", disabled)

    def test_counter_partition_rejects_point_loss(self) -> None:
        BENCHMARK.validate_overlap("enabled", overlap_record("enabled"))
        BENCHMARK.validate_overlap("disabled", overlap_record("disabled"))
        broken = overlap_record("enabled")
        broken["shell_points"] -= 1
        with self.assertRaisesRegex(RuntimeError, "dispatch proof"):
            BENCHMARK.validate_overlap("enabled", broken)

    def test_summary_requires_three_percent_stable_improvement(self) -> None:
        negative = BENCHMARK.summarize(samples(1.01))
        self.assertFalse(negative["promote_default"])
        self.assertEqual(
            negative["cuda_transfer_diagnostics"]["enabled"]["h2d_bytes"],
            [265000],
        )
        self.assertEqual(
            negative["cuda_transfer_diagnostics"]["disabled"]["h2d_bytes"],
            [135400],
        )
        self.assertEqual(
            negative["decision"], "retain-halo-curl-overlap-opt-in"
        )
        positive = BENCHMARK.summarize(samples(1.04))
        self.assertTrue(positive["promote_default"])
        self.assertEqual(
            positive["decision"], "enable-halo-curl-overlap-default"
        )

    def test_exact_comparator_separates_cache_traffic_from_physics(self) -> None:
        identity = {
            "schema_version": 4,
            "mpi_ranks": 2,
            "pixels": 16,
            "cells": 4096,
            "warmup_steps": 4,
            "steps": 10,
            "loop_tile_base_db": 0,
            "source_enabled": False,
            "overlap_material": True,
            "requested_transport": "cuda-aware",
            "selected_transport": "cuda-aware",
            "completion_policy": "waitsome",
            "initial_condition": "trigonometric-v1",
            "cpu_calls": 0,
        }
        enabled_result = {**identity, **observable(1.0), "h2d_bytes": 265000}
        disabled_result = {**identity, **observable(1.0), "h2d_bytes": 135400}
        enabled_halo = overlap_record("enabled")
        enabled_halo.update(
            checks=20,
            eligible=20,
            launches=20,
            full_points=2000,
            interior_points=1800,
            shell_points=200,
        )
        disabled_halo = overlap_record("disabled")
        disabled_halo.update(checks=20, skipped_disabled=20)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            enabled_path = root / "enabled.log"
            disabled_path = root / "disabled.log"

            def write_log(path: pathlib.Path, result: dict, halo: dict) -> None:
                path.write_text(
                    COMPARATOR.BASE.RESULT_PREFIX
                    + json.dumps(result, sort_keys=True)
                    + "\n"
                    + COMPARATOR.BASE.SOURCE_PROFILE_PREFIX
                    + json.dumps({"profile": "single-ez-v1"}, sort_keys=True)
                    + "\n"
                    + COMPARATOR.BASE.INITIAL_CONDITION_PREFIX
                    + json.dumps({"profile": "trigonometric-v1"}, sort_keys=True)
                    + "\n"
                    + COMPARATOR.BASE.INITIALIZATION_TIMING_PREFIX
                    + json.dumps(
                        {"applications_per_rank": 1, "max_seconds": 0.25},
                        sort_keys=True,
                    )
                    + "\n"
                    + COMPARATOR.HALO_PREFIX
                    + json.dumps(halo, sort_keys=True)
                    + "\n"
                    + COMPARATOR.BASE.QUALIFICATION_PREFIX
                    + path.name
                    + ":PASS\n",
                    encoding="utf-8",
                )

            write_log(enabled_path, enabled_result, enabled_halo)
            write_log(disabled_path, disabled_result, disabled_halo)
            with mock.patch.object(
                sys,
                "argv",
                ["compare", str(enabled_path), str(disabled_path)],
            ):
                self.assertEqual(COMPARATOR.main(), 0)

            disabled_result["energy"] += 1.0
            write_log(disabled_path, disabled_result, disabled_halo)
            with mock.patch.object(
                sys,
                "argv",
                ["compare", str(enabled_path), str(disabled_path)],
            ):
                with self.assertRaisesRegex(ValueError, "physical observables"):
                    COMPARATOR.main()


if __name__ == "__main__":
    unittest.main()
