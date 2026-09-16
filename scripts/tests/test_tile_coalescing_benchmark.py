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
    "gpmeep_tile_coalescing_benchmark_test",
    SCRIPTS / "benchmark-tile-coalescing.py",
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)

COMPARATOR_SPEC = importlib.util.spec_from_file_location(
    "gpmeep_tile_coalescing_comparator_test",
    SCRIPTS / "compare-tile-coalescing-observables.py",
)
assert COMPARATOR_SPEC is not None and COMPARATOR_SPEC.loader is not None
COMPARATOR = importlib.util.module_from_spec(COMPARATOR_SPEC)
sys.modules[COMPARATOR_SPEC.name] = COMPARATOR
COMPARATOR_SPEC.loader.exec_module(COMPARATOR)


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
        "cpu_calls": 0,
        "cuda_calls": 100,
        "phase_calls": {"curl": {"cpu_calls": 0, "cuda_calls": 10}},
        "h2d_bytes": 0,
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
        "loop_tile_base_db": (
            profile["reference_loop_tile_base_db"]
            if condition == "reference"
            else profile["loop_tile_base_db"]
        ),
        "bfast": False,
        "source_enabled": profile["source_enabled"],
        "overlap_material": profile["overlap_material"],
        "requested_transport": profile["transport"],
        "selected_transport": profile["transport"],
        "completion_policy": profile["completion_policy"],
        "initial_condition": profile["initial_condition"],
    }
    tile = (
        {
            "curl_chunk_phases": 10,
            "curl_input_tiles": 100,
            "update_eh_chunk_phases": 0,
            "update_eh_input_tiles": 0,
        }
        if condition == "enabled"
        else {
            "curl_chunk_phases": 0,
            "curl_input_tiles": 0,
            "update_eh_chunk_phases": 0,
            "update_eh_input_tiles": 0,
        }
    )
    markers = (
        (BENCHMARK.BASE.HELPER.SOURCE_PROFILE_PREFIX,
         {"profile": "single-ez-v1"}),
        (BENCHMARK.BASE.HELPER.INITIAL_CONDITION_PREFIX,
         {"profile": "trigonometric-v1"}),
        (BENCHMARK.BASE.HELPER.INITIALIZATION_TIMING_PREFIX,
         {"applications_per_rank": 1, "max_seconds": 0.25}),
        (BENCHMARK.TILE_PREFIX, tile),
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


def synthetic_samples(
    legacy_speedup: float, reference_ratio: float
) -> list[dict]:
    result = []
    for kind, pair, position, condition in BENCHMARK.expected_schedule():
        if condition == "enabled":
            seconds = 10.0 / legacy_speedup
        elif condition == "disabled":
            seconds = 10.0
        else:
            seconds = (10.0 / legacy_speedup) / reference_ratio
        result.append(
            {
                "kind": kind,
                "pair": pair,
                "position": position,
                "condition": condition,
                "external_process_wall_seconds": seconds + 10.0,
                "records": {
                    "benchmark": observable(seconds),
                    "devices": [
                        {"rank": 0, "uuid": "gpu-a"},
                        {"rank": 1, "uuid": "gpu-b"},
                    ],
                },
            }
        )
    return result


class TileCoalescingBenchmarkTests(unittest.TestCase):
    def test_validate_sample_requires_initialization_timing_marker(self) -> None:
        output = valid_worker_output()
        records = BENCHMARK.validate_sample("enabled", output)
        self.assertEqual(
            records["benchmark"]["initial_condition"], "trigonometric-v1"
        )
        missing = "".join(
            line + "\n"
            for line in output.splitlines()
            if not line.startswith(
                BENCHMARK.BASE.HELPER.INITIALIZATION_TIMING_PREFIX
            )
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            BENCHMARK.validate_sample("enabled", missing)

    def test_exact_comparator_requires_dispatch_and_physics(self) -> None:
        identity = {
            "schema_version": 4,
            "mpi_ranks": 2,
            "pixels": 16,
            "cells": 4096,
            "warmup_steps": 4,
            "steps": 10,
            "loop_tile_base_db": 128,
            "source_enabled": True,
            "overlap_material": False,
            "requested_transport": "cuda-aware",
            "selected_transport": "cuda-aware",
            "completion_policy": "waitsome",
            "initial_condition": "trigonometric-v1",
            "cpu_calls": 0,
            "cuda_calls": 20,
        }
        physical = {
            "cut_probe_values": [[0.1, 0.0]],
            "cut_probe_l2": 0.1,
            "ez_l2": 1.0,
            "ez_weighted_checksum": 2.0,
            "energy": 3.0,
            "dft_norm": 4.0,
            "remote_cut_scalars": 8,
            "remote_cut_l2": 5.0,
            "mpi_messages": 20,
            "mpi_scalars": 40,
            "cuda_aware_bytes": 160,
            "pinned_bytes": 0,
        }
        enabled_result = {**identity, **physical, "h2d_bytes": 100}
        disabled_result = {**identity, **physical, "h2d_bytes": 1000}
        enabled_tile = {
            "curl_chunk_phases": 20,
            "curl_input_tiles": 80,
            "update_eh_chunk_phases": 0,
            "update_eh_input_tiles": 0,
        }
        disabled_tile = {key: 0 for key in COMPARATOR.TILE_KEYS}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            enabled_path = root / "enabled.log"
            disabled_path = root / "disabled.log"

            def write_log(path: pathlib.Path, result: dict, tile: dict) -> None:
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
                    + COMPARATOR.TILE_PREFIX
                    + json.dumps(tile, sort_keys=True)
                    + "\n"
                    + COMPARATOR.BASE.QUALIFICATION_PREFIX
                    + path.name
                    + ":PASS\n",
                    encoding="utf-8",
                )

            write_log(enabled_path, enabled_result, enabled_tile)
            write_log(disabled_path, disabled_result, disabled_tile)
            with mock.patch.object(
                sys,
                "argv",
                ["compare", str(enabled_path), str(disabled_path)],
            ):
                self.assertEqual(COMPARATOR.main(), 0)

            disabled_result["energy"] += 1.0
            write_log(disabled_path, disabled_result, disabled_tile)
            with mock.patch.object(
                sys,
                "argv",
                ["compare", str(enabled_path), str(disabled_path)],
            ):
                with self.assertRaisesRegex(ValueError, "physical observables"):
                    COMPARATOR.main()

            disabled_result["energy"] -= 1.0
            enabled_tile["curl_input_tiles"] = enabled_tile["curl_chunk_phases"]
            write_log(enabled_path, enabled_result, enabled_tile)
            write_log(disabled_path, disabled_result, disabled_tile)
            with mock.patch.object(
                sys,
                "argv",
                ["compare", str(enabled_path), str(disabled_path)],
            ):
                with self.assertRaisesRegex(ValueError, "dispatch proof"):
                    COMPARATOR.main()

    def test_profile_hash_schedule_and_dependency_are_fixed(self) -> None:
        BENCHMARK.BASE.assert_fixed_profile()
        schedule = BENCHMARK.expected_schedule()
        self.assertEqual(len(schedule), 21)
        self.assertEqual(
            schedule[:6],
            [
                ("warmup", 0, 1, "enabled"),
                ("warmup", 0, 2, "disabled"),
                ("warmup", 0, 3, "reference"),
                ("measured", 1, 1, "enabled"),
                ("measured", 1, 2, "disabled"),
                ("reference", 1, 3, "reference"),
            ],
        )
        self.assertEqual(
            BENCHMARK.BASE.ADDITIONAL_RUNNER_SOURCES,
            {"benchmark-boundary-eh-overlap.py": BENCHMARK.BASE_PATH},
        )

    def test_condition_environment_is_disjoint_and_reference_is_no_tile(self) -> None:
        hostile = {
            "MEEP_GPU_DISABLE_TILE_COALESCING": "1",
            "MEEP_GPU_EXPECT_TILE_COALESCING": "1",
            "MEEP_GPU_EXPECT_NO_TILE_COALESCING": "1",
            "MEEP_GPU_MULTI_LOOP_TILE_BASE_DB": "128",
        }
        enabled = BENCHMARK.condition_environment(hostile, "enabled")
        self.assertEqual(enabled["MEEP_GPU_EXPECT_TILE_COALESCING"], "1")
        self.assertNotIn("MEEP_GPU_DISABLE_TILE_COALESCING", enabled)
        self.assertNotIn("MEEP_GPU_EXPECT_NO_TILE_COALESCING", enabled)
        disabled = BENCHMARK.condition_environment(hostile, "disabled")
        self.assertEqual(disabled["MEEP_GPU_DISABLE_TILE_COALESCING"], "1")
        self.assertEqual(disabled["MEEP_GPU_EXPECT_NO_TILE_COALESCING"], "1")
        self.assertNotIn("MEEP_GPU_EXPECT_TILE_COALESCING", disabled)
        reference = BENCHMARK.condition_environment(hostile, "reference")
        self.assertEqual(reference["MEEP_GPU_MULTI_LOOP_TILE_BASE_DB"], "0")
        self.assertEqual(reference["MEEP_GPU_EXPECT_NO_TILE_COALESCING"], "1")
        self.assertNotIn("MEEP_GPU_DISABLE_TILE_COALESCING", reference)

    def test_counter_contract_rejects_missing_or_control_coalescing(self) -> None:
        enabled = {
            "curl_chunk_phases": 10,
            "curl_input_tiles": 100,
            "update_eh_chunk_phases": 0,
            "update_eh_input_tiles": 0,
        }
        disabled = {key: 0 for key in enabled}
        BENCHMARK.validate_tile_coalescing("enabled", enabled)
        BENCHMARK.validate_tile_coalescing("disabled", disabled)
        BENCHMARK.validate_tile_coalescing("reference", disabled)
        with self.assertRaisesRegex(RuntimeError, "did not execute"):
            BENCHMARK.validate_tile_coalescing("enabled", disabled)
        missing_phase = dict(enabled)
        missing_phase["update_eh_input_tiles"] = 1
        with self.assertRaisesRegex(RuntimeError, "did not execute"):
            BENCHMARK.validate_tile_coalescing("enabled", missing_phase)
        uncoalesced_phase = dict(enabled)
        uncoalesced_phase["update_eh_chunk_phases"] = 10
        uncoalesced_phase["update_eh_input_tiles"] = 10
        with self.assertRaisesRegex(RuntimeError, "did not execute"):
            BENCHMARK.validate_tile_coalescing("enabled", uncoalesced_phase)
        with self.assertRaisesRegex(RuntimeError, "unexpectedly"):
            BENCHMARK.validate_tile_coalescing("reference", enabled)

    def test_summary_requires_large_recovery_and_no_tile_parity(self) -> None:
        passing = BENCHMARK.summarize(synthetic_samples(10.0, 1.0))
        self.assertTrue(passing["promote_default"])
        self.assertEqual(
            passing["decision"], "retain-gpu-tile-coalescing-default"
        )
        self.assertEqual(
            passing["rank_device_uuid_mapping"],
            [
                {"rank": 0, "uuid": "gpu-a"},
                {"rank": 1, "uuid": "gpu-b"},
            ],
        )
        self.assertEqual(passing["reference_pairs_within_limit"], 6)
        slow_recovery = BENCHMARK.summarize(synthetic_samples(4.0, 1.0))
        self.assertFalse(slow_recovery["promote_default"])
        slow_reference = BENCHMARK.summarize(synthetic_samples(10.0, 1.2))
        self.assertFalse(slow_reference["promote_default"])
        self.assertEqual(
            slow_reference["decision"], "disable-gpu-tile-coalescing-default"
        )

    def test_summary_rejects_cross_sample_device_remapping(self) -> None:
        remapped = synthetic_samples(10.0, 1.0)
        remapped[-1]["records"]["devices"] = [
            {"rank": 0, "uuid": "gpu-b"},
            {"rank": 1, "uuid": "gpu-a"},
        ]
        with self.assertRaisesRegex(RuntimeError, "mapping changed"):
            BENCHMARK.summarize(remapped)


if __name__ == "__main__":
    unittest.main()
