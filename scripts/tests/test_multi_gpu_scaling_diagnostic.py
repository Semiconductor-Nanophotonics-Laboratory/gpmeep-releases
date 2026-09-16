from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import multi_gpu_benchmark as mgb  # noqa: E402

RUNNER_PATH = SCRIPTS / "run-multi-gpu-scaling-diagnostic.py"
SPEC = importlib.util.spec_from_file_location("multi_gpu_scaling_diagnostic", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load multi-GPU scaling diagnostic")
diag = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diag
SPEC.loader.exec_module(diag)


def valid_phase_calls(condition: str = "default") -> dict[str, dict[str, int | str]]:
    phases = {
        phase: {
            "cpu_calls": 0,
            "cuda_calls": 60,
            "expectation": "cuda_required",
        }
        for phase in mgb.REQUIRED_CUDA_PHASES
    }
    phases["polarization"] = {
        "cpu_calls": 0,
        "cuda_calls": 0,
        "expectation": "not_applicable",
    }
    phases["dft"]["cuda_calls"] = 160
    if diag.CONDITIONS[condition]["disable_phase_batched_curl"]:
        phases["curl"]["cuda_calls"] = 120
    if diag.CONDITIONS[condition]["disable_phase_batched_update_eh"]:
        phases["update_eh"]["cuda_calls"] = 120
    if diag.CONDITIONS[condition]["disable_phase_batched_source"]:
        phases["source"]["cuda_calls"] = 120
    return phases


def valid_record(
    ranks: int = 1,
    *,
    pixels: int = 192,
    condition: str = "default",
    seconds: float | None = None,
) -> dict:
    steps = diag.PROFILE["measured_steps"]
    if seconds is None:
        seconds = 2.0 if ranks == 1 else 1.25
    cells = pixels**3
    probes = [[0.1 + 0.01 * index, 0.0] for index in range(mgb.PROBE_COUNT)]
    probe_l2 = math.sqrt(sum(real * real + imag * imag for real, imag in probes))
    phase_calls = valid_phase_calls(condition)
    multi = ranks > 1
    return {
        "schema_version": mgb.SCHEMA_VERSION,
        "mpi_ranks": ranks,
        "pixels": pixels,
        "cells": cells,
        "warmup_steps": diag.PROFILE["warmup_steps"],
        "steps": steps,
        "loop_tile_base_db": diag.CONDITIONS[condition]["loop_tile_base_db"],
        "bfast": False,
        "source_enabled": True,
        "overlap_material": False,
        "seconds": seconds,
        "mcells_per_second": cells * steps / seconds / 1e6,
        "requested_transport": "pinned",
        "selected_transport": "pinned" if multi else "none",
        "completion_policy": "waitsome",
        "initial_condition": diag.PROFILE["initial_condition"],
        "cut_probe_values": probes,
        "cut_probe_l2": probe_l2,
        "ez_l2": 1.2,
        "ez_weighted_checksum": 1.7,
        "energy": 2.1,
        "dft_norm": 0.35,
        "remote_cut_scalars": 120 if multi else 0,
        "remote_cut_l2": 0.42 if multi else 0.0,
        "cpu_calls": 0,
        "cuda_calls": sum(item["cuda_calls"] for item in phase_calls.values()),
        "phase_calls": phase_calls,
        "h2d_bytes": 32,
        "d2h_bytes": 16,
        "mpi_messages": 24 if multi else 0,
        "mpi_scalars": 240 if multi else 0,
        "cuda_aware_bytes": 0,
        "pinned_bytes": 960 if multi else 0,
    }


def json_line(prefix: str, value: dict) -> str:
    return prefix + json.dumps(value, separators=(",", ":")) + "\n"


def feature_records(condition: str, ranks: int = 1) -> dict[str, dict]:
    tile = {
        "curl_chunk_phases": 2,
        "curl_input_tiles": 20,
        "update_eh_chunk_phases": 0,
        "update_eh_input_tiles": 0,
    }
    if diag.CONDITIONS[condition]["tile_expectation"] == "not-coalesced":
        tile = {key: 0 for key in tile}
    multi = ranks == 2
    disabled = condition == "disable-boundary-descriptor-fast-replay"
    return {
        "tile_coalescing": tile,
        "phase_batch_policy": {
            "curl_automatic_checks": 0,
            "curl_automatic_selected": 0,
            "curl_automatic_rejected": 0,
            "curl_forced_batches": 80 * ranks,
            "curl_batched_operations": 640 * ranks,
            "curl_unbatched_operations": 0,
            "update_eh_automatic_checks": 0,
            "update_eh_automatic_selected": 0,
            "update_eh_automatic_rejected": 0,
            "update_eh_forced_batches": 80 * ranks,
            "update_eh_batched_operations": 640 * ranks,
            "update_eh_unbatched_operations": 0,
        },
        "curl_phase_replay": {
            "disabled_ranks": 0,
            "checks": (
                0
                if diag.CONDITIONS[condition]["disable_phase_batched_curl"]
                else 2 * 80 * ranks
            ),
            "hits": (
                0
                if diag.CONDITIONS[condition]["disable_phase_batched_curl"]
                else 2 * 80 * ranks
            ),
            "unready": 0,
            "generation_misses": 0,
            "mirror_misses": 0,
        },
        "boundary_phase_graph": {"creations": 1, "launches": 80},
        "eager_mpi": {
            "receive_start_calls": 0,
            "send_start_calls": 0,
            "receive_requests": 0,
            "send_requests": 0,
        },
        "receive_pingpong": {
            "warmup_secondary_allocations": 0,
            "selections": 0,
            "secondary_selections": 0,
        },
        "boundary_eh_overlap": {
            "checks": 160 * ranks,
            "warmup_skipped_cold_topology": 0,
            "warmup_launched_h": 0,
            "warmup_launched_e": 0,
            "eligible_ranks": ranks if multi else 0,
            "eligible": 160 * ranks if multi else 0,
            "launched_h": 80 * ranks if multi else 0,
            "launched_e": 80 * ranks if multi else 0,
            "skipped_disabled": 0,
            "skipped_unsupported_schedule": 0,
            "skipped_no_remote": 0 if multi else 160 * ranks,
            "skipped_cold_topology": 0,
            "rejected": 0,
        },
        "halo_curl_overlap": {
            "checks": 80 * ranks,
            "warmup_skipped_cold_topology": 0,
            "warmup_launches": 0,
            "eligible": 80 * ranks if multi else 0,
            "launches": 80 * ranks if multi else 0,
            "skipped_disabled": 0,
            "skipped_unsupported_schedule": 0,
            "skipped_no_remote": 0 if multi else 80 * ranks,
            "skipped_cold_topology": 0,
            "rejected_feature": 0,
            "rejected_small": 0,
            "full_points": 0,
            "interior_points": 0,
            "shell_points": 0,
        },
        "dft_phase_sharing": {
            "batch_calls": 80,
            "submitted_updates": 160,
            "phase_preparation_launches": 80,
            "phase_reuses": 80,
            "update_kernel_launches": 160,
            "maximum_batch_size": 2,
        },
        "dft_multi_monitor_batch": {
            "automatic_checks": 80,
            "automatic_selected": 0,
            "automatic_rejected": 80,
            "forced_batches": 0,
            "batched_updates": 0,
            "unbatched_updates": 160,
            "plan_uploads": 0,
            "plan_reuses": 0,
            "metadata_host_to_device_bytes": 0,
        },
        "dft_warmup_plan": {
            "metadata_host_to_device_bytes": 0,
            "plan_reuses": 0,
            "plan_uploads": 0,
        },
        "boundary_descriptor_replay": {
            "disabled": disabled,
            "disabled_ranks": ranks if disabled else 0,
            "gather_fast_replays": 0,
            "gather_full_validations": 0,
            "scatter_fast_replays": 0 if ranks == 1 or disabled else 20,
            "scatter_full_validations": 20 if multi and disabled else 0,
        },
    }


def valid_output(item: dict) -> str:
    lines = [
        json_line(mgb.RECORD_PREFIX, valid_record(
            item["ranks"], pixels=item["pixels"], condition=item["condition"]
        ))
    ]
    for rank in range(item["ranks"]):
        lines.append(
            json_line(
                mgb.DEVICE_RECORD_PREFIX,
                {
                    "rank": rank,
                    "ordinal": rank,
                    "uuid": f"{rank + 1:032x}",
                    "name": "Synthetic GPU",
                    "compute_major": 8,
                    "compute_minor": 6,
                    "compatible": True,
                },
            )
        )
    features = feature_records(item["condition"], item["ranks"])
    for name, (prefix, _) in diag.FEATURE_SCHEMAS.items():
        lines.append(json_line(prefix, features[name]))
    lines.append(
        json_line(
            diag.SOURCE_PROFILE_PREFIX,
            {"profile": diag.PROFILE["source_profile"]},
        )
    )
    lines.append(
        json_line(
            diag.INITIAL_CONDITION_PREFIX,
            {"profile": diag.PROFILE["initial_condition"]},
        )
    )
    lines.append(
        json_line(
            diag.INITIALIZATION_TIMING_PREFIX,
            {"applications_per_rank": 1, "max_seconds": 0.25},
        )
    )
    lines.append("\nField time usage:\n")
    for index, label in enumerate(diag.TIMING_LABELS_BY_RANK[item["ranks"]], 1):
        lines.append(f"    {label}: {0.01 * index:.3f} s\n")
    lines.append("\n")
    return "".join(lines)


def synthetic_sample(item: dict) -> dict:
    records = diag.parse_sample(valid_output(item), item)
    return {
        "schedule": item,
        "records": records,
        "process_wall_seconds": (
            records["benchmark"]["seconds"] + 0.5
        ),
    }


class FixedProfileTests(unittest.TestCase):
    def test_profile_hash_and_release_separation_are_fixed(self) -> None:
        diag.assert_fixed_profile()
        self.assertEqual(diag.PROFILE["evidence_class"], "development-diagnostic-only")
        self.assertFalse(diag.PROFILE["release_qualification"])
        self.assertEqual(diag.PROFILE["source_profile"], "dual-electric-v1")

    def test_schedule_is_complete_balanced_and_unique(self) -> None:
        schedule = diag.expected_schedule()
        self.assertEqual(len(schedule), 234)
        self.assertEqual(
            [item["position"] for item in schedule], list(range(1, 235))
        )
        identities = {
            (
                item["matrix"], item["pixels"], item["condition"],
                item["repeat"], item["ranks"],
            )
            for item in schedule
        }
        self.assertEqual(len(identities), len(schedule))
        for pixels in (64, 96, 128, 160, 224, 256):
            for condition in diag.PHASE_SCREEN_CONDITION_ORDER:
                for repeat in (1, 2, 3):
                    for ranks in (1, 2):
                        self.assertIn(
                            (
                                "size-screen", pixels, condition,
                                repeat, ranks,
                            ),
                            identities,
                        )
        for condition in diag.CONDITION_ORDER:
            for repeat in (1, 2, 3):
                for ranks in (1, 2):
                    self.assertIn(
                        (
                            "causal-controls", 192, condition, repeat, ranks
                        ),
                        identities,
                    )

    def test_runner_reuses_contained_executor(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("MGB.run_logged", source)
        self.assertNotIn("subprocess.run", source)
        self.assertNotIn("subprocess.Popen", source)

    def test_even_repeat_reverses_ranks_and_boundary_ab_order(self) -> None:
        causal = [
            item for item in diag.expected_schedule()
            if item["matrix"] == "causal-controls"
        ]
        for repeat, expected_ranks in (
            (1, [1, 2]), (2, [2, 1]), (3, [1, 2])
        ):
            selected = [item for item in causal if item["repeat"] == repeat]
            for offset in range(0, len(selected), 2):
                self.assertEqual(
                    [item["ranks"] for item in selected[offset:offset + 2]],
                    expected_ranks,
                )
        self.assertEqual(
            [item["condition"] for item in causal[:4:2]],
            ["default", "disable-boundary-descriptor-fast-replay"],
        )
        repeat_two = [item for item in causal if item["repeat"] == 2]
        self.assertEqual(
            [item["condition"] for item in repeat_two[:4:2]],
            ["disable-boundary-descriptor-fast-replay", "default"],
        )

    def test_size_screen_reverses_condition_order_with_repeat(self) -> None:
        screen = [
            item for item in diag.expected_schedule()
            if item["matrix"] == "size-screen"
        ]
        for repeat, expected in (
            (1, list(diag.PHASE_SCREEN_CONDITION_ORDER)),
            (2, list(reversed(diag.PHASE_SCREEN_CONDITION_ORDER))),
            (3, list(diag.PHASE_SCREEN_CONDITION_ORDER)),
        ):
            first_pixels = next(
                item["pixels"] for item in screen if item["repeat"] == repeat
            )
            selected = [
                item for item in screen
                if item["repeat"] == repeat
                and item["pixels"] == first_pixels
                and item["ranks"] == (1 if repeat % 2 else 2)
            ]
            self.assertEqual(
                [item["condition"] for item in selected], expected
            )


class ConditionEnvironmentTests(unittest.TestCase):
    def test_every_causal_condition_has_exact_toggle_semantics(self) -> None:
        base = {"HOME": "/isolated"}
        expected = {
            "default": {
                "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
            },
            "disable-phase-batched-curl": {
                "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
                "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL"
            },
            "disable-phase-batched-update-eh": {
                "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
                "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH"
            },
            "disable-both-phase-batches": {
                "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
                "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH",
            },
            "disable-phase-batched-source": {
                "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
                "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE",
            },
            "disable-tile-coalescing": {
                "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
                "MEEP_GPU_DISABLE_TILE_COALESCING"
            },
            "disable-all-three": {
                "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
                "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH",
                "MEEP_GPU_DISABLE_TILE_COALESCING",
            },
            "no-tile-reference": {
                "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
            },
            "disable-boundary-descriptor-fast-replay": {
                "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
                "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
                "MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY",
            },
        }
        toggle_keys = {
            "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
            "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
            "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
            "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL",
            "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH",
            "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE",
            "MEEP_GPU_DISABLE_TILE_COALESCING",
            "MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY",
        }
        for condition, toggles in expected.items():
            with self.subTest(condition=condition):
                environment = diag.condition_environment(base, condition, 192, 2)
                self.assertEqual(toggle_keys & environment.keys(), toggles)
                self.assertEqual(
                    environment["MEEP_GPU_MULTI_LOOP_TILE_BASE_DB"],
                    str(diag.CONDITIONS[condition]["loop_tile_base_db"]),
                )
                self.assertEqual(
                    environment["MEEP_GPU_MULTI_INITIAL_CONDITION"], "affine-v1"
                )
                self.assertEqual(
                    environment["MEEP_GPU_MULTI_SOURCE_PROFILE"],
                    "dual-electric-v1",
                )
                self.assertEqual(
                    environment["MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE"], "1"
                )
                expectations = {
                    "MEEP_GPU_EXPECT_TILE_COALESCING",
                    "MEEP_GPU_EXPECT_NO_TILE_COALESCING",
                } & environment.keys()
                self.assertEqual(len(expectations), 1)
        no_tile = diag.condition_environment(base, "no-tile-reference", 192, 2)
        self.assertNotIn("MEEP_GPU_DISABLE_TILE_COALESCING", no_tile)
        self.assertEqual(no_tile["MEEP_GPU_MULTI_LOOP_TILE_BASE_DB"], "0")

    def test_boundary_descriptor_worker_gates_are_rank_and_condition_exact(self) -> None:
        default_two = diag.condition_environment({}, "default", 192, 2)
        self.assertEqual(
            default_two["MEEP_GPU_EXPECT_BOUNDARY_DESCRIPTOR_FAST_REPLAY"], "1"
        )
        self.assertNotIn(
            "MEEP_GPU_EXPECT_BOUNDARY_GATHER_DESCRIPTOR_FAST_REPLAY", default_two
        )
        disabled_two = diag.condition_environment(
            {}, "disable-boundary-descriptor-fast-replay", 192, 2
        )
        self.assertEqual(
            disabled_two["MEEP_GPU_EXPECT_NO_BOUNDARY_DESCRIPTOR_FAST_REPLAY"], "1"
        )
        disabled_one = diag.condition_environment(
            {}, "disable-boundary-descriptor-fast-replay", 192, 1
        )
        self.assertNotIn(
            "MEEP_GPU_EXPECT_NO_BOUNDARY_DESCRIPTOR_FAST_REPLAY", disabled_one
        )

    def test_ld_preload_is_scoped_to_the_mpi_worker_not_mpiexec(self) -> None:
        command = diag.isolated_command(
            {"HOME": "/isolated"},
            pathlib.Path("/proc/10/fd/3"),
            pathlib.Path("/proc/10/fd/4"),
            pathlib.Path("/proc/10/fd/5"),
            2,
        )
        mpiexec_index = command.index("/proc/10/fd/3")
        preload_index = command.index("LD_PRELOAD=/proc/10/fd/5")
        self.assertGreater(preload_index, mpiexec_index)
        self.assertEqual(command[preload_index - 1], "/usr/bin/env")
        host_index = command.index("--host")
        self.assertEqual(command[host_index + 1], "localhost:2")
        self.assertNotIn("localhost", command)

        singleton = diag.isolated_command(
            {"HOME": "/isolated"},
            pathlib.Path("/proc/10/fd/3"),
            pathlib.Path("/proc/10/fd/4"),
            pathlib.Path("/proc/10/fd/5"),
            1,
        )
        singleton_host_index = singleton.index("--host")
        self.assertEqual(singleton[singleton_host_index + 1], "localhost:1")


class StrictParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.item = {
            "position": 1,
            "matrix": "size-screen",
            "pixels": 192,
            "condition": "default",
            "repeat": 1,
            "ranks": 2,
        }

    def test_complete_sample_binds_worker_devices_features_and_times(self) -> None:
        parsed = diag.parse_sample(valid_output(self.item), self.item)
        self.assertEqual(parsed["benchmark"]["mpi_ranks"], 2)
        self.assertEqual([item["rank"] for item in parsed["devices"]], [0, 1])
        self.assertEqual(
            set(parsed["feature_markers"]), set(diag.FEATURE_SCHEMAS)
        )
        self.assertEqual(parsed["initial_condition"]["profile"], "affine-v1")
        self.assertEqual(
            parsed["initialization_timing"],
            {"applications_per_rank": 1, "max_seconds": 0.25},
        )
        self.assertIn("time stepping", parsed["print_times"])

    def test_no_tile_worker_profile_is_strictly_accepted(self) -> None:
        item = dict(self.item, condition="no-tile-reference")
        parsed = diag.parse_sample(valid_output(item), item)
        self.assertEqual(parsed["benchmark"]["loop_tile_base_db"], 0)
        self.assertTrue(
            all(value == 0 for value in parsed["feature_markers"]["tile_coalescing"].values())
        )

    def test_incomplete_expected_worker_profile_is_rejected(self) -> None:
        record = valid_record()
        encoded = json_line(mgb.RECORD_PREFIX, record)
        with self.assertRaisesRegex(mgb.EvidenceError, "exactly the fixed worker keys"):
            mgb.extract_record(encoded, expected_worker_profile={"loop_tile_base_db": 128})

    def test_legacy_v3_worker_record_fails_closed(self) -> None:
        record = valid_record(2)
        record["schema_version"] = 3
        current = json_line(mgb.RECORD_PREFIX, valid_record(2))
        legacy = json_line("gpmeep-multi-benchmark-v3:", record)
        output = valid_output(self.item).replace(current, legacy)
        with self.assertRaises(diag.DiagnosticError):
            diag.parse_sample(output, self.item)

    def test_duplicate_feature_marker_is_rejected(self) -> None:
        output = valid_output(self.item)
        tile = feature_records("default")["tile_coalescing"]
        output += json_line(diag.TILE_PREFIX, tile)
        with self.assertRaisesRegex(diag.DiagnosticError, "exactly one tile"):
            diag.parse_sample(output, self.item)

    def test_unknown_gpmeep_marker_is_rejected(self) -> None:
        with self.assertRaisesRegex(diag.DiagnosticError, "unknown gpmeep marker"):
            diag.parse_sample(
                valid_output(self.item) + "gpmeep-surprise-v1:{}\n", self.item
            )

    def test_initial_condition_and_setup_timing_are_exact_and_bound(self) -> None:
        output = valid_output(self.item)
        missing = "".join(
            line
            for line in output.splitlines(keepends=True)
            if not line.startswith(diag.INITIAL_CONDITION_PREFIX)
        )
        with self.assertRaisesRegex(diag.DiagnosticError, "exactly one initial"):
            diag.parse_sample(missing, self.item)

        wrong_marker = output.replace(
            json_line(
                diag.INITIAL_CONDITION_PREFIX,
                {"profile": diag.PROFILE["initial_condition"]},
            ),
            json_line(
                diag.INITIAL_CONDITION_PREFIX,
                {"profile": "trigonometric-v1"},
            ),
        )
        with self.assertRaisesRegex(diag.DiagnosticError, "marker, worker record"):
            diag.parse_sample(wrong_marker, self.item)

        for timing in (
            {"applications_per_rank": 2, "max_seconds": 0.25},
            {"applications_per_rank": 1, "max_seconds": 0.0},
            {"applications_per_rank": 1, "max_seconds": "0.25"},
        ):
            candidate = output.replace(
                json_line(
                    diag.INITIALIZATION_TIMING_PREFIX,
                    {"applications_per_rank": 1, "max_seconds": 0.25},
                ),
                json_line(diag.INITIALIZATION_TIMING_PREFIX, timing),
            )
            with self.subTest(timing=timing):
                with self.assertRaises(diag.DiagnosticError):
                    diag.parse_sample(candidate, self.item)
        with self.assertRaisesRegex(diag.DiagnosticError, "unknown gpmeep marker"):
            diag.reject_unknown_gpmeep_streams(
                valid_output(self.item), "gpmeep-stderr-surprise-v1:{}\n"
            )

    def test_source_profile_marker_is_exact_and_bound(self) -> None:
        output = valid_output(self.item)
        source_line = json_line(
            diag.SOURCE_PROFILE_PREFIX,
            {"profile": diag.PROFILE["source_profile"]},
        )
        for candidate, message in (
            (output.replace(source_line, ""), "exactly one source profile"),
            (output.replace(source_line, source_line * 2), "exactly one source profile"),
            (
                output.replace(
                    source_line,
                    json_line(
                        diag.SOURCE_PROFILE_PREFIX,
                        {"profile": "single-ez-v1"},
                    ),
                ),
                "fixed diagnostic profile",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(diag.DiagnosticError, message):
                    diag.parse_sample(candidate, self.item)

    def test_feature_schema_and_nonnegative_counters_are_exact(self) -> None:
        prefix, keys = diag.FEATURE_SCHEMAS["tile_coalescing"]
        record = feature_records("default")["tile_coalescing"]
        for mutation in ("missing", "extra", "negative"):
            candidate = dict(record)
            if mutation == "missing":
                candidate.pop(next(iter(keys)))
            elif mutation == "extra":
                candidate["extra"] = 0
            else:
                candidate["curl_chunk_phases"] = -1
            with self.subTest(mutation=mutation):
                with self.assertRaises(diag.DiagnosticError):
                    diag.exactly_one_feature(
                        json_line(prefix, candidate), prefix, "tile", keys
                    )

    def test_tile_expectation_mismatch_is_rejected(self) -> None:
        zero = {key: 0 for key in feature_records("default")["tile_coalescing"]}
        with self.assertRaisesRegex(diag.DiagnosticError, "did not execute"):
            diag.validate_tile_record("default", zero)
        with self.assertRaisesRegex(diag.DiagnosticError, "reported coalescing"):
            diag.validate_tile_record(
                "disable-tile-coalescing",
                feature_records("default")["tile_coalescing"],
            )

    def test_feature_relations_are_fail_closed(self) -> None:
        benchmark = valid_record(2)
        base = feature_records("default", 2)
        mutations = []
        boundary = copy.deepcopy(base)
        boundary["boundary_eh_overlap"]["checks"] -= 1
        mutations.append(boundary)
        halo = copy.deepcopy(base)
        halo["halo_curl_overlap"]["full_points"] = 1
        mutations.append(halo)
        dft = copy.deepcopy(base)
        dft["dft_phase_sharing"]["submitted_updates"] -= 1
        mutations.append(dft)
        eager = copy.deepcopy(base)
        eager["eager_mpi"]["receive_start_calls"] = 1
        mutations.append(eager)
        pingpong = copy.deepcopy(base)
        pingpong["receive_pingpong"]["secondary_selections"] = 1
        mutations.append(pingpong)
        replay_conservation = copy.deepcopy(base)
        replay_conservation["curl_phase_replay"]["hits"] -= 1
        mutations.append(replay_conservation)
        replay_optout = copy.deepcopy(base)
        replay_optout["curl_phase_replay"]["disabled_ranks"] = 1
        mutations.append(replay_optout)
        replay_no_hit = copy.deepcopy(base)
        replay_no_hit["curl_phase_replay"]["unready"] = (
            replay_no_hit["curl_phase_replay"]["checks"]
        )
        replay_no_hit["curl_phase_replay"]["hits"] = 0
        mutations.append(replay_no_hit)
        for features in mutations:
            with self.subTest(features=features):
                with self.assertRaises(diag.DiagnosticError):
                    diag.validate_feature_contract(self.item, benchmark, features)

        disabled_item = dict(self.item, condition="disable-phase-batched-curl")
        disabled_features = feature_records(disabled_item["condition"], 2)
        disabled_features["curl_phase_replay"]["checks"] = 1
        disabled_features["curl_phase_replay"]["unready"] = 1
        with self.assertRaisesRegex(diag.DiagnosticError, "disabled curl batching"):
            diag.validate_feature_contract(
                disabled_item,
                valid_record(2, condition=disabled_item["condition"]),
                disabled_features,
            )

    def test_boundary_descriptor_v1_boolean_and_optout_semantics_are_exact(self) -> None:
        prefix, keys = diag.FEATURE_SCHEMAS["boundary_descriptor_replay"]
        record = feature_records("default", 2)["boundary_descriptor_replay"]
        bad_type = dict(record, disabled=0)
        with self.assertRaisesRegex(diag.DiagnosticError, "must be boolean"):
            diag.exactly_one_feature(
                json_line(prefix, bad_type), prefix,
                "boundary_descriptor_replay", keys,
            )
        disabled_item = dict(
            self.item, condition="disable-boundary-descriptor-fast-replay"
        )
        disabled = feature_records(disabled_item["condition"], 2)
        disabled["boundary_descriptor_replay"]["disabled_ranks"] = 1
        with self.assertRaisesRegex(diag.DiagnosticError, "full validation"):
            diag.validate_feature_contract(
                disabled_item, valid_record(2, condition=disabled_item["condition"]),
                disabled,
            )

    def test_two_rank_no_remote_counters_are_rejected_even_if_partition_balances(self) -> None:
        for feature_name in ("boundary_eh_overlap", "halo_curl_overlap"):
            features = feature_records("default", 2)
            feature = features[feature_name]
            feature["eligible"] -= 1
            if feature_name == "boundary_eh_overlap":
                feature["launched_e"] -= 1
            feature["skipped_no_remote"] += 1
            with self.subTest(feature=feature_name):
                with self.assertRaisesRegex(diag.DiagnosticError, "no-remote"):
                    diag.validate_feature_contract(
                        self.item, valid_record(2), features
                    )

    def test_print_times_requires_the_exact_rank_specific_label_order(self) -> None:
        valid = valid_output(self.item)
        parsed = diag.parse_sample(valid, self.item)
        self.assertEqual(
            tuple(parsed["print_times"]), diag.TIMING_LABELS_BY_RANK[2]
        )
        zero_deviation = valid.replace(
            "    time stepping: 0.010 s\n",
            "    time stepping: 0.010 s +/- 0 s\n",
        )
        self.assertEqual(
            diag.parse_sample(zero_deviation, self.item)["print_times"]
            ["time stepping"]["stddev_seconds"],
            0.0,
        )
        swapped = valid.replace(
            "    copying boundaries: 0.020 s\n"
            "    1-1 communication: 0.030 s\n",
            "    1-1 communication: 0.030 s\n"
            "    copying boundaries: 0.020 s\n",
        )
        with self.assertRaisesRegex(diag.DiagnosticError, "labels/order"):
            diag.parse_sample(swapped, self.item)

    def test_print_times_rejects_missing_duplicate_and_malformed_rows(self) -> None:
        cases = (
            "Field time usage:\n    copying boundaries: 0.1 s\n\n",
            "Field time usage:\n    time stepping: 0.1 s\n    time stepping: 0.2 s\n\n",
            "Field time usage:\n    time stepping: nan s\n\n",
            "Field time usage:\n    unknown timer: 0.1 s\n\n",
        )
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(diag.DiagnosticError):
                    diag.parse_print_times(value, 2)


class MatrixValidationTests(unittest.TestCase):
    def complete_samples(self) -> list[dict]:
        return [synthetic_sample(item) for item in diag.expected_schedule()]

    def test_complete_matrix_summarizes_without_release_claim(self) -> None:
        summary = diag.summarize(self.complete_samples())
        self.assertEqual(summary["sample_count"], 234)
        self.assertFalse(summary["release_qualification"]["evaluated"])
        self.assertFalse(summary["release_qualification"]["pass"])
        self.assertEqual(
            summary["rank_device_uuid_mapping"],
            {"1": [f"{1:032x}"], "2": [f"{1:032x}", f"{2:032x}"]},
        )
        self.assertEqual(
            summary["rank_device_identity_mapping"]["1"][0]["name"],
            "Synthetic GPU",
        )
        self.assertEqual(len(summary["configurations"]), 39)
        self.assertEqual(
            len(summary["phase_batch_crossover"]["effects"]), 84
        )
        self.assertEqual(
            len(summary["phase_batch_crossover"]["policy_candidates"]), 12
        )
        self.assertTrue(
            all(
                item["classification"] == "indeterminate"
                for item in summary["phase_batch_crossover"]["effects"]
            )
        )
        binding = summary["phase_batch_crossover"]["hardware_binding"]
        self.assertEqual(
            binding["rank_device_identity_mapping"],
            summary["rank_device_identity_mapping"],
        )
        unsigned = dict(binding)
        digest = unsigned.pop("binding_sha256")
        self.assertEqual(digest, diag.canonical_sha256(unsigned))
        self.assertIn("fresh crossover run", binding["reuse_contract"])
        markdown = diag.markdown_report(summary)
        self.assertIn(
            "# C++ multi-GPU phase-batch crossover diagnostic", markdown
        )
        self.assertIn(binding["binding_sha256"], markdown)
        self.assertIn("Synthetic GPU", markdown)
        self.assertIn("## Phase-batch crossover", markdown)
        self.assertIn("`curl-and-update-eh-combined`", markdown)
        self.assertIn("`indeterminate`", markdown)
        self.assertIn("## Hardware-bound policy candidates", markdown)
        self.assertIn("`leave-phase-batching-disabled`", markdown)
        self.assertEqual(
            summary["timing_domains"]["initialize_field_setup_seconds"],
            "untimed setup; never included in FDTD speedup",
        )
        for configuration in summary["configurations"]:
            for rank_summary in configuration["rank_summaries"].values():
                self.assertEqual(rank_summary["initial_condition"], "affine-v1")
                self.assertEqual(
                    rank_summary["median_initialize_field_setup_seconds"], 0.25
                )

    def test_changed_physical_uuid_mapping_is_rejected(self) -> None:
        samples = self.complete_samples()
        samples[-1]["records"]["devices"][0]["uuid"] = f"{9:032x}"
        with self.assertRaisesRegex(diag.DiagnosticError, "UUID mapping changed"):
            diag.validate_matrix(samples)

    def test_changed_physical_capability_is_rejected(self) -> None:
        samples = self.complete_samples()
        samples[-1]["records"]["devices"][0]["compute_minor"] = 9
        with self.assertRaisesRegex(diag.DiagnosticError, "identity mapping"):
            diag.validate_matrix(samples)

    def test_changed_observable_is_rejected(self) -> None:
        samples = self.complete_samples()
        target = next(
            sample
            for sample in samples
            if sample["schedule"]["condition"] != "default"
        )
        target["records"]["benchmark"]["energy"] *= 2
        with self.assertRaisesRegex(diag.DiagnosticError, "physical observable"):
            diag.validate_matrix(samples)

    def test_reordered_matrix_is_rejected(self) -> None:
        samples = self.complete_samples()
        samples[0], samples[1] = samples[1], samples[0]
        with self.assertRaisesRegex(diag.DiagnosticError, "reordered"):
            diag.validate_matrix(samples)

    def test_every_phase_batch_causal_class_is_checked(self) -> None:
        samples = self.complete_samples()
        target = next(
            sample for sample in samples
            if sample["schedule"]["condition"] == "disable-tile-coalescing"
            and sample["schedule"]["repeat"] == 1
            and sample["schedule"]["ranks"] == 2
        )
        target["records"]["benchmark"]["phase_calls"]["curl"]["cuda_calls"] += 1
        with self.assertRaisesRegex(diag.DiagnosticError, "causal class"):
            diag.validate_matrix(samples)

    def test_source_batching_requires_exact_two_to_one_call_ratio(self) -> None:
        samples = self.complete_samples()
        target = next(
            sample for sample in samples
            if sample["schedule"]["matrix"] == "causal-controls"
            and sample["schedule"]["condition"] == "disable-phase-batched-source"
            and sample["schedule"]["repeat"] == 1
            and sample["schedule"]["ranks"] == 2
        )
        target["records"]["benchmark"]["phase_calls"]["source"][
            "cuda_calls"
        ] += 1
        with self.assertRaisesRegex(diag.DiagnosticError, "causal call counts"):
            diag.validate_matrix(samples)


class PhaseBatchCrossoverTests(unittest.TestCase):
    def test_effect_classification_requires_every_paired_repeat(self) -> None:
        self.assertEqual(
            diag.classify_phase_effect([1.04, 1.05, 1.06]),
            "phase-batched-faster",
        )
        self.assertEqual(
            diag.classify_phase_effect([0.95, 0.96, 0.94]),
            "per-operation-faster",
        )
        self.assertEqual(
            diag.classify_phase_effect([1.04, 1.01, 1.05]),
            "indeterminate",
        )
        with self.assertRaisesRegex(diag.DiagnosticError, "paired sample"):
            diag.classify_phase_effect([1.1, 1.1])

    @staticmethod
    def configurations(*, one_point_only: bool = False) -> list[dict]:
        values = []
        large = {256} if one_point_only else {224, 256}
        for pixels in diag.PROFILE["size_screen_pixels"]:
            for condition in diag.PHASE_SCREEN_CONDITION_ORDER:
                by_rank = {}
                for ranks in diag.PROFILE["ranks"]:
                    seconds = 1.0
                    if pixels in large and condition == "default":
                        seconds = 0.8
                    by_rank[str(ranks)] = {
                        "median_seconds": seconds,
                        "seconds_by_repeat": {
                            str(repeat): seconds
                            for repeat in range(1, diag.PROFILE["repeats"] + 1)
                        },
                    }
                values.append(
                    {
                        "pixels": pixels,
                        "condition": condition,
                        "rank_summaries": by_rank,
                    }
                )
        return values

    def test_candidate_threshold_requires_two_point_stable_suffix(self) -> None:
        summary = diag.summarize_phase_batch_crossover(self.configurations())
        candidate = next(
            item for item in summary["policy_candidates"]
            if item["effect"] == "curl-and-update-eh-combined"
            and item["ranks"] == 1
        )
        self.assertEqual(candidate["status"], "OBSERVED_THRESHOLD_CANDIDATE")
        self.assertEqual(candidate["minimum_pixels"], 224)
        self.assertEqual(candidate["minimum_nominal_local_cells"], 224**3)
        self.assertEqual(len(candidate["supporting_points"]), 2)

        one_point = diag.summarize_phase_batch_crossover(
            self.configurations(one_point_only=True)
        )
        rejected = next(
            item for item in one_point["policy_candidates"]
            if item["effect"] == "curl-and-update-eh-combined"
            and item["ranks"] == 1
        )
        self.assertEqual(rejected["status"], "NO_STABLE_BATCHED_SUFFIX")
        self.assertEqual(
            rejected["recommendation"], "leave-phase-batching-disabled"
        )


class EvidenceLifecycleTests(unittest.TestCase):
    def test_stable_snapshot_binds_exact_payload_and_detects_later_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source.py"
            snapshot = root / "archive" / "source.py"
            snapshot.parent.mkdir()
            source.write_text("value = 1\n", encoding="utf-8")
            binding = diag.stable_snapshot(source, snapshot, root)
            self.assertEqual(snapshot.read_bytes(), source.read_bytes())
            diag.verify_snapshot_window(binding, snapshot, root)
            source.write_text("value = 2\n", encoding="utf-8")
            with self.assertRaises(diag.SourceWindowError):
                diag.verify_snapshot_window(binding, snapshot, root)

    def test_failed_marker_is_atomic_hash_bound_and_never_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            (output / "partial.log").write_text("observed\n", encoding="utf-8")
            try:
                raise diag.SourceWindowError("source moved")
            except diag.SourceWindowError as error:
                diag.seal_failed(
                    output,
                    error,
                    run_id="diagnostic-test",
                    expected_receipt_id="a" * 64,
                    verified_receipt_id=None,
                )
            self.assertFalse((output / "COMPLETE.json").exists())
            marker = json.loads((output / "FAILED.json").read_text())
            report = json.loads((output / "failure-report.json").read_text())
            manifest = json.loads((output / "artifacts.sha256.json").read_text())
            self.assertEqual(marker["state"], "FAILED")
            terminal = json.loads((output / "TERMINAL.json").read_text())
            self.assertEqual(terminal["state"], "FAILED")
            self.assertEqual(
                marker["terminal_claim"],
                diag.stable_file_record(output / "TERMINAL.json", output),
            )
            self.assertFalse(report["source_identity_proven"])
            self.assertTrue(marker["complete_marker_absent"])
            self.assertTrue(manifest["complete"])
            self.assertEqual(
                marker["failure_report"],
                diag.stable_file_record(output / "failure-report.json", output),
            )
            self.assertEqual(
                marker["manifest"],
                diag.stable_file_record(output / "artifacts.sha256.json", output),
            )

    def test_recorded_artifact_tamper_is_rejected_before_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            artifact = output / "sample.log"
            artifact.write_text("original\n", encoding="utf-8")
            record = diag.stable_file_record(artifact, output)
            diag.verify_recorded_file(record, output, "sample log")
            artifact.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(diag.DiagnosticError):
                diag.verify_recorded_file(record, output, "sample log")

    def test_binary_snapshot_binds_bytes_and_permissions_and_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "worker"
            destination = root / "archive/worker"
            destination.parent.mkdir()
            source.write_bytes(b"synthetic ELF payload")
            os.chmod(source, 0o751)
            binding = diag.stable_binary_snapshot(
                source, destination, root, executable_required=True
            )
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(destination.stat().st_mode & 0o7777, 0o751)
            diag.verify_binary_snapshot(binding, destination, root)
            destination.write_bytes(b"tampered")
            with self.assertRaises(diag.SourceWindowError):
                diag.verify_binary_snapshot(binding, destination, root)

    def test_persistent_execution_handle_is_read_only_and_inode_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            snapshot = root / "worker"
            snapshot.write_bytes(b"immutable execution bytes")
            os.chmod(snapshot, 0o751)
            descriptor, binding = diag.create_stable_execution_handle(
                snapshot, root, "unit-worker"
            )
            try:
                diag.verify_stable_execution_handle(binding, descriptor)
                with self.assertRaises(OSError):
                    os.pwrite(descriptor, b"X", 0)
                self.assertEqual(binding["mode"], 0o751)
                self.assertEqual(
                    binding["source_snapshot"],
                    diag.stable_file_record(snapshot, root),
                )
            finally:
                os.close(descriptor)

    def test_persistent_handle_keeps_original_on_path_replace_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            snapshot = root / "worker"
            snapshot.write_bytes(b"receipt worker bytes")
            os.chmod(snapshot, 0o755)
            descriptor, binding = diag.create_stable_execution_handle(
                snapshot, root, "replace-restore-worker"
            )
            try:
                replacement = root / "replacement"
                replacement.write_bytes(b"attacker worker bytes")
                os.chmod(replacement, 0o755)
                os.replace(replacement, snapshot)
                snapshot.write_bytes(b"receipt worker bytes")
                os.chmod(snapshot, 0o755)
                self.assertEqual(
                    os.pread(descriptor, binding["size_bytes"], 0),
                    b"receipt worker bytes",
                )
                self.assertNotEqual(snapshot.stat().st_ino, binding["inode"])
                with self.assertRaisesRegex(
                    diag.SourceWindowError, "stable execution handle changed"
                ):
                    diag.verify_stable_execution_handle(binding, descriptor)
            finally:
                os.close(descriptor)

    def test_in_place_equal_length_write_restore_changes_ctime_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            snapshot = root / "worker"
            original = b"receipt-worker"
            malicious = b"attacker-bytes"
            self.assertEqual(len(original), len(malicious))
            snapshot.write_bytes(original)
            os.chmod(snapshot, 0o755)
            descriptor, binding = diag.create_stable_execution_handle(
                snapshot, root, "in-place-worker"
            )
            try:
                snapshot.write_bytes(malicious)
                self.assertEqual(os.pread(descriptor, len(malicious), 0), malicious)
                snapshot.write_bytes(original)
                os.chmod(snapshot, 0o755)
                self.assertEqual(os.pread(descriptor, len(original), 0), original)
                self.assertNotEqual(snapshot.stat().st_ctime_ns, binding["ctime_ns"])
                with self.assertRaisesRegex(
                    diag.SourceWindowError, "stable execution handle changed"
                ):
                    diag.verify_stable_execution_handle(binding, descriptor)
            finally:
                os.close(descriptor)

    def test_loader_trace_requires_fixed_fd_preload_and_rejects_rpath_original(self) -> None:
        execution = pathlib.Path("/proc/123/fd/9")
        original = pathlib.Path("/build/src/.libs/libmeep.so.38.0.0")
        accepted = diag.validate_loader_trace(
            f"\t{execution} (0x1)\n", execution, original, "libmeep.so.38"
        )
        self.assertIn(str(execution), accepted["resolved_preload_line"])
        with self.assertRaisesRegex(diag.DiagnosticError, "fixed open image"):
            diag.validate_loader_trace(
                "\tlibmeep.so.38 => /build/src/.libs/libmeep.so.38 (0x1)\n",
                execution,
                original,
                "libmeep.so.38",
            )

    def test_output_claim_and_manifest_reject_reuse_symlinks_and_hardlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "result"
            diag.claim_output_directory(output)
            with self.assertRaisesRegex(diag.DiagnosticError, "must be absent"):
                diag.claim_output_directory(output)
            symlink = output / "escape"
            symlink.symlink_to(root)
            with self.assertRaisesRegex(diag.DiagnosticError, "symlink"):
                diag.build_manifest(output)
            symlink.unlink()
            source = root / "external"
            source.write_text("external\n", encoding="utf-8")
            os.link(source, output / "hardlink")
            with self.assertRaisesRegex(diag.DiagnosticError, "hard links"):
                diag.build_manifest(output)

    def test_output_claim_rejects_a_symlink_in_any_lexical_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            real = root / "real"
            (real / "sub").mkdir(parents=True)
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(diag.DiagnosticError, "canonical"):
                diag.claim_output_directory(link / "sub/result")

    def test_provenance_receipt_verification_reuses_proven_qualification_module(self) -> None:
        self.assertIs(
            diag.PROVENANCE._load_qualification_contract_from_source(),
            diag.QUALIFICATION,
        )
        with tempfile.TemporaryDirectory() as temporary:
            fake_adjacent = pathlib.Path(temporary) / "gpmeep_qualification_contract.py"
            fake_adjacent.write_text(
                "def validate_v2_receipt(*args): return None\n", encoding="utf-8"
            )
            original_file = diag.PROVENANCE.__file__
            try:
                diag.PROVENANCE.__file__ = str(pathlib.Path(temporary) / "provenance.py")
                self.assertIs(
                    diag.PROVENANCE._load_qualification_contract_from_source(),
                    diag.QUALIFICATION,
                )
            finally:
                diag.PROVENANCE.__file__ = original_file

    def test_complete_and_failed_markers_are_exclusive_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            diag.exclusive_publish_json(output / "COMPLETE.json", {"state": "COMPLETE"})
            with self.assertRaisesRegex(diag.DiagnosticError, "already exists"):
                diag.exclusive_publish_json(
                    output / "COMPLETE.json", {"state": "replacement"}
                )
            with self.assertRaisesRegex(diag.DiagnosticError, "COMPLETE"):
                diag.seal_failed(
                    output,
                    diag.DiagnosticError("late failure"),
                    run_id="diagnostic-test",
                    expected_receipt_id="a" * 64,
                    verified_receipt_id="a" * 64,
                )

    def test_terminal_claim_allows_only_one_state_publisher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            complete = {
                "schema_version": 1,
                "state": "COMPLETE",
                "run_id": "terminal-race-test",
                "expected_receipt_id": "a" * 64,
                "verified_receipt_id": "a" * 64,
            }
            diag.publish_terminal_marker(output, "COMPLETE", complete)
            with self.assertRaisesRegex(diag.DiagnosticError, "opposite"):
                diag.publish_terminal_marker(
                    output,
                    "FAILED",
                    dict(complete, state="FAILED"),
                )
            self.assertTrue((output / "COMPLETE.json").is_file())
            self.assertFalse((output / "FAILED.json").exists())

    def test_archived_inner_reexec_proves_exact_transitive_source_before_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "result"
            diag.claim_output_directory(output)
            run_id = "archived-inner-self-test"
            capability = "1" * 64
            capability_sha256 = hashlib.sha256(capability.encode()).hexdigest()
            diag.exclusive_publish_json(
                output / "RUNNING.json",
                {
                    "schema_version": diag.DIAGNOSTIC_SCHEMA_VERSION,
                    "state": "CLAIMED",
                    "run_id": run_id,
                    "created_at_utc": diag.utc_now(),
                    "outer_pid": os.getpid(),
                    "capability_sha256": capability_sha256,
                },
            )
            bindings, runtime_identity, archived_runner = diag._source_snapshots(output)
            receipt_source = root / "receipt.json"
            receipt_source.write_text("{}\n", encoding="utf-8")
            receipt_snapshot = output / "archive/input/build-provenance.json"
            receipt_snapshot.parent.mkdir()
            receipt_binding = diag.stable_snapshot(
                receipt_source, receipt_snapshot, output
            )
            control_python_path = pathlib.Path(sys.executable).resolve()
            invocation = {
                "schema_version": diag.DIAGNOSTIC_SCHEMA_VERSION,
                "run_id": run_id,
                "outer_pid": os.getpid(),
                "output": str(output),
                "repo": str(diag.REPO.resolve()),
                "expected_receipt_id": "a" * 64,
                "requested_mpiexec": str((root / "mpiexec").resolve()),
                "receipt_snapshot": receipt_binding,
                "source_bindings": bindings,
                "runtime_source_identity": runtime_identity,
                "capability_sha256": capability_sha256,
                "bootstrap_sha256": diag.INNER_BOOTSTRAP_SHA256,
                "control_python": diag.stable_open_identity(control_python_path),
                "authoritative_inner_stage": {
                    "mechanism": (
                        "isolated-bootstrap-compile-exec-archived-source"
                    ),
                    "bootstrap_sha256": diag.INNER_BOOTSTRAP_SHA256,
                    "runner": bindings["runner"]["observed_snapshot"],
                    "transitive_dependencies": {
                        name: bindings[name]["observed_snapshot"]
                        for name in sorted(bindings)
                        if name != "runner"
                    },
                },
            }
            invocation_path = output / "archive/control/inner-invocation.json"
            invocation_path.parent.mkdir()
            diag.atomic_write_json(invocation_path, invocation)
            transport_log = root / "inner-transport.log"
            run = mgb.run_logged(
                [
                    str(control_python_path), "-I", "-B", "-c",
                    diag.INNER_BOOTSTRAP,
                    str(archived_runner),
                    bindings["runner"]["observed_snapshot"]["sha256"],
                    diag.INNER_BOOTSTRAP_SHA256,
                    "--internal-authoritative",
                    "--invocation", str(invocation_path),
                    "--invocation-sha256", diag.sha256_file(invocation_path),
                    "--capability", capability,
                ],
                transport_log,
                outer_timeout=30,
            )
            self.assertNotEqual(run["returncode"], 0)
            self.assertIn("invalid snapshotted build receipt", run["stderr"])
            self.assertTrue((output / "FAILED.json").is_file())
            self.assertFalse((output / "COMPLETE.json").exists())

    def test_process_executable_identity_rejects_replace_run_restore_forgery(self) -> None:
        actual = diag.stable_open_identity(
            pathlib.Path("/proc/self/exe"), process_executable=True
        )
        self.assertEqual(diag.verify_control_python_identity(actual), actual)
        forged = dict(actual, inode=actual["inode"] + 1)
        with self.assertRaisesRegex(diag.DiagnosticError, "process executable"):
            diag.verify_control_python_identity(forged)

    def test_direct_internal_and_missing_normal_args_fail_closed_without_type_errors(self) -> None:
        with self.assertRaisesRegex(diag.DiagnosticError, "lacks authority"):
            diag.main(["--internal-authoritative"])
        with self.assertRaisesRegex(diag.DiagnosticError, "normal invocation requires"):
            diag.main([])

    def test_invalid_top_level_receipt_id_is_sealed_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "result"
            with self.assertRaisesRegex(diag.DiagnosticError, "lowercase SHA"):
                diag.main(
                    [
                        "--build-receipt", str(pathlib.Path(temporary) / "missing.json"),
                        "--expected-receipt-id", "invalid",
                        "--output", str(output),
                        "--mpiexec", str(pathlib.Path(temporary) / "mpiexec"),
                    ]
                )
            self.assertTrue((output / "FAILED.json").is_file())
            self.assertFalse((output / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
