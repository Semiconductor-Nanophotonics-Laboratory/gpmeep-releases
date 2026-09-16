from __future__ import annotations

import copy
import json
import math
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import multi_gpu_benchmark as mgb  # noqa: E402


def valid_phase_calls() -> dict:
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
    return phases


def valid_record(ranks: int = 1, transport: str = "pinned") -> dict:
    pixels = 96
    steps = 20
    seconds = 2.0 if ranks == 1 else 1.2
    cells = pixels**3
    probes = [[0.10 + 0.01 * index, 0.0] for index in range(mgb.PROBE_COUNT)]
    probe_l2 = math.sqrt(
        sum(real * real + imag * imag for real, imag in probes)
    )
    multi = ranks > 1
    selected = transport if multi else "none"
    phase_calls = valid_phase_calls()
    return {
        "schema_version": mgb.SCHEMA_VERSION,
        "mpi_ranks": ranks,
        "pixels": pixels,
        "cells": cells,
        "warmup_steps": 6,
        "steps": steps,
        "loop_tile_base_db": 128,
        "bfast": False,
        "source_enabled": True,
        "overlap_material": False,
        "seconds": seconds,
        "mcells_per_second": cells * steps / seconds / 1e6,
        "requested_transport": transport,
        "selected_transport": selected,
        "completion_policy": "waitsome",
        "initial_condition": "trigonometric-v1",
        "cut_probe_values": probes,
        "cut_probe_l2": probe_l2,
        "ez_l2": 1.2,
        "ez_weighted_checksum": 1.7,
        "energy": 2.1,
        "dft_norm": 0.35,
        "remote_cut_scalars": 120 if multi else 0,
        "remote_cut_l2": 0.42 if multi else 0.0,
        "cpu_calls": sum(item["cpu_calls"] for item in phase_calls.values()),
        "cuda_calls": sum(item["cuda_calls"] for item in phase_calls.values()),
        "phase_calls": phase_calls,
        "h2d_bytes": 32,
        "d2h_bytes": 16,
        "mpi_messages": 24 if multi else 0,
        "mpi_scalars": 240 if multi else 0,
        "cuda_aware_bytes": 960 if multi and transport == "cuda-aware" else 0,
        "pinned_bytes": 960 if multi and transport == "pinned" else 0,
    }


def encoded(record: dict) -> str:
    return mgb.RECORD_PREFIX + json.dumps(record, separators=(",", ":")) + "\n"


def encoded_initialization(
    profile: str = "trigonometric-v1",
    source_profile: str = "single-ez-v1",
    applications: int = 1,
    seconds: float = 0.25,
) -> str:
    return "".join(
        (
            mgb.SOURCE_PROFILE_PREFIX
            + json.dumps({"profile": source_profile}, separators=(",", ":"))
            + "\n",
            mgb.INITIAL_CONDITION_PREFIX
            + json.dumps({"profile": profile}, separators=(",", ":"))
            + "\n",
            mgb.INITIALIZATION_TIMING_PREFIX
            + json.dumps(
                {
                    "applications_per_rank": applications,
                    "max_seconds": seconds,
                },
                separators=(",", ":"),
            )
            + "\n",
        )
    )


def valid_sample(repeat_index: int, ranks: int, seconds: float) -> dict:
    record = valid_record(ranks)
    record["seconds"] = seconds
    record["mcells_per_second"] = (
        record["cells"] * record["steps"] / seconds / 1e6
    )
    return {
        "repeat_index": repeat_index,
        "mpi_ranks": ranks,
        "record": record,
        "devices": [
            {"uuid": f"{ordinal + 1:032x}"} for ordinal in range(ranks)
        ],
    }


class StrictRecordTests(unittest.TestCase):
    def test_valid_one_rank_record(self) -> None:
        record = mgb.extract_record(
            "Meep progress\n" + encoded(valid_record()),
            expected_ranks=1,
            expected_pixels=96,
            expected_warmup_steps=6,
            expected_steps=20,
            expected_transport="pinned",
        )
        self.assertEqual(record["mpi_ranks"], 1)

    def test_initialization_markers_are_exact_and_separate_from_fdtd(self) -> None:
        records = mgb.extract_initialization_records(
            encoded_initialization(), expected_profile="trigonometric-v1"
        )
        self.assertEqual(
            records["initial_condition"], {"profile": "trigonometric-v1"}
        )
        self.assertEqual(
            records["initialize_field_timing"]["max_seconds"], 0.25
        )
        self.assertIn(
            "excluded from FDTD seconds",
            records["initialize_field_timing"]["timing_domain"],
        )
        self.assertEqual(
            mgb.extract_source_profile_record(
                encoded_initialization(), expected_profile="single-ez-v1"
            ),
            {"profile": "single-ez-v1"},
        )
        for source_profile in ("dual-electric-v1", "unknown-v1"):
            with self.subTest(source_profile=source_profile):
                with self.assertRaisesRegex(mgb.EvidenceError, "differs"):
                    mgb.extract_source_profile_record(
                        encoded_initialization(source_profile=source_profile),
                        expected_profile="single-ez-v1",
                    )
        source_line = (
            mgb.SOURCE_PROFILE_PREFIX + '{"profile":"single-ez-v1"}\n'
        )
        for output in ("", source_line * 2):
            with self.subTest(source_marker_count=output.count(source_line)):
                with self.assertRaisesRegex(mgb.EvidenceError, "exactly one"):
                    mgb.extract_source_profile_record(
                        output, expected_profile="single-ez-v1"
                    )
        for output, message in (
            (encoded_initialization(profile="affine-v1"), "differs"),
            (encoded_initialization(applications=2), "application count"),
            (encoded_initialization(seconds=0.0), "finite and positive"),
            (
                encoded_initialization() + encoded_initialization(),
                "exactly one",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(mgb.EvidenceError, message):
                    mgb.extract_initialization_records(
                        output, expected_profile="trigonometric-v1"
                    )
        for invalid_expected in (0, -1, True):
            with self.subTest(invalid_expected=invalid_expected):
                with self.assertRaisesRegex(
                    mgb.EvidenceError, "positive integer"
                ):
                    mgb.extract_initialization_records(
                        encoded_initialization(applications=invalid_expected),
                        expected_profile="trigonometric-v1",
                        expected_applications_per_rank=invalid_expected,
                    )

    def test_reserved_marker_streams_fail_closed(self) -> None:
        stdout = encoded(valid_record()) + encoded_initialization()
        mgb.reject_unknown_worker_markers(stdout, "ordinary diagnostic\n")
        mgb.reject_unknown_worker_markers(
            "".join(
                prefix + "{}\n"
                for prefix in mgb.WORKER_STDOUT_GPMEEP_PREFIXES
            ),
            "",
        )
        for candidate, stderr, message in (
            (
                stdout
                + 'gpmeep-initial-condition-v2:{"profile":"trigonometric-v1"}\n',
                "",
                "mixed-version",
            ),
            (
                stdout + 'gpmeep-multi-benchmark-v3:{"schema_version":3}\n',
                "",
                "mixed-version",
            ),
            (
                stdout,
                'gpmeep-initial-condition-v1:{"profile":"trigonometric-v1"}\n',
                "stderr",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(mgb.EvidenceError, message):
                    mgb.reject_unknown_worker_markers(candidate, stderr)

    def test_valid_multi_rank_pinned_record(self) -> None:
        record = mgb.extract_record(
            encoded(valid_record(2)),
            expected_ranks=2,
            expected_transport="host",
        )
        self.assertGreater(record["remote_cut_l2"], 0)

    def test_valid_multi_rank_cuda_aware_record(self) -> None:
        record = mgb.extract_record(
            encoded(valid_record(2, "cuda-aware")),
            expected_ranks=2,
            expected_transport="device",
        )
        self.assertEqual(record["selected_transport"], "cuda-aware")

    def test_duplicate_record_is_rejected(self) -> None:
        text = encoded(valid_record()) * 2
        with self.assertRaisesRegex(mgb.EvidenceError, "exactly one"):
            mgb.extract_record(text)

    def test_missing_record_is_rejected(self) -> None:
        with self.assertRaisesRegex(mgb.EvidenceError, "found 0"):
            mgb.extract_record("1,96,884736,20\n")

    def test_duplicate_json_key_is_rejected(self) -> None:
        payload = '{"schema_version":4,"schema_version":4}'
        with self.assertRaisesRegex(mgb.EvidenceError, "duplicate JSON key"):
            mgb.extract_record(mgb.RECORD_PREFIX + payload)

    def test_legacy_v3_prefix_and_schema_fail_closed(self) -> None:
        legacy = valid_record()
        legacy["schema_version"] = 3
        payload = json.dumps(legacy, separators=(",", ":"))
        with self.assertRaisesRegex(mgb.EvidenceError, "found 0"):
            mgb.extract_record("gpmeep-multi-benchmark-v3:" + payload)
        with self.assertRaisesRegex(mgb.EvidenceError, "schema_version must be 4"):
            mgb.extract_record(mgb.RECORD_PREFIX + payload)

    def test_unknown_key_is_rejected(self) -> None:
        record = valid_record()
        record["untrusted"] = 1
        with self.assertRaisesRegex(mgb.EvidenceError, "key mismatch"):
            mgb.extract_record(encoded(record))

    def test_missing_key_is_rejected(self) -> None:
        record = valid_record()
        del record["dft_norm"]
        with self.assertRaisesRegex(mgb.EvidenceError, "key mismatch"):
            mgb.extract_record(encoded(record))

    def test_nonfinite_value_is_rejected(self) -> None:
        record = valid_record()
        record["seconds"] = float("nan")
        with self.assertRaisesRegex(mgb.EvidenceError, "non-finite JSON"):
            mgb.extract_record(encoded(record))

    def test_boolean_integer_is_rejected(self) -> None:
        record = valid_record()
        record["steps"] = True
        with self.assertRaisesRegex(mgb.EvidenceError, "must be an integer"):
            mgb.extract_record(encoded(record))

    def test_fixed_worker_profile_is_exact_and_type_checked(self) -> None:
        mutations = {
            "loop_tile_base_db": 64,
            "bfast": True,
            "source_enabled": False,
            "overlap_material": True,
            "completion_policy": "waitall",
            "initial_condition": "affine-v1",
        }
        for key, value in mutations.items():
            with self.subTest(key=key, value=value):
                record = valid_record()
                record[key] = value
                with self.assertRaisesRegex(
                    mgb.EvidenceError, f"worker profile requires {key}"
                ):
                    mgb.extract_record(encoded(record))
        wrong_types = {
            "loop_tile_base_db": True,
            "bfast": 0,
            "source_enabled": 1,
            "overlap_material": 0,
            "completion_policy": False,
            "initial_condition": 1,
        }
        for key, value in wrong_types.items():
            with self.subTest(key=key, wrong_type=value):
                record = valid_record()
                record[key] = value
                with self.assertRaises(mgb.EvidenceError):
                    mgb.extract_record(encoded(record))

    def test_zero_global_observable_is_rejected(self) -> None:
        for key in (
            "cut_probe_l2",
            "ez_l2",
            "ez_weighted_checksum",
            "energy",
            "dft_norm",
        ):
            with self.subTest(key=key):
                record = valid_record()
                record[key] = 0.0
                with self.assertRaises(mgb.EvidenceError):
                    mgb.extract_record(encoded(record))

    def test_zero_probe_vector_is_rejected(self) -> None:
        record = valid_record()
        record["cut_probe_values"] = [[0.0, 0.0]] * mgb.PROBE_COUNT
        record["cut_probe_l2"] = 1.0
        with self.assertRaisesRegex(mgb.EvidenceError, "identically zero"):
            mgb.extract_record(encoded(record))

    def test_inconsistent_probe_norm_is_rejected(self) -> None:
        record = valid_record()
        record["cut_probe_l2"] *= 2
        with self.assertRaisesRegex(mgb.EvidenceError, "does not match"):
            mgb.extract_record(encoded(record))

    def test_mismatched_fixed_work_is_rejected(self) -> None:
        record = valid_record()
        record["cells"] += 1
        with self.assertRaisesRegex(mgb.EvidenceError, "pixels cubed"):
            mgb.extract_record(encoded(record))

    def test_mismatched_expected_work_is_rejected(self) -> None:
        with self.assertRaisesRegex(mgb.EvidenceError, "expected 192"):
            mgb.extract_record(encoded(valid_record()), expected_pixels=192)

    def test_inconsistent_throughput_is_rejected(self) -> None:
        record = valid_record()
        record["mcells_per_second"] *= 1.1
        with self.assertRaisesRegex(mgb.EvidenceError, "inconsistent"):
            mgb.extract_record(encoded(record))

    def test_cpu_dispatch_is_rejected(self) -> None:
        record = valid_record()
        record["phase_calls"]["curl"]["cpu_calls"] = 1
        record["cpu_calls"] = 1
        with self.assertRaisesRegex(mgb.EvidenceError, "required phase"):
            mgb.extract_record(encoded(record))

    def test_every_required_phase_rejects_cpu_or_zero_cuda(self) -> None:
        for phase in mgb.REQUIRED_CUDA_PHASES:
            for field, value in (("cpu_calls", 1), ("cuda_calls", 0)):
                with self.subTest(phase=phase, field=field):
                    record = valid_record()
                    record["phase_calls"][phase][field] = value
                    with self.assertRaisesRegex(mgb.EvidenceError, "required phase"):
                        mgb.extract_record(encoded(record))

    def test_phase_schema_and_expectations_are_exact(self) -> None:
        mutations = []
        missing = valid_record()
        del missing["phase_calls"]["dft"]
        mutations.append((missing, "phase_calls key mismatch"))
        extra = valid_record()
        extra["phase_calls"]["unknown"] = extra["phase_calls"]["curl"]
        mutations.append((extra, "phase_calls key mismatch"))
        wrong_expectation = valid_record()
        wrong_expectation["phase_calls"]["dft"]["expectation"] = "optional"
        mutations.append((wrong_expectation, "expectation"))
        extra_nested = valid_record()
        extra_nested["phase_calls"]["dft"]["extra"] = 1
        mutations.append((extra_nested, "invalid schema"))
        for record, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(mgb.EvidenceError, message):
                    mgb.extract_record(encoded(record))

    def test_not_applicable_polarization_must_remain_zero(self) -> None:
        for field in ("cpu_calls", "cuda_calls"):
            with self.subTest(field=field):
                record = valid_record()
                record["phase_calls"]["polarization"][field] = 1
                with self.assertRaisesRegex(mgb.EvidenceError, "not-applicable"):
                    mgb.extract_record(encoded(record))

    def test_top_level_phase_aggregates_are_exact(self) -> None:
        for field in ("cpu_calls", "cuda_calls"):
            for delta in (-1, 1):
                with self.subTest(field=field, delta=delta):
                    record = valid_record()
                    record[field] += delta
                    with self.assertRaisesRegex(
                        mgb.EvidenceError, "aggregate|nonnegative"
                    ):
                        mgb.extract_record(encoded(record))

    def test_multi_rank_zero_remote_cut_is_rejected(self) -> None:
        record = valid_record(2)
        record["remote_cut_l2"] = 0.0
        with self.assertRaisesRegex(mgb.EvidenceError, "vacuous remote-cut"):
            mgb.extract_record(encoded(record))

    def test_multi_rank_mixed_transport_is_rejected(self) -> None:
        record = valid_record(2)
        record["cuda_aware_bytes"] = 960
        with self.assertRaisesRegex(mgb.EvidenceError, "exactly one MPI transport"):
            mgb.extract_record(encoded(record))

    def test_transport_counter_mismatch_is_rejected(self) -> None:
        record = valid_record(2)
        record["selected_transport"] = "cuda-aware"
        with self.assertRaisesRegex(mgb.EvidenceError, "byte counters"):
            mgb.extract_record(encoded(record))

    def test_transport_bytes_must_equal_fp32_scalar_count(self) -> None:
        record = valid_record(2)
        record["pinned_bytes"] -= 4
        with self.assertRaisesRegex(mgb.EvidenceError, "FP32 size"):
            mgb.extract_record(encoded(record))

    def test_pinned_transport_requires_bidirectional_staging(self) -> None:
        for key in ("h2d_bytes", "d2h_bytes"):
            with self.subTest(key=key):
                record = valid_record(2)
                record[key] = 0
                with self.assertRaisesRegex(mgb.EvidenceError, "H2D and D2H"):
                    mgb.extract_record(encoded(record))

    def test_rank_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(mgb.EvidenceError, "expected 4"):
            mgb.extract_record(encoded(valid_record(2)), expected_ranks=4)


class ConfigurationTests(unittest.TestCase):
    def test_rank_list_is_exact_and_requires_baseline(self) -> None:
        self.assertEqual(mgb.parse_rank_list("1,2,4"), [1, 2, 4])
        for invalid in ("", "1,,2", "0,1", "1,2,2", "2,4", "1,2x"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(mgb.EvidenceError):
                    mgb.parse_rank_list(invalid)

    def test_probability_rejects_nonfinite_and_out_of_range(self) -> None:
        self.assertEqual(mgb.parse_probability("gate", "0.55"), 0.55)
        for invalid in ("nan", "inf", "-0.1", "1.1", "text"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(mgb.EvidenceError):
                    mgb.parse_probability("gate", invalid)

    def test_boolean_flags_are_strict(self) -> None:
        self.assertFalse(mgb.parse_boolean_flag("flag", "0"))
        self.assertTrue(mgb.parse_boolean_flag("flag", "1"))
        for invalid in ("", "true", "false", "2", " 1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(mgb.EvidenceError):
                    mgb.parse_boolean_flag("flag", invalid)

    def test_comparison_detects_probe_change(self) -> None:
        reference = valid_record()
        candidate = copy.deepcopy(reference)
        candidate["cut_probe_values"][3][0] *= 1.1
        candidate["cut_probe_l2"] = math.sqrt(
            sum(
                real * real + imag * imag
                for real, imag in candidate["cut_probe_values"]
            )
        )
        comparison = mgb.compare_record(candidate, reference, 0.002)
        self.assertFalse(comparison["observable_gate"])
        self.assertGreater(comparison["cut_probe_max_relative_error"], 0.002)

    def test_repeated_timings_use_median_and_every_sample_correctness(self) -> None:
        samples = []
        for repeat_index, one_second, two_seconds in (
            (1, 10.0, 6.0),
            (2, 11.0, 7.0),
            (3, 100.0, 80.0),
        ):
            samples.append(valid_sample(repeat_index, 1, one_second))
            samples.append(valid_sample(repeat_index, 2, two_seconds))
        summaries, passed = mgb.summarize_samples(
            samples, [1, 2], 3, 0.002, 0.5
        )
        self.assertTrue(passed)
        one, two = summaries
        self.assertEqual(one["seconds"], 11.0)
        self.assertEqual(two["seconds"], 7.0)
        self.assertEqual(two["seconds_statistic"], "median")
        self.assertAlmostEqual(two["speedup_vs_1gpu"], 11.0 / 7.0)
        self.assertTrue(two["all_sample_observable_gate"])
        self.assertTrue(all("comparison_vs_reference" in sample for sample in samples))

        samples[-1]["record"]["energy"] *= 1.1
        _, passed = mgb.summarize_samples(samples, [1, 2], 3, 0.002, 0.5)
        self.assertFalse(passed)

    def test_repeat_matrix_rejects_missing_duplicate_or_changed_gpu_mapping(self) -> None:
        base = [
            valid_sample(repeat_index, ranks, 2.0 if ranks == 1 else 1.2)
            for repeat_index in (1, 2, 3)
            for ranks in (1, 2)
        ]
        missing = copy.deepcopy(base[:-1])
        duplicate = copy.deepcopy(base)
        duplicate[-1]["repeat_index"] = 2
        changed_mapping = copy.deepcopy(base)
        changed_mapping[-1]["devices"][1]["uuid"] = "f" * 32
        changed_rank_zero = copy.deepcopy(base)
        for sample in changed_rank_zero:
            if sample["mpi_ranks"] == 2:
                sample["devices"][0]["uuid"] = "e" * 32
        for samples, message in (
            (missing, "incomplete/duplicate"),
            (duplicate, "incomplete/duplicate"),
            (changed_mapping, "mapping changed"),
            (changed_rank_zero, "canonical prefix"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(mgb.EvidenceError, message):
                    mgb.validate_sample_matrix(samples, [1, 2], 3)

        cross_configuration = [
            valid_sample(repeat_index, ranks, 2.0)
            for repeat_index in (1, 2, 3)
            for ranks in (1, 2, 4)
        ]
        for sample in cross_configuration:
            if sample["mpi_ranks"] == 4:
                sample["devices"][1]["uuid"] = "e" * 32
        with self.assertRaisesRegex(mgb.EvidenceError, "canonical prefix"):
            mgb.validate_sample_matrix(cross_configuration, [1, 2, 4], 3)

    def test_absent_build_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(mgb.EvidenceError, "required MPI build"):
                mgb.load_build_receipt(
                    pathlib.Path(directory), pathlib.Path(directory) / "build"
                )

    def test_python_mpi_receipt_and_artifacts_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            build = repo / "build"
            build.mkdir()
            receipt_path = build / "build-provenance.json"
            receipt_path.write_text("{}\n", encoding="utf-8")
            artifacts = {}
            for name in ("gpu_mpi_performance", "gpu_step_db_test"):
                path = build / name
                path.write_bytes(name.encode())
                artifacts[name] = {
                    "path": str(path),
                    "sha256": mgb.sha256_file(path),
                }
            receipt = {
                "build_kind": "cuda-mpi-python-fp32",
                "receipt_id": "1" * 64,
                "build_input_id": "2" * 64,
                "artifact_set_id": "3" * 64,
                "artifacts": artifacts,
            }
            with mock.patch.object(
                mgb, "verify_build_receipt", return_value=receipt
            ):
                loaded = mgb.load_build_receipt(repo, build)
            self.assertEqual(loaded["receipt_id"], "1" * 64)
            self.assertEqual(set(loaded["verified_artifacts"]), set(artifacts))

            receipt["build_kind"] = "cuda-mpi-fp32"
            with mock.patch.object(
                mgb, "verify_build_receipt", return_value=receipt
            ):
                with self.assertRaisesRegex(mgb.EvidenceError, "wrong build_kind"):
                    mgb.load_build_receipt(repo, build)

    def test_device_identity_records_require_unique_physical_gpus(self) -> None:
        def device(rank: int, uuid: str) -> str:
            return mgb.DEVICE_RECORD_PREFIX + json.dumps(
                {
                    "rank": rank,
                    "ordinal": rank,
                    "uuid": uuid,
                    "name": "Synthetic GPU",
                    "compute_major": 8,
                    "compute_minor": 6,
                    "compatible": True,
                },
                separators=(",", ":"),
            )

        records = mgb.extract_device_records(
            device(1, "2" * 32) + "\n" + device(0, "1" * 32) + "\n", 2
        )
        self.assertEqual([item["rank"] for item in records], [0, 1])
        with self.assertRaisesRegex(mgb.EvidenceError, "distinct physical"):
            mgb.extract_device_records(
                device(0, "1" * 32) + "\n" + device(1, "1" * 32) + "\n", 2
            )

        with self.assertRaisesRegex(mgb.EvidenceError, "32 lowercase hex"):
            mgb.extract_device_records(device(0, "GPU-a") + "\n", 1)

    def test_release_gate_requires_exact_physical_1_2_4_scope(self) -> None:
        partial = mgb.qualification_gates(True, [1, 2], [1, 2], [], False)
        self.assertTrue(partial["development"]["pass"])
        self.assertFalse(partial["release"]["pass"])
        self.assertEqual(partial["qualification"], "development-only")

        workload = {
            "quick": False,
            "pixels": 192,
            "warmup_steps": 12,
            "measured_steps": 80,
            **mgb.FIXED_WORKER_PROFILE,
            "source_profile": "single-ez-v1",
        }
        missing_profile = mgb.qualification_gates(
            True, [1, 2, 4], [1, 2, 4], [], True
        )
        self.assertFalse(missing_profile["release"]["pass"])

        release = mgb.qualification_gates(
            True,
            [1, 2, 4],
            [1, 2, 4],
            [],
            True,
            workload=workload,
            repeat_count=3,
            timing_statistic="median",
            transport="pinned",
            minimum_efficiency=0.55,
            observable_tolerance=0.002,
        )
        self.assertTrue(release["release"]["pass"])
        self.assertEqual(release["qualification"], "release")

        skipped = mgb.qualification_gates(
            True,
            [1, 2, 4],
            [1, 2],
            [{"ranks": 4, "reason": "not visible"}],
            True,
        )
        self.assertFalse(skipped["release"]["pass"])

    def test_every_release_profile_dimension_is_fail_closed(self) -> None:
        workload = {
            "quick": False,
            "pixels": 192,
            "warmup_steps": 12,
            "measured_steps": 80,
            **mgb.FIXED_WORKER_PROFILE,
            "source_profile": "single-ez-v1",
        }
        mutations = (
            ({**workload, "quick": True}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "pixels": 96}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "warmup_steps": 11}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "measured_steps": 79}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "loop_tile_base_db": 64}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "bfast": True}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "source_enabled": False}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "source_profile": "dual-electric-v1"}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "overlap_material": True}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "completion_policy": "waitall"}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "quick": 0}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "loop_tile_base_db": True}, 3, "median", "pinned", 0.55, 0.002),
            ({**workload, "bfast": 0}, 3, "median", "pinned", 0.55, 0.002),
            (workload, 2, "median", "pinned", 0.55, 0.002),
            (workload, 3, "single", "pinned", 0.55, 0.002),
            (workload, 3, "median", "cuda-aware", 0.55, 0.002),
            (workload, 3, "median", "pinned", 0.54, 0.002),
            (workload, 3, "median", "pinned", 0.55, 0.003),
        )
        for candidate, repeats, statistic, transport, efficiency, tolerance in mutations:
            with self.subTest(
                candidate=candidate,
                repeats=repeats,
                statistic=statistic,
                transport=transport,
                efficiency=efficiency,
                tolerance=tolerance,
            ):
                gate = mgb.qualification_gates(
                    True,
                    [1, 2, 4],
                    [1, 2, 4],
                    [],
                    True,
                    workload=candidate,
                    repeat_count=repeats,
                    timing_statistic=statistic,
                    transport=transport,
                    minimum_efficiency=efficiency,
                    observable_tolerance=tolerance,
                )
                self.assertFalse(gate["release"]["pass"])
                self.assertFalse(gate["release"]["profile_pass"])
                self.assertTrue(gate["release"]["profile_failures"])

    def test_complete_marker_binds_atomic_report_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            report_path = root / "report.json"
            complete_path = root / "COMPLETE.json"
            report = {
                "schema_version": mgb.SCHEMA_VERSION,
                "state": "COMPLETE",
                "run_id": "synthetic-run",
            }
            marker = mgb.publish_complete(
                root,
                report_path,
                complete_path,
                report,
                controller_pass=True,
                release_pass=False,
            )
            self.assertEqual(marker["report"]["sha256"], mgb.sha256_file(report_path))
            self.assertEqual(
                marker,
                json.loads(complete_path.read_text(encoding="utf-8")),
            )
            self.assertFalse(marker["release_pass"])

    def test_individual_log_record_detects_post_run_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            log = root / "sample.log"
            complete = root / "COMPLETE.json"
            mgb._write_run_log(log, ["worker", "--flag"], 0, 1.25, "ok\n", "")
            original = mgb.file_record(log, root)
            self.assertEqual(original["sha256"], mgb.sha256_file(log))
            self.assertEqual(mgb.verify_file_record(original, root, "log"), log)
            log.write_text(log.read_text(encoding="utf-8") + "tampered\n")
            self.assertNotEqual(original["sha256"], mgb.sha256_file(log))
            with self.assertRaisesRegex(mgb.EvidenceError, "size differs|SHA-256"):
                mgb.verify_file_record(original, root, "log")
            self.assertFalse(complete.exists())

    def test_device_query_requires_one_exact_record(self) -> None:
        self.assertEqual(mgb._device_count("compatible_cuda_devices=2\n"), 2)
        for invalid in (
            "compatible_cuda_devices=2x\n",
            " compatible_cuda_devices=2\n",
            "compatible_cuda_devices=2\ncompatible_cuda_devices=2\n",
            "",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(mgb.EvidenceError):
                    mgb._device_count(invalid)

    def test_cuda_aware_transport_flags_apply_to_every_mpi_lane(self) -> None:
        base = {"HOME": "/safe/home", "OMP_NUM_THREADS": "1"}
        pinned = mgb._mpi_transport_environment(base, "pinned")
        self.assertEqual(pinned["MEEP_GPU_MPI_TRANSPORT"], "pinned")
        self.assertNotIn("OMPI_MCA_opal_cuda_support", pinned)
        with mock.patch.dict(
            mgb.os.environ,
            {"MEEP_GPU_MPI_MCA_BTL": "self,smcuda"},
        ):
            aware = mgb._mpi_transport_environment(base, "cuda-aware")
        self.assertEqual(aware["MEEP_GPU_MPI_TRANSPORT"], "cuda-aware")
        self.assertEqual(aware["OMPI_MCA_opal_cuda_support"], "true")
        self.assertEqual(aware["OMPI_MCA_btl"], "self,smcuda")
        self.assertEqual(aware["UCX_MEMTYPE_CACHE"], "n")
        self.assertEqual(base, {"HOME": "/safe/home", "OMP_NUM_THREADS": "1"})

    def test_base_environment_pins_mpi_configuration_to_source_and_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            home = repo / "home"
            home.mkdir()
            parameter_path = (
                repo / "environment" / "openmpi-qualification-mca-params.conf"
            )
            parameter_path.parent.mkdir()
            parameter_path.write_text("mca_base_component_show_load_errors=1\n")
            prefix = repo / ".envs" / "meep-gpu-cuda-mpi"
            (prefix / "lib" / "openmpi").mkdir(parents=True)
            (prefix / "lib" / "pmix").mkdir(parents=True)
            with mock.patch.dict(
                mgb.os.environ, {"HOME": str(home)}, clear=True
            ):
                environment = mgb._forwarded_base_environment(repo)
            parameter_file = str(parameter_path)
            self.assertEqual(
                environment["OMPI_MCA_mca_base_param_files"], parameter_file
            )
            self.assertEqual(
                environment["PMIX_MCA_mca_base_param_files"], parameter_file
            )
            self.assertEqual(
                environment["PRTE_MCA_mca_base_param_files"], parameter_file
            )
            self.assertEqual(
                environment["OMPI_MCA_mca_base_component_path"],
                str(prefix / "lib" / "openmpi"),
            )
            self.assertEqual(
                environment["PMIX_MCA_mca_base_component_path"],
                str(prefix / "lib" / "pmix"),
            )

    def test_device_query_keeps_only_identity_and_canonical_mpi_environment(self) -> None:
        base = {
            "HOME": "/safe/home",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMP_NUM_THREADS": "1",
            "MEEP_GPU_MPI_TRANSPORT": "cuda-aware",
            "MEEP_GPU_MULTI_INITIAL_CONDITION": "trigonometric-v1",
            "MEEP_GPU_MULTI_SOURCE_PROFILE": "single-ez-v1",
            "OMPI_MCA_opal_cuda_support": "true",
            **{
                key: f"canonical/{key}"
                for key in mgb.CANONICAL_MPI_ENVIRONMENT_KEYS
            },
        }
        query = mgb._device_query_environment(base)
        self.assertEqual(
            set(query),
            {
                "HOME",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                *mgb.CANONICAL_MPI_ENVIRONMENT_KEYS,
            },
        )
        self.assertNotIn("MEEP_GPU_MPI_TRANSPORT", query)
        self.assertNotIn("MEEP_GPU_MULTI_INITIAL_CONDITION", query)
        self.assertNotIn("MEEP_GPU_MULTI_SOURCE_PROFILE", query)
        self.assertNotIn("OMPI_MCA_opal_cuda_support", query)
        with self.assertRaisesRegex(
            mgb.EvidenceError, "missing canonical MPI configuration"
        ):
            mgb._device_query_environment({"HOME": "/safe/home"})

    def test_every_mpi_command_is_wrapped_by_timeout(self) -> None:
        command = mgb._timed_mpi_command(
            ["mamba", "run"], 37, 4, pathlib.Path("/tmp/worker")
        )
        timeout_index = command.index("/usr/bin/timeout")
        mpirun_index = command.index("mpirun")
        self.assertLess(timeout_index, mpirun_index)
        self.assertEqual(command[timeout_index + 1 : timeout_index + 4], [
            "--signal=TERM",
            "--kill-after=5s",
            "37s",
        ])
        self.assertEqual(command[mpirun_index + 4], "4")

    def test_run_logged_reaps_pipe_holding_descendant_after_timeout(self) -> None:
        child_cases = {
            "same_group_no_pipes": (
                "import signal,time;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)",
                "subprocess.DEVNULL",
            ),
            "setsid_pipe_holder": (
                "import os,signal,time;os.setsid();"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)",
                "None",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, (child_program, child_stdout) in child_cases.items():
                with self.subTest(name=name):
                    program = (
                        "import subprocess,sys,time;"
                        "child=subprocess.Popen([sys.executable,'-c',"
                        + repr(child_program)
                        + f"],stdout={child_stdout},stderr={child_stdout});"
                        "print('descendant_pid='+str(child.pid),flush=True);"
                        "time.sleep(60)"
                    )
                    log = pathlib.Path(directory) / f"{name}.log"
                    with mock.patch.object(
                        mgb, "TERMINATION_GRACE_SECONDS", 0.25
                    ):
                        with self.assertRaises(mgb.EvidenceError) as caught:
                            mgb.run_logged(
                                [sys.executable, "-u", "-c", program],
                                log,
                                outer_timeout=0.2,
                            )
                    self.assertIsInstance(
                        caught.exception.__cause__, subprocess.TimeoutExpired
                    )
                    text = log.read_text(encoding="utf-8")
                    self.assertIn("returncode: PYTHON_TIMEOUT", text)
                    match = re.search(r"descendant_pid=([0-9]+)", text)
                    self.assertIsNotNone(match)
                    pid = int(match.group(1))
                    deadline = time.monotonic() + 1.0
                    while pathlib.Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertFalse(
                        pathlib.Path(f"/proc/{pid}").exists(),
                        f"descendant {pid} survived containment",
                    )

    def test_run_logged_reaps_daemon_after_successful_launcher_exit(self) -> None:
        child_program = (
            "import signal,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
        )
        program = (
            "import subprocess,sys;"
            "child=subprocess.Popen([sys.executable,'-c',"
            + repr(child_program)
            + "],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
            "print('descendant_pid='+str(child.pid),flush=True)"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "descendant-leak.log"
            with mock.patch.object(mgb, "TERMINATION_GRACE_SECONDS", 0.25):
                with self.assertRaisesRegex(
                    mgb.EvidenceError, "left surviving descendants"
                ):
                    mgb.run_logged(
                        [sys.executable, "-u", "-c", program], log, 2
                    )
            text = log.read_text(encoding="utf-8")
            self.assertIn("returncode: DESCENDANT_LEAK", text)
            match = re.search(r"descendant_pid=([0-9]+)", text)
            self.assertIsNotNone(match)
            pid = int(match.group(1))
            self.assertFalse(pathlib.Path(f"/proc/{pid}").exists())

    def test_subreaper_state_is_restored_if_baseline_capture_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            setter = mock.Mock()
            with (
                mock.patch.object(mgb, "_child_subreaper_state", return_value=False),
                mock.patch.object(mgb, "_set_child_subreaper", setter),
                mock.patch.object(
                    mgb, "_process_descendants", side_effect=OSError("EMFILE")
                ),
                self.assertRaisesRegex(OSError, "EMFILE"),
            ):
                mgb.run_logged(
                    [sys.executable, "-c", "pass"],
                    pathlib.Path(directory) / "unused.log",
                    1,
                )
            self.assertEqual(
                setter.call_args_list,
                [mock.call(True), mock.call(False)],
            )

    def test_missing_root_procfs_tree_fails_closed(self) -> None:
        with mock.patch.object(
            mgb.pathlib.Path,
            "iterdir",
            side_effect=FileNotFoundError("procfs unavailable"),
        ):
            with self.assertRaisesRegex(
                mgb.EvidenceError, "required process task tree is unavailable"
            ):
                mgb._process_descendants(mgb.os.getpid())

    def test_partial_pidfd_snapshot_is_closed_or_returned_for_cleanup(self) -> None:
        failure = OSError(1, "EPERM")
        with (
            mock.patch.object(
                mgb, "_process_descendants", return_value={111, 222}
            ),
            mock.patch.object(mgb, "_pidfd_has_exited", return_value=False),
            mock.patch.object(mgb.os, "close") as close,
        ):
            with mock.patch.object(
                mgb, "_pidfd_open", side_effect=[51, failure]
            ):
                with self.assertRaises(mgb.EvidenceError):
                    mgb._snapshot_run_pidfds(set())
            close.assert_called_once_with(51)

            close.reset_mock()
            errors: list[str] = []
            with mock.patch.object(
                mgb, "_pidfd_open", side_effect=[52, failure]
            ):
                handles = mgb._snapshot_run_pidfds(set(), errors)
            self.assertEqual(handles, {111: 52})
            self.assertTrue(errors)
            close.assert_not_called()

    def test_pidfd_signal_capability_fails_before_process_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            setter = mock.Mock()
            popen = mock.Mock()
            with (
                mock.patch.object(mgb, "_child_subreaper_state", return_value=False),
                mock.patch.object(mgb, "_set_child_subreaper", setter),
                mock.patch.object(mgb, "_process_descendants", return_value=set()),
                mock.patch.object(mgb, "_pidfd_open", return_value=91),
                mock.patch.object(
                    mgb,
                    "_pidfd_send_signal",
                    side_effect=OSError(1, "EPERM"),
                ),
                mock.patch.object(mgb.os, "close"),
                mock.patch.object(mgb.subprocess, "Popen", popen),
                self.assertRaises(OSError),
            ):
                mgb.run_logged(
                    [sys.executable, "-c", "pass"],
                    pathlib.Path(directory) / "unused.log",
                    1,
                )
            popen.assert_not_called()
            self.assertEqual(
                setter.call_args_list,
                [mock.call(True), mock.call(False)],
            )

    def test_run_logged_rejects_concurrent_controller_thread(self) -> None:
        release = threading.Event()
        ready = threading.Event()

        def background() -> None:
            ready.set()
            release.wait(2)

        thread = threading.Thread(target=background)
        thread.start()
        try:
            self.assertTrue(ready.wait(1))
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(
                    mgb.EvidenceError, "single-threaded controller"
                ):
                    mgb.run_logged(
                        [sys.executable, "-c", "pass"],
                        pathlib.Path(directory) / "unused.log",
                        1,
                    )
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
