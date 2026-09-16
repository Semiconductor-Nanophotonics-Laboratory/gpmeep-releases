from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_boundary_eh_overlap_comparator",
    SCRIPTS / "compare-boundary-eh-overlap-observables.py",
)
assert SPEC is not None and SPEC.loader is not None
COMPARATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARATOR)


def result_record() -> dict:
    return {
        "schema_version": 4,
        "mpi_ranks": 2,
        "pixels": 16,
        "cells": 16**3,
        "warmup_steps": 4,
        "steps": 10,
        "source_enabled": False,
        "overlap_material": True,
        "requested_transport": "cuda-aware",
        "selected_transport": "cuda-aware",
        "completion_policy": "waitsome",
        "initial_condition": "trigonometric-v1",
        "cut_probe_values": [[0.25, 0.0], [0.5, 0.0]],
        "cut_probe_l2": 0.75,
        "ez_l2": 1.25,
        "ez_weighted_checksum": 2.5,
        "energy": 3.5,
        "dft_norm": 4.5,
        "remote_cut_scalars": 24,
        "remote_cut_l2": 5.5,
        "cpu_calls": 0,
        "cuda_calls": 100,
        "phase_calls": {"curl": {"cpu_calls": 0, "cuda_calls": 20}},
        "h2d_bytes": 128,
        "d2h_bytes": 64,
        "mpi_messages": 80,
        "mpi_scalars": 160,
        "cuda_aware_bytes": 640,
        "pinned_bytes": 0,
    }


def overlap_record(*, enabled: bool) -> dict:
    if enabled:
        return {
            "checks": 40,
            "eligible_ranks": 2,
            "eligible": 40,
            "launched_h": 20,
            "launched_e": 20,
            "skipped_disabled": 0,
            "skipped_unsupported_schedule": 0,
            "skipped_no_remote": 0,
            "skipped_cold_topology": 0,
            "rejected": 0,
        }
    return {
        "checks": 40,
        "eligible_ranks": 0,
        "eligible": 0,
        "launched_h": 0,
        "launched_e": 0,
        "skipped_disabled": 40,
        "skipped_unsupported_schedule": 0,
        "skipped_no_remote": 0,
        "skipped_cold_topology": 0,
        "rejected": 0,
    }


def write_log(path: pathlib.Path, result: dict, overlap: dict) -> None:
    path.write_text(
        COMPARATOR.RESULT_PREFIX
        + json.dumps(result, separators=(",", ":"))
        + "\n"
        + COMPARATOR.SOURCE_PROFILE_PREFIX
        + json.dumps({"profile": "single-ez-v1"}, separators=(",", ":"))
        + "\n"
        + COMPARATOR.INITIAL_CONDITION_PREFIX
        + json.dumps({"profile": "trigonometric-v1"}, separators=(",", ":"))
        + "\n"
        + COMPARATOR.INITIALIZATION_TIMING_PREFIX
        + json.dumps(
            {"applications_per_rank": 1, "max_seconds": 0.25},
            separators=(",", ":"),
        )
        + "\n"
        + COMPARATOR.OVERLAP_PREFIX
        + json.dumps(overlap, separators=(",", ":"))
        + "\n"
        + COMPARATOR.QUALIFICATION_PREFIX
        + path.name
        + ":PASS\n",
        encoding="utf-8",
    )


class BoundaryEhOverlapComparatorTests(unittest.TestCase):
    def test_exact_fixed_workload_and_counter_partition_pass(self) -> None:
        enabled = result_record()
        disabled = result_record()
        COMPARATOR.validate_fixed_workload(enabled)
        COMPARATOR.validate_fixed_workload(disabled)
        self.assertEqual(
            COMPARATOR.selected(enabled, COMPARATOR.OBSERVABLE_KEYS),
            COMPARATOR.selected(disabled, COMPARATOR.OBSERVABLE_KEYS),
        )
        COMPARATOR.validate_overlap_counters(
            overlap_record(enabled=True), enabled=True, expected_checks=40
        )
        COMPARATOR.validate_overlap_counters(
            overlap_record(enabled=False), enabled=False, expected_checks=40
        )

    def test_wrong_workload_identity_is_rejected(self) -> None:
        record = result_record()
        record["source_enabled"] = True
        with self.assertRaisesRegex(ValueError, "source-free"):
            COMPARATOR.validate_fixed_workload(record)
        record = result_record()
        record["cells"] += 1
        with self.assertRaisesRegex(ValueError, "pixels cubed"):
            COMPARATOR.validate_fixed_workload(record)
        record = result_record()
        record["initial_condition"] = "affine-v1"
        with self.assertRaisesRegex(ValueError, "trigonometric-v1"):
            COMPARATOR.validate_fixed_workload(record)

    def test_initialization_and_qualification_markers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            baseline = root / "baseline.log"
            record = result_record()
            write_log(baseline, record, overlap_record(enabled=True))
            COMPARATOR.validate_initialization_records(baseline, record)
            text = baseline.read_text(encoding="utf-8")
            initial = (
                COMPARATOR.INITIAL_CONDITION_PREFIX
                + '{"profile":"trigonometric-v1"}\n'
            )
            source_profile = (
                COMPARATOR.SOURCE_PROFILE_PREFIX
                + '{"profile":"single-ez-v1"}\n'
            )
            timing = (
                COMPARATOR.INITIALIZATION_TIMING_PREFIX
                + '{"applications_per_rank":1,"max_seconds":0.25}\n'
            )
            qualification = (
                COMPARATOR.QUALIFICATION_PREFIX + baseline.name + ":PASS\n"
            )
            cases = (
                (text.replace(source_profile, ""), "exactly one"),
                (
                    text.replace(
                        source_profile,
                        COMPARATOR.SOURCE_PROFILE_PREFIX
                        + '{"profile":"dual-electric-v1"}\n',
                    ),
                    "release profile",
                ),
                (text.replace(initial, ""), "exactly one"),
                (text.replace(initial, initial * 2), "exactly one"),
                (
                    text.replace(
                        initial,
                        COMPARATOR.INITIAL_CONDITION_PREFIX
                        + '{"profile":"affine-v1"}\n',
                    ),
                    "release profile differ",
                ),
                (
                    text.replace(
                        timing,
                        COMPARATOR.INITIALIZATION_TIMING_PREFIX
                        + '{"applications_per_rank":true,"max_seconds":0.25}\n',
                    ),
                    "one application",
                ),
                (
                    text.replace(
                        timing,
                        COMPARATOR.INITIALIZATION_TIMING_PREFIX
                        + '{"applications_per_rank":1,"max_seconds":Infinity}\n',
                    ),
                    "non-finite",
                ),
                (
                    text.replace(
                        timing,
                        COMPARATOR.INITIALIZATION_TIMING_PREFIX
                        + '{"applications_per_rank":1,"max_seconds":0.25,"extra":1}\n',
                    ),
                    "exact schema",
                ),
                (
                    text.replace(
                        qualification,
                        'gpmeep-initial-condition-v2:'
                        '{"profile":"trigonometric-v1"}\n'
                        + qualification,
                    ),
                    "mixed-version",
                ),
                (
                    text.replace(
                        qualification,
                        'gpmeep-multi-benchmark-v3:{"schema_version":3}\n'
                        + qualification,
                    ),
                    "mixed-version",
                ),
                (text.replace(qualification, ""), "exact terminal"),
                (
                    text.replace(qualification, qualification * 2),
                    "exact terminal",
                ),
                (
                    text.replace(
                        qualification,
                        "launcher-prefix " + qualification,
                    ),
                    "exact terminal",
                ),
                (
                    text + "nonterminal-data\n",
                    "exact terminal",
                ),
            )
            for index, (candidate, message) in enumerate(cases):
                with self.subTest(index=index, message=message):
                    path = root / f"case-{index}.log"
                    candidate = candidate.replace(
                        qualification,
                        COMPARATOR.QUALIFICATION_PREFIX + path.name + ":PASS\n",
                    )
                    path.write_text(candidate, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        COMPARATOR.validate_initialization_records(path, record)

    def test_counter_misclassification_and_partial_launch_are_rejected(self) -> None:
        counters = overlap_record(enabled=True)
        counters["rejected"] = 1
        with self.assertRaisesRegex(ValueError, "classified exactly once"):
            COMPARATOR.validate_overlap_counters(
                counters, enabled=True, expected_checks=40
            )
        counters = overlap_record(enabled=True)
        counters["launched_h"] -= 1
        with self.assertRaisesRegex(ValueError, "launched exactly once"):
            COMPARATOR.validate_overlap_counters(
                counters, enabled=True, expected_checks=40
            )

    def test_mixed_rank_profile_requires_partial_eligibility(self) -> None:
        record = result_record()
        record["source_enabled"] = True
        record["overlap_material"] = False
        COMPARATOR.validate_fixed_workload(record, "mixed")
        counters = overlap_record(enabled=True)
        counters.update(
            eligible_ranks=1,
            eligible=10,
            launched_h=0,
            launched_e=10,
            rejected=30,
        )
        COMPARATOR.validate_overlap_counters(
            counters,
            enabled=True,
            expected_checks=40,
            mixed=True,
            expected_ranks=2,
        )
        counters["eligible_ranks"] = 2
        with self.assertRaisesRegex(ValueError, "asymmetric"):
            COMPARATOR.validate_overlap_counters(
                counters,
                enabled=True,
                expected_checks=40,
                mixed=True,
                expected_ranks=2,
            )
    def test_duplicate_and_nonfinite_json_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            COMPARATOR.strict_json_loads('{"checks":1,"checks":1}')
        with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
            COMPARATOR.strict_json_loads('{"energy":Infinity}')

    def test_duplicate_record_and_changed_observable_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            enabled_path = root / "enabled.log"
            disabled_path = root / "disabled.log"
            enabled = result_record()
            disabled = result_record()
            disabled["energy"] += 1.0
            write_log(enabled_path, enabled, overlap_record(enabled=True))
            write_log(disabled_path, disabled, overlap_record(enabled=False))
            self.assertNotEqual(
                COMPARATOR.selected(enabled, COMPARATOR.OBSERVABLE_KEYS),
                COMPARATOR.selected(disabled, COMPARATOR.OBSERVABLE_KEYS),
            )
            with enabled_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    COMPARATOR.RESULT_PREFIX + json.dumps(enabled) + "\n"
                )
            with self.assertRaisesRegex(ValueError, "expected exactly one"):
                COMPARATOR.one_record(
                    enabled_path, COMPARATOR.RESULT_PREFIX
                )


if __name__ == "__main__":
    unittest.main()
