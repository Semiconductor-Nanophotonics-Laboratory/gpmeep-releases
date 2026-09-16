#!/usr/bin/env python3
"""Tests for independent fixed-work AuNP performance replay."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKLOAD_DIR = ROOT / "scripts" / "user-workloads"
sys.path.insert(0, str(WORKLOAD_DIR))


def load_module(name: str):
    path = WORKLOAD_DIR / name
    module_name = f"gpmeep_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMMON = load_module("common.py")
sys.modules["common"] = COMMON
PERFORMANCE = load_module("run_aunp_performance_workload.py")
sys.modules["run_aunp_performance_workload"] = PERFORMANCE
VERIFIER = load_module("verify_aunp_performance_workload.py")


class VerifyAuNPPerformanceWorkloadTests(unittest.TestCase):
    def fixture(self, root: pathlib.Path):
        output = root / "lane"
        output.mkdir()
        (output / "summary.json").write_text("{}\n", encoding="utf-8")
        (output / "COMPLETE").write_text("{}\n", encoding="utf-8")
        receipt_path = root / "receipt.json"
        receipt_path.write_text("{}\n", encoding="utf-8")
        source_start = {
            "file_count": 1,
            "sha256": "1" * 64,
            "source_manifest_schema_version": 2,
            "source_manifest_sha256": "2" * 64,
        }
        receipt = {
            "receipt_id": "3" * 64,
            "build_input_id": "4" * 64,
            "artifact_set_id": "5" * 64,
            "source_start": source_start,
        }
        receipt_record = {
            **receipt,
            "path": str(receipt_path),
            "size_bytes": receipt_path.stat().st_size,
            "sha256": COMMON.sha256_file(receipt_path),
        }
        summary = {
            "build_receipt": receipt_record,
            "rank_records": [
                {
                    "rank": 0,
                    "build_receipt_id": receipt["receipt_id"],
                    "runtime_artifacts": {
                        "libmeep": {
                            "path": str(receipt_path),
                            "size_bytes": receipt_path.stat().st_size,
                            "sha256": COMMON.sha256_file(receipt_path),
                        }
                    },
                    "records": [
                        {"timestep_delta": PERFORMANCE.MINIMUM_TIMESTEPS_PER_PHASE}
                        for _ in PERFORMANCE.PHASE_SPECS
                    ],
                }
            ],
            "fdtd_wall_seconds": 2.0,
            "phase_wall_seconds": [1.0, 1.0],
            "workload_end_to_end_seconds": 5.0,
        }
        return output, receipt_path, receipt, summary

    def test_replay_binds_receipt_runtime_and_raw_timesteps(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            output, receipt_path, receipt, summary = self.fixture(root)
            provenance = types.ModuleType("gpmeep_provenance")

            class ProvenanceError(RuntimeError):
                pass

            provenance.ProvenanceError = ProvenanceError
            provenance.verify_build_receipt = lambda path, repo: receipt
            with mock.patch.dict(
                sys.modules, {"gpmeep_provenance": provenance}
            ), mock.patch.object(
                VERIFIER.performance,
                "validate_performance_output",
                return_value=summary,
            ), mock.patch.object(
                VERIFIER,
                "_external_record_path",
                return_value=receipt_path,
            ), mock.patch.object(
                VERIFIER, "validate_runtime_against_receipt"
            ) as validate_runtime:
                report = VERIFIER.replay(ROOT, output, "cpu", 1)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(
                report["phase_timestep_deltas"],
                [PERFORMANCE.MINIMUM_TIMESTEPS_PER_PHASE]
                * len(PERFORMANCE.PHASE_SPECS),
            )
            validate_runtime.assert_called_once()

    def test_rank_receipt_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            output, receipt_path, receipt, summary = self.fixture(root)
            summary["rank_records"][0]["build_receipt_id"] = "0" * 64
            provenance = types.ModuleType("gpmeep_provenance")

            class ProvenanceError(RuntimeError):
                pass

            provenance.ProvenanceError = ProvenanceError
            provenance.verify_build_receipt = lambda path, repo: receipt
            with mock.patch.dict(
                sys.modules, {"gpmeep_provenance": provenance}
            ), mock.patch.object(
                VERIFIER.performance,
                "validate_performance_output",
                return_value=summary,
            ), mock.patch.object(
                VERIFIER,
                "_external_record_path",
                return_value=receipt_path,
            ):
                with self.assertRaisesRegex(COMMON.WorkloadError, "receipt ID differs"):
                    VERIFIER.replay(ROOT, output, "cpu", 1)

    def test_result_publication_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw) / "replay.json"
            VERIFIER._write_exclusive(path, b"first\n")
            with self.assertRaises(FileExistsError):
                VERIFIER._write_exclusive(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")


if __name__ == "__main__":
    unittest.main()
