#!/usr/bin/env python3
"""Tests for replaying M1 with its immutable pre-M2 verifier tree."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/replay_external_m1.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_external_m1_replay_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


def source_row(path: str, digest: str) -> dict[str, object]:
    return {
        "kind": "file",
        "mode_octal": "0664",
        "path": path,
        "sha256": digest,
        "size_bytes": 1,
    }


class ExternalM1ReplayTests(unittest.TestCase):
    def test_requires_exactly_one_passing_comparator_replay(self) -> None:
        for invocation_count in (0, 2):
            with self.subTest(invocation_count=invocation_count):
                with tempfile.TemporaryDirectory() as raw:
                    root = pathlib.Path(raw).resolve()
                    repo = root / "old-repo"
                    workload = repo / "scripts/user-workloads"
                    workload.mkdir(parents=True)
                    output = root / "evidence"
                    output.mkdir()
                    calls = "\n    ".join(
                        "matrix.COMPARATOR.compare_ters(None)"
                        for _index in range(invocation_count)
                    ) or "pass"
                    (workload / "run_hybrid_ters_matrix.py").write_text(
                        f"""
class Comparator:
    @staticmethod
    def compare_ters(arguments):
        return {{"outcome": "PASS"}}
class Matrix:
    COMPARATOR = Comparator()
matrix = Matrix()
def verify_hybrid_complete(output):
    {calls}
    return {{}}
""".lstrip(),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        REPLAY.ReplayError,
                        "did not replay one exact comparison",
                    ):
                        REPLAY.replay(repo, output)

    def test_replays_old_verifier_and_derives_source_partition(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "old-repo"
            workload = repo / "scripts/user-workloads"
            workload.mkdir(parents=True)
            output = root / "evidence"
            output.mkdir()
            receipt_path = repo / "receipt.json"
            source = {
                "algorithm": "fixture",
                "file_count": 2,
                "files": [
                    source_row(
                        "scripts/python-validation/m3_feature_plan.py",
                        "1" * 64,
                    ),
                    source_row("src/a.cpp", "2" * 64),
                ],
                "missing_paths": [],
                "sha256": "3" * 64,
                "source_manifest_schema_version": 1,
                "source_manifest_sha256": "4" * 64,
            }
            receipt_path.write_text(
                json.dumps(
                    {
                        "receipt_id": "5" * 64,
                        "source_start": source,
                        "source_end": source,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            verifier = workload / "run_hybrid_ters_matrix.py"
            verifier.write_text(
                """
import hashlib
class Matrix:
    class Comparator:
        @staticmethod
        def compare_ters(arguments):
            return {
                "schema": "fixture-comparison-v1",
                "comparisons": [
                    {"label": "cuda-fp32-1g", "outcome": "PASS"}
                ],
                "outcome": "PASS",
            }
    COMPARATOR = Comparator()

    @staticmethod
    def stable_file_record(path, root, label):
        value = path.read_bytes()
        return {
            "path": str(path.relative_to(root)),
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
matrix = Matrix()
def verify_hybrid_complete(output):
    matrix.COMPARATOR.compare_ters(None)
    return {
        "source_snapshot_sha256": "3" * 64,
        "gpu_devices": ["GPU-a", "GPU-b"],
        "tasks": [{"task": {"backend": "cuda", "build_receipt_id": "5" * 64}}],
        "performance": {"speed": {"outcome": "PASS"}},
        "timing_stability": {"cuda": {"outcome": "PASS"}},
        "exact_comparison": {},
    }
""".lstrip(),
                encoding="utf-8",
            )
            (output / "CHECKPOINT.json").write_text(
                json.dumps(
                    {
                        "contract": {
                            "builds": {
                                "shared_fp32": {
                                    "receipt_path": str(receipt_path)
                                }
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            verifier_sha256 = hashlib.sha256(verifier.read_bytes()).hexdigest()
            result = REPLAY.replay(repo, output)
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(result["task_count"], 1)
        self.assertEqual(result["exact_comparison"]["outcome"], "PASS")
        self.assertEqual(
            result["shared_fp32_source_identity"]["retained_file_count"], 1
        )
        self.assertEqual(
            result["verifier"]["sha256"],
            verifier_sha256,
        )


if __name__ == "__main__":
    unittest.main()
