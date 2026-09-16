#!/usr/bin/env python3
"""Tests for validation-script-independent build source identities."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/m3_source_equivalence.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_source_equivalence", SOURCE)
assert SPEC is not None and SPEC.loader is not None
IDENTITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IDENTITY
SPEC.loader.exec_module(IDENTITY)


def row(path: str, digest: str) -> dict[str, object]:
    return {
        "kind": "file",
        "mode_octal": "0664",
        "path": path,
        "sha256": digest,
        "size_bytes": 10,
    }


def receipt(script_digest: str = "1" * 64, core_digest: str = "2" * 64):
    source = {
        "algorithm": "fixture",
        "file_count": 2,
        "files": [
            row("scripts/python-validation/m3_feature_plan.py", script_digest),
            row("src/step.cpp", core_digest),
        ],
        "missing_paths": [],
        "sha256": "3" * 64,
        "source_manifest_schema_version": 1,
        "source_manifest_sha256": "4" * 64,
    }
    return {
        "receipt_id": "5" * 64,
        "source_start": source,
        "source_end": json.loads(json.dumps(source)),
    }


class M3SourceEquivalenceTests(unittest.TestCase):
    def derive(self, root: pathlib.Path, value: dict[str, object]):
        path = root / "receipt.json"
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return IDENTITY.derive(path)

    def write(
        self, root: pathlib.Path, name: str, value: dict[str, object]
    ) -> pathlib.Path:
        path = root / name
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def test_validation_script_changes_do_not_change_core_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            first = self.derive(root, receipt(script_digest="1" * 64))
            second = self.derive(root, receipt(script_digest="9" * 64))
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["retained_file_count"], 1)
        self.assertEqual(first["excluded_file_count"], 1)

    def test_non_validation_source_change_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            first = self.derive(root, receipt(core_digest="2" * 64))
            second = self.derive(root, receipt(core_digest="8" * 64))
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_unlisted_build_script_change_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            first_receipt = receipt()
            second_receipt = receipt()
            for value, digest in (
                (first_receipt, "6" * 64),
                (second_receipt, "7" * 64),
            ):
                source = value["source_start"]
                source["files"].append(
                    row("scripts/build-meep-cuda-mpi-python.sh", digest)
                )
                source["files"].sort(key=lambda item: item["path"])
                source["file_count"] = 3
                value["source_end"] = json.loads(json.dumps(source))
            first = self.derive(root, first_receipt)
            second = self.derive(root, second_receipt)
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_compare_reports_exact_allowlisted_delta(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            first_path = self.write(
                root, "first.json", receipt(script_digest="1" * 64)
            )
            second_path = self.write(
                root, "second.json", receipt(script_digest="9" * 64)
            )
            result = IDENTITY.compare(first_path, second_path)
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(
            result["delta"]["modified_paths"],
            ["scripts/python-validation/m3_feature_plan.py"],
        )
        self.assertEqual(result["delta"]["unexpected_paths"], [])

    def test_compare_rejects_unallowlisted_delta(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            first_path = self.write(root, "first.json", receipt(core_digest="2" * 64))
            second_path = self.write(
                root, "second.json", receipt(core_digest="8" * 64)
            )
            with self.assertRaisesRegex(Exception, "escapes"):
                IDENTITY.compare(first_path, second_path)

    def test_start_end_duplicate_and_unsafe_path_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            changed = receipt()
            changed["source_end"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(Exception, "snapshot differs"):
                self.derive(root, changed)
            duplicated = receipt()
            duplicated["source_start"]["files"][1]["path"] = (
                "scripts/python-validation/m3_feature_plan.py"
            )
            duplicated["source_end"] = json.loads(
                json.dumps(duplicated["source_start"])
            )
            with self.assertRaisesRegex(Exception, "path inventory"):
                self.derive(root, duplicated)
            unsafe = receipt()
            unsafe["source_start"]["files"][0]["path"] = "../escape"
            unsafe["source_end"] = json.loads(json.dumps(unsafe["source_start"]))
            with self.assertRaisesRegex(Exception, "row identity"):
                self.derive(root, unsafe)


if __name__ == "__main__":
    unittest.main()
