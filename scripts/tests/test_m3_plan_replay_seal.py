#!/usr/bin/env python3
"""Tests for the deterministic final M3 plan-replay seal."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/seal_m3_plan_replay.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_plan_replay_seal", SOURCE)
assert SPEC is not None and SPEC.loader is not None
SEAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SEAL
SPEC.loader.exec_module(SEAL)


class M3PlanReplaySealTests(unittest.TestCase):
    def test_derive_binds_clean_commit_code_policy_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            plan_root = root / "plan"
            repo.mkdir()
            plan_root.mkdir()
            responses = ["a" * 40, "", "a" * 40, ""]
            with mock.patch.object(SEAL, "_git", side_effect=responses), mock.patch.object(
                SEAL, "_code_records", return_value={"sealer": {"sha256": "b" * 64}}
            ), mock.patch.object(
                SEAL.m3_feature_plan,
                "load_release_plan",
                return_value={"counts": {"features": 42}},
            ):
                result = SEAL.derive(repo, plan_root)
        self.assertEqual(result["git_commit"], "a" * 40)
        self.assertEqual(result["outcome"], "PASS")
        self.assertIn("excluded_paths", result["source_equivalence_policy"])

    def test_dirty_repository_fails_before_plan_loading(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            plan_root = root / "plan"
            repo.mkdir()
            plan_root.mkdir()
            with mock.patch.object(
                SEAL, "_git", side_effect=["a" * 40, " M source.cpp"]
            ), mock.patch.object(
                SEAL.m3_feature_plan, "load_release_plan"
            ) as loader:
                with self.assertRaisesRegex(SEAL.SealError, "not a clean"):
                    SEAL.derive(repo, plan_root)
            loader.assert_not_called()

    def test_publication_is_exclusive_and_exactly_replayed(self) -> None:
        value = {
            "schema": SEAL.SCHEMA,
            "outcome": "PASS",
            "repository": "/repo",
            "plan_root": "/plan",
            "git_commit": "a" * 40,
        }
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw).resolve() / "seal.json"
            SEAL._publish(path, value)
            with mock.patch.object(SEAL, "derive", return_value=value):
                self.assertEqual(SEAL.verify(path), value)
            with self.assertRaisesRegex(SEAL.SealError, "fresh path"):
                SEAL._publish(path, value)
            path.write_text('{"schema":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(SEAL.SealError, "not a PASS"):
                SEAL.verify(path)

    def test_duplicate_json_and_symlink_inputs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            path = root / "seal.json"
            path.write_text('{"schema":"one","schema":"two"}\n', encoding="utf-8")
            with self.assertRaisesRegex(SEAL.SealError, "duplicate JSON key"):
                SEAL.load(path)
            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(SEAL.SealError, "bounded regular file"):
                SEAL.load(link)


if __name__ == "__main__":
    unittest.main()
