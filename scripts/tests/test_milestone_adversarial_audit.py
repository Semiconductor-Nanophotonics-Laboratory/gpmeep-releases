#!/usr/bin/env python3
"""Tests for durable adversarial subagent milestone-audit receipts."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/seal_milestone_adversarial_audit.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_milestone_audit", SOURCE)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def draft(repo: pathlib.Path, evidence: pathlib.Path) -> dict[str, object]:
    return {
        "schema": AUDIT.DRAFT_SCHEMA,
        "milestone": "M1",
        "auditor": {"agent_id": "agent-123", "task_name": "m1_adversarial"},
        "source": {"repo": str(repo), "commit": "a" * 40},
        "evidence": {"root": str(evidence), "terminal_name": "COMPLETE"},
        "scope": ["terminal replay", "numerical comparison", "performance gates"],
        "findings": [
            {
                "id": "M1-AUDIT-001",
                "severity": "medium",
                "status": "closed",
                "summary": "fixture finding",
                "resolution": "fixture correction",
                "verification": "fixture replay passed",
            }
        ],
        "review": "Independent adversarial fixture review.",
    }


class MilestoneAdversarialAuditTests(unittest.TestCase):
    def fixture(self, root: pathlib.Path):
        repo = root / "repo"
        evidence = root / "evidence"
        repo.mkdir()
        evidence.mkdir()
        AUDIT.publication._publish(
            evidence / "COMPLETE", {"schema": "fixture", "outcome": "PASS"}
        )
        return repo, evidence

    def derive(self, value: dict[str, object]) -> dict[str, object]:
        with mock.patch.object(
            AUDIT, "_source_state"
        ) as source_state:
            source_state.return_value = {
                "repo": value["source"]["repo"],
                "commit": value["source"]["commit"],
                "status_porcelain": "",
            }
            return AUDIT.derive(value)

    def test_receipt_binds_agent_source_terminal_and_closed_findings(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo, evidence = self.fixture(pathlib.Path(raw).resolve())
            result = self.derive(draft(repo, evidence))
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(result["finding_counts"]["medium"], 1)
        self.assertEqual(result["unresolved_counts"]["medium"], 0)
        self.assertEqual(result["evidence"]["terminal"]["sha256"].__len__(), 64)

    def test_unresolved_material_finding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo, evidence = self.fixture(pathlib.Path(raw).resolve())
            value = draft(repo, evidence)
            value["findings"][0]["status"] = "open"
            with self.assertRaisesRegex(AUDIT.AuditError, "unresolved material"):
                self.derive(value)

    def test_changed_commit_and_terminal_failure_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo, evidence = self.fixture(pathlib.Path(raw).resolve())
            value = draft(repo, evidence)
            with mock.patch.object(
                AUDIT,
                "_source_state",
                return_value={
                    "repo": str(repo),
                    "commit": "b" * 40,
                    "status_porcelain": "",
                },
            ):
                with self.assertRaisesRegex(AUDIT.AuditError, "commit differs"):
                    AUDIT.derive(value)
            (evidence / "COMPLETE").write_text(
                '{"outcome":"FAIL"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(AUDIT.AuditError, "terminal is not a PASS"):
                self.derive(value)

    def test_published_receipt_replays_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo, evidence = self.fixture(root)
            value = self.derive(draft(repo, evidence))
            output = root / "audit.json"
            AUDIT.publication._publish(output, value)
            with mock.patch.object(AUDIT, "derive", return_value=value):
                self.assertEqual(AUDIT.verify(output), value)
            value["review"] = "tampered"
            output.write_text(
                AUDIT.json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
            )
            with mock.patch.object(AUDIT, "derive", return_value={}):
                with self.assertRaisesRegex(AUDIT.AuditError, "not exactly"):
                    AUDIT.verify(output)


if __name__ == "__main__":
    unittest.main()
