#!/usr/bin/env python3
"""Tests for the non-circular final M3 post-audit release receipt."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/seal_m3_post_audit_release.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_post_audit", SOURCE)
assert SPEC is not None and SPEC.loader is not None
SEAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SEAL
SPEC.loader.exec_module(SEAL)


COMMIT = "a" * 40


def report() -> dict[str, object]:
    return {
        "schema": "gpmeep-m3-release-aggregation-v3",
        "requirement_closure": {
            "outcome": "PASS_WITH_EXPLICIT_M4_DEFERRED_SCOPE",
            "achieved_count": 11,
            "deferred_count": 3,
            "achieved": [
                {
                    "requirement_id": f"fixture.achieved.{index}",
                    "outcome": "PASS",
                }
                for index in range(11)
            ],
            "deferred": [
                {
                    "requirement_id": requirement_id,
                    "outcome": "DEFERRED_TO_M4_COORDINATION",
                }
                for requirement_id in (
                    "m4.portable_gpu_distribution",
                    "m4.private_github_release",
                    "m4.long_horizon_dispatch",
                )
            ],
        },
        "adversarial_audits": {
            "m1": {"outcome": "PASS"},
            "m2": {"outcome": "PASS"},
        },
    }


class M3PostAuditReleaseTests(unittest.TestCase):
    def fixture(self, root: pathlib.Path):
        repo = root / "repo"
        aggregate = root / "aggregate"
        repo.mkdir()
        aggregate.mkdir()
        SEAL.publication._publish(
            aggregate / "M3_COMPLETE",
            {"schema": "fixture", "outcome": "PASS"},
        )
        audit = root / "m3-audit.json"
        SEAL.publication._publish(
            audit, {"schema": "fixture", "outcome": "PASS"}
        )
        terminal = SEAL.publication._stable_file_record(
            aggregate / "M3_COMPLETE", "fixture terminal"
        )
        audit_value = {
            "milestone": "M3",
            "source": {
                "repo": str(repo),
                "commit": COMMIT,
                "status_porcelain": "",
            },
            "evidence": {
                "root": str(aggregate),
                "terminal_name": "M3_COMPLETE",
                "terminal": terminal,
            },
            "auditor": {"agent_id": "agent-123", "task_name": "m3_adversarial"},
            "finding_counts": {
                severity: 0 for severity in SEAL.audit_seal.SEVERITIES
            },
            "unresolved_counts": {
                severity: 0 for severity in SEAL.audit_seal.SEVERITIES
            },
        }
        return repo, aggregate, audit, audit_value

    def derive(
        self,
        repo: pathlib.Path,
        aggregate: pathlib.Path,
        audit: pathlib.Path,
        audit_value: dict[str, object],
        aggregate_report: dict[str, object] | None = None,
    ) -> dict[str, object]:
        source = {"repo": str(repo), "commit": COMMIT, "status_porcelain": ""}
        with mock.patch.object(
            SEAL, "_source_state", return_value=source
        ), mock.patch.object(
            SEAL.aggregation,
            "verify_complete",
            return_value=aggregate_report or report(),
        ), mock.patch.object(
            SEAL.audit_seal, "verify", return_value=audit_value
        ):
            return SEAL.derive(repo, aggregate, audit)

    def test_binds_aggregate_all_audits_and_m4_stop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo, aggregate, audit, audit_value = self.fixture(
                pathlib.Path(raw).resolve()
            )
            value = self.derive(repo, aggregate, audit, audit_value)
        self.assertEqual(value["outcome"], "PASS")
        self.assertEqual(
            value["release_status"], "M1_M3_COMPLETE_M4_COORDINATION_REQUIRED"
        )
        self.assertEqual(value["next_action"], "STOP_FOR_M4_COORDINATION")
        self.assertEqual(value["adversarial_milestones"], ["M1", "M2", "M3"])
        self.assertEqual(
            value["aggregation"]["requested_scope"]["achieved_count"], 11
        )

    def test_m3_audit_binding_and_material_closure_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo, aggregate, audit, audit_value = self.fixture(
                pathlib.Path(raw).resolve()
            )
            for mutation in ("commit", "root", "material"):
                changed = {
                    **audit_value,
                    "source": dict(audit_value["source"]),
                    "evidence": dict(audit_value["evidence"]),
                    "unresolved_counts": dict(audit_value["unresolved_counts"]),
                }
                if mutation == "commit":
                    changed["source"]["commit"] = "b" * 40
                elif mutation == "root":
                    changed["evidence"]["root"] = str(repo)
                else:
                    changed["unresolved_counts"]["medium"] = 1
                with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    SEAL.PostAuditError, "audit binding"
                ):
                    self.derive(repo, aggregate, audit, changed)

    def test_requested_scope_must_be_exact_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo, aggregate, audit, audit_value = self.fixture(
                pathlib.Path(raw).resolve()
            )
            changed = report()
            changed["requirement_closure"]["achieved"].pop()
            with self.assertRaisesRegex(SEAL.PostAuditError, "requested-scope"):
                self.derive(repo, aggregate, audit, audit_value, changed)
            changed = report()
            changed["requirement_closure"]["deferred"][0][
                "requirement_id"
            ] = "m4.unknown"
            with self.assertRaisesRegex(SEAL.PostAuditError, "M4 deferred"):
                self.derive(repo, aggregate, audit, audit_value, changed)

    def test_publication_is_exclusive_and_exactly_replayed(self) -> None:
        value = {
            "schema": SEAL.SCHEMA,
            "outcome": "PASS",
            "release_status": "M1_M3_COMPLETE_M4_COORDINATION_REQUIRED",
            "next_action": "STOP_FOR_M4_COORDINATION",
            "source": {"repo": "/repo"},
            "aggregation": {"root": "/aggregate"},
            "m3_adversarial_audit": {"receipt": {"path": "/audit"}},
        }
        with tempfile.TemporaryDirectory() as raw:
            path = pathlib.Path(raw).resolve() / "release.json"
            SEAL.publication._publish(path, value)
            with mock.patch.object(SEAL, "derive", return_value=value):
                self.assertEqual(SEAL.verify(path), value)
            with self.assertRaisesRegex(SEAL.publication.SealError, "fresh path"):
                SEAL.publication._publish(path, value)
            path.write_text('{"schema":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(SEAL.PostAuditError, "not an exact PASS"):
                SEAL.verify(path)


if __name__ == "__main__":
    unittest.main()
