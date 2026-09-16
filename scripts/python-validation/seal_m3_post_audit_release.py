#!/usr/bin/env python3
"""Seal the post-audit M1--M3 release closure without a circular dependency."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_m3_release_aggregation as aggregation  # noqa: E402
import seal_m3_plan_replay as publication  # noqa: E402
import seal_milestone_adversarial_audit as audit_seal  # noqa: E402


SCHEMA = "gpmeep-m3-post-audit-release-v1"
PREFIX = "GPMEEP_M3_POST_AUDIT_RELEASE="
MATERIAL_SEVERITIES = {"critical", "high", "medium"}
MANDATORY_M4_REQUIREMENTS = {
    "m4.portable_gpu_distribution",
    "m4.private_github_release",
}
OPTIONAL_M4_REQUIREMENTS = {"m4.long_horizon_dispatch"}


class PostAuditError(RuntimeError):
    """Raised when the final aggregate and M3 audit do not close together."""


def _source_state(repo: pathlib.Path) -> dict[str, str]:
    commit = publication._git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    status = publication._git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if publication.COMMIT.fullmatch(commit) is None or status:
        raise PostAuditError("post-audit source is not a clean fixed commit")
    return {"repo": str(repo), "commit": commit, "status_porcelain": status}


def _code_records() -> dict[str, dict[str, Any]]:
    paths = {
        "sealer": pathlib.Path(__file__),
        "aggregator": pathlib.Path(aggregation.__file__),
        "audit_sealer": pathlib.Path(audit_seal.__file__),
        "publication_sealer": pathlib.Path(publication.__file__),
    }
    return {
        name: publication._stable_file_record(path, f"post-audit {name}")
        for name, path in paths.items()
    }


def _validate_report(report: dict[str, Any]) -> dict[str, Any]:
    closure = report.get("requirement_closure")
    if not isinstance(closure, dict):
        raise PostAuditError("aggregate requested-scope closure is absent")
    achieved = closure.get("achieved")
    deferred = closure.get("deferred")
    if (
        closure.get("outcome") != "PASS_WITH_EXPLICIT_M4_DEFERRED_SCOPE"
        or closure.get("achieved_count") != 11
        or not isinstance(achieved, list)
        or len(achieved) != 11
        or any(
            not isinstance(item, dict) or item.get("outcome") != "PASS"
            for item in achieved
        )
        or len({item.get("requirement_id") for item in achieved}) != 11
        or not isinstance(deferred, list)
        or closure.get("deferred_count") != len(deferred)
    ):
        raise PostAuditError("aggregate requested-scope closure differs")
    deferred_ids = {
        item.get("requirement_id")
        for item in deferred
        if isinstance(item, dict)
        and item.get("outcome") == "DEFERRED_TO_M4_COORDINATION"
    }
    allowed = MANDATORY_M4_REQUIREMENTS | OPTIONAL_M4_REQUIREMENTS
    if (
        len(deferred_ids) != len(deferred)
        or not MANDATORY_M4_REQUIREMENTS <= deferred_ids
        or not deferred_ids <= allowed
    ):
        raise PostAuditError("aggregate M4 deferred scope differs")
    audits = report.get("adversarial_audits")
    if (
        not isinstance(audits, dict)
        or set(audits) != {"m1", "m2"}
        or any(
            not isinstance(value, dict) or value.get("outcome") != "PASS"
            for value in audits.values()
        )
    ):
        raise PostAuditError("aggregate M1/M2 adversarial closure differs")
    return {
        "outcome": closure["outcome"],
        "achieved_count": closure["achieved_count"],
        "deferred_count": closure["deferred_count"],
        "deferred_requirement_ids": sorted(deferred_ids),
    }


def derive(
    repo: pathlib.Path,
    aggregation_root: pathlib.Path,
    audit_path: pathlib.Path,
) -> dict[str, Any]:
    repo = publication._canonical_directory(repo, "post-audit source repository")
    aggregation_root = publication._canonical_directory(
        aggregation_root, "M3 aggregation root"
    )
    audit_path = pathlib.Path(os.path.abspath(audit_path))
    before_source = _source_state(repo)
    before_code = _code_records()
    terminal_path = aggregation_root / "M3_COMPLETE"
    before_terminal = publication._stable_file_record(
        terminal_path, "M3 aggregate terminal"
    )
    before_audit = publication._stable_file_record(
        audit_path, "M3 adversarial audit receipt"
    )
    report = aggregation.verify_complete(aggregation_root)
    audit = audit_seal.verify(audit_path)
    audit_evidence = audit.get("evidence")
    unresolved = audit.get("unresolved_counts")
    expected_audit_source = {
        "repo": str(repo),
        "commit": before_source["commit"],
        "status_porcelain": "",
    }
    if (
        audit.get("milestone") != "M3"
        or audit.get("source") != expected_audit_source
        or not isinstance(audit_evidence, dict)
        or audit_evidence.get("root") != str(aggregation_root)
        or audit_evidence.get("terminal_name") != "M3_COMPLETE"
        or audit_evidence.get("terminal") != before_terminal
        or not isinstance(unresolved, dict)
        or any(
            unresolved.get(severity) != 0
            for severity in MATERIAL_SEVERITIES
        )
    ):
        raise PostAuditError("M3 adversarial audit binding differs")
    requested_scope = _validate_report(report)
    after_source = _source_state(repo)
    after_code = _code_records()
    after_terminal = publication._stable_file_record(
        terminal_path, "M3 aggregate terminal"
    )
    after_audit = publication._stable_file_record(
        audit_path, "M3 adversarial audit receipt"
    )
    if (
        after_source != before_source
        or after_code != before_code
        or after_terminal != before_terminal
        or after_audit != before_audit
    ):
        raise PostAuditError("post-audit inputs changed during full replay")
    return {
        "schema": SCHEMA,
        "outcome": "PASS",
        "release_status": "M1_M3_COMPLETE_M4_COORDINATION_REQUIRED",
        "next_action": "STOP_FOR_M4_COORDINATION",
        "source": before_source,
        "aggregation": {
            "root": str(aggregation_root),
            "terminal": before_terminal,
            "report_schema": report["schema"],
            "requested_scope": requested_scope,
            "embedded_adversarial_audits": ["M1", "M2"],
        },
        "m3_adversarial_audit": {
            "receipt": before_audit,
            "auditor": audit["auditor"],
            "finding_counts": audit["finding_counts"],
            "unresolved_counts": audit["unresolved_counts"],
        },
        "adversarial_milestones": ["M1", "M2", "M3"],
        "code": before_code,
    }


def verify(path: pathlib.Path) -> dict[str, Any]:
    retained = publication.load(path)
    if (
        not isinstance(retained, dict)
        or retained.get("schema") != SCHEMA
        or retained.get("outcome") != "PASS"
        or retained.get("release_status")
        != "M1_M3_COMPLETE_M4_COORDINATION_REQUIRED"
        or retained.get("next_action") != "STOP_FOR_M4_COORDINATION"
    ):
        raise PostAuditError("post-audit release receipt is not an exact PASS")
    try:
        expected = derive(
            pathlib.Path(retained["source"]["repo"]),
            pathlib.Path(retained["aggregation"]["root"]),
            pathlib.Path(retained["m3_adversarial_audit"]["receipt"]["path"]),
        )
    except (KeyError, TypeError) as exc:
        raise PostAuditError("post-audit release receipt paths differ") from exc
    if retained != expected:
        raise PostAuditError("post-audit release receipt was not exactly re-derived")
    return retained


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    operations = parser.add_subparsers(dest="operation", required=True)
    create = operations.add_parser("create", allow_abbrev=False)
    create.add_argument("--repo", required=True, type=pathlib.Path)
    create.add_argument("--aggregation-root", required=True, type=pathlib.Path)
    create.add_argument("--m3-audit", required=True, type=pathlib.Path)
    create.add_argument("--output", required=True, type=pathlib.Path)
    replay = operations.add_parser("verify", allow_abbrev=False)
    replay.add_argument("--input", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.operation == "create":
        repo = publication._canonical_directory(args.repo, "source repository")
        aggregation_root = publication._canonical_directory(
            args.aggregation_root, "M3 aggregation root"
        )
        audit_path = pathlib.Path(os.path.abspath(args.m3_audit))
        output = pathlib.Path(os.path.abspath(args.output))
        if (
            output == audit_path
            or output == repo
            or repo in output.parents
            or output == aggregation_root
            or aggregation_root in output.parents
        ):
            raise PostAuditError("post-audit output overlaps an input")
        publication._publish(
            output, derive(repo, aggregation_root, audit_path)
        )
        value = verify(output)
        path = output
    else:
        path = args.input
        value = verify(path)
    digest = publication._stable_file_record(
        path, "post-audit release receipt"
    )["sha256"]
    print(
        PREFIX
        + json.dumps(
            {
                "schema": SCHEMA,
                "outcome": value["outcome"],
                "release_status": value["release_status"],
                "sha256": digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        ValueError,
        PostAuditError,
        publication.SealError,
        audit_seal.AuditError,
        aggregation.WorkloadError,
    ) as error:
        print(f"post-audit release error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
