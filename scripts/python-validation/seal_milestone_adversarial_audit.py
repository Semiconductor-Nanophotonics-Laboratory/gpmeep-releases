#!/usr/bin/env python3
"""Seal and replay an adversarial subagent milestone-audit receipt."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from collections import Counter
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import seal_m3_plan_replay as publication  # noqa: E402


DRAFT_SCHEMA = "gpmeep-adversarial-audit-draft-v1"
SCHEMA = "gpmeep-adversarial-audit-receipt-v1"
MILESTONES = {"M1", "M2", "M3"}
SEVERITIES = {"critical", "high", "medium", "low", "info"}
STATUSES = {"closed", "open"}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
HIGH_SEVERITIES = {"critical", "high", "medium"}
MAXIMUM_REVIEW_BYTES = 4 * 1024**2


class AuditError(RuntimeError):
    """Raised when an adversarial audit is incomplete or has changed."""


def _text(value: Any, label: str, maximum: int = 16_384) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > maximum
    ):
        raise AuditError(f"adversarial audit {label} differs")
    return value


def _source_state(repo: pathlib.Path) -> dict[str, str]:
    commit = publication._git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    status = publication._git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if publication.COMMIT.fullmatch(commit) is None or status:
        raise AuditError("adversarial audit source is not a clean fixed commit")
    return {"repo": str(repo), "commit": commit, "status_porcelain": status}


def _finding(value: Any, milestone: str) -> dict[str, str]:
    keys = {"id", "severity", "status", "summary", "resolution", "verification"}
    if not isinstance(value, dict) or set(value) != keys:
        raise AuditError("adversarial audit finding schema differs")
    finding_id = _text(value["id"], "finding ID", 128)
    if SAFE_ID.fullmatch(finding_id) is None or not finding_id.startswith(
        f"{milestone}-"
    ):
        raise AuditError("adversarial audit finding ID differs")
    severity = value["severity"]
    status = value["status"]
    if severity not in SEVERITIES or status not in STATUSES:
        raise AuditError("adversarial audit finding classification differs")
    summary = _text(value["summary"], "finding summary")
    resolution = _text(value["resolution"], "finding resolution")
    verification = _text(value["verification"], "finding verification")
    if severity in HIGH_SEVERITIES and status != "closed":
        raise AuditError("adversarial audit has an unresolved material finding")
    return {
        "id": finding_id,
        "severity": severity,
        "status": status,
        "summary": summary,
        "resolution": resolution,
        "verification": verification,
    }


def derive(draft: Any) -> dict[str, Any]:
    keys = {
        "schema",
        "milestone",
        "auditor",
        "source",
        "evidence",
        "scope",
        "findings",
        "review",
    }
    if not isinstance(draft, dict) or set(draft) != keys:
        raise AuditError("adversarial audit draft schema differs")
    if draft.get("schema") != DRAFT_SCHEMA or draft.get("milestone") not in MILESTONES:
        raise AuditError("adversarial audit draft identity differs")
    milestone = draft["milestone"]
    auditor = draft["auditor"]
    if not isinstance(auditor, dict) or set(auditor) != {"agent_id", "task_name"}:
        raise AuditError("adversarial audit agent identity differs")
    auditor = {
        "agent_id": _text(auditor["agent_id"], "agent ID", 128),
        "task_name": _text(auditor["task_name"], "task name", 256),
    }
    if SAFE_ID.fullmatch(auditor["agent_id"]) is None:
        raise AuditError("adversarial audit agent ID is not canonical")

    source = draft["source"]
    evidence = draft["evidence"]
    if not isinstance(source, dict) or set(source) != {"repo", "commit"}:
        raise AuditError("adversarial audit source binding differs")
    if not isinstance(evidence, dict) or set(evidence) != {"root", "terminal_name"}:
        raise AuditError("adversarial audit evidence binding differs")
    repo = publication._canonical_directory(
        pathlib.Path(_text(source["repo"], "source repository", 4096)),
        "audit source repository",
    )
    root = publication._canonical_directory(
        pathlib.Path(_text(evidence["root"], "evidence root", 4096)),
        "audit evidence root",
    )
    source_state = _source_state(repo)
    if source.get("commit") != source_state["commit"]:
        raise AuditError("adversarial audit source commit differs")
    terminal_name = _text(evidence["terminal_name"], "terminal name", 128)
    terminal_relative = pathlib.PurePosixPath(terminal_name)
    if (
        terminal_relative.name != terminal_name
        or terminal_relative.is_absolute()
        or "\\" in terminal_name
    ):
        raise AuditError("adversarial audit terminal name is unsafe")
    terminal_path = root / terminal_name
    terminal_record = publication._stable_file_record(
        terminal_path, "adversarial audit terminal"
    )
    terminal = publication.load(terminal_path)
    if terminal.get("outcome") != "PASS":
        raise AuditError("adversarial audit target terminal is not a PASS")
    if terminal_record != publication._stable_file_record(
        terminal_path, "adversarial audit terminal"
    ):
        raise AuditError("adversarial audit terminal changed during replay")

    scope = draft["scope"]
    if (
        not isinstance(scope, list)
        or not scope
        or len(scope) > 256
        or len(scope) != len(set(scope))
        or any(
            not isinstance(value, str)
            or not value.strip()
            or len(value.encode("utf-8")) > 4096
            for value in scope
        )
    ):
        raise AuditError("adversarial audit scope differs")
    findings_raw = draft["findings"]
    if not isinstance(findings_raw, list) or len(findings_raw) > 4096:
        raise AuditError("adversarial audit finding inventory differs")
    findings = [_finding(value, milestone) for value in findings_raw]
    if len({value["id"] for value in findings}) != len(findings):
        raise AuditError("adversarial audit finding IDs are not unique")
    review = _text(draft["review"], "review", MAXIMUM_REVIEW_BYTES)
    counts = Counter(value["severity"] for value in findings)
    unresolved = Counter(
        value["severity"] for value in findings if value["status"] == "open"
    )
    if _source_state(repo) != source_state:
        raise AuditError("adversarial audit source changed during replay")
    return {
        "schema": SCHEMA,
        "outcome": "PASS",
        "milestone": milestone,
        "auditor": auditor,
        "source": source_state,
        "evidence": {
            "root": str(root),
            "terminal_name": terminal_name,
            "terminal": terminal_record,
        },
        "scope": scope,
        "findings": findings,
        "finding_counts": {
            severity: counts[severity] for severity in sorted(SEVERITIES)
        },
        "unresolved_counts": {
            severity: unresolved[severity] for severity in sorted(SEVERITIES)
        },
        "review": review,
    }


def _draft_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    try:
        return {
            "schema": DRAFT_SCHEMA,
            "milestone": receipt["milestone"],
            "auditor": receipt["auditor"],
            "source": {
                "repo": receipt["source"]["repo"],
                "commit": receipt["source"]["commit"],
            },
            "evidence": {
                "root": receipt["evidence"]["root"],
                "terminal_name": receipt["evidence"]["terminal_name"],
            },
            "scope": receipt["scope"],
            "findings": receipt["findings"],
            "review": receipt["review"],
        }
    except (KeyError, TypeError) as exc:
        raise AuditError("adversarial audit receipt fields differ") from exc


def verify(path: pathlib.Path) -> dict[str, Any]:
    retained = publication.load(path)
    expected_keys = {
        "schema",
        "outcome",
        "milestone",
        "auditor",
        "source",
        "evidence",
        "scope",
        "findings",
        "finding_counts",
        "unresolved_counts",
        "review",
    }
    if (
        set(retained) != expected_keys
        or retained.get("schema") != SCHEMA
        or retained.get("outcome") != "PASS"
    ):
        raise AuditError("adversarial audit receipt is not an exact PASS")
    derived = derive(_draft_from_receipt(retained))
    if retained != derived:
        raise AuditError("adversarial audit receipt was not exactly re-derived")
    return retained


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    operations = parser.add_subparsers(dest="operation", required=True)
    create = operations.add_parser("create", allow_abbrev=False)
    create.add_argument("--draft", required=True, type=pathlib.Path)
    create.add_argument("--output", required=True, type=pathlib.Path)
    replay = operations.add_parser("verify", allow_abbrev=False)
    replay.add_argument("--input", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.operation == "create":
        draft = publication.load(args.draft)
        value = derive(draft)
        output = pathlib.Path(os.path.abspath(args.output))
        repo = pathlib.Path(value["source"]["repo"])
        evidence = pathlib.Path(value["evidence"]["root"])
        if (
            output == repo
            or repo in output.parents
            or output == evidence
            or evidence in output.parents
        ):
            raise AuditError("adversarial audit output overlaps an input root")
        publication._publish(output, value)
        value = verify(output)
        path = output
    else:
        path = args.input
        value = verify(path)
    digest = publication._stable_file_record(
        path, "adversarial audit receipt"
    )["sha256"]
    print(
        "GPMEEP_ADVERSARIAL_AUDIT="
        + json.dumps(
            {
                "schema": SCHEMA,
                "outcome": value["outcome"],
                "milestone": value["milestone"],
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
    except (OSError, AuditError, publication.SealError, ValueError) as error:
        print(f"adversarial audit error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
