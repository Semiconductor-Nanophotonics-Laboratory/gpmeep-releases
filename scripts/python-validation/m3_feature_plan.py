#!/usr/bin/env python3
"""Load and route the frozen 42-feature M3 GPU evidence contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import re
import stat
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m3_execution_plan as execution_plan  # noqa: E402
import m3_specialized_plan as specialized_plan  # noqa: E402


SCHEMA = "gpmeep-m3-feature-plan-v1"
REQUIREMENTS_NAME = "M3_GPU_FEATURE_REQUIREMENTS_DRAFT.tsv"
EVIDENCE_NAME = "M3_GPU_FEATURE_EVIDENCE_PLAN_DRAFT.tsv"
EXPECTED_SHA256 = {
    REQUIREMENTS_NAME: "45f3ac7511f7b38b78a6bbd63d4cc09a78073fe72b2da1bb002c569a1cef44ba",
    EVIDENCE_NAME: "2912cb14a24cb9709a98a530c182532355b5bb2ae1093cf5fdf80e91abd74a55",
}
REQUIREMENT_FIELDS = (
    "feature_id",
    "family",
    "feature",
    "primary_oracle",
    "required_evidence",
    "topology_or_performance",
)
EVIDENCE_FIELDS = (
    "feature_id",
    "primary_current_receipt_evidence",
    "independent_or_native_evidence",
    "required_specialized_counter_or_gate",
    "release_topology",
    "status",
)
FAMILY_COUNTS = {
    "solver_physics": 19,
    "monitor_transform_adjoint": 13,
    "distributed_policy": 10,
}
STATUS_BY_FEATURE = {
    "distributed.cuda_aware_mpi": "planned_capability_gated",
    "policy.long_horizon_small": "planned_policy_gap_allowed",
}
MILESTONES = {"milestone:M2-AuNP", "milestone:M3-example-matrix"}
FEATURE_ID = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+\Z")


class PlanError(RuntimeError):
    """Raised when the feature contract or its execution routing differs."""


def _stable_bytes(path: pathlib.Path) -> bytes:
    path = pathlib.Path(os.path.abspath(path))
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PlanError("M3 feature input is not a regular non-symlink file")
    if before.st_size > 4 * 1024 * 1024:
        raise PlanError("M3 feature input exceeds its size bound")
    with path.open("rb") as handle:
        value = handle.read()
        descriptor = os.fstat(handle.fileno())
    after = path.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(descriptor) or identity(before) != identity(after):
        raise PlanError("M3 feature input changed while it was read")
    return value


def _rows(value: bytes, fields: tuple[str, ...], label: str) -> list[dict[str, str]]:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PlanError(f"{label} is not strict UTF-8") from exc
    if (
        "\r" in text
        or "\x00" in text
        or '"' in text
        or not text.endswith("\n")
        or any(line.count("\t") != len(fields) - 1 for line in text.splitlines())
    ):
        raise PlanError(f"{label} has noncanonical text encoding")
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t", strict=True)
        if tuple(reader.fieldnames or ()) != fields:
            raise PlanError(f"{label} header differs")
        rows = list(reader)
    except csv.Error as exc:
        raise PlanError(f"{label} is malformed TSV: {exc}") from exc
    if (
        len(rows) != 42
        or any(set(row) != set(fields) for row in rows)
        or any(any(not value for value in row.values()) for row in rows)
    ):
        raise PlanError(f"{label} must contain exactly 42 complete rows")
    return rows


def _tokens(value: str, label: str, *, allow_milestone: bool) -> list[str]:
    tokens = value.split(";")
    if len(tokens) != len(set(tokens)) or any(not token for token in tokens):
        raise PlanError(f"M3 feature {label} tokens differ")
    for token in tokens:
        if token.startswith("milestone:"):
            if not allow_milestone or token not in MILESTONES:
                raise PlanError(f"M3 feature {label} milestone differs")
            continue
        path = pathlib.PurePosixPath(token)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in token
            or path.as_posix() != token
            or not token.startswith(("python/", "scripts/", "tests/"))
        ):
            raise PlanError(f"M3 feature {label} path differs")
    return tokens


def parse_contract(requirements: bytes, evidence: bytes) -> list[dict[str, Any]]:
    requirement_rows = _rows(requirements, REQUIREMENT_FIELDS, REQUIREMENTS_NAME)
    evidence_rows = _rows(evidence, EVIDENCE_FIELDS, EVIDENCE_NAME)
    requirement_ids = [row["feature_id"] for row in requirement_rows]
    evidence_ids = [row["feature_id"] for row in evidence_rows]
    if (
        requirement_ids != evidence_ids
        or len(requirement_ids) != len(set(requirement_ids))
        or any(FEATURE_ID.fullmatch(value) is None for value in requirement_ids)
    ):
        raise PlanError("M3 feature ID inventory differs")
    family_counts = {
        family: sum(row["family"] == family for row in requirement_rows)
        for family in FAMILY_COUNTS
    }
    if family_counts != FAMILY_COUNTS:
        raise PlanError("M3 feature family partition differs")
    features = []
    for requirement, route in zip(requirement_rows, evidence_rows, strict=True):
        feature_id = requirement["feature_id"]
        expected_family = (
            "solver_physics"
            if feature_id.startswith("solver.")
            else "monitor_transform_adjoint"
            if feature_id.startswith("monitor.")
            else "distributed_policy"
        )
        expected_status = STATUS_BY_FEATURE.get(feature_id, "planned")
        if (
            requirement["family"] != expected_family
            or route["status"] != expected_status
            or any(
                len(requirement[name].encode("utf-8")) > 4096
                for name in REQUIREMENT_FIELDS[2:]
            )
            or any(
                len(route[name].encode("utf-8")) > 4096
                for name in EVIDENCE_FIELDS[1:5]
            )
        ):
            raise PlanError(f"M3 feature row differs: {feature_id}")
        features.append(
            {
                **requirement,
                "primary_current_receipt_evidence": _tokens(
                    route["primary_current_receipt_evidence"],
                    "primary evidence",
                    allow_milestone=True,
                ),
                "independent_or_native_evidence": _tokens(
                    route["independent_or_native_evidence"],
                    "independent evidence",
                    allow_milestone=False,
                ),
                "required_specialized_counter_or_gate": route[
                    "required_specialized_counter_or_gate"
                ],
                "release_topology": route["release_topology"],
                "status": route["status"],
            }
        )
    return features


def _repo_file(repo: pathlib.Path, value: str) -> pathlib.Path:
    path = repo.joinpath(*pathlib.PurePosixPath(value).parts)
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or path.resolve(strict=True) != path
    ):
        raise PlanError(f"M3 feature evidence is not a repository file: {value}")
    return path


def _execution_routes(
    execution: dict[str, Any],
    specialized: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    routes: dict[str, list[dict[str, str]]] = {}

    def add(path: str, kind: str, unit_id: str) -> None:
        routes.setdefault(path, []).append({"kind": kind, "unit_id": unit_id})

    for row in (*execution["selected"], *execution["stronger"]):
        add(row["case_path"], "paired", row["unit_id"])
    for row in execution["host"]:
        for path in row["covers_examples"]:
            add(path, "host", row["unit_id"])
    driver_paths = {
        "run-mpi-python-validation": "scripts/run-mpi-python-validation.py",
        "run-near2far-mpi-qualification": (
            "scripts/run-near2far-mpi-qualification.py"
        ),
        "run-mpi-adjoint-benchmark": "scripts/run-mpi-adjoint-benchmark.py",
    }
    for row in specialized["units"]:
        add(row["target_path"], "specialized", row["unit_id"])
        if row["driver"] in driver_paths:
            add(driver_paths[row["driver"]], "specialized", row["unit_id"])
    return routes


def load_release_plan(plan_root: pathlib.Path, repo: pathlib.Path) -> dict[str, Any]:
    plan_root = pathlib.Path(os.path.abspath(plan_root))
    repo = pathlib.Path(os.path.abspath(repo))
    for root, label in ((plan_root, "plan root"), (repo, "repository")):
        if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
            raise PlanError(f"M3 feature {label} differs")
    payloads = {}
    inputs = {}
    for name, digest in EXPECTED_SHA256.items():
        path = plan_root / name
        value = _stable_bytes(path)
        actual = hashlib.sha256(value).hexdigest()
        if actual != digest:
            raise PlanError(f"M3 feature {name} SHA-256 differs")
        payloads[name] = value
        inputs[name] = {
            "path": str(path.resolve(strict=True)),
            "size_bytes": len(value),
            "sha256": actual,
        }
    features = parse_contract(
        payloads[REQUIREMENTS_NAME], payloads[EVIDENCE_NAME]
    )
    execution = execution_plan.load_release_plan(plan_root, repo)
    specialized = specialized_plan.load_release_plan(plan_root, repo)
    routes = _execution_routes(execution, specialized)
    for feature in features:
        primary_routes = []
        for evidence in feature["primary_current_receipt_evidence"]:
            if evidence in MILESTONES:
                route = [{"kind": "milestone", "unit_id": evidence}]
            else:
                _repo_file(repo, evidence)
                route = routes.get(evidence, [])
            if not route:
                raise PlanError(
                    f"M3 primary evidence has no execution route: {evidence}"
                )
            primary_routes.append({"evidence": evidence, "routes": route})
        for evidence in feature["independent_or_native_evidence"]:
            _repo_file(repo, evidence)
        feature["primary_routes"] = primary_routes
    return {
        "schema": SCHEMA,
        "inputs": inputs,
        "features": features,
        "component_plans": {
            "execution": execution["inputs"],
            "specialized": specialized["input"],
        },
        "counts": {
            "features": 42,
            "solver_physics": 19,
            "monitor_transform_adjoint": 13,
            "distributed_policy": 10,
            "ordinary_release": 40,
            "capability_gated": 1,
            "policy_gap_allowed": 1,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-root", required=True, type=pathlib.Path)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            load_release_plan(args.plan_root, args.repo),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, PlanError, ValueError) as error:
        print(f"M3 feature plan error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
