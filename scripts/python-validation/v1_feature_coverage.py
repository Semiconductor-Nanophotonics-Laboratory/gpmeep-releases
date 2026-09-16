#!/usr/bin/env python3
"""Validate the self-contained gpmeep v1 feature-coverage contract.

This module validates the plan, not completed executions.  A route is only
eligible for release evidence after a separate execution aggregator binds its
CPU/GPU outputs to this exact manifest digest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import stat
import sys
from collections import Counter, defaultdict
from typing import Any


SCHEMA = "gpmeep-v1-feature-coverage-v1"
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "v1_feature_coverage.json"
DEFAULT_RUNNER_MANIFEST = SCRIPT_DIR / "manifest.json"
MAX_INPUT_BYTES = 2 * 1024 * 1024
EXPECTED_ROUTE_COUNT = 32
EXPECTED_FEATURE_CONTRACT_SHA256 = (
    "5cedc727435619863d966fc7aef7ea8fa947c8ac92c28050844dbcd50984fcf2"
)
EXPECTED_ROUTE_CONTRACT_SHA256 = (
    "ed2d1491d8a34d16f7b06bf6675698aef2b3b58af83d49b8528b21084c7600c6"
)

EXPECTED_FEATURES = {
    "solver.real_cartesian",
    "solver.complex_bloch",
    "solver.dimensions",
    "solver.pml_conductivity",
    "solver.absorber",
    "solver.pec",
    "solver.standard_sources",
    "solver.custom_sources",
    "solver.stochastic_sources",
    "solver.dispersion",
    "solver.kerr",
    "solver.gain_multilevel",
    "solver.anisotropy",
    "solver.gyrotropy",
    "solver.dynamic_material",
    "solver.material_grid",
    "solver.symmetry",
    "solver.dynamic_stop_harminv",
    "solver.cw",
    "monitor.dft_accumulation",
    "monitor.dft_reduction",
    "monitor.dft_materialization",
    "monitor.dft_checkpoint",
    "monitor.ldos",
    "monitor.eigenmode_overlap",
    "monitor.n2f_2d",
    "monitor.n2f_3d",
    "monitor.n2f_cyl",
    "monitor.cyl_to_cartesian",
    "monitor.n2f_adjoint",
    "monitor.adjoint_cartesian",
    "monitor.adjoint_cyl",
    "distributed.device_ownership",
    "distributed.pinned_mpi",
    "distributed.cuda_aware_mpi",
    "distributed.multi_rank",
    "distributed.multi_gpu_correctness",
    "distributed.strong_scaling",
    "policy.auto_small",
    "policy.auto_production",
    "policy.long_horizon_small",
    "evidence.resume_durability",
}
EXPECTED_FAMILY_COUNTS = {
    "solver_physics": 19,
    "monitor_transform_adjoint": 13,
    "distributed_policy": 10,
}
SPECIAL_POLICIES = {
    "distributed.cuda_aware_mpi": "capability_gated",
    "policy.long_horizon_small": "policy_gap_allowed",
}
FEATURE_ID = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+\Z")
ROUTE_ID = re.compile(r"[a-z][a-z0-9_.-]*:[A-Za-z0-9_.-]+\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")

ALLOWED_EVIDENCE = {
    "analytic_or_embedded_oracle",
    "auto_policy",
    "capability_probe",
    "device_uuid_binding",
    "direct_gpu2_comparison",
    "directional_derivative",
    "multi_rank_work",
    "observable_equivalence",
    "performance_repeats",
    "pinned_transport_counters",
    "resume_replay",
    "specialized_counter",
    "strict_cuda_no_fallback",
}
ALLOWED_TOPOLOGIES = {"cpu", "cpu8", "auto", "gpu1", "gpu2"}
ROUTE_KINDS = {"runner_case", "mpi_case", "specialized", "sealed_anchor"}
TOP_LEVEL_KEYS = {
    "schema",
    "description",
    "scheme_scope",
    "expected_feature_count",
    "maximum_route_count",
    "features",
    "routes",
}
FEATURE_KEYS = {
    "id",
    "family",
    "policy",
    "minimum_routes",
    "required_evidence",
}
COMMON_ROUTE_KEYS = {
    "id",
    "tier",
    "kind",
    "topologies",
    "evidence",
    "features",
}
ROUTE_KIND_KEYS = {
    "runner_case": {"case_id"},
    "mpi_case": {"case_id"},
    "specialized": {"driver", "target"},
    "sealed_anchor": {
        "terminal_relative_path",
        "source_commit",
        "source_snapshot_sha256",
        "build_receipt_id",
        "complete_sha256",
        "report_sha256",
        "audit_relative_path",
        "audit_sha256",
    },
}
GPU2_FEATURES = {
    "monitor.cyl_to_cartesian",
    "distributed.device_ownership",
    "distributed.pinned_mpi",
    "distributed.cuda_aware_mpi",
    "distributed.multi_rank",
    "distributed.multi_gpu_correctness",
    "distributed.strong_scaling",
}
AUTO_FEATURES = {
    "policy.auto_small",
    "policy.auto_production",
    "policy.long_horizon_small",
}
DERIVATIVE_FEATURES = {
    "solver.material_grid",
    "monitor.n2f_adjoint",
    "monitor.adjoint_cartesian",
    "monitor.adjoint_cyl",
}


class CoveragePlanError(RuntimeError):
    """Raised when the v1 coverage contract is incomplete or ambiguous."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CoveragePlanError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _stable_bytes(path: pathlib.Path) -> bytes:
    path = pathlib.Path(os.path.abspath(path))
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CoveragePlanError(f"input is not a regular non-symlink file: {path}")
    if before.st_size > MAX_INPUT_BYTES:
        raise CoveragePlanError(f"input exceeds {MAX_INPUT_BYTES} bytes: {path}")
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
        raise CoveragePlanError(f"input changed while read: {path}")
    return value


def _load_json(path: pathlib.Path) -> tuple[dict[str, Any], bytes]:
    payload = _stable_bytes(path)
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CoveragePlanError(f"non-finite JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CoveragePlanError(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise CoveragePlanError(f"JSON input must be an object: {path}")
    return value, payload


def _safe_repo_file(repo: pathlib.Path, relative: str) -> pathlib.Path:
    if not isinstance(relative, str):
        raise CoveragePlanError("repository path must be a string")
    pure = pathlib.PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in relative
        or pure.as_posix() != relative
    ):
        raise CoveragePlanError(f"noncanonical repository path: {relative!r}")
    path = repo.joinpath(*pure.parts)
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or path.resolve(strict=True) != path
    ):
        raise CoveragePlanError(f"path is not a regular repository file: {relative}")
    return path


def _load_runner_cases(
    repo: pathlib.Path, runner_manifest: pathlib.Path
) -> tuple[dict[str, dict[str, Any]], str]:
    runner_path = _safe_repo_file(
        repo, "scripts/python-validation/run_validation.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "gpmeep_v1_runner_contract", runner_path
    )
    if module_spec is None or module_spec.loader is None:
        raise CoveragePlanError("cannot load Python validation runner contract")
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    manifest, payload = _load_json(runner_manifest)
    try:
        cases = runner.materialize_cases(manifest, repo)
    except Exception as error:  # runner defines its own fail-closed exception
        raise CoveragePlanError(f"runner manifest is invalid: {error}") from error
    return {case["id"]: case for case in cases}, hashlib.sha256(payload).hexdigest()


def _string_set(value: Any, label: str, allowed: set[str]) -> set[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise CoveragePlanError(f"{label} must be a nonempty unique string array")
    result = set(value)
    unknown = result - allowed
    if unknown:
        raise CoveragePlanError(f"{label} has unknown values: {sorted(unknown)}")
    return result


def _validate_features(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_FEATURES):
        raise CoveragePlanError("feature array must contain exactly 42 entries")
    features: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(value):
        if not isinstance(feature, dict) or set(feature) != FEATURE_KEYS:
            raise CoveragePlanError(f"feature {index} has an invalid schema")
        feature_id = feature["id"]
        if (
            not isinstance(feature_id, str)
            or FEATURE_ID.fullmatch(feature_id) is None
            or feature_id in features
        ):
            raise CoveragePlanError(f"feature {index} has an invalid or duplicate ID")
        expected_family = (
            "solver_physics"
            if feature_id.startswith("solver.")
            else "monitor_transform_adjoint"
            if feature_id.startswith("monitor.")
            else "distributed_policy"
        )
        expected_policy = SPECIAL_POLICIES.get(feature_id, "ordinary")
        minimum = feature["minimum_routes"]
        if (
            feature["family"] != expected_family
            or feature["policy"] != expected_policy
            or not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or minimum <= 0
        ):
            raise CoveragePlanError(f"feature contract differs: {feature_id}")
        _string_set(
            feature["required_evidence"],
            f"{feature_id}.required_evidence",
            ALLOWED_EVIDENCE,
        )
        features[feature_id] = feature
    if set(features) != EXPECTED_FEATURES:
        missing = sorted(EXPECTED_FEATURES - set(features))
        extra = sorted(set(features) - EXPECTED_FEATURES)
        raise CoveragePlanError(f"feature inventory differs; missing={missing} extra={extra}")
    counts = Counter(feature["family"] for feature in features.values())
    if dict(counts) != EXPECTED_FAMILY_COUNTS:
        raise CoveragePlanError(f"feature family counts differ: {dict(counts)}")
    if _canonical_sha256(value) != EXPECTED_FEATURE_CONTRACT_SHA256:
        raise CoveragePlanError("frozen feature contract digest differs")
    return features


def _validate_runner_route(
    route: dict[str, Any], cases: dict[str, dict[str, Any]]
) -> None:
    case_id = route["case_id"]
    case = cases.get(case_id)
    if case is None:
        raise CoveragePlanError(f"route case is absent from runner inventory: {case_id}")
    comparison = case.get("comparison", {})
    if (
        case.get("disposition") != "run"
        or case.get("compute_scope") != "fdtd_cuda"
        or case.get("gpu_contract") != "cuda_dispatch"
        or comparison.get("mode") not in {"embedded_oracle", "json_metrics"}
    ):
        raise CoveragePlanError(
            f"route case is not a runnable numerical CUDA oracle: {case_id}"
        )
    if route["kind"] == "mpi_case" and "gpu2" not in route["topologies"]:
        raise CoveragePlanError(f"MPI route lacks gpu2 topology: {route['id']}")


def _validate_anchor(
    route: dict[str, Any], evidence_root: pathlib.Path | None
) -> dict[str, Any]:
    if COMMIT.fullmatch(route["source_commit"]) is None:
        raise CoveragePlanError(f"anchor source commit is invalid: {route['id']}")
    for name in (
        "source_snapshot_sha256",
        "build_receipt_id",
        "complete_sha256",
        "report_sha256",
        "audit_sha256",
    ):
        if SHA256.fullmatch(route[name]) is None:
            raise CoveragePlanError(f"anchor {name} is invalid: {route['id']}")
    relative = route["terminal_relative_path"]
    if not isinstance(relative, str):
        raise CoveragePlanError(f"anchor terminal path is invalid: {route['id']}")
    pure = pathlib.PurePosixPath(relative)
    if (
        pure.is_absolute()
        or len(pure.parts) != 1
        or pure.parts[0] in {"", ".", ".."}
        or pure.as_posix() != relative
    ):
        raise CoveragePlanError(f"anchor terminal path is invalid: {route['id']}")
    if evidence_root is None:
        return {"checked": False, "reason": "evidence root not requested"}
    terminal = evidence_root / relative
    if terminal.is_symlink() or not terminal.is_dir():
        raise CoveragePlanError(f"anchor terminal directory is absent: {terminal}")
    complete, complete_payload = _load_json(terminal / "COMPLETE")
    report, report_payload = _load_json(terminal / "report.json")
    complete_sha = hashlib.sha256(complete_payload).hexdigest()
    report_sha = hashlib.sha256(report_payload).hexdigest()
    if complete_sha != route["complete_sha256"]:
        raise CoveragePlanError(f"anchor COMPLETE SHA-256 differs: {route['id']}")
    if report_sha != route["report_sha256"]:
        raise CoveragePlanError(f"anchor report SHA-256 differs: {route['id']}")
    if (
        complete.get("outcome") != "PASS"
        or complete.get("report", {}).get("sha256") != report_sha
        or report.get("outcome") != "PASS"
        or report.get("source_snapshot_sha256")
        != route["source_snapshot_sha256"]
        or report.get("build", {}).get("source_sha256")
        != route["source_snapshot_sha256"]
        or report.get("build", {}).get("receipt_id")
        != route["build_receipt_id"]
    ):
        raise CoveragePlanError(f"anchor is not a sealed PASS: {route['id']}")
    audit_relative = route["audit_relative_path"]
    if not isinstance(audit_relative, str):
        raise CoveragePlanError(f"anchor audit path is invalid: {route['id']}")
    audit_pure = pathlib.PurePosixPath(audit_relative)
    if (
        audit_pure.is_absolute()
        or len(audit_pure.parts) != 2
        or audit_pure.parts[0] != "audits"
        or any(part in {"", ".", ".."} for part in audit_pure.parts)
        or audit_pure.as_posix() != audit_relative
    ):
        raise CoveragePlanError(f"anchor audit path is invalid: {route['id']}")
    audit_path = evidence_root.joinpath(*audit_pure.parts)
    audit_payload = _stable_bytes(audit_path)
    audit_sha = hashlib.sha256(audit_payload).hexdigest()
    if audit_sha != route["audit_sha256"]:
        raise CoveragePlanError(f"anchor audit SHA-256 differs: {route['id']}")
    return {
        "checked": True,
        "terminal": str(terminal.resolve(strict=True)),
        "complete_sha256": complete_sha,
        "report_sha256": report_sha,
        "source_snapshot_sha256": route["source_snapshot_sha256"],
        "build_receipt_id": route["build_receipt_id"],
        "audit": str(audit_path.resolve(strict=True)),
        "audit_sha256": audit_sha,
    }


def validate_plan(
    plan: dict[str, Any],
    *,
    repo: pathlib.Path,
    runner_manifest: pathlib.Path,
    evidence_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    repo = pathlib.Path(os.path.abspath(repo))
    if repo.is_symlink() or not repo.is_dir() or repo.resolve(strict=True) != repo:
        raise CoveragePlanError("repository root must be a canonical directory")
    if evidence_root is not None:
        evidence_root = pathlib.Path(os.path.abspath(evidence_root))
        if (
            evidence_root.is_symlink()
            or not evidence_root.is_dir()
            or evidence_root.resolve(strict=True) != evidence_root
        ):
            raise CoveragePlanError(
                "evidence root must be a canonical non-symlink directory"
            )
    if set(plan) != TOP_LEVEL_KEYS or plan.get("schema") != SCHEMA:
        raise CoveragePlanError("coverage manifest top-level schema differs")
    if (
        not isinstance(plan["description"], str)
        or not plan["description"].strip()
        or plan["scheme_scope"] != "excluded_by_user_requirement"
        or plan["expected_feature_count"] != len(EXPECTED_FEATURES)
    ):
        raise CoveragePlanError("coverage manifest metadata differs")
    if plan["maximum_route_count"] != EXPECTED_ROUTE_COUNT:
        raise CoveragePlanError("deduplication cap differs from the frozen contract")
    features = _validate_features(plan["features"])
    cases, runner_manifest_sha = _load_runner_cases(repo, runner_manifest)

    routes_value = plan["routes"]
    if (
        not isinstance(routes_value, list)
        or len(routes_value) != EXPECTED_ROUTE_COUNT
    ):
        raise CoveragePlanError("route count differs from the frozen deduplication cap")
    routes: dict[str, dict[str, Any]] = {}
    assignments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anchors: dict[str, Any] = {}
    for index, route in enumerate(routes_value):
        if not isinstance(route, dict) or route.get("kind") not in ROUTE_KINDS:
            raise CoveragePlanError(f"route {index} has an unknown schema or kind")
        expected_keys = COMMON_ROUTE_KEYS | ROUTE_KIND_KEYS[route["kind"]]
        if set(route) != expected_keys:
            raise CoveragePlanError(f"route {index} keys differ for {route['kind']}")
        route_id = route["id"]
        if (
            not isinstance(route_id, str)
            or ROUTE_ID.fullmatch(route_id) is None
            or route_id in routes
        ):
            raise CoveragePlanError(f"route {index} has an invalid or duplicate ID")
        tier = route["tier"]
        if not isinstance(tier, int) or isinstance(tier, bool) or tier not in range(4):
            raise CoveragePlanError(f"route tier is invalid: {route_id}")
        topologies = _string_set(
            route["topologies"], f"{route_id}.topologies", ALLOWED_TOPOLOGIES
        )
        evidence = _string_set(
            route["evidence"], f"{route_id}.evidence", ALLOWED_EVIDENCE
        )
        feature_ids = _string_set(
            route["features"], f"{route_id}.features", EXPECTED_FEATURES
        )
        if route["kind"] in {"runner_case", "mpi_case"}:
            _validate_runner_route(route, cases)
            if not {"cpu", "gpu1"}.issubset(topologies):
                raise CoveragePlanError(f"runner route lacks CPU/GPU1 pair: {route_id}")
            if not {"observable_equivalence", "strict_cuda_no_fallback"}.issubset(
                evidence
            ):
                raise CoveragePlanError(f"runner route lacks baseline evidence: {route_id}")
        elif route["kind"] == "specialized":
            _safe_repo_file(repo, route["driver"])
            _safe_repo_file(repo, route["target"])
            if "gpu2" not in topologies:
                raise CoveragePlanError(f"specialized route lacks gpu2: {route_id}")
        else:
            if tier != 0:
                raise CoveragePlanError(f"sealed anchor must be tier 0: {route_id}")
            anchors[route_id] = _validate_anchor(route, evidence_root)
        routes[route_id] = route
        for feature_id in feature_ids:
            assignments[feature_id].append(route)

    for feature_id, feature in features.items():
        assigned = assignments.get(feature_id, [])
        if len(assigned) < feature["minimum_routes"]:
            raise CoveragePlanError(
                f"feature {feature_id} has {len(assigned)} routes; "
                f"requires {feature['minimum_routes']}"
            )
        supplied = set().union(*(set(route["evidence"]) for route in assigned))
        missing_evidence = set(feature["required_evidence"]) - supplied
        if missing_evidence:
            raise CoveragePlanError(
                f"feature {feature_id} lacks evidence {sorted(missing_evidence)}"
            )
        topologies = set().union(*(set(route["topologies"]) for route in assigned))
        if feature_id in GPU2_FEATURES and "gpu2" not in topologies:
            raise CoveragePlanError(f"feature {feature_id} lacks a GPU2 route")
        if feature_id in AUTO_FEATURES and "auto" not in topologies:
            raise CoveragePlanError(f"feature {feature_id} lacks an auto-policy route")
        if feature_id in DERIVATIVE_FEATURES and any(
            "directional_derivative" not in route["evidence"] for route in assigned
        ):
            raise CoveragePlanError(
                f"feature {feature_id} has a route without directional derivative"
            )

    if _canonical_sha256(routes_value) != EXPECTED_ROUTE_CONTRACT_SHA256:
        raise CoveragePlanError("frozen route contract digest differs")

    route_counts = Counter(route["tier"] for route in routes.values())
    policy_counts = Counter(feature["policy"] for feature in features.values())
    return {
        "schema": SCHEMA,
        "outcome": "PASS",
        "feature_count": len(features),
        "family_counts": dict(sorted(EXPECTED_FAMILY_COUNTS.items())),
        "policy_counts": dict(sorted(policy_counts.items())),
        "route_count": len(routes),
        "route_counts_by_tier": {
            str(tier): route_counts.get(tier, 0) for tier in range(4)
        },
        "runner_case_count": sum(
            route["kind"] == "runner_case" for route in routes.values()
        ),
        "mpi_case_count": sum(
            route["kind"] == "mpi_case" for route in routes.values()
        ),
        "specialized_count": sum(
            route["kind"] == "specialized" for route in routes.values()
        ),
        "sealed_anchor_count": sum(
            route["kind"] == "sealed_anchor" for route in routes.values()
        ),
        "runner_manifest_sha256": runner_manifest_sha,
        "anchors": anchors,
        "uncovered_features": [],
        "scheme_scope": plan["scheme_scope"],
    }


def load_and_validate(
    manifest: pathlib.Path,
    *,
    repo: pathlib.Path,
    runner_manifest: pathlib.Path,
    evidence_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    plan, payload = _load_json(manifest)
    report = validate_plan(
        plan,
        repo=repo,
        runner_manifest=runner_manifest,
        evidence_root=evidence_root,
    )
    report["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=SCRIPT_DIR.parent.parent)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--runner-manifest", type=pathlib.Path, default=DEFAULT_RUNNER_MANIFEST
    )
    parser.add_argument("--evidence-root", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_and_validate(
        args.manifest,
        repo=args.repo,
        runner_manifest=args.runner_manifest,
        evidence_root=args.evidence_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CoveragePlanError, OSError, ValueError) as error:
        print(f"v1 feature coverage error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
