#!/usr/bin/env python3
"""Replay and seal the complete M1--M3 gpmeep release evidence closure."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
from typing import Any, Callable


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DEFAULT = SCRIPT_DIR.parents[1]
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import m3_feature_plan as feature_plan  # noqa: E402
import m3_source_equivalence as source_equivalence  # noqa: E402
import run_hybrid_aunp_matrix as aunp_matrix  # noqa: E402
import run_m3_host_matrix as host_matrix  # noqa: E402
import run_m3_paired_matrix as paired_matrix  # noqa: E402
import run_m3_repeated_performance_matrix as repeated_matrix  # noqa: E402
import run_m3_specialized_matrix as specialized_matrix  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
import seal_m3_plan_replay as plan_replay_seal  # noqa: E402
import seal_milestone_adversarial_audit as audit_seal  # noqa: E402
from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    atomic_write_text,
    file_record,
)


SCHEMA = "gpmeep-m3-release-aggregation-v3"
COMPLETE_SCHEMA = "gpmeep-m3-release-aggregation-complete-v3"
COMPONENT_TERMINALS = {
    "m1": "COMPLETE",
    "m2": "COMPLETE",
    "paired": "PAIRED_COMPLETE",
    "host": "HOST_COMPLETE",
    "specialized": "SPECIALIZED_COMPLETE",
    "performance": "PERFORMANCE_COMPLETE",
}
M3_COMPONENTS = ("paired", "host", "specialized", "performance")
EXPECTED_PERFORMANCE_UNITS = {
    "performance-m1-ters",
    "performance-m2-aunp",
    "performance-edge-emitter-3d",
    "performance-metasurface-crossover",
    "performance-near2far-transform",
    "performance-materialgrid-adjoint",
    "performance-long-horizon-policy",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument("--m1-replay-repo", required=True, type=pathlib.Path)
    parser.add_argument("--plan-root", required=True, type=pathlib.Path)
    parser.add_argument("--plan-replay", required=True, type=pathlib.Path)
    parser.add_argument("--m1-audit", required=True, type=pathlib.Path)
    parser.add_argument("--m2-audit", required=True, type=pathlib.Path)
    parser.add_argument("--m1-output", required=True, type=pathlib.Path)
    parser.add_argument("--m2-output", required=True, type=pathlib.Path)
    parser.add_argument("--paired-output", required=True, type=pathlib.Path)
    parser.add_argument("--host-output", required=True, type=pathlib.Path)
    parser.add_argument("--specialized-output", required=True, type=pathlib.Path)
    parser.add_argument("--performance-output", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def _canonical_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(path))
    try:
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise WorkloadError(f"M3 aggregate {label} is unavailable: {exc}") from exc
    if lexical.is_symlink() or not lexical.is_dir() or resolved != lexical:
        raise WorkloadError(f"M3 aggregate {label} is not a canonical directory")
    return lexical


def _canonical_file(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(path))
    try:
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise WorkloadError(f"M3 aggregate {label} is unavailable: {exc}") from exc
    if lexical.is_symlink() or not lexical.is_file() or resolved != lexical:
        raise WorkloadError(f"M3 aggregate {label} is not a canonical file")
    if metadata.st_size > 64 * 1024**2:
        raise WorkloadError(f"M3 aggregate {label} exceeds its size bound")
    return lexical


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    components = {
        name: _canonical_directory(
            pathlib.Path(getattr(args, f"{name}_output")), f"{name} output"
        )
        for name in COMPONENT_TERMINALS
    }
    return {
        "schema": SCHEMA,
        "repo": str(_canonical_directory(args.repo, "repository")),
        "m1_replay_repo": str(
            _canonical_directory(args.m1_replay_repo, "M1 replay repository")
        ),
        "plan_root": str(_canonical_directory(args.plan_root, "plan root")),
        "plan_replay": str(_canonical_file(args.plan_replay, "plan replay")),
        "audits": {
            name: str(
                _canonical_file(
                    pathlib.Path(getattr(args, f"{name}_audit")),
                    f"{name} adversarial audit",
                )
            )
            for name in ("m1", "m2")
        },
        "components": {name: str(path) for name, path in components.items()},
    }


def validate_contract(
    contract: Any,
) -> tuple[
    pathlib.Path,
    pathlib.Path,
    pathlib.Path,
    pathlib.Path,
    dict[str, pathlib.Path],
    dict[str, pathlib.Path],
]:
    if (
        not isinstance(contract, dict)
        or set(contract)
        != {
            "schema",
            "repo",
            "m1_replay_repo",
            "plan_root",
            "plan_replay",
            "audits",
            "components",
        }
        or contract.get("schema") != SCHEMA
        or not isinstance(contract.get("components"), dict)
        or set(contract["components"]) != set(COMPONENT_TERMINALS)
        or not isinstance(contract.get("audits"), dict)
        or set(contract["audits"]) != {"m1", "m2"}
    ):
        raise WorkloadError("M3 aggregate contract schema differs")
    repo = _canonical_directory(pathlib.Path(contract["repo"]), "repository")
    m1_replay_repo = _canonical_directory(
        pathlib.Path(contract["m1_replay_repo"]), "M1 replay repository"
    )
    plan_root = _canonical_directory(
        pathlib.Path(contract["plan_root"]), "plan root"
    )
    plan_replay = _canonical_file(
        pathlib.Path(contract["plan_replay"]), "plan replay"
    )
    audits = {
        name: _canonical_file(
            pathlib.Path(value), f"{name} adversarial audit"
        )
        for name, value in contract["audits"].items()
    }
    components = {
        name: _canonical_directory(pathlib.Path(value), f"{name} output")
        for name, value in contract["components"].items()
    }
    if (
        str(repo) != contract["repo"]
        or str(m1_replay_repo) != contract["m1_replay_repo"]
        or str(plan_root) != contract["plan_root"]
        or str(plan_replay) != contract["plan_replay"]
        or {name: str(path) for name, path in audits.items()}
        != contract["audits"]
        or {
        name: str(path) for name, path in components.items()
        }
        != contract["components"]
    ):
        raise WorkloadError("M3 aggregate contract paths are not canonical")
    return repo, m1_replay_repo, plan_root, plan_replay, audits, components


def _audit_reports(
    audits: dict[str, pathlib.Path],
    repo: pathlib.Path,
    m1_replay_repo: pathlib.Path,
    components: dict[str, pathlib.Path],
) -> dict[str, dict[str, Any]]:
    reports = {name: audit_seal.verify(path) for name, path in audits.items()}
    expected = {
        "m1": {
            "milestone": "M1",
            "repo": str(m1_replay_repo),
            "root": str(components["m1"]),
        },
        "m2": {
            "milestone": "M2",
            "repo": str(repo),
            "root": str(components["m2"]),
        },
    }
    for name, report in reports.items():
        wanted = expected[name]
        if (
            report.get("outcome") != "PASS"
            or report.get("milestone") != wanted["milestone"]
            or report.get("source", {}).get("repo") != wanted["repo"]
            or report.get("evidence", {}).get("root") != wanted["root"]
            or report.get("evidence", {}).get("terminal_name") != "COMPLETE"
            or any(
                report.get("unresolved_counts", {}).get(severity) != 0
                for severity in audit_seal.HIGH_SEVERITIES
            )
        ):
            raise WorkloadError(f"M3 aggregate {name} adversarial audit differs")
    return reports


def _external_m1_report(
    output: pathlib.Path, replay_repo: pathlib.Path
) -> dict[str, Any]:
    helper = (SCRIPT_DIR / "replay_external_m1.py").resolve(strict=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": "/dev/null",
        }
    )
    try:
        process = subprocess.run(
            [
                sys.executable,
                str(helper),
                "--repo",
                str(replay_repo),
                "--output",
                str(output),
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=43_200,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise WorkloadError(f"external M1 replay process failed: {exc}") from exc
    prefix = "GPMEEP_EXTERNAL_M1_REPLAY="
    records = [line for line in process.stdout.splitlines() if line.startswith(prefix)]
    if (
        process.returncode != 0
        or len(process.stdout.encode("utf-8")) > 16 * 1024**2
        or len(process.stderr.encode("utf-8")) > 16 * 1024**2
        or len(records) != 1
    ):
        raise WorkloadError(
            "external M1 replay did not publish one bounded PASS record: "
            + process.stderr[-4096:]
        )
    try:
        report = json.loads(records[0][len(prefix) :])
    except json.JSONDecodeError as exc:
        raise WorkloadError("external M1 replay record is invalid JSON") from exc
    expected_keys = {
        "schema",
        "outcome",
        "source_snapshot_sha256",
        "gpu_devices",
        "tasks",
        "task_count",
        "performance",
        "timing_stability",
        "exact_comparison",
        "shared_fp32_source_identity",
        "shared_fp32_receipt",
        "verifier",
    }
    if (
        not isinstance(report, dict)
        or set(report) != expected_keys
        or report.get("schema") != "gpmeep-external-m1-terminal-replay-v2"
        or report.get("outcome") != "PASS"
        or not isinstance(report.get("exact_comparison"), dict)
        or report["exact_comparison"].get("outcome") != "PASS"
        or report.get("task_count") != len(report.get("tasks", []))
    ):
        raise WorkloadError("external M1 replay record differs")
    return report


def _component_reports(
    paths: dict[str, pathlib.Path], m1_replay_repo: pathlib.Path
) -> dict[str, dict[str, Any]]:
    verifiers: dict[str, Callable[[pathlib.Path], dict[str, Any]]] = {
        "m2": aunp_matrix.verify_hybrid_complete,
        "paired": paired_matrix.verify_complete,
        "host": host_matrix.verify_complete,
        "specialized": specialized_matrix.verify_complete,
        "performance": repeated_matrix.verify_complete,
    }
    reports = {
        "m1": _external_m1_report(paths["m1"], m1_replay_repo),
        **{
            name: verifiers[name](paths[name])
            for name in COMPONENT_TERMINALS
            if name != "m1"
        },
    }
    if any(report.get("outcome") != "PASS" for report in reports.values()):
        raise WorkloadError("M3 aggregate component is not an exact PASS")
    return reports


def _task_index(report: dict[str, Any], label: str) -> dict[str, tuple[int, dict[str, Any]]]:
    result: dict[str, tuple[int, dict[str, Any]]] = {}
    tasks = report.get("tasks")
    if not isinstance(tasks, list):
        raise WorkloadError(f"M3 aggregate {label} task inventory differs")
    for index, record in enumerate(tasks):
        task = record.get("task") if isinstance(record, dict) else None
        unit_id = task.get("unit_id") if isinstance(task, dict) else None
        if not isinstance(unit_id, str) or unit_id in result:
            raise WorkloadError(f"M3 aggregate {label} task identity differs")
        result[unit_id] = (index, record)
    return result


def _receipt_and_hardware_closure(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    m1 = reports["m1"]
    m2 = reports["m2"]
    m3 = [reports[name] for name in M3_COMPONENTS]
    receipt_id = m2.get("build", {}).get("receipt_id")
    source_sha256 = m2.get("source_snapshot_sha256")
    if not all(
        isinstance(value, str) and len(value) == 64
        for value in (receipt_id, source_sha256)
    ):
        raise WorkloadError("M3 aggregate M2 receipt/source identity differs")
    m3_receipts = [report.get("contract", {}).get("build_receipt", {}) for report in m3]
    if any(
        record.get("receipt_id") != receipt_id
        or record.get("source_sha256") != source_sha256
        for record in m3_receipts
    ):
        raise WorkloadError("M3 aggregate build receipt closure differs")
    cuda_receipts = {
        record.get("task", {}).get("build_receipt_id")
        for record in m1.get("tasks", [])
        if record.get("task", {}).get("backend") == "cuda"
    }
    all_m1_receipts = {
        record.get("task", {}).get("build_receipt_id")
        for record in m1.get("tasks", [])
    }
    m1_receipt = m1.get("shared_fp32_receipt")
    if not isinstance(m1_receipt, dict) or not isinstance(
        m1_receipt.get("path"), str
    ):
        raise WorkloadError("M3 aggregate M1 receipt record differs")
    if m1_receipt != absolute_file_record(
        pathlib.Path(m1_receipt["path"]), "M3 aggregate M1 FP32 receipt"
    ):
        raise WorkloadError("M3 aggregate M1 receipt changed after replay")
    source_bridge = source_equivalence.compare(
        pathlib.Path(m1_receipt["path"]),
        pathlib.Path(m2["build"]["receipt_path"]),
    )
    m1_identity = source_bridge["first"]
    final_identity = source_bridge["second"]
    if (
        len(cuda_receipts) != 1
        or len(all_m1_receipts) != 2
        or not isinstance(m1_identity, dict)
        or m1_identity.get("receipt_id") not in cuda_receipts
        or m1_identity.get("full_source_sha256")
        != m1.get("source_snapshot_sha256")
        or final_identity.get("receipt_id") != receipt_id
        or final_identity.get("full_source_sha256") != source_sha256
        or m1.get("shared_fp32_source_identity") != m1_identity
        or m1_identity.get("schema") != source_equivalence.SCHEMA
        or final_identity.get("schema") != source_equivalence.SCHEMA
        or m1_identity.get("policy") != final_identity.get("policy")
        or m1_identity.get("retained_file_count")
        != final_identity.get("retained_file_count")
        or m1_identity.get("sha256") != final_identity.get("sha256")
    ):
        raise WorkloadError(
            "M3 aggregate M1/final non-validation source closure differs"
        )
    m1_fp32_receipt_id = next(iter(cuda_receipts))
    devices = m1.get("gpu_devices")
    if (
        not isinstance(devices, list)
        or len(devices) != 2
        or len(set(devices)) != 2
        or m2.get("gpu_devices") != devices
        or reports["paired"].get("contract", {}).get("gpu_device") != devices[0]
        or reports["host"].get("contract", {}).get("gpu_device") != devices[0]
        or reports["specialized"].get("contract", {}).get("gpu_devices") != devices
        or reports["performance"].get("contract", {}).get("gpu_devices") != devices
    ):
        raise WorkloadError("M3 aggregate physical GPU identity closure differs")
    return {
        "receipt_id": receipt_id,
        "source_sha256": source_sha256,
        "m1_fp32_receipt_id": m1_fp32_receipt_id,
        "m1_source_sha256": m1["source_snapshot_sha256"],
        "m1_cpu_fp64_receipt_id": next(
            value for value in all_m1_receipts if value != m1_fp32_receipt_id
        ),
        "non_validation_source_identity": {
            "schema": final_identity["schema"],
            "policy": final_identity["policy"],
            "retained_file_count": final_identity["retained_file_count"],
            "sha256": final_identity["sha256"],
        },
        "m1_to_final_source_bridge": source_bridge,
        "gpu_devices": devices,
    }


def _runtime_root_closure(
    repo: pathlib.Path,
    plan_root: pathlib.Path,
    reports: dict[str, dict[str, Any]],
) -> None:
    paired = reports["paired"]["contract"]
    host = reports["host"]["contract"]
    specialized = reports["specialized"]["contract"]
    performance = reports["performance"]["contract"]
    observed_repos = {
        paired["paths"]["repo"],
        host["repo"],
        specialized["paths"]["repo"],
        performance["paths"]["repo"],
    }
    observed_plans = {
        paired["paths"]["plan_root"],
        host["plan_root"],
        specialized["paths"]["plan_root"],
        performance["paths"]["plan_root"],
    }
    if observed_repos != {str(repo)} or observed_plans != {str(plan_root)}:
        raise WorkloadError("M3 aggregate runtime root closure differs")


def _source_records(repo: pathlib.Path, plan: dict[str, Any]) -> dict[str, Any]:
    paths = sorted(
        {
            evidence
            for feature in plan["features"]
            for field in (
                "primary_current_receipt_evidence",
                "independent_or_native_evidence",
            )
            for evidence in feature[field]
            if not evidence.startswith("milestone:")
        }
    )
    return {
        relative: matrix.stable_file_record(
            repo.joinpath(*pathlib.PurePosixPath(relative).parts),
            repo,
            f"M3 aggregate source evidence {relative}",
        )
        for relative in paths
    }


def _route_reference(
    route: dict[str, str],
    indexes: dict[str, dict[str, tuple[int, dict[str, Any]]]],
) -> dict[str, Any]:
    kind = route["kind"]
    unit_id = route["unit_id"]
    if kind == "milestone":
        if unit_id == "milestone:M1-TERS":
            return {"kind": kind, "unit_id": unit_id, "component": "m1", "outcome": "PASS"}
        if unit_id == "milestone:M2-AuNP":
            return {"kind": kind, "unit_id": unit_id, "component": "m2", "outcome": "PASS"}
        if unit_id == "milestone:M3-example-matrix":
            return {
                "kind": kind,
                "unit_id": unit_id,
                "components": list(M3_COMPONENTS),
                "outcome": "PASS",
            }
        raise WorkloadError("M3 aggregate milestone route differs")
    if kind == "performance":
        if unit_id not in indexes["performance"]:
            raise WorkloadError(f"M3 aggregate performance route is absent: {unit_id}")
        return {"kind": kind, "unit_id": unit_id, "component": "performance", "outcome": "PASS"}
    if kind not in {"paired", "host", "specialized"}:
        raise WorkloadError(f"M3 aggregate route kind differs: {kind}")
    try:
        index, _record = indexes[kind][unit_id]
    except KeyError as exc:
        raise WorkloadError(f"M3 aggregate routed unit is absent: {kind}:{unit_id}") from exc
    return {
        "kind": kind,
        "unit_id": unit_id,
        "component": kind,
        "task_index": index,
        "outcome": "PASS",
    }


def _feature_results(
    plan: dict[str, Any], reports: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexes = {
        kind: _task_index(reports[kind], kind)
        for kind in ("paired", "host", "specialized")
    }
    unit_results = reports["performance"].get("unit_results")
    if not isinstance(unit_results, dict):
        raise WorkloadError("M3 aggregate performance unit results differ")
    indexes["performance"] = {
        name: (index, record)
        for index, (name, record) in enumerate(sorted(unit_results.items()))
    }
    core = indexes["specialized"].get("specialized-mpi-python-core")
    if core is None:
        raise WorkloadError("M3 aggregate CUDA-aware capability evidence is absent")
    core_evidence = core[1].get("task_replay", {}).get("evidence", {})
    capability_executed = (
        core_evidence.get("driver") == "run-mpi-python-validation"
        and "cpu_vs_cuda_aware" in core_evidence.get("comparison_names", [])
    )
    if not capability_executed:
        raise WorkloadError("M3 aggregate CUDA-aware capability lane was not verified")
    long_horizon = unit_results.get("performance-long-horizon-policy", {})
    diagnostic = long_horizon.get("statistics", {}).get(
        "automatic_policy", {}
    ).get("forced-cuda-diagnostic", {})
    if type(diagnostic.get("missed_speedup_m4_gap")) is not bool:
        raise WorkloadError("M3 aggregate long-horizon policy diagnosis differs")

    results = []
    for feature in plan["features"]:
        routed = []
        for primary in feature["primary_routes"]:
            routed.append(
                {
                    "evidence": primary["evidence"],
                    "routes": [
                        _route_reference(route, indexes) for route in primary["routes"]
                    ],
                }
            )
        status = feature["status"]
        if status == "planned":
            outcome = "PASS"
        elif status == "planned_capability_gated":
            outcome = "PASS_CAPABILITY"
        elif status == "planned_policy_gap_allowed":
            outcome = (
                "PASS_POLICY_GAP_RETAINED"
                if diagnostic["missed_speedup_m4_gap"]
                else "PASS"
            )
        else:
            raise WorkloadError(f"M3 aggregate feature status differs: {status}")
        results.append(
            {
                "feature_id": feature["feature_id"],
                "family": feature["family"],
                "outcome": outcome,
                "contract": feature,
                "primary_execution_evidence": routed,
                "independent_source_evidence": feature[
                    "independent_or_native_evidence"
                ],
            }
        )
    outcomes = {value: sum(row["outcome"] == value for row in results) for value in {
        "PASS", "PASS_CAPABILITY", "PASS_POLICY_GAP_RETAINED"
    }}
    if len(results) != 42 or sum(outcomes.values()) != 42:
        raise WorkloadError("M3 aggregate feature outcome inventory differs")
    return results, {
        "outcome": "PASS",
        "feature_count": 42,
        "outcome_counts": outcomes,
        "capability": {
            "cuda_aware_mpi_executed": True,
            "status": "PASS_CAPABILITY",
        },
        "long_horizon_policy": {
            "missed_speedup_m4_gap": diagnostic["missed_speedup_m4_gap"],
            "status": next(
                row["outcome"]
                for row in results
                if row["feature_id"] == "policy.long_horizon_small"
            ),
        },
    }


def _specialized_evidence(
    report: dict[str, Any], unit_id: str
) -> dict[str, Any]:
    try:
        _index, record = _task_index(report, "specialized")[unit_id]
        evidence = record["task_replay"]["evidence"]
    except (KeyError, TypeError) as exc:
        raise WorkloadError(f"M3 aggregate specialized evidence is absent: {unit_id}") from exc
    if not isinstance(evidence, dict):
        raise WorkloadError(f"M3 aggregate specialized evidence differs: {unit_id}")
    return evidence


def _performance_summary(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    near2far = _specialized_evidence(
        reports["specialized"], "specialized-near2far-mpi-qualification"
    )
    adjoint = _specialized_evidence(
        reports["specialized"], "specialized-mpi-adjoint-qualification"
    )
    if (
        near2far.get("driver") != "run-near2far-mpi-qualification"
        or [row.get("name") for row in near2far.get("performance_scenarios", [])]
        != ["single-point", "batch"]
        or adjoint.get("driver") != "run-mpi-adjoint-benchmark"
    ):
        raise WorkloadError("M3 aggregate specialized performance evidence differs")
    repeated = reports["performance"].get("unit_results", {})
    repeated_ids = {
        "performance-edge-emitter-3d",
        "performance-metasurface-crossover",
        "performance-long-horizon-policy",
    }
    if set(repeated) != repeated_ids or any(
        value.get("statistics", {}).get("outcome") != "PASS"
        for value in repeated.values()
    ):
        raise WorkloadError("M3 aggregate repeated performance closure differs")
    result = {
        "performance-m1-ters": {
            "outcome": "PASS",
            "timing_contract": "controller-rederived-fixed-window-fdtd-wall",
            "performance": reports["m1"]["performance"],
            "timing_stability": reports["m1"]["timing_stability"],
        },
        "performance-m2-aunp": {
            "outcome": "PASS",
            "timing_contract": "controller-rederived-fixed-work-phase-wall",
            "performance": reports["m2"]["performance"],
            "timing_stability": reports["m2"]["timing_stability"],
        },
        "performance-near2far-transform": {
            "outcome": "PASS",
            "timing_contract": "rederived-native-batch-lane-median",
            "scenarios": near2far["performance_scenarios"],
        },
        "performance-materialgrid-adjoint": {
            "outcome": "PASS",
            "timing_contract": "rederived-forward-adjoint-trajectory-wall",
            "headline_metric": adjoint["headline_metric"],
            "raw_samples": adjoint["raw_samples"],
            "cpu_over_cuda_single_median_speedup": adjoint[
                "cpu_over_cuda_single_median_speedup"
            ],
            "cpu_over_cuda_multi_median_speedup": adjoint[
                "cpu_over_cuda_multi_median_speedup"
            ],
            "cuda_single_over_cuda_multi_median_speedup": adjoint[
                "cuda_single_over_cuda_multi_median_speedup"
            ],
        },
    }
    for unit_id in sorted(repeated):
        result[unit_id] = {
            "outcome": "PASS",
            "statistics": repeated[unit_id]["statistics"],
            "sample_count": repeated[unit_id]["sample_count"],
        }
    if set(result) != EXPECTED_PERFORMANCE_UNITS:
        raise WorkloadError("M3 aggregate seven-unit performance inventory differs")
    return result


def _component_summary(
    paths: dict[str, pathlib.Path], reports: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        name: {
            "outcome": "PASS",
            "output": str(paths[name]),
            "terminal": absolute_file_record(
                paths[name] / COMPONENT_TERMINALS[name],
                f"M3 aggregate {name} terminal",
            ),
            "task_count": len(reports[name].get("tasks", [])),
        }
        for name in COMPONENT_TERMINALS
    }


def _requirement_closure(
    reports: dict[str, dict[str, Any]],
    closure: dict[str, Any],
    correctness: dict[str, Any],
    feature_summary: dict[str, Any],
    performance: dict[str, Any],
) -> dict[str, Any]:
    if (
        feature_summary.get("feature_count") != 42
        or correctness.get("paired_cpu_cuda_unit_count") != 70
        or len(closure.get("gpu_devices", [])) != 2
        or set(performance) != EXPECTED_PERFORMANCE_UNITS
        or reports["m1"].get("outcome") != "PASS"
        or reports["m2"].get("outcome") != "PASS"
    ):
        raise WorkloadError("M3 aggregate high-level requirement closure differs")
    achieved = [
        {
            "requirement_id": "user.ters.correctness",
            "requirement": "Supplied TERS result agrees between CPU FP64 and CUDA FP32",
            "outcome": "PASS",
            "evidence": "correctness.m1_exact_comparison",
        },
        {
            "requirement_id": "user.ters.performance",
            "requirement": "Supplied TERS uses CPU8, GPU1, and GPU2 release speed gates",
            "outcome": "PASS",
            "evidence": "performance.performance-m1-ters",
        },
        {
            "requirement_id": "user.aunp.correctness",
            "requirement": (
                "Supplied AuNP GPU results agree with the packaged CPU-FP64 "
                "oracle and each other"
            ),
            "outcome": "PASS",
            "evidence": "correctness.m2_exact_comparisons",
        },
        {
            "requirement_id": "user.aunp.performance",
            "requirement": "Supplied AuNP uses current-hardware CPU8, GPU1, and GPU2 speed gates",
            "outcome": "PASS",
            "evidence": "performance.performance-m2-aunp",
        },
        {
            "requirement_id": "solver.fp32_cuda",
            "requirement": "Authoritative GPU release lane uses the qualified FP32 CUDA build",
            "outcome": "PASS",
            "evidence": "build_and_hardware_closure.receipt_id",
        },
        {
            "requirement_id": "distributed.multi_gpu",
            "requirement": (
                "One- and two-GPU correctness, performance, and specialized "
                "paths execute on two physical GPUs"
            ),
            "outcome": "PASS",
            "evidence": "build_and_hardware_closure.gpu_devices",
        },
        {
            "requirement_id": "examples.python_deduplicated",
            "requirement": (
                "Broad deduplicated Python examples and stronger tests pass "
                "CPU/CUDA comparison"
            ),
            "outcome": "PASS",
            "evidence": "correctness.paired_cpu_cuda_units",
        },
        {
            "requirement_id": "features.broad_surface",
            "requirement": (
                "All 42 planned solver, monitor, adjoint, distributed, and "
                "policy obligations resolve to current evidence"
            ),
            "outcome": "PASS",
            "evidence": "features",
        },
        {
            "requirement_id": "examples.adjoint",
            "requirement": (
                "Adjoint and inverse-design paths include numerical and repeated "
                "performance qualification"
            ),
            "outcome": "PASS",
            "evidence": "performance.performance-materialgrid-adjoint",
        },
        {
            "requirement_id": "build.isolated_python_no_scheme",
            "requirement": (
                "Dedicated receipt-bound CPU/CUDA MPI Python environments are "
                "isolated and Scheme is excluded"
            ),
            "outcome": "PASS",
            "evidence": "build_and_hardware_closure",
        },
        {
            "requirement_id": "evidence.durable",
            "requirement": (
                "Controller checkpoints, raw comparisons, timings, telemetry, "
                "and terminal hashes remain replayable"
            ),
            "outcome": "PASS",
            "evidence": "components",
        },
    ]
    deferred = [
        {
            "requirement_id": "m4.portable_gpu_distribution",
            "requirement": (
                "Build and install on additional GPU architectures as an "
                "easy-to-install package"
            ),
            "outcome": "DEFERRED_TO_M4_COORDINATION",
            "reason": (
                "M1--M3 qualify the current hardware; cross-machine packaging "
                "starts only after the requested M4 coordination stop."
            ),
        },
        {
            "requirement_id": "m4.private_github_release",
            "requirement": (
                "Publish the final distributable gpmeep repository to the "
                "user's private GitHub"
            ),
            "outcome": "DEFERRED_TO_M4_COORDINATION",
            "reason": (
                "The user required a detailed M1--M3 report and coordination "
                "before M4 publication work."
            ),
        },
    ]
    policy_gap = feature_summary["long_horizon_policy"]
    if policy_gap["missed_speedup_m4_gap"]:
        deferred.append(
            {
                "requirement_id": "m4.long_horizon_dispatch",
                "requirement": (
                    "Automatically select CUDA for every repeatably beneficial "
                    "long-horizon small-domain case"
                ),
                "outcome": "DEFERRED_TO_M4_COORDINATION",
                "reason": (
                    "The forced-CUDA diagnostic found a retained "
                    "automatic-dispatch optimization gap."
                ),
            }
        )
    return {
        "outcome": "PASS_WITH_EXPLICIT_M4_DEFERRED_SCOPE",
        "achieved_count": len(achieved),
        "deferred_count": len(deferred),
        "achieved": achieved,
        "deferred": deferred,
    }


def _m1_exact_comparison(report: dict[str, Any]) -> dict[str, Any]:
    comparison = report.get("exact_comparison")
    candidates = comparison.get("comparisons") if isinstance(comparison, dict) else None
    if (
        not isinstance(comparison, dict)
        or comparison.get("schema") != "gpmeep-ters-comparison-v1"
        or comparison.get("outcome") != "PASS"
        or not isinstance(candidates, list)
        or len(candidates) != 2
        or [item.get("label") for item in candidates]
        != ["cuda-fp32-1g", "cuda-fp32-2g"]
    ):
        raise WorkloadError("M3 aggregate M1 exact comparison inventory differs")
    tolerance_names = {
        "field_nrmse",
        "field_peak_linf",
        "field_energy_relative",
        "field_peak_relative",
        "phase_rms",
    }
    metric_names = {
        "nrmse",
        "peak_normalized_linf",
        "energy_relative_error",
        "peak_relative_error",
        "masked_phase_rms_rad",
    }
    for candidate in candidates:
        tolerance = candidate.get("tolerance")
        fields = candidate.get("fields")
        if (
            candidate.get("comparison_class") != "cuda-fp64-fp32"
            or candidate.get("outcome") != "PASS"
            or not isinstance(tolerance, dict)
            or not tolerance_names.issubset(tolerance)
            or not isinstance(fields, dict)
            or not fields
            or candidate.get("gap_enhancement", {}).get("outcome") != "PASS"
        ):
            raise WorkloadError("M3 aggregate M1 exact comparison content differs")
        for field in fields.values():
            if (
                not isinstance(field, dict)
                or field.get("outcome") != "PASS"
                or not metric_names.issubset(field)
                or any(
                    isinstance(field[name], bool)
                    or not isinstance(field[name], (int, float))
                    for name in metric_names
                )
            ):
                raise WorkloadError("M3 aggregate M1 exact field evidence differs")
    return comparison


def _m2_exact_comparisons(
    output: pathlib.Path, report: dict[str, Any]
) -> list[dict[str, Any]]:
    result = []
    for index, record in enumerate(report.get("exact_comparisons", [])):
        spec = record.get("spec") if isinstance(record, dict) else None
        report_record = record.get("report") if isinstance(record, dict) else None
        if not isinstance(spec, dict) or not isinstance(report_record, dict):
            raise WorkloadError("M3 aggregate M2 exact comparison record differs")
        retained = matrix.load_stable_json_record(
            output,
            report_record,
            f"M3 aggregate M2 exact comparison {index + 1}",
        )
        comparisons = retained.get("comparisons") if isinstance(retained, dict) else None
        if (
            retained.get("schema") != "gpmeep-aunp-r4000-comparison-v2"
            or retained.get("outcome") != "PASS"
            or not isinstance(comparisons, list)
            or len(comparisons) != 2
            or any(value.get("outcome") != "PASS" for value in comparisons)
        ):
            raise WorkloadError("M3 aggregate M2 exact comparison content differs")
        result.append({"spec": spec, "comparison": retained})
    if len(result) != 2:
        raise WorkloadError("M3 aggregate M2 exact comparison inventory differs")
    return result


def _paired_correctness_units(
    output: pathlib.Path, report: dict[str, Any]
) -> list[dict[str, Any]]:
    result = []
    for index, record in enumerate(report.get("tasks", [])):
        task = record.get("task") if isinstance(record, dict) else None
        archive = record.get("archive") if isinstance(record, dict) else None
        runner_record = archive.get("runner_report") if isinstance(archive, dict) else None
        if not isinstance(task, dict) or not isinstance(runner_record, dict):
            raise WorkloadError("M3 aggregate paired correctness record differs")
        runner = matrix.load_stable_json_record(
            output,
            runner_record,
            f"M3 aggregate paired runner report {index + 1}",
        )
        selected = [
            value
            for value in runner.get("results", [])
            if isinstance(value, dict) and value.get("selected") is True
        ]
        configuration = runner.get("configuration", {})
        performance_contract = configuration.get("performance_evidence", {})
        if (
            len(selected) != 1
            or selected[0].get("path") != task.get("case_path")
            or selected[0].get("outcome") != "PASS"
            or performance_contract.get("valid_for_speed_gate") is not False
        ):
            raise WorkloadError("M3 aggregate paired selected result differs")
        runs = selected[0].get("runs")
        comparison = selected[0].get("comparison")
        if (
            not isinstance(runs, dict)
            or set(runs) != {"cpu", "cuda"}
            or not isinstance(comparison, dict)
            or comparison.get("outcome") != "PASS"
        ):
            raise WorkloadError("M3 aggregate paired CPU/CUDA result differs")
        durations = {}
        for backend in ("cpu", "cuda"):
            duration = runs[backend].get("duration_seconds")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or duration <= 0
            ):
                raise WorkloadError("M3 aggregate paired duration differs")
            durations[backend] = float(duration)
        if (
            comparison.get("cpu_seconds") != runs["cpu"]["duration_seconds"]
            or comparison.get("cuda_seconds") != runs["cuda"]["duration_seconds"]
        ):
            raise WorkloadError("M3 aggregate paired comparison timing differs")
        result.append(
            {
                "unit_id": task.get("unit_id"),
                "case_path": task.get("case_path"),
                "release_role": task.get("release_role"),
                "comparison": comparison,
                "diagnostic_process_wall_seconds": durations,
                "timing_valid_for_speed_gate": False,
            }
        )
    return result


def _repeated_correctness(
    report: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    result = {}
    for unit_id, unit in sorted(report.get("unit_results", {}).items()):
        comparisons = unit.get("numerical_comparisons")
        if not isinstance(comparisons, dict) or any(
            not isinstance(value, dict)
            or value.get("outcome") != "PASS"
            or value.get("failure_count") != 0
            for value in comparisons.values()
        ):
            raise WorkloadError("M3 aggregate repeated correctness differs")
        result[unit_id] = comparisons
    return result


def _correctness_summary(
    paths: dict[str, pathlib.Path], reports: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    repeated_pairs = sum(
        len(value.get("numerical_comparisons", {}))
        for value in reports["performance"]["unit_results"].values()
    )
    m2_comparisons = _m2_exact_comparisons(paths["m2"], reports["m2"])
    paired_units = _paired_correctness_units(paths["paired"], reports["paired"])
    repeated_comparisons = _repeated_correctness(reports["performance"])
    return {
        "outcome": "PASS",
        "m1_exact_comparison_count": 1,
        "m1_exact_comparison": _m1_exact_comparison(reports["m1"]),
        "m2_exact_comparison_count": len(m2_comparisons),
        "m2_exact_comparisons": m2_comparisons,
        "paired_cpu_cuda_unit_count": len(paired_units),
        "paired_cpu_cuda_units": paired_units,
        "host_compatibility_unit_count": len(reports["host"]["tasks"]),
        "specialized_unit_count": len(reports["specialized"]["tasks"]),
        "repeated_cpu_gpu_numerical_comparison_count": repeated_pairs,
        "repeated_cpu_gpu_numerical_comparisons": repeated_comparisons,
    }


def derive_report(contract: dict[str, Any]) -> dict[str, Any]:
    (
        repo,
        m1_replay_repo,
        plan_root,
        plan_replay,
        audit_paths,
        paths,
    ) = validate_contract(contract)
    sealed_plan = plan_replay_seal.verify(plan_replay)
    if (
        sealed_plan.get("repository") != str(repo)
        or sealed_plan.get("plan_root") != str(plan_root)
    ):
        raise WorkloadError("M3 aggregate final plan-replay roots differ")
    audits = _audit_reports(
        audit_paths, repo, m1_replay_repo, paths
    )
    reports = _component_reports(paths, m1_replay_repo)
    _runtime_root_closure(repo, plan_root, reports)
    closure = _receipt_and_hardware_closure(reports)
    plan = feature_plan.load_release_plan(plan_root, repo)
    if sealed_plan.get("plan") != plan:
        raise WorkloadError("M3 aggregate final plan replay differs")
    features, feature_summary = _feature_results(plan, reports)
    performance = _performance_summary(reports)
    correctness = _correctness_summary(paths, reports)
    requirements = _requirement_closure(
        reports, closure, correctness, feature_summary, performance
    )
    evidence_code = {
        "aggregator": matrix.stable_file_record(
            pathlib.Path(__file__), repo, "M3 aggregate controller"
        ),
        "feature_plan": matrix.stable_file_record(
            SCRIPT_DIR / "m3_feature_plan.py", repo, "M3 feature plan loader"
        ),
        "external_m1_replay": matrix.stable_file_record(
            SCRIPT_DIR / "replay_external_m1.py", repo, "M3 external M1 replay"
        ),
        "source_equivalence": matrix.stable_file_record(
            SCRIPT_DIR / "m3_source_equivalence.py",
            repo,
            "M3 source equivalence verifier",
        ),
        "plan_replay_seal": matrix.stable_file_record(
            SCRIPT_DIR / "seal_m3_plan_replay.py",
            repo,
            "M3 final plan-replay sealer",
        ),
        "audit_seal": matrix.stable_file_record(
            SCRIPT_DIR / "seal_milestone_adversarial_audit.py",
            repo,
            "M3 milestone adversarial-audit sealer",
        ),
    }
    return {
        "schema": SCHEMA,
        "outcome": "PASS",
        "contract": contract,
        "build_and_hardware_closure": closure,
        "components": _component_summary(paths, reports),
        "adversarial_audits": audits,
        "requirement_closure": requirements,
        "feature_plan": plan,
        "feature_summary": feature_summary,
        "features": features,
        "source_evidence": _source_records(repo, plan),
        "performance": performance,
        "correctness": correctness,
        "evidence_code": evidence_code,
    }


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.4f}"


def _scientific(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:.6e}"


def _m1_error_rows(
    report: dict[str, Any],
) -> list[tuple[str, str, str, str, str, str, str]]:
    rows = []
    comparison = report["correctness"]["m1_exact_comparison"]
    for candidate in comparison["comparisons"]:
        tolerance = candidate["tolerance"]
        for field_name, field in sorted(candidate["fields"].items()):
            rows.append(
                (
                    candidate["label"],
                    field_name,
                    _scientific(field["nrmse"])
                    + " / "
                    + _scientific(tolerance["field_nrmse"]),
                    _scientific(field["peak_normalized_linf"])
                    + " / "
                    + _scientific(tolerance["field_peak_linf"]),
                    _scientific(field["energy_relative_error"])
                    + " / "
                    + _scientific(tolerance["field_energy_relative"]),
                    _scientific(field["masked_phase_rms_rad"])
                    + " / "
                    + _scientific(tolerance["phase_rms"]),
                    field["outcome"],
                )
            )
    return rows


def _paired_diagnostic_rows(
    report: dict[str, Any],
) -> list[tuple[str, str, str, str, str, str, str]]:
    rows = []
    for unit in report["correctness"]["paired_cpu_cuda_units"]:
        comparison = unit["comparison"]
        cpu_seconds = unit["diagnostic_process_wall_seconds"]["cpu"]
        cuda_seconds = unit["diagnostic_process_wall_seconds"]["cuda"]
        rows.append(
            (
                unit["unit_id"],
                unit["case_path"],
                str(comparison.get("mode")),
                _number(cpu_seconds),
                _number(cuda_seconds),
                _number(cpu_seconds / cuda_seconds) + "x",
                str(comparison["outcome"]),
            )
        )
    return rows


def _m2_error_rows(
    report: dict[str, Any],
) -> list[tuple[str, str, str, str, str, str]]:
    rows = []
    for direction in report["correctness"]["m2_exact_comparisons"]:
        direction_name = str(direction["spec"].get("name"))
        for comparison in direction["comparison"]["comparisons"]:
            maxima = {"max_abs": [], "rmse": [], "nrmse": []}
            polarizations = comparison.get("polarizations", {})
            if isinstance(polarizations, dict):
                for polarization in polarizations.values():
                    arrays = (
                        polarization.get("arrays", {})
                        if isinstance(polarization, dict)
                        else {}
                    )
                    for metrics in arrays.values():
                        if not isinstance(metrics, dict):
                            continue
                        for name in maxima:
                            value = metrics.get(name)
                            if (
                                not isinstance(value, bool)
                                and isinstance(value, (int, float))
                            ):
                                maxima[name].append(float(value))
            rows.append(
                (
                    direction_name,
                    str(comparison.get("label")),
                    _scientific(max(maxima["max_abs"], default=math.nan)),
                    _scientific(max(maxima["rmse"], default=math.nan)),
                    _scientific(max(maxima["nrmse"], default=math.nan)),
                    str(comparison["outcome"]),
                )
            )
    return rows


def _performance_rows(report: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for unit_id in ("performance-m1-ters", "performance-m2-aunp"):
        unit = report["performance"][unit_id]
        for name, metric in sorted(unit["performance"].items()):
            rows.append(
                (
                    unit_id,
                    name,
                    _number(metric.get("median_speedup")) + "x",
                    ">=" + _number(metric.get("minimum")) + "x",
                    str(metric.get("outcome")),
                )
            )
    near2far = report["performance"]["performance-near2far-transform"]
    for scenario in near2far["scenarios"]:
        for name, field, gate in (
            (
                "cpu8-to-cuda1",
                "cpu_to_one_gpu_speedup",
                scenario.get("required_one_gpu_speedup"),
            ),
            ("cpu8-to-cuda2", "cpu_to_two_gpu_speedup", None),
            ("cuda1-to-cuda2", "one_to_two_gpu_scaling", None),
        ):
            rows.append(
                (
                    "performance-near2far-transform",
                    f"{scenario['name']}:{name}",
                    _number(scenario.get(field)) + "x",
                    ">=" + _number(gate) + "x" if gate is not None else "verified",
                    "PASS",
                )
            )
    adjoint = report["performance"]["performance-materialgrid-adjoint"]
    for name, field in (
        ("cpu8-to-cuda1", "cpu_over_cuda_single_median_speedup"),
        ("cpu8-to-cuda2", "cpu_over_cuda_multi_median_speedup"),
        ("cuda1-to-cuda2", "cuda_single_over_cuda_multi_median_speedup"),
    ):
        rows.append(
            (
                "performance-materialgrid-adjoint",
                name,
                _number(adjoint.get(field)) + "x",
                "verified",
                "PASS",
            )
        )
    for unit_id in (
        "performance-edge-emitter-3d",
        "performance-metasurface-crossover",
        "performance-long-horizon-policy",
    ):
        statistics = report["performance"][unit_id]["statistics"]
        for name, metric in sorted(statistics.get("speedups", {}).items()):
            rows.append(
                (
                    unit_id,
                    name,
                    _number(metric.get("conservative_speedup")) + "x",
                    ">=" + _number(metric.get("minimum")) + "x",
                    str(metric.get("outcome")),
                )
            )
        for name, metric in sorted(statistics.get("automatic_policy", {}).items()):
            if "maximum_paired_slowdown" in metric:
                observed = _number(metric["maximum_paired_slowdown"]) + "x slowdown"
                gate = "<=" + _number(metric.get("maximum")) + "x"
                outcome = str(metric.get("outcome"))
            else:
                observed = _number(metric.get("conservative_speedup")) + "x"
                gate = "diagnostic"
                outcome = (
                    "M4_GAP"
                    if metric.get("missed_speedup_m4_gap") is True
                    else "PASS"
                )
            rows.append((unit_id, name, observed, gate, outcome))
    return rows


def _feature_route_ids(feature: dict[str, Any]) -> str:
    route_ids = sorted(
        {
            route["unit_id"]
            for primary in feature["primary_execution_evidence"]
            for route in primary["routes"]
        }
    )
    return "<br>".join(f"`{unit_id}`" for unit_id in route_ids)


def render_markdown(report: dict[str, Any]) -> str:
    closure = report["build_and_hardware_closure"]
    lines = [
        "# gpmeep M1--M3 release qualification",
        "",
        "Outcome: **PASS**",
        "",
        "## Build and hardware closure",
        "",
        f"- Final FP32 receipt: `{closure['receipt_id']}`",
        f"- Final source snapshot: `{closure['source_sha256']}`",
        f"- M1 FP32 receipt: `{closure['m1_fp32_receipt_id']}`",
        f"- M1 source snapshot: `{closure['m1_source_sha256']}`",
        "- Retained non-validation source files: "
        f"{closure['non_validation_source_identity']['retained_file_count']}",
        "- Retained source identity: "
        f"`{closure['non_validation_source_identity']['sha256']}`",
        "- Physical GPUs: " + ", ".join(f"`{item}`" for item in closure["gpu_devices"]),
        "",
        "## Requested-scope closure",
        "",
        "| Requirement ID | Outcome | Evidence/reason |",
        "| --- | --- | --- |",
    ]
    for item in report["requirement_closure"]["achieved"]:
        lines.append(
            f"| `{item['requirement_id']}` | {item['outcome']} | "
            f"`{item['evidence']}` |"
        )
    for item in report["requirement_closure"]["deferred"]:
        lines.append(
            f"| `{item['requirement_id']}` | {item['outcome']} | "
            f"{item['reason']} |"
        )
    lines.extend(
        (
            "",
            "## Component closure",
            "",
            "| Component | Tasks | Outcome |",
            "| --- | ---: | --- |",
        )
    )
    for name, component in report["components"].items():
        lines.append(
            f"| {name} | {component['task_count']} | {component['outcome']} |"
        )
    lines.extend(
        (
            "",
            "## Adversarial milestone audits",
            "",
            "| Milestone | Auditor task | Findings | Open material findings | Outcome |",
            "| --- | --- | ---: | ---: | --- |",
        )
    )
    for name, audit in sorted(report["adversarial_audits"].items()):
        finding_count = sum(audit["finding_counts"].values())
        material_open = sum(
            audit["unresolved_counts"][severity]
            for severity in sorted(audit_seal.HIGH_SEVERITIES)
        )
        lines.append(
            f"| {audit['milestone']} | `{audit['auditor']['task_name']}` | "
            f"{finding_count} | {material_open} | {audit['outcome']} |"
        )
    lines.extend(
        (
            "",
            "## Correctness coverage",
            "",
            "| Evidence class | Count |",
            "| --- | ---: |",
        )
    )
    for name, value in report["correctness"].items():
        if name.endswith("_count"):
            lines.append(f"| {name} | {value} |")
    lines.extend(
        (
            "",
            "### M1 TERS exact CPU FP64 vs CUDA FP32 error",
            "",
            "Each cell is `observed / gate`; raw re-derived values remain in report.json.",
            "",
            "| CUDA lane | Field | NRMSE | Peak L-inf | Energy relative | "
            "Phase RMS (rad) | Outcome |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        )
    )
    for candidate, field, nrmse, linf, energy, phase, outcome in _m1_error_rows(
        report
    ):
        lines.append(
            f"| {candidate} | {field} | {nrmse} | {linf} | {energy} | "
            f"{phase} | {outcome} |"
        )
    lines.extend(
        (
            "",
            "### M2 AuNP exact GPU/oracle and one-/two-GPU error",
            "",
            "The maxima summarize all retained spectral arrays; full per-array and "
            "resonance metrics are retained in report.json.",
            "",
            "| Direction | Candidate | Max absolute | Max RMSE | Max NRMSE | Outcome |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        )
    )
    for direction, candidate, maximum, rmse, nrmse, outcome in _m2_error_rows(
        report
    ):
        lines.append(
            f"| {direction} | {candidate} | {maximum} | {rmse} | {nrmse} | "
            f"{outcome} |"
        )
    lines.extend(
        (
            "",
            "### Deduplicated Python example CPU/CUDA correctness",
            "",
            "These one-shot whole-process times include startup and are diagnostic only; "
            "they are not performance claims.",
            "",
            "| Unit | Case | Comparison | CPU process (s) | CUDA process (s) | "
            "Diagnostic ratio | Outcome |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        )
    )
    for unit, case, mode, cpu, cuda, ratio, outcome in _paired_diagnostic_rows(
        report
    ):
        lines.append(
            f"| `{unit}` | `{case}` | {mode} | {cpu} | {cuda} | {ratio} | "
            f"{outcome} |"
        )
    lines.extend(
        (
            "",
            "## Repeated performance",
            "",
            "Only controller-approved phase timings are shown; cold process startup "
            "is not a speed claim.",
            "",
            "| Unit | Comparison/policy | Observed | Gate | Outcome |",
            "| --- | --- | ---: | ---: | --- |",
        )
    )
    for unit_id, name, observed, gate, outcome in _performance_rows(report):
        lines.append(f"| {unit_id} | {name} | {observed} | {gate} | {outcome} |")
    summary = report["feature_summary"]
    lines.extend(
        (
            "",
            "## Feature closure",
            "",
            f"All {summary['feature_count']} feature obligations resolve to current evidence.",
            "",
            "| Outcome class | Count |",
            "| --- | ---: |",
        )
    )
    for name, value in sorted(summary["outcome_counts"].items()):
        lines.append(f"| {name} | {value} |")
    lines.extend(
        (
            "",
            "### Feature-by-feature execution evidence",
            "",
            "| Feature ID | Family | Outcome | Executed evidence |",
            "| --- | --- | --- | --- |",
        )
    )
    for feature in report["features"]:
        lines.append(
            f"| `{feature['feature_id']}` | {feature['family']} | "
            f"{feature['outcome']} | {_feature_route_ids(feature)} |"
        )
    gap = summary["long_horizon_policy"]
    lines.extend(
        (
            "",
            "CUDA-aware MPI is capability-qualified rather than inferred from pinned staging.",
            "The long-horizon automatic-dispatch result is retained as "
            f"`{gap['status']}`; any missed speedup remains explicit M4 work.",
            "",
        )
    )
    return "\n".join(lines)


def _load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        with stable.file_object() as handle:
            try:
                value = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise WorkloadError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return value


def _terminal_payload(output: pathlib.Path) -> dict[str, Any]:
    return {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": file_record(output / "report.json", output),
        "markdown": file_record(output / "report.md", output),
    }


def verify_complete(
    output: pathlib.Path, terminal_name: str = "M3_COMPLETE"
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {"M3_COMPLETE", "PENDING_M3_COMPLETE"}:
        raise WorkloadError("M3 aggregate terminal filename differs")
    terminal = _load_json(output / terminal_name, "M3 aggregate terminal")
    if (
        set(terminal) != {"schema", "outcome", "report", "markdown"}
        or terminal.get("schema") != COMPLETE_SCHEMA
        or terminal.get("outcome") != "PASS"
    ):
        raise WorkloadError("M3 aggregate terminal is not an exact PASS")
    report = matrix.load_stable_json_record(
        output, terminal["report"], "M3 aggregate report"
    )
    markdown_record = terminal["markdown"]
    if not isinstance(markdown_record, dict):
        raise WorkloadError("M3 aggregate Markdown record differs")
    markdown_relative = matrix._safe_output_relative(
        markdown_record.get("path"), "M3 aggregate Markdown report"
    )
    with StableFile(
        output.joinpath(*markdown_relative.parts),
        "M3 aggregate Markdown report",
        expected=markdown_record,
    ) as stable:
        with stable.file_object() as handle:
            markdown = handle.read().decode("utf-8", errors="strict")
    expected_keys = {
        "schema",
        "outcome",
        "contract",
        "build_and_hardware_closure",
        "components",
        "adversarial_audits",
        "requirement_closure",
        "feature_plan",
        "feature_summary",
        "features",
        "source_evidence",
        "performance",
        "correctness",
        "evidence_code",
    }
    if (
        not isinstance(report, dict)
        or set(report) != expected_keys
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
        or report != derive_report(report.get("contract"))
        or markdown != render_markdown(report)
        or terminal != _terminal_payload(output)
    ):
        raise WorkloadError("M3 aggregate report was not fully re-derived")
    return report


def publish_terminal(output: pathlib.Path) -> None:
    atomic_write_json(output / "PENDING_M3_COMPLETE", _terminal_payload(output))
    verify_complete(output, "PENDING_M3_COMPLETE")
    os.replace(output / "PENDING_M3_COMPLETE", output / "M3_COMPLETE")


def validate_output(output: pathlib.Path, contract: dict[str, Any]) -> pathlib.Path:
    output = pathlib.Path(os.path.abspath(output))
    if output.exists() or output.is_symlink():
        raise WorkloadError("M3 aggregate output already exists")
    immutable = [
        pathlib.Path(contract["repo"]),
        pathlib.Path(contract["m1_replay_repo"]),
        pathlib.Path(contract["plan_root"]),
        *(pathlib.Path(value) for value in contract["components"].values()),
    ]
    if any(
        output == path or output in path.parents or path in output.parents
        for path in immutable
    ):
        raise WorkloadError("M3 aggregate output overlaps an immutable evidence root")
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = build_contract(args)
    output = validate_output(args.output, contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    try:
        report = derive_report(contract)
        atomic_write_json(output / "report.json", report)
        atomic_write_text(output / "report.md", render_markdown(report))
        publish_terminal(output)
        return 0
    except Exception as exc:
        atomic_write_json(
            output / "FAILED.json",
            {
                "schema": SCHEMA,
                "outcome": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, WorkloadError, feature_plan.PlanError, ValueError) as error:
        print(f"M3 release aggregation error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
