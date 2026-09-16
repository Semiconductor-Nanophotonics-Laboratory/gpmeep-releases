#!/usr/bin/env python3
"""Seal the user-revised AuNP and Meep-example release scope."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DEFAULT = SCRIPT_DIR.parents[1]
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_hybrid_aunp_matrix as aunp_matrix  # noqa: E402
import run_m3_host_matrix as host_matrix  # noqa: E402
import run_m3_paired_matrix as paired_matrix  # noqa: E402
import run_m3_specialized_matrix as specialized_matrix  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
import seal_m3_plan_replay as plan_replay_seal  # noqa: E402
import seal_milestone_adversarial_audit as audit_seal  # noqa: E402
from common import WorkloadError, atomic_write_json, atomic_write_text, file_record  # noqa: E402


SCHEMA = "gpmeep-current-scope-release-v1"
COMPLETE_SCHEMA = "gpmeep-current-scope-release-complete-v1"
TERMINAL_NAME = "M3_COMPLETE"
PENDING_TERMINAL_NAME = "PENDING_M3_COMPLETE"
HOR_DEFERRAL_SCHEMA = "gpmeep-user-deferred-workload-v1"
EXPECTED_COMPONENT_COUNTS = {"paired": 70, "host": 4, "specialized": 5}
TIMING_FIELDS = {"fdtd_wall_seconds", "workload_end_to_end_seconds"}
PERFORMANCE_COMPARISONS = {
    "cpu-fp32-to-one-gpu",
    "cpu-fp32-to-two-gpu",
    "one-gpu-to-two-gpu",
}
MATERIAL_SEVERITIES = {"critical", "high", "medium"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument("--plan-replay", required=True, type=pathlib.Path)
    parser.add_argument("--hor-deferral", required=True, type=pathlib.Path)
    parser.add_argument("--aunp-output", required=True, type=pathlib.Path)
    parser.add_argument("--aunp-audit", required=True, type=pathlib.Path)
    parser.add_argument("--paired-output", required=True, type=pathlib.Path)
    parser.add_argument("--host-output", required=True, type=pathlib.Path)
    parser.add_argument("--specialized-output", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def _canonical_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    return plan_replay_seal._canonical_directory(path, label)


def _stable_json(path: pathlib.Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record, payload = plan_replay_seal._stable_file_value(path, label)
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WorkloadError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return record, value


def _exact_record(value: Any, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "size_bytes", "sha256"}
        or not isinstance(value["path"], str)
    ):
        raise WorkloadError(f"{label} record differs")
    expected = plan_replay_seal._stable_file_record(pathlib.Path(value["path"]), label)
    if value != expected:
        raise WorkloadError(f"{label} changed after the Hor_TM deferral")
    return expected


def validate_hor_deferral(path: pathlib.Path) -> dict[str, Any]:
    receipt_record, receipt = _stable_json(path, "Hor_TM user deferral receipt")
    decision = receipt.get("decision")
    semantics = receipt.get("qualification_semantics")
    evidence = receipt.get("evidence")
    if (
        receipt.get("schema") != HOR_DEFERRAL_SCHEMA
        or not isinstance(decision, dict)
        or decision.get("status") != "USER_DEFERRED_LONG_RUNTIME"
        or not isinstance(semantics, dict)
        or semantics.get("correctness_outcome") != "NOT_EVALUATED"
        or semantics.get("performance_outcome") != "NOT_EVALUATED"
        or semantics.get("release_gate_eligible") is not False
        or semantics.get("completed_contract_tasks") != 0
        or not isinstance(evidence, dict)
        or set(evidence)
        != {
            "checkpoint",
            "controller_terminal",
            "gpu_telemetry",
            "host_telemetry",
            "journal",
            "progress_log",
        }
    ):
        raise WorkloadError("Hor_TM user deferral semantics differ")
    source = _exact_record(receipt.get("source"), "deferred Hor_TM source")
    evidence_records = {}
    for name, value in evidence.items():
        if not isinstance(value, dict):
            raise WorkloadError("Hor_TM deferred evidence schema differs")
        record = {key: value[key] for key in ("path", "size_bytes", "sha256") if key in value}
        evidence_records[name] = _exact_record(record, f"deferred Hor_TM {name}")
    return {
        "outcome": "USER_DEFERRED_LONG_RUNTIME",
        "correctness": "NOT_EVALUATED",
        "performance": "NOT_EVALUATED",
        "release_gate_eligible": False,
        "receipt": receipt_record,
        "source": source,
        "evidence": evidence_records,
    }


def _positive_samples(value: Any, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0.0
            for item in value
        )
    ):
        raise WorkloadError(f"AuNP {label} must contain exactly two positive samples")
    return [float(item) for item in value]


def validate_aunp_performance(report: dict[str, Any]) -> dict[str, Any]:
    performance = report.get("performance")
    diagnostics = report.get("timing_diagnostics")
    if (
        report.get("outcome") != "PASS"
        or report.get("performance_repeats") != 2
        or not isinstance(performance, dict)
        or set(performance) != TIMING_FIELDS
        or not isinstance(diagnostics, dict)
        or set(diagnostics) != TIMING_FIELDS
    ):
        raise WorkloadError("AuNP two-run performance inventory differs")
    for timing_name, comparisons in performance.items():
        if not isinstance(comparisons, dict) or set(comparisons) != PERFORMANCE_COMPARISONS:
            raise WorkloadError(f"AuNP {timing_name} comparison inventory differs")
        for comparison_name, metric in comparisons.items():
            expected_keys = {
                "reference_role",
                "candidate_role",
                "reference_samples_seconds",
                "candidate_samples_seconds",
                "reference_mean_seconds",
                "candidate_mean_seconds",
                "mean_speedup",
                "minimum",
                "outcome",
            }
            if not isinstance(metric, dict) or set(metric) != expected_keys:
                raise WorkloadError(f"AuNP {comparison_name} metric schema differs")
            reference = _positive_samples(
                metric["reference_samples_seconds"], f"{comparison_name} reference"
            )
            candidate = _positive_samples(
                metric["candidate_samples_seconds"], f"{comparison_name} candidate"
            )
            reference_mean = statistics.fmean(reference)
            candidate_mean = statistics.fmean(candidate)
            speedup = reference_mean / candidate_mean
            if (
                not math.isclose(metric["reference_mean_seconds"], reference_mean, rel_tol=1e-12)
                or not math.isclose(metric["candidate_mean_seconds"], candidate_mean, rel_tol=1e-12)
                or not math.isclose(metric["mean_speedup"], speedup, rel_tol=1e-12)
                or metric.get("outcome") != "PASS"
                or speedup < float(metric["minimum"])
                or "median_speedup" in metric
                or "conservative_speedup" in metric
            ):
                raise WorkloadError(f"AuNP {comparison_name} arithmetic-mean gate differs")
    for timing_name, roles in diagnostics.items():
        if not isinstance(roles, dict) or len(roles) != 3:
            raise WorkloadError(f"AuNP {timing_name} diagnostic inventory differs")
        for role, diagnostic in roles.items():
            if not isinstance(diagnostic, dict) or diagnostic.get("release_gate") is not None:
                raise WorkloadError(f"AuNP {role} timing dispersion became a release gate")
            samples = _positive_samples(diagnostic.get("samples_seconds"), f"{role} diagnostic")
            if diagnostic.get("sample_count") != 2 or not math.isclose(
                diagnostic.get("arithmetic_mean_seconds"), statistics.fmean(samples), rel_tol=1e-12
            ):
                raise WorkloadError(f"AuNP {role} timing diagnostic mean differs")
    return {
        "measured_runs_per_topology": 2,
        "aggregation": "arithmetic_mean",
        "dispersion_role": "diagnostic_only",
        "metrics": performance,
        "timing_diagnostics": diagnostics,
    }


def _task_ids(report: dict[str, Any], label: str, expected_count: int) -> set[str]:
    tasks = report.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != expected_count:
        raise WorkloadError(f"{label} task count differs")
    identifiers = set()
    for record in tasks:
        task = record.get("task") if isinstance(record, dict) else None
        unit_id = task.get("unit_id") if isinstance(task, dict) else None
        if not isinstance(unit_id, str) or not unit_id or unit_id in identifiers:
            raise WorkloadError(f"{label} task identity differs")
        identifiers.add(unit_id)
    return identifiers


def feature_closure(
    plan: dict[str, Any], component_ids: dict[str, set[str]]
) -> list[dict[str, Any]]:
    features = plan.get("features")
    if not isinstance(features, list) or len(features) != 42:
        raise WorkloadError("current-scope feature inventory differs")
    results = []
    for feature in features:
        routes = feature.get("primary_routes") if isinstance(feature, dict) else None
        if not isinstance(routes, list) or not routes:
            raise WorkloadError("current-scope feature route is absent")
        evidence_results = []
        for evidence in routes:
            candidates = evidence.get("routes") if isinstance(evidence, dict) else None
            if not isinstance(candidates, list) or not candidates:
                raise WorkloadError("current-scope evidence route differs")
            passed = []
            for route in candidates:
                kind = route.get("kind") if isinstance(route, dict) else None
                unit_id = route.get("unit_id") if isinstance(route, dict) else None
                if kind == "milestone":
                    ok = unit_id in {"milestone:M2-AuNP", "milestone:M3-example-matrix"}
                else:
                    ok = isinstance(unit_id, str) and unit_id in component_ids.get(kind, set())
                if ok:
                    passed.append({"kind": kind, "unit_id": unit_id})
            if not passed:
                raise WorkloadError(f"feature evidence did not pass: {evidence.get('evidence')}")
            evidence_results.append({"evidence": evidence["evidence"], "passing_routes": passed})
        results.append(
            {
                "feature_id": feature["feature_id"],
                "family": feature["family"],
                "planned_status": feature["status"],
                "outcome": "PASS",
                "evidence": evidence_results,
            }
        )
    if len({item["feature_id"] for item in results}) != 42:
        raise WorkloadError("current-scope feature IDs are not unique")
    return results


def _component_terminal(root: pathlib.Path, name: str) -> dict[str, Any]:
    terminals = {
        "aunp": "COMPLETE",
        "paired": "PAIRED_COMPLETE",
        "host": "HOST_COMPLETE",
        "specialized": "SPECIALIZED_COMPLETE",
    }
    return plan_replay_seal._stable_file_record(
        root / terminals[name], f"current-scope {name} terminal"
    )


def _build_hardware_closure(
    aunp: dict[str, Any], components: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    receipt_id = aunp.get("build", {}).get("receipt_id")
    source_sha256 = aunp.get("source_snapshot_sha256")
    devices = aunp.get("gpu_devices")
    if (
        not isinstance(receipt_id, str)
        or not isinstance(source_sha256, str)
        or not isinstance(devices, list)
        or len(devices) != 2
        or len(set(devices)) != 2
        or aunp.get("cpu_physical_cores") != 8
    ):
        raise WorkloadError("AuNP build or hardware identity differs")
    for name, report in components.items():
        contract = report.get("contract", {})
        build = contract.get("build_receipt", {})
        if build.get("receipt_id") != receipt_id or build.get("source_sha256") != source_sha256:
            raise WorkloadError(f"{name} build receipt differs from AuNP")
        if name == "specialized" and contract.get("gpu_devices") != devices:
            raise WorkloadError("specialized GPU identity differs from AuNP")
        if name in {"paired", "host"} and contract.get("gpu_device") not in devices:
            raise WorkloadError(f"{name} GPU identity differs from AuNP")
    return {
        "receipt_id": receipt_id,
        "source_sha256": source_sha256,
        "cpu_baseline": "8 physical MPI ranks",
        "gpu_devices": devices,
        "shared_identical_fp32_build": True,
    }


def _audit_summary(
    audit_path: pathlib.Path,
    aunp_root: pathlib.Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    audit = audit_seal.verify(audit_path)
    unresolved = audit.get("unresolved_counts", {})
    evidence = audit.get("evidence", {})
    if (
        audit.get("milestone") != "M2"
        or audit.get("source") != {
            "repo": source["repository"],
            "commit": source["git_commit"],
            "status_porcelain": "",
        }
        or evidence.get("root") != str(aunp_root)
        or evidence.get("terminal_name") != "COMPLETE"
        or any(unresolved.get(level) != 0 for level in MATERIAL_SEVERITIES)
    ):
        raise WorkloadError("AuNP adversarial audit binding differs")
    return {
        "outcome": "PASS",
        "receipt": plan_replay_seal._stable_file_record(audit_path, "AuNP audit receipt"),
        "auditor": audit["auditor"],
        "finding_counts": audit["finding_counts"],
        "unresolved_counts": audit["unresolved_counts"],
    }


def derive(contract: dict[str, str]) -> dict[str, Any]:
    expected_keys = {
        "repo",
        "plan_replay",
        "hor_deferral",
        "aunp_output",
        "aunp_audit",
        "paired_output",
        "host_output",
        "specialized_output",
    }
    if not isinstance(contract, dict) or set(contract) != expected_keys:
        raise WorkloadError("current-scope aggregation contract differs")
    repo = _canonical_directory(pathlib.Path(contract["repo"]), "current-scope repository")
    roots = {
        name: _canonical_directory(pathlib.Path(contract[f"{name}_output"]), f"{name} output")
        for name in ("aunp", "paired", "host", "specialized")
    }
    plan_replay = plan_replay_seal.verify(pathlib.Path(contract["plan_replay"]))
    if plan_replay.get("repository") != str(repo):
        raise WorkloadError("current-scope plan repository differs")
    source = {
        "repository": str(repo),
        "git_commit": plan_replay["git_commit"],
        "git_status_porcelain": plan_replay["git_status_porcelain"],
    }
    hor = validate_hor_deferral(pathlib.Path(contract["hor_deferral"]))
    aunp = aunp_matrix.verify_hybrid_complete(roots["aunp"])
    components = {
        "paired": paired_matrix.verify_complete(roots["paired"]),
        "host": host_matrix.verify_complete(roots["host"]),
        "specialized": specialized_matrix.verify_complete(roots["specialized"]),
    }
    if any(report.get("outcome") != "PASS" for report in components.values()):
        raise WorkloadError("official Meep component is not a PASS")
    component_ids = {
        name: _task_ids(report, name, EXPECTED_COMPONENT_COUNTS[name])
        for name, report in components.items()
    }
    features = feature_closure(plan_replay["plan"], component_ids)
    performance = validate_aunp_performance(aunp)
    audit = _audit_summary(
        pathlib.Path(contract["aunp_audit"]), roots["aunp"], source
    )
    terminals = {
        name: _component_terminal(root, name) for name, root in roots.items()
    }
    return {
        "schema": SCHEMA,
        "outcome": "PASS",
        "scope": {
            "Hor_TM": "USER_DEFERRED_LONG_RUNTIME",
            "user_workload": "AuNP",
            "performance_evidence": "AuNP only; exactly two measured runs per topology; arithmetic means",
            "official_examples": "deduplicated Python correctness and GPU topology; timings are diagnostic only",
            "scheme_examples": "excluded",
        },
        "contract": contract,
        "source": source,
        "plan_replay": plan_replay_seal._stable_file_record(
            pathlib.Path(contract["plan_replay"]), "current-scope plan replay"
        ),
        "hor_deferral": hor,
        "aunp": {
            "outcome": "PASS",
            "terminal": terminals["aunp"],
            "exact_comparison_count": len(aunp.get("exact_comparisons", [])),
            "performance": performance,
            "adversarial_audit": audit,
        },
        "official_meep_examples": {
            "outcome": "PASS",
            "performance_release_gate": False,
            "timings_are_diagnostic_only": True,
            "paired_cpu_cuda_units": EXPECTED_COMPONENT_COUNTS["paired"],
            "host_compatibility_units": EXPECTED_COMPONENT_COUNTS["host"],
            "specialized_correctness_units": EXPECTED_COMPONENT_COUNTS["specialized"],
            "total_physical_units": sum(EXPECTED_COMPONENT_COUNTS.values()),
            "feature_count": len(features),
            "features": features,
            "terminals": {name: terminals[name] for name in components},
        },
        "build_and_hardware": _build_hardware_closure(aunp, components),
        "next_action": "STOP_FOR_M4_COORDINATION_AFTER_M3_ADVERSARIAL_AUDIT",
    }


def markdown_report(report: dict[str, Any]) -> str:
    build = report["build_and_hardware"]
    examples = report["official_meep_examples"]
    lines = [
        "# gpmeep current-scope qualification",
        "",
        "- Outcome: **PASS**",
        "- Hor_TM: user-deferred, correctness/performance not evaluated",
        "- Performance release evidence: AuNP only, two measured runs per topology",
        f"- Build receipt: `{build['receipt_id']}`",
        f"- GPUs: `{', '.join(build['gpu_devices'])}`",
        "",
        "## AuNP arithmetic-mean performance",
        "",
        "| Timing | Comparison | Reference mean | Candidate mean | Speedup | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for timing_name, metrics in report["aunp"]["performance"]["metrics"].items():
        for comparison_name, metric in metrics.items():
            lines.append(
                f"| {timing_name} | {comparison_name} | "
                f"{metric['reference_mean_seconds']:.6f}s | "
                f"{metric['candidate_mean_seconds']:.6f}s | "
                f"{metric['mean_speedup']:.4f}x | {metric['minimum']:.4f}x |"
            )
    lines.extend(
        [
            "",
            "## Official Meep Python validation",
            "",
            f"- Paired CPU/CUDA units: {examples['paired_cpu_cuda_units']}",
            f"- Host compatibility units: {examples['host_compatibility_units']}",
            f"- Specialized correctness/topology units: {examples['specialized_correctness_units']}",
            f"- Covered GPU feature contracts: {examples['feature_count']}",
            "- Example timings are retained as diagnostics and are not performance claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _terminal_payload(output: pathlib.Path) -> dict[str, Any]:
    return {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": file_record(output / "report.json", output),
        "markdown": file_record(output / "report.md", output),
        "aggregator": plan_replay_seal._stable_file_record(
            pathlib.Path(__file__), "current-scope aggregator"
        ),
    }


def verify_complete(
    output: pathlib.Path, terminal_name: str = TERMINAL_NAME
) -> dict[str, Any]:
    output = _canonical_directory(output, "current-scope aggregation output")
    if terminal_name not in {TERMINAL_NAME, PENDING_TERMINAL_NAME}:
        raise WorkloadError("current-scope terminal filename differs")
    _terminal_record, terminal = _stable_json(
        output / terminal_name, "current-scope terminal"
    )
    expected_terminal_keys = {"schema", "outcome", "report", "markdown", "aggregator"}
    if (
        set(terminal) != expected_terminal_keys
        or terminal.get("schema") != COMPLETE_SCHEMA
        or terminal.get("outcome") != "PASS"
    ):
        raise WorkloadError("current-scope terminal is not an exact PASS")
    report = matrix.load_stable_json_record(
        output, terminal["report"], "current-scope report"
    )
    expected = derive(report.get("contract"))
    if report != expected:
        raise WorkloadError("current-scope report was not exactly re-derived")
    markdown_record = file_record(output / "report.md", output)
    if terminal["markdown"] != markdown_record or (output / "report.md").read_text(
        encoding="utf-8"
    ) != markdown_report(report):
        raise WorkloadError("current-scope Markdown report differs")
    if terminal != _terminal_payload(output):
        raise WorkloadError("current-scope terminal payload differs")
    return report


def _contract_from_args(args: argparse.Namespace) -> dict[str, str]:
    repo = _canonical_directory(args.repo, "current-scope repository")
    return {
        "repo": str(repo),
        "plan_replay": str(pathlib.Path(os.path.abspath(args.plan_replay))),
        "hor_deferral": str(pathlib.Path(os.path.abspath(args.hor_deferral))),
        "aunp_output": str(_canonical_directory(args.aunp_output, "AuNP output")),
        "aunp_audit": str(pathlib.Path(os.path.abspath(args.aunp_audit))),
        "paired_output": str(_canonical_directory(args.paired_output, "paired output")),
        "host_output": str(_canonical_directory(args.host_output, "host output")),
        "specialized_output": str(
            _canonical_directory(args.specialized_output, "specialized output")
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = _contract_from_args(args)
    output = pathlib.Path(os.path.abspath(args.output))
    repo = pathlib.Path(contract["repo"])
    input_roots = {
        pathlib.Path(contract[name])
        for name in ("aunp_output", "paired_output", "host_output", "specialized_output")
    }
    if output == repo or repo in output.parents or any(
        output == root or root in output.parents for root in input_roots
    ):
        raise WorkloadError("current-scope output overlaps an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    report = derive(contract)
    atomic_write_json(output / "report.json", report)
    atomic_write_text(output / "report.md", markdown_report(report))
    atomic_write_json(output / PENDING_TERMINAL_NAME, _terminal_payload(output))
    verify_complete(output, PENDING_TERMINAL_NAME)
    os.replace(output / PENDING_TERMINAL_NAME, output / TERMINAL_NAME)
    value = verify_complete(output)
    print(
        "GPMEEP_CURRENT_SCOPE_RELEASE="
        + json.dumps(
            {
                "schema": SCHEMA,
                "outcome": value["outcome"],
                "output": str(output),
                "terminal_sha256": file_record(output / TERMINAL_NAME, output)["sha256"],
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
    except (OSError, ValueError, WorkloadError, plan_replay_seal.SealError) as exc:
        print(f"current-scope release error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
