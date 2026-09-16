#!/usr/bin/env python3
"""Combine same-binary legacy/analytic MaterialGrid benchmark evidence."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import platform
import statistics
import sys
import traceback
import uuid
from typing import Any

from gpmeep_benchmark_evidence import (
    atomic_write_json,
    atomic_write_text,
    compare_gradient_vectors,
    sha256_file,
)
from gpmeep_provenance import git_output, source_snapshot


PROVENANCE_FIELDS = (
    "git_head",
    "git_status_porcelain",
    "source_snapshot",
    "relevant_source_sha256",
    "build_receipt",
    "runtime_artifacts",
    "meep_module",
    "python",
    "platform",
    "cuda_toolchain",
    "nvidia_driver",
)
POSTPROCESSOR_SOURCES = (
    "scripts/compare-material-gradient-benchmarks.py",
    "scripts/gpmeep_benchmark_evidence.py",
    "scripts/gpmeep_provenance.py",
)
FINAL_ARTIFACTS = (
    "report.json",
    "report.md",
    "COMPLETE",
    "FAILED.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-report", type=pathlib.Path, required=True)
    parser.add_argument("--analytic-report", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--gradient-atol", type=float, default=2e-7)
    parser.add_argument("--gradient-rtol", type=float, default=5e-5)
    parser.add_argument("--minimum-cuda-evaluation-improvement", type=float, default=1.10)
    parser.add_argument("--minimum-cuda-wall-improvement", type=float, default=1.10)
    parser.add_argument("--minimum-gradient-improvement", type=float, default=50.0)
    parser.add_argument("--minimum-optimized-cuda-speedup", type=float, default=2.85)
    parser.add_argument("--maximum-forward-adjoint-regression", type=float, default=1.02)
    return parser.parse_args(argv)


def load_report(path: pathlib.Path, expected_path: str) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {path}: {error}") from error
    if report.get("schema_version") != 3 or report.get("state") != "COMPLETE":
        raise RuntimeError(f"{path} is not a complete benchmark report")
    marker_path = path.parent / "COMPLETE"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {marker_path}: {error}") from error
    if (
        marker.get("state") != "COMPLETE"
        or pathlib.Path(str(marker.get("report", ""))).resolve() != path
        or marker.get("report_sha256") != sha256_file(path)
        or marker.get("run_id") != report.get("run_id")
    ):
        raise RuntimeError(f"{marker_path} does not authenticate {path}")
    if not report.get("gate", {}).get("pass"):
        raise RuntimeError(f"{path} did not pass its standalone gates")
    actual_path = report.get("workload", {}).get("material_gradient_path")
    if actual_path != expected_path:
        raise RuntimeError(
            f"{path} expected material path {expected_path!r}, got {actual_path!r}"
        )
    return report


def measured_runs(report: dict[str, Any], backend: str) -> list[dict[str, Any]]:
    runs = sorted(
        (
            run
            for run in report["runs"]
            if run["backend"] == backend and not run["warmup"]
        ),
        key=lambda run: run["iteration"],
    )
    if not runs:
        raise RuntimeError(f"report contains no measured {backend} runs")
    expected_repeats = report.get("workload", {}).get("repeats")
    if (
        isinstance(expected_repeats, bool)
        or not isinstance(expected_repeats, int)
        or expected_repeats < 1
        or len(runs) != expected_repeats
    ):
        raise RuntimeError(
            f"report {backend} measured-run count does not match workload repeats"
        )
    expected_iterations = list(range(expected_repeats))
    if [run["iteration"] for run in runs] != expected_iterations:
        raise RuntimeError(f"report has incomplete {backend} iterations")
    return runs


def validate_run_matrix(report: dict[str, Any]) -> None:
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise RuntimeError("report run matrix is absent")
    if any(run.get("backend") not in ("cpu", "cuda") for run in runs):
        raise RuntimeError("report contains an unknown backend")
    expected_warmups = report.get("workload", {}).get("warmups")
    if (
        isinstance(expected_warmups, bool)
        or not isinstance(expected_warmups, int)
        or expected_warmups < 0
    ):
        raise RuntimeError("report workload warmup count is invalid")
    for backend in ("cpu", "cuda"):
        measured_runs(report, backend)
        warmups = sorted(
            (
                run
                for run in runs
                if run["backend"] == backend and run["warmup"] is True
            ),
            key=lambda run: run["iteration"],
        )
        if len(warmups) != expected_warmups or [
            run["iteration"] for run in warmups
        ] != list(range(expected_warmups)):
            raise RuntimeError(f"report has incomplete {backend} warmups")
    expected_total = 2 * (
        int(report["workload"]["repeats"]) + expected_warmups
    )
    if len(runs) != expected_total:
        raise RuntimeError("report contains duplicate or surplus runs")


def phase_median(runs: list[dict[str, Any]], field: str) -> float:
    return statistics.median(
        float(run["result"]["timing"][field]) for run in runs
    )


def wall_median(runs: list[dict[str, Any]]) -> float:
    return statistics.median(float(run["process_seconds"]) for run in runs)


def phase_values(runs: list[dict[str, Any]], field: str) -> list[float]:
    values = [float(run["result"]["timing"][field]) for run in runs]
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise RuntimeError(f"run timing {field!r} must be finite and positive")
    return values


def wall_values(runs: list[dict[str, Any]]) -> list[float]:
    values = [float(run["process_seconds"]) for run in runs]
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise RuntimeError("process timings must be finite and positive")
    return values


def conservative_improvement(
    reference_values: list[float], optimized_values: list[float]
) -> float:
    if not reference_values or not optimized_values:
        raise RuntimeError("conservative comparison requires nonempty samples")
    if not all(
        math.isfinite(value) and value > 0
        for value in reference_values + optimized_values
    ):
        raise RuntimeError("conservative comparison timings must be positive and finite")
    return min(reference_values) / max(optimized_values)


def validate_standalone_timing(report: dict[str, Any]) -> None:
    cpu = measured_runs(report, "cpu")
    cuda = measured_runs(report, "cuda")
    cpu_values = phase_values(cpu, "elapsed_seconds")
    cuda_values = phase_values(cuda, "elapsed_seconds")
    expected = {
        "cpu_seconds": cpu_values,
        "cuda_seconds": cuda_values,
        "cpu_median_seconds": statistics.median(cpu_values),
        "cuda_median_seconds": statistics.median(cuda_values),
        "speedup_cpu_over_cuda": statistics.median(cpu_values)
        / statistics.median(cuda_values),
    }
    actual = report.get("timing", {})
    for name, value in expected.items():
        if actual.get(name) != value:
            raise RuntimeError(f"standalone timing field {name!r} is stale or invalid")


def prepare_output(output: pathlib.Path, run_id: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in FINAL_ARTIFACTS:
        (output / name).unlink(missing_ok=True)
    atomic_write_json(
        output / "state.json",
        {"schema_version": 1, "state": "RUNNING", "run_id": run_id},
    )


def mark_failed(output: pathlib.Path, run_id: str, error: BaseException) -> None:
    for name in ("report.json", "report.md", "COMPLETE"):
        (output / name).unlink(missing_ok=True)
    failure = {
        "schema_version": 1,
        "state": "FAILED",
        "run_id": run_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }
    atomic_write_json(output / "FAILED.json", failure)
    atomic_write_json(output / "state.json", failure)


def collect_postprocessor_provenance(
    repo: pathlib.Path, snapshot: dict[str, Any]
) -> dict[str, Any]:
    return {
        "git_head": git_output(repo, "rev-parse", "HEAD").strip(),
        "git_status_porcelain": git_output(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines(),
        "source_snapshot": snapshot,
        "source_sha256": {
            relative: sha256_file(repo / relative)
            for relative in POSTPROCESSOR_SOURCES
        },
        "python": {
            "executable": str(pathlib.Path(sys.executable).resolve()),
            "sha256": sha256_file(pathlib.Path(sys.executable).resolve()),
            "version": sys.version,
        },
        "platform": platform.platform(),
        "command": [
            str(pathlib.Path(sys.executable).resolve()),
            str(pathlib.Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    }


def publish_complete(
    output: pathlib.Path,
    run_id: str,
    report: dict[str, Any],
    markdown: str,
) -> None:
    if report.get("state") != "COMPLETE" or report.get("gate", {}).get("pass") is not True:
        raise RuntimeError("refusing to publish an unqualified comparison as COMPLETE")
    report_path = output / "report.json"
    markdown_path = output / "report.md"
    atomic_write_text(markdown_path, markdown)
    report["report_markdown"] = {
        "path": str(markdown_path.resolve()),
        "sha256": sha256_file(markdown_path),
    }
    atomic_write_json(report_path, report)
    complete = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "report_markdown": str(markdown_path.resolve()),
        "report_markdown_sha256": sha256_file(markdown_path),
    }
    atomic_write_json(output / "COMPLETE", complete)
    atomic_write_json(output / "state.json", complete)


def execute(
    args: argparse.Namespace, repo: pathlib.Path, run_id: str
) -> int:
    if min(
        args.minimum_cuda_evaluation_improvement,
        args.minimum_cuda_wall_improvement,
        args.minimum_gradient_improvement,
        args.minimum_optimized_cuda_speedup,
    ) <= 1:
        raise ValueError("all speedup/improvement thresholds must exceed one")
    if args.maximum_forward_adjoint_regression < 1:
        raise ValueError("maximum forward/adjoint regression must be at least one")
    if args.gradient_atol < 0 or args.gradient_rtol < 0:
        raise ValueError("gradient tolerances must be nonnegative")

    snapshot_before = source_snapshot(repo)
    input_hashes = {
        "legacy": sha256_file(args.legacy_report.resolve()),
        "analytic": sha256_file(args.analytic_report.resolve()),
    }
    legacy = load_report(args.legacy_report.resolve(), "legacy")
    analytic = load_report(args.analytic_report.resolve(), "analytic-required")

    for report in (legacy, analytic):
        validate_run_matrix(report)
        validate_standalone_timing(report)
    for field in PROVENANCE_FIELDS:
        if legacy["provenance"].get(field) != analytic["provenance"].get(field):
            raise RuntimeError(f"legacy/analytic provenance differs for {field}")
    if analytic["provenance"].get("source_snapshot") != snapshot_before:
        raise RuntimeError(
            "comparison source snapshot differs from the benchmark/build snapshot"
        )

    legacy_workload = dict(legacy["workload"])
    analytic_workload = dict(analytic["workload"])
    legacy_workload.pop("material_gradient_path", None)
    analytic_workload.pop("material_gradient_path", None)
    if legacy_workload != analytic_workload:
        raise RuntimeError("legacy/analytic workload configuration differs")

    expected_count = int(legacy["workload"]["design_variables"])
    comparisons: list[dict[str, Any]] = []
    objective_comparisons: list[dict[str, Any]] = []
    backend_results: dict[str, Any] = {}
    all_gradient_pass = True
    all_forward_adjoint_pass = True
    for backend in ("cpu", "cuda"):
        legacy_runs = measured_runs(legacy, backend)
        analytic_runs = measured_runs(analytic, backend)
        if len(legacy_runs) != len(analytic_runs):
            raise RuntimeError(f"legacy/analytic {backend} repeat counts differ")

        for legacy_run, analytic_run in zip(legacy_runs, analytic_runs):
            comparison = compare_gradient_vectors(
                legacy_run["result"],
                analytic_run["result"],
                atol=args.gradient_atol,
                rtol=args.gradient_rtol,
                expected_count=expected_count,
            )
            comparison.update(
                {"backend": backend, "iteration": legacy_run["iteration"]}
            )
            comparisons.append(comparison)
            all_gradient_pass = all_gradient_pass and comparison["pass"]
            legacy_objective = float(
                legacy_run["result"]["result"]["objective"]
            )
            analytic_objective = float(
                analytic_run["result"]["result"]["objective"]
            )
            objective_error = abs(analytic_objective - legacy_objective)
            objective_allowed = args.gradient_atol + args.gradient_rtol * max(
                abs(legacy_objective), abs(analytic_objective)
            )
            objective_comparisons.append(
                {
                    "backend": backend,
                    "iteration": legacy_run["iteration"],
                    "legacy": legacy_objective,
                    "analytic": analytic_objective,
                    "absolute_error": objective_error,
                    "allowed_error": objective_allowed,
                    "pass": objective_error <= objective_allowed,
                }
            )

        legacy_evaluation = phase_median(legacy_runs, "elapsed_seconds")
        analytic_evaluation = phase_median(analytic_runs, "elapsed_seconds")
        legacy_gradient = phase_median(legacy_runs, "gradient_seconds")
        analytic_gradient = phase_median(analytic_runs, "gradient_seconds")
        legacy_evaluation_values = phase_values(legacy_runs, "elapsed_seconds")
        analytic_evaluation_values = phase_values(analytic_runs, "elapsed_seconds")
        legacy_gradient_values = phase_values(legacy_runs, "gradient_seconds")
        analytic_gradient_values = phase_values(analytic_runs, "gradient_seconds")
        legacy_wall_values = wall_values(legacy_runs)
        analytic_wall_values = wall_values(analytic_runs)
        legacy_forward_adjoint = phase_median(
            legacy_runs, "forward_seconds"
        ) + phase_median(legacy_runs, "adjoint_seconds")
        analytic_forward_adjoint = phase_median(
            analytic_runs, "forward_seconds"
        ) + phase_median(analytic_runs, "adjoint_seconds")
        forward_adjoint_ratio = analytic_forward_adjoint / legacy_forward_adjoint
        forward_adjoint_pass = (
            forward_adjoint_ratio <= args.maximum_forward_adjoint_regression
        )
        all_forward_adjoint_pass = (
            all_forward_adjoint_pass and forward_adjoint_pass
        )
        backend_results[backend] = {
            "legacy_evaluation_median_seconds": legacy_evaluation,
            "analytic_evaluation_median_seconds": analytic_evaluation,
            "evaluation_improvement": legacy_evaluation / analytic_evaluation,
            "conservative_evaluation_improvement": conservative_improvement(
                legacy_evaluation_values, analytic_evaluation_values
            ),
            "legacy_gradient_median_seconds": legacy_gradient,
            "analytic_gradient_median_seconds": analytic_gradient,
            "gradient_improvement": legacy_gradient / analytic_gradient,
            "conservative_gradient_improvement": conservative_improvement(
                legacy_gradient_values, analytic_gradient_values
            ),
            "legacy_wall_median_seconds": wall_median(legacy_runs),
            "analytic_wall_median_seconds": wall_median(analytic_runs),
            "wall_improvement": wall_median(legacy_runs) / wall_median(analytic_runs),
            "conservative_wall_improvement": conservative_improvement(
                legacy_wall_values, analytic_wall_values
            ),
            "legacy_forward_adjoint_median_seconds": legacy_forward_adjoint,
            "analytic_forward_adjoint_median_seconds": analytic_forward_adjoint,
            "forward_adjoint_ratio": forward_adjoint_ratio,
            "forward_adjoint_pass": forward_adjoint_pass,
        }

    analytic_cpu_runs = measured_runs(analytic, "cpu")
    analytic_cuda_runs = measured_runs(analytic, "cuda")
    optimized_cuda_speedup = (
        phase_median(analytic_cpu_runs, "elapsed_seconds")
        / phase_median(analytic_cuda_runs, "elapsed_seconds")
    )
    conservative_optimized_cuda_speedup = conservative_improvement(
        phase_values(analytic_cpu_runs, "elapsed_seconds"),
        phase_values(analytic_cuda_runs, "elapsed_seconds"),
    )
    gates = {
        "cross_path_gradient_pass": all_gradient_pass,
        "cross_path_objective_pass": all(
            comparison["pass"] for comparison in objective_comparisons
        ),
        "forward_adjoint_regression_pass": all_forward_adjoint_pass,
        "cuda_evaluation_improvement_pass": backend_results["cuda"][
            "conservative_evaluation_improvement"
        ]
        >= args.minimum_cuda_evaluation_improvement,
        "cuda_wall_improvement_pass": backend_results["cuda"][
            "conservative_wall_improvement"
        ]
        >= args.minimum_cuda_wall_improvement,
        "cpu_gradient_improvement_pass": backend_results["cpu"][
            "conservative_gradient_improvement"
        ]
        >= args.minimum_gradient_improvement,
        "cuda_gradient_improvement_pass": backend_results["cuda"][
            "conservative_gradient_improvement"
        ]
        >= args.minimum_gradient_improvement,
        "optimized_cuda_speedup_pass": conservative_optimized_cuda_speedup
        >= args.minimum_optimized_cuda_speedup,
    }
    gates["pass"] = all(gates.values())

    if input_hashes != {
        "legacy": sha256_file(args.legacy_report.resolve()),
        "analytic": sha256_file(args.analytic_report.resolve()),
    }:
        raise RuntimeError("an input report changed during comparison")
    snapshot_after = source_snapshot(repo)
    if snapshot_after != snapshot_before:
        raise RuntimeError("repository source changed during comparison")
    postprocessor_provenance = collect_postprocessor_provenance(
        repo, snapshot_after
    )
    report = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "inputs": {
            "legacy_report": {
                "path": str(args.legacy_report.resolve()),
                "sha256": input_hashes["legacy"],
            },
            "analytic_report": {
                "path": str(args.analytic_report.resolve()),
                "sha256": input_hashes["analytic"],
            },
        },
        "provenance": {
            field: analytic["provenance"][field] for field in PROVENANCE_FIELDS
        },
        "postprocessor_provenance": postprocessor_provenance,
        "workload": analytic_workload,
        "thresholds": {
            "gradient_atol": args.gradient_atol,
            "gradient_rtol": args.gradient_rtol,
            "minimum_cuda_evaluation_improvement": args.minimum_cuda_evaluation_improvement,
            "minimum_cuda_wall_improvement": args.minimum_cuda_wall_improvement,
            "minimum_gradient_improvement": args.minimum_gradient_improvement,
            "minimum_optimized_cuda_speedup": args.minimum_optimized_cuda_speedup,
            "maximum_forward_adjoint_regression": args.maximum_forward_adjoint_regression,
        },
        "backend_results": backend_results,
        "optimized_cuda_speedup_cpu_over_cuda": optimized_cuda_speedup,
        "conservative_optimized_cuda_speedup_cpu_over_cuda": (
            conservative_optimized_cuda_speedup
        ),
        "gradient_comparisons": comparisons,
        "objective_comparisons": objective_comparisons,
        "gate": gates,
    }
    markdown = "\n".join(
        (
            "# MaterialGrid analytic-gradient A/B",
            "",
            "- CPU evaluation improvement: "
            f"{backend_results['cpu']['evaluation_improvement']:.3f}×",
            "- CUDA evaluation improvement: "
            f"{backend_results['cuda']['evaluation_improvement']:.3f}×",
            "- Conservative CUDA evaluation improvement: "
            f"{backend_results['cuda']['conservative_evaluation_improvement']:.3f}×",
            f"- CUDA wall improvement: {backend_results['cuda']['wall_improvement']:.3f}×",
            "- Conservative CUDA wall improvement: "
            f"{backend_results['cuda']['conservative_wall_improvement']:.3f}×",
            f"- CPU gradient improvement: {backend_results['cpu']['gradient_improvement']:.1f}×",
            f"- CUDA gradient improvement: {backend_results['cuda']['gradient_improvement']:.1f}×",
            f"- Optimized CPU/CUDA speedup: {optimized_cuda_speedup:.3f}×",
            "- Conservative optimized CPU/CUDA speedup: "
            f"{conservative_optimized_cuda_speedup:.3f}×",
            f"- Cross-path gradient gate: {'PASS' if all_gradient_pass else 'FAIL'}",
            f"- Overall: {'PASS' if gates['pass'] else 'FAIL'}",
            "",
        )
    )
    if not gates["pass"]:
        failed_gates = [
            name
            for name, value in gates.items()
            if name.endswith("_pass") and value is False
        ]
        raise RuntimeError(
            "MaterialGrid comparison qualification gates failed: "
            + ", ".join(failed_gates)
        )
    publish_complete(args.output, run_id, report, markdown)
    print(markdown, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = pathlib.Path(__file__).resolve().parent.parent
    run_id = uuid.uuid4().hex
    prepare_output(args.output, run_id)
    try:
        return execute(args, repo, run_id)
    except BaseException as error:
        mark_failed(args.output, run_id, error)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
