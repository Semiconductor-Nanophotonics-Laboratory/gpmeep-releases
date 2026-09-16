#!/usr/bin/env python3
"""Replay one completed M3 distributed-feature unit from raw evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
import types
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DEFAULT = SCRIPT_DIR.parents[1]
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_m3_mpi_example_case as mpi_example  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
from common import StableFile, WorkloadError, file_record  # noqa: E402


SCHEMA = "gpmeep-m3-specialized-task-replay-v1"


def _source_module(name: str, path: pathlib.Path) -> types.ModuleType:
    with StableFile(path, f"M3 specialized verifier {name}") as stable:
        path = stable.path
        with stable.file_object() as handle:
            value = handle.read()
        stable.verify_unchanged()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(value, str(path), "exec"), module.__dict__)
    return module


def _load(path: pathlib.Path, label: str) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        if stable.initial_stat.st_size > 512 * 1024**2:
            raise WorkloadError(f"{label} exceeds its replay bound")
        with stable.file_object() as handle:
            try:
                value = json.load(
                    handle,
                    object_pairs_hook=mpi_example.validation.unique_json_object,
                )
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                mpi_example.validation.ManifestError,
            ) as exc:
                raise WorkloadError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return value


def task_name(unit: dict[str, Any]) -> str:
    order = unit.get("launch_order")
    unit_id = unit.get("unit_id")
    if type(order) is not int or not isinstance(unit_id, str):
        raise WorkloadError("M3 specialized task identity is invalid")
    return f"{order:02d}-{unit_id}"


def task_raw(output: pathlib.Path, unit: dict[str, Any]) -> pathlib.Path:
    return output / "raw" / task_name(unit)


def task_attestation(output: pathlib.Path, unit: dict[str, Any]) -> pathlib.Path:
    return output / "attestations" / f"{task_name(unit)}.json"


def _record(path: pathlib.Path, output: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        return file_record(path, output)
    except (OSError, WorkloadError) as exc:
        raise WorkloadError(f"M3 specialized {label} record failed: {exc}") from exc


def _python_example(
    raw: pathlib.Path, output: pathlib.Path, expected: dict[str, Any] | None
) -> dict[str, Any]:
    report = mpi_example.verify_complete(raw)
    value = {
        "driver": "python-mpi-example-case",
        "terminal": _record(raw / "COMPLETE", output, "MPI example COMPLETE"),
        "report": _record(raw / "report.json", output, "MPI example report"),
        "case_path": report["case_path"],
        "receipt_id": report["receipt_id"],
        "comparison_names": sorted(report["comparisons"]),
    }
    if expected is not None and expected != value:
        raise WorkloadError("M3 MPI example replay differs from retained evidence")
    return value


def _mpi_python(
    raw: pathlib.Path,
    output: pathlib.Path,
    repo: pathlib.Path,
    expected: dict[str, Any] | None,
) -> dict[str, Any]:
    module = _source_module(
        "gpmeep_m3_mpi_python_replay", repo / "scripts/run-mpi-python-validation.py"
    )
    module.verify_complete_publication(raw, repo)
    report = _load(raw / "report.json", "M3 MPI Python report")
    gate = report.get("gate", {})
    if gate.get("pass") is not True or any(value is not True for value in gate.values()):
        raise WorkloadError("M3 MPI Python report gate is not an exact PASS")
    value = {
        "driver": "run-mpi-python-validation",
        "terminal": _record(raw / "COMPLETE", output, "MPI Python COMPLETE"),
        "report": _record(raw / "report.json", output, "MPI Python report"),
        "artifact_manifest": _record(
            raw / "artifacts.sha256.json", output, "MPI Python artifact manifest"
        ),
        "receipt_id": report["build_receipt"]["receipt_id"],
        "comparison_names": sorted(report["comparisons"]),
    }
    if expected is not None and expected != value:
        raise WorkloadError("M3 MPI Python replay differs from retained evidence")
    return value


def _near2far(
    raw: pathlib.Path,
    output: pathlib.Path,
    repo: pathlib.Path,
    unit: dict[str, Any],
    expected: dict[str, Any] | None,
) -> dict[str, Any]:
    module = _source_module(
        "gpmeep_m3_near2far_replay", repo / "scripts/run-near2far-mpi-qualification.py"
    )
    attestation = task_attestation(output, unit)
    manifest_candidate = _load(
        raw / module.BUNDLE_MANIFEST_NAME, "M3 Near2Far evidence manifest"
    )
    bundle_id = (
        expected.get("bundle_id") if expected is not None else manifest_candidate.get("bundle_id")
    )
    if not isinstance(bundle_id, str) or re.fullmatch(r"[0-9a-f]{64}", bundle_id) is None:
        raise WorkloadError("M3 Near2Far bundle identity is invalid")
    manifest = module.verify_evidence_bundle(
        raw,
        repo,
        expected_bundle_id=bundle_id,
        expected_qualification_tier="release",
        release_attestation=attestation,
    )
    if manifest.get("bundle_id") != bundle_id:
        raise WorkloadError("M3 Near2Far verified bundle identity differs")
    report = _load(raw / "report.json", "M3 Near2Far report")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list) or [item.get("name") for item in scenarios] != [
        "single-point",
        "batch",
    ]:
        raise WorkloadError("M3 Near2Far scenario inventory differs")
    value = {
        "driver": "run-near2far-mpi-qualification",
        "terminal": _record(raw / "COMPLETE", output, "Near2Far COMPLETE"),
        "report": _record(raw / "report.json", output, "Near2Far report"),
        "manifest": _record(
            raw / module.BUNDLE_MANIFEST_NAME, output, "Near2Far manifest"
        ),
        "attestation": _record(attestation, output, "Near2Far attestation"),
        "bundle_id": bundle_id,
        "scenario_names": ["single-point", "batch"],
        "performance_scenarios": scenarios,
        "receipt_id": report["runtime_provenance"]["build_receipt"]["receipt_id"],
    }
    if expected is not None and expected != value:
        raise WorkloadError("M3 Near2Far replay differs from retained evidence")
    return value


def _adjoint(
    raw: pathlib.Path,
    output: pathlib.Path,
    repo: pathlib.Path,
    receipt: pathlib.Path,
    expected: dict[str, Any] | None,
) -> dict[str, Any]:
    module = _source_module(
        "gpmeep_m3_adjoint_public_replay",
        repo / "scripts/verify-mpi-adjoint-evidence.py",
    )
    summary = module.verify(
        argparse.Namespace(repo=repo, build_receipt=receipt, output=raw)
    )
    if summary.get("state") != "VERIFIED":
        raise WorkloadError("M3 MPI adjoint public verifier did not pass")
    value = {
        "driver": "run-mpi-adjoint-benchmark",
        "terminal": _record(raw / "COMPLETE", output, "MPI adjoint COMPLETE"),
        "report": _record(raw / "report.json", output, "MPI adjoint report"),
        "receipt_id": summary["receipt_id"],
        "run_id": summary["run_id"],
        "raw_samples": summary["raw_samples"],
        "headline_metric": summary["headline_metric"],
        "cpu_over_cuda_single_median_speedup": summary[
            "cpu_over_cuda_single_median_speedup"
        ],
        "cpu_over_cuda_multi_median_speedup": summary[
            "cpu_over_cuda_multi_median_speedup"
        ],
        "cuda_single_over_cuda_multi_median_speedup": summary[
            "cuda_single_over_cuda_multi_median_speedup"
        ],
        "maximum_gradient_absolute_error": summary[
            "maximum_gradient_absolute_error"
        ],
        "maximum_gradient_relative_l2_error": summary[
            "maximum_gradient_relative_l2_error"
        ],
        "maximum_objective_absolute_error": summary[
            "maximum_objective_absolute_error"
        ],
    }
    if expected is not None and expected != value:
        raise WorkloadError("M3 MPI adjoint replay differs from retained evidence")
    return value


def verify_task(
    unit: dict[str, Any],
    output: pathlib.Path,
    contract: dict[str, Any],
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    repo = pathlib.Path(contract["paths"]["repo"]).resolve(strict=True)
    receipt = pathlib.Path(contract["paths"]["build_receipt"]).resolve(strict=True)
    raw = task_raw(output, unit).resolve(strict=True)
    driver = unit.get("driver")
    if driver == "python-mpi-example-case":
        value = _python_example(raw, output, expected)
        if value["case_path"] != unit.get("target_path"):
            raise WorkloadError("M3 MPI example target differs from its plan")
    elif driver == "run-mpi-python-validation":
        value = _mpi_python(raw, output, repo, expected)
    elif driver == "run-near2far-mpi-qualification":
        value = _near2far(raw, output, repo, unit, expected)
    elif driver == "run-mpi-adjoint-benchmark":
        value = _adjoint(raw, output, repo, receipt, expected)
    else:
        raise WorkloadError(f"unsupported M3 specialized driver: {driver!r}")
    if value.get("receipt_id") != contract["build_receipt"]["receipt_id"]:
        raise WorkloadError("M3 specialized task receipt identity differs")
    return {"schema": SCHEMA, "unit_id": unit["unit_id"], "evidence": value}
