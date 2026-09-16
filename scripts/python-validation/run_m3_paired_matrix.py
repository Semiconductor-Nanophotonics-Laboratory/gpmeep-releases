#!/usr/bin/env python3
"""Run the 70 receipt-bound M3 CPU/strict-CUDA paired validation units."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import pathlib
import stat
import sys
import time
from dataclasses import dataclass
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DEFAULT = SCRIPT_DIR.parents[1]
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import archive_validation  # noqa: E402
import gpmeep_provenance as provenance  # noqa: E402
import m3_execution_plan  # noqa: E402
import run_observed_process as observed  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
import verify_telemetry_v2 as telemetry_replay  # noqa: E402
from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    file_record,
)


SCHEMA = "gpmeep-m3-paired-matrix-v1"
CHECKPOINT_SCHEMA = "gpmeep-m3-paired-checkpoint-v1"
JOURNAL_SCHEMA = "gpmeep-m3-paired-journal-v1"
COMPLETE_SCHEMA = "gpmeep-m3-paired-complete-v1"
TASK_COUNT = 70
INITIAL_FREE_BYTES = 40 * 1024**3
PER_TASK_FREE_BYTES = 5 * 1024**3
MINIMUM_FREE_INODES = 100_000
MINIMUM_HOST_AVAILABLE_BYTES = 4 * 1024**3
MINIMUM_CGROUP_FREE_BYTES = 4 * 1024**3
MINIMUM_GPU_FREE_MEMORY_FRACTION = 0.20
MAXIMUM_GPU_TEMPERATURE_CELSIUS = 90.0
MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES = 95_000.0


@dataclass(frozen=True)
class Task:
    index: int
    unit_id: str
    section: str
    case_path: str
    release_role: str
    purpose: str
    plan_launch_order: int
    covered_example: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument("--plan-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--python", required=True, type=pathlib.Path)
    parser.add_argument("--build-python", required=True, type=pathlib.Path)
    parser.add_argument("--install-prefix", required=True, type=pathlib.Path)
    parser.add_argument("--build-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--gpu-device", required=True)
    parser.add_argument("--unit-timeout-seconds", type=float, default=24 * 3600.0)
    parser.add_argument("--stdout-limit-mib", type=int, default=32)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.unit_timeout_seconds)
        or not 3600.0 <= args.unit_timeout_seconds <= 48 * 3600.0
    ):
        raise WorkloadError("M3 paired unit timeout must be in [1h,48h]")
    if not 1 <= args.stdout_limit_mib <= 1024:
        raise WorkloadError("M3 paired stdout limit must be in [1,1024] MiB")
    if (
        not math.isfinite(args.telemetry_interval_seconds)
        or not 1.0 <= args.telemetry_interval_seconds <= 60.0
    ):
        raise WorkloadError("M3 paired telemetry interval must be in [1s,60s]")


def normalized_gpu(value: str) -> str:
    return matrix._normalized_uuid(value)


def build_tasks(plan: dict[str, Any]) -> list[Task]:
    if plan.get("schema") != m3_execution_plan.SCHEMA:
        raise WorkloadError("M3 paired execution plan schema differs")
    rows = list(plan.get("selected", [])) + list(plan.get("stronger", []))
    tasks = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise WorkloadError("M3 paired plan row is invalid")
        tasks.append(
            Task(
                index=index,
                unit_id=row["unit_id"],
                section=row["section"],
                case_path=row["case_path"],
                release_role=row["release_role"],
                purpose=row["purpose"],
                plan_launch_order=row["launch_order"],
                covered_example=row["covered_example"],
            )
        )
    if len(tasks) != TASK_COUNT or [task.index for task in tasks] != list(
        range(1, TASK_COUNT + 1)
    ):
        raise WorkloadError("M3 paired task inventory differs")
    if len({task.unit_id for task in tasks}) != TASK_COUNT:
        raise WorkloadError("M3 paired task IDs are not unique")
    return tasks


def task_spec(task: Task) -> dict[str, Any]:
    return {
        "index": task.index,
        "unit_id": task.unit_id,
        "section": task.section,
        "case_path": task.case_path,
        "backends": ["cpu", "cuda"],
        "release_role": task.release_role,
        "purpose": task.purpose,
        "plan_launch_order": task.plan_launch_order,
        "covered_example": task.covered_example,
    }


def task_name(task: Task) -> str:
    return f"{task.index:03d}-{task.unit_id}"


def task_raw_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("raw", task_name(task))


def task_archive_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("archives", task_name(task))


def task_runtime_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("runtime", task_name(task))


def task_log_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("logs", f"{task_name(task)}.log")


def task_telemetry_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("telemetry", task_name(task))


def task_command(
    task: Task, contract: dict[str, Any], output: pathlib.Path
) -> list[str]:
    paths = contract["paths"]
    return [
        paths["python"],
        paths["runner"],
        "--repo",
        paths["repo"],
        "--manifest",
        paths["manifest"],
        "--build-python",
        paths["build_python"],
        "--install-prefix",
        paths["install_prefix"],
        "--python",
        paths["python"],
        "--output",
        str(output.joinpath(*task_raw_relative(task).parts)),
        "--backend",
        "both",
        "--tier",
        "full",
        "--case",
        task.case_path,
    ]


def capacity_snapshot(path: pathlib.Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    usage = os.statvfs(path)
    return {
        "path": str(path),
        "free_bytes": usage.f_bavail * usage.f_frsize,
        "total_bytes": usage.f_blocks * usage.f_frsize,
        "free_inodes": usage.f_favail,
        "total_inodes": usage.f_files,
    }


def validate_capacity(record: Any, initial: bool) -> None:
    keys = {"path", "free_bytes", "total_bytes", "free_inodes", "total_inodes"}
    if (
        not isinstance(record, dict)
        or set(record) != keys
        or not isinstance(record["path"], str)
        or not pathlib.Path(record["path"]).is_absolute()
        or any(
            type(record[name]) is not int or record[name] < 0
            for name in ("free_bytes", "total_bytes", "free_inodes", "total_inodes")
        )
        or record["free_bytes"] > record["total_bytes"]
        or record["free_inodes"] > record["total_inodes"]
    ):
        raise WorkloadError("M3 paired capacity record is invalid")
    minimum = INITIAL_FREE_BYTES if initial else PER_TASK_FREE_BYTES
    if record["free_bytes"] < minimum:
        raise WorkloadError("M3 paired free-space gate failed")
    if record["free_inodes"] < MINIMUM_FREE_INODES:
        raise WorkloadError("M3 paired free-inode gate failed")


def _stable_code_records(
    repo: pathlib.Path,
) -> tuple[dict[str, StableFile], dict[str, dict[str, Any]]]:
    paths = {
        "controller": pathlib.Path(__file__),
        "plan_loader": SCRIPT_DIR / "m3_execution_plan.py",
        "runner": SCRIPT_DIR / "run_validation.py",
        "archive_replay": SCRIPT_DIR / "archive_validation.py",
        "manifest": SCRIPT_DIR / "manifest.json",
        "provenance": SCRIPT_DIR.parent / "gpmeep_provenance.py",
        "qualification_contract": (
            SCRIPT_DIR.parent / "gpmeep_qualification_contract.py"
        ),
        "observed_runner": USER_WORKLOAD_DIR / "run_observed_process.py",
        "matrix_common": USER_WORKLOAD_DIR / "run_user_workload_matrix.py",
        "host_telemetry": USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py",
        "gpu_telemetry": USER_WORKLOAD_DIR / "capture_gpu_telemetry_v2.py",
        "telemetry_replay": USER_WORKLOAD_DIR / "verify_telemetry_v2.py",
        "workload_common": USER_WORKLOAD_DIR / "common.py",
    }
    stables = {name: StableFile(path, f"M3 paired {name}") for name, path in paths.items()}
    records = {
        name: matrix.stable_identity_record(stable, repo)
        for name, stable in stables.items()
    }
    return stables, records


def _receipt_record(
    receipt_path: pathlib.Path, receipt: dict[str, Any]
) -> dict[str, Any]:
    required = ("receipt_id", "build_input_id", "artifact_set_id")
    if any(not isinstance(receipt.get(name), str) for name in required):
        raise WorkloadError("M3 paired build receipt IDs are invalid")
    source_sha256 = receipt.get("source_start", {}).get("sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise WorkloadError("M3 paired build source identity is invalid")
    return {
        "path": str(receipt_path),
        "size_bytes": receipt_path.stat().st_size,
        "sha256": provenance.sha256_file(receipt_path),
        "receipt_id": receipt["receipt_id"],
        "build_input_id": receipt["build_input_id"],
        "artifact_set_id": receipt["artifact_set_id"],
        "source_sha256": source_sha256,
    }


def build_contract(
    args: argparse.Namespace,
    repo: pathlib.Path,
    plan: dict[str, Any],
    tasks: list[Task],
    code: dict[str, Any],
    receipt: dict[str, Any],
    gpu: str,
    nvidia_smi: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "paths": {
            "repo": str(repo),
            "plan_root": str(args.plan_root.resolve(strict=True)),
            "python": str(args.python.resolve(strict=True)),
            "build_python": str(args.build_python.resolve(strict=True)),
            "install_prefix": str(args.install_prefix.resolve(strict=True)),
            "build_receipt": str(args.build_receipt.resolve(strict=True)),
            "runner": str((SCRIPT_DIR / "run_validation.py").resolve(strict=True)),
            "manifest": str((SCRIPT_DIR / "manifest.json").resolve(strict=True)),
        },
        "plan": plan,
        "tasks": [task_spec(task) for task in tasks],
        "evidence_code": code,
        "build_receipt": receipt,
        "gpu_device": gpu,
        "nvidia_smi": nvidia_smi,
        "limits": {
            "unit_timeout_seconds": args.unit_timeout_seconds,
            "stdout_limit_bytes": args.stdout_limit_mib * 1024**2,
            "telemetry_interval_seconds": args.telemetry_interval_seconds,
        },
        "gates": {
            "initial_free_bytes": INITIAL_FREE_BYTES,
            "per_task_free_bytes": PER_TASK_FREE_BYTES,
            "minimum_free_inodes": MINIMUM_FREE_INODES,
            "minimum_host_available_bytes": MINIMUM_HOST_AVAILABLE_BYTES,
            "minimum_cgroup_free_bytes": MINIMUM_CGROUP_FREE_BYTES,
            "minimum_gpu_free_memory_fraction": MINIMUM_GPU_FREE_MEMORY_FRACTION,
            "maximum_gpu_temperature_celsius": MAXIMUM_GPU_TEMPERATURE_CELSIUS,
            "maximum_host_temperature_millidegrees": (
                MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES
            ),
        },
    }


def _paths_overlap(first: pathlib.Path, second: pathlib.Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_output_isolation(
    output: pathlib.Path, repo: pathlib.Path, plan_root: pathlib.Path
) -> None:
    output = pathlib.Path(os.path.abspath(output))
    repo = repo.resolve(strict=True)
    plan_root = plan_root.resolve(strict=True)
    if _paths_overlap(output, repo) or _paths_overlap(output, plan_root):
        raise WorkloadError("M3 paired output overlaps an immutable input root")


def validate_contract(
    contract: Any,
) -> tuple[pathlib.Path, dict[str, Any], list[Task], str]:
    keys = {
        "schema",
        "paths",
        "plan",
        "tasks",
        "evidence_code",
        "build_receipt",
        "gpu_device",
        "nvidia_smi",
        "limits",
        "gates",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != keys
        or contract.get("schema") != SCHEMA
    ):
        raise WorkloadError("M3 paired sealed contract schema differs")
    path_keys = {
        "repo",
        "plan_root",
        "python",
        "build_python",
        "install_prefix",
        "build_receipt",
        "runner",
        "manifest",
    }
    paths = contract["paths"]
    if (
        not isinstance(paths, dict)
        or set(paths) != path_keys
        or any(not isinstance(paths[name], str) for name in path_keys)
    ):
        raise WorkloadError("M3 paired sealed path contract differs")
    resolved = {}
    for name in path_keys:
        path = pathlib.Path(paths[name])
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise WorkloadError(f"M3 paired sealed path is not canonical: {name}")
        resolved[name] = path
    repo = resolved["repo"]
    if not repo.is_dir() or not resolved["plan_root"].is_dir():
        raise WorkloadError("M3 paired sealed input root type differs")
    if (
        not resolved["python"].is_file()
        or not resolved["build_python"].is_dir()
        or not resolved["install_prefix"].is_dir()
        or not resolved["build_receipt"].is_file()
        or not resolved["runner"].is_file()
        or not resolved["manifest"].is_file()
    ):
        raise WorkloadError("M3 paired sealed runtime path type differs")
    if (
        resolved["runner"] != (SCRIPT_DIR / "run_validation.py").resolve(strict=True)
        or resolved["manifest"] != (SCRIPT_DIR / "manifest.json").resolve(strict=True)
        or resolved["build_python"].parent / "build-provenance.json"
        != resolved["build_receipt"]
    ):
        raise WorkloadError("M3 paired sealed runtime path relationship differs")
    plan = m3_execution_plan.load_release_plan(resolved["plan_root"], repo)
    tasks = build_tasks(plan)
    if contract["plan"] != plan or contract["tasks"] != [
        task_spec(task) for task in tasks
    ]:
        raise WorkloadError("M3 paired sealed plan was not re-derived")
    expected_code = {
        "controller": pathlib.Path(__file__),
        "plan_loader": SCRIPT_DIR / "m3_execution_plan.py",
        "runner": SCRIPT_DIR / "run_validation.py",
        "archive_replay": SCRIPT_DIR / "archive_validation.py",
        "manifest": SCRIPT_DIR / "manifest.json",
        "provenance": SCRIPT_DIR.parent / "gpmeep_provenance.py",
        "qualification_contract": (
            SCRIPT_DIR.parent / "gpmeep_qualification_contract.py"
        ),
        "observed_runner": USER_WORKLOAD_DIR / "run_observed_process.py",
        "matrix_common": USER_WORKLOAD_DIR / "run_user_workload_matrix.py",
        "host_telemetry": USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py",
        "gpu_telemetry": USER_WORKLOAD_DIR / "capture_gpu_telemetry_v2.py",
        "telemetry_replay": USER_WORKLOAD_DIR / "verify_telemetry_v2.py",
        "workload_common": USER_WORKLOAD_DIR / "common.py",
    }
    code = contract["evidence_code"]
    if not isinstance(code, dict) or set(code) != set(expected_code):
        raise WorkloadError("M3 paired evidence-code inventory differs")
    for name, expected_path in expected_code.items():
        stable = matrix.stable_record(repo, code[name], f"M3 paired {name}")
        try:
            if stable.path != expected_path.resolve(strict=True):
                raise WorkloadError(f"M3 paired evidence-code path differs: {name}")
        finally:
            stable.close()
    try:
        receipt = provenance.verify_build_receipt(resolved["build_receipt"], repo)
    except provenance.ProvenanceError as exc:
        raise WorkloadError(f"M3 paired build receipt replay failed: {exc}") from exc
    if contract["build_receipt"] != _receipt_record(
        resolved["build_receipt"], receipt
    ):
        raise WorkloadError("M3 paired build receipt record differs")
    gpu = normalized_gpu(contract["gpu_device"])
    if gpu != contract["gpu_device"]:
        raise WorkloadError("M3 paired GPU UUID spelling differs")
    if contract["nvidia_smi"] != absolute_file_record(
        pathlib.Path("/usr/bin/nvidia-smi"), "nvidia-smi"
    ):
        raise WorkloadError("M3 paired nvidia-smi identity differs")
    limits = contract["limits"]
    if not isinstance(limits, dict) or set(limits) != {
        "unit_timeout_seconds",
        "stdout_limit_bytes",
        "telemetry_interval_seconds",
    }:
        raise WorkloadError("M3 paired sealed limits differ")
    timeout = limits["unit_timeout_seconds"]
    interval = limits["telemetry_interval_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or not 3600.0 <= timeout <= 48 * 3600.0
        or type(limits["stdout_limit_bytes"]) is not int
        or not 1024**2 <= limits["stdout_limit_bytes"] <= 1024**3
        or limits["stdout_limit_bytes"] % 1024**2
        or isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or not 1.0 <= interval <= 60.0
    ):
        raise WorkloadError("M3 paired sealed limit value differs")
    expected_gates = {
        "initial_free_bytes": INITIAL_FREE_BYTES,
        "per_task_free_bytes": PER_TASK_FREE_BYTES,
        "minimum_free_inodes": MINIMUM_FREE_INODES,
        "minimum_host_available_bytes": MINIMUM_HOST_AVAILABLE_BYTES,
        "minimum_cgroup_free_bytes": MINIMUM_CGROUP_FREE_BYTES,
        "minimum_gpu_free_memory_fraction": MINIMUM_GPU_FREE_MEMORY_FRACTION,
        "maximum_gpu_temperature_celsius": MAXIMUM_GPU_TEMPERATURE_CELSIUS,
        "maximum_host_temperature_millidegrees": (
            MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES
        ),
    }
    if contract["gates"] != expected_gates:
        raise WorkloadError("M3 paired sealed gates differ")
    return repo, plan, tasks, gpu


def clean_environment(
    contract: dict[str, Any], runtime: pathlib.Path
) -> dict[str, str]:
    runtime.mkdir(parents=True, exist_ok=False)
    for name in ("home", "cache", "tmp"):
        (runtime / name).mkdir()
    python = pathlib.Path(contract["paths"]["python"])
    return {
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "HOME": str(runtime / "home"),
        "XDG_CACHE_HOME": str(runtime / "cache"),
        "TMPDIR": str(runtime / "tmp"),
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": contract["gpu_device"],
    }


def observer_specs(
    task: Task,
    root_pid: int,
    output: pathlib.Path,
    contract: dict[str, Any],
) -> list[observed.ObserverSpec]:
    telemetry = output.joinpath(*task_telemetry_relative(task).parts)
    python = contract["paths"]["python"]
    interval = contract["limits"]["telemetry_interval_seconds"]
    label = task_name(task)
    return [
        observed.ObserverSpec(
            "host",
            (
                python,
                str(USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py"),
                "--root-pid",
                str(root_pid),
                "--output",
                str(telemetry / "host.jsonl"),
                "--label",
                label,
                "--interval-seconds",
                repr(interval),
            ),
            telemetry / "host-monitor.log",
        ),
        observed.ObserverSpec(
            "gpu",
            (
                python,
                str(USER_WORKLOAD_DIR / "capture_gpu_telemetry_v2.py"),
                "--root-pid",
                str(root_pid),
                "--output",
                str(telemetry / "gpu.jsonl"),
                "--label",
                label,
                "--selected-uuid",
                contract["gpu_device"],
                "--interval-seconds",
                repr(interval),
            ),
            telemetry / "gpu-monitor.log",
        ),
    ]


def _code_sha256(contract: dict[str, Any], name: str) -> str:
    try:
        value = contract["evidence_code"][name]["sha256"]
    except (KeyError, TypeError) as exc:
        raise WorkloadError(f"M3 paired code identity is absent: {name}") from exc
    if not isinstance(value, str) or len(value) != 64:
        raise WorkloadError(f"M3 paired code digest is invalid: {name}")
    return value


def derive_telemetry(
    task: Task,
    process: dict[str, Any],
    output: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    root = output.joinpath(*task_telemetry_relative(task).parts)
    gates = contract["gates"]
    return telemetry_replay.verify_pair(
        host_stream=root / "host.jsonl",
        host_terminal=root / "host.jsonl.COMPLETE.json",
        gpu_stream=root / "gpu.jsonl",
        gpu_terminal=root / "gpu.jsonl.COMPLETE.json",
        expected_root_pid=process["command_pid"],
        expected_label=task_name(task),
        expected_host_monitor_sha256=_code_sha256(contract, "host_telemetry"),
        expected_gpu_uuids=(contract["gpu_device"],),
        expected_gpu_monitor_sha256=_code_sha256(contract, "gpu_telemetry"),
        expected_nvidia_smi_sha256=contract["nvidia_smi"]["sha256"],
        minimum_host_available_bytes=gates["minimum_host_available_bytes"],
        minimum_cgroup_free_bytes=gates["minimum_cgroup_free_bytes"],
        maximum_host_temperature_millidegrees=(
            gates["maximum_host_temperature_millidegrees"]
        ),
        minimum_gpu_free_memory_fraction=(
            gates["minimum_gpu_free_memory_fraction"]
        ),
        maximum_gpu_temperature_celsius=(
            gates["maximum_gpu_temperature_celsius"]
        ),
    )


def publish_telemetry(
    task: Task,
    process: dict[str, Any],
    observers: list[dict[str, Any]],
    output: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    specs = observer_specs(task, process["command_pid"], output, contract)
    if len(observers) != len(specs):
        raise WorkloadError("M3 paired observer inventory differs")
    for record, spec in zip(observers, specs, strict=True):
        observed.verify_observer_record(output, record, spec)
    report = derive_telemetry(task, process, output, contract)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    atomic_write_json(path, report)
    return matrix.stable_file_record(path, output, "M3 paired telemetry replay")


def validate_telemetry(
    task: Task,
    process: dict[str, Any],
    observers: list[dict[str, Any]],
    retained: Any,
    output: pathlib.Path,
    contract: dict[str, Any],
) -> None:
    specs = observer_specs(task, process["command_pid"], output, contract)
    if len(observers) != len(specs):
        raise WorkloadError("M3 paired observer inventory differs")
    for record, spec in zip(observers, specs, strict=True):
        observed.verify_observer_record(output, record, spec)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    expected = matrix.stable_file_record(path, output, "M3 paired telemetry replay")
    if retained != expected:
        raise WorkloadError("M3 paired telemetry replay record differs")
    with StableFile(path, "M3 paired telemetry replay") as stable:
        with stable.file_object() as handle:
            saved = json.load(handle)
    if saved != derive_telemetry(task, process, output, contract):
        raise WorkloadError("M3 paired telemetry replay was not re-derived")


def _load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        with stable.file_object() as handle:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return value


def derive_archive_record(
    task: Task,
    output: pathlib.Path,
    contract: dict[str, Any],
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    archive = output.joinpath(*task_archive_relative(task).parts)
    if expected_manifest_sha256 is None:
        complete = archive_validation.verify_archive(archive)
    else:
        complete = archive_validation.verify_archive(
            archive,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_receipt_id=contract["build_receipt"]["receipt_id"],
        )
    if complete.get("build_receipt_id") != contract["build_receipt"]["receipt_id"]:
        raise WorkloadError("M3 paired archive receipt differs")
    report_path = archive.joinpath(*archive_validation.RUNNER_ROOT.parts) / "report.json"
    report = _load_json(report_path, "M3 paired runner report")
    selected = [item for item in report.get("results", []) if item.get("selected")]
    configuration = report.get("configuration", {})
    if (
        report.get("exit_code") != 0
        or len(selected) != 1
        or selected[0].get("path") != task.case_path
        or selected[0].get("outcome") != "PASS"
        or set(selected[0].get("runs", {})) != {"cpu", "cuda"}
        or any(
            not run.get("outcome", "").startswith("PASS")
            for run in selected[0]["runs"].values()
        )
        or selected[0].get("comparison", {}).get("outcome") != "PASS"
        or configuration.get("case_ids") != [task.case_path]
        or configuration.get("backends") != ["cpu", "cuda"]
        or configuration.get("tiers") != ["full"]
        or configuration.get("performance_evidence", {}).get(
            "valid_for_speed_gate"
        )
        is not False
        or configuration.get("performance_evidence", {}).get(
            "concurrency_detected"
        )
        is not False
    ):
        raise WorkloadError("M3 paired selected runner result is not an exact PASS")
    return {
        "path": task_archive_relative(task).as_posix(),
        "archive_complete": matrix.stable_file_record(
            archive / archive_validation.COMPLETE_NAME,
            output,
            "M3 paired ARCHIVE_COMPLETE",
        ),
        "archive_manifest": matrix.stable_file_record(
            archive / archive_validation.MANIFEST_NAME,
            output,
            "M3 paired ARCHIVE_MANIFEST",
        ),
        "runner_report": matrix.stable_file_record(
            report_path, output, "M3 paired runner report"
        ),
        "archive_manifest_sha256": complete["archive_manifest_sha256"],
        "report_sha256": complete["report_sha256"],
        "receipt_id": complete["build_receipt_id"],
    }


def validate_archive_record(
    task: Task, retained: Any, output: pathlib.Path, contract: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(retained, dict) or set(retained) != {
        "path",
        "archive_complete",
        "archive_manifest",
        "runner_report",
        "archive_manifest_sha256",
        "report_sha256",
        "receipt_id",
    }:
        raise WorkloadError("M3 paired archive record schema differs")
    if retained["path"] != task_archive_relative(task).as_posix():
        raise WorkloadError("M3 paired archive path differs")
    derived = derive_archive_record(
        task,
        output,
        contract,
        expected_manifest_sha256=retained["archive_manifest_sha256"],
    )
    if retained != derived:
        raise WorkloadError("M3 paired archive record was not re-derived")
    return retained


def write_checkpoint(
    output: pathlib.Path,
    contract: dict[str, Any],
    hardware_before: dict[str, Any],
    capacity_before: dict[str, Any],
    completed: list[dict[str, Any]],
) -> None:
    atomic_write_json(
        output / "CHECKPOINT.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "contract": contract,
            "hardware_before": hardware_before,
            "capacity_before": capacity_before,
            "completed_tasks": completed,
        },
    )


def append_event(
    output: pathlib.Path, journal: dict[str, Any], event: dict[str, Any]
) -> None:
    journal["events"].append(event)
    atomic_write_json(output / "JOURNAL.json", journal)


def expected_journal(
    journal: Any, contract: dict[str, Any], completed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if (
        not isinstance(journal, dict)
        or set(journal) != {"schema", "contract", "events"}
        or journal["schema"] != JOURNAL_SCHEMA
        or journal["contract"] != contract
        or not isinstance(journal["events"], list)
        or not journal["events"]
    ):
        raise WorkloadError("M3 paired journal identity differs")
    first = journal["events"][0]
    if first.get("state") != "controller-started" or not math.isfinite(
        float(first.get("unix_seconds", math.nan))
    ):
        raise WorkloadError("M3 paired journal start event differs")
    expected = [first]
    for record in completed:
        expected.extend(
            (
                {"state": "task-started", "task": record["task"]},
                {
                    "state": "task-process-ended",
                    "task": record["task"],
                    "process": record["process"],
                },
                {
                    "state": "task-validated",
                    "task": record["task"],
                    "archive_complete": record["archive"]["archive_complete"],
                },
            )
        )
    if journal["events"][: len(expected)] != expected:
        raise WorkloadError("M3 paired journal disagrees with checkpoint")
    return expected


def validate_completed_task(
    task: Task,
    retained: Any,
    output: pathlib.Path,
    contract: dict[str, Any],
    gpu: str,
) -> dict[str, Any]:
    keys = {
        "task",
        "process",
        "observers",
        "telemetry_replay",
        "archive",
        "hardware",
        "capacity_before",
        "capacity_after",
    }
    if not isinstance(retained, dict) or set(retained) != keys:
        raise WorkloadError("M3 paired completed-task record schema differs")
    if retained["task"] != task_spec(task):
        raise WorkloadError("M3 paired completed-task order differs")
    matrix._verify_process_record(
        output,
        retained["process"],
        task_log_relative(task).as_posix(),
        task_name(task),
    )
    if retained["process"]["command"] != task_command(task, contract, output):
        raise WorkloadError("M3 paired task command differs")
    validate_telemetry(
        task,
        retained["process"],
        retained["observers"],
        retained["telemetry_replay"],
        output,
        contract,
    )
    matrix.validate_hardware_evidence(retained["hardware"], [gpu])
    validate_capacity(retained["capacity_before"], False)
    validate_capacity(retained["capacity_after"], False)
    validate_archive_record(task, retained["archive"], output, contract)
    return retained


def _load_checkpoint(output: pathlib.Path, contract: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _load_json(output / "CHECKPOINT.json", "M3 paired checkpoint")
    if set(checkpoint) != {
        "schema",
        "contract",
        "hardware_before",
        "capacity_before",
        "completed_tasks",
    } or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise WorkloadError("M3 paired checkpoint schema differs")
    if checkpoint["contract"] != contract:
        raise WorkloadError("M3 paired resume invocation differs")
    if not isinstance(checkpoint["completed_tasks"], list):
        raise WorkloadError("M3 paired checkpoint task inventory is invalid")
    return checkpoint


def _terminal_payload(output: pathlib.Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": file_record(output / "report.json", output),
        "journal": file_record(output / "JOURNAL.json", output),
        "checkpoint": report["checkpoint"],
    }


def verify_complete(
    output: pathlib.Path, terminal_name: str = "PAIRED_COMPLETE"
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {"PAIRED_COMPLETE", "PENDING_PAIRED_COMPLETE"}:
        raise WorkloadError("M3 paired terminal filename is invalid")
    complete = _load_json(output / terminal_name, "M3 paired COMPLETE")
    if set(complete) != {
        "schema",
        "outcome",
        "report",
        "journal",
        "checkpoint",
    } or complete.get("schema") != COMPLETE_SCHEMA or complete.get("outcome") != "PASS":
        raise WorkloadError("M3 paired COMPLETE is not an exact PASS")
    report = matrix.load_stable_json_record(
        output, complete["report"], "M3 paired report"
    )
    journal = matrix.load_stable_json_record(
        output, complete["journal"], "M3 paired journal"
    )
    checkpoint = matrix.load_stable_json_record(
        output, complete["checkpoint"], "M3 paired checkpoint"
    )
    report_keys = {
        "schema",
        "outcome",
        "contract",
        "tasks",
        "hardware",
        "capacity_initial",
        "capacity_final",
        "resume_history",
        "checkpoint",
    }
    checkpoint_keys = {
        "schema",
        "contract",
        "hardware_before",
        "capacity_before",
        "completed_tasks",
    }
    if (
        not isinstance(report, dict)
        or set(report) != report_keys
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
    ):
        raise WorkloadError("M3 paired report is not a PASS")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != checkpoint_keys
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
    ):
        raise WorkloadError("M3 paired checkpoint schema differs")
    contract = checkpoint.get("contract")
    _repo, _plan, tasks, gpu = validate_contract(contract)
    if report["contract"] != contract:
        raise WorkloadError("M3 paired report contract differs")
    completed = checkpoint.get("completed_tasks")
    if (
        not isinstance(completed, list)
        or len(completed) != TASK_COUNT
        or report.get("tasks") != completed
    ):
        raise WorkloadError("M3 paired completed-task inventory differs")
    for retained, task in zip(completed, tasks, strict=True):
        validate_completed_task(task, retained, output, contract, gpu)
    if expected_journal(journal, contract, completed) != journal["events"]:
        raise WorkloadError("M3 paired journal has unsealed events")
    matrix.validate_hardware_evidence(report.get("hardware"), [gpu])
    if checkpoint.get("hardware_before") != report["hardware"]["before"]:
        raise WorkloadError("M3 paired initial hardware differs")
    if checkpoint.get("capacity_before") != report.get("capacity_initial"):
        raise WorkloadError("M3 paired initial capacity differs")
    validate_capacity(report["capacity_initial"], True)
    validate_capacity(report["capacity_final"], False)
    if report.get("resume_history") != matrix.resume_history_records(output):
        raise WorkloadError("M3 paired resume history differs")
    if complete != _terminal_payload(output, report):
        raise WorkloadError("M3 paired COMPLETE payload differs")
    return report


def publish_terminal(output: pathlib.Path, report: dict[str, Any]) -> None:
    atomic_write_json(
        output / "PENDING_PAIRED_COMPLETE", _terminal_payload(output, report)
    )
    verify_complete(output, "PENDING_PAIRED_COMPLETE")
    os.replace(output / "PENDING_PAIRED_COMPLETE", output / "PAIRED_COMPLETE")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    repo = args.repo.resolve(strict=True)
    validate_output_isolation(args.output, repo, args.plan_root)
    if args.resume:
        if args.output.is_symlink():
            raise WorkloadError("M3 paired resume output is a symlink")
        output = args.output.resolve(strict=True)
        if (output / "PAIRED_COMPLETE").exists():
            raise WorkloadError("completed M3 paired matrix cannot be resumed")
    else:
        if args.output.is_symlink():
            raise WorkloadError("M3 paired output is a symlink")
        output = pathlib.Path(os.path.abspath(args.output))
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.parent.resolve(strict=True) != output.parent:
            raise WorkloadError("M3 paired output parent contains a symlink")
        output.mkdir(exist_ok=False)
    lock_path = output / "LOCK"
    if args.resume and not stat.S_ISREG(lock_path.lstat().st_mode):
        raise WorkloadError("M3 paired lock is not a regular file")
    lock = lock_path.open("r+" if args.resume else "x")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise WorkloadError(f"could not acquire M3 paired lock: {exc}") from exc

    code_stables: dict[str, StableFile] = {}
    receipt_stable: StableFile | None = None
    nvidia_stable: StableFile | None = None
    try:
        plan = m3_execution_plan.load_release_plan(args.plan_root, repo)
        tasks = build_tasks(plan)
        gpu = normalized_gpu(args.gpu_device)
        code_stables, code_records = _stable_code_records(repo)
        receipt_path = args.build_receipt.resolve(strict=True)
        receipt_stable = StableFile(receipt_path, "M3 paired build receipt")
        receipt = provenance.verify_build_receipt(receipt_path, repo)
        receipt_record = _receipt_record(receipt_path, receipt)
        if args.build_python.resolve(strict=True).parent / "build-provenance.json" != receipt_path:
            raise WorkloadError("M3 paired Python and build receipt prefixes differ")
        nvidia_stable = StableFile(pathlib.Path("/usr/bin/nvidia-smi"), "nvidia-smi")
        nvidia_record = absolute_file_record(nvidia_stable.path, "nvidia-smi")
        contract = build_contract(
            args,
            repo,
            plan,
            tasks,
            code_records,
            receipt_record,
            gpu,
            nvidia_record,
        )
        validate_contract(contract)
        current_hardware = matrix.hardware_snapshot((gpu,))
        current_capacity = capacity_snapshot(output)
        completed: list[dict[str, Any]] = []
        if args.resume:
            checkpoint = _load_checkpoint(output, contract)
            hardware_before = checkpoint["hardware_before"]
            capacity_before = checkpoint["capacity_before"]
            matrix.validate_hardware_evidence(
                {"before": hardware_before, "after": current_hardware}, [gpu]
            )
            validate_capacity(capacity_before, True)
            if len(checkpoint["completed_tasks"]) > TASK_COUNT:
                raise WorkloadError("M3 paired checkpoint has too many tasks")
            for retained, task in zip(
                checkpoint["completed_tasks"], tasks, strict=False
            ):
                completed.append(
                    validate_completed_task(task, retained, output, contract, gpu)
                )
            journal = _load_json(output / "JOURNAL.json", "M3 paired journal")
            prefix = expected_journal(journal, contract, completed)
            partial = [pathlib.PurePosixPath("PENDING_PAIRED_COMPLETE")]
            if len(completed) < TASK_COUNT:
                task = tasks[len(completed)]
                partial.extend(
                    (
                        task_raw_relative(task),
                        task_archive_relative(task),
                        task_runtime_relative(task),
                        task_log_relative(task),
                        task_telemetry_relative(task),
                    )
                )
            matrix.archive_interrupted_attempt(output, partial)
            journal["events"] = prefix
            atomic_write_json(output / "JOURNAL.json", journal)
            atomic_write_json(
                output / "RUNNING.json",
                {
                    "schema": SCHEMA,
                    "resumed_unix_seconds": time.time(),
                    "completed_tasks": len(completed),
                },
            )
        else:
            hardware_before = current_hardware
            capacity_before = current_capacity
            validate_capacity(capacity_before, True)
            journal = {
                "schema": JOURNAL_SCHEMA,
                "contract": contract,
                "events": [
                    {"state": "controller-started", "unix_seconds": time.time()}
                ],
            }
            atomic_write_json(output / "JOURNAL.json", journal)
            atomic_write_json(
                output / "RUNNING.json",
                {"schema": SCHEMA, "started_unix_seconds": time.time()},
            )
            write_checkpoint(
                output, contract, hardware_before, capacity_before, completed
            )

        for task in tasks[len(completed) :]:
            before_capacity = capacity_snapshot(output)
            validate_capacity(before_capacity, False)
            before_hardware = matrix.hardware_snapshot((gpu,))
            runtime = output.joinpath(*task_runtime_relative(task).parts)
            environment = clean_environment(contract, runtime)
            append_event(
                output, journal, {"state": "task-started", "task": task_spec(task)}
            )
            process, observer_records = observed.run_bounded_observed(
                task_command(task, contract, output),
                environment,
                output,
                output.joinpath(*task_log_relative(task).parts),
                contract["limits"]["unit_timeout_seconds"],
                contract["limits"]["stdout_limit_bytes"],
                lambda pid, selected=task: observer_specs(
                    selected, pid, output, contract
                ),
                observer_environment=environment,
                observer_exit_timeout_seconds=(
                    contract["limits"]["telemetry_interval_seconds"] * 3 + 30
                ),
                working_directory=repo,
            )
            append_event(
                output,
                journal,
                {
                    "state": "task-process-ended",
                    "task": task_spec(task),
                    "process": process,
                },
            )
            if process["returncode"] != 0 or process["timed_out"] or process[
                "output_limited"
            ]:
                raise WorkloadError(f"M3 paired task failed: {task_name(task)}")
            after_hardware = matrix.hardware_snapshot((gpu,))
            hardware = {"before": before_hardware, "after": after_hardware}
            matrix.validate_hardware_evidence(hardware, [gpu])
            telemetry_record = publish_telemetry(
                task, process, observer_records, output, contract
            )
            raw = output.joinpath(*task_raw_relative(task).parts)
            archive = output.joinpath(*task_archive_relative(task).parts)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive_validation.seal_archive(raw, archive, repo, receipt_path)
            archive_record = derive_archive_record(task, output, contract)
            retained = {
                "task": task_spec(task),
                "process": process,
                "observers": observer_records,
                "telemetry_replay": telemetry_record,
                "archive": archive_record,
                "hardware": hardware,
                "capacity_before": before_capacity,
                "capacity_after": capacity_snapshot(output),
            }
            completed.append(retained)
            append_event(
                output,
                journal,
                {
                    "state": "task-validated",
                    "task": task_spec(task),
                    "archive_complete": archive_record["archive_complete"],
                },
            )
            write_checkpoint(
                output, contract, hardware_before, capacity_before, completed
            )

        hardware_after = matrix.hardware_snapshot((gpu,))
        hardware = {"before": hardware_before, "after": hardware_after}
        matrix.validate_hardware_evidence(hardware, [gpu])
        for stable in code_stables.values():
            stable.verify_unchanged()
        receipt_stable.verify_unchanged()
        nvidia_stable.verify_unchanged()
        report = {
            "schema": SCHEMA,
            "outcome": "PASS",
            "contract": contract,
            "tasks": completed,
            "hardware": hardware,
            "capacity_initial": capacity_before,
            "capacity_final": capacity_snapshot(output),
            "resume_history": matrix.resume_history_records(output),
            "checkpoint": matrix.stable_file_record(
                output / "CHECKPOINT.json", output, "M3 paired checkpoint"
            ),
        }
        atomic_write_json(output / "report.json", report)
        publish_terminal(output, report)
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "outcome": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "failed_unix_seconds": time.time(),
            "partial_artifacts": matrix.collect_partial_evidence(output),
        }
        if (output / "JOURNAL.json").is_file():
            failure["journal"] = file_record(output / "JOURNAL.json", output)
        atomic_write_json(output / "FAILED.json", failure)
        raise
    finally:
        for stable in code_stables.values():
            stable.close(verify=False)
        if receipt_stable is not None:
            receipt_stable.close(verify=False)
        if nvidia_stable is not None:
            nvidia_stable.close(verify=False)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, WorkloadError, ValueError, json.JSONDecodeError) as error:
        print(f"M3 paired matrix error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
