#!/usr/bin/env python3
"""Run the four resumable CPU-only M3 host compatibility units."""

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
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import m3_execution_plan  # noqa: E402
import run_m3_host_oracle as host_oracle  # noqa: E402
import run_m3_paired_matrix as paired_matrix  # noqa: E402
import run_observed_process as observed  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
import run_validation  # noqa: E402
import verify_telemetry_v2 as telemetry_replay  # noqa: E402
from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    atomic_write_json,
    file_record,
)


SCHEMA = "gpmeep-m3-host-matrix-v1"
CHECKPOINT_SCHEMA = "gpmeep-m3-host-checkpoint-v1"
JOURNAL_SCHEMA = "gpmeep-m3-host-journal-v1"
COMPLETE_SCHEMA = "gpmeep-m3-host-matrix-complete-v1"
TASK_COUNT = 4
INITIAL_FREE_BYTES = 20 * 1024**3
PER_TASK_FREE_BYTES = 5 * 1024**3
MINIMUM_FREE_INODES = 100_000
MINIMUM_HOST_AVAILABLE_BYTES = 4 * 1024**3
MINIMUM_CGROUP_FREE_BYTES = 4 * 1024**3
MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES = 95_000.0
TELEMETRY_INTERVAL_SECONDS = 5.0
STDOUT_LIMIT_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class Task:
    index: int
    unit_id: str
    driver: str
    target_path: str
    covers_examples: tuple[str, ...]
    oracle_contract: str
    timeout_seconds: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--paired-output", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def build_tasks(plan: dict[str, Any]) -> list[Task]:
    rows = plan.get("host")
    if plan.get("schema") != m3_execution_plan.SCHEMA or not isinstance(rows, list):
        raise WorkloadError("M3 host execution plan schema differs")
    tasks = []
    for index, row in enumerate(rows, start=1):
        tasks.append(
            Task(
                index=index,
                unit_id=row["unit_id"],
                driver=row["driver"],
                target_path=row["target_path"],
                covers_examples=tuple(row["covers_examples"]),
                oracle_contract=row["oracle_contract"],
                timeout_seconds=row["timeout_seconds"],
            )
        )
    if (
        len(tasks) != TASK_COUNT
        or [task.index for task in tasks] != [1, 2, 3, 4]
        or {task.unit_id for task in tasks} != set(host_oracle.ORACLES)
    ):
        raise WorkloadError("M3 host task inventory differs")
    return tasks


def task_spec(task: Task) -> dict[str, Any]:
    return {
        "index": task.index,
        "unit_id": task.unit_id,
        "driver": task.driver,
        "target_path": task.target_path,
        "backend": "cpu",
        "covers_examples": list(task.covers_examples),
        "oracle_contract": task.oracle_contract,
        "timeout_seconds": task.timeout_seconds,
        "counts_toward_cuda_coverage": False,
        "counts_toward_gpu_speed": False,
    }


def task_name(task: Task) -> str:
    return f"{task.index:02d}-{task.unit_id}"


def task_output_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("units", task_name(task))


def task_runtime_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("runtime", task_name(task))


def task_log_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("logs", f"{task_name(task)}.log")


def task_telemetry_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("telemetry", task_name(task))


def _external_record(path: pathlib.Path, label: str) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        return {
            "path": str(stable.path),
            "size_bytes": stable.initial_stat.st_size,
            "sha256": stable.sha256,
        }


def paired_identity(
    paired_output: pathlib.Path, paired_report: dict[str, Any]
) -> dict[str, Any]:
    return {
        "path": str(paired_output),
        "complete": _external_record(
            paired_output / "PAIRED_COMPLETE", "M3 paired COMPLETE"
        ),
        "report": _external_record(
            paired_output / "report.json", "M3 paired report"
        ),
        "checkpoint": _external_record(
            paired_output / "CHECKPOINT.json", "M3 paired checkpoint"
        ),
        "receipt_id": paired_report["contract"]["build_receipt"]["receipt_id"],
        "source_sha256": paired_report["contract"]["build_receipt"][
            "source_sha256"
        ],
    }


def producer_identity(
    paired_output: pathlib.Path, paired_report: dict[str, Any]
) -> dict[str, Any]:
    matches = [
        record
        for record in paired_report["tasks"]
        if record["task"]["case_path"]
        == "python/examples/dipole_in_vacuum_1D.py"
    ]
    if len(matches) != 1:
        raise WorkloadError("M3 plot producer task inventory differs")
    archive = matches[0]["archive"]
    relative = pathlib.PurePosixPath(archive["path"])
    if (
        relative.is_absolute()
        or not relative.parts
        or "." in relative.parts
        or ".." in relative.parts
        or "\\" in archive["path"]
    ):
        raise WorkloadError("M3 plot producer archive record path is unsafe")
    path = paired_output.joinpath(*relative.parts).resolve(strict=True)
    if path.parent != (paired_output / "archives").resolve(strict=True):
        raise WorkloadError("M3 plot producer archive path differs")
    return {
        "path": str(path),
        "archive_manifest_sha256": archive["archive_manifest_sha256"],
        "receipt_id": archive["receipt_id"],
        "archive_complete": archive["archive_complete"],
    }


def _code_records(repo: pathlib.Path) -> tuple[dict[str, StableFile], dict[str, Any]]:
    paths = {
        "controller": pathlib.Path(__file__),
        "host_oracle": SCRIPT_DIR / "run_m3_host_oracle.py",
        "paired_controller": SCRIPT_DIR / "run_m3_paired_matrix.py",
        "plan_loader": SCRIPT_DIR / "m3_execution_plan.py",
        "validation_runtime": SCRIPT_DIR / "run_validation.py",
        "observed_runner": USER_WORKLOAD_DIR / "run_observed_process.py",
        "matrix_common": USER_WORKLOAD_DIR / "run_user_workload_matrix.py",
        "host_telemetry": USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py",
        "telemetry_replay": USER_WORKLOAD_DIR / "verify_telemetry_v2.py",
        "workload_common": USER_WORKLOAD_DIR / "common.py",
    }
    stables = {name: StableFile(path, f"M3 host {name}") for name, path in paths.items()}
    records = {
        name: matrix.stable_identity_record(stable, repo)
        for name, stable in stables.items()
    }
    return stables, records


def build_contract(
    paired_output: pathlib.Path,
    paired_report: dict[str, Any],
    plan: dict[str, Any],
    tasks: list[Task],
    runtime: dict[str, Any],
    code: dict[str, Any],
) -> dict[str, Any]:
    paired_contract = paired_report["contract"]
    return {
        "schema": SCHEMA,
        "paired": paired_identity(paired_output, paired_report),
        "producer": producer_identity(paired_output, paired_report),
        "repo": paired_contract["paths"]["repo"],
        "plan_root": paired_contract["paths"]["plan_root"],
        "python": paired_contract["paths"]["python"],
        "build_python": paired_contract["paths"]["build_python"],
        "install_prefix": paired_contract["paths"]["install_prefix"],
        "build_receipt": paired_contract["build_receipt"],
        "gpu_device": paired_contract["gpu_device"],
        "plan": plan,
        "tasks": [task_spec(task) for task in tasks],
        "runtime_contract": runtime,
        "evidence_code": code,
        "limits": {
            "stdout_limit_bytes": STDOUT_LIMIT_BYTES,
            "telemetry_interval_seconds": TELEMETRY_INTERVAL_SECONDS,
        },
        "gates": {
            "initial_free_bytes": INITIAL_FREE_BYTES,
            "per_task_free_bytes": PER_TASK_FREE_BYTES,
            "minimum_free_inodes": MINIMUM_FREE_INODES,
            "minimum_host_available_bytes": MINIMUM_HOST_AVAILABLE_BYTES,
            "minimum_cgroup_free_bytes": MINIMUM_CGROUP_FREE_BYTES,
            "maximum_host_temperature_millidegrees": (
                MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES
            ),
        },
    }


def validate_contract(
    contract: Any,
    *,
    verified_paired_report: dict[str, Any] | None = None,
) -> tuple[pathlib.Path, dict[str, Any], list[Task], str, dict[str, Any]]:
    keys = {
        "schema",
        "paired",
        "producer",
        "repo",
        "plan_root",
        "python",
        "build_python",
        "install_prefix",
        "build_receipt",
        "gpu_device",
        "plan",
        "tasks",
        "runtime_contract",
        "evidence_code",
        "limits",
        "gates",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != keys
        or contract.get("schema") != SCHEMA
    ):
        raise WorkloadError("M3 host sealed contract schema differs")
    paired_output = pathlib.Path(contract["paired"]["path"])
    paired_report = (
        paired_matrix.verify_complete(paired_output)
        if verified_paired_report is None
        else verified_paired_report
    )
    if contract["paired"] != paired_identity(paired_output, paired_report):
        raise WorkloadError("M3 host paired identity differs")
    repo = pathlib.Path(contract["repo"])
    plan_root = pathlib.Path(contract["plan_root"])
    if (
        not repo.is_absolute()
        or repo.resolve(strict=True) != repo
        or not plan_root.is_absolute()
        or plan_root.resolve(strict=True) != plan_root
    ):
        raise WorkloadError("M3 host sealed root path differs")
    plan = m3_execution_plan.load_release_plan(plan_root, repo)
    tasks = build_tasks(plan)
    if contract["plan"] != plan or contract["tasks"] != [
        task_spec(task) for task in tasks
    ]:
        raise WorkloadError("M3 host plan was not re-derived")
    if contract["producer"] != producer_identity(paired_output, paired_report):
        raise WorkloadError("M3 host producer identity differs")
    paired_contract = paired_report["contract"]
    for name in (
        "repo",
        "plan_root",
        "python",
        "build_python",
        "install_prefix",
    ):
        expected = paired_contract["paths"].get(name, paired_contract.get(name))
        if contract[name] != expected:
            raise WorkloadError(f"M3 host paired runtime path differs: {name}")
    if (
        contract["build_receipt"] != paired_contract["build_receipt"]
        or contract["gpu_device"] != paired_contract["gpu_device"]
    ):
        raise WorkloadError("M3 host build or GPU identity differs from paired")
    gpu = paired_matrix.normalized_gpu(contract["gpu_device"])
    if gpu != contract["gpu_device"]:
        raise WorkloadError("M3 host GPU UUID spelling differs")
    runtime_snapshot = run_validation.validation_snapshot(
        repo,
        pathlib.Path(contract["build_python"]),
        pathlib.Path(contract["python"]),
    )
    if not runtime_snapshot.get("available"):
        raise WorkloadError(
            "M3 host runtime replay failed: "
            + "; ".join(runtime_snapshot.get("problems", []))
        )
    runtime = runtime_snapshot["runtime_contract"]
    if contract["runtime_contract"] != runtime:
        raise WorkloadError("M3 host runtime contract differs")
    expected_code = {
        "controller": pathlib.Path(__file__),
        "host_oracle": SCRIPT_DIR / "run_m3_host_oracle.py",
        "paired_controller": SCRIPT_DIR / "run_m3_paired_matrix.py",
        "plan_loader": SCRIPT_DIR / "m3_execution_plan.py",
        "validation_runtime": SCRIPT_DIR / "run_validation.py",
        "observed_runner": USER_WORKLOAD_DIR / "run_observed_process.py",
        "matrix_common": USER_WORKLOAD_DIR / "run_user_workload_matrix.py",
        "host_telemetry": USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py",
        "telemetry_replay": USER_WORKLOAD_DIR / "verify_telemetry_v2.py",
        "workload_common": USER_WORKLOAD_DIR / "common.py",
    }
    code = contract["evidence_code"]
    if not isinstance(code, dict) or set(code) != set(expected_code):
        raise WorkloadError("M3 host evidence-code inventory differs")
    for name, expected_path in expected_code.items():
        stable = matrix.stable_record(repo, code[name], f"M3 host {name}")
        try:
            if stable.path != expected_path.resolve(strict=True):
                raise WorkloadError(f"M3 host code path differs: {name}")
        finally:
            stable.close()
    if contract["limits"] != {
        "stdout_limit_bytes": STDOUT_LIMIT_BYTES,
        "telemetry_interval_seconds": TELEMETRY_INTERVAL_SECONDS,
    }:
        raise WorkloadError("M3 host sealed limits differ")
    if contract["gates"] != {
        "initial_free_bytes": INITIAL_FREE_BYTES,
        "per_task_free_bytes": PER_TASK_FREE_BYTES,
        "minimum_free_inodes": MINIMUM_FREE_INODES,
        "minimum_host_available_bytes": MINIMUM_HOST_AVAILABLE_BYTES,
        "minimum_cgroup_free_bytes": MINIMUM_CGROUP_FREE_BYTES,
        "maximum_host_temperature_millidegrees": (
            MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES
        ),
    }:
        raise WorkloadError("M3 host sealed gates differ")
    return repo, plan, tasks, gpu, paired_report


def task_command(
    task: Task, contract: dict[str, Any], output: pathlib.Path
) -> list[str]:
    command = [
        contract["python"],
        str(SCRIPT_DIR / "run_m3_host_oracle.py"),
        "--repo",
        contract["repo"],
        "--output",
        str(output.joinpath(*task_output_relative(task).parts)),
        "--unit-id",
        task.unit_id,
    ]
    if task.unit_id == "host-plot-radiation-pattern":
        command.extend(
            (
                "--producer-archive",
                contract["producer"]["path"],
                "--expected-producer-manifest-sha256",
                contract["producer"]["archive_manifest_sha256"],
                "--expected-receipt-id",
                contract["producer"]["receipt_id"],
            )
        )
    return command


def clean_environment(
    contract: dict[str, Any], runtime_root: pathlib.Path
) -> dict[str, str]:
    runtime_root.mkdir(parents=True, exist_ok=False)
    for name in ("home", "cache", "tmp", "matplotlib"):
        (runtime_root / name).mkdir()
    runtime = contract["runtime_contract"]
    libmeep = pathlib.Path(runtime["libmeep"]["path"])
    installed = pathlib.Path(runtime["installed_environment"])
    python = pathlib.Path(contract["python"])
    return {
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "PYTHONPATH": f"{SCRIPT_DIR}:{contract['build_python']}",
        "LD_LIBRARY_PATH": f"{libmeep.parent}:{installed / 'lib'}",
        "HOME": str(runtime_root / "home"),
        "XDG_CACHE_HOME": str(runtime_root / "cache"),
        "TMPDIR": str(runtime_root / "tmp"),
        "MPLCONFIGDIR": str(runtime_root / "matplotlib"),
        "MPLBACKEND": "Agg",
        "FONTCONFIG_FILE": runtime["fontconfig_file"]["path"],
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "MEEP_GPU_BACKEND": "cpu",
        "GPMEEP_VALIDATION_EXPECTED_BACKEND": "cpu",
        "JAX_PLATFORMS": "cpu",
        "CUDA_CACHE_DISABLE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def observer_spec(
    task: Task, root_pid: int, output: pathlib.Path, contract: dict[str, Any]
) -> observed.ObserverSpec:
    telemetry = output.joinpath(*task_telemetry_relative(task).parts)
    return observed.ObserverSpec(
        "host",
        (
            contract["python"],
            str(USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py"),
            "--root-pid",
            str(root_pid),
            "--output",
            str(telemetry / "host.jsonl"),
            "--label",
            task_name(task),
            "--interval-seconds",
            repr(TELEMETRY_INTERVAL_SECONDS),
        ),
        telemetry / "host-monitor.log",
    )


def derive_telemetry(
    task: Task,
    process: dict[str, Any],
    output: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    telemetry = output.joinpath(*task_telemetry_relative(task).parts)
    return telemetry_replay.verify_pair(
        host_stream=telemetry / "host.jsonl",
        host_terminal=telemetry / "host.jsonl.COMPLETE.json",
        gpu_stream=None,
        gpu_terminal=None,
        expected_root_pid=process["command_pid"],
        expected_label=task_name(task),
        expected_host_monitor_sha256=contract["evidence_code"]["host_telemetry"][
            "sha256"
        ],
        minimum_host_available_bytes=MINIMUM_HOST_AVAILABLE_BYTES,
        minimum_cgroup_free_bytes=MINIMUM_CGROUP_FREE_BYTES,
        maximum_host_temperature_millidegrees=(
            MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES
        ),
    )


def publish_telemetry(
    task: Task,
    process: dict[str, Any],
    observer_records: list[dict[str, Any]],
    output: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    spec = observer_spec(task, process["command_pid"], output, contract)
    if len(observer_records) != 1:
        raise WorkloadError("M3 host observer inventory differs")
    observed.verify_observer_record(output, observer_records[0], spec)
    report = derive_telemetry(task, process, output, contract)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    atomic_write_json(path, report)
    return matrix.stable_file_record(path, output, "M3 host telemetry replay")


def validate_telemetry(
    task: Task,
    process: dict[str, Any],
    observer_records: list[dict[str, Any]],
    retained: Any,
    output: pathlib.Path,
    contract: dict[str, Any],
) -> None:
    spec = observer_spec(task, process["command_pid"], output, contract)
    if len(observer_records) != 1:
        raise WorkloadError("M3 host observer inventory differs")
    observed.verify_observer_record(output, observer_records[0], spec)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    expected = matrix.stable_file_record(path, output, "M3 host telemetry replay")
    if retained != expected:
        raise WorkloadError("M3 host telemetry replay record differs")
    with StableFile(path, "M3 host telemetry replay") as stable:
        with stable.file_object() as handle:
            report = json.load(handle)
    if report != derive_telemetry(task, process, output, contract):
        raise WorkloadError("M3 host telemetry was not re-derived")


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
    required = {"path", "free_bytes", "total_bytes", "free_inodes", "total_inodes"}
    if (
        not isinstance(record, dict)
        or set(record) != required
        or not isinstance(record["path"], str)
        or not pathlib.Path(record["path"]).is_absolute()
        or any(
            type(record[name]) is not int or record[name] < 0
            for name in ("free_bytes", "total_bytes", "free_inodes", "total_inodes")
        )
        or record["free_bytes"] > record["total_bytes"]
        or record["free_inodes"] > record["total_inodes"]
    ):
        raise WorkloadError("M3 host capacity record differs")
    minimum = INITIAL_FREE_BYTES if initial else PER_TASK_FREE_BYTES
    if record["free_bytes"] < minimum or record["free_inodes"] < MINIMUM_FREE_INODES:
        raise WorkloadError("M3 host capacity gate failed")


def _plot_kwargs(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "producer_archive": pathlib.Path(contract["producer"]["path"]),
        "expected_manifest_sha256": contract["producer"][
            "archive_manifest_sha256"
        ],
        "expected_receipt_id": contract["producer"]["receipt_id"],
    }


def derive_unit_output(
    task: Task, output: pathlib.Path, contract: dict[str, Any]
) -> dict[str, Any]:
    unit_output = output.joinpath(*task_output_relative(task).parts)
    kwargs = _plot_kwargs(contract) if task.unit_id == "host-plot-radiation-pattern" else {}
    host_oracle.validate_complete(
        unit_output, pathlib.Path(contract["repo"]), task.unit_id, **kwargs
    )
    return {
        "path": task_output_relative(task).as_posix(),
        "complete": matrix.stable_file_record(
            unit_output / "COMPLETE", output, "M3 host unit COMPLETE"
        ),
        "report": matrix.stable_file_record(
            unit_output / "report.json", output, "M3 host unit report"
        ),
    }


def validate_unit_output(
    task: Task, retained: Any, output: pathlib.Path, contract: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(retained, dict) or set(retained) != {"path", "complete", "report"}:
        raise WorkloadError("M3 host unit-output record differs")
    derived = derive_unit_output(task, output, contract)
    if retained != derived:
        raise WorkloadError("M3 host unit output was not re-derived")
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
        raise WorkloadError("M3 host journal identity differs")
    first = journal["events"][0]
    if first.get("state") != "controller-started" or not math.isfinite(
        float(first.get("unix_seconds", math.nan))
    ):
        raise WorkloadError("M3 host journal start differs")
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
                    "complete": record["output"]["complete"],
                },
            )
        )
    if journal["events"][: len(expected)] != expected:
        raise WorkloadError("M3 host journal disagrees with checkpoint")
    return expected


def validate_completed_task(
    task: Task,
    record: Any,
    output: pathlib.Path,
    contract: dict[str, Any],
    gpu: str,
) -> dict[str, Any]:
    keys = {
        "task",
        "process",
        "observers",
        "telemetry_replay",
        "output",
        "hardware",
        "capacity_before",
        "capacity_after",
    }
    if not isinstance(record, dict) or set(record) != keys:
        raise WorkloadError("M3 host completed-task record differs")
    if record["task"] != task_spec(task):
        raise WorkloadError("M3 host completed-task order differs")
    matrix._verify_process_record(
        output,
        record["process"],
        task_log_relative(task).as_posix(),
        task_name(task),
    )
    if record["process"]["command"] != task_command(task, contract, output):
        raise WorkloadError("M3 host task command differs")
    validate_telemetry(
        task,
        record["process"],
        record["observers"],
        record["telemetry_replay"],
        output,
        contract,
    )
    matrix.validate_hardware_evidence(record["hardware"], [gpu])
    validate_capacity(record["capacity_before"], False)
    validate_capacity(record["capacity_after"], False)
    validate_unit_output(task, record["output"], output, contract)
    return record


def _load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        with stable.file_object() as handle:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return value


def _load_checkpoint(output: pathlib.Path, contract: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _load_json(output / "CHECKPOINT.json", "M3 host checkpoint")
    if set(checkpoint) != {
        "schema",
        "contract",
        "hardware_before",
        "capacity_before",
        "completed_tasks",
    } or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise WorkloadError("M3 host checkpoint schema differs")
    if checkpoint["contract"] != contract:
        raise WorkloadError("M3 host resume invocation differs")
    if not isinstance(checkpoint["completed_tasks"], list):
        raise WorkloadError("M3 host checkpoint task inventory differs")
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
    output: pathlib.Path, terminal_name: str = "HOST_COMPLETE"
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {"HOST_COMPLETE", "PENDING_HOST_COMPLETE"}:
        raise WorkloadError("M3 host terminal filename differs")
    complete = _load_json(output / terminal_name, "M3 host matrix COMPLETE")
    if (
        set(complete) != {"schema", "outcome", "report", "journal", "checkpoint"}
        or complete.get("schema") != COMPLETE_SCHEMA
        or complete.get("outcome") != "PASS"
    ):
        raise WorkloadError("M3 host matrix COMPLETE is not an exact PASS")
    report = matrix.load_stable_json_record(
        output, complete["report"], "M3 host matrix report"
    )
    journal = matrix.load_stable_json_record(
        output, complete["journal"], "M3 host matrix journal"
    )
    checkpoint = matrix.load_stable_json_record(
        output, complete["checkpoint"], "M3 host matrix checkpoint"
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
    if (
        not isinstance(report, dict)
        or set(report) != report_keys
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
    ):
        raise WorkloadError("M3 host matrix report differs")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint)
        != {"schema", "contract", "hardware_before", "capacity_before", "completed_tasks"}
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
    ):
        raise WorkloadError("M3 host matrix checkpoint differs")
    contract = checkpoint["contract"]
    _repo, _plan, tasks, gpu, _paired = validate_contract(contract)
    if report["contract"] != contract:
        raise WorkloadError("M3 host report contract differs")
    completed = checkpoint["completed_tasks"]
    if (
        not isinstance(completed, list)
        or len(completed) != TASK_COUNT
        or report["tasks"] != completed
    ):
        raise WorkloadError("M3 host completed-task inventory differs")
    for task, record in zip(tasks, completed, strict=True):
        validate_completed_task(task, record, output, contract, gpu)
    if expected_journal(journal, contract, completed) != journal["events"]:
        raise WorkloadError("M3 host journal has unsealed events")
    matrix.validate_hardware_evidence(report["hardware"], [gpu])
    if checkpoint["hardware_before"] != report["hardware"]["before"]:
        raise WorkloadError("M3 host initial hardware differs")
    if checkpoint["capacity_before"] != report["capacity_initial"]:
        raise WorkloadError("M3 host initial capacity differs")
    validate_capacity(report["capacity_initial"], True)
    validate_capacity(report["capacity_final"], False)
    if report["resume_history"] != matrix.resume_history_records(output):
        raise WorkloadError("M3 host resume history differs")
    if complete != _terminal_payload(output, report):
        raise WorkloadError("M3 host terminal payload differs")
    return report


def publish_terminal(output: pathlib.Path, report: dict[str, Any]) -> None:
    atomic_write_json(
        output / "PENDING_HOST_COMPLETE", _terminal_payload(output, report)
    )
    verify_complete(output, "PENDING_HOST_COMPLETE")
    os.replace(output / "PENDING_HOST_COMPLETE", output / "HOST_COMPLETE")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paired_output = args.paired_output.resolve(strict=True)
    paired_report = paired_matrix.verify_complete(paired_output)
    paired_contract = paired_report["contract"]
    repo = pathlib.Path(paired_contract["paths"]["repo"])
    plan_root = pathlib.Path(paired_contract["paths"]["plan_root"])
    output_absolute = pathlib.Path(os.path.abspath(args.output))
    if any(
        paired_matrix._paths_overlap(output_absolute, root)
        for root in (repo, plan_root, paired_output)
    ):
        raise WorkloadError("M3 host output overlaps an immutable input root")
    if args.resume:
        if args.output.is_symlink():
            raise WorkloadError("M3 host resume output is a symlink")
        output = args.output.resolve(strict=True)
        if (output / "HOST_COMPLETE").exists():
            raise WorkloadError("completed M3 host matrix cannot be resumed")
    else:
        if args.output.is_symlink():
            raise WorkloadError("M3 host output is a symlink")
        output = output_absolute
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.parent.resolve(strict=True) != output.parent:
            raise WorkloadError("M3 host output parent contains a symlink")
        output.mkdir(exist_ok=False)
    lock_path = output / "LOCK"
    if args.resume and not stat.S_ISREG(lock_path.lstat().st_mode):
        raise WorkloadError("M3 host lock is not a regular file")
    lock = lock_path.open("r+" if args.resume else "x")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise WorkloadError(f"could not acquire M3 host lock: {exc}") from exc

    code_stables: dict[str, StableFile] = {}
    paired_stables: list[StableFile] = []
    try:
        plan = m3_execution_plan.load_release_plan(plan_root, repo)
        tasks = build_tasks(plan)
        snapshot = run_validation.validation_snapshot(
            repo,
            pathlib.Path(paired_contract["paths"]["build_python"]),
            pathlib.Path(paired_contract["paths"]["python"]),
        )
        if not snapshot.get("available"):
            raise WorkloadError(
                "M3 host runtime preflight failed: "
                + "; ".join(snapshot.get("problems", []))
            )
        code_stables, code = _code_records(repo)
        for name in ("PAIRED_COMPLETE", "report.json", "CHECKPOINT.json"):
            paired_stables.append(StableFile(paired_output / name, f"M3 paired {name}"))
        contract = build_contract(
            paired_output,
            paired_report,
            plan,
            tasks,
            snapshot["runtime_contract"],
            code,
        )
        validate_contract(contract, verified_paired_report=paired_report)
        gpu = paired_matrix.normalized_gpu(contract["gpu_device"])
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
                raise WorkloadError("M3 host checkpoint has too many tasks")
            for task, record in zip(
                tasks, checkpoint["completed_tasks"], strict=False
            ):
                completed.append(
                    validate_completed_task(task, record, output, contract, gpu)
                )
            journal = _load_json(output / "JOURNAL.json", "M3 host journal")
            prefix = expected_journal(journal, contract, completed)
            partial = [pathlib.PurePosixPath("PENDING_HOST_COMPLETE")]
            if len(completed) < TASK_COUNT:
                task = tasks[len(completed)]
                partial.extend(
                    (
                        task_output_relative(task),
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
            capacity_before_task = capacity_snapshot(output)
            validate_capacity(capacity_before_task, False)
            hardware_before_task = matrix.hardware_snapshot((gpu,))
            runtime = output.joinpath(*task_runtime_relative(task).parts)
            environment = clean_environment(contract, runtime)
            append_event(
                output, journal, {"state": "task-started", "task": task_spec(task)}
            )
            process, observers = observed.run_bounded_observed(
                task_command(task, contract, output),
                environment,
                output,
                output.joinpath(*task_log_relative(task).parts),
                task.timeout_seconds,
                STDOUT_LIMIT_BYTES,
                lambda pid, selected=task: [
                    observer_spec(selected, pid, output, contract)
                ],
                observer_environment=environment,
                observer_exit_timeout_seconds=(
                    TELEMETRY_INTERVAL_SECONDS * 3 + 30
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
                raise WorkloadError(f"M3 host task failed: {task_name(task)}")
            hardware_after_task = matrix.hardware_snapshot((gpu,))
            hardware = {
                "before": hardware_before_task,
                "after": hardware_after_task,
            }
            matrix.validate_hardware_evidence(hardware, [gpu])
            telemetry_record = publish_telemetry(
                task, process, observers, output, contract
            )
            output_record = derive_unit_output(task, output, contract)
            record = {
                "task": task_spec(task),
                "process": process,
                "observers": observers,
                "telemetry_replay": telemetry_record,
                "output": output_record,
                "hardware": hardware,
                "capacity_before": capacity_before_task,
                "capacity_after": capacity_snapshot(output),
            }
            completed.append(record)
            append_event(
                output,
                journal,
                {
                    "state": "task-validated",
                    "task": task_spec(task),
                    "complete": output_record["complete"],
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
        for stable in paired_stables:
            stable.verify_unchanged()
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
                output / "CHECKPOINT.json", output, "M3 host checkpoint"
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
        for stable in paired_stables:
            stable.close(verify=False)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, WorkloadError, ValueError, json.JSONDecodeError) as error:
        print(f"M3 host matrix error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
