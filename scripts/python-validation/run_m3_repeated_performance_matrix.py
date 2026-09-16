#!/usr/bin/env python3
"""Run the resumable 72-sample M3 repeated-example performance matrix."""

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
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DEFAULT = SCRIPT_DIR.parents[1]
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import gpmeep_provenance as provenance  # noqa: E402
import m3_performance_plan as performance_plan  # noqa: E402
import m3_repeated_performance_replay as performance_replay  # noqa: E402
import run_m3_performance_sample as performance_sample  # noqa: E402
import run_observed_process as observed  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
import run_validation as validation  # noqa: E402
import verify_telemetry_v2 as telemetry_replay  # noqa: E402
from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    file_record,
)


SCHEMA = "gpmeep-m3-repeated-performance-matrix-v1"
CHECKPOINT_SCHEMA = "gpmeep-m3-repeated-performance-checkpoint-v1"
JOURNAL_SCHEMA = "gpmeep-m3-repeated-performance-journal-v1"
COMPLETE_SCHEMA = "gpmeep-m3-repeated-performance-complete-v1"
TASK_COUNT = 72
INITIAL_FREE_BYTES = 40 * 1024**3
PER_TASK_FREE_BYTES = 5 * 1024**3
MINIMUM_FREE_INODES = 100_000
MINIMUM_HOST_AVAILABLE_BYTES = 4 * 1024**3
MINIMUM_CGROUP_FREE_BYTES = 4 * 1024**3
MINIMUM_GPU_FREE_MEMORY_FRACTION = 0.20
MAXIMUM_GPU_TEMPERATURE_CELSIUS = 90.0
MAXIMUM_HOST_TEMPERATURE_MILLIDEGREES = 95_000.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument("--plan-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--python", required=True, type=pathlib.Path)
    parser.add_argument("--build-python", required=True, type=pathlib.Path)
    parser.add_argument("--install-prefix", required=True, type=pathlib.Path)
    parser.add_argument("--build-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--mpiexec", required=True, type=pathlib.Path)
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument("--stdout-limit-mib", type=int, default=64)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.stdout_limit_mib <= 1024:
        raise WorkloadError("M3 repeated stdout limit must be in [1,1024] MiB")
    if (
        isinstance(args.telemetry_interval_seconds, bool)
        or not isinstance(args.telemetry_interval_seconds, (int, float))
        or not math.isfinite(float(args.telemetry_interval_seconds))
        or not 1.0 <= args.telemetry_interval_seconds <= 60.0
    ):
        raise WorkloadError("M3 repeated telemetry interval must be in [1s,60s]")


def normalized_devices(value: str) -> tuple[str, str]:
    return performance_sample.normalized_devices(value)


def task_raw_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath(
        "samples", performance_replay.task_name(task)
    )


def task_runtime_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath(
        "runtime", performance_replay.task_name(task)
    )


def task_log_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath(
        "logs", f"{performance_replay.task_name(task)}.log"
    )


def task_telemetry_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath(
        "telemetry", performance_replay.task_name(task)
    )


def task_command(
    task: dict[str, Any], contract: dict[str, Any], output: pathlib.Path
) -> list[str]:
    paths = contract["paths"]
    return [
        paths["python"],
        paths["sample_runner"],
        "--repo",
        paths["repo"],
        "--manifest",
        paths["manifest"],
        "--output",
        str(output.joinpath(*task_raw_relative(task).parts)),
        "--case",
        task["target"],
        "--topology",
        task["topology"],
        "--sample-kind",
        task["sample_kind"],
        "--cycle-index",
        str(task["cycle_index"]),
        "--python",
        paths["python"],
        "--build-python",
        paths["build_python"],
        "--install-prefix",
        paths["install_prefix"],
        "--build-receipt",
        paths["build_receipt"],
        "--mpiexec",
        paths["mpiexec"],
        "--gpu-devices",
        ",".join(contract["gpu_devices"]),
        "--timeout-seconds",
        str(task["timeout_seconds"]),
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
    if (
        not isinstance(record, dict)
        or set(record)
        != {"path", "free_bytes", "total_bytes", "free_inodes", "total_inodes"}
        or not isinstance(record["path"], str)
        or not pathlib.Path(record["path"]).is_absolute()
        or any(
            type(record[name]) is not int or record[name] < 0
            for name in record
            if name != "path"
        )
        or record["free_bytes"] > record["total_bytes"]
        or record["free_inodes"] > record["total_inodes"]
    ):
        raise WorkloadError("M3 repeated capacity record is invalid")
    if record["free_bytes"] < (
        INITIAL_FREE_BYTES if initial else PER_TASK_FREE_BYTES
    ):
        raise WorkloadError("M3 repeated free-space gate failed")
    if record["free_inodes"] < MINIMUM_FREE_INODES:
        raise WorkloadError("M3 repeated free-inode gate failed")


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
        "CUDA_VISIBLE_DEVICES": ",".join(contract["gpu_devices"]),
    }


def _code_paths() -> dict[str, pathlib.Path]:
    return {
        "controller": pathlib.Path(__file__),
        "plan_loader": SCRIPT_DIR / "m3_performance_plan.py",
        "statistics_replay": SCRIPT_DIR / "m3_repeated_performance_replay.py",
        "sample_runner": SCRIPT_DIR / "run_m3_performance_sample.py",
        "automatic_common": SCRIPT_DIR / "run_m3_auto_example_case.py",
        "mpi_example_common": SCRIPT_DIR / "run_m3_mpi_example_case.py",
        "mpi_rank_launcher": SCRIPT_DIR / "m3_mpi_rank_launcher.py",
        "example_oracle": SCRIPT_DIR / "run_example_oracle.py",
        "validation_runner": SCRIPT_DIR / "run_validation.py",
        "manifest": SCRIPT_DIR / "manifest.json",
        "provenance": SCRIPT_DIR.parent / "gpmeep_provenance.py",
        "observed_runner": USER_WORKLOAD_DIR / "run_observed_process.py",
        "matrix_common": USER_WORKLOAD_DIR / "run_user_workload_matrix.py",
        "host_telemetry": USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py",
        "gpu_telemetry": USER_WORKLOAD_DIR / "capture_gpu_telemetry_v2.py",
        "telemetry_replay": USER_WORKLOAD_DIR / "verify_telemetry_v2.py",
        "workload_common": USER_WORKLOAD_DIR / "common.py",
    }


def _code_records(
    repo: pathlib.Path,
) -> tuple[dict[str, StableFile], dict[str, dict[str, Any]]]:
    stables = {
        name: StableFile(path, f"M3 repeated {name}")
        for name, path in _code_paths().items()
    }
    return stables, {
        name: matrix.stable_identity_record(stable, repo)
        for name, stable in stables.items()
    }


def _receipt_record(
    receipt_path: pathlib.Path, receipt: dict[str, Any]
) -> dict[str, Any]:
    identifiers = ("receipt_id", "build_input_id", "artifact_set_id")
    if any(not isinstance(receipt.get(name), str) for name in identifiers):
        raise WorkloadError("M3 repeated build receipt IDs are invalid")
    source_sha256 = receipt.get("source_start", {}).get("sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise WorkloadError("M3 repeated build source identity is invalid")
    return {
        "path": str(receipt_path),
        "size_bytes": receipt_path.stat().st_size,
        "sha256": provenance.sha256_file(receipt_path),
        "receipt_id": receipt["receipt_id"],
        "build_input_id": receipt["build_input_id"],
        "artifact_set_id": receipt["artifact_set_id"],
        "source_sha256": source_sha256,
    }


def _runtime_paths(args: argparse.Namespace, repo: pathlib.Path) -> dict[str, str]:
    return {
        "repo": str(repo),
        "plan_root": str(args.plan_root.resolve(strict=True)),
        "python": str(args.python.resolve(strict=True)),
        "build_python": str(args.build_python.resolve(strict=True)),
        "install_prefix": str(args.install_prefix.resolve(strict=True)),
        "build_receipt": str(args.build_receipt.resolve(strict=True)),
        "mpiexec": str(args.mpiexec.resolve(strict=True)),
        "manifest": str((SCRIPT_DIR / "manifest.json").resolve(strict=True)),
        "sample_runner": str(
            (SCRIPT_DIR / "run_m3_performance_sample.py").resolve(strict=True)
        ),
    }


def _authoritative_paths(
    repo: pathlib.Path, plan_root: pathlib.Path
) -> dict[str, pathlib.Path]:
    return {
        "repo": repo,
        "plan_root": plan_root,
        "python": (
            repo / ".envs/meep-gpu-cuda-mpi/bin/python3.11"
        ).resolve(strict=True),
        "build_python": (
            repo / "build/meep-cuda-mpi-python-fp32/python"
        ).resolve(strict=True),
        "install_prefix": (
            repo / "install/meep-cuda-mpi-python-fp32"
        ).resolve(strict=True),
        "build_receipt": (
            repo / "build/meep-cuda-mpi-python-fp32/build-provenance.json"
        ).resolve(strict=True),
        "mpiexec": (
            repo / ".envs/meep-gpu-cuda-mpi/bin/mpirun"
        ).resolve(strict=True),
        "manifest": (SCRIPT_DIR / "manifest.json").resolve(strict=True),
        "sample_runner": (
            SCRIPT_DIR / "run_m3_performance_sample.py"
        ).resolve(strict=True),
    }


def build_contract(
    args: argparse.Namespace,
    repo: pathlib.Path,
    plan: dict[str, Any],
    tasks: list[dict[str, Any]],
    code: dict[str, Any],
    receipt: dict[str, Any],
    devices: tuple[str, str],
    nvidia_smi: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "paths": _runtime_paths(args, repo),
        "plan": plan,
        "tasks": tasks,
        "evidence_code": code,
        "build_receipt": receipt,
        "gpu_devices": list(devices),
        "nvidia_smi": nvidia_smi,
        "limits": {
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


def validate_contract(
    contract: Any,
) -> tuple[pathlib.Path, dict[str, Any], list[dict[str, Any]], tuple[str, str]]:
    keys = {
        "schema",
        "paths",
        "plan",
        "tasks",
        "evidence_code",
        "build_receipt",
        "gpu_devices",
        "nvidia_smi",
        "limits",
        "gates",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != keys
        or contract.get("schema") != SCHEMA
    ):
        raise WorkloadError("M3 repeated sealed contract schema differs")
    paths = contract["paths"]
    expected_paths = {
        "repo",
        "plan_root",
        "python",
        "build_python",
        "install_prefix",
        "build_receipt",
        "mpiexec",
        "manifest",
        "sample_runner",
    }
    if (
        not isinstance(paths, dict)
        or set(paths) != expected_paths
        or any(not isinstance(paths[name], str) for name in expected_paths)
    ):
        raise WorkloadError("M3 repeated sealed path inventory differs")
    resolved = {name: pathlib.Path(value) for name, value in paths.items()}
    if any(
        not path.is_absolute() or path.resolve(strict=True) != path
        for path in resolved.values()
    ):
        raise WorkloadError("M3 repeated runtime path is not canonical")
    directories = {"repo", "plan_root", "build_python", "install_prefix"}
    if any(
        (name in directories) != path.is_dir()
        for name, path in resolved.items()
    ):
        raise WorkloadError("M3 repeated runtime path type differs")
    if any(
        not os.access(resolved[name], os.X_OK) for name in ("python", "mpiexec")
    ):
        raise WorkloadError("M3 repeated runtime executable differs")
    if resolved != _authoritative_paths(resolved["repo"], resolved["plan_root"]):
        raise WorkloadError("M3 repeated runtime is not authoritative")
    plan = performance_plan.load_release_plan(
        resolved["plan_root"], resolved["repo"]
    )
    tasks = performance_replay.build_tasks(plan)
    if contract["plan"] != plan or contract["tasks"] != tasks:
        raise WorkloadError("M3 repeated sealed plan was not re-derived")
    code = contract["evidence_code"]
    expected_code = _code_paths()
    if not isinstance(code, dict) or set(code) != set(expected_code):
        raise WorkloadError("M3 repeated evidence-code inventory differs")
    for name, expected_path in expected_code.items():
        stable = matrix.stable_record(
            resolved["repo"], code[name], f"M3 repeated {name}"
        )
        try:
            if stable.path != expected_path.resolve(strict=True):
                raise WorkloadError(f"M3 repeated code path differs: {name}")
        finally:
            stable.close()
    try:
        receipt = provenance.verify_build_receipt(
            resolved["build_receipt"], resolved["repo"]
        )
    except provenance.ProvenanceError as exc:
        raise WorkloadError(f"M3 repeated receipt replay failed: {exc}") from exc
    if contract["build_receipt"] != _receipt_record(
        resolved["build_receipt"], receipt
    ):
        raise WorkloadError("M3 repeated receipt record differs")
    devices = normalized_devices(",".join(contract["gpu_devices"]))
    if list(devices) != contract["gpu_devices"]:
        raise WorkloadError("M3 repeated GPU inventory differs")
    if contract["nvidia_smi"] != absolute_file_record(
        pathlib.Path("/usr/bin/nvidia-smi"), "nvidia-smi"
    ):
        raise WorkloadError("M3 repeated nvidia-smi identity differs")
    limits = contract["limits"]
    if (
        not isinstance(limits, dict)
        or set(limits) != {"stdout_limit_bytes", "telemetry_interval_seconds"}
        or type(limits["stdout_limit_bytes"]) is not int
        or not 1024**2 <= limits["stdout_limit_bytes"] <= 1024**3
        or limits["stdout_limit_bytes"] % 1024**2
        or isinstance(limits["telemetry_interval_seconds"], bool)
        or not isinstance(limits["telemetry_interval_seconds"], (int, float))
        or not math.isfinite(float(limits["telemetry_interval_seconds"]))
        or not 1.0 <= limits["telemetry_interval_seconds"] <= 60.0
    ):
        raise WorkloadError("M3 repeated sealed limits differ")
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
        raise WorkloadError("M3 repeated sealed gates differ")
    return resolved["repo"], plan, tasks, devices


def observer_specs(
    task: dict[str, Any],
    root_pid: int,
    output: pathlib.Path,
    contract: dict[str, Any],
) -> list[observed.ObserverSpec]:
    root = output.joinpath(*task_telemetry_relative(task).parts)
    interval = contract["limits"]["telemetry_interval_seconds"]
    python = contract["paths"]["python"]
    label = performance_replay.task_name(task)
    specs = [
        observed.ObserverSpec(
            "host",
            (
                python,
                str(USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py"),
                "--root-pid",
                str(root_pid),
                "--output",
                str(root / "host.jsonl"),
                "--label",
                label,
                "--interval-seconds",
                repr(interval),
            ),
            root / "host-monitor.log",
        )
    ]
    selected_devices = task_gpu_devices(task, contract)
    if selected_devices:
        selected_flags = tuple(
            item
            for device in selected_devices
            for item in ("--selected-uuid", device)
        )
        specs.append(
            observed.ObserverSpec(
                "gpu",
                (
                    python,
                    str(USER_WORKLOAD_DIR / "capture_gpu_telemetry_v2.py"),
                    "--root-pid",
                    str(root_pid),
                    "--output",
                    str(root / "gpu.jsonl"),
                    "--label",
                    label,
                    *selected_flags,
                    "--interval-seconds",
                    repr(interval),
                ),
                root / "gpu-monitor.log",
            )
        )
    return specs


def task_gpu_devices(
    task: dict[str, Any], contract: dict[str, Any]
) -> tuple[str, ...]:
    devices = tuple(contract["gpu_devices"])
    topology = task["topology"]
    if topology == "cuda1" or (
        topology == "auto1"
        and task["unit_id"] == "performance-metasurface-crossover"
    ):
        return devices[:1]
    if topology == "cuda2":
        return devices
    return ()


def _code_sha256(contract: dict[str, Any], name: str) -> str:
    value = contract.get("evidence_code", {}).get(name, {}).get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise WorkloadError(f"M3 repeated code digest differs: {name}")
    return value


def derive_telemetry(
    task: dict[str, Any],
    process: dict[str, Any],
    output: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    root = output.joinpath(*task_telemetry_relative(task).parts)
    gates = contract["gates"]
    selected_devices = task_gpu_devices(task, contract)
    return telemetry_replay.verify_pair(
        host_stream=root / "host.jsonl",
        host_terminal=root / "host.jsonl.COMPLETE.json",
        gpu_stream=(root / "gpu.jsonl") if selected_devices else None,
        gpu_terminal=(
            (root / "gpu.jsonl.COMPLETE.json") if selected_devices else None
        ),
        expected_root_pid=process["command_pid"],
        expected_label=performance_replay.task_name(task),
        expected_host_monitor_sha256=_code_sha256(contract, "host_telemetry"),
        expected_gpu_uuids=selected_devices,
        expected_gpu_monitor_sha256=(
            _code_sha256(contract, "gpu_telemetry")
            if selected_devices
            else None
        ),
        expected_nvidia_smi_sha256=(
            contract["nvidia_smi"]["sha256"] if selected_devices else None
        ),
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
    task: dict[str, Any],
    process: dict[str, Any],
    observers: list[dict[str, Any]],
    output: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    specs = observer_specs(task, process["command_pid"], output, contract)
    if len(observers) != len(specs):
        raise WorkloadError("M3 repeated observer inventory differs")
    for record, spec in zip(observers, specs, strict=True):
        observed.verify_observer_record(output, record, spec)
    replay = derive_telemetry(task, process, output, contract)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    atomic_write_json(path, replay)
    return matrix.stable_file_record(path, output, "M3 repeated telemetry replay")


def validate_telemetry(
    task: dict[str, Any],
    process: dict[str, Any],
    observers: list[dict[str, Any]],
    retained: Any,
    output: pathlib.Path,
    contract: dict[str, Any],
) -> None:
    specs = observer_specs(task, process["command_pid"], output, contract)
    if len(observers) != len(specs):
        raise WorkloadError("M3 repeated observer inventory differs")
    for record, spec in zip(observers, specs, strict=True):
        observed.verify_observer_record(output, record, spec)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    expected = matrix.stable_file_record(path, output, "M3 repeated telemetry replay")
    if retained != expected:
        raise WorkloadError("M3 repeated telemetry replay record differs")
    with StableFile(path, "M3 repeated telemetry replay") as stable:
        with stable.file_object() as handle:
            saved = json.load(handle)
    if saved != derive_telemetry(task, process, output, contract):
        raise WorkloadError("M3 repeated telemetry was not re-derived")


def derive_sample_replay(
    task: dict[str, Any], output: pathlib.Path, contract: dict[str, Any]
) -> dict[str, Any]:
    root = output.joinpath(*task_raw_relative(task).parts)
    report = performance_sample.verify_complete(root)
    if (
        report["case_path"] != task["target"]
        or report["topology"] != task["topology"]
        or report["sample_kind"] != task["sample_kind"]
        or report["cycle_index"] != task["cycle_index"]
        or report["gpu_devices"] != contract["gpu_devices"]
        or report["receipt_id"] != contract["build_receipt"]["receipt_id"]
        or report["timing"]["contract"] != task["timing_contract"]
    ):
        raise WorkloadError("M3 repeated child sample identity differs")
    return {
        "complete": matrix.stable_file_record(
            root / "COMPLETE", output, "M3 repeated sample COMPLETE"
        ),
        "report": matrix.stable_file_record(
            root / "report.json", output, "M3 repeated sample report"
        ),
        "timing": report["timing"],
    }


def validate_completed_task(
    task: dict[str, Any],
    record: Any,
    output: pathlib.Path,
    contract: dict[str, Any],
    devices: tuple[str, str],
) -> dict[str, Any]:
    keys = {
        "task",
        "process",
        "observers",
        "telemetry_replay",
        "sample_replay",
        "hardware",
        "capacity_before",
        "capacity_after",
    }
    if not isinstance(record, dict) or set(record) != keys:
        raise WorkloadError("M3 repeated completed-task schema differs")
    if record["task"] != task:
        raise WorkloadError("M3 repeated completed-task order differs")
    matrix._verify_process_record(
        output,
        record["process"],
        task_log_relative(task).as_posix(),
        performance_replay.task_name(task),
    )
    if record["process"]["command"] != task_command(task, contract, output):
        raise WorkloadError("M3 repeated task command differs")
    validate_telemetry(
        task,
        record["process"],
        record["observers"],
        record["telemetry_replay"],
        output,
        contract,
    )
    matrix.validate_hardware_evidence(record["hardware"], list(devices))
    validate_capacity(record["capacity_before"], False)
    validate_capacity(record["capacity_after"], False)
    expected_replay = derive_sample_replay(task, output, contract)
    if record["sample_replay"] != expected_replay:
        raise WorkloadError("M3 repeated sample replay differs")
    return record


def derive_unit_results(
    output: pathlib.Path,
    contract: dict[str, Any],
    completed: list[dict[str, Any]],
) -> dict[str, Any]:
    plan = contract["plan"]
    task_records = {
        performance_replay.task_name(record["task"]): record for record in completed
    }
    expected_names = {
        performance_replay.task_name(task) for task in contract["tasks"]
    }
    if set(task_records) != expected_names or len(task_records) != TASK_COUNT:
        raise WorkloadError("M3 repeated result task inventory differs")
    cases = validation.materialize_cases(
        validation.load_json(pathlib.Path(contract["paths"]["manifest"])),
        pathlib.Path(contract["paths"]["repo"]),
    )
    results = {}
    for unit in performance_replay.repeated_units(plan):
        unit_tasks = [
            task for task in contract["tasks"] if task["unit_id"] == unit["unit_id"]
        ]
        reports = [
            performance_sample.verify_complete(
                output.joinpath(*task_raw_relative(task).parts)
            )
            for task in unit_tasks
        ]
        statistics_result = performance_replay.derive_statistics(
            plan, unit["unit_id"], reports
        )
        if statistics_result["outcome"] != "PASS":
            raise WorkloadError(
                f"M3 repeated performance gate failed: {unit['unit_id']}"
            )
        matches = [case for case in cases if case["path"] == unit["target"]]
        if len(matches) != 1:
            raise WorkloadError("M3 repeated numerical case identity differs")
        case = matches[0]
        by_key = {
            (report["sample_kind"], report["cycle_index"], report["topology"]): report
            for report in reports
        }
        comparisons = {}
        for kind, cycles in (("warmup", (0,)), ("measured", range(1, 6))):
            for cycle in cycles:
                reference = by_key[(kind, cycle, "cpu8")]
                reference_stdout = output.joinpath(
                    *task_raw_relative(
                        next(
                            task
                            for task in unit_tasks
                            if task["sample_kind"] == kind
                            and task["cycle_index"] == cycle
                            and task["topology"] == "cpu8"
                        )
                    ).parts,
                    "lane",
                    "stdout.log",
                )
                for topology in unit["topologies"]:
                    if topology == "cpu8":
                        continue
                    candidate = by_key[(kind, cycle, topology)]
                    candidate_task = next(
                        task
                        for task in unit_tasks
                        if task["sample_kind"] == kind
                        and task["cycle_index"] == cycle
                        and task["topology"] == topology
                    )
                    candidate_stdout = output.joinpath(
                        *task_raw_relative(candidate_task).parts,
                        "lane",
                        "stdout.log",
                    )
                    name = f"{kind}-{cycle:02d}-cpu8-vs-{topology}"
                    comparison = validation.compare_json_metric_files(
                        reference_stdout,
                        candidate_stdout,
                        case["comparison"],
                        temporary_parent=output / "comparison-work" / unit["unit_id"] / name,
                        expected_cpu_stdout={
                            "available": True,
                            **reference["lane"]["stdout"],
                        },
                        expected_cuda_stdout={
                            "available": True,
                            **candidate["lane"]["stdout"],
                        },
                    )
                    if comparison.get("outcome") != "PASS" or comparison.get(
                        "failure_count"
                    ) != 0:
                        raise WorkloadError(
                            f"M3 repeated numerical comparison failed: {unit['unit_id']}:{name}"
                        )
                    comparisons[name] = comparison
        results[unit["unit_id"]] = {
            "statistics": statistics_result,
            "numerical_comparisons": comparisons,
            "sample_count": len(reports),
        }
    return results


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
        or journal.get("schema") != JOURNAL_SCHEMA
        or journal.get("contract") != contract
        or not isinstance(journal.get("events"), list)
        or not journal["events"]
    ):
        raise WorkloadError("M3 repeated journal identity differs")
    first = journal["events"][0]
    if first.get("state") != "controller-started" or not math.isfinite(
        float(first.get("unix_seconds", math.nan))
    ):
        raise WorkloadError("M3 repeated journal start differs")
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
                    "sample_replay": record["sample_replay"],
                },
            )
        )
    if journal["events"][: len(expected)] != expected:
        raise WorkloadError("M3 repeated journal disagrees with checkpoint")
    return expected


def _load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        with stable.file_object() as handle:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return value


def _load_checkpoint(
    output: pathlib.Path, contract: dict[str, Any]
) -> dict[str, Any]:
    checkpoint = _load_json(output / "CHECKPOINT.json", "M3 repeated checkpoint")
    if (
        set(checkpoint)
        != {
            "schema",
            "contract",
            "hardware_before",
            "capacity_before",
            "completed_tasks",
        }
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("contract") != contract
        or not isinstance(checkpoint.get("completed_tasks"), list)
    ):
        raise WorkloadError("M3 repeated checkpoint differs")
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
    output: pathlib.Path, terminal_name: str = "PERFORMANCE_COMPLETE"
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {
        "PERFORMANCE_COMPLETE",
        "PENDING_PERFORMANCE_COMPLETE",
    }:
        raise WorkloadError("M3 repeated terminal filename differs")
    terminal = _load_json(output / terminal_name, "M3 repeated COMPLETE")
    if (
        set(terminal) != {"schema", "outcome", "report", "journal", "checkpoint"}
        or terminal.get("schema") != COMPLETE_SCHEMA
        or terminal.get("outcome") != "PASS"
    ):
        raise WorkloadError("M3 repeated COMPLETE differs")
    report = matrix.load_stable_json_record(
        output, terminal["report"], "M3 repeated report"
    )
    journal = matrix.load_stable_json_record(
        output, terminal["journal"], "M3 repeated journal"
    )
    checkpoint = matrix.load_stable_json_record(
        output, terminal["checkpoint"], "M3 repeated checkpoint"
    )
    report_keys = {
        "schema",
        "outcome",
        "contract",
        "tasks",
        "unit_results",
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
        or not isinstance(checkpoint, dict)
        or set(checkpoint) != checkpoint_keys
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
    ):
        raise WorkloadError("M3 repeated terminal report differs")
    contract = checkpoint["contract"]
    _repo, _plan, tasks, devices = validate_contract(contract)
    completed = checkpoint["completed_tasks"]
    if (
        report["contract"] != contract
        or not isinstance(completed, list)
        or len(completed) != TASK_COUNT
        or report["tasks"] != completed
    ):
        raise WorkloadError("M3 repeated completed-task inventory differs")
    for task, record in zip(tasks, completed, strict=True):
        validate_completed_task(task, record, output, contract, devices)
    if expected_journal(journal, contract, completed) != journal["events"]:
        raise WorkloadError("M3 repeated journal has unsealed events")
    if report["unit_results"] != derive_unit_results(output, contract, completed):
        raise WorkloadError("M3 repeated unit result replay differs")
    matrix.validate_hardware_evidence(report["hardware"], list(devices))
    if checkpoint["hardware_before"] != report["hardware"]["before"]:
        raise WorkloadError("M3 repeated initial hardware differs")
    if checkpoint["capacity_before"] != report["capacity_initial"]:
        raise WorkloadError("M3 repeated initial capacity differs")
    validate_capacity(report["capacity_initial"], True)
    validate_capacity(report["capacity_final"], False)
    if report["resume_history"] != matrix.resume_history_records(output):
        raise WorkloadError("M3 repeated resume history differs")
    if terminal != _terminal_payload(output, report):
        raise WorkloadError("M3 repeated terminal payload differs")
    return report


def publish_terminal(output: pathlib.Path, report: dict[str, Any]) -> None:
    atomic_write_json(
        output / "PENDING_PERFORMANCE_COMPLETE", _terminal_payload(output, report)
    )
    verify_complete(output, "PENDING_PERFORMANCE_COMPLETE")
    os.replace(
        output / "PENDING_PERFORMANCE_COMPLETE", output / "PERFORMANCE_COMPLETE"
    )


def validate_output_isolation(
    output: pathlib.Path, repo: pathlib.Path, plan_root: pathlib.Path
) -> None:
    output = pathlib.Path(os.path.abspath(output))
    for immutable in (repo.resolve(strict=True), plan_root.resolve(strict=True)):
        if (
            output == immutable
            or output in immutable.parents
            or immutable in output.parents
        ):
            raise WorkloadError("M3 repeated output overlaps an immutable input")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    repo = args.repo.resolve(strict=True)
    plan_root = args.plan_root.resolve(strict=True)
    validate_output_isolation(args.output, repo, plan_root)
    output_absolute = pathlib.Path(os.path.abspath(args.output))
    if args.resume:
        if args.output.is_symlink():
            raise WorkloadError("M3 repeated resume output is a symlink")
        output = args.output.resolve(strict=True)
        if (output / "PERFORMANCE_COMPLETE").exists():
            raise WorkloadError("completed M3 repeated matrix cannot be resumed")
    else:
        if args.output.is_symlink():
            raise WorkloadError("M3 repeated output is a symlink")
        output = output_absolute
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.parent.resolve(strict=True) != output.parent:
            raise WorkloadError("M3 repeated output parent contains a symlink")
        output.mkdir(exist_ok=False)
    lock_path = output / "LOCK"
    if args.resume and not stat.S_ISREG(lock_path.lstat().st_mode):
        raise WorkloadError("M3 repeated lock is not a regular file")
    lock = lock_path.open("r+" if args.resume else "x")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise WorkloadError(f"could not acquire M3 repeated lock: {exc}") from exc

    code_stables: dict[str, StableFile] = {}
    target_stables: list[StableFile] = []
    plan_stable: StableFile | None = None
    receipt_stable: StableFile | None = None
    nvidia_stable: StableFile | None = None
    try:
        plan = performance_plan.load_release_plan(plan_root, repo)
        tasks = performance_replay.build_tasks(plan)
        if len(tasks) != TASK_COUNT:
            raise WorkloadError("M3 repeated task count differs")
        devices = normalized_devices(args.gpu_devices)
        code_stables, code = _code_records(repo)
        plan_stable = StableFile(
            pathlib.Path(plan["input"]["path"]), "M3 repeated performance plan"
        )
        target_stables = [
            StableFile(
                repo.joinpath(*pathlib.PurePosixPath(unit["target"]).parts),
                f"M3 repeated target {unit['unit_id']}",
            )
            for unit in performance_replay.repeated_units(plan)
        ]
        receipt_path = args.build_receipt.resolve(strict=True)
        receipt_stable = StableFile(receipt_path, "M3 repeated build receipt")
        try:
            receipt = provenance.verify_build_receipt(receipt_path, repo)
        except provenance.ProvenanceError as exc:
            raise WorkloadError(
                f"M3 repeated receipt preflight failed: {exc}"
            ) from exc
        receipt_record = _receipt_record(receipt_path, receipt)
        nvidia_stable = StableFile(
            pathlib.Path("/usr/bin/nvidia-smi"), "M3 repeated nvidia-smi"
        )
        contract = build_contract(
            args,
            repo,
            plan,
            tasks,
            code,
            receipt_record,
            devices,
            absolute_file_record(nvidia_stable.path, "nvidia-smi"),
        )
        validate_contract(contract)
        current_hardware = matrix.hardware_snapshot(devices)
        current_capacity = capacity_snapshot(output)
        completed: list[dict[str, Any]] = []
        if args.resume:
            checkpoint = _load_checkpoint(output, contract)
            hardware_before = checkpoint["hardware_before"]
            capacity_before = checkpoint["capacity_before"]
            matrix.validate_hardware_evidence(
                {"before": hardware_before, "after": current_hardware},
                list(devices),
            )
            validate_capacity(capacity_before, True)
            if len(checkpoint["completed_tasks"]) > TASK_COUNT:
                raise WorkloadError("M3 repeated checkpoint has too many tasks")
            for task, record in zip(
                tasks, checkpoint["completed_tasks"], strict=False
            ):
                completed.append(
                    validate_completed_task(task, record, output, contract, devices)
                )
            journal = _load_json(output / "JOURNAL.json", "M3 repeated journal")
            prefix = expected_journal(journal, contract, completed)
            partial = [pathlib.PurePosixPath("PENDING_PERFORMANCE_COMPLETE")]
            if len(completed) < TASK_COUNT:
                current = tasks[len(completed)]
                partial.extend(
                    (
                        task_raw_relative(current),
                        task_runtime_relative(current),
                        task_log_relative(current),
                        task_telemetry_relative(current),
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
            before_hardware = matrix.hardware_snapshot(devices)
            runtime = output.joinpath(*task_runtime_relative(task).parts)
            environment = clean_environment(contract, runtime)
            append_event(output, journal, {"state": "task-started", "task": task})
            process, observer_records = observed.run_bounded_observed(
                task_command(task, contract, output),
                environment,
                output,
                output.joinpath(*task_log_relative(task).parts),
                float(task["timeout_seconds"]) + 300.0,
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
                {"state": "task-process-ended", "task": task, "process": process},
            )
            if (
                process["returncode"] != 0
                or process["timed_out"]
                or process["output_limited"]
            ):
                raise WorkloadError(
                    f"M3 repeated task failed: {performance_replay.task_name(task)}"
                )
            after_hardware = matrix.hardware_snapshot(devices)
            hardware = {"before": before_hardware, "after": after_hardware}
            matrix.validate_hardware_evidence(hardware, list(devices))
            telemetry_record = publish_telemetry(
                task, process, observer_records, output, contract
            )
            sample_replay = derive_sample_replay(task, output, contract)
            record = {
                "task": task,
                "process": process,
                "observers": observer_records,
                "telemetry_replay": telemetry_record,
                "sample_replay": sample_replay,
                "hardware": hardware,
                "capacity_before": before_capacity,
                "capacity_after": capacity_snapshot(output),
            }
            completed.append(record)
            append_event(
                output,
                journal,
                {
                    "state": "task-validated",
                    "task": task,
                    "sample_replay": sample_replay,
                },
            )
            write_checkpoint(
                output, contract, hardware_before, capacity_before, completed
            )

        unit_results = derive_unit_results(output, contract, completed)
        hardware_after = matrix.hardware_snapshot(devices)
        hardware = {"before": hardware_before, "after": hardware_after}
        matrix.validate_hardware_evidence(hardware, list(devices))
        for stable in code_stables.values():
            stable.verify_unchanged()
        for stable in target_stables:
            stable.verify_unchanged()
        plan_stable.verify_unchanged()
        receipt_stable.verify_unchanged()
        nvidia_stable.verify_unchanged()
        report = {
            "schema": SCHEMA,
            "outcome": "PASS",
            "contract": contract,
            "tasks": completed,
            "unit_results": unit_results,
            "hardware": hardware,
            "capacity_initial": capacity_before,
            "capacity_final": capacity_snapshot(output),
            "resume_history": matrix.resume_history_records(output),
            "checkpoint": matrix.stable_file_record(
                output / "CHECKPOINT.json", output, "M3 repeated checkpoint"
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
        for stable in target_stables:
            stable.close(verify=False)
        if plan_stable is not None:
            plan_stable.close(verify=False)
        if receipt_stable is not None:
            receipt_stable.close(verify=False)
        if nvidia_stable is not None:
            nvidia_stable.close(verify=False)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        WorkloadError,
        ValueError,
        json.JSONDecodeError,
        performance_plan.PlanError,
    ) as error:
        print(f"M3 repeated performance matrix error: {error}", file=sys.stderr)
        raise SystemExit(2)
