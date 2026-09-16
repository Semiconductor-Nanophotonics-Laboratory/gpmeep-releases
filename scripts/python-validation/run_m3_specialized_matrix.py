#!/usr/bin/env python3
"""Run the resumable five-unit M3 distributed-feature correctness matrix."""

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
import m3_specialized_plan as specialized_plan  # noqa: E402
import m3_specialized_replay as specialized_replay  # noqa: E402
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


SCHEMA = "gpmeep-m3-specialized-matrix-v1"
CHECKPOINT_SCHEMA = "gpmeep-m3-specialized-checkpoint-v1"
JOURNAL_SCHEMA = "gpmeep-m3-specialized-journal-v1"
COMPLETE_SCHEMA = "gpmeep-m3-specialized-complete-v1"
TASK_COUNT = 5
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
    parser.add_argument("--stdout-limit-mib", type=int, default=256)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.stdout_limit_mib <= 1024:
        raise WorkloadError("M3 specialized stdout limit must be in [1,1024] MiB")
    if (
        not math.isfinite(args.telemetry_interval_seconds)
        or not 1.0 <= args.telemetry_interval_seconds <= 60.0
    ):
        raise WorkloadError("M3 specialized telemetry interval must be in [1s,60s]")


def normalized_devices(value: str) -> tuple[str, str]:
    devices = tuple(
        matrix._normalized_uuid(item) for item in value.split(",") if item
    )
    identities = {
        item.lower().removeprefix("gpu-").replace("-", "") for item in devices
    }
    if len(devices) != 2 or len(identities) != 2:
        raise WorkloadError("M3 specialized matrix requires two distinct GPU UUIDs")
    return devices  # type: ignore[return-value]


def build_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema") != specialized_plan.SCHEMA:
        raise WorkloadError("M3 specialized plan schema differs")
    tasks = plan.get("units")
    if (
        not isinstance(tasks, list)
        or len(tasks) != TASK_COUNT
        or [task.get("launch_order") for task in tasks]
        != list(range(1, TASK_COUNT + 1))
        or len({task.get("unit_id") for task in tasks}) != TASK_COUNT
    ):
        raise WorkloadError("M3 specialized task inventory differs")
    return tasks


def task_name(task: dict[str, Any]) -> str:
    return specialized_replay.task_name(task)


def task_raw_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("raw", task_name(task))


def task_runtime_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("runtime", task_name(task))


def task_log_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("logs", f"{task_name(task)}.log")


def task_telemetry_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("telemetry", task_name(task))


def task_attestation_relative(task: dict[str, Any]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("attestations", f"{task_name(task)}.json")


def task_outer_timeout(task: dict[str, Any]) -> float:
    timeout = float(task["timeout_seconds"])
    if task["driver"] == "python-mpi-example-case":
        return timeout * 3.0 + 300.0
    return timeout


def task_command(
    task: dict[str, Any], contract: dict[str, Any], output: pathlib.Path
) -> list[str]:
    paths = contract["paths"]
    raw = output.joinpath(*task_raw_relative(task).parts)
    driver = task["driver"]
    if driver == "python-mpi-example-case":
        return [
            paths["python"],
            paths["mpi_example_runner"],
            "--repo",
            paths["repo"],
            "--manifest",
            paths["manifest"],
            "--output",
            str(raw),
            "--case",
            task["target_path"],
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
            "--cpu-ranks",
            "8",
            "--gpu-devices",
            ",".join(contract["gpu_devices"]),
            "--timeout-seconds",
            str(task["timeout_seconds"]),
        ]
    if driver == "run-mpi-python-validation":
        return [
            paths["python"],
            paths["mpi_python_runner"],
            "--build-receipt",
            paths["build_receipt"],
            "--output",
            str(raw),
            "--devices",
            ",".join(str(value) for value in contract["gpu_ordinals"]),
            "--timeout",
            str(task["timeout_seconds"]),
        ]
    if driver == "run-near2far-mpi-qualification":
        attestation = output.joinpath(*task_attestation_relative(task).parts)
        return [
            paths["python"],
            paths["near2far_runner"],
            "--executable",
            paths["gpu_step_db"],
            "--output",
            str(raw),
            "--batch-targets",
            "512",
            "--frequency-count",
            "12",
            "--repetitions",
            "7",
            "--cpu-threads",
            "8",
            "--small-minimum-speedup",
            "1.0",
            "--batch-minimum-speedup",
            "1.5",
            "--batch-minimum-two-gpu-scaling",
            "1.1",
            "--lane-timeout-seconds",
            "600",
            "--qualification-tier",
            "release",
            "--build-receipt",
            paths["build_receipt"],
            "--write-release-attestation",
            str(attestation),
        ]
    if driver == "run-mpi-adjoint-benchmark":
        return [
            paths["python"],
            paths["adjoint_runner"],
            "--build-receipt",
            paths["build_receipt"],
            "--output",
            str(raw),
            "--profile",
            "m19-cpu-gpu-adjoint-release-v4",
        ]
    raise WorkloadError(f"M3 specialized driver is unsupported: {driver!r}")


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
        raise WorkloadError("M3 specialized capacity record is invalid")
    if record["free_bytes"] < (
        INITIAL_FREE_BYTES if initial else PER_TASK_FREE_BYTES
    ):
        raise WorkloadError("M3 specialized free-space gate failed")
    if record["free_inodes"] < MINIMUM_FREE_INODES:
        raise WorkloadError("M3 specialized free-inode gate failed")


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
        "plan_loader": SCRIPT_DIR / "m3_specialized_plan.py",
        "task_replay": SCRIPT_DIR / "m3_specialized_replay.py",
        "mpi_example_runner": SCRIPT_DIR / "run_m3_mpi_example_case.py",
        "mpi_rank_launcher": SCRIPT_DIR / "m3_mpi_rank_launcher.py",
        "validation_runner": SCRIPT_DIR / "run_validation.py",
        "manifest": SCRIPT_DIR / "manifest.json",
        "provenance": SCRIPT_DIR.parent / "gpmeep_provenance.py",
        "observed_runner": USER_WORKLOAD_DIR / "run_observed_process.py",
        "matrix_common": USER_WORKLOAD_DIR / "run_user_workload_matrix.py",
        "host_telemetry": USER_WORKLOAD_DIR / "capture_host_telemetry_v2.py",
        "gpu_telemetry": USER_WORKLOAD_DIR / "capture_gpu_telemetry_v2.py",
        "telemetry_replay": USER_WORKLOAD_DIR / "verify_telemetry_v2.py",
        "workload_common": USER_WORKLOAD_DIR / "common.py",
        "mpi_python_runner": SCRIPT_DIR.parent / "run-mpi-python-validation.py",
        "near2far_runner": SCRIPT_DIR.parent / "run-near2far-mpi-qualification.py",
        "adjoint_runner": SCRIPT_DIR.parent / "run-mpi-adjoint-benchmark.py",
        "adjoint_replay": SCRIPT_DIR.parent / "verify-mpi-adjoint-evidence.py",
    }


def _code_records(
    repo: pathlib.Path,
) -> tuple[dict[str, StableFile], dict[str, dict[str, Any]]]:
    stables = {
        name: StableFile(path, f"M3 specialized {name}")
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
        raise WorkloadError("M3 specialized build receipt IDs are invalid")
    source_sha256 = receipt.get("source_start", {}).get("sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise WorkloadError("M3 specialized build source identity is invalid")
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
        "mpi_example_runner": str(
            (SCRIPT_DIR / "run_m3_mpi_example_case.py").resolve(strict=True)
        ),
        "mpi_python_runner": str(
            (SCRIPT_DIR.parent / "run-mpi-python-validation.py").resolve(strict=True)
        ),
        "near2far_runner": str(
            (SCRIPT_DIR.parent / "run-near2far-mpi-qualification.py").resolve(
                strict=True
            )
        ),
        "adjoint_runner": str(
            (SCRIPT_DIR.parent / "run-mpi-adjoint-benchmark.py").resolve(
                strict=True
            )
        ),
        "gpu_step_db": str(
            (repo / "build/meep-cuda-mpi-python-fp32/tests/gpu-step-db").resolve(
                strict=True
            )
        ),
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
        "gpu_ordinals": [0, 1],
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


def _paths_overlap(first: pathlib.Path, second: pathlib.Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_output_isolation(
    output: pathlib.Path, repo: pathlib.Path, plan_root: pathlib.Path
) -> None:
    output = pathlib.Path(os.path.abspath(output))
    for immutable in (repo.resolve(strict=True), plan_root.resolve(strict=True)):
        if _paths_overlap(output, immutable):
            raise WorkloadError(
                "M3 specialized output overlaps an immutable input root"
            )


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
        "mpi_example_runner": (
            SCRIPT_DIR / "run_m3_mpi_example_case.py"
        ).resolve(strict=True),
        "mpi_python_runner": (
            SCRIPT_DIR.parent / "run-mpi-python-validation.py"
        ).resolve(strict=True),
        "near2far_runner": (
            SCRIPT_DIR.parent / "run-near2far-mpi-qualification.py"
        ).resolve(strict=True),
        "adjoint_runner": (
            SCRIPT_DIR.parent / "run-mpi-adjoint-benchmark.py"
        ).resolve(strict=True),
        "gpu_step_db": (
            repo / "build/meep-cuda-mpi-python-fp32/tests/gpu-step-db"
        ).resolve(strict=True),
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
        "gpu_ordinals",
        "nvidia_smi",
        "limits",
        "gates",
    }
    if (
        not isinstance(contract, dict)
        or set(contract) != keys
        or contract.get("schema") != SCHEMA
    ):
        raise WorkloadError("M3 specialized sealed contract schema differs")
    paths = contract["paths"]
    expected_path_keys = {
        "repo",
        "plan_root",
        "python",
        "build_python",
        "install_prefix",
        "build_receipt",
        "mpiexec",
        "manifest",
        "mpi_example_runner",
        "mpi_python_runner",
        "near2far_runner",
        "adjoint_runner",
        "gpu_step_db",
    }
    if (
        not isinstance(paths, dict)
        or set(paths) != expected_path_keys
        or any(not isinstance(paths[name], str) for name in expected_path_keys)
    ):
        raise WorkloadError("M3 specialized sealed path inventory differs")
    resolved: dict[str, pathlib.Path] = {}
    for name, value in paths.items():
        path = pathlib.Path(value)
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise WorkloadError(f"M3 specialized path is not canonical: {name}")
        resolved[name] = path
    directory_names = {"repo", "plan_root", "build_python", "install_prefix"}
    if any(
        (name in directory_names) != path.is_dir()
        for name, path in resolved.items()
    ):
        raise WorkloadError("M3 specialized runtime path type differs")
    if any(
        not os.access(resolved[name], os.X_OK)
        for name in ("python", "mpiexec", "gpu_step_db")
    ):
        raise WorkloadError("M3 specialized runtime executable is not executable")
    authoritative = _authoritative_paths(resolved["repo"], resolved["plan_root"])
    if resolved != authoritative:
        raise WorkloadError("M3 specialized runtime is not authoritative")
    plan = specialized_plan.load_release_plan(
        resolved["plan_root"], resolved["repo"]
    )
    tasks = build_tasks(plan)
    if contract["plan"] != plan or contract["tasks"] != tasks:
        raise WorkloadError("M3 specialized sealed plan was not re-derived")
    expected_code = _code_paths()
    code = contract["evidence_code"]
    if not isinstance(code, dict) or set(code) != set(expected_code):
        raise WorkloadError("M3 specialized evidence-code inventory differs")
    for name, expected_path in expected_code.items():
        stable = matrix.stable_record(
            resolved["repo"], code[name], f"M3 specialized {name}"
        )
        try:
            if stable.path != expected_path.resolve(strict=True):
                raise WorkloadError(
                    f"M3 specialized evidence-code path differs: {name}"
                )
        finally:
            stable.close()
    try:
        receipt = provenance.verify_build_receipt(
            resolved["build_receipt"], resolved["repo"]
        )
    except provenance.ProvenanceError as exc:
        raise WorkloadError(
            f"M3 specialized build receipt replay failed: {exc}"
        ) from exc
    if contract["build_receipt"] != _receipt_record(
        resolved["build_receipt"], receipt
    ):
        raise WorkloadError("M3 specialized build receipt record differs")
    gpu_devices = contract["gpu_devices"]
    if (
        not isinstance(gpu_devices, list)
        or any(not isinstance(item, str) for item in gpu_devices)
    ):
        raise WorkloadError("M3 specialized GPU device inventory differs")
    devices = normalized_devices(",".join(gpu_devices))
    if list(devices) != gpu_devices or contract["gpu_ordinals"] != [0, 1]:
        raise WorkloadError("M3 specialized GPU mapping differs")
    if contract["nvidia_smi"] != absolute_file_record(
        pathlib.Path("/usr/bin/nvidia-smi"), "nvidia-smi"
    ):
        raise WorkloadError("M3 specialized nvidia-smi identity differs")
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
        raise WorkloadError("M3 specialized sealed limits differ")
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
        raise WorkloadError("M3 specialized sealed gates differ")
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
    label = task_name(task)
    selected_flags = tuple(
        item
        for device in contract["gpu_devices"]
        for item in ("--selected-uuid", device)
    )
    return [
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
        ),
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
        ),
    ]


def _code_sha256(contract: dict[str, Any], name: str) -> str:
    try:
        value = contract["evidence_code"][name]["sha256"]
    except (KeyError, TypeError) as exc:
        raise WorkloadError(
            f"M3 specialized code identity is absent: {name}"
        ) from exc
    if not isinstance(value, str) or len(value) != 64:
        raise WorkloadError(f"M3 specialized code digest is invalid: {name}")
    return value


def derive_telemetry(
    task: dict[str, Any],
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
        expected_gpu_uuids=tuple(contract["gpu_devices"]),
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
    task: dict[str, Any],
    process: dict[str, Any],
    observers: list[dict[str, Any]],
    output: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    specs = observer_specs(task, process["command_pid"], output, contract)
    if len(observers) != len(specs):
        raise WorkloadError("M3 specialized observer inventory differs")
    for record, spec in zip(observers, specs, strict=True):
        observed.verify_observer_record(output, record, spec)
    report = derive_telemetry(task, process, output, contract)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    atomic_write_json(path, report)
    return matrix.stable_file_record(
        path, output, "M3 specialized telemetry replay"
    )


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
        raise WorkloadError("M3 specialized observer inventory differs")
    for record, spec in zip(observers, specs, strict=True):
        observed.verify_observer_record(output, record, spec)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    expected = matrix.stable_file_record(
        path, output, "M3 specialized telemetry replay"
    )
    if retained != expected:
        raise WorkloadError("M3 specialized telemetry replay record differs")
    with StableFile(path, "M3 specialized telemetry replay") as stable:
        with stable.file_object() as handle:
            saved = json.load(handle)
    if saved != derive_telemetry(task, process, output, contract):
        raise WorkloadError("M3 specialized telemetry was not re-derived")


def derive_task_replay(
    task: dict[str, Any], output: pathlib.Path, contract: dict[str, Any]
) -> dict[str, Any]:
    return specialized_replay.verify_task(task, output, contract)


def validate_task_replay(
    task: dict[str, Any],
    retained: Any,
    output: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(retained, dict)
        or set(retained) != {"schema", "unit_id", "evidence"}
        or retained.get("schema") != specialized_replay.SCHEMA
        or retained.get("unit_id") != task["unit_id"]
        or not isinstance(retained.get("evidence"), dict)
    ):
        raise WorkloadError("M3 specialized task replay schema differs")
    derived = specialized_replay.verify_task(
        task, output, contract, expected=retained["evidence"]
    )
    if retained != derived:
        raise WorkloadError("M3 specialized task replay was not re-derived")
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
        or journal.get("schema") != JOURNAL_SCHEMA
        or journal.get("contract") != contract
        or not isinstance(journal.get("events"), list)
        or not journal["events"]
    ):
        raise WorkloadError("M3 specialized journal identity differs")
    first = journal["events"][0]
    if first.get("state") != "controller-started" or not math.isfinite(
        float(first.get("unix_seconds", math.nan))
    ):
        raise WorkloadError("M3 specialized journal start differs")
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
                    "task_replay": record["task_replay"],
                },
            )
        )
    if journal["events"][: len(expected)] != expected:
        raise WorkloadError("M3 specialized journal disagrees with checkpoint")
    return expected


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
        "task_replay",
        "hardware",
        "capacity_before",
        "capacity_after",
    }
    if not isinstance(record, dict) or set(record) != keys:
        raise WorkloadError("M3 specialized completed-task schema differs")
    if record["task"] != task:
        raise WorkloadError("M3 specialized completed-task order differs")
    matrix._verify_process_record(
        output,
        record["process"],
        task_log_relative(task).as_posix(),
        task_name(task),
    )
    if record["process"]["command"] != task_command(task, contract, output):
        raise WorkloadError("M3 specialized task command differs")
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
    validate_task_replay(task, record["task_replay"], output, contract)
    return record


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
    checkpoint = _load_json(
        output / "CHECKPOINT.json", "M3 specialized checkpoint"
    )
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
        raise WorkloadError("M3 specialized checkpoint differs")
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
    output: pathlib.Path, terminal_name: str = "SPECIALIZED_COMPLETE"
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {
        "SPECIALIZED_COMPLETE",
        "PENDING_SPECIALIZED_COMPLETE",
    }:
        raise WorkloadError("M3 specialized terminal filename differs")
    terminal = _load_json(
        output / terminal_name, "M3 specialized matrix COMPLETE"
    )
    if (
        set(terminal) != {"schema", "outcome", "report", "journal", "checkpoint"}
        or terminal.get("schema") != COMPLETE_SCHEMA
        or terminal.get("outcome") != "PASS"
    ):
        raise WorkloadError("M3 specialized COMPLETE is not an exact PASS")
    report = matrix.load_stable_json_record(
        output, terminal["report"], "M3 specialized matrix report"
    )
    journal = matrix.load_stable_json_record(
        output, terminal["journal"], "M3 specialized matrix journal"
    )
    checkpoint = matrix.load_stable_json_record(
        output, terminal["checkpoint"], "M3 specialized matrix checkpoint"
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
        or not isinstance(checkpoint, dict)
        or set(checkpoint) != checkpoint_keys
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
    ):
        raise WorkloadError("M3 specialized terminal report differs")
    contract = checkpoint["contract"]
    _repo, _plan, tasks, devices = validate_contract(contract)
    completed = checkpoint["completed_tasks"]
    if (
        report["contract"] != contract
        or not isinstance(completed, list)
        or len(completed) != TASK_COUNT
        or report["tasks"] != completed
    ):
        raise WorkloadError("M3 specialized completed-task inventory differs")
    for task, record in zip(tasks, completed, strict=True):
        validate_completed_task(task, record, output, contract, devices)
    if expected_journal(journal, contract, completed) != journal["events"]:
        raise WorkloadError("M3 specialized journal has unsealed events")
    matrix.validate_hardware_evidence(report["hardware"], list(devices))
    if checkpoint["hardware_before"] != report["hardware"]["before"]:
        raise WorkloadError("M3 specialized initial hardware differs")
    if checkpoint["capacity_before"] != report["capacity_initial"]:
        raise WorkloadError("M3 specialized initial capacity differs")
    validate_capacity(report["capacity_initial"], True)
    validate_capacity(report["capacity_final"], False)
    if report["resume_history"] != matrix.resume_history_records(output):
        raise WorkloadError("M3 specialized resume history differs")
    if terminal != _terminal_payload(output, report):
        raise WorkloadError("M3 specialized terminal payload differs")
    return report


def publish_terminal(output: pathlib.Path, report: dict[str, Any]) -> None:
    atomic_write_json(
        output / "PENDING_SPECIALIZED_COMPLETE", _terminal_payload(output, report)
    )
    verify_complete(output, "PENDING_SPECIALIZED_COMPLETE")
    os.replace(
        output / "PENDING_SPECIALIZED_COMPLETE", output / "SPECIALIZED_COMPLETE"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    repo = args.repo.resolve(strict=True)
    plan_root = args.plan_root.resolve(strict=True)
    validate_output_isolation(args.output, repo, plan_root)
    output_absolute = pathlib.Path(os.path.abspath(args.output))
    if args.resume:
        if args.output.is_symlink():
            raise WorkloadError("M3 specialized resume output is a symlink")
        output = args.output.resolve(strict=True)
        if (output / "SPECIALIZED_COMPLETE").exists():
            raise WorkloadError("completed M3 specialized matrix cannot be resumed")
    else:
        if args.output.is_symlink():
            raise WorkloadError("M3 specialized output is a symlink")
        output = output_absolute
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.parent.resolve(strict=True) != output.parent:
            raise WorkloadError(
                "M3 specialized output parent contains a symlink"
            )
        output.mkdir(exist_ok=False)
    lock_path = output / "LOCK"
    if args.resume and not stat.S_ISREG(lock_path.lstat().st_mode):
        raise WorkloadError("M3 specialized lock is not a regular file")
    lock = lock_path.open("r+" if args.resume else "x")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise WorkloadError(
            f"could not acquire M3 specialized lock: {exc}"
        ) from exc

    code_stables: dict[str, StableFile] = {}
    target_stables: list[StableFile] = []
    receipt_stable: StableFile | None = None
    plan_stable: StableFile | None = None
    nvidia_stable: StableFile | None = None
    try:
        plan = specialized_plan.load_release_plan(plan_root, repo)
        tasks = build_tasks(plan)
        devices = normalized_devices(args.gpu_devices)
        code_stables, code = _code_records(repo)
        plan_stable = StableFile(
            pathlib.Path(plan["input"]["path"]), "M3 specialized plan"
        )
        target_stables = [
            StableFile(
                repo.joinpath(
                    *pathlib.PurePosixPath(task["target_path"]).parts
                ),
                f"M3 specialized target {task['unit_id']}",
            )
            for task in tasks
        ]
        receipt_path = args.build_receipt.resolve(strict=True)
        receipt_stable = StableFile(
            receipt_path, "M3 specialized build receipt"
        )
        try:
            receipt = provenance.verify_build_receipt(receipt_path, repo)
        except provenance.ProvenanceError as exc:
            raise WorkloadError(
                f"M3 specialized build receipt preflight failed: {exc}"
            ) from exc
        receipt_record = _receipt_record(receipt_path, receipt)
        nvidia_stable = StableFile(
            pathlib.Path("/usr/bin/nvidia-smi"), "M3 specialized nvidia-smi"
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
        current_hardware = matrix.hardware_snapshot(list(devices))
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
                raise WorkloadError(
                    "M3 specialized checkpoint has too many tasks"
                )
            for task, record in zip(
                tasks, checkpoint["completed_tasks"], strict=False
            ):
                completed.append(
                    validate_completed_task(
                        task, record, output, contract, devices
                    )
                )
            journal = _load_json(
                output / "JOURNAL.json", "M3 specialized journal"
            )
            prefix = expected_journal(journal, contract, completed)
            partial = [pathlib.PurePosixPath("PENDING_SPECIALIZED_COMPLETE")]
            if len(completed) < TASK_COUNT:
                current = tasks[len(completed)]
                partial.extend(
                    (
                        task_raw_relative(current),
                        task_runtime_relative(current),
                        task_log_relative(current),
                        task_telemetry_relative(current),
                        task_attestation_relative(current),
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
            before_hardware = matrix.hardware_snapshot(list(devices))
            runtime = output.joinpath(*task_runtime_relative(task).parts)
            environment = clean_environment(contract, runtime)
            append_event(
                output, journal, {"state": "task-started", "task": task}
            )
            process, observer_records = observed.run_bounded_observed(
                task_command(task, contract, output),
                environment,
                output,
                output.joinpath(*task_log_relative(task).parts),
                task_outer_timeout(task),
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
                    "task": task,
                    "process": process,
                },
            )
            if (
                process["returncode"] != 0
                or process["timed_out"]
                or process["output_limited"]
            ):
                raise WorkloadError(
                    f"M3 specialized task failed: {task_name(task)}"
                )
            after_hardware = matrix.hardware_snapshot(list(devices))
            hardware = {"before": before_hardware, "after": after_hardware}
            matrix.validate_hardware_evidence(hardware, list(devices))
            telemetry_record = publish_telemetry(
                task, process, observer_records, output, contract
            )
            replay_record = derive_task_replay(task, output, contract)
            record = {
                "task": task,
                "process": process,
                "observers": observer_records,
                "telemetry_replay": telemetry_record,
                "task_replay": replay_record,
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
                    "task_replay": replay_record,
                },
            )
            write_checkpoint(
                output, contract, hardware_before, capacity_before, completed
            )

        hardware_after = matrix.hardware_snapshot(list(devices))
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
            "hardware": hardware,
            "capacity_initial": capacity_before,
            "capacity_final": capacity_snapshot(output),
            "resume_history": matrix.resume_history_records(output),
            "checkpoint": matrix.stable_file_record(
                output / "CHECKPOINT.json",
                output,
                "M3 specialized checkpoint",
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
        specialized_plan.PlanError,
    ) as error:
        print(f"M3 specialized matrix error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
