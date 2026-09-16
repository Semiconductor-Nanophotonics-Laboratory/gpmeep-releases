#!/usr/bin/env python3
"""Run the release AuNP hybrid exact/performance qualification matrix."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import pathlib
import shutil
import stat
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_aunp_performance_workload as performance_adapter  # noqa: E402
import run_observed_process as observed  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
import verify_aunp_performance_workload as performance_replay  # noqa: E402
import verify_telemetry_v2 as telemetry_replay  # noqa: E402
from common import (  # noqa: E402
    AUNP_ARCHIVE_SHA256,
    StableFile,
    WorkloadError,
    absolute_file_record,
    aunp_qualification_profile,
    atomic_write_json,
    atomic_write_text,
    file_record,
    verify_file,
)


SCHEMA = "gpmeep-hybrid-aunp-matrix-v3"
CHECKPOINT_SCHEMA = "gpmeep-hybrid-aunp-checkpoint-v3"
COMPLETE_SCHEMA = "gpmeep-hybrid-aunp-complete-v3"
JOURNAL_SCHEMA = "gpmeep-hybrid-aunp-journal-v3"
RELEASE_PERFORMANCE_REPEATS = 2
RELEASE_GATES = {
    "minimum_one_gpu_speedup": (1.5, math.inf, "minimum"),
    "minimum_two_gpu_speedup": (2.0, math.inf, "minimum"),
    "minimum_multi_gpu_scaling": (1.1, math.inf, "minimum"),
    "minimum_gpu_free_memory_fraction": (0.20, 0.99, "minimum"),
    "maximum_gpu_temperature_celsius": (1.0, 90.0, "maximum"),
    "maximum_host_temperature_millidegrees": (1.0, 95_000.0, "maximum"),
    "minimum_host_available_bytes": (4 * 1024**3, math.inf, "minimum"),
    "minimum_cgroup_free_bytes": (4 * 1024**3, math.inf, "minimum"),
    "minimum_free_workspace_bytes": (40 * 1024**3, math.inf, "minimum"),
    "minimum_free_exact_lane_bytes": (20 * 1024**3, math.inf, "minimum"),
    "minimum_free_inodes": (100_000, math.inf, "minimum"),
}


@dataclass(frozen=True)
class Task:
    role: str
    lane: matrix.Lane
    repeat: int
    kind: str
    counts_toward_performance: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--fp32-python", required=True, type=pathlib.Path)
    parser.add_argument("--fp32-mpiexec", required=True, type=pathlib.Path)
    parser.add_argument("--fp32-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--cpu-ranks", type=int)
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument(
        "--performance-repeats", type=int, default=RELEASE_PERFORMANCE_REPEATS
    )
    parser.add_argument("--exact-timeout-seconds", type=float, default=72 * 3600.0)
    parser.add_argument("--performance-timeout-seconds", type=float, default=6 * 3600.0)
    parser.add_argument("--comparison-timeout-seconds", type=float, default=12 * 3600.0)
    parser.add_argument("--stdout-limit-mib", type=int, default=128)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=5.0)
    parser.add_argument("--minimum-one-gpu-speedup", type=float, default=1.5)
    parser.add_argument("--minimum-two-gpu-speedup", type=float, default=2.0)
    parser.add_argument("--minimum-multi-gpu-scaling", type=float, default=1.1)
    parser.add_argument(
        "--minimum-gpu-free-memory-fraction", type=float, default=0.20
    )
    parser.add_argument(
        "--maximum-gpu-temperature-celsius", type=float, default=90.0
    )
    parser.add_argument(
        "--maximum-host-temperature-millidegrees", type=int, default=95_000
    )
    parser.add_argument(
        "--minimum-host-available-bytes", type=int, default=4 * 1024**3
    )
    parser.add_argument(
        "--minimum-cgroup-free-bytes", type=int, default=4 * 1024**3
    )
    parser.add_argument(
        "--minimum-free-workspace-bytes", type=int, default=40 * 1024**3
    )
    parser.add_argument(
        "--minimum-free-exact-lane-bytes", type=int, default=20 * 1024**3
    )
    parser.add_argument("--minimum-free-inodes", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def release_gate_values(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in RELEASE_GATES}


def validate_release_gates(gates: Any) -> None:
    if not isinstance(gates, dict) or set(gates) != set(RELEASE_GATES):
        raise WorkloadError("AuNP release gate inventory is not exact")
    for name, (lower, upper, direction) in RELEASE_GATES.items():
        value = gates[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise WorkloadError(f"AuNP release gate {name} is not finite")
        numeric = float(value)
        valid = lower <= numeric <= upper
        if not valid:
            raise WorkloadError(f"AuNP release gate {name} is outside release bounds")
        default_boundary = lower if direction == "minimum" else upper
        if direction == "minimum" and numeric < default_boundary:
            raise WorkloadError(f"AuNP release gate {name} was weakened")
        if direction == "maximum" and numeric > default_boundary:
            raise WorkloadError(f"AuNP release gate {name} was weakened")


def validate_args(args: argparse.Namespace) -> None:
    if args.performance_repeats != RELEASE_PERFORMANCE_REPEATS:
        raise WorkloadError("AuNP release qualification requires exactly two repeats")
    for name in (
        "exact_timeout_seconds",
        "performance_timeout_seconds",
        "comparison_timeout_seconds",
    ):
        value = getattr(args, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise WorkloadError(f"{name} must be positive and finite")
    if not 1 <= args.stdout_limit_mib <= 1024:
        raise WorkloadError("stdout limit must be in [1,1024] MiB")
    if (
        not math.isfinite(args.telemetry_interval_seconds)
        or not 1.0 <= args.telemetry_interval_seconds <= 60.0
    ):
        raise WorkloadError("telemetry interval must be in [1,60] seconds")
    validate_release_gates(release_gate_values(args))


def normalized_devices(value: str) -> tuple[str, str]:
    devices = tuple(matrix._normalized_uuid(item) for item in value.split(","))
    normalized = {
        item.lower().removeprefix("gpu-").replace("-", "") for item in devices
    }
    if len(devices) != 2 or len(normalized) != 2:
        raise WorkloadError("AuNP qualification requires two distinct GPU UUIDs")
    return devices


def build_tasks(
    fp32: matrix.Build,
    cpu_ranks: int,
    devices: tuple[str, str],
    repeats: int = RELEASE_PERFORMANCE_REPEATS,
) -> list[Task]:
    cpu = matrix.Lane("cpu-fp32", fp32, "cpu", cpu_ranks, ())
    gpu1 = matrix.Lane("cuda-fp32-1g", fp32, "cuda", 1, devices[:1])
    gpu2 = matrix.Lane("cuda-fp32-2g", fp32, "cuda", 2, devices)
    tasks = [
        Task("smoke-cuda-fp32-1g", gpu1, 0, "smoke", False),
        Task("smoke-cuda-fp32-2g", gpu2, 0, "smoke", False),
        Task("exact-cuda-fp32-1g", gpu1, 0, "exact", False),
        Task("exact-cuda-fp32-2g", gpu2, 0, "exact", False),
    ]
    performance = (
        (f"performance-cpu-fp32-{cpu_ranks}r", cpu),
        ("performance-cuda-fp32-1g", gpu1),
        ("performance-cuda-fp32-2g", gpu2),
    )
    for repeat in range(repeats):
        ordered = performance if repeat % 2 == 0 else tuple(reversed(performance))
        tasks.extend(
            Task(role, lane, repeat, "performance", True)
            for role, lane in ordered
        )
    return tasks


def task_spec(task: Task) -> dict[str, Any]:
    return {
        "role": task.role,
        "repeat": task.repeat,
        "kind": task.kind,
        "counts_toward_performance": task.counts_toward_performance,
        "backend": task.lane.backend,
        "ranks": task.lane.ranks,
        "devices": list(task.lane.devices),
        "build_receipt_id": task.lane.build.receipt["receipt_id"],
    }


def build_contract_record(build: matrix.Build) -> dict[str, Any]:
    return {
        "python": str(build.python),
        "mpiexec": str(build.mpiexec),
        "receipt_path": str(build.receipt_path),
        "receipt_id": build.receipt["receipt_id"],
        "build_input_id": build.receipt["build_input_id"],
        "artifact_set_id": build.receipt["artifact_set_id"],
        "source_sha256": matrix._source_identity(build),
    }


def task_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("lanes", task.role, f"repeat-{task.repeat:02d}")


def task_runtime_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("runtime", task.role, f"repeat-{task.repeat:02d}")


def task_log_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("logs", task.role, f"repeat-{task.repeat:02d}.log")


def task_telemetry_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath(
        "telemetry", task.role, f"repeat-{task.repeat:02d}"
    )


def task_replay_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("replays", task.role, f"repeat-{task.repeat:02d}.json")


def task_command(task: Task, archive: pathlib.Path, output: pathlib.Path) -> list[str]:
    adapter = (
        SCRIPT_DIR / "run_aunp_workload.py"
        if task.kind == "exact"
        else SCRIPT_DIR / "run_aunp_performance_workload.py"
    )
    return [
        str(task.lane.build.mpiexec),
        "--bind-to",
        "core",
        "--map-by",
        "core:PE=1",
        "--report-bindings",
        "-n",
        str(task.lane.ranks),
        str(task.lane.build.python),
        "-u",
        str(adapter),
        "--archive",
        str(archive),
        "--output",
        str(output),
        "--build-receipt",
        str(task.lane.build.receipt_path),
        "--expected-backend",
        task.lane.backend,
    ]


def exact_comparison_specs() -> list[dict[str, str]]:
    return [
        {
            "name": "gpu1-reference",
            "reference_role": "exact-cuda-fp32-1g",
            "candidate_role": "exact-cuda-fp32-2g",
            "candidate_class": "repeat-cuda-fp32",
        },
        {
            "name": "gpu2-reference",
            "reference_role": "exact-cuda-fp32-2g",
            "candidate_role": "exact-cuda-fp32-1g",
            "candidate_class": "repeat-cuda-fp32",
        },
    ]


def capacity_snapshot(path: pathlib.Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    usage = shutil.disk_usage(path)
    filesystem = os.statvfs(path)
    return {
        "path": str(path),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "free_inodes": filesystem.f_favail,
        "total_inodes": filesystem.f_files,
    }


def validate_capacity(snapshot: Any, gates: dict[str, Any], task: Task | None) -> None:
    validate_capacity_record(snapshot)
    minimum_bytes = (
        gates["minimum_free_workspace_bytes"]
        if task is None
        else gates["minimum_free_exact_lane_bytes"]
    )
    if snapshot["free_bytes"] < minimum_bytes:
        raise WorkloadError("AuNP workspace free-space gate failed")
    if snapshot["free_inodes"] < gates["minimum_free_inodes"]:
        raise WorkloadError("AuNP workspace free-inode gate failed")


def validate_capacity_record(snapshot: Any) -> None:
    required = {"path", "free_bytes", "total_bytes", "free_inodes", "total_inodes"}
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != required
        or not isinstance(snapshot["path"], str)
        or not pathlib.Path(snapshot["path"]).is_absolute()
        or any(
            type(snapshot[name]) is not int or snapshot[name] < 0
            for name in ("free_bytes", "total_bytes", "free_inodes", "total_inodes")
        )
        or snapshot["free_bytes"] > snapshot["total_bytes"]
        or snapshot["free_inodes"] > snapshot["total_inodes"]
    ):
        raise WorkloadError("AuNP workspace-capacity record is invalid")


def contract(
    args: argparse.Namespace,
    archive: pathlib.Path,
    evidence_code: dict[str, Any],
    fp32: matrix.Build,
    cpu_ranks: int,
    devices: tuple[str, str],
    tasks: list[Task],
    nvidia_smi: dict[str, Any],
) -> dict[str, Any]:
    return {
        "input": {
            "path": str(archive),
            "size_bytes": archive.stat().st_size,
            "sha256": AUNP_ARCHIVE_SHA256,
        },
        "evidence_code": evidence_code,
        "build": build_contract_record(fp32),
        "cpu_ranks": cpu_ranks,
        "gpu_devices": list(devices),
        "performance_repeats": args.performance_repeats,
        "qualification_profile": aunp_qualification_profile(),
        "fixed_window_meep_time": performance_adapter.FIXED_WINDOW_MEEP_TIME,
        "minimum_timesteps_per_phase": (
            performance_adapter.MINIMUM_TIMESTEPS_PER_PHASE
        ),
        "limits": {
            "exact_timeout_seconds": args.exact_timeout_seconds,
            "performance_timeout_seconds": args.performance_timeout_seconds,
            "comparison_timeout_seconds": args.comparison_timeout_seconds,
            "stdout_limit_bytes": args.stdout_limit_mib * 1024 * 1024,
            "telemetry_interval_seconds": args.telemetry_interval_seconds,
        },
        "gates": release_gate_values(args),
        "nvidia_smi": nvidia_smi,
        "tasks": [task_spec(task) for task in tasks],
        "exact_comparisons": exact_comparison_specs(),
    }


def reconstruct_contract_tasks(
    sealed_contract: dict[str, Any],
) -> tuple[matrix.Build, tuple[str, str], list[Task]]:
    contract_keys = {
        "input",
        "evidence_code",
        "build",
        "cpu_ranks",
        "gpu_devices",
        "performance_repeats",
        "qualification_profile",
        "fixed_window_meep_time",
        "minimum_timesteps_per_phase",
        "limits",
        "gates",
        "nvidia_smi",
        "tasks",
        "exact_comparisons",
    }
    if not isinstance(sealed_contract, dict) or set(sealed_contract) != contract_keys:
        raise WorkloadError("AuNP sealed contract is invalid")
    input_record = sealed_contract["input"]
    if (
        not isinstance(input_record, dict)
        or set(input_record) != {"path", "size_bytes", "sha256"}
        or not isinstance(input_record["path"], str)
        or not pathlib.Path(input_record["path"]).is_absolute()
        or type(input_record["size_bytes"]) is not int
        or input_record["size_bytes"] <= 0
        or input_record["sha256"] != AUNP_ARCHIVE_SHA256
    ):
        raise WorkloadError("AuNP sealed input contract is invalid")
    limits = sealed_contract["limits"]
    limit_keys = {
        "exact_timeout_seconds",
        "performance_timeout_seconds",
        "comparison_timeout_seconds",
        "stdout_limit_bytes",
        "telemetry_interval_seconds",
    }
    if not isinstance(limits, dict) or set(limits) != limit_keys:
        raise WorkloadError("AuNP sealed execution limits are invalid")
    for name in (
        "exact_timeout_seconds",
        "performance_timeout_seconds",
        "comparison_timeout_seconds",
    ):
        value = limits[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise WorkloadError(f"AuNP sealed {name} is invalid")
    if (
        type(limits["stdout_limit_bytes"]) is not int
        or not 1024**2 <= limits["stdout_limit_bytes"] <= 1024**3
        or limits["stdout_limit_bytes"] % 1024**2
    ):
        raise WorkloadError("AuNP sealed stdout limit is invalid")
    interval = limits["telemetry_interval_seconds"]
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or not 1.0 <= interval <= 60.0
    ):
        raise WorkloadError("AuNP sealed telemetry interval is invalid")
    record = sealed_contract.get("build")
    if not isinstance(record, dict):
        raise WorkloadError("AuNP sealed build contract is invalid")
    try:
        build = matrix.load_build(
            "shared-fp32",
            pathlib.Path(record["python"]),
            pathlib.Path(record["mpiexec"]),
            pathlib.Path(record["receipt_path"]),
            "cuda-mpi-python-fp32",
            True,
            True,
        )
    except (KeyError, TypeError) as exc:
        raise WorkloadError("AuNP sealed build contract is incomplete") from exc
    if record != build_contract_record(build):
        raise WorkloadError("AuNP sealed build contract was not re-derived")
    cpu_ranks = sealed_contract.get("cpu_ranks")
    repeats = sealed_contract.get("performance_repeats")
    device_values = sealed_contract.get("gpu_devices")
    if (
        type(cpu_ranks) is not int
        or cpu_ranks < 2
        or repeats != RELEASE_PERFORMANCE_REPEATS
        or not isinstance(device_values, list)
        or not all(isinstance(value, str) for value in device_values)
    ):
        raise WorkloadError("AuNP sealed task topology is invalid")
    if sealed_contract.get("qualification_profile") != aunp_qualification_profile():
        raise WorkloadError("AuNP sealed TM-only qualification profile differs")
    devices = normalized_devices(",".join(device_values))
    if list(devices) != device_values:
        raise WorkloadError("AuNP sealed GPU UUID spelling is not canonical")
    tasks = build_tasks(build, cpu_ranks, devices, repeats)
    if sealed_contract.get("tasks") != [task_spec(task) for task in tasks]:
        raise WorkloadError("AuNP sealed task schedule is not canonical")
    if sealed_contract.get("exact_comparisons") != exact_comparison_specs():
        raise WorkloadError("AuNP two-direction exact comparison contract differs")
    validate_release_gates(sealed_contract.get("gates"))
    if (
        sealed_contract.get("fixed_window_meep_time")
        != performance_adapter.FIXED_WINDOW_MEEP_TIME
        or sealed_contract.get("minimum_timesteps_per_phase")
        != performance_adapter.MINIMUM_TIMESTEPS_PER_PHASE
    ):
        raise WorkloadError("AuNP fixed-work performance contract differs")
    return build, devices, tasks


def _speedup_metrics(
    timings: dict[str, list[float]],
    cpu_role: str,
    gates: dict[str, Any],
) -> dict[str, Any]:
    pairs = (
        (
            "cpu-fp32-to-one-gpu",
            cpu_role,
            "performance-cuda-fp32-1g",
            gates["minimum_one_gpu_speedup"],
        ),
        (
            "cpu-fp32-to-two-gpu",
            cpu_role,
            "performance-cuda-fp32-2g",
            gates["minimum_two_gpu_speedup"],
        ),
        (
            "one-gpu-to-two-gpu",
            "performance-cuda-fp32-1g",
            "performance-cuda-fp32-2g",
            gates["minimum_multi_gpu_scaling"],
        ),
    )
    result = {}
    for name, reference, candidate, minimum in pairs:
        reference_values = timings[reference]
        candidate_values = timings[candidate]
        if (
            len(reference_values) != RELEASE_PERFORMANCE_REPEATS
            or len(candidate_values) != RELEASE_PERFORMANCE_REPEATS
        ):
            raise WorkloadError("AuNP mean speedup requires exactly two paired samples")
        reference_mean = statistics.fmean(reference_values)
        candidate_mean = statistics.fmean(candidate_values)
        mean_speedup = reference_mean / candidate_mean
        result[name] = {
            "reference_role": reference,
            "candidate_role": candidate,
            "reference_samples_seconds": reference_values,
            "candidate_samples_seconds": candidate_values,
            "reference_mean_seconds": reference_mean,
            "candidate_mean_seconds": candidate_mean,
            "mean_speedup": mean_speedup,
            "minimum": minimum,
            "outcome": "PASS" if mean_speedup >= minimum else "FAIL",
        }
    return result


def _timing_diagnostic(values: list[float]) -> dict[str, Any]:
    if len(values) != RELEASE_PERFORMANCE_REPEATS:
        raise WorkloadError("AuNP timing diagnostic requires exactly two samples")
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values)
    return {
        "sample_count": len(values),
        "samples_seconds": values,
        "arithmetic_mean_seconds": mean,
        "population_stddev_seconds": deviation,
        "coefficient_of_variation": deviation / mean,
        "release_gate": None,
    }


def performance_report(
    completed: list[dict[str, Any]],
    cpu_ranks: int,
    gates: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_release_gates(gates)
    cpu_role = f"performance-cpu-fp32-{cpu_ranks}r"
    roles = {
        cpu_role: [],
        "performance-cuda-fp32-1g": [],
        "performance-cuda-fp32-2g": [],
    }
    work_contracts = set()
    for record in completed:
        task = record.get("task", {})
        role = task.get("role")
        if role not in roles or task.get("counts_toward_performance") is not True:
            continue
        timesteps = record.get("phase_timestep_deltas")
        if (
            not isinstance(timesteps, list)
            or len(timesteps) != len(performance_adapter.PHASE_SPECS)
            or any(
                type(value) is not int
                or value < performance_adapter.MINIMUM_TIMESTEPS_PER_PHASE
                for value in timesteps
            )
        ):
            raise WorkloadError("AuNP performance timestep contract is invalid")
        work_contracts.add(tuple(timesteps))
        roles[role].append(record)
    if len(work_contracts) != 1 or any(
        len(records) != RELEASE_PERFORMANCE_REPEATS for records in roles.values()
    ):
        raise WorkloadError("AuNP performance work/repeat inventory differs")
    timing_fields = ("fdtd_wall_seconds", "workload_end_to_end_seconds")
    performance = {}
    diagnostics = {}
    for field in timing_fields:
        timings = {
            role: [float(record[field]) for record in records]
            for role, records in roles.items()
        }
        if any(
            not math.isfinite(value) or value <= 0
            for values in timings.values()
            for value in values
        ):
            raise WorkloadError(f"AuNP performance {field} has invalid timing")
        performance[field] = _speedup_metrics(timings, cpu_role, gates)
        diagnostics[field] = {
            role: _timing_diagnostic(values) for role, values in timings.items()
        }
    return performance, diagnostics


def performance_outcome(performance: dict[str, Any]) -> str:
    speed_values = [
        metric
        for timing in performance.values()
        for metric in timing.values()
    ]
    return (
        "PASS"
        if all(item.get("outcome") == "PASS" for item in speed_values)
        else "FAIL"
    )


def clock_event_report(
    output: pathlib.Path, completed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Re-derive bound per-task clock-event evidence and release policy."""

    tasks = []
    performance_active = 0
    nonperformance_active = 0
    for record in completed:
        task = record.get("task", {})
        if task.get("backend") != "cuda":
            continue
        replay = matrix.load_stable_json_record(
            output,
            record.get("telemetry_replay", {}),
            f"AuNP {task.get('role')} clock-event telemetry",
        )
        gpu = replay.get("gpu") if isinstance(replay, dict) else None
        expected_policy = (
            "reject-active-during-alive-performance-samples"
            if task.get("counts_toward_performance") is True
            else "disclose-active-reasons"
        )
        devices = task.get("devices")
        reasons = gpu.get("clock_event_reasons") if isinstance(gpu, dict) else None
        if (
            replay.get("status") != "PASS"
            or not isinstance(devices, list)
            or not devices
            or gpu.get("clock_event_policy") != expected_policy
            or not isinstance(reasons, dict)
            or set(reasons) != set(devices)
            or gpu.get("process_coverage", {}).get("outcome") != "PASS"
        ):
            raise WorkloadError("AuNP clock-event telemetry identity differs")
        per_gpu = {}
        active = 0
        for uuid in devices:
            retained = reasons[uuid]
            if not isinstance(retained, dict) or set(retained) != set(
                telemetry_replay.CLOCK_EVENT_REASON_METRICS
            ):
                raise WorkloadError("AuNP clock-event reason inventory differs")
            per_gpu[uuid] = {}
            for name in telemetry_replay.CLOCK_EVENT_REASON_METRICS:
                value = retained[name]
                if not isinstance(value, dict) or set(value) != {
                    "active_sample_count",
                    "active_alive_sample_count",
                    "first_active_unix_seconds",
                    "last_active_unix_seconds",
                }:
                    raise WorkloadError("AuNP clock-event reason schema differs")
                total = value["active_sample_count"]
                alive = value["active_alive_sample_count"]
                if (
                    type(total) is not int
                    or type(alive) is not int
                    or total < 0
                    or alive < 0
                    or alive > total
                ):
                    raise WorkloadError("AuNP clock-event reason count differs")
                active += alive
                per_gpu[uuid][name] = value
        if task.get("counts_toward_performance") is True:
            performance_active += active
            if active:
                raise WorkloadError(
                    "AuNP measured performance contains an active clock-event reason"
                )
        else:
            nonperformance_active += active
        tasks.append(
            {
                "role": task["role"],
                "repeat": task["repeat"],
                "kind": task["kind"],
                "counts_toward_performance": task["counts_toward_performance"],
                "policy": expected_policy,
                "active_alive_sample_count": active,
                "per_gpu": per_gpu,
            }
        )
    if not tasks:
        raise WorkloadError("AuNP clock-event telemetry has no GPU tasks")
    return {
        "schema": "gpmeep-aunp-clock-event-report-v1",
        "performance_policy": "zero-active-clock-event-reasons",
        "nonperformance_policy": "disclose-active-clock-event-reasons",
        "performance_active_alive_sample_count": performance_active,
        "nonperformance_active_alive_sample_count": nonperformance_active,
        "performance_outcome": "PASS" if performance_active == 0 else "FAIL",
        "tasks": tasks,
    }


def task_label(task: Task) -> str:
    return f"{task.role}-repeat-{task.repeat:02d}"


def observer_specs(
    task: Task,
    root_pid: int,
    output: pathlib.Path,
    sealed_contract: dict[str, Any],
) -> list[observed.ObserverSpec]:
    telemetry = output.joinpath(*task_telemetry_relative(task).parts)
    interval = sealed_contract["limits"]["telemetry_interval_seconds"]
    label = task_label(task)
    host = observed.ObserverSpec(
        "host",
        (
            str(task.lane.build.python),
            str(SCRIPT_DIR / "capture_host_telemetry_v2.py"),
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
    )
    specs = [host]
    if task.lane.backend == "cuda":
        command = [
            str(task.lane.build.python),
            str(SCRIPT_DIR / "capture_gpu_telemetry_v2.py"),
            "--root-pid",
            str(root_pid),
            "--output",
            str(telemetry / "gpu.jsonl"),
            "--label",
            label,
        ]
        for uuid in task.lane.devices:
            command.extend(("--selected-uuid", uuid))
        command.extend(("--interval-seconds", repr(interval)))
        specs.append(
            observed.ObserverSpec(
                "gpu", tuple(command), telemetry / "gpu-monitor.log"
            )
        )
    return specs


def _code_sha256(sealed_contract: dict[str, Any], name: str) -> str:
    evidence = sealed_contract.get("evidence_code")
    if not isinstance(evidence, dict) or not isinstance(evidence.get(name), dict):
        raise WorkloadError(f"AuNP evidence code {name} is absent")
    digest = evidence[name].get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise WorkloadError(f"AuNP evidence code {name} digest is invalid")
    return digest


def derive_telemetry_replay(
    task: Task,
    process: dict[str, Any],
    output: pathlib.Path,
    sealed_contract: dict[str, Any],
) -> dict[str, Any]:
    telemetry = output.joinpath(*task_telemetry_relative(task).parts)
    gates = sealed_contract["gates"]
    gpu = task.lane.backend == "cuda"
    return telemetry_replay.verify_pair(
        host_stream=telemetry / "host.jsonl",
        host_terminal=telemetry / "host.jsonl.COMPLETE.json",
        gpu_stream=(telemetry / "gpu.jsonl") if gpu else None,
        gpu_terminal=(telemetry / "gpu.jsonl.COMPLETE.json") if gpu else None,
        expected_root_pid=process["command_pid"],
        expected_label=task_label(task),
        expected_host_monitor_sha256=_code_sha256(
            sealed_contract, "host_telemetry"
        ),
        expected_gpu_uuids=task.lane.devices,
        expected_gpu_monitor_sha256=(
            _code_sha256(sealed_contract, "gpu_telemetry") if gpu else None
        ),
        expected_nvidia_smi_sha256=(
            sealed_contract["nvidia_smi"]["sha256"] if gpu else None
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
        expected_interval_seconds=sealed_contract["limits"][
            "telemetry_interval_seconds"
        ],
        expected_process_started_unix_seconds=process["started_unix_seconds"],
        expected_process_ended_unix_seconds=process["ended_unix_seconds"],
        reject_active_clock_event_reasons=task.counts_toward_performance,
    )


def publish_telemetry_replay(
    task: Task,
    process: dict[str, Any],
    observer_records: list[dict[str, Any]],
    output: pathlib.Path,
    sealed_contract: dict[str, Any],
) -> dict[str, Any]:
    specs = observer_specs(task, process["command_pid"], output, sealed_contract)
    if len(observer_records) != len(specs):
        raise WorkloadError("AuNP task observer inventory differs")
    for record, spec in zip(observer_records, specs, strict=True):
        observed.verify_observer_record(output, record, spec)
    report = derive_telemetry_replay(task, process, output, sealed_contract)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    atomic_write_json(path, report)
    return matrix.stable_file_record(
        path, output, f"{task.role} telemetry replay"
    )


def validate_telemetry_replay(
    task: Task,
    process: dict[str, Any],
    observer_records: list[dict[str, Any]],
    record: Any,
    output: pathlib.Path,
    sealed_contract: dict[str, Any],
) -> dict[str, Any]:
    specs = observer_specs(task, process["command_pid"], output, sealed_contract)
    if len(observer_records) != len(specs):
        raise WorkloadError("AuNP task observer inventory differs")
    for retained, spec in zip(observer_records, specs, strict=True):
        observed.verify_observer_record(output, retained, spec)
    path = output.joinpath(*task_telemetry_relative(task).parts) / "replay.json"
    expected_record = matrix.stable_file_record(
        path, output, f"{task.role} telemetry replay"
    )
    if record != expected_record:
        raise WorkloadError("AuNP task telemetry replay record differs")
    with StableFile(path, f"{task.role} telemetry replay") as stable:
        with stable.file_object() as handle:
            retained_report = json.load(handle)
    if retained_report != derive_telemetry_replay(
        task, process, output, sealed_contract
    ):
        raise WorkloadError("AuNP task telemetry replay was not re-derived")
    return record


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def derive_lane_evidence(
    task: Task,
    lane_output: pathlib.Path,
    output: pathlib.Path,
    *,
    publish_replay: bool,
) -> dict[str, Any]:
    complete = matrix.stable_file_record(
        lane_output / "COMPLETE", output, f"{task.role} COMPLETE"
    )
    replay_record = None
    if task.kind == "exact":
        validated = matrix.validate_lane_output(task.lane, lane_output, "aunp")
        lane = matrix.COMPARATOR.load_lane(
            lane_output, matrix.COMPARATOR.AUNP_SCHEMA
        )
        matrix.COMPARATOR.validate_aunp_exact_stage_resume(lane)
        timesteps = [
            int(item["timestep_delta"])
            for item in validated["timestep_contract"]
        ]
        provenance = _canonical(validated["provenance"])
        values = {
            "fdtd_wall_seconds": validated["fdtd_wall_seconds"],
            "phase_wall_seconds": validated["phase_wall_seconds"],
            "phase_timestep_deltas": timesteps,
            "workload_end_to_end_seconds": validated[
                "workload_end_to_end_seconds"
            ],
            "provenance": provenance,
        }
    else:
        replay = performance_replay.replay(
            REPO, lane_output, task.lane.backend, task.lane.ranks
        )
        replay_path = output.joinpath(*task_replay_relative(task).parts)
        if publish_replay:
            atomic_write_json(replay_path, replay)
        with StableFile(replay_path, f"{task.role} lane replay") as stable:
            with stable.file_object() as handle:
                retained = json.load(handle)
        if retained != replay:
            raise WorkloadError(f"{task.role} lane replay was not re-derived")
        replay_record = matrix.stable_file_record(
            replay_path, output, f"{task.role} lane replay"
        )
        values = {
            "fdtd_wall_seconds": replay["fdtd_wall_seconds"],
            "phase_wall_seconds": replay["phase_wall_seconds"],
            "phase_timestep_deltas": replay["phase_timestep_deltas"],
            "workload_end_to_end_seconds": replay[
                "workload_end_to_end_seconds"
            ],
            "provenance": {
                "receipt_id": replay["receipt_id"],
                "source_start": replay["source_start"],
            },
        }
    return {**values, "complete": complete, "lane_replay": replay_record}


def comparison_relative(spec: dict[str, str]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("comparisons", spec["name"])


def comparison_log_relative(spec: dict[str, str]) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("logs", "comparisons", f"{spec['name']}.log")


def comparison_command(
    spec: dict[str, str],
    build: matrix.Build,
    archive: pathlib.Path,
    output: pathlib.Path,
    exact_paths: dict[str, pathlib.Path],
) -> list[str]:
    return matrix.comparison_command(
        build.python,
        "aunp",
        output.joinpath(*comparison_relative(spec).parts),
        exact_paths[spec["reference_role"]],
        [
            (
                spec["candidate_role"],
                spec["candidate_class"],
                exact_paths[spec["candidate_role"]],
            )
        ],
        archive,
    )


def derive_comparison_report(
    spec: dict[str, str],
    archive: pathlib.Path,
    exact_paths: dict[str, pathlib.Path],
) -> dict[str, Any]:
    args = argparse.Namespace(
        reference=exact_paths[spec["reference_role"]],
        candidate=[
            (
                spec["candidate_role"],
                spec["candidate_class"],
                exact_paths[spec["candidate_role"]],
            )
        ],
        expected_archive=archive,
    )
    return matrix.COMPARATOR.compare_aunp(args)


def validate_comparison(
    output: pathlib.Path,
    record: Any,
    spec: dict[str, str],
    build: matrix.Build,
    archive: pathlib.Path,
    exact_paths: dict[str, pathlib.Path],
) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {
        "spec",
        "process",
        "complete",
        "report",
    }:
        raise WorkloadError("AuNP exact comparison record is not exact")
    if record["spec"] != spec:
        raise WorkloadError("AuNP exact comparison direction differs")
    matrix._verify_process_record(
        output,
        record["process"],
        comparison_log_relative(spec).as_posix(),
        f"AuNP comparison {spec['name']}",
    )
    expected_command = comparison_command(
        spec, build, archive, output, exact_paths
    )
    if record["process"]["command"] != expected_command:
        raise WorkloadError("AuNP exact comparison command differs")
    directory = output.joinpath(*comparison_relative(spec).parts)
    complete = matrix.stable_file_record(
        directory / "COMPLETE", output, f"AuNP comparison {spec['name']} COMPLETE"
    )
    report_record = matrix.stable_file_record(
        directory / "report.json", output, f"AuNP comparison {spec['name']} report"
    )
    if record["complete"] != complete or record["report"] != report_record:
        raise WorkloadError("AuNP exact comparison artifact record differs")
    with StableFile(directory / "report.json", "AuNP exact comparison report") as stable:
        with stable.file_object() as handle:
            retained = json.load(handle)
    # ``compare_aunp`` represents rank-indexed provenance maps with integer
    # keys in memory, while JSON necessarily restores those object keys as
    # strings.  Compare the retained report with the same JSON-canonical form
    # that the comparison subprocess published; otherwise every multi-rank
    # exact comparison is rejected despite a byte-for-byte reproducible PASS
    # report.
    derived = _canonical(derive_comparison_report(spec, archive, exact_paths))
    if retained != derived or derived.get("outcome") != "PASS":
        raise WorkloadError("AuNP exact comparison did not independently re-derive PASS")
    comparisons = derived.get("comparisons")
    if (
        not isinstance(comparisons, list)
        or len(comparisons) != 2
        or comparisons[0].get("label") != "packaged-upstream-expected"
        or comparisons[0].get("outcome") != "PASS"
        or comparisons[1].get("label") != spec["candidate_role"]
        or comparisons[1].get("comparison_class") != spec["candidate_class"]
        or comparisons[1].get("outcome") != "PASS"
    ):
        raise WorkloadError("AuNP exact comparison does not cover oracle and peer")
    return record


def write_checkpoint(
    output: pathlib.Path,
    sealed_contract: dict[str, Any],
    hardware_before: dict[str, Any],
    capacity_before: dict[str, Any],
    completed: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    atomic_write_json(
        output / "CHECKPOINT.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "contract": sealed_contract,
            "hardware_before": hardware_before,
            "capacity_before": capacity_before,
            "completed_tasks": completed,
            "exact_comparisons": comparisons,
        },
    )


def load_checkpoint(
    output: pathlib.Path, sealed_contract: dict[str, Any]
) -> dict[str, Any]:
    with StableFile(output / "CHECKPOINT.json", "AuNP hybrid checkpoint") as stable:
        with stable.file_object() as handle:
            checkpoint = json.load(handle)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "schema",
        "contract",
        "hardware_before",
        "capacity_before",
        "completed_tasks",
        "exact_comparisons",
    }:
        raise WorkloadError("AuNP hybrid checkpoint schema is not exact")
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise WorkloadError("AuNP hybrid checkpoint version differs")
    if checkpoint["contract"] != sealed_contract:
        raise WorkloadError("AuNP hybrid resume invocation differs from checkpoint")
    if not isinstance(checkpoint["completed_tasks"], list) or not isinstance(
        checkpoint["exact_comparisons"], list
    ):
        raise WorkloadError("AuNP hybrid checkpoint inventories are invalid")
    return checkpoint


def append_event(output: pathlib.Path, journal: dict[str, Any], event: dict[str, Any]) -> None:
    journal["events"].append(event)
    atomic_write_json(output / "JOURNAL.json", journal)


def journal_prefix(
    journal: Any,
    sealed_contract: dict[str, Any],
    completed: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(comparisons) > len(exact_comparison_specs()):
        raise WorkloadError("AuNP hybrid journal has too many comparisons")
    if comparisons and len(completed) < 4:
        raise WorkloadError("AuNP comparisons precede both exact tasks")
    if len(completed) > 4 and len(comparisons) != 2:
        raise WorkloadError("AuNP performance tasks precede exact comparisons")
    if (
        not isinstance(journal, dict)
        or journal.get("schema") != JOURNAL_SCHEMA
        or journal.get("contract") != sealed_contract
        or not isinstance(journal.get("events"), list)
        or not journal["events"]
    ):
        raise WorkloadError("AuNP hybrid journal identity differs")
    first = journal["events"][0]
    if first.get("state") != "controller-started" or not math.isfinite(
        float(first.get("unix_seconds", math.nan))
    ):
        raise WorkloadError("AuNP hybrid journal has no controller start")
    expected = [first]
    comparison_index = 0
    for index, record in enumerate(completed):
        spec = record["task"]
        expected.extend(
            (
                {"state": "task-started", "task": spec},
                {
                    "state": "task-process-ended",
                    "task": spec,
                    "process": record["process"],
                },
                {
                    "state": "task-validated",
                    "task": spec,
                    "complete": record["complete"],
                },
            )
        )
        if index == 3:
            for comparison in comparisons:
                expected.extend(
                    (
                        {
                            "state": "comparison-process-ended",
                            "spec": comparison["spec"],
                            "process": comparison["process"],
                        },
                        {
                            "state": "comparison-validated",
                            "spec": comparison["spec"],
                            "complete": comparison["complete"],
                        },
                    )
                )
                comparison_index += 1
    if comparison_index != len(comparisons):
        raise WorkloadError("AuNP comparisons precede both exact tasks")
    if journal["events"][: len(expected)] != expected:
        raise WorkloadError("AuNP hybrid journal disagrees with checkpoint")
    return expected


def validate_completed_task(
    output: pathlib.Path,
    record: Any,
    task: Task,
    archive: pathlib.Path,
    devices: tuple[str, str],
    sealed_contract: dict[str, Any],
) -> dict[str, Any]:
    lane_keys = {
        "fdtd_wall_seconds",
        "phase_wall_seconds",
        "phase_timestep_deltas",
        "workload_end_to_end_seconds",
        "provenance",
        "complete",
        "lane_replay",
    }
    keys = {
        "task",
        "output",
        "process",
        "observers",
        "telemetry_replay",
        "hardware",
        "capacity_before",
        "capacity_after",
    } | lane_keys
    if not isinstance(record, dict) or set(record) != keys:
        raise WorkloadError("AuNP completed-task record schema is not exact")
    if record["task"] != task_spec(task):
        raise WorkloadError("AuNP completed-task order differs")
    relative = task_relative(task)
    if record["output"] != relative.as_posix():
        raise WorkloadError("AuNP completed-task output path differs")
    matrix._verify_process_record(
        output, record["process"], task_log_relative(task).as_posix(), task.role
    )
    lane_output = output.joinpath(*relative.parts)
    if record["process"]["command"] != task_command(task, archive, lane_output):
        raise WorkloadError("AuNP completed-task command differs")
    matrix.validate_hardware_evidence(record["hardware"], list(devices))
    validate_capacity(record["capacity_before"], sealed_contract["gates"], task)
    validate_capacity_record(record["capacity_after"])
    validate_telemetry_replay(
        task,
        record["process"],
        record["observers"],
        record["telemetry_replay"],
        output,
        sealed_contract,
    )
    derived = derive_lane_evidence(
        task, lane_output, output, publish_replay=False
    )
    if any(record[name] != derived[name] for name in lane_keys):
        raise WorkloadError("AuNP completed-task lane evidence was not re-derived")
    return record


def _exact_paths(
    output: pathlib.Path, completed: list[dict[str, Any]]
) -> dict[str, pathlib.Path]:
    return {
        record["task"]["role"]: output / record["output"]
        for record in completed
        if record["task"]["kind"] == "exact"
    }


def run_missing_comparisons(
    output: pathlib.Path,
    comparisons: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    journal: dict[str, Any],
    build: matrix.Build,
    archive: pathlib.Path,
    args: argparse.Namespace,
    checkpoint_callback: Any = None,
) -> None:
    exact_paths = _exact_paths(output, completed)
    if set(exact_paths) != {"exact-cuda-fp32-1g", "exact-cuda-fp32-2g"}:
        raise WorkloadError("AuNP exact comparisons precede complete exact lanes")
    environment = {
        "PATH": f"{build.python.parent}:/usr/bin:/bin",
        "HOME": str(output / "runtime" / "comparisons"),
        "XDG_CACHE_HOME": str(output / "runtime" / "comparisons"),
        "TMPDIR": "/tmp",
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for spec in exact_comparison_specs()[len(comparisons) :]:
        directory = output.joinpath(*comparison_relative(spec).parts)
        command = comparison_command(spec, build, archive, output, exact_paths)
        process = matrix.run_bounded(
            command,
            environment,
            output.joinpath(*comparison_log_relative(spec).parts),
            args.comparison_timeout_seconds,
            args.stdout_limit_mib * 1024 * 1024,
        )
        append_event(
            output,
            journal,
            {"state": "comparison-process-ended", "spec": spec, "process": process},
        )
        if process["returncode"] != 0 or process["timed_out"] or process[
            "output_limited"
        ]:
            raise WorkloadError(f"AuNP exact comparison failed: {spec['name']}")
        record = {
            "spec": spec,
            "process": process,
            "complete": matrix.stable_file_record(
                directory / "COMPLETE", output, f"AuNP {spec['name']} COMPLETE"
            ),
            "report": matrix.stable_file_record(
                directory / "report.json", output, f"AuNP {spec['name']} report"
            ),
        }
        validate_comparison(output, record, spec, build, archive, exact_paths)
        comparisons.append(record)
        append_event(
            output,
            journal,
            {
                "state": "comparison-validated",
                "spec": spec,
                "complete": record["complete"],
            },
        )
        if checkpoint_callback is not None:
            checkpoint_callback()


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# gpmeep hybrid AuNP qualification",
        "",
        f"Outcome: **{report['outcome']}**",
        "",
        "Both GPU exact TM-only lanes (minimum DFT window 20 um/c, decay 5e-8) "
        "are directly checked against the packaged CPU-FP64 spectra/resonance "
        "oracle and against each other in both directions.",
        "",
        "Performance uses exactly two measured runs per topology. Every raw time is "
        "retained, and release speedups are ratios of arithmetic-mean times. Timing "
        "dispersion is diagnostic only.",
        "",
        "| Timing | Metric | Reference mean | Candidate mean | Mean speedup | Gate | Outcome |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for timing, metrics in report["performance"].items():
        for name, metric in metrics.items():
            lines.append(
                f"| {timing} | {name} | {metric['reference_mean_seconds']:.6f}s | "
                f"{metric['candidate_mean_seconds']:.6f}s | "
                f"{metric['mean_speedup']:.4f}x | "
                f"{metric['minimum']:.4f}x | **{metric['outcome']}** |"
            )
    lines.extend(("", f"Resume boundaries: {len(report['resume_history'])}.", ""))
    clock_events = report["clock_event_telemetry"]
    lines.extend(
        (
            "## Bound thermal/throttle evidence",
            "",
            "Measured GPU performance requires zero active software-thermal, "
            "hardware-thermal, hardware-power-brake, and software-power-cap "
            "clock-event samples.",
            "",
            f"Performance active samples: "
            f"{clock_events['performance_active_alive_sample_count']} "
            f"(**{clock_events['performance_outcome']}**).",
            f"Non-performance active samples (disclosed, not timed): "
            f"{clock_events['nonperformance_active_alive_sample_count']}.",
            "",
        )
    )
    return "\n".join(lines)


def publish_terminal(output: pathlib.Path, outcome: str) -> None:
    if outcome != "PASS":
        raise WorkloadError("AuNP hybrid performance gates failed")
    atomic_write_json(
        output / "PENDING_COMPLETE",
        {
            "schema": COMPLETE_SCHEMA,
            "outcome": outcome,
            "report": file_record(output / "report.json", output),
            "markdown": file_record(output / "report.md", output),
            "journal": file_record(output / "JOURNAL.json", output),
        },
    )
    verify_hybrid_complete(output, "PENDING_COMPLETE")
    os.replace(output / "PENDING_COMPLETE", output / "COMPLETE")


def verify_hybrid_complete(
    output: pathlib.Path, terminal_name: str = "COMPLETE"
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {"COMPLETE", "PENDING_COMPLETE"}:
        raise WorkloadError("AuNP hybrid terminal filename is invalid")
    with StableFile(output / terminal_name, "AuNP hybrid COMPLETE") as stable:
        with stable.file_object() as handle:
            complete = json.load(handle)
    if set(complete) != {"schema", "outcome", "report", "markdown", "journal"} or (
        complete.get("schema") != COMPLETE_SCHEMA or complete.get("outcome") != "PASS"
    ):
        raise WorkloadError("AuNP hybrid COMPLETE is not an exact PASS")
    report = matrix.load_stable_json_record(
        output, complete["report"], "AuNP hybrid report"
    )
    matrix.stable_record(output, complete["markdown"], "AuNP hybrid Markdown").close()
    journal = matrix.load_stable_json_record(
        output, complete["journal"], "AuNP hybrid journal"
    )
    checkpoint = matrix.load_stable_json_record(
        output, report.get("recovery_checkpoint", {}), "AuNP hybrid checkpoint"
    )
    report_keys = {
        "schema",
        "outcome",
        "input",
        "evidence_code",
        "source_snapshot_sha256",
        "build",
        "cpu_physical_cores",
        "gpu_devices",
        "performance_repeats",
        "qualification_profile",
        "fixed_window_meep_time",
        "tasks",
        "exact_comparisons",
        "performance",
        "timing_diagnostics",
        "clock_event_telemetry",
        "hardware",
        "capacity_initial",
        "capacity_final",
        "resume_history",
        "recovery_checkpoint",
    }
    checkpoint_keys = {
        "schema",
        "contract",
        "hardware_before",
        "capacity_before",
        "completed_tasks",
        "exact_comparisons",
    }
    if (
        not isinstance(report, dict)
        or set(report) != report_keys
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
    ):
        raise WorkloadError("AuNP hybrid report is not a PASS")
    if not isinstance(checkpoint, dict) or set(checkpoint) != checkpoint_keys or (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
    ):
        raise WorkloadError("AuNP hybrid checkpoint is not exact")
    sealed_contract = checkpoint.get("contract")
    build, devices, tasks = reconstruct_contract_tasks(sealed_contract)
    evidence_code = sealed_contract["evidence_code"]
    expected_code = {
        "controller": pathlib.Path(__file__),
        "matrix_common": SCRIPT_DIR / "run_user_workload_matrix.py",
        "observed_runner": SCRIPT_DIR / "run_observed_process.py",
        "exact_adapter": SCRIPT_DIR / "run_aunp_workload.py",
        "exact_stage_state": SCRIPT_DIR / "aunp_exact_resume.py",
        "performance_adapter": SCRIPT_DIR / "run_aunp_performance_workload.py",
        "performance_replay": SCRIPT_DIR / "verify_aunp_performance_workload.py",
        "host_telemetry": SCRIPT_DIR / "capture_host_telemetry_v2.py",
        "gpu_telemetry": SCRIPT_DIR / "capture_gpu_telemetry_v2.py",
        "telemetry_replay": SCRIPT_DIR / "verify_telemetry_v2.py",
        "comparator": SCRIPT_DIR / "compare_user_workloads.py",
        "common": SCRIPT_DIR / "common.py",
    }
    if not isinstance(evidence_code, dict) or set(evidence_code) != set(expected_code):
        raise WorkloadError("AuNP evidence-code inventory differs")
    for name, path in expected_code.items():
        stable = matrix.stable_record(REPO, evidence_code[name], f"AuNP {name}")
        if stable.path != path.resolve(strict=True):
            stable.close(verify=False)
            raise WorkloadError(f"AuNP evidence-code path differs: {name}")
        stable.close()
    input_record = sealed_contract["input"]
    if input_record.get("sha256") != AUNP_ARCHIVE_SHA256:
        raise WorkloadError("AuNP hybrid archive digest differs")
    with StableFile(
        pathlib.Path(input_record["path"]), "AuNP hybrid archive", expected=input_record
    ):
        pass
    if (
        report["input"] != input_record
        or report["evidence_code"] != evidence_code
        or report["build"] != sealed_contract["build"]
        or report["cpu_physical_cores"] != sealed_contract["cpu_ranks"]
        or report["gpu_devices"] != sealed_contract["gpu_devices"]
        or report["performance_repeats"]
        != sealed_contract["performance_repeats"]
        or report["qualification_profile"]
        != sealed_contract["qualification_profile"]
        or report["fixed_window_meep_time"]
        != sealed_contract["fixed_window_meep_time"]
    ):
        raise WorkloadError("AuNP hybrid report differs from its sealed contract")
    nvidia_record = sealed_contract.get("nvidia_smi")
    if nvidia_record != absolute_file_record(
        pathlib.Path("/usr/bin/nvidia-smi"), "nvidia-smi"
    ):
        raise WorkloadError("AuNP hybrid nvidia-smi identity differs")
    completed = checkpoint.get("completed_tasks")
    comparisons = checkpoint.get("exact_comparisons")
    if (
        not isinstance(completed, list)
        or len(completed) != len(tasks)
        or not isinstance(comparisons, list)
        or len(comparisons) != 2
        or report.get("tasks") != completed
        or report.get("exact_comparisons") != comparisons
    ):
        raise WorkloadError("AuNP hybrid task/comparison inventory is incomplete")
    archive = pathlib.Path(input_record["path"]).resolve(strict=True)
    for record, task in zip(completed, tasks, strict=True):
        validate_completed_task(
            output, record, task, archive, devices, sealed_contract
        )
    exact_paths = _exact_paths(output, completed)
    for record, spec in zip(comparisons, exact_comparison_specs(), strict=True):
        validate_comparison(output, record, spec, build, archive, exact_paths)
    performance, diagnostics = performance_report(
        completed, sealed_contract["cpu_ranks"], sealed_contract["gates"]
    )
    if (
        report.get("performance") != performance
        or report.get("timing_diagnostics") != diagnostics
        or performance_outcome(performance) != "PASS"
    ):
        raise WorkloadError("AuNP hybrid performance evidence was not re-derived")
    derived_clock_events = clock_event_report(output, completed)
    if (
        report.get("clock_event_telemetry") != derived_clock_events
        or derived_clock_events.get("performance_outcome") != "PASS"
    ):
        raise WorkloadError("AuNP clock-event evidence was not re-derived")
    if checkpoint["completed_tasks"] != completed or checkpoint[
        "exact_comparisons"
    ] != comparisons:
        raise WorkloadError("AuNP hybrid checkpoint differs from report")
    matrix.validate_hardware_evidence(report.get("hardware"), list(devices))
    if checkpoint["hardware_before"] != report["hardware"]["before"]:
        raise WorkloadError("AuNP hybrid initial hardware differs")
    if report.get("capacity_initial") != checkpoint.get("capacity_before"):
        raise WorkloadError("AuNP hybrid initial capacity differs")
    validate_capacity(report["capacity_initial"], sealed_contract["gates"], None)
    validate_capacity_record(report["capacity_final"])
    actual_history = matrix.resume_history_records(output)
    if report.get("resume_history") != actual_history:
        raise WorkloadError("AuNP hybrid resume history differs")
    if report.get("source_snapshot_sha256") != matrix._source_identity(build):
        raise WorkloadError("AuNP hybrid source snapshot differs")
    if report.get("build") != build_contract_record(build):
        raise WorkloadError("AuNP hybrid report build differs")
    expected_events = journal_prefix(
        journal, sealed_contract, completed, comparisons
    )
    if journal["events"] != expected_events:
        raise WorkloadError("AuNP hybrid journal contains unsealed events")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    archive = verify_file(args.archive, AUNP_ARCHIVE_SHA256, "AuNP archive")
    if args.resume:
        if args.output.is_symlink():
            raise WorkloadError("AuNP resume output is a symlink")
        output = args.output.resolve(strict=True)
        if (output / "COMPLETE").exists():
            raise WorkloadError("completed AuNP hybrid evidence cannot be resumed")
    else:
        if args.output.is_symlink():
            raise WorkloadError("AuNP output is a symlink")
        output = pathlib.Path(os.path.abspath(args.output))
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.parent.resolve(strict=True) != output.parent:
            raise WorkloadError("AuNP output parent contains a symlink")
        output.mkdir(exist_ok=False)
    lock_path = output / "LOCK"
    if args.resume and not stat.S_ISREG(lock_path.lstat().st_mode):
        raise WorkloadError("AuNP hybrid lock is not a regular file")
    lock = lock_path.open("r+" if args.resume else "x")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise WorkloadError(f"could not acquire AuNP hybrid lock: {exc}") from exc

    archive_stable = StableFile(
        archive,
        "AuNP archive",
        expected={
            "size_bytes": archive.stat().st_size,
            "sha256": AUNP_ARCHIVE_SHA256,
        },
    )
    code_stables: dict[str, StableFile] = {}
    nvidia_stable: StableFile | None = None
    try:
        code_paths = {
            "controller": pathlib.Path(__file__),
            "matrix_common": SCRIPT_DIR / "run_user_workload_matrix.py",
            "observed_runner": SCRIPT_DIR / "run_observed_process.py",
            "exact_adapter": SCRIPT_DIR / "run_aunp_workload.py",
            "exact_stage_state": SCRIPT_DIR / "aunp_exact_resume.py",
            "performance_adapter": SCRIPT_DIR / "run_aunp_performance_workload.py",
            "performance_replay": SCRIPT_DIR / "verify_aunp_performance_workload.py",
            "host_telemetry": SCRIPT_DIR / "capture_host_telemetry_v2.py",
            "gpu_telemetry": SCRIPT_DIR / "capture_gpu_telemetry_v2.py",
            "telemetry_replay": SCRIPT_DIR / "verify_telemetry_v2.py",
            "comparator": SCRIPT_DIR / "compare_user_workloads.py",
            "common": SCRIPT_DIR / "common.py",
        }
        code_stables = {
            name: StableFile(path, f"AuNP hybrid {name}")
            for name, path in code_paths.items()
        }
        evidence_code = {
            name: matrix.stable_identity_record(stable, REPO)
            for name, stable in code_stables.items()
        }
        nvidia_stable = StableFile(
            pathlib.Path("/usr/bin/nvidia-smi"), "AuNP nvidia-smi"
        )
        nvidia_smi = absolute_file_record(nvidia_stable.path, "AuNP nvidia-smi")
        build = matrix.load_build(
            "shared-fp32",
            args.fp32_python,
            args.fp32_mpiexec,
            args.fp32_receipt,
            "cuda-mpi-python-fp32",
            True,
            True,
        )
        available = matrix.physical_core_count()
        cpu_ranks = args.cpu_ranks or available
        if cpu_ranks < 2 or cpu_ranks > available:
            raise WorkloadError(f"cpu-ranks must be in [2,{available}]")
        devices = normalized_devices(args.gpu_devices)
        tasks = build_tasks(build, cpu_ranks, devices, args.performance_repeats)
        sealed_contract = contract(
            args,
            archive,
            evidence_code,
            build,
            cpu_ranks,
            devices,
            tasks,
            nvidia_smi,
        )
        current_hardware = matrix.hardware_snapshot(devices)
        current_capacity = capacity_snapshot(output)
        completed: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        if args.resume:
            checkpoint = load_checkpoint(output, sealed_contract)
            hardware_before = checkpoint["hardware_before"]
            capacity_before = checkpoint["capacity_before"]
            matrix.validate_hardware_evidence(
                {"before": hardware_before, "after": current_hardware}, list(devices)
            )
            validate_capacity(capacity_before, sealed_contract["gates"], None)
            if len(checkpoint["completed_tasks"]) > len(tasks):
                raise WorkloadError("AuNP checkpoint contains too many tasks")
            for record, task in zip(
                checkpoint["completed_tasks"], tasks, strict=False
            ):
                completed.append(
                    validate_completed_task(
                        output, record, task, archive, devices, sealed_contract
                    )
                )
            comparisons = checkpoint["exact_comparisons"]
            if len(comparisons) > 2:
                raise WorkloadError("AuNP checkpoint contains too many comparisons")
            if comparisons and len(completed) < 4:
                raise WorkloadError("AuNP comparisons precede exact lanes")
            if len(completed) > 4 and len(comparisons) != 2:
                raise WorkloadError(
                    "AuNP performance checkpoint precedes exact comparisons"
                )
            exact_paths = _exact_paths(output, completed)
            for record, spec in zip(
                comparisons, exact_comparison_specs(), strict=False
            ):
                validate_comparison(
                    output, record, spec, build, archive, exact_paths
                )
            with StableFile(output / "JOURNAL.json", "AuNP hybrid journal") as stable:
                with stable.file_object() as handle:
                    journal = json.load(handle)
            expected_events = journal_prefix(
                journal, sealed_contract, completed, comparisons
            )
            partial_paths: list[pathlib.PurePosixPath] = []
            partial_paths.append(pathlib.PurePosixPath("PENDING_COMPLETE"))
            if len(completed) >= 4 and len(comparisons) < 2:
                spec = exact_comparison_specs()[len(comparisons)]
                partial_paths.extend(
                    (comparison_relative(spec), comparison_log_relative(spec))
                )
            elif len(completed) < len(tasks):
                task = tasks[len(completed)]
                if task.kind != "exact":
                    partial_paths.append(task_relative(task))
                partial_paths.extend(
                    (
                        task_runtime_relative(task),
                        task_log_relative(task),
                        task_telemetry_relative(task),
                        task_replay_relative(task),
                    )
                )
            matrix.archive_interrupted_attempt(output, partial_paths)
            journal["events"] = expected_events
            atomic_write_json(output / "JOURNAL.json", journal)
            atomic_write_json(
                output / "RUNNING.json",
                {
                    "schema": SCHEMA,
                    "resumed_unix_seconds": time.time(),
                    "completed_tasks": len(completed),
                    "completed_comparisons": len(comparisons),
                },
            )
        else:
            hardware_before = current_hardware
            capacity_before = current_capacity
            validate_capacity(capacity_before, sealed_contract["gates"], None)
            journal = {
                "schema": JOURNAL_SCHEMA,
                "contract": sealed_contract,
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
                output,
                sealed_contract,
                hardware_before,
                capacity_before,
                completed,
                comparisons,
            )

        for task in tasks[len(completed) :]:
            if task.kind == "performance" and len(comparisons) < 2:
                run_missing_comparisons(
                    output,
                    comparisons,
                    completed,
                    journal,
                    build,
                    archive,
                    args,
                    lambda: write_checkpoint(
                        output,
                        sealed_contract,
                        hardware_before,
                        capacity_before,
                        completed,
                        comparisons,
                    ),
                )
            capacity_task_before = capacity_snapshot(output)
            validate_capacity(capacity_task_before, sealed_contract["gates"], task)
            relative = task_relative(task)
            lane_output = output.joinpath(*relative.parts)
            lane_output.parent.mkdir(parents=True, exist_ok=True)
            runtime = output.joinpath(*task_runtime_relative(task).parts)
            environment = matrix.clean_environment(task.lane, runtime)
            before = matrix.hardware_snapshot(devices)
            append_event(
                output, journal, {"state": "task-started", "task": task_spec(task)}
            )
            process, observer_records = observed.run_bounded_observed(
                task_command(task, archive, lane_output),
                environment,
                output,
                output.joinpath(*task_log_relative(task).parts),
                (
                    args.exact_timeout_seconds
                    if task.kind == "exact"
                    else args.performance_timeout_seconds
                ),
                args.stdout_limit_mib * 1024 * 1024,
                lambda pid, selected=task: observer_specs(
                    selected, pid, output, sealed_contract
                ),
                observer_environment=environment,
                observer_exit_timeout_seconds=(
                    args.telemetry_interval_seconds * 3 + 30
                ),
                working_directory=REPO,
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
                raise WorkloadError(f"AuNP hybrid task failed: {task.role}")
            after = matrix.hardware_snapshot(devices)
            hardware = {"before": before, "after": after}
            matrix.validate_hardware_evidence(hardware, list(devices))
            lane_evidence = derive_lane_evidence(
                task, lane_output, output, publish_replay=True
            )
            telemetry_record = publish_telemetry_replay(
                task,
                process,
                observer_records,
                output,
                sealed_contract,
            )
            record = {
                "task": task_spec(task),
                "output": relative.as_posix(),
                "process": process,
                "observers": observer_records,
                "telemetry_replay": telemetry_record,
                "hardware": hardware,
                "capacity_before": capacity_task_before,
                "capacity_after": capacity_snapshot(output),
                **lane_evidence,
            }
            completed.append(record)
            append_event(
                output,
                journal,
                {
                    "state": "task-validated",
                    "task": task_spec(task),
                    "complete": record["complete"],
                },
            )
            write_checkpoint(
                output,
                sealed_contract,
                hardware_before,
                capacity_before,
                completed,
                comparisons,
            )

        if len(comparisons) < 2:
            run_missing_comparisons(
                output,
                comparisons,
                completed,
                journal,
                build,
                archive,
                args,
                lambda: write_checkpoint(
                    output,
                    sealed_contract,
                    hardware_before,
                    capacity_before,
                    completed,
                    comparisons,
                ),
            )
        performance, diagnostics = performance_report(
            completed, cpu_ranks, sealed_contract["gates"]
        )
        clock_events = clock_event_report(output, completed)
        outcome = performance_outcome(performance)
        hardware_after = matrix.hardware_snapshot(devices)
        hardware = {"before": hardware_before, "after": hardware_after}
        matrix.validate_hardware_evidence(hardware, list(devices))
        archive_stable.verify_unchanged()
        nvidia_stable.verify_unchanged()
        for stable in code_stables.values():
            stable.verify_unchanged()
        report = {
            "schema": SCHEMA,
            "outcome": outcome,
            "input": sealed_contract["input"],
            "evidence_code": evidence_code,
            "source_snapshot_sha256": matrix._source_identity(build),
            "build": build_contract_record(build),
            "cpu_physical_cores": cpu_ranks,
            "gpu_devices": list(devices),
            "performance_repeats": args.performance_repeats,
            "qualification_profile": aunp_qualification_profile(),
            "fixed_window_meep_time": performance_adapter.FIXED_WINDOW_MEEP_TIME,
            "tasks": completed,
            "exact_comparisons": comparisons,
            "performance": performance,
            "timing_diagnostics": diagnostics,
            "clock_event_telemetry": clock_events,
            "hardware": hardware,
            "capacity_initial": capacity_before,
            "capacity_final": capacity_snapshot(output),
            "resume_history": matrix.resume_history_records(output),
            "recovery_checkpoint": matrix.stable_file_record(
                output / "CHECKPOINT.json", output, "AuNP hybrid checkpoint"
            ),
        }
        atomic_write_json(output / "report.json", report)
        atomic_write_text(output / "report.md", markdown_report(report))
        publish_terminal(output, outcome)
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
        if nvidia_stable is not None:
            nvidia_stable.close(verify=False)
        archive_stable.close(verify=False)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WorkloadError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"hybrid AuNP matrix error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
