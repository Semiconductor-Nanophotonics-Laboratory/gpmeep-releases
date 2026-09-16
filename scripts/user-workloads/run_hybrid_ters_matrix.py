#!/usr/bin/env python3
"""Run the release TERS hybrid exact-correctness/performance qualification."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import pathlib
import stat
import sys
import time
from dataclasses import dataclass
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_user_workload_matrix as matrix  # noqa: E402
from common import (  # noqa: E402
    MINIMUM_TERS_PERFORMANCE_TIMESTEPS_PER_PHASE,
    MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME,
    StableFile,
    TERS_SHA256,
    TERS_PERFORMANCE_ADAPTATION,
    TERS_PERFORMANCE_MEASUREMENT_MODE,
    WorkloadError,
    atomic_write_json,
    atomic_write_text,
    file_record,
    verify_file,
)


SCHEMA = "gpmeep-hybrid-ters-matrix-v1"
CHECKPOINT_SCHEMA = "gpmeep-hybrid-ters-checkpoint-v1"
COMPLETE_SCHEMA = "gpmeep-hybrid-ters-complete-v1"
MINIMUM_PERFORMANCE_REPEATS = 3
MAXIMUM_PERFORMANCE_REPEATS = 9
DEFAULT_WINDOW_MEEP_TIME = MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
RELEASE_PERFORMANCE_GATES = {
    "maximum_timing_cv": (0.0, 0.15),
    "minimum_one_gpu_speedup": (1.5, math.inf),
    "minimum_two_gpu_speedup": (2.0, math.inf),
    "minimum_multi_gpu_scaling": (1.1, math.inf),
}


@dataclass(frozen=True)
class Task:
    role: str
    lane: matrix.Lane
    repeat: int
    measurement_mode: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    for prefix in ("cpu-fp64", "fp32"):
        parser.add_argument(f"--{prefix}-python", required=True, type=pathlib.Path)
        parser.add_argument(f"--{prefix}-mpiexec", required=True, type=pathlib.Path)
        parser.add_argument(f"--{prefix}-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--cpu-ranks", type=int, default=None)
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument("--performance-repeats", type=int, default=5)
    parser.add_argument(
        "--performance-window-meep-time",
        type=float,
        default=DEFAULT_WINDOW_MEEP_TIME,
    )
    parser.add_argument("--exact-timeout-seconds", type=float, default=7 * 24 * 3600.0)
    parser.add_argument("--performance-timeout-seconds", type=float, default=6 * 3600.0)
    parser.add_argument("--comparison-timeout-seconds", type=float, default=12 * 3600.0)
    parser.add_argument("--stdout-limit-mib", type=int, default=128)
    parser.add_argument("--maximum-timing-cv", type=float, default=0.15)
    parser.add_argument("--minimum-one-gpu-speedup", type=float, default=1.5)
    parser.add_argument("--minimum-two-gpu-speedup", type=float, default=2.0)
    parser.add_argument("--minimum-multi-gpu-scaling", type=float, default=1.1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not MINIMUM_PERFORMANCE_REPEATS <= args.performance_repeats <= MAXIMUM_PERFORMANCE_REPEATS:
        raise WorkloadError(
            f"performance repeats must be in [{MINIMUM_PERFORMANCE_REPEATS},"
            f"{MAXIMUM_PERFORMANCE_REPEATS}]"
        )
    if not math.isfinite(args.performance_window_meep_time) or not (
        MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
        <= args.performance_window_meep_time
        <= 1.0
    ):
        raise WorkloadError(
            "performance window must be in "
            f"[{MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME},1] Meep time"
        )
    for name in (
        "exact_timeout_seconds",
        "performance_timeout_seconds",
        "comparison_timeout_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise WorkloadError(f"{name} must be positive and finite")
    if not 1 <= args.stdout_limit_mib <= 1024:
        raise WorkloadError("stdout limit must be in [1,1024] MiB")
    validate_release_performance_gates(
        {name: getattr(args, name) for name in RELEASE_PERFORMANCE_GATES}
    )


def validate_release_performance_gates(gates: Any) -> None:
    if not isinstance(gates, dict) or set(gates) != set(RELEASE_PERFORMANCE_GATES):
        raise WorkloadError("hybrid release performance gate inventory is not exact")
    for name, (minimum, maximum) in RELEASE_PERFORMANCE_GATES.items():
        value = gates[name]
        invalid = (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        )
        if not invalid:
            invalid = (
                not minimum < float(value) <= maximum
                if name == "maximum_timing_cv"
                else not minimum <= float(value) <= maximum
            )
        if invalid:
            raise WorkloadError(f"hybrid release performance gate {name} is invalid")


def normalized_devices(value: str) -> tuple[str, str]:
    devices = tuple(matrix._normalized_uuid(item) for item in value.split(","))
    normalized = {
        item.lower().removeprefix("gpu-").replace("-", "") for item in devices
    }
    if len(devices) != 2 or len(normalized) != 2:
        raise WorkloadError("hybrid qualification requires two distinct GPU UUIDs")
    return devices


def build_tasks(
    cpu_fp64: matrix.Build,
    fp32: matrix.Build,
    cpu_ranks: int,
    devices: tuple[str, str],
    repeats: int,
) -> list[Task]:
    exact = (
        Task(
            f"exact-cpu-fp64-{cpu_ranks}r",
            matrix.Lane("cpu-fp64", cpu_fp64, "cpu", cpu_ranks, ()),
            0,
            "exact",
        ),
        Task(
            "exact-cuda-fp32-1g",
            matrix.Lane("cuda-fp32-1g", fp32, "cuda", 1, devices[:1]),
            0,
            "exact",
        ),
        Task(
            "exact-cuda-fp32-2g",
            matrix.Lane("cuda-fp32-2g", fp32, "cuda", 2, devices),
            0,
            "exact",
        ),
    )
    performance_lanes = (
        (
            f"performance-cpu-fp32-{cpu_ranks}r",
            matrix.Lane("cpu-fp32", fp32, "cpu", cpu_ranks, ()),
        ),
        (
            "performance-cuda-fp32-1g",
            matrix.Lane("cuda-fp32-1g", fp32, "cuda", 1, devices[:1]),
        ),
        (
            "performance-cuda-fp32-2g",
            matrix.Lane("cuda-fp32-2g", fp32, "cuda", 2, devices),
        ),
    )
    tasks = list(exact)
    for repeat in range(repeats):
        ordered = performance_lanes if repeat % 2 == 0 else tuple(reversed(performance_lanes))
        tasks.extend(
            Task(role, lane, repeat, TERS_PERFORMANCE_MEASUREMENT_MODE)
            for role, lane in ordered
        )
    return tasks


def task_spec(task: Task) -> dict[str, Any]:
    return {
        "role": task.role,
        "repeat": task.repeat,
        "measurement_mode": task.measurement_mode,
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


def reconstruct_contract_tasks(
    sealed_contract: dict[str, Any],
) -> tuple[matrix.Build, matrix.Build, tuple[str, str], list[Task]]:
    """Re-open both receipts and independently reconstruct the required schedule."""

    if not isinstance(sealed_contract, dict):
        raise WorkloadError("hybrid sealed contract is invalid")
    builds = sealed_contract.get("builds")
    if not isinstance(builds, dict) or set(builds) != {"cpu_fp64", "shared_fp32"}:
        raise WorkloadError("hybrid build contract inventory is not exact")
    build_specs = {
        "cpu_fp64": ("cpu-fp64", "cpu-mpi-python-fp64", False, False),
        "shared_fp32": ("shared-fp32", "cuda-mpi-python-fp32", True, True),
    }
    loaded: dict[str, matrix.Build] = {}
    for key, (name, kind, single, cuda) in build_specs.items():
        record = builds[key]
        if not isinstance(record, dict):
            raise WorkloadError(f"hybrid {key} build contract is invalid")
        try:
            loaded[key] = matrix.load_build(
                name,
                pathlib.Path(record["python"]),
                pathlib.Path(record["mpiexec"]),
                pathlib.Path(record["receipt_path"]),
                kind,
                single,
                cuda,
            )
        except (KeyError, TypeError) as exc:
            raise WorkloadError(f"hybrid {key} build contract is incomplete") from exc
        if record != build_contract_record(loaded[key]):
            raise WorkloadError(f"hybrid {key} build contract was not re-derived")
    cpu_fp64 = loaded["cpu_fp64"]
    shared_fp32 = loaded["shared_fp32"]
    if matrix._source_identity(cpu_fp64) != matrix._source_identity(shared_fp32):
        raise WorkloadError("hybrid CPU/GPU builds do not share one source snapshot")
    if cpu_fp64.receipt["receipt_id"] == shared_fp32.receipt["receipt_id"]:
        raise WorkloadError("hybrid FP64 and FP32 receipts are not distinct")
    cpu_ranks = sealed_contract.get("cpu_ranks")
    repeats = sealed_contract.get("performance_repeats")
    devices_value = sealed_contract.get("gpu_devices")
    if (
        type(cpu_ranks) is not int
        or cpu_ranks < 2
        or type(repeats) is not int
        or not MINIMUM_PERFORMANCE_REPEATS <= repeats <= MAXIMUM_PERFORMANCE_REPEATS
        or not isinstance(devices_value, list)
        or not all(isinstance(value, str) for value in devices_value)
    ):
        raise WorkloadError("hybrid task topology contract is invalid")
    devices = normalized_devices(",".join(devices_value))
    if list(devices) != devices_value:
        raise WorkloadError("hybrid GPU device spelling is not canonical")
    tasks = build_tasks(cpu_fp64, shared_fp32, cpu_ranks, devices, repeats)
    if sealed_contract.get("tasks") != [task_spec(task) for task in tasks]:
        raise WorkloadError("hybrid sealed task schedule is not canonical")
    return cpu_fp64, shared_fp32, devices, tasks


def task_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("lanes", task.role, f"repeat-{task.repeat:02d}")


def task_log_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("logs", task.role, f"repeat-{task.repeat:02d}.log")


def task_runtime_relative(task: Task) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath("runtime", task.role, f"repeat-{task.repeat:02d}")


def task_command(
    task: Task,
    source: pathlib.Path,
    output: pathlib.Path,
    window: float,
) -> list[str]:
    command = matrix.lane_command(task.lane, "ters", source, output)
    if task.measurement_mode != "exact":
        command.extend(("--performance-window-meep-time", repr(window)))
    return command


def contract(
    args: argparse.Namespace,
    source: pathlib.Path,
    evidence_code: dict[str, Any],
    cpu_fp64: matrix.Build,
    fp32: matrix.Build,
    cpu_ranks: int,
    devices: tuple[str, str],
    tasks: list[Task],
) -> dict[str, Any]:
    return {
        "input": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": TERS_SHA256,
        },
        "evidence_code": evidence_code,
        "builds": {
            "cpu_fp64": build_contract_record(cpu_fp64),
            "shared_fp32": build_contract_record(fp32),
        },
        "cpu_ranks": cpu_ranks,
        "gpu_devices": list(devices),
        "performance_repeats": args.performance_repeats,
        "performance_window_meep_time": args.performance_window_meep_time,
        "limits": {
            "exact_timeout_seconds": args.exact_timeout_seconds,
            "performance_timeout_seconds": args.performance_timeout_seconds,
            "comparison_timeout_seconds": args.comparison_timeout_seconds,
            "stdout_limit_bytes": args.stdout_limit_mib * 1024 * 1024,
        },
        "gates": {
            "maximum_timing_cv": args.maximum_timing_cv,
            "minimum_one_gpu_speedup": args.minimum_one_gpu_speedup,
            "minimum_two_gpu_speedup": args.minimum_two_gpu_speedup,
            "minimum_multi_gpu_scaling": args.minimum_multi_gpu_scaling,
        },
        "tasks": [task_spec(task) for task in tasks],
    }


def write_checkpoint(
    output: pathlib.Path,
    sealed_contract: dict[str, Any],
    hardware_before: dict[str, Any],
    completed: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
) -> None:
    atomic_write_json(
        output / "CHECKPOINT.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "contract": sealed_contract,
            "hardware_before": hardware_before,
            "completed_tasks": completed,
            "exact_comparison": comparison,
        },
    )


def load_checkpoint(
    output: pathlib.Path, sealed_contract: dict[str, Any]
) -> dict[str, Any]:
    with StableFile(output / "CHECKPOINT.json", "hybrid checkpoint") as stable:
        with stable.file_object() as handle:
            checkpoint = json.load(handle)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "schema",
        "contract",
        "hardware_before",
        "completed_tasks",
        "exact_comparison",
    }:
        raise WorkloadError("hybrid checkpoint schema is not exact")
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise WorkloadError("hybrid checkpoint version differs")
    if checkpoint["contract"] != sealed_contract:
        raise WorkloadError("hybrid resume invocation differs from checkpoint")
    if not isinstance(checkpoint["completed_tasks"], list):
        raise WorkloadError("hybrid checkpoint task inventory is invalid")
    return checkpoint


def validate_lane_mode(summary: dict[str, Any], task: Task, window: float) -> None:
    if summary.get("measurement_mode") != task.measurement_mode:
        raise WorkloadError(f"{task.role} published the wrong measurement mode")
    expected_window = None if task.measurement_mode == "exact" else window
    if summary.get("performance_window_meep_time") != expected_window:
        raise WorkloadError(f"{task.role} published the wrong performance window")
    overrides = summary.get("physics_overrides")
    if task.measurement_mode == "exact":
        if overrides != []:
            raise WorkloadError(f"{task.role} exact execution was modified")
    elif overrides != [TERS_PERFORMANCE_ADAPTATION]:
        raise WorkloadError(f"{task.role} does not disclose its timing adaptation")


def validate_performance_timestep_contract(
    timestep_contract: Any, window: float
) -> None:
    if not isinstance(timestep_contract, list) or len(timestep_contract) != 2:
        raise WorkloadError("hybrid performance sample must contain exactly two phases")
    for phase in timestep_contract:
        if not isinstance(phase, dict) or set(phase) != {"timestep_delta", "meep_time"}:
            raise WorkloadError("hybrid performance timestep contract is not exact")
        timestep_delta = phase["timestep_delta"]
        meep_time = phase["meep_time"]
        if (
            type(timestep_delta) is not int
            or timestep_delta < MINIMUM_TERS_PERFORMANCE_TIMESTEPS_PER_PHASE
            or isinstance(meep_time, bool)
            or not isinstance(meep_time, (int, float))
            or not math.isfinite(float(meep_time))
            or float(meep_time) < window
        ):
            raise WorkloadError(
                "hybrid performance sample is shorter than the minimum work contract"
            )


def validate_completed_task(
    output: pathlib.Path,
    task: Task,
    record: dict[str, Any],
    source: pathlib.Path,
    devices: tuple[str, str],
    window: float,
) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("task") != task_spec(task):
        raise WorkloadError("hybrid checkpoint task order differs")
    relative = task_relative(task)
    if record.get("output") != relative.as_posix():
        raise WorkloadError("hybrid checkpoint task output differs")
    process = record.get("process")
    matrix._verify_process_record(
        output, process, task_log_relative(task).as_posix(), task.role
    )
    expected_command = task_command(
        task, source, output.joinpath(*relative.parts), window
    )
    if process.get("command") != expected_command:
        raise WorkloadError("hybrid checkpoint command differs")
    matrix.validate_hardware_evidence(record.get("hardware"), list(devices))
    validated = matrix.validate_lane_output(
        task.lane, output.joinpath(*relative.parts), "ters"
    )
    validate_lane_mode(validated["summary"], task, window)
    if task.measurement_mode != "exact":
        validate_performance_timestep_contract(validated["timestep_contract"], window)
    lane = matrix.COMPARATOR.load_lane(
        output.joinpath(*relative.parts), matrix.COMPARATOR.TERS_SCHEMA
    )
    matrix.COMPARATOR._ters_files(
        lane, allow_performance_window=task.measurement_mode != "exact"
    )
    complete = matrix.stable_file_record(
        output.joinpath(*relative.parts) / "COMPLETE",
        output,
        f"{task.role} COMPLETE",
    )
    recomputed = {
        "task": task_spec(task),
        "output": relative.as_posix(),
        "process": process,
        "hardware": record["hardware"],
        "fdtd_wall_seconds": validated["fdtd_wall_seconds"],
        "workload_end_to_end_seconds": validated["workload_end_to_end_seconds"],
        "phase_wall_seconds": validated["phase_wall_seconds"],
        "timestep_contract": validated["timestep_contract"],
        "provenance": json.loads(
            json.dumps(validated["provenance"], sort_keys=True, separators=(",", ":"))
        ),
        "complete": complete,
    }
    if record != recomputed:
        raise WorkloadError(f"{task.role} checkpoint record was not re-derived")
    return record


def journal_prefix(
    journal: dict[str, Any],
    sealed_contract: dict[str, Any],
    completed: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if (
        not isinstance(journal, dict)
        or journal.get("schema") != "gpmeep-hybrid-ters-journal-v1"
        or journal.get("contract") != sealed_contract
        or not isinstance(journal.get("events"), list)
        or not journal["events"]
    ):
        raise WorkloadError("hybrid resume journal identity differs")
    first = journal["events"][0]
    if first.get("state") != "controller-started" or not math.isfinite(
        float(first.get("unix_seconds", math.nan))
    ):
        raise WorkloadError("hybrid resume journal has no controller start")
    expected = [first]
    for record in completed:
        spec = record["task"]
        expected.extend(
            (
                {"state": "task-started", "task": spec},
                {"state": "task-process-ended", "task": spec, "process": record["process"]},
                {"state": "task-validated", "task": spec, "complete": record["complete"]},
            )
        )
    if comparison is not None:
        expected.extend(
            (
                {"state": "comparison-process-ended", "process": comparison["process"]},
                {"state": "comparison-validated", "complete": comparison["complete"]},
            )
        )
    if journal["events"][: len(expected)] != expected:
        raise WorkloadError("hybrid journal disagrees with checkpoint")
    return expected


def append_event(output: pathlib.Path, journal: dict[str, Any], event: dict[str, Any]) -> None:
    journal["events"].append(event)
    atomic_write_json(output / "JOURNAL.json", journal)


def validate_exact_comparison(output: pathlib.Path, record: dict[str, Any]) -> None:
    matrix._verify_process_record(
        output,
        record.get("process"),
        "logs/comparisons/exact.log",
        "hybrid exact comparison",
    )
    complete_record = matrix.stable_file_record(
        output / "comparisons" / "exact" / "COMPLETE",
        output,
        "hybrid exact comparison COMPLETE",
    )
    if record.get("complete") != complete_record:
        raise WorkloadError("hybrid exact comparison marker differs")
    with StableFile(
        output / "comparisons" / "exact" / "COMPLETE",
        "hybrid exact comparison COMPLETE",
    ) as stable:
        with stable.file_object() as handle:
            complete = json.load(handle)
    if complete.get("outcome") != "PASS":
        raise WorkloadError("hybrid exact numerical comparison did not pass")


def performance_report(
    args: argparse.Namespace, completed: list[dict[str, Any]], cpu_ranks: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    roles = {
        f"performance-cpu-fp32-{cpu_ranks}r": [],
        "performance-cuda-fp32-1g": [],
        "performance-cuda-fp32-2g": [],
    }
    contracts: set[str] = set()
    for record in completed:
        role = record["task"]["role"]
        if role not in roles:
            continue
        validate_performance_timestep_contract(
            record.get("timestep_contract"), args.performance_window_meep_time
        )
        roles[role].append(record)
        contracts.add(
            json.dumps(record["timestep_contract"], sort_keys=True, separators=(",", ":"))
        )
    if len(contracts) != 1 or any(
        len(records) != args.performance_repeats for records in roles.values()
    ):
        raise WorkloadError("hybrid performance work contract/repeat inventory differs")
    timings = {
        role: [float(record["fdtd_wall_seconds"]) for record in records]
        for role, records in roles.items()
    }
    cpu, gpu1, gpu2 = roles
    metrics = {
        "cpu-fp32-to-one-gpu": {
            **matrix.conservative_speedup(timings[cpu], timings[gpu1]),
            "minimum": args.minimum_one_gpu_speedup,
        },
        "cpu-fp32-to-two-gpu": {
            **matrix.conservative_speedup(timings[cpu], timings[gpu2]),
            "minimum": args.minimum_two_gpu_speedup,
        },
        "one-gpu-to-two-gpu": {
            **matrix.conservative_speedup(timings[gpu1], timings[gpu2]),
            "minimum": args.minimum_multi_gpu_scaling,
        },
    }
    for metric in metrics.values():
        metric["outcome"] = (
            "PASS" if metric["conservative_speedup"] >= metric["minimum"] else "FAIL"
        )
    stability = {
        role: matrix.timing_stability(values, args.maximum_timing_cv)
        for role, values in timings.items()
    }
    return metrics, stability


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# gpmeep hybrid TERS qualification",
        "",
        f"Outcome: **{report['outcome']}**",
        "",
        "Exact numerical correctness uses one unmodified full CPU-FP64 run and one full "
        "run for each CUDA topology. Performance statistics use explicitly adapted fixed "
        "Meep-time windows and are never presented as full-workload results.",
        "",
        "| Metric | Median | Conservative | Gate | Outcome |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, metric in report["performance"].items():
        lines.append(
            f"| {name} | {metric['median_speedup']:.4f}x | "
            f"{metric['conservative_speedup']:.4f}x | {metric['minimum']:.4f}x | "
            f"**{metric['outcome']}** |"
        )
    lines.extend(("", f"Resume boundaries: {len(report['resume_history'])}.", ""))
    return "\n".join(lines)


def verify_hybrid_complete(output: pathlib.Path) -> dict[str, Any]:
    """Independently re-derive a sealed hybrid PASS from retained evidence."""

    output = output.resolve(strict=True)
    with StableFile(output / "COMPLETE", "hybrid COMPLETE") as stable:
        with stable.file_object() as handle:
            complete = json.load(handle)
    if set(complete) != {"schema", "outcome", "report", "markdown", "journal"} or (
        complete["schema"] != COMPLETE_SCHEMA or complete["outcome"] != "PASS"
    ):
        raise WorkloadError("hybrid COMPLETE is not an exact PASS")
    report = matrix.load_stable_json_record(output, complete["report"], "hybrid report")
    matrix.stable_record(output, complete["markdown"], "hybrid Markdown").close()
    journal = matrix.load_stable_json_record(output, complete["journal"], "hybrid journal")
    if report.get("schema") != SCHEMA or report.get("outcome") != "PASS":
        raise WorkloadError("hybrid report is not a PASS")
    evidence_code = report.get("evidence_code")
    expected_code = {
        "controller": pathlib.Path(__file__),
        "matrix_common": SCRIPT_DIR / "run_user_workload_matrix.py",
        "adapter": SCRIPT_DIR / "run_ters_workload.py",
        "comparator": SCRIPT_DIR / "compare_user_workloads.py",
        "common": SCRIPT_DIR / "common.py",
    }
    if not isinstance(evidence_code, dict) or set(evidence_code) != set(expected_code):
        raise WorkloadError("hybrid evidence-code inventory is not exact")
    for name, path in expected_code.items():
        stable = matrix.stable_record(REPO, evidence_code[name], f"hybrid {name}")
        if stable.path != path.resolve(strict=True):
            stable.close(verify=False)
            raise WorkloadError(f"hybrid {name} path is not canonical")
        stable.close()
    input_record = report.get("input")
    if not isinstance(input_record, dict) or input_record.get("sha256") != TERS_SHA256:
        raise WorkloadError("hybrid input is not the exact TERS source")
    with StableFile(pathlib.Path(input_record["path"]), "hybrid input", expected=input_record):
        pass
    checkpoint = matrix.load_stable_json_record(
        output, report.get("recovery_checkpoint", {}), "hybrid checkpoint"
    )
    if set(checkpoint) != {
        "schema",
        "contract",
        "hardware_before",
        "completed_tasks",
        "exact_comparison",
    } or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise WorkloadError("hybrid checkpoint is not exact")
    sealed_contract = checkpoint["contract"]
    repeats = report.get("performance_repeats")
    cpu_ranks = report.get("cpu_physical_cores")
    devices = report.get("gpu_devices")
    window = report.get("performance_window_meep_time")
    if (
        sealed_contract.get("input") != input_record
        or sealed_contract.get("evidence_code") != evidence_code
        or sealed_contract.get("performance_repeats") != repeats
        or sealed_contract.get("cpu_ranks") != cpu_ranks
        or sealed_contract.get("gpu_devices") != devices
        or sealed_contract.get("performance_window_meep_time") != window
        or type(repeats) is not int
        or not MINIMUM_PERFORMANCE_REPEATS <= repeats <= MAXIMUM_PERFORMANCE_REPEATS
        or type(cpu_ranks) is not int
        or cpu_ranks < 2
        or not isinstance(devices, list)
        or len(devices) != 2
        or isinstance(window, bool)
        or not isinstance(window, (int, float))
        or not math.isfinite(float(window))
        or not MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME <= float(window) <= 1.0
    ):
        raise WorkloadError("hybrid checkpoint/report contract differs")
    cpu_fp64, shared_fp32, contract_devices, expected_tasks = reconstruct_contract_tasks(
        sealed_contract
    )
    if list(contract_devices) != devices:
        raise WorkloadError("hybrid reconstructed GPU inventory differs")
    source_snapshot = matrix._source_identity(cpu_fp64)
    if (
        source_snapshot != matrix._source_identity(shared_fp32)
        or report.get("source_snapshot_sha256") != source_snapshot
    ):
        raise WorkloadError("hybrid source/build snapshot was not re-derived")
    matrix.validate_hardware_evidence(report.get("hardware"), devices)
    if checkpoint.get("hardware_before") != report["hardware"]["before"]:
        raise WorkloadError("hybrid initial hardware checkpoint differs")
    actual_history = matrix.resume_history_records(output)
    if report.get("resume_history") != actual_history:
        raise WorkloadError("hybrid resume history differs")
    completed = report.get("tasks")
    expected_count = len(expected_tasks)
    if not isinstance(completed, list) or len(completed) != expected_count:
        raise WorkloadError("hybrid task inventory is incomplete")
    if checkpoint.get("completed_tasks") != completed:
        raise WorkloadError("hybrid checkpoint task inventory differs")
    task_specs = sealed_contract.get("tasks")
    if task_specs != [task_spec(task) for task in expected_tasks]:
        raise WorkloadError("hybrid sealed task schedule is incomplete")

    exact_paths: dict[str, pathlib.Path] = {}
    observed_outputs: set[str] = set()
    source_path = pathlib.Path(input_record["path"]).resolve(strict=True)
    for index, record in enumerate(completed):
        expected_task = expected_tasks[index]
        expected_spec = task_spec(expected_task)
        spec = record.get("task")
        if not isinstance(spec, dict) or spec != expected_spec:
            raise WorkloadError("hybrid task order differs from its contract")
        expected_mode = expected_task.measurement_mode
        if spec.get("measurement_mode") != expected_mode:
            raise WorkloadError("hybrid exact/performance task boundary differs")
        expected_relative = task_relative(expected_task)
        if record.get("output") != expected_relative.as_posix():
            raise WorkloadError("hybrid task output path is not canonical")
        relative = matrix._safe_output_relative(record.get("output"), "hybrid task")
        if relative.as_posix() in observed_outputs:
            raise WorkloadError("hybrid task output path was reused")
        observed_outputs.add(relative.as_posix())
        lane_output = output.joinpath(*relative.parts)
        expected_log = task_log_relative(expected_task).as_posix()
        matrix._verify_process_record(
            output, record.get("process"), expected_log, spec["role"]
        )
        if record["process"].get("command") != task_command(
            expected_task, source_path, lane_output, float(window)
        ):
            raise WorkloadError("hybrid task command is not canonical")
        matrix.validate_hardware_evidence(record.get("hardware"), devices)
        complete_record = matrix.stable_file_record(
            lane_output / "COMPLETE", output, f"{spec['role']} COMPLETE"
        )
        if record.get("complete") != complete_record:
            raise WorkloadError("hybrid task COMPLETE record differs")
        lane = matrix.COMPARATOR.load_lane(lane_output, matrix.COMPARATOR.TERS_SCHEMA)
        summary = lane["summary"]
        if (
            summary.get("expected_backend") != expected_task.lane.backend
            or summary.get("single_precision")
            is not expected_task.lane.build.single_precision
            or summary.get("mpi_size") != expected_task.lane.ranks
        ):
            raise WorkloadError("hybrid task topology/precision evidence differs")
        validate_lane_mode(summary, expected_task, float(window))
        if expected_mode == "exact":
            exact_paths[spec["role"]] = lane_output
            matrix.COMPARATOR._ters_files(lane)
        else:
            matrix.COMPARATOR._ters_files(lane, allow_performance_window=True)
        provenance = matrix.COMPARATOR.validate_lane_provenance(lane)
        if (
            provenance.get("receipt_id") != expected_task.lane.build.receipt["receipt_id"]
            or provenance.get("source_start")
            != expected_task.lane.build.receipt.get("source_start")
        ):
            raise WorkloadError("hybrid task build provenance differs")
        gpu_mapping = provenance.get("gpu_devices")
        expected_gpu_values = {
            value.lower().removeprefix("gpu-").replace("-", "")
            for value in expected_task.lane.devices
        }
        if not isinstance(gpu_mapping, dict) or (
            expected_task.lane.backend == "cpu"
            and gpu_mapping != {}
        ) or (
            expected_task.lane.backend == "cuda"
            and (
                set(gpu_mapping) != set(range(expected_task.lane.ranks))
                or set(gpu_mapping.values()) != expected_gpu_values
            )
        ):
            raise WorkloadError("hybrid task used the wrong GPU rank mapping")
        derived = matrix.derive_lane_metrics(summary, "ters", spec["role"])
        if expected_mode != "exact":
            validate_performance_timestep_contract(
                derived["timestep_contract"], float(window)
            )
        for name in (
            "fdtd_wall_seconds",
            "workload_end_to_end_seconds",
            "phase_wall_seconds",
            "timestep_contract",
        ):
            if record.get(name) != derived[name]:
                raise WorkloadError(f"hybrid {spec['role']} {name} was not re-derived")
        published_provenance = json.loads(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        )
        if record.get("provenance") != published_provenance:
            raise WorkloadError("hybrid task provenance was not re-derived")

    comparison = report.get("exact_comparison")
    if not isinstance(comparison, dict) or checkpoint.get("exact_comparison") != comparison:
        raise WorkloadError("hybrid exact comparison checkpoint differs")
    validate_exact_comparison(output, comparison)
    reference_role = f"exact-cpu-fp64-{cpu_ranks}r"
    comparator_args = argparse.Namespace(
        reference=exact_paths[reference_role],
        candidate=[
            (
                "cuda-fp32-1g",
                "cuda-fp64-fp32",
                exact_paths["exact-cuda-fp32-1g"],
            ),
            (
                "cuda-fp32-2g",
                "cuda-fp64-fp32",
                exact_paths["exact-cuda-fp32-2g"],
            ),
        ],
    )
    recomputed_comparison = matrix.COMPARATOR.compare_ters(comparator_args)
    if recomputed_comparison.get("outcome") != "PASS":
        raise WorkloadError("hybrid exact numerical comparison does not re-derive PASS")

    gates = sealed_contract.get("gates", {})
    validate_release_performance_gates(gates)
    derived_args = argparse.Namespace(
        performance_repeats=repeats,
        performance_window_meep_time=float(window),
        maximum_timing_cv=gates.get("maximum_timing_cv"),
        minimum_one_gpu_speedup=gates.get("minimum_one_gpu_speedup"),
        minimum_two_gpu_speedup=gates.get("minimum_two_gpu_speedup"),
        minimum_multi_gpu_scaling=gates.get("minimum_multi_gpu_scaling"),
    )
    performance, stability = performance_report(derived_args, completed, cpu_ranks)
    if report.get("performance") != performance or report.get("timing_stability") != stability:
        raise WorkloadError("hybrid performance evidence was not re-derived")
    if not all(value["outcome"] == "PASS" for value in performance.values()) or not all(
        value["outcome"] == "PASS" for value in stability.values()
    ):
        raise WorkloadError("hybrid performance gates do not pass")
    expected_events = journal_prefix(journal, sealed_contract, completed, comparison)
    if journal["events"] != expected_events:
        raise WorkloadError("hybrid journal contains unsealed events")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    source = verify_file(args.input, TERS_SHA256, "TERS workload input")
    if args.resume:
        output = args.output.resolve(strict=True)
        if (output / "COMPLETE").exists():
            raise WorkloadError("completed hybrid evidence cannot be resumed")
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    lock_path = output / "LOCK"
    if args.resume and not stat.S_ISREG(lock_path.lstat().st_mode):
        raise WorkloadError("hybrid resume lock is not a regular file")
    lock = lock_path.open("r+" if args.resume else "x")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise WorkloadError(f"could not acquire hybrid matrix lock: {exc}") from exc

    source_stable = StableFile(
        source,
        "TERS workload input",
        expected={"size_bytes": source.stat().st_size, "sha256": TERS_SHA256},
    )
    code_stables: dict[str, StableFile] = {}
    try:
        code_paths = {
            "controller": pathlib.Path(__file__),
            "matrix_common": SCRIPT_DIR / "run_user_workload_matrix.py",
            "adapter": SCRIPT_DIR / "run_ters_workload.py",
            "comparator": SCRIPT_DIR / "compare_user_workloads.py",
            "common": SCRIPT_DIR / "common.py",
        }
        code_stables = {
            name: StableFile(path, f"hybrid {name}") for name, path in code_paths.items()
        }
        evidence_code = {
            name: matrix.stable_identity_record(stable, REPO)
            for name, stable in code_stables.items()
        }
        cpu_fp64 = matrix.load_build(
            "cpu-fp64",
            args.cpu_fp64_python,
            args.cpu_fp64_mpiexec,
            args.cpu_fp64_receipt,
            "cpu-mpi-python-fp64",
            False,
            False,
        )
        fp32 = matrix.load_build(
            "shared-fp32",
            args.fp32_python,
            args.fp32_mpiexec,
            args.fp32_receipt,
            "cuda-mpi-python-fp32",
            True,
            True,
        )
        if matrix._source_identity(cpu_fp64) != matrix._source_identity(fp32):
            raise WorkloadError("hybrid CPU/GPU builds do not share one source snapshot")
        available = matrix.physical_core_count()
        cpu_ranks = args.cpu_ranks or available
        if cpu_ranks < 2 or cpu_ranks > available:
            raise WorkloadError(f"cpu-ranks must be in [2,{available}]")
        devices = normalized_devices(args.gpu_devices)
        current_hardware = matrix.hardware_snapshot(devices)
        tasks = build_tasks(cpu_fp64, fp32, cpu_ranks, devices, args.performance_repeats)
        sealed_contract = contract(
            args,
            source,
            evidence_code,
            cpu_fp64,
            fp32,
            cpu_ranks,
            devices,
            tasks,
        )
        completed: list[dict[str, Any]] = []
        comparison: dict[str, Any] | None = None
        if args.resume:
            checkpoint = load_checkpoint(output, sealed_contract)
            hardware_before = checkpoint["hardware_before"]
            matrix.validate_hardware_evidence(
                {"before": hardware_before, "after": current_hardware}, list(devices)
            )
            checkpoint_completed = checkpoint["completed_tasks"]
            if len(checkpoint_completed) > len(tasks):
                raise WorkloadError("hybrid checkpoint contains too many tasks")
            for index, record in enumerate(checkpoint_completed):
                completed.append(
                    validate_completed_task(
                        output,
                        tasks[index],
                        record,
                        source,
                        devices,
                        args.performance_window_meep_time,
                    )
                )
            comparison = checkpoint["exact_comparison"]
            if comparison is not None:
                if len(completed) != len(tasks):
                    raise WorkloadError("hybrid comparison precedes unfinished tasks")
                validate_exact_comparison(output, comparison)
            with StableFile(output / "JOURNAL.json", "hybrid journal") as stable:
                with stable.file_object() as handle:
                    journal = json.load(handle)
            expected_events = journal_prefix(
                journal, sealed_contract, completed, comparison
            )
            partial_paths: list[pathlib.PurePosixPath] = []
            if len(completed) < len(tasks):
                task = tasks[len(completed)]
                partial_paths.extend(
                    (task_relative(task), task_runtime_relative(task), task_log_relative(task))
                )
            elif comparison is None:
                partial_paths.extend(
                    (
                        pathlib.PurePosixPath("comparisons", "exact"),
                        pathlib.PurePosixPath("logs", "comparisons", "exact.log"),
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
                    "comparison_complete": comparison is not None,
                },
            )
        else:
            hardware_before = current_hardware
            journal = {
                "schema": "gpmeep-hybrid-ters-journal-v1",
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
            write_checkpoint(output, sealed_contract, hardware_before, completed, comparison)

        for task in tasks[len(completed) :]:
            relative = task_relative(task)
            lane_output = output.joinpath(*relative.parts)
            lane_output.parent.mkdir(parents=True, exist_ok=True)
            runtime = output.joinpath(*task_runtime_relative(task).parts)
            environment = matrix.clean_environment(task.lane, runtime)
            log = output.joinpath(*task_log_relative(task).parts)
            before = matrix.hardware_snapshot(devices)
            append_event(output, journal, {"state": "task-started", "task": task_spec(task)})
            process = matrix.run_bounded(
                task_command(
                    task, source, lane_output, args.performance_window_meep_time
                ),
                environment,
                log,
                (
                    args.exact_timeout_seconds
                    if task.measurement_mode == "exact"
                    else args.performance_timeout_seconds
                ),
                args.stdout_limit_mib * 1024 * 1024,
            )
            append_event(
                output,
                journal,
                {"state": "task-process-ended", "task": task_spec(task), "process": process},
            )
            if process["returncode"] != 0 or process["timed_out"] or process["output_limited"]:
                raise WorkloadError(f"hybrid task failed: {task.role}")
            after = matrix.hardware_snapshot(devices)
            hardware = {"before": before, "after": after}
            matrix.validate_hardware_evidence(hardware, list(devices))
            validated = matrix.validate_lane_output(task.lane, lane_output, "ters")
            validate_lane_mode(
                validated["summary"], task, args.performance_window_meep_time
            )
            if task.measurement_mode != "exact":
                validate_performance_timestep_contract(
                    validated["timestep_contract"],
                    args.performance_window_meep_time,
                )
            published_lane = matrix.COMPARATOR.load_lane(
                lane_output, matrix.COMPARATOR.TERS_SCHEMA
            )
            matrix.COMPARATOR._ters_files(
                published_lane,
                allow_performance_window=task.measurement_mode != "exact",
            )
            complete_record = matrix.stable_file_record(
                lane_output / "COMPLETE", output, f"{task.role} COMPLETE"
            )
            record = {
                "task": task_spec(task),
                "output": relative.as_posix(),
                "process": process,
                "hardware": hardware,
                "fdtd_wall_seconds": validated["fdtd_wall_seconds"],
                "workload_end_to_end_seconds": validated[
                    "workload_end_to_end_seconds"
                ],
                "phase_wall_seconds": validated["phase_wall_seconds"],
                "timestep_contract": validated["timestep_contract"],
                "provenance": json.loads(
                    json.dumps(
                        validated["provenance"], sort_keys=True, separators=(",", ":")
                    )
                ),
                "complete": complete_record,
            }
            completed.append(record)
            append_event(
                output,
                journal,
                {
                    "state": "task-validated",
                    "task": task_spec(task),
                    "complete": complete_record,
                },
            )
            write_checkpoint(output, sealed_contract, hardware_before, completed, comparison)

        if comparison is None:
            exact_by_role = {
                record["task"]["role"]: output / record["output"]
                for record in completed
                if record["task"]["measurement_mode"] == "exact"
            }
            reference = exact_by_role[f"exact-cpu-fp64-{cpu_ranks}r"]
            candidates = [
                (
                    "cuda-fp32-1g",
                    "cuda-fp64-fp32",
                    exact_by_role["exact-cuda-fp32-1g"],
                ),
                (
                    "cuda-fp32-2g",
                    "cuda-fp64-fp32",
                    exact_by_role["exact-cuda-fp32-2g"],
                ),
            ]
            comparison_output = output / "comparisons" / "exact"
            comparison_environment = {
                "PATH": f"{fp32.python.parent}:/usr/bin:/bin",
                "HOME": str(output / "runtime"),
                "XDG_CACHE_HOME": str(output / "runtime"),
                "TMPDIR": "/tmp",
                "LC_ALL": "C",
                "LANG": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
            process = matrix.run_bounded(
                matrix.comparison_command(
                    fp32.python,
                    "ters",
                    comparison_output,
                    reference,
                    candidates,
                    source,
                ),
                comparison_environment,
                output / "logs" / "comparisons" / "exact.log",
                args.comparison_timeout_seconds,
                args.stdout_limit_mib * 1024 * 1024,
            )
            append_event(
                output, journal, {"state": "comparison-process-ended", "process": process}
            )
            if process["returncode"] != 0 or process["timed_out"] or process["output_limited"]:
                raise WorkloadError("hybrid exact comparison process failed")
            complete_record = matrix.stable_file_record(
                comparison_output / "COMPLETE", output, "hybrid comparison COMPLETE"
            )
            comparison = {"process": process, "complete": complete_record}
            validate_exact_comparison(output, comparison)
            append_event(
                output,
                journal,
                {"state": "comparison-validated", "complete": complete_record},
            )
            write_checkpoint(output, sealed_contract, hardware_before, completed, comparison)

        performance, stability = performance_report(args, completed, cpu_ranks)
        outcome = (
            "PASS"
            if all(value["outcome"] == "PASS" for value in performance.values())
            and all(value["outcome"] == "PASS" for value in stability.values())
            else "FAIL"
        )
        hardware_after = matrix.hardware_snapshot(devices)
        hardware = {"before": hardware_before, "after": hardware_after}
        matrix.validate_hardware_evidence(hardware, list(devices))
        source_stable.verify_unchanged()
        report = {
            "schema": SCHEMA,
            "outcome": outcome,
            "input": sealed_contract["input"],
            "evidence_code": evidence_code,
            "source_snapshot_sha256": matrix._source_identity(cpu_fp64),
            "cpu_physical_cores": cpu_ranks,
            "gpu_devices": list(devices),
            "performance_repeats": args.performance_repeats,
            "performance_window_meep_time": args.performance_window_meep_time,
            "hardware": hardware,
            "tasks": completed,
            "exact_comparison": comparison,
            "performance": performance,
            "timing_stability": stability,
            "recovery_checkpoint": matrix.stable_file_record(
                output / "CHECKPOINT.json", output, "hybrid checkpoint"
            ),
            "resume_history": matrix.resume_history_records(output),
            "qualification_semantics": {
                "exact_correctness": (
                    "one complete unmodified CPU-FP64 run compared against one complete "
                    "unmodified one-GPU and two-GPU FP32 run"
                ),
                "performance": (
                    "fixed Meep-time samples with exact geometry, sources, dispersive "
                    "materials, DFT monitors, and postprocessing; not full-workload timing"
                ),
            },
        }
        atomic_write_json(output / "report.json", report)
        atomic_write_text(output / "report.md", markdown_report(report))
        (output / "RUNNING.json").unlink()
        terminal = {
            "schema": COMPLETE_SCHEMA if outcome == "PASS" else f"{SCHEMA}-failed-v1",
            "outcome": outcome,
            "report": file_record(output / "report.json", output),
            "markdown": file_record(output / "report.md", output),
            "journal": file_record(output / "JOURNAL.json", output),
        }
        atomic_write_json(output / ("COMPLETE" if outcome == "PASS" else "FAILED.json"), terminal)
        if outcome == "PASS":
            verify_hybrid_complete(output)
        return 0 if outcome == "PASS" else 3
    except Exception as exc:
        running = output / "RUNNING.json"
        if running.exists():
            running.unlink()
        failure = {
            "schema": f"{SCHEMA}-failed-v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "partial_artifacts": matrix.collect_partial_evidence(output),
        }
        if (output / "JOURNAL.json").is_file():
            failure["journal"] = file_record(output / "JOURNAL.json", output)
        atomic_write_json(output / "FAILED.json", failure)
        raise
    finally:
        for stable in code_stables.values():
            stable.close(verify=False)
        source_stable.close(verify=False)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WorkloadError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"hybrid TERS matrix error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
