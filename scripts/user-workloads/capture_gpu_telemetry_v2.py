#!/usr/bin/env python3
"""Capture source-bound NVIDIA telemetry for one sealed task process tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import signal
import stat
import subprocess
import sys
import time
from typing import Any


SAMPLE_SCHEMA = "gpmeep-gpu-telemetry-sample-v3"
COMPLETE_SCHEMA = "gpmeep-gpu-telemetry-complete-v3"
NVIDIA_SMI = pathlib.Path("/usr/bin/nvidia-smi")

# query name, retained name, representation, required on every selected device
GPU_FIELD_SPECS = (
    ("index", "index", "int", True),
    ("uuid", "uuid", "string", True),
    ("name", "name", "string", True),
    ("pstate", "pstate", "string", False),
    ("temperature.gpu", "temperature_celsius", "float", True),
    ("temperature.gpu.tlimit", "temperature_limit_celsius", "float", False),
    ("power.draw", "power_draw_watts", "float", False),
    ("power.limit", "power_limit_watts", "float", False),
    ("clocks.sm", "sm_clock_mhz", "float", False),
    ("clocks.mem", "memory_clock_mhz", "float", False),
    ("memory.used", "memory_used_mib", "float", True),
    ("memory.total", "memory_total_mib", "float", True),
    ("utilization.gpu", "gpu_utilization_percent", "float", False),
    ("utilization.memory", "memory_utilization_percent", "float", False),
    (
        "clocks_event_reasons.sw_thermal_slowdown",
        "clock_event_sw_thermal_slowdown",
        "bool",
        True,
    ),
    (
        "clocks_event_reasons.hw_thermal_slowdown",
        "clock_event_hw_thermal_slowdown",
        "bool",
        True,
    ),
    (
        "clocks_event_reasons.hw_power_brake_slowdown",
        "clock_event_hw_power_brake_slowdown",
        "bool",
        True,
    ),
    (
        "clocks_event_reasons.sw_power_cap",
        "clock_event_sw_power_cap",
        "bool",
        True,
    ),
    (
        "ecc.errors.corrected.volatile.total",
        "ecc_corrected_volatile_total",
        "int",
        False,
    ),
    (
        "ecc.errors.uncorrected.volatile.total",
        "ecc_uncorrected_volatile_total",
        "int",
        False,
    ),
    (
        "ecc.errors.corrected.aggregate.total",
        "ecc_corrected_aggregate_total",
        "int",
        False,
    ),
    (
        "ecc.errors.uncorrected.aggregate.total",
        "ecc_uncorrected_aggregate_total",
        "int",
        False,
    ),
    (
        "retired_pages.single_bit_ecc.count",
        "retired_pages_single_bit_count",
        "int",
        False,
    ),
    (
        "retired_pages.double_bit.count",
        "retired_pages_double_bit_count",
        "int",
        False,
    ),
    ("retired_pages.pending", "retired_pages_pending", "bool", False),
    ("remapped_rows.correctable", "remapped_rows_correctable", "int", False),
    (
        "remapped_rows.uncorrectable",
        "remapped_rows_uncorrectable",
        "int",
        False,
    ),
    ("remapped_rows.pending", "remapped_rows_pending", "bool", False),
    ("remapped_rows.failure", "remapped_rows_failure", "bool", False),
)
GPU_QUERY_FIELDS = tuple(spec[0] for spec in GPU_FIELD_SPECS)
UNSUPPORTED_VALUES = {"n/a", "not available", "not supported", "unsupported"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root-pid", required=True, type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--selected-uuid", action="append", required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_file_record(path: pathlib.Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"telemetry artifact is not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def atomic_exclusive_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeError("GPU telemetry terminal write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def proc_identity(pid: int) -> dict[str, Any] | None:
    try:
        raw = (pathlib.Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8"
        )
        close = raw.rfind(")")
        if close <= 1:
            return None
        fields = raw[close + 2 :].split()
        if len(fields) <= 19:
            return None
        return {"state": fields[0], "start_time_ticks": int(fields[19])}
    except (OSError, IndexError, ValueError):
        return None


def run_query(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
    )
    if completed.returncode != 0:
        error = completed.stderr.strip().replace("\n", " ")
        raise RuntimeError(f"query returned {completed.returncode}: {error}")
    return completed.stdout


def csv_rows(raw: str) -> list[list[str]]:
    return [
        [value.strip() for value in row]
        for row in csv.reader(io.StringIO(raw), skipinitialspace=True)
        if row and any(value.strip() for value in row)
    ]


def parse_metric(raw: str, kind: str, *, required: bool) -> dict[str, Any]:
    text = raw.strip()
    normalized = text.strip("[]").strip().lower()
    if normalized in UNSUPPORTED_VALUES:
        if required:
            raise ValueError("required GPU metric is unsupported")
        return {"supported": False, "value": None, "raw": text}
    if kind == "string":
        value: Any = text
        if not value:
            raise ValueError("empty GPU metric")
    elif kind == "int":
        value = int(text)
        if value < 0:
            raise ValueError("negative GPU counter")
    elif kind == "float":
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("nonfinite GPU metric")
    elif kind == "bool":
        choices = {
            "yes": True,
            "true": True,
            "enabled": True,
            "active": True,
            "no": False,
            "false": False,
            "disabled": False,
            "inactive": False,
            "not active": False,
        }
        if normalized not in choices:
            raise ValueError(f"unrecognized GPU boolean metric: {text}")
        value = choices[normalized]
    else:
        raise ValueError(f"unknown metric representation: {kind}")
    return {"supported": True, "value": value, "raw": text}


def parse_gpu_inventory(raw: str) -> list[dict[str, Any]]:
    parsed = []
    for row in csv_rows(raw):
        if len(row) != len(GPU_FIELD_SPECS):
            raise ValueError("GPU inventory row has the wrong field count")
        record = {}
        for value, (_query, retained, kind, required) in zip(
            row, GPU_FIELD_SPECS, strict=True
        ):
            metric = parse_metric(value, kind, required=required)
            if retained in {"index", "uuid", "name"}:
                record[retained] = metric["value"]
            else:
                record[retained] = metric
        parsed.append(record)
    if not parsed:
        raise ValueError("GPU inventory is empty")
    uuids = [str(row["uuid"]) for row in parsed]
    indices = [int(row["index"]) for row in parsed]
    if len(set(uuids)) != len(uuids) or len(set(indices)) != len(indices):
        raise ValueError("GPU inventory contains a duplicate UUID or index")
    return parsed


def parse_compute_apps(raw: str) -> list[dict[str, Any]]:
    parsed = []
    for row in csv_rows(raw):
        if len(row) != 4:
            raise ValueError("compute-app row has the wrong field count")
        pid = int(row[1])
        if pid <= 0 or not row[0] or not row[2]:
            raise ValueError("compute-app identity is invalid")
        parsed.append(
            {
                "gpu_uuid": row[0],
                "pid": pid,
                "process_name": row[2],
                "used_gpu_memory_mib": parse_metric(
                    row[3], "float", required=False
                ),
            }
        )
    return parsed


def new_output_paths(requested: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    if requested.is_symlink():
        raise FileExistsError("GPU telemetry output is a symlink")
    output = pathlib.Path(os.path.abspath(requested))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve(strict=True) != output.parent:
        raise RuntimeError("GPU telemetry output parent contains a symlink")
    complete = output.with_name(output.name + ".COMPLETE.json")
    if output.exists() or output.is_symlink() or complete.exists() or complete.is_symlink():
        raise FileExistsError("GPU telemetry output or terminal already exists")
    return output, complete


def capture(selected: tuple[str, ...]) -> dict[str, Any]:
    inventory = []
    for uuid in selected:
        rows = parse_gpu_inventory(
            run_query(
                [
                    str(NVIDIA_SMI),
                    f"--id={uuid}",
                    "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
                    "--format=csv,noheader,nounits",
                ]
            )
        )
        if len(rows) != 1 or rows[0]["uuid"] != uuid:
            raise ValueError("GPU query differs from the sealed UUID")
        inventory.append(rows[0])
    applications = parse_compute_apps(
        run_query(
            [
                str(NVIDIA_SMI),
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    selected_set = set(selected)
    applications = [
        row for row in applications if row["gpu_uuid"] in selected_set
    ]
    return {"gpus": inventory, "compute_applications": applications}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = tuple(args.selected_uuid)
    if args.root_pid <= 1 or not args.label:
        raise ValueError("GPU telemetry root PID and label are invalid")
    if (
        len(selected) not in {1, 2}
        or len(set(selected)) != len(selected)
        or any(not value.startswith("GPU-") for value in selected)
    ):
        raise ValueError("one or two distinct canonical GPU UUIDs are required")
    if (
        not math.isfinite(args.interval_seconds)
        or not 1.0 <= args.interval_seconds <= 60.0
    ):
        raise ValueError("GPU telemetry interval must be in [1,60] seconds")
    output, complete = new_output_paths(args.output)
    initial_identity = proc_identity(args.root_pid)
    if initial_identity is None or initial_identity["state"] in {"Z", "X"}:
        raise RuntimeError("GPU telemetry root process is not alive at capture start")
    root_start_time_ticks = int(initial_identity["start_time_ticks"])
    source_before = absolute_file_record(pathlib.Path(__file__))
    nvidia_smi_before = absolute_file_record(NVIDIA_SMI)
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    interrupted: list[str] = []

    def handle_signal(signum: int, _frame: Any) -> None:
        interrupted.append(signal.Signals(signum).name)

    previous_handlers = {}
    for selected_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[selected_signal] = signal.getsignal(selected_signal)
        signal.signal(selected_signal, handle_signal)

    started = time.time()
    started_monotonic = time.monotonic()
    sequence = 0
    query_error_count = 0
    outcome = "PASS"
    terminal_error = None
    try:
        deadline = time.monotonic()
        while True:
            identity = proc_identity(args.root_pid)
            root_alive = bool(
                identity is not None
                and identity["state"] not in {"Z", "X"}
                and int(identity["start_time_ticks"]) == root_start_time_ticks
            )
            sample: dict[str, Any] = {
                "schema": SAMPLE_SCHEMA,
                "sequence": sequence,
                "label": args.label,
                "captured_unix_seconds": time.time(),
                "captured_monotonic_seconds": time.monotonic(),
                "root_pid": args.root_pid,
                "root_start_time_ticks": root_start_time_ticks,
                "root_alive": root_alive,
                "selected_gpu_uuids": list(selected),
            }
            try:
                sample.update(capture(selected))
                sample["query_error"] = None
            except Exception as exc:
                sample.update({"gpus": [], "compute_applications": []})
                sample["query_error"] = f"{type(exc).__name__}: {exc}"
                query_error_count += 1
                outcome = "FAIL"
            payload = (
                json.dumps(sample, allow_nan=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise RuntimeError("GPU telemetry append made no progress")
                offset += written
            os.fsync(descriptor)
            sequence += 1
            if interrupted:
                outcome = "INTERRUPTED"
                terminal_error = f"received {interrupted[0]}"
                break
            if not root_alive:
                break
            deadline += args.interval_seconds
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()
    except Exception as exc:
        outcome = "FAIL"
        terminal_error = f"{type(exc).__name__}: {exc}"
    finally:
        os.fsync(descriptor)
        os.close(descriptor)
        for selected_signal, handler in previous_handlers.items():
            signal.signal(selected_signal, handler)

    source_after = absolute_file_record(pathlib.Path(__file__))
    nvidia_smi_after = absolute_file_record(NVIDIA_SMI)
    if source_before != source_after or nvidia_smi_before != nvidia_smi_after:
        outcome = "FAIL"
        terminal_error = "GPU telemetry monitor or nvidia-smi identity changed"
    terminal = {
        "schema": COMPLETE_SCHEMA,
        "outcome": outcome,
        "terminal_error": terminal_error,
        "label": args.label,
        "root_pid": args.root_pid,
        "root_start_time_ticks": root_start_time_ticks,
        "selected_gpu_uuids": list(selected),
        "interval_seconds": args.interval_seconds,
        "sample_count": sequence,
        "query_error_count": query_error_count,
        "started_unix_seconds": started,
        "ended_unix_seconds": time.time(),
        "started_monotonic_seconds": started_monotonic,
        "ended_monotonic_seconds": time.monotonic(),
        "telemetry": absolute_file_record(output),
        "monitor_source_before": source_before,
        "monitor_source_after": source_after,
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
    }
    atomic_exclusive_json(complete, terminal)
    return 0 if outcome == "PASS" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"GPU telemetry error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
