#!/usr/bin/env python3
"""Independently replay final host and NVIDIA telemetry streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
from typing import Any


HOST_SAMPLE_SCHEMA = "gpmeep-host-process-telemetry-sample-v2"
HOST_COMPLETE_SCHEMA = "gpmeep-host-process-telemetry-complete-v2"
GPU_SAMPLE_SCHEMA = "gpmeep-gpu-telemetry-sample-v3"
GPU_COMPLETE_SCHEMA = "gpmeep-gpu-telemetry-complete-v3"
REPORT_SCHEMA = "gpmeep-telemetry-replay-v3"
UNSUPPORTED_VALUES = {"n/a", "not available", "not supported", "unsupported"}
GPU_REQUIRED_METRICS = {
    "temperature_celsius": "float",
    "memory_used_mib": "float",
    "memory_total_mib": "float",
    "clock_event_sw_thermal_slowdown": "bool",
    "clock_event_hw_thermal_slowdown": "bool",
    "clock_event_hw_power_brake_slowdown": "bool",
    "clock_event_sw_power_cap": "bool",
}
CLOCK_EVENT_REASON_METRICS = (
    "clock_event_sw_thermal_slowdown",
    "clock_event_hw_thermal_slowdown",
    "clock_event_hw_power_brake_slowdown",
    "clock_event_sw_power_cap",
)
GPU_OPTIONAL_METRICS = {
    "pstate": "string",
    "temperature_limit_celsius": "float",
    "power_draw_watts": "float",
    "power_limit_watts": "float",
    "sm_clock_mhz": "float",
    "memory_clock_mhz": "float",
    "gpu_utilization_percent": "float",
    "memory_utilization_percent": "float",
    "ecc_corrected_volatile_total": "int",
    "ecc_uncorrected_volatile_total": "int",
    "ecc_corrected_aggregate_total": "int",
    "ecc_uncorrected_aggregate_total": "int",
    "retired_pages_single_bit_count": "int",
    "retired_pages_double_bit_count": "int",
    "retired_pages_pending": "bool",
    "remapped_rows_correctable": "int",
    "remapped_rows_uncorrectable": "int",
    "remapped_rows_pending": "bool",
    "remapped_rows_failure": "bool",
}
MONOTONIC_COUNTER_METRICS = (
    "ecc_corrected_volatile_total",
    "ecc_uncorrected_volatile_total",
    "ecc_corrected_aggregate_total",
    "ecc_uncorrected_aggregate_total",
    "retired_pages_single_bit_count",
    "retired_pages_double_bit_count",
    "remapped_rows_correctable",
    "remapped_rows_uncorrectable",
)
FAULT_BOOLEAN_METRICS = (
    "retired_pages_pending",
    "remapped_rows_pending",
    "remapped_rows_failure",
)


def fail(message: str) -> None:
    raise ValueError(message)


def exact_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{name} is not an integer")
    return value


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{name} is not finite")
    return result


def nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{name} is not a nonempty string")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_report(path: pathlib.Path, value: dict[str, Any]) -> pathlib.Path:
    if path.is_symlink():
        raise FileExistsError("telemetry replay report is a symlink")
    output = pathlib.Path(os.path.abspath(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve(strict=True) != output.parent:
        raise RuntimeError("telemetry replay report parent contains a symlink")
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeError("telemetry replay report write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return output


def read_jsonl(path: pathlib.Path) -> tuple[pathlib.Path, bytes, list[dict[str, Any]]]:
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        fail(f"final telemetry is empty or lacks a final newline: {resolved}")
    records = []
    for number, raw in enumerate(payload.splitlines(), 1):
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(f"invalid JSONL record {number} in {resolved}: {exc}")
        if not isinstance(value, dict):
            fail(f"JSONL record {number} is not an object")
        records.append(value)
    return resolved, payload, records


def load_json(path: pathlib.Path, name: str) -> tuple[pathlib.Path, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid {name} JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{name} is not an object")
    return resolved, value


def check_file_record(
    record: Any, path: pathlib.Path, payload: bytes, name: str
) -> None:
    if not isinstance(record, dict):
        fail(f"{name} file record is not an object")
    recorded_path = nonempty_string(record.get("path"), f"{name}.path")
    if pathlib.Path(recorded_path).resolve(strict=True) != path.resolve(strict=True):
        fail(f"{name} path does not match")
    if exact_int(record.get("size_bytes"), f"{name}.size_bytes") != len(payload):
        fail(f"{name} size does not match")
    if record.get("sha256") != sha256_bytes(payload):
        fail(f"{name} SHA-256 does not match")


def check_bound_file(
    before: Any,
    after: Any,
    expected_sha256: str,
    name: str,
) -> dict[str, Any]:
    if not expected_sha256 or len(expected_sha256) != 64:
        fail(f"{name} requires an externally sealed SHA-256")
    if not isinstance(before, dict) or not isinstance(after, dict) or before != after:
        fail(f"{name} before/after identity differs")
    path = pathlib.Path(nonempty_string(before.get("path"), f"{name}.path"))
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    check_file_record(before, resolved, payload, name)
    if before.get("sha256") != expected_sha256:
        fail(f"{name} differs from its externally sealed SHA-256")
    return before


def gap_summary(times: list[float], name: str) -> dict[str, float | None]:
    gaps = [right - left for left, right in zip(times, times[1:])]
    if any(gap <= 0 for gap in gaps):
        fail(f"{name} timestamps are not strictly increasing")
    return {
        "minimum": min(gaps) if gaps else None,
        "median": statistics.median(gaps) if gaps else None,
        "maximum": max(gaps) if gaps else None,
    }


def verify_terminal_timing(
    terminal: dict[str, Any], times: list[float], name: str
) -> float:
    interval = finite_number(terminal.get("interval_seconds"), f"{name}.interval")
    started = finite_number(terminal.get("started_unix_seconds"), f"{name}.started")
    ended = finite_number(terminal.get("ended_unix_seconds"), f"{name}.ended")
    if not 1.0 <= interval <= 60.0:
        fail(f"{name} interval lies outside [1,60] seconds")
    if not started <= times[0] <= times[-1] <= ended:
        fail(f"{name} time bounds do not contain every sample")
    return interval


def verify_host(
    stream_path: pathlib.Path,
    terminal_path: pathlib.Path,
    *,
    expected_root_pid: int,
    expected_label: str,
    expected_monitor_sha256: str,
    minimum_available_bytes: int,
    minimum_cgroup_free_bytes: int,
    maximum_temperature_millidegrees: int,
) -> dict[str, Any]:
    stream, payload, records = read_jsonl(stream_path)
    _terminal_file, terminal = load_json(terminal_path, "host terminal")
    if expected_root_pid <= 1 or not expected_label:
        fail("host replay external process identity is invalid")
    if minimum_available_bytes < 0 or minimum_cgroup_free_bytes < 0:
        fail("host replay memory gates are negative")
    if maximum_temperature_millidegrees <= 0:
        fail("host replay temperature gate is invalid")

    root_start: int | None = None
    times: list[float] = []
    seen_dead = False
    alive_samples = 0
    maximum_rss = 0
    minimum_available: int | None = None
    minimum_swap_free: int | None = None
    minimum_cgroup_free: int | None = None
    maximum_temperature: dict[str, Any] | None = None
    process_identities: set[tuple[int, int]] = set()
    process_pids: set[int] = set()
    cgroup_support: bool | None = None
    cgroup_path: str | None = None
    host_capacity: tuple[int, int] | None = None
    cgroup_capacity: tuple[int | None, int | None] | None = None

    for sequence, record in enumerate(records):
        prefix = f"host[{sequence}]"
        if record.get("schema") != HOST_SAMPLE_SCHEMA:
            fail(f"{prefix} has the wrong schema")
        if exact_int(record.get("sequence"), f"{prefix}.sequence") != sequence:
            fail(f"{prefix} has a non-contiguous sequence")
        if exact_int(record.get("root_pid"), f"{prefix}.root_pid") != expected_root_pid:
            fail(f"{prefix} differs from the sealed root PID")
        if record.get("label") != expected_label:
            fail(f"{prefix} differs from the sealed label")
        this_start = exact_int(
            record.get("root_start_time_ticks"), f"{prefix}.root_start_time_ticks"
        )
        root_start = this_start if root_start is None else root_start
        if this_start != root_start or this_start <= 0:
            fail("host root process start identity changed")
        times.append(
            finite_number(record.get("captured_unix_seconds"), f"{prefix}.captured")
        )
        alive = record.get("root_alive")
        if not isinstance(alive, bool):
            fail(f"{prefix}.root_alive is not Boolean")
        if seen_dead and alive:
            fail("host root became alive after a terminal sample")
        seen_dead = seen_dead or not alive
        alive_samples += int(alive)

        processes = record.get("processes")
        if not isinstance(processes, list):
            fail(f"{prefix}.processes is not a list")
        if exact_int(record.get("process_count"), f"{prefix}.process_count") != len(
            processes
        ):
            fail(f"{prefix} process count differs from its list")
        pids = []
        rss_sum = 0
        for index, process in enumerate(processes):
            if not isinstance(process, dict):
                fail(f"{prefix}.processes[{index}] is not an object")
            pid = exact_int(process.get("pid"), "process.pid")
            ppid = exact_int(process.get("ppid"), "process.ppid")
            pgrp = exact_int(process.get("pgrp"), "process.pgrp")
            start = exact_int(process.get("start_time_ticks"), "process.start")
            processor = exact_int(process.get("processor"), "process.processor")
            rss = exact_int(process.get("rss_bytes"), "process.rss_bytes")
            state = nonempty_string(process.get("state"), "process.state")
            nonempty_string(process.get("comm"), "process.comm")
            if (
                pid <= 0
                or ppid < 0
                or pgrp <= 0
                or start <= 0
                or processor < 0
                or rss < 0
                or len(state) != 1
            ):
                fail(f"{prefix} has invalid process telemetry")
            pids.append(pid)
            rss_sum += rss
            if alive:
                process_identities.add((pid, start))
                process_pids.add(pid)
        if pids != sorted(set(pids)):
            fail(f"{prefix} process PIDs are duplicate or unsorted")
        process_by_pid = {item["pid"]: item for item in processes}
        if alive and expected_root_pid not in pids:
            fail(f"{prefix} says root alive but omits its process")
        root_records = [item for item in processes if item["pid"] == expected_root_pid]
        if alive and (
            len(root_records) != 1
            or root_records[0]["start_time_ticks"] != root_start
            or root_records[0]["state"] in {"Z", "X"}
        ):
            fail(f"{prefix} root process identity differs from the stream")
        if not alive and processes:
            fail(f"{prefix} retains processes after the root exited")
        if alive:
            for pid in pids:
                if pid == expected_root_pid:
                    continue
                visited = set()
                current = pid
                while current != expected_root_pid:
                    if current in visited or current not in process_by_pid:
                        fail(f"{prefix} process inventory is not rooted at the task")
                    visited.add(current)
                    current = process_by_pid[current]["ppid"]
        if exact_int(record.get("process_rss_bytes"), f"{prefix}.rss") != rss_sum:
            fail(f"{prefix} RSS sum differs from its process records")
        maximum_rss = max(maximum_rss, rss_sum)

        memory = record.get("memory_bytes")
        if not isinstance(memory, dict):
            fail(f"{prefix}.memory_bytes is not an object")
        memory_values = {
            key: exact_int(memory.get(key), f"{prefix}.memory.{key}")
            for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
        }
        if (
            memory_values["MemTotal"] <= 0
            or memory_values["MemAvailable"] < 0
            or memory_values["MemAvailable"] > memory_values["MemTotal"]
            or memory_values["SwapTotal"] < 0
            or memory_values["SwapFree"] < 0
            or memory_values["SwapFree"] > memory_values["SwapTotal"]
        ):
            fail(f"{prefix} has invalid host memory telemetry")
        capacity = (memory_values["MemTotal"], memory_values["SwapTotal"])
        host_capacity = capacity if host_capacity is None else host_capacity
        if capacity != host_capacity:
            fail("host memory capacity changed within the stream")
        if memory_values["MemAvailable"] < minimum_available_bytes:
            fail(f"{prefix} violates the host available-memory gate")
        minimum_available = (
            memory_values["MemAvailable"]
            if minimum_available is None
            else min(minimum_available, memory_values["MemAvailable"])
        )
        minimum_swap_free = (
            memory_values["SwapFree"]
            if minimum_swap_free is None
            else min(minimum_swap_free, memory_values["SwapFree"])
        )

        loads = record.get("load_average")
        if not isinstance(loads, list) or len(loads) != 3:
            fail(f"{prefix}.load_average has the wrong shape")
        if any(
            finite_number(value, f"{prefix}.load_average") < 0 for value in loads
        ):
            fail(f"{prefix} has a negative load average")

        cgroup = record.get("cgroup_memory")
        if not isinstance(cgroup, dict) or not isinstance(cgroup.get("supported"), bool):
            fail(f"{prefix}.cgroup_memory is invalid")
        supported = cgroup["supported"]
        cgroup_support = supported if cgroup_support is None else cgroup_support
        if supported != cgroup_support:
            fail("host cgroup support changed within the stream")
        keys = (
            "memory_current",
            "memory_max",
            "memory_swap_current",
            "memory_swap_max",
        )
        if not supported:
            if cgroup.get("path") is not None or any(
                cgroup.get(key) is not None for key in keys
            ):
                fail(f"{prefix} unsupported cgroup metrics are not null")
        else:
            this_path = nonempty_string(cgroup.get("path"), f"{prefix}.cgroup.path")
            cgroup_path = this_path if cgroup_path is None else cgroup_path
            if this_path != cgroup_path:
                fail("host cgroup path changed within the stream")
            current = exact_int(cgroup.get("memory_current"), "cgroup.current")
            swap_current = exact_int(
                cgroup.get("memory_swap_current"), "cgroup.swap_current"
            )
            if current < 0 or swap_current < 0:
                fail(f"{prefix} has a negative cgroup memory value")
            maximum = cgroup.get("memory_max")
            if maximum is not None:
                maximum = exact_int(maximum, "cgroup.max")
                if maximum <= 0 or current > maximum:
                    fail(f"{prefix} has invalid cgroup memory limits")
                free = maximum - current
                if free < minimum_cgroup_free_bytes:
                    fail(f"{prefix} violates the cgroup memory gate")
                minimum_cgroup_free = (
                    free
                    if minimum_cgroup_free is None
                    else min(minimum_cgroup_free, free)
                )
            swap_max = cgroup.get("memory_swap_max")
            if swap_max is not None:
                swap_max = exact_int(swap_max, "cgroup.swap_max")
                if swap_max < 0 or swap_current > swap_max:
                    fail(f"{prefix} has invalid cgroup swap limits")
            capacity = (maximum, swap_max)
            cgroup_capacity = capacity if cgroup_capacity is None else cgroup_capacity
            if capacity != cgroup_capacity:
                fail("host cgroup memory limits changed within the stream")

        temperatures = record.get("temperatures")
        if not isinstance(temperatures, list):
            fail(f"{prefix}.temperatures is not a list")
        sensors = set()
        for sensor in temperatures:
            if not isinstance(sensor, dict):
                fail(f"{prefix} has a non-object temperature sensor")
            device = nonempty_string(sensor.get("device"), "temperature.device")
            label = nonempty_string(sensor.get("label"), "temperature.label")
            value = exact_int(
                sensor.get("millidegrees_celsius"), "temperature.value"
            )
            if (device, label) in sensors or not -40000 <= value <= 200000:
                fail(f"{prefix} has duplicate or implausible temperature telemetry")
            sensors.add((device, label))
            if value > maximum_temperature_millidegrees:
                fail(f"{prefix} violates the host temperature gate")
            if (
                maximum_temperature is None
                or value > maximum_temperature["millidegrees_celsius"]
            ):
                maximum_temperature = {
                    "device": device,
                    "label": label,
                    "millidegrees_celsius": value,
                }

    if not records or records[-1].get("root_alive") is not False:
        fail("final host stream does not end with root_alive=false")
    if alive_samples == 0:
        fail("host monitor never observed the workload alive")
    gap = gap_summary(times, "host capture")
    if terminal.get("schema") != HOST_COMPLETE_SCHEMA:
        fail("host terminal has the wrong schema")
    if terminal.get("outcome") != "PASS" or terminal.get("terminal_error") is not None:
        fail("host terminal outcome is not PASS")
    if (
        terminal.get("root_pid") != expected_root_pid
        or terminal.get("root_start_time_ticks") != root_start
        or terminal.get("label") != expected_label
    ):
        fail("host terminal identity differs from the stream")
    if exact_int(terminal.get("sample_count"), "host terminal sample_count") != len(
        records
    ):
        fail("host terminal sample count differs from the stream")
    verify_terminal_timing(terminal, times, "host terminal")
    check_file_record(terminal.get("telemetry"), stream, payload, "host telemetry")
    monitor = check_bound_file(
        terminal.get("monitor_source_before"),
        terminal.get("monitor_source_after"),
        expected_monitor_sha256,
        "host monitor",
    )
    return {
        "schema": "gpmeep-host-telemetry-replay-v2",
        "status": "PASS",
        "path": str(stream),
        "sha256": sha256_bytes(payload),
        "sample_count": len(records),
        "alive_sample_count": alive_samples,
        "root_pid": expected_root_pid,
        "root_start_time_ticks": root_start,
        "label": expected_label,
        "first_capture_unix_seconds": times[0],
        "last_capture_unix_seconds": times[-1],
        "cadence_seconds": gap,
        "maximum_process_rss_bytes": maximum_rss,
        "minimum_mem_available_bytes": minimum_available,
        "minimum_swap_free_bytes": minimum_swap_free,
        "minimum_cgroup_free_bytes": minimum_cgroup_free,
        "maximum_temperature": maximum_temperature,
        "observed_process_pids": sorted(process_pids),
        "observed_process_identities": [
            {"pid": pid, "start_time_ticks": start}
            for pid, start in sorted(process_identities)
        ],
        "monitor_source": monitor,
    }


def metric_value(metric: Any, kind: str, name: str, *, required: bool) -> Any:
    if not isinstance(metric, dict) or set(metric) != {"supported", "value", "raw"}:
        fail(f"{name} metric representation is invalid")
    supported = metric["supported"]
    raw = metric["raw"]
    if not isinstance(supported, bool) or not isinstance(raw, str) or not raw:
        fail(f"{name} metric support or raw value is invalid")
    normalized = raw.strip().strip("[]").strip().lower()
    if not supported:
        if required or metric["value"] is not None or normalized not in UNSUPPORTED_VALUES:
            fail(f"{name} unsupported metric representation is inconsistent")
        return None
    if normalized in UNSUPPORTED_VALUES:
        fail(f"{name} supported metric contains an unsupported raw value")
    value = metric["value"]
    try:
        if kind == "float":
            retained = finite_number(value, name)
            parsed = float(raw)
            if not math.isfinite(parsed) or retained != parsed:
                fail(f"{name} retained value differs from raw telemetry")
            return retained
        if kind == "int":
            retained = exact_int(value, name)
            if retained != int(raw) or retained < 0:
                fail(f"{name} retained value differs from raw telemetry")
            return retained
        if kind == "bool":
            if not isinstance(value, bool):
                fail(f"{name} Boolean value is invalid")
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
            if normalized not in choices or choices[normalized] is not value:
                fail(f"{name} retained value differs from raw telemetry")
            return value
        if kind == "string":
            retained = nonempty_string(value, name)
            if retained != raw:
                fail(f"{name} retained value differs from raw telemetry")
            return retained
    except ValueError as exc:
        fail(f"{name} cannot be parsed from raw telemetry: {exc}")
    fail(f"{name} has an unknown metric representation")


def verify_gpu(
    stream_path: pathlib.Path,
    terminal_path: pathlib.Path,
    *,
    expected_root_pid: int,
    expected_root_start_time_ticks: int,
    expected_label: str,
    expected_uuids: tuple[str, ...],
    expected_monitor_sha256: str,
    expected_nvidia_smi_sha256: str,
    host_process_pids: set[int],
    minimum_free_memory_fraction: float,
    maximum_temperature_celsius: float,
    expected_interval_seconds: float | None,
    expected_process_started_unix_seconds: float | None,
    expected_process_ended_unix_seconds: float | None,
    reject_active_clock_event_reasons: bool,
) -> dict[str, Any]:
    stream, payload, records = read_jsonl(stream_path)
    _terminal_file, terminal = load_json(terminal_path, "GPU terminal")
    if (
        expected_root_pid <= 1
        or expected_root_start_time_ticks <= 0
        or not expected_label
        or len(expected_uuids) not in {1, 2}
        or len(set(expected_uuids)) != len(expected_uuids)
        or any(not value.startswith("GPU-") for value in expected_uuids)
    ):
        fail("GPU replay external identity is invalid")
    if (
        not math.isfinite(minimum_free_memory_fraction)
        or not 0.0 <= minimum_free_memory_fraction < 1.0
        or not math.isfinite(maximum_temperature_celsius)
        or maximum_temperature_celsius <= 0
    ):
        fail("GPU replay memory or temperature gate is invalid")
    if not isinstance(reject_active_clock_event_reasons, bool):
        fail("GPU clock-event policy is not Boolean")
    if reject_active_clock_event_reasons and (
        expected_interval_seconds is None
        or expected_process_started_unix_seconds is None
        or expected_process_ended_unix_seconds is None
    ):
        fail("release performance clock-event policy requires sealed coverage")

    times: list[float] = []
    monotonic_times: list[float] = []
    seen_dead = False
    alive_samples = 0
    observed_apps: set[tuple[str, int, str]] = set()
    observed_workload_uuids: set[str] = set()
    extrema: dict[str, dict[str, float]] = {
        uuid: {
            "peak_memory_used_mib": 0.0,
            "minimum_free_memory_fraction": 1.0,
            "peak_temperature_celsius": -math.inf,
            "peak_power_draw_watts": 0.0,
            "peak_gpu_utilization_percent": 0.0,
        }
        for uuid in expected_uuids
    }
    reliability_baseline: dict[str, dict[str, tuple[bool, Any]]] = {
        uuid: {} for uuid in expected_uuids
    }
    device_identity: dict[str, tuple[int, str, float]] = {}
    clock_event_reasons: dict[str, dict[str, dict[str, Any]]] = {
        uuid: {
            name: {
                "active_sample_count": 0,
                "active_alive_sample_count": 0,
                "first_active_unix_seconds": None,
                "last_active_unix_seconds": None,
            }
            for name in CLOCK_EVENT_REASON_METRICS
        }
        for uuid in expected_uuids
    }

    for sequence, record in enumerate(records):
        prefix = f"gpu[{sequence}]"
        if record.get("schema") != GPU_SAMPLE_SCHEMA:
            fail(f"{prefix} has the wrong schema")
        if exact_int(record.get("sequence"), f"{prefix}.sequence") != sequence:
            fail(f"{prefix} has a non-contiguous sequence")
        if (
            exact_int(record.get("root_pid"), f"{prefix}.root_pid")
            != expected_root_pid
            or exact_int(record.get("root_start_time_ticks"), f"{prefix}.root_start")
            != expected_root_start_time_ticks
            or record.get("label") != expected_label
        ):
            fail(f"{prefix} differs from the sealed process identity")
        selected = record.get("selected_gpu_uuids")
        if selected != list(expected_uuids):
            fail(f"{prefix} selected UUID order differs from the external seal")
        times.append(
            finite_number(record.get("captured_unix_seconds"), f"{prefix}.captured")
        )
        monotonic_times.append(
            finite_number(
                record.get("captured_monotonic_seconds"),
                f"{prefix}.captured_monotonic",
            )
        )
        alive = record.get("root_alive")
        if not isinstance(alive, bool):
            fail(f"{prefix}.root_alive is not Boolean")
        if seen_dead and alive:
            fail("GPU root became alive after a terminal sample")
        seen_dead = seen_dead or not alive
        alive_samples += int(alive)
        if record.get("query_error") is not None:
            fail(f"{prefix} contains a GPU query error")
        gpus = record.get("gpus")
        apps = record.get("compute_applications")
        if not isinstance(gpus, list) or not isinstance(apps, list):
            fail(f"{prefix} device or compute-app inventory is invalid")
        if len(gpus) != len(expected_uuids):
            fail(f"{prefix} GPU inventory count differs from the selection")
        if [gpu.get("uuid") if isinstance(gpu, dict) else None for gpu in gpus] != list(
            expected_uuids
        ):
            fail(f"{prefix} GPU inventory is duplicate or reordered")
        indices = []
        for gpu in gpus:
            uuid = nonempty_string(gpu.get("uuid"), f"{prefix}.uuid")
            index = exact_int(gpu.get("index"), f"{prefix}.{uuid}.index")
            nonempty_string(gpu.get("name"), f"{prefix}.{uuid}.name")
            if index < 0:
                fail(f"{prefix}.{uuid} has a negative device index")
            indices.append(index)
            values = {
                name: metric_value(gpu.get(name), kind, f"{prefix}.{uuid}.{name}", required=True)
                for name, kind in GPU_REQUIRED_METRICS.items()
            }
            optional = {
                name: metric_value(
                    gpu.get(name), kind, f"{prefix}.{uuid}.{name}", required=False
                )
                for name, kind in GPU_OPTIONAL_METRICS.items()
            }
            expected_gpu_keys = {
                "index",
                "uuid",
                "name",
                *GPU_REQUIRED_METRICS,
                *GPU_OPTIONAL_METRICS,
            }
            if set(gpu) != expected_gpu_keys:
                fail(f"{prefix}.{uuid} metric inventory differs")
            used = values["memory_used_mib"]
            total = values["memory_total_mib"]
            temperature = values["temperature_celsius"]
            for name in CLOCK_EVENT_REASON_METRICS:
                if values[name]:
                    retained_reason = clock_event_reasons[uuid][name]
                    retained_reason["active_sample_count"] += 1
                    retained_reason["active_alive_sample_count"] += int(alive)
                    retained_reason["first_active_unix_seconds"] = (
                        times[-1]
                        if retained_reason["first_active_unix_seconds"] is None
                        else retained_reason["first_active_unix_seconds"]
                    )
                    retained_reason["last_active_unix_seconds"] = times[-1]
                    if reject_active_clock_event_reasons and alive:
                        fail(
                            f"{prefix}.{uuid}.{name} is active during a "
                            "release performance process"
                        )
            identity = (index, gpu["name"], total)
            device_identity.setdefault(uuid, identity)
            if identity != device_identity[uuid]:
                fail(f"{prefix}.{uuid} device identity changed")
            if used < 0 or total <= 0 or used > total:
                fail(f"{prefix}.{uuid} has invalid device memory")
            free_fraction = 1.0 - used / total
            if free_fraction < minimum_free_memory_fraction:
                fail(f"{prefix}.{uuid} violates the GPU free-memory gate")
            if not -100 <= temperature <= maximum_temperature_celsius:
                fail(f"{prefix}.{uuid} violates the GPU temperature gate")
            temperature_limit = optional["temperature_limit_celsius"]
            if temperature_limit is not None and temperature >= temperature_limit:
                fail(f"{prefix}.{uuid} reached its reported temperature limit")
            for name in ("gpu_utilization_percent", "memory_utilization_percent"):
                if optional[name] is not None and not 0 <= optional[name] <= 100:
                    fail(f"{prefix}.{uuid}.{name} lies outside [0,100]")
            for name in (
                "power_draw_watts",
                "power_limit_watts",
                "sm_clock_mhz",
                "memory_clock_mhz",
            ):
                if optional[name] is not None and optional[name] < 0:
                    fail(f"{prefix}.{uuid}.{name} is negative")

            current = extrema[uuid]
            current["peak_memory_used_mib"] = max(
                current["peak_memory_used_mib"], used
            )
            current["minimum_free_memory_fraction"] = min(
                current["minimum_free_memory_fraction"], free_fraction
            )
            current["peak_temperature_celsius"] = max(
                current["peak_temperature_celsius"], temperature
            )
            if optional["power_draw_watts"] is not None:
                current["peak_power_draw_watts"] = max(
                    current["peak_power_draw_watts"], optional["power_draw_watts"]
                )
            if optional["gpu_utilization_percent"] is not None:
                current["peak_gpu_utilization_percent"] = max(
                    current["peak_gpu_utilization_percent"],
                    optional["gpu_utilization_percent"],
                )

            for name in MONOTONIC_COUNTER_METRICS + FAULT_BOOLEAN_METRICS:
                metric = gpu[name]
                state = (metric["supported"], optional[name])
                baseline = reliability_baseline[uuid].setdefault(name, state)
                if state != baseline:
                    fail(f"{prefix}.{uuid}.{name} changed during the workload")
                if name in FAULT_BOOLEAN_METRICS and state == (True, True):
                    fail(f"{prefix}.{uuid}.{name} reports an active fault")
        if len(set(indices)) != len(indices):
            fail(f"{prefix} GPU indices are duplicate")

        app_identities = set()
        for index, app in enumerate(apps):
            if not isinstance(app, dict):
                fail(f"{prefix}.apps[{index}] is not an object")
            uuid = nonempty_string(app.get("gpu_uuid"), "compute_app.gpu_uuid")
            pid = exact_int(app.get("pid"), "compute_app.pid")
            name = nonempty_string(app.get("process_name"), "compute_app.name")
            app_memory = metric_value(
                app.get("used_gpu_memory_mib"),
                "float",
                "compute_app.used_gpu_memory_mib",
                required=False,
            )
            identity = (uuid, pid)
            if (
                uuid not in expected_uuids
                or pid <= 0
                or identity in app_identities
                or (app_memory is not None and app_memory < 0)
            ):
                fail(f"{prefix} has an invalid or duplicate compute application")
            app_identities.add(identity)
            if pid not in host_process_pids:
                fail(f"{prefix} contains a compute app outside the host process tree")
            observed_apps.add((uuid, pid, name))
            if alive:
                observed_workload_uuids.add(uuid)
        if not alive and apps:
            fail(f"{prefix} idle boundary still contains compute applications")

    if not records or records[-1].get("root_alive") is not False:
        fail("final GPU stream does not end with root_alive=false")
    if alive_samples == 0:
        fail("GPU monitor never observed the workload alive")
    missing_apps = set(expected_uuids) - observed_workload_uuids
    if missing_apps:
        fail(f"GPU workload process was never observed on: {sorted(missing_apps)}")
    gap_summary(times, "GPU wall-clock capture")
    gap = gap_summary(monotonic_times, "GPU monotonic capture")
    if terminal.get("schema") != GPU_COMPLETE_SCHEMA:
        fail("GPU terminal has the wrong schema")
    if terminal.get("outcome") != "PASS" or terminal.get("terminal_error") is not None:
        fail("GPU terminal outcome is not PASS")
    if (
        terminal.get("root_pid") != expected_root_pid
        or terminal.get("root_start_time_ticks") != expected_root_start_time_ticks
        or terminal.get("label") != expected_label
        or terminal.get("selected_gpu_uuids") != list(expected_uuids)
    ):
        fail("GPU terminal identity differs from the stream")
    if exact_int(terminal.get("sample_count"), "GPU terminal sample_count") != len(
        records
    ):
        fail("GPU terminal sample count differs from the stream")
    if exact_int(terminal.get("query_error_count"), "GPU terminal errors") != 0:
        fail("GPU terminal records query errors")
    interval = verify_terminal_timing(terminal, times, "GPU terminal")
    if expected_interval_seconds is not None:
        expected_interval = finite_number(
            expected_interval_seconds, "GPU expected interval"
        )
        if interval != expected_interval:
            fail("GPU terminal interval differs from the sealed interval")
    else:
        expected_interval = interval
    terminal_monotonic_start = finite_number(
        terminal.get("started_monotonic_seconds"), "GPU terminal monotonic start"
    )
    terminal_monotonic_end = finite_number(
        terminal.get("ended_monotonic_seconds"), "GPU terminal monotonic end"
    )
    if not (
        terminal_monotonic_start
        <= monotonic_times[0]
        <= monotonic_times[-1]
        <= terminal_monotonic_end
    ):
        fail("GPU monotonic terminal bounds do not contain every sample")
    maximum_gap = expected_interval * 2.0 + 1.0
    if gap["maximum"] is not None and gap["maximum"] > maximum_gap:
        fail("GPU telemetry cadence has an uncovered gap")
    if (
        expected_process_started_unix_seconds is None
    ) != (expected_process_ended_unix_seconds is None):
        fail("GPU process coverage requires both start and end")
    coverage = None
    if expected_process_started_unix_seconds is not None:
        process_started = finite_number(
            expected_process_started_unix_seconds, "GPU process start"
        )
        process_ended = finite_number(
            expected_process_ended_unix_seconds, "GPU process end"
        )
        if process_ended < process_started:
            fail("GPU process coverage bounds are reversed")
        if times[0] > process_started + maximum_gap:
            fail("GPU telemetry has a start-coverage gap")
        if times[-1] < process_ended:
            fail("GPU telemetry has an end-coverage gap")
        coverage = {
            "process_started_unix_seconds": process_started,
            "process_ended_unix_seconds": process_ended,
            "maximum_allowed_gap_seconds": maximum_gap,
            "start_lag_seconds": times[0] - process_started,
            "end_tail_seconds": times[-1] - process_ended,
            "outcome": "PASS",
        }
    check_file_record(terminal.get("telemetry"), stream, payload, "GPU telemetry")
    monitor = check_bound_file(
        terminal.get("monitor_source_before"),
        terminal.get("monitor_source_after"),
        expected_monitor_sha256,
        "GPU monitor",
    )
    nvidia_smi = check_bound_file(
        terminal.get("nvidia_smi_before"),
        terminal.get("nvidia_smi_after"),
        expected_nvidia_smi_sha256,
        "nvidia-smi",
    )
    return {
        "schema": "gpmeep-gpu-telemetry-replay-v3",
        "status": "PASS",
        "path": str(stream),
        "sha256": sha256_bytes(payload),
        "sample_count": len(records),
        "alive_sample_count": alive_samples,
        "root_pid": expected_root_pid,
        "root_start_time_ticks": expected_root_start_time_ticks,
        "label": expected_label,
        "selected_gpu_uuids": list(expected_uuids),
        "first_capture_unix_seconds": times[0],
        "last_capture_unix_seconds": times[-1],
        "cadence_seconds": gap,
        "process_coverage": coverage,
        "clock_event_policy": (
            "reject-active-during-alive-performance-samples"
            if reject_active_clock_event_reasons
            else "disclose-active-reasons"
        ),
        "clock_event_reasons": clock_event_reasons,
        "observed_compute_applications": [
            {"gpu_uuid": uuid, "pid": pid, "process_name": name}
            for uuid, pid, name in sorted(observed_apps)
        ],
        "per_gpu_extrema": extrema,
        "monitor_source": monitor,
        "nvidia_smi": nvidia_smi,
    }


def verify_pair(
    *,
    host_stream: pathlib.Path,
    host_terminal: pathlib.Path,
    gpu_stream: pathlib.Path | None,
    gpu_terminal: pathlib.Path | None,
    expected_root_pid: int,
    expected_label: str,
    expected_host_monitor_sha256: str,
    expected_gpu_uuids: tuple[str, ...] = (),
    expected_gpu_monitor_sha256: str | None = None,
    expected_nvidia_smi_sha256: str | None = None,
    minimum_host_available_bytes: int = 4 * 1024**3,
    minimum_cgroup_free_bytes: int = 4 * 1024**3,
    maximum_host_temperature_millidegrees: int = 95_000,
    minimum_gpu_free_memory_fraction: float = 0.20,
    maximum_gpu_temperature_celsius: float = 90.0,
    expected_interval_seconds: float | None = None,
    expected_process_started_unix_seconds: float | None = None,
    expected_process_ended_unix_seconds: float | None = None,
    reject_active_clock_event_reasons: bool = False,
) -> dict[str, Any]:
    host = verify_host(
        host_stream,
        host_terminal,
        expected_root_pid=expected_root_pid,
        expected_label=expected_label,
        expected_monitor_sha256=expected_host_monitor_sha256,
        minimum_available_bytes=minimum_host_available_bytes,
        minimum_cgroup_free_bytes=minimum_cgroup_free_bytes,
        maximum_temperature_millidegrees=maximum_host_temperature_millidegrees,
    )
    if (gpu_stream is None) != (gpu_terminal is None):
        fail("GPU stream and terminal must be supplied together")
    if gpu_stream is None:
        if expected_gpu_uuids or expected_gpu_monitor_sha256 or expected_nvidia_smi_sha256:
            fail("GPU external seals were supplied without GPU telemetry")
        gpu = None
    else:
        if expected_gpu_monitor_sha256 is None or expected_nvidia_smi_sha256 is None:
            fail("final GPU replay requires monitor and nvidia-smi SHA-256 seals")
        gpu = verify_gpu(
            gpu_stream,
            gpu_terminal,
            expected_root_pid=expected_root_pid,
            expected_root_start_time_ticks=host["root_start_time_ticks"],
            expected_label=expected_label,
            expected_uuids=expected_gpu_uuids,
            expected_monitor_sha256=expected_gpu_monitor_sha256,
            expected_nvidia_smi_sha256=expected_nvidia_smi_sha256,
            host_process_pids=set(host["observed_process_pids"]),
            minimum_free_memory_fraction=minimum_gpu_free_memory_fraction,
            maximum_temperature_celsius=maximum_gpu_temperature_celsius,
            expected_interval_seconds=expected_interval_seconds,
            expected_process_started_unix_seconds=(
                expected_process_started_unix_seconds
            ),
            expected_process_ended_unix_seconds=expected_process_ended_unix_seconds,
            reject_active_clock_event_reasons=reject_active_clock_event_reasons,
        )
        if max(
            host["first_capture_unix_seconds"], gpu["first_capture_unix_seconds"]
        ) > min(host["last_capture_unix_seconds"], gpu["last_capture_unix_seconds"]):
            fail("host and GPU telemetry capture intervals do not overlap")
    verifier_path = pathlib.Path(__file__).resolve(strict=True)
    verifier_payload = verifier_path.read_bytes()
    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "host": host,
        "gpu": gpu,
        "verifier_source": {
            "path": str(verifier_path),
            "size_bytes": len(verifier_payload),
            "sha256": sha256_bytes(verifier_payload),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--host-stream", required=True, type=pathlib.Path)
    parser.add_argument("--host-terminal", required=True, type=pathlib.Path)
    parser.add_argument("--gpu-stream", type=pathlib.Path)
    parser.add_argument("--gpu-terminal", type=pathlib.Path)
    parser.add_argument("--expected-root-pid", required=True, type=int)
    parser.add_argument("--expected-label", required=True)
    parser.add_argument("--expected-host-monitor-sha256", required=True)
    parser.add_argument("--expected-gpu-uuid", action="append", default=[])
    parser.add_argument("--expected-gpu-monitor-sha256")
    parser.add_argument("--expected-nvidia-smi-sha256")
    parser.add_argument(
        "--minimum-host-available-bytes", type=int, default=4 * 1024**3
    )
    parser.add_argument(
        "--minimum-cgroup-free-bytes", type=int, default=4 * 1024**3
    )
    parser.add_argument(
        "--maximum-host-temperature-millidegrees", type=int, default=95_000
    )
    parser.add_argument(
        "--minimum-gpu-free-memory-fraction", type=float, default=0.20
    )
    parser.add_argument(
        "--maximum-gpu-temperature-celsius", type=float, default=90.0
    )
    parser.add_argument("--expected-interval-seconds", type=float)
    parser.add_argument("--expected-process-started-unix-seconds", type=float)
    parser.add_argument("--expected-process-ended-unix-seconds", type=float)
    parser.add_argument("--reject-active-clock-event-reasons", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_pair(
        host_stream=args.host_stream,
        host_terminal=args.host_terminal,
        gpu_stream=args.gpu_stream,
        gpu_terminal=args.gpu_terminal,
        expected_root_pid=args.expected_root_pid,
        expected_label=args.expected_label,
        expected_host_monitor_sha256=args.expected_host_monitor_sha256,
        expected_gpu_uuids=tuple(args.expected_gpu_uuid),
        expected_gpu_monitor_sha256=args.expected_gpu_monitor_sha256,
        expected_nvidia_smi_sha256=args.expected_nvidia_smi_sha256,
        minimum_host_available_bytes=args.minimum_host_available_bytes,
        minimum_cgroup_free_bytes=args.minimum_cgroup_free_bytes,
        maximum_host_temperature_millidegrees=(
            args.maximum_host_temperature_millidegrees
        ),
        minimum_gpu_free_memory_fraction=args.minimum_gpu_free_memory_fraction,
        maximum_gpu_temperature_celsius=args.maximum_gpu_temperature_celsius,
        expected_interval_seconds=args.expected_interval_seconds,
        expected_process_started_unix_seconds=(
            args.expected_process_started_unix_seconds
        ),
        expected_process_ended_unix_seconds=args.expected_process_ended_unix_seconds,
        reject_active_clock_event_reasons=(
            args.reject_active_clock_event_reasons
        ),
    )
    if args.output is None:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        atomic_report(args.output, report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"telemetry v2 replay failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2)
