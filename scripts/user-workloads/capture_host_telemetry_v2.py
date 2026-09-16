#!/usr/bin/env python3
"""Capture durable host/process telemetry for one sealed task process tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import stat
import sys
import time
from typing import Any


SAMPLE_SCHEMA = "gpmeep-host-process-telemetry-sample-v2"
COMPLETE_SCHEMA = "gpmeep-host-process-telemetry-complete-v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root-pid", required=True, type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--label", required=True)
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
                raise RuntimeError("host telemetry terminal write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def proc_records() -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            close = raw.rfind(")")
            if close <= 1:
                continue
            fields = raw[close + 2 :].split()
            if len(fields) <= 36:
                continue
            status = (entry / "status").read_text(encoding="utf-8")
            rss_bytes = 0
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    rss_bytes = int(line.split()[1]) * 1024
                    break
            pid = int(entry.name)
            records[pid] = {
                "pid": pid,
                "ppid": int(fields[1]),
                "pgrp": int(fields[2]),
                "state": fields[0],
                "start_time_ticks": int(fields[19]),
                "processor": int(fields[36]),
                "rss_bytes": rss_bytes,
                "comm": raw[raw.find("(") + 1 : close],
            }
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            ValueError,
            IndexError,
        ):
            continue
    return records


def descendants(
    records: dict[int, dict[str, Any]], root_pid: int
) -> list[dict[str, Any]]:
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, record in records.items():
            if pid not in selected and record["ppid"] in selected:
                selected.add(pid)
                changed = True
    return [records[pid] for pid in sorted(selected) if pid in records]


def memory_snapshot() -> dict[str, int]:
    result: dict[str, int] = {}
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in required:
            result[key] = int(value.split()[0]) * 1024
    if set(result) != required:
        raise RuntimeError("host memory telemetry is incomplete")
    return result


def _cgroup_number(path: pathlib.Path) -> int | None:
    raw = path.read_text(encoding="utf-8").strip()
    if raw == "max":
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"negative cgroup value: {path}")
    return value


def cgroup_location(root_pid: int) -> tuple[str, pathlib.Path] | None:
    unified = None
    for line in (pathlib.Path("/proc") / str(root_pid) / "cgroup").read_text(
        encoding="utf-8"
    ).splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0" and controllers == "":
            unified = pathlib.PurePosixPath(relative)
            break
    if unified is None:
        return None
    if not unified.is_absolute() or ".." in unified.parts:
        raise RuntimeError("task has an invalid unified cgroup path")
    directory = pathlib.Path("/sys/fs/cgroup").joinpath(*unified.parts[1:])
    return unified.as_posix(), directory


def cgroup_snapshot(
    location: tuple[str, pathlib.Path] | None,
) -> dict[str, Any]:
    if location is None:
        return {
            "supported": False,
            "path": None,
            "memory_current": None,
            "memory_max": None,
            "memory_swap_current": None,
            "memory_swap_max": None,
        }
    unified, directory = location
    values = {
        name.replace(".", "_"): _cgroup_number(directory / name)
        for name in ("memory.current", "memory.max", "memory.swap.current", "memory.swap.max")
    }
    return {"supported": True, "path": unified, **values}


def temperature_snapshot() -> list[dict[str, Any]]:
    result = []
    for directory in sorted(pathlib.Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            device = (directory / "name").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        for source in sorted(directory.glob("temp*_input")):
            try:
                value = int(source.read_text(encoding="utf-8").strip())
                label_path = source.with_name(source.name.replace("_input", "_label"))
                label = (
                    label_path.read_text(encoding="utf-8").strip()
                    if label_path.is_file()
                    else source.stem
                )
            except (OSError, ValueError):
                continue
            result.append(
                {
                    "device": device,
                    "label": label,
                    "millidegrees_celsius": value,
                }
            )
    return result


def new_output_paths(requested: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    if requested.is_symlink():
        raise FileExistsError("host telemetry output is a symlink")
    output = pathlib.Path(os.path.abspath(requested))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve(strict=True) != output.parent:
        raise RuntimeError("host telemetry output parent contains a symlink")
    complete = output.with_name(output.name + ".COMPLETE.json")
    if output.exists() or output.is_symlink() or complete.exists() or complete.is_symlink():
        raise FileExistsError("host telemetry output or terminal already exists")
    return output, complete


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.root_pid <= 1 or not args.label:
        raise ValueError("host telemetry root PID and label are invalid")
    if (
        not math.isfinite(args.interval_seconds)
        or not 1.0 <= args.interval_seconds <= 60.0
    ):
        raise ValueError("host telemetry interval must be in [1,60] seconds")
    output, complete = new_output_paths(args.output)
    initial_records = proc_records()
    initial_root = initial_records.get(args.root_pid)
    if initial_root is None or initial_root["state"] in {"Z", "X"}:
        raise RuntimeError("host telemetry root process is not alive at capture start")
    root_start_time_ticks = int(initial_root["start_time_ticks"])
    initial_cgroup_location = cgroup_location(args.root_pid)
    source_before = absolute_file_record(pathlib.Path(__file__))
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
    sequence = 0
    outcome = "PASS"
    terminal_error = None
    try:
        deadline = time.monotonic()
        while True:
            records = proc_records()
            root = records.get(args.root_pid)
            root_alive = bool(
                root is not None
                and root["state"] not in {"Z", "X"}
                and int(root["start_time_ticks"]) == root_start_time_ticks
            )
            selected = descendants(records, args.root_pid)
            sample = {
                "schema": SAMPLE_SCHEMA,
                "sequence": sequence,
                "label": args.label,
                "captured_unix_seconds": time.time(),
                "root_pid": args.root_pid,
                "root_start_time_ticks": root_start_time_ticks,
                "root_alive": root_alive,
                "processes": selected,
                "process_count": len(selected),
                "process_rss_bytes": sum(
                    int(record["rss_bytes"]) for record in selected
                ),
                "load_average": list(os.getloadavg()),
                "memory_bytes": memory_snapshot(),
                "cgroup_memory": cgroup_snapshot(initial_cgroup_location),
                "temperatures": temperature_snapshot(),
            }
            payload = (
                json.dumps(sample, allow_nan=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise RuntimeError("host telemetry append made no progress")
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
    if source_before != source_after:
        outcome = "FAIL"
        terminal_error = "host telemetry monitor source changed"
    terminal = {
        "schema": COMPLETE_SCHEMA,
        "outcome": outcome,
        "terminal_error": terminal_error,
        "label": args.label,
        "root_pid": args.root_pid,
        "root_start_time_ticks": root_start_time_ticks,
        "interval_seconds": args.interval_seconds,
        "sample_count": sequence,
        "started_unix_seconds": started,
        "ended_unix_seconds": time.time(),
        "telemetry": absolute_file_record(output),
        "monitor_source_before": source_before,
        "monitor_source_after": source_after,
    }
    atomic_exclusive_json(complete, terminal)
    return 0 if outcome == "PASS" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"host telemetry error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
