#!/usr/bin/env python3
"""Run one bounded workload with fail-closed sibling observer processes."""

from __future__ import annotations

import math
import os
import pathlib
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable


from common import WorkloadError, file_record


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]


@dataclass(frozen=True)
class ObserverSpec:
    label: str
    command: tuple[str, ...]
    log_path: pathlib.Path


@dataclass
class RunningObserver:
    spec: ObserverSpec
    process: subprocess.Popen[bytes]
    log: Any
    started_unix_seconds: float
    started_monotonic: float


def _proc_group_alive(process_group: int) -> bool:
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            if int(fields[2]) == process_group and fields[0] not in {"Z", "X"}:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    for selected_signal, wait_seconds in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(process.pid, selected_signal)
        except ProcessLookupError:
            if process.poll() is None:
                process.wait(timeout=1.0)
            return
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            process.poll()
            if not _proc_group_alive(process.pid):
                if process.poll() is None:
                    process.wait(timeout=1.0)
                return
            time.sleep(0.05)
    raise WorkloadError(f"could not terminate process group {process.pid}")


def _validate_inputs(
    command: list[str],
    evidence_root: pathlib.Path,
    log_path: pathlib.Path,
    timeout_seconds: float,
    output_limit_bytes: int,
    observer_exit_timeout_seconds: float,
) -> pathlib.Path:
    if not command or not all(isinstance(item, str) and item for item in command):
        raise WorkloadError("observed workload command is invalid")
    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or type(output_limit_bytes) is not int
        or output_limit_bytes <= 0
        or not math.isfinite(observer_exit_timeout_seconds)
        or observer_exit_timeout_seconds <= 0
    ):
        raise WorkloadError("observed workload limits must be positive and finite")
    root = evidence_root.resolve(strict=True)
    for path, label in ((log_path, "workload log"),):
        absolute = pathlib.Path(os.path.abspath(path))
        try:
            absolute.relative_to(root)
        except ValueError as exc:
            raise WorkloadError(f"observed {label} escapes the evidence root") from exc
    return root


def _start_observers(
    specs: list[ObserverSpec], environment: dict[str, str], root: pathlib.Path
) -> list[RunningObserver]:
    if not specs or len({spec.label for spec in specs}) != len(specs):
        raise WorkloadError("observer inventory is empty or has duplicate labels")
    running = []
    try:
        for spec in specs:
            if (
                not spec.label
                or not spec.command
                or not all(isinstance(item, str) and item for item in spec.command)
            ):
                raise WorkloadError("observer specification is invalid")
            path = pathlib.Path(os.path.abspath(spec.log_path))
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise WorkloadError("observer log escapes the evidence root") from exc
            if path.is_symlink():
                raise WorkloadError("observer log is a symlink")
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open("xb")
            started_wall = time.time()
            started = time.monotonic()
            try:
                process = subprocess.Popen(
                    list(spec.command),
                    cwd=root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            except Exception:
                log.close()
                raise
            running.append(
                RunningObserver(spec, process, log, started_wall, started)
            )
    except Exception:
        for observer in running:
            if observer.process.poll() is None:
                _terminate_group(observer.process)
            observer.log.close()
        raise
    return running


def _finish_observers(
    running: list[RunningObserver],
    root: pathlib.Path,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    for observer in running:
        remaining = deadline - time.monotonic()
        try:
            observer.process.wait(timeout=max(remaining, 0.0))
        except subprocess.TimeoutExpired:
            _terminate_group(observer.process)
    ended = time.time()
    records = []
    for observer in running:
        observer.log.flush()
        os.fsync(observer.log.fileno())
        observer.log.close()
        records.append(
            {
                "label": observer.spec.label,
                "command": list(observer.spec.command),
                "command_pid": observer.process.pid,
                "returncode": observer.process.returncode,
                "started_unix_seconds": observer.started_unix_seconds,
                "ended_unix_seconds": ended,
                "wall_seconds": time.monotonic() - observer.started_monotonic,
                "log": file_record(
                    pathlib.Path(os.path.abspath(observer.spec.log_path)), root
                ),
            }
        )
    return records


def run_bounded_observed(
    command: list[str],
    environment: dict[str, str],
    evidence_root: pathlib.Path,
    log_path: pathlib.Path,
    timeout_seconds: float,
    output_limit_bytes: int,
    observer_factory: Callable[[int], list[ObserverSpec]],
    *,
    observer_environment: dict[str, str] | None = None,
    observer_exit_timeout_seconds: float = 90.0,
    observer_output_limit_bytes: int = 8 * 1024 * 1024,
    working_directory: pathlib.Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one task, killing it if a required observer exits early or fails."""

    root = _validate_inputs(
        command,
        evidence_root,
        log_path,
        timeout_seconds,
        output_limit_bytes,
        observer_exit_timeout_seconds,
    )
    if type(observer_output_limit_bytes) is not int or observer_output_limit_bytes <= 0:
        raise WorkloadError("observer output limit must be positive")
    working_directory = (
        REPO if working_directory is None else working_directory.resolve(strict=True)
    )
    if not working_directory.is_dir():
        raise WorkloadError("observed workload working directory is invalid")
    log_path = pathlib.Path(os.path.abspath(log_path))
    if log_path.is_symlink():
        raise WorkloadError("observed workload log is a symlink")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_wall = time.time()
    started = time.monotonic()
    peak_rss = 0
    output_bytes = 0
    timed_out = False
    output_limited = False
    interrupted_signal: signal.Signals | None = None
    observer_failure: str | None = None
    previous_handlers: dict[signal.Signals, Any] = {}

    def interrupt_controller(selected_signal: int, _frame: Any) -> None:
        nonlocal interrupted_signal
        interrupted_signal = signal.Signals(selected_signal)

    for selected_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[selected_signal] = signal.getsignal(selected_signal)
        signal.signal(selected_signal, interrupt_controller)

    process: subprocess.Popen[bytes] | None = None
    running: list[RunningObserver] = []
    selector = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        if process.stdout is None:
            raise WorkloadError("could not capture observed workload output")
        specs = observer_factory(process.pid)
        running = _start_observers(
            specs, observer_environment or environment, root
        )
        selector.register(process.stdout, selectors.EVENT_READ)
        with log_path.open("xb") as log:
            while True:
                if interrupted_signal is not None:
                    _terminate_group(process)
                    break
                elapsed = time.monotonic() - started
                peak_rss = max(peak_rss, _process_group_rss_bytes(process.pid))
                if elapsed > timeout_seconds:
                    timed_out = True
                    _terminate_group(process)
                for observer in running:
                    returncode = observer.process.poll()
                    observer_bytes = os.fstat(observer.log.fileno()).st_size
                    if observer_bytes > observer_output_limit_bytes:
                        observer_failure = (
                            f"observer {observer.spec.label} exceeded its output limit"
                        )
                        _terminate_group(process)
                        break
                    if returncode is not None and process.poll() is None:
                        observer_failure = (
                            f"observer {observer.spec.label} exited early with "
                            f"return code {returncode}"
                        )
                        _terminate_group(process)
                        break
                events = selector.select(timeout=0.25)
                for key, _ in events:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = output_limit_bytes - output_bytes
                    captured = chunk[:remaining]
                    log.write(captured)
                    output_bytes += len(captured)
                    if len(chunk) > remaining:
                        output_limited = True
                        _terminate_group(process)
                        break
                if timed_out or output_limited or observer_failure is not None:
                    break
                if process.poll() is not None and not selector.get_map():
                    break
            if process.poll() is None:
                process.wait(timeout=5.0)
            log.flush()
            os.fsync(log.fileno())
    finally:
        for selected_signal, handler in previous_handlers.items():
            signal.signal(selected_signal, handler)
        selector.close()
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if process is not None and process.poll() is None:
            _terminate_group(process)

    workload_ended_wall = time.time()
    workload_wall_seconds = time.monotonic() - started
    observer_records = _finish_observers(
        running, root, observer_exit_timeout_seconds
    )
    if interrupted_signal is not None:
        raise WorkloadError(f"controller received {interrupted_signal.name}")
    if observer_failure is not None:
        raise WorkloadError(observer_failure)
    failed_observers = [
        record for record in observer_records if record["returncode"] != 0
    ]
    if failed_observers:
        raise WorkloadError(
            "observer process failed: "
            + ", ".join(
                f"{record['label']}={record['returncode']}"
                for record in failed_observers
            )
        )
    return (
        {
            "command": command,
            "command_pid": process.pid,
            "returncode": process.returncode,
            "timed_out": timed_out,
            "output_limited": output_limited,
            "output_bytes": output_bytes,
            "peak_process_group_rss_bytes": peak_rss,
            "started_unix_seconds": started_wall,
            "ended_unix_seconds": workload_ended_wall,
            "wall_seconds": workload_wall_seconds,
            "log": file_record(log_path, root),
        },
        observer_records,
    )


def _process_group_rss_bytes(process_group: int) -> int:
    total = 0
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            if int(fields[2]) != process_group:
                continue
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total


def verify_observer_record(
    evidence_root: pathlib.Path,
    record: Any,
    expected: ObserverSpec,
) -> dict[str, Any]:
    keys = {
        "label",
        "command",
        "command_pid",
        "returncode",
        "started_unix_seconds",
        "ended_unix_seconds",
        "wall_seconds",
        "log",
    }
    if not isinstance(record, dict) or set(record) != keys:
        raise WorkloadError("observer process record schema is not exact")
    if (
        record["label"] != expected.label
        or record["command"] != list(expected.command)
        or type(record["command_pid"]) is not int
        or record["command_pid"] <= 1
        or record["returncode"] != 0
    ):
        raise WorkloadError(f"observer {expected.label} identity or outcome differs")
    started = record["started_unix_seconds"]
    ended = record["ended_unix_seconds"]
    wall = record["wall_seconds"]
    if (
        isinstance(started, bool)
        or not isinstance(started, (int, float))
        or isinstance(ended, bool)
        or not isinstance(ended, (int, float))
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not all(math.isfinite(float(value)) for value in (started, ended, wall))
        or ended < started
        or wall <= 0
    ):
        raise WorkloadError(f"observer {expected.label} timing differs")
    root = evidence_root.resolve(strict=True)
    expected_path = pathlib.Path(os.path.abspath(expected.log_path))
    expected_record = file_record(expected_path, root)
    if record["log"] != expected_record:
        raise WorkloadError(f"observer {expected.label} log record differs")
    return record
