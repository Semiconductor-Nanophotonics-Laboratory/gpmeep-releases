#!/usr/bin/env python3
"""Focused tests for source-bound host/process telemetry v2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "user-workloads" / "capture_host_telemetry_v2.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_host_telemetry_v2", SOURCE)
assert SPEC is not None and SPEC.loader is not None
TELEMETRY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TELEMETRY
SPEC.loader.exec_module(TELEMETRY)


def process(pid: int, ppid: int, *, start: int = 100) -> dict[str, object]:
    return {
        "pid": pid,
        "ppid": ppid,
        "pgrp": 42,
        "state": "R",
        "start_time_ticks": start,
        "processor": 0,
        "rss_bytes": pid,
        "comm": f"fixture-{pid}",
    }


MEMORY = {
    "MemTotal": 1000,
    "MemAvailable": 800,
    "SwapTotal": 100,
    "SwapFree": 90,
}
CGROUP = {
    "supported": True,
    "path": "/fixture",
    "memory_current": 200,
    "memory_max": 1000,
    "memory_swap_current": 0,
    "memory_swap_max": 100,
}


class HostTelemetryTests(unittest.TestCase):
    def test_descendants_are_transitive_and_sorted(self) -> None:
        records = {
            44: process(44, 43),
            42: process(42, 1),
            43: process(43, 42),
            99: process(99, 1),
        }
        self.assertEqual(
            [record["pid"] for record in TELEMETRY.descendants(records, 42)],
            [42, 43, 44],
        )

    def test_cgroup_unsupported_is_explicit_not_zero(self) -> None:
        self.assertEqual(
            TELEMETRY.cgroup_snapshot(None),
            {
                "supported": False,
                "path": None,
                "memory_current": None,
                "memory_max": None,
                "memory_swap_current": None,
                "memory_swap_max": None,
            },
        )

    def test_main_writes_contiguous_stream_and_source_bound_terminal(self) -> None:
        root = process(42, 1)
        child = process(43, 42, start=101)
        records = [{42: root}, {42: root, 43: child}, {}]
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw) / "host.jsonl"
            argv = [
                "--root-pid",
                "42",
                "--output",
                str(output),
                "--label",
                "fixture",
                "--interval-seconds",
                "1",
            ]
            with mock.patch.object(
                TELEMETRY, "proc_records", side_effect=records
            ), mock.patch.object(
                TELEMETRY, "cgroup_location", return_value=("/fixture", pathlib.Path("/ignored"))
            ), mock.patch.object(
                TELEMETRY, "cgroup_snapshot", return_value=CGROUP
            ), mock.patch.object(
                TELEMETRY, "memory_snapshot", return_value=MEMORY
            ), mock.patch.object(
                TELEMETRY, "temperature_snapshot", return_value=[]
            ), mock.patch.object(
                TELEMETRY.os, "getloadavg", return_value=(1.0, 1.0, 1.0)
            ), mock.patch.object(TELEMETRY.time, "sleep", return_value=None):
                self.assertEqual(TELEMETRY.main(argv), 0)

            samples = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([item["sequence"] for item in samples], [0, 1])
            self.assertEqual([item["root_alive"] for item in samples], [True, False])
            self.assertEqual(samples[0]["root_start_time_ticks"], 100)
            self.assertEqual(samples[0]["process_count"], 2)
            self.assertEqual(samples[0]["process_rss_bytes"], 85)
            terminal_path = output.with_name(output.name + ".COMPLETE.json")
            terminal = json.loads(terminal_path.read_text())
            self.assertEqual(terminal["outcome"], "PASS")
            self.assertEqual(terminal["sample_count"], 2)
            self.assertEqual(
                terminal["telemetry"]["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                terminal["monitor_source_before"],
                terminal["monitor_source_after"],
            )
            self.assertEqual(
                terminal["monitor_source_before"]["path"], str(SOURCE)
            )

    def test_pid_reuse_is_terminal_not_a_new_workload(self) -> None:
        initial = process(42, 1, start=100)
        replacement = process(42, 1, start=999)
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw) / "host.jsonl"
            with mock.patch.object(
                TELEMETRY,
                "proc_records",
                side_effect=[{42: initial}, {42: replacement}],
            ), mock.patch.object(
                TELEMETRY, "cgroup_location", return_value=None
            ), mock.patch.object(
                TELEMETRY, "cgroup_snapshot", return_value=TELEMETRY.cgroup_snapshot(None)
            ), mock.patch.object(
                TELEMETRY, "memory_snapshot", return_value=MEMORY
            ), mock.patch.object(
                TELEMETRY, "temperature_snapshot", return_value=[]
            ), mock.patch.object(
                TELEMETRY.os, "getloadavg", return_value=(0.0, 0.0, 0.0)
            ):
                self.assertEqual(
                    TELEMETRY.main(
                        [
                            "--root-pid",
                            "42",
                            "--output",
                            str(output),
                            "--label",
                            "fixture",
                            "--interval-seconds",
                            "1",
                        ]
                    ),
                    0,
                )
            sample = json.loads(output.read_text().strip())
            self.assertFalse(sample["root_alive"])
            self.assertEqual(sample["root_start_time_ticks"], 100)

    def test_existing_symlink_and_invalid_interval_fail_before_capture(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = pathlib.Path(raw)
            target = directory / "target"
            target.write_text("fixture\n")
            output = directory / "host.jsonl"
            output.symlink_to(target)
            root = process(42, 1)
            with mock.patch.object(TELEMETRY, "proc_records", return_value={42: root}):
                with self.assertRaises(FileExistsError):
                    TELEMETRY.main(
                        [
                            "--root-pid",
                            "42",
                            "--output",
                            str(output),
                            "--label",
                            "fixture",
                        ]
                    )
                with self.assertRaisesRegex(ValueError, "interval"):
                    TELEMETRY.main(
                        [
                            "--root-pid",
                            "42",
                            "--output",
                            str(directory / "new.jsonl"),
                            "--label",
                            "fixture",
                            "--interval-seconds",
                            "0.1",
                        ]
                    )

    def test_dangling_output_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw) / "host.jsonl"
            output.symlink_to(pathlib.Path(raw) / "missing-target")
            with self.assertRaisesRegex(FileExistsError, "symlink"):
                TELEMETRY.new_output_paths(output)


if __name__ == "__main__":
    unittest.main()
