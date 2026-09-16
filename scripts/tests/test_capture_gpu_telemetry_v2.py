#!/usr/bin/env python3
"""Focused tests for source-bound NVIDIA telemetry v2."""

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
SOURCE = ROOT / "scripts" / "user-workloads" / "capture_gpu_telemetry_v2.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_gpu_telemetry_v2", SOURCE)
assert SPEC is not None and SPEC.loader is not None
TELEMETRY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TELEMETRY
SPEC.loader.exec_module(TELEMETRY)


def gpu_row(uuid: str = "GPU-one", index: int = 0) -> str:
    values = [
        str(index),
        uuid,
        "NVIDIA Fixture",
        "P8",
        "33",
        "[N/A]",
        "21.47",
        "450.00",
        "210",
        "405",
        "1",
        "24564",
        "0",
        "0",
        "Not Active",
        "Not Active",
        "Not Active",
        "Not Active",
        "[N/A]",
        "[N/A]",
        "[N/A]",
        "[N/A]",
        "[N/A]",
        "[N/A]",
        "[N/A]",
        "0",
        "0",
        "No",
        "No",
    ]
    return ", ".join(values) + "\n"


def captured(uuid: str = "GPU-one") -> dict[str, object]:
    return {
        "gpus": TELEMETRY.parse_gpu_inventory(gpu_row(uuid)),
        "compute_applications": [
            {
                "gpu_uuid": uuid,
                "pid": 43,
                "process_name": "fixture",
                "used_gpu_memory_mib": {
                    "supported": True,
                    "value": 100.0,
                    "raw": "100",
                },
            }
        ],
    }


class GpuTelemetryTests(unittest.TestCase):
    def test_inventory_preserves_unsupported_and_zero_supported_metrics(self) -> None:
        row = TELEMETRY.parse_gpu_inventory(gpu_row())[0]
        self.assertEqual(row["uuid"], "GPU-one")
        self.assertEqual(
            row["ecc_corrected_volatile_total"],
            {"supported": False, "value": None, "raw": "[N/A]"},
        )
        self.assertEqual(
            row["remapped_rows_correctable"],
            {"supported": True, "value": 0, "raw": "0"},
        )
        self.assertEqual(
            row["remapped_rows_pending"],
            {"supported": True, "value": False, "raw": "No"},
        )
        self.assertEqual(
            row["clock_event_sw_thermal_slowdown"],
            {"supported": True, "value": False, "raw": "Not Active"},
        )

    def test_required_clock_event_reason_cannot_be_unsupported(self) -> None:
        values = gpu_row().strip().split(", ")
        values[14] = "[N/A]"
        with self.assertRaisesRegex(ValueError, "required"):
            TELEMETRY.parse_gpu_inventory(", ".join(values) + "\n")

    def test_required_metric_cannot_be_unsupported(self) -> None:
        values = gpu_row().strip().split(", ")
        values[10] = "[N/A]"
        with self.assertRaisesRegex(ValueError, "required"):
            TELEMETRY.parse_gpu_inventory(", ".join(values) + "\n")

    def test_compute_apps_uses_csv_and_explicit_optional_memory(self) -> None:
        rows = TELEMETRY.parse_compute_apps(
            'GPU-one, 43, "fixture,worker", [N/A]\n'
        )
        self.assertEqual(rows[0]["process_name"], "fixture,worker")
        self.assertFalse(rows[0]["used_gpu_memory_mib"]["supported"])

    def test_capture_queries_selected_uuid_and_filters_unselected_apps(self) -> None:
        outputs = [
            gpu_row("GPU-one", 0),
            gpu_row("GPU-two", 1),
            "GPU-one, 43, worker, 10\nGPU-other, 99, other, 20\n",
        ]
        with mock.patch.object(TELEMETRY, "run_query", side_effect=outputs) as query:
            result = TELEMETRY.capture(("GPU-one", "GPU-two"))
        self.assertEqual(
            [row["uuid"] for row in result["gpus"]], ["GPU-one", "GPU-two"]
        )
        self.assertEqual(len(result["compute_applications"]), 1)
        self.assertIn("--id=GPU-one", query.call_args_list[0].args[0])
        self.assertIn("--id=GPU-two", query.call_args_list[1].args[0])

    def test_main_writes_root_bound_stream_and_terminal(self) -> None:
        identity = {"state": "R", "start_time_ticks": 100}
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw) / "gpu.jsonl"
            argv = [
                "--root-pid",
                "42",
                "--output",
                str(output),
                "--label",
                "fixture",
                "--selected-uuid",
                "GPU-one",
                "--interval-seconds",
                "1",
            ]
            with mock.patch.object(
                TELEMETRY,
                "proc_identity",
                side_effect=[identity, identity, None],
            ), mock.patch.object(
                TELEMETRY, "capture", return_value=captured()
            ), mock.patch.object(TELEMETRY.time, "sleep", return_value=None):
                self.assertEqual(TELEMETRY.main(argv), 0)
            samples = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([sample["sequence"] for sample in samples], [0, 1])
            self.assertEqual(
                [sample["root_alive"] for sample in samples], [True, False]
            )
            self.assertEqual(samples[0]["root_start_time_ticks"], 100)
            terminal = json.loads(
                output.with_name(output.name + ".COMPLETE.json").read_text()
            )
            self.assertEqual(terminal["outcome"], "PASS")
            self.assertEqual(terminal["query_error_count"], 0)
            self.assertEqual(
                terminal["telemetry"]["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                terminal["monitor_source_before"],
                terminal["monitor_source_after"],
            )
            self.assertEqual(
                terminal["nvidia_smi_before"], terminal["nvidia_smi_after"]
            )

    def test_query_error_is_retained_and_terminal_fails(self) -> None:
        identity = {"state": "R", "start_time_ticks": 100}
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw) / "gpu.jsonl"
            with mock.patch.object(
                TELEMETRY, "proc_identity", side_effect=[identity, None]
            ), mock.patch.object(
                TELEMETRY, "capture", side_effect=RuntimeError("fixture failure")
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
                            "--selected-uuid",
                            "GPU-one",
                            "--interval-seconds",
                            "1",
                        ]
                    ),
                    3,
                )
            sample = json.loads(output.read_text().strip())
            self.assertIn("fixture failure", sample["query_error"])
            terminal = json.loads(
                output.with_name(output.name + ".COMPLETE.json").read_text()
            )
            self.assertEqual(terminal["outcome"], "FAIL")
            self.assertEqual(terminal["query_error_count"], 1)

    def test_cli_rejects_duplicate_or_noncanonical_uuid(self) -> None:
        base = [
            "--root-pid",
            "42",
            "--output",
            "/tmp/unused-gpu-telemetry",
            "--label",
            "fixture",
        ]
        with self.assertRaisesRegex(ValueError, "one or two"):
            TELEMETRY.main(
                base
                + [
                    "--selected-uuid",
                    "GPU-one",
                    "--selected-uuid",
                    "GPU-one",
                ]
            )
        with self.assertRaisesRegex(ValueError, "one or two"):
            TELEMETRY.main(base + ["--selected-uuid", "0"])

    def test_dangling_output_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw) / "gpu.jsonl"
            output.symlink_to(pathlib.Path(raw) / "missing-target")
            with self.assertRaisesRegex(FileExistsError, "symlink"):
                TELEMETRY.new_output_paths(output)


if __name__ == "__main__":
    unittest.main()
