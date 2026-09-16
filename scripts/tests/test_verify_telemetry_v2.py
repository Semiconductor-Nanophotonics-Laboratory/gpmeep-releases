#!/usr/bin/env python3
"""Mutation-focused tests for independent telemetry v2 replay."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "user-workloads" / "verify_telemetry_v2.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_verify_telemetry_v2", SOURCE)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY
SPEC.loader.exec_module(VERIFY)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: pathlib.Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": digest(payload),
    }


def metric(value, raw: str | None = None) -> dict[str, object]:
    if value is None:
        return {"supported": False, "value": None, "raw": raw or "[N/A]"}
    if raw is None:
        if isinstance(value, bool):
            raw = "Yes" if value else "No"
        else:
            raw = str(value)
    return {"supported": True, "value": value, "raw": raw}


def process(pid: int, ppid: int, start: int) -> dict[str, object]:
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


def host_sample(sequence: int, alive: bool) -> dict[str, object]:
    processes = [process(42, 1, 100), process(43, 42, 101)] if alive else []
    return {
        "schema": VERIFY.HOST_SAMPLE_SCHEMA,
        "sequence": sequence,
        "label": "fixture",
        "captured_unix_seconds": 10.0 + 2.0 * sequence,
        "root_pid": 42,
        "root_start_time_ticks": 100,
        "root_alive": alive,
        "processes": processes,
        "process_count": len(processes),
        "process_rss_bytes": sum(item["rss_bytes"] for item in processes),
        "load_average": [1.0, 1.0, 1.0],
        "memory_bytes": {
            "MemTotal": 16 * 1024**3,
            "MemAvailable": 8 * 1024**3,
            "SwapTotal": 1024**3,
            "SwapFree": 1024**3,
        },
        "cgroup_memory": {
            "supported": False,
            "path": None,
            "memory_current": None,
            "memory_max": None,
            "memory_swap_current": None,
            "memory_swap_max": None,
        },
        "temperatures": [
            {
                "device": "coretemp",
                "label": "Package id 0",
                "millidegrees_celsius": 50_000,
            }
        ],
    }


def gpu_device(uuid: str, index: int) -> dict[str, object]:
    return {
        "index": index,
        "uuid": uuid,
        "name": "NVIDIA Fixture",
        "pstate": metric("P0"),
        "temperature_celsius": metric(40.0, "40"),
        "temperature_limit_celsius": metric(None),
        "power_draw_watts": metric(100.0, "100"),
        "power_limit_watts": metric(450.0, "450"),
        "sm_clock_mhz": metric(1800.0, "1800"),
        "memory_clock_mhz": metric(10000.0, "10000"),
        "memory_used_mib": metric(1.0, "1"),
        "memory_total_mib": metric(100.0, "100"),
        "gpu_utilization_percent": metric(100.0, "100"),
        "memory_utilization_percent": metric(1.0, "1"),
        "clock_event_sw_thermal_slowdown": metric(False, "Not Active"),
        "clock_event_hw_thermal_slowdown": metric(False, "Not Active"),
        "clock_event_hw_power_brake_slowdown": metric(False, "Not Active"),
        "clock_event_sw_power_cap": metric(False, "Not Active"),
        "ecc_corrected_volatile_total": metric(None),
        "ecc_uncorrected_volatile_total": metric(None),
        "ecc_corrected_aggregate_total": metric(None),
        "ecc_uncorrected_aggregate_total": metric(None),
        "retired_pages_single_bit_count": metric(None),
        "retired_pages_double_bit_count": metric(None),
        "retired_pages_pending": metric(None),
        "remapped_rows_correctable": metric(0),
        "remapped_rows_uncorrectable": metric(0),
        "remapped_rows_pending": metric(False),
        "remapped_rows_failure": metric(False),
    }


def gpu_sample(
    sequence: int, alive: bool, uuids: tuple[str, ...]
) -> dict[str, object]:
    apps = (
        [
            {
                "gpu_uuid": uuid,
                "pid": 43,
                "process_name": "fixture-worker",
                "used_gpu_memory_mib": metric(1.0, "1"),
            }
            for uuid in uuids
        ]
        if alive
        else []
    )
    return {
        "schema": VERIFY.GPU_SAMPLE_SCHEMA,
        "sequence": sequence,
        "label": "fixture",
        "captured_unix_seconds": 10.5 + 2.0 * sequence,
        "captured_monotonic_seconds": 20.5 + 2.0 * sequence,
        "root_pid": 42,
        "root_start_time_ticks": 100,
        "root_alive": alive,
        "selected_gpu_uuids": list(uuids),
        "gpus": [gpu_device(uuid, index) for index, uuid in enumerate(uuids)],
        "compute_applications": apps,
        "query_error": None,
    }


class Fixture:
    def __init__(self, root: pathlib.Path, uuids: tuple[str, ...] = ("GPU-one",)):
        self.root = root
        self.uuids = uuids
        self.host_stream = root / "host.jsonl"
        self.host_terminal = root / "host.COMPLETE.json"
        self.gpu_stream = root / "gpu.jsonl"
        self.gpu_terminal = root / "gpu.COMPLETE.json"
        self.host_monitor = root / "host-monitor.py"
        self.gpu_monitor = root / "gpu-monitor.py"
        self.nvidia_smi = root / "nvidia-smi"
        self.host_monitor.write_bytes(b"host monitor\n")
        self.gpu_monitor.write_bytes(b"gpu monitor\n")
        self.nvidia_smi.write_bytes(b"nvidia-smi\n")
        self.host_records = [host_sample(0, True), host_sample(1, False)]
        self.gpu_records = [
            gpu_sample(0, True, uuids),
            gpu_sample(1, False, uuids),
        ]
        self.refresh()

    @staticmethod
    def write_jsonl(path: pathlib.Path, values: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(
                json.dumps(value, allow_nan=False, sort_keys=True) + "\n"
                for value in values
            ),
            encoding="utf-8",
        )

    def refresh(self) -> None:
        self.write_jsonl(self.host_stream, self.host_records)
        self.write_jsonl(self.gpu_stream, self.gpu_records)
        host_source = file_record(self.host_monitor)
        gpu_source = file_record(self.gpu_monitor)
        nvidia = file_record(self.nvidia_smi)
        self.host_terminal.write_text(
            json.dumps(
                {
                    "schema": VERIFY.HOST_COMPLETE_SCHEMA,
                    "outcome": "PASS",
                    "terminal_error": None,
                    "label": "fixture",
                    "root_pid": 42,
                    "root_start_time_ticks": 100,
                    "interval_seconds": 2.0,
                    "sample_count": len(self.host_records),
                    "started_unix_seconds": 9.9,
                    "ended_unix_seconds": 12.1,
                    "telemetry": file_record(self.host_stream),
                    "monitor_source_before": host_source,
                    "monitor_source_after": copy.deepcopy(host_source),
                },
                sort_keys=True,
            )
            + "\n"
        )
        self.gpu_terminal.write_text(
            json.dumps(
                {
                    "schema": VERIFY.GPU_COMPLETE_SCHEMA,
                    "outcome": "PASS",
                    "terminal_error": None,
                    "label": "fixture",
                    "root_pid": 42,
                    "root_start_time_ticks": 100,
                    "selected_gpu_uuids": list(self.uuids),
                    "interval_seconds": 2.0,
                    "sample_count": len(self.gpu_records),
                    "query_error_count": 0,
                    "started_unix_seconds": 10.4,
                    "ended_unix_seconds": 12.6,
                    "started_monotonic_seconds": 20.4,
                    "ended_monotonic_seconds": 22.6,
                    "telemetry": file_record(self.gpu_stream),
                    "monitor_source_before": gpu_source,
                    "monitor_source_after": copy.deepcopy(gpu_source),
                    "nvidia_smi_before": nvidia,
                    "nvidia_smi_after": copy.deepcopy(nvidia),
                },
                sort_keys=True,
            )
            + "\n"
        )

    def verify(self, **overrides):
        values = {
            "host_stream": self.host_stream,
            "host_terminal": self.host_terminal,
            "gpu_stream": self.gpu_stream,
            "gpu_terminal": self.gpu_terminal,
            "expected_root_pid": 42,
            "expected_label": "fixture",
            "expected_host_monitor_sha256": digest(self.host_monitor.read_bytes()),
            "expected_gpu_uuids": self.uuids,
            "expected_gpu_monitor_sha256": digest(self.gpu_monitor.read_bytes()),
            "expected_nvidia_smi_sha256": digest(self.nvidia_smi.read_bytes()),
            "expected_interval_seconds": 2.0,
            "expected_process_started_unix_seconds": 9.5,
            "expected_process_ended_unix_seconds": 12.0,
        }
        values.update(overrides)
        return VERIFY.verify_pair(**values)


class TelemetryReplayTests(unittest.TestCase):
    def with_fixture(self, uuids=("GPU-one",)):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Fixture(pathlib.Path(temporary.name), uuids)

    def test_valid_pair_passes_and_derives_extrema(self) -> None:
        fixture = self.with_fixture()
        report = fixture.verify()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["host"]["maximum_process_rss_bytes"], 85)
        self.assertEqual(
            report["gpu"]["per_gpu_extrema"]["GPU-one"][
                "minimum_free_memory_fraction"
            ],
            0.99,
        )

    def test_host_rss_mutation_is_rejected(self) -> None:
        fixture = self.with_fixture()
        fixture.host_records[0]["process_rss_bytes"] -= 1
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "RSS sum"):
            fixture.verify()

    def test_host_process_inventory_must_be_rooted_at_task(self) -> None:
        fixture = self.with_fixture()
        outsider = process(44, 1, 102)
        fixture.host_records[0]["processes"].append(outsider)
        fixture.host_records[0]["process_count"] += 1
        fixture.host_records[0]["process_rss_bytes"] += outsider["rss_bytes"]
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "not rooted"):
            fixture.verify()

    def test_truncated_final_stream_is_rejected(self) -> None:
        fixture = self.with_fixture()
        fixture.host_stream.write_bytes(fixture.host_stream.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "final newline"):
            fixture.verify()

    def test_sequence_gap_is_rejected(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[1]["sequence"] = 2
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            fixture.verify()

    def test_terminal_hash_mutation_is_rejected(self) -> None:
        fixture = self.with_fixture()
        terminal = json.loads(fixture.gpu_terminal.read_text())
        terminal["telemetry"]["sha256"] = "0" * 64
        fixture.gpu_terminal.write_text(json.dumps(terminal) + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            fixture.verify()

    def test_terminal_time_bounds_are_replayed(self) -> None:
        fixture = self.with_fixture()
        terminal = json.loads(fixture.host_terminal.read_text())
        terminal["ended_unix_seconds"] = 11.0
        fixture.host_terminal.write_text(json.dumps(terminal) + "\n")
        with self.assertRaisesRegex(ValueError, "time bounds"):
            fixture.verify()

    def test_external_monitor_and_executable_seals_are_required(self) -> None:
        fixture = self.with_fixture()
        with self.assertRaisesRegex(ValueError, "externally sealed"):
            fixture.verify(expected_host_monitor_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "externally sealed"):
            fixture.verify(expected_nvidia_smi_sha256="0" * 64)

    def test_two_gpu_reordering_is_rejected(self) -> None:
        fixture = self.with_fixture(("GPU-one", "GPU-two"))
        fixture.gpu_records[0]["gpus"].reverse()
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "reordered"):
            fixture.verify()

    def test_external_uuid_order_need_not_equal_device_index_order(self) -> None:
        fixture = self.with_fixture(("GPU-one", "GPU-two"))
        for record in fixture.gpu_records:
            record["gpus"][0]["index"] = 1
            record["gpus"][1]["index"] = 0
        fixture.refresh()
        self.assertEqual(fixture.verify()["status"], "PASS")

    def test_gpu_headroom_boundary_is_rejected(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[0]["gpus"][0]["memory_used_mib"] = metric(81.0, "81")
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "free-memory"):
            fixture.verify()

    def test_reliability_counter_change_is_rejected(self) -> None:
        fixture = self.with_fixture()
        for record in fixture.gpu_records:
            record["gpus"][0]["remapped_rows_correctable"] = metric(0)
        fixture.gpu_records[1]["gpus"][0]["remapped_rows_correctable"] = metric(1)
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "changed"):
            fixture.verify()

    def test_unsupported_metric_cannot_hide_zero(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[0]["gpus"][0]["ecc_corrected_volatile_total"] = {
            "supported": False,
            "value": 0,
            "raw": "[N/A]",
        }
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            fixture.verify()

    def test_retained_metric_must_match_raw_value(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[0]["gpus"][0]["memory_used_mib"] = {
            "supported": True,
            "value": 1.0,
            "raw": "2",
        }
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "differs from raw"):
            fixture.verify()

    def test_active_reliability_fault_is_rejected(self) -> None:
        fixture = self.with_fixture()
        for record in fixture.gpu_records:
            record["gpus"][0]["remapped_rows_failure"] = metric(True)
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "active fault"):
            fixture.verify()

    def test_missing_workload_application_is_rejected(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[0]["compute_applications"] = []
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "never observed"):
            fixture.verify()

    def test_external_or_idle_application_is_rejected(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[0]["compute_applications"][0]["pid"] = 99
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "outside"):
            fixture.verify()
        fixture = self.with_fixture()
        fixture.gpu_records[1]["compute_applications"] = copy.deepcopy(
            fixture.gpu_records[0]["compute_applications"]
        )
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "idle boundary"):
            fixture.verify()

    def test_root_start_identity_mismatch_is_rejected(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[0]["root_start_time_ticks"] = 999
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "process identity"):
            fixture.verify()

    def test_temperature_gate_is_rejected(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[0]["gpus"][0]["temperature_celsius"] = metric(
            91.0, "91"
        )
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "temperature"):
            fixture.verify()

    def test_clock_event_reason_must_be_supported_and_present(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[0]["gpus"][0][
            "clock_event_sw_thermal_slowdown"
        ] = metric(None)
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            fixture.verify()
        fixture = self.with_fixture()
        del fixture.gpu_records[0]["gpus"][0][
            "clock_event_sw_thermal_slowdown"
        ]
        fixture.refresh()
        with self.assertRaisesRegex(ValueError, "metric representation"):
            fixture.verify()

    def test_active_clock_event_is_disclosed_for_exact_and_rejected_for_performance(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[0]["gpus"][0][
            "clock_event_sw_thermal_slowdown"
        ] = metric(True, "Active")
        fixture.refresh()
        disclosed = fixture.verify(reject_active_clock_event_reasons=False)
        reason = disclosed["gpu"]["clock_event_reasons"]["GPU-one"][
            "clock_event_sw_thermal_slowdown"
        ]
        self.assertEqual(reason["active_alive_sample_count"], 1)
        self.assertEqual(
            disclosed["gpu"]["clock_event_policy"],
            "disclose-active-reasons",
        )
        with self.assertRaisesRegex(ValueError, "release performance"):
            fixture.verify(reject_active_clock_event_reasons=True)

    def test_clock_event_performance_policy_requires_sealed_coverage(self) -> None:
        fixture = self.with_fixture()
        with self.assertRaisesRegex(ValueError, "requires sealed coverage"):
            fixture.verify(
                reject_active_clock_event_reasons=True,
                expected_process_started_unix_seconds=None,
                expected_process_ended_unix_seconds=None,
            )

    def test_gpu_cadence_and_process_coverage_gaps_are_rejected(self) -> None:
        fixture = self.with_fixture()
        fixture.gpu_records[1]["captured_monotonic_seconds"] = 30.0
        fixture.refresh()
        terminal = json.loads(fixture.gpu_terminal.read_text())
        terminal["ended_monotonic_seconds"] = 30.1
        fixture.gpu_terminal.write_text(json.dumps(terminal) + "\n")
        with self.assertRaisesRegex(ValueError, "cadence"):
            fixture.verify()

        fixture = self.with_fixture()
        with self.assertRaisesRegex(ValueError, "start-coverage"):
            fixture.verify(expected_process_started_unix_seconds=0.0)

        fixture = self.with_fixture()
        with self.assertRaisesRegex(ValueError, "end-coverage"):
            fixture.verify(expected_process_ended_unix_seconds=20.0)

    def test_capture_interval_must_overlap(self) -> None:
        fixture = self.with_fixture()
        for record in fixture.gpu_records:
            record["captured_unix_seconds"] += 100.0
        fixture.refresh()
        terminal = json.loads(fixture.gpu_terminal.read_text())
        terminal["started_unix_seconds"] += 100.0
        terminal["ended_unix_seconds"] += 100.0
        fixture.gpu_terminal.write_text(json.dumps(terminal) + "\n")
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            fixture.verify(
                expected_process_started_unix_seconds=None,
                expected_process_ended_unix_seconds=None,
            )

    def test_cpu_only_host_replay_is_supported(self) -> None:
        fixture = self.with_fixture()
        report = VERIFY.verify_pair(
            host_stream=fixture.host_stream,
            host_terminal=fixture.host_terminal,
            gpu_stream=None,
            gpu_terminal=None,
            expected_root_pid=42,
            expected_label="fixture",
            expected_host_monitor_sha256=digest(fixture.host_monitor.read_bytes()),
        )
        self.assertIsNone(report["gpu"])

    def test_report_publication_is_atomic_exclusive_and_symlink_safe(self) -> None:
        fixture = self.with_fixture()
        report = fixture.verify()
        output = fixture.root / "report.json"
        self.assertEqual(VERIFY.atomic_report(output, report), output)
        self.assertEqual(json.loads(output.read_text())["status"], "PASS")
        with self.assertRaises(FileExistsError):
            VERIFY.atomic_report(output, report)
        dangling = fixture.root / "dangling-report.json"
        dangling.symlink_to(fixture.root / "missing")
        with self.assertRaisesRegex(FileExistsError, "symlink"):
            VERIFY.atomic_report(dangling, report)


if __name__ == "__main__":
    unittest.main()
