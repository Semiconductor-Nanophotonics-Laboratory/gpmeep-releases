#!/usr/bin/env python3
"""Contract tests for the four-unit M3 host compatibility controller."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "run_m3_host_matrix.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_host_matrix", SOURCE)
assert SPEC is not None and SPEC.loader is not None
HOST = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HOST
SPEC.loader.exec_module(HOST)
GPU = "GPU-11111111-1111-1111-1111-111111111111"


def fake_plan():
    rows = []
    values = [
        (
            "host-eps-fit-lorentzian",
            "python-validation-exact-case",
            "python/examples/eps_fit_lorentzian.py",
            ["python/examples/eps_fit_lorentzian.py"],
            600,
        ),
        (
            "host-mpb-regression-suite",
            "python-unittest-file",
            "python/tests/test_mpb.py",
            [f"python/examples/mpb_{index}.py" for index in range(12)],
            7200,
        ),
        (
            "host-parallel-wvgs-mpb",
            "runpy-finite-array-adapter",
            "python/examples/parallel-wvgs-mpb.py",
            ["python/examples/parallel-wvgs-mpb.py"],
            3600,
        ),
        (
            "host-plot-radiation-pattern",
            "render-from-sealed-producer-metrics",
            "python/examples/plot_radiation_pattern_dipole.py",
            ["python/examples/plot_radiation_pattern_dipole.py"],
            600,
        ),
    ]
    for order, (unit, driver, target, covered, timeout) in enumerate(values, 1):
        rows.append(
            {
                "launch_order": order,
                "unit_id": unit,
                "driver": driver,
                "target_path": target,
                "covers_examples": covered,
                "oracle_contract": f"oracle {order}",
                "timeout_seconds": timeout,
            }
        )
    return {"schema": HOST.m3_execution_plan.SCHEMA, "host": rows}


def paired_fixture(root: pathlib.Path, plan_root: pathlib.Path):
    paired = root / "paired"
    archive = paired / "archives" / "042-dipole"
    archive.mkdir(parents=True)
    for name in ("PAIRED_COMPLETE", "report.json", "CHECKPOINT.json"):
        (paired / name).write_text(f"fixture {name}\n")
    contract = {
        "paths": {
            "repo": str(ROOT),
            "plan_root": str(plan_root),
            "python": str(pathlib.Path(sys.executable).resolve()),
            "build_python": str(root / "build" / "python"),
            "install_prefix": str(root / "install"),
        },
        "build_receipt": {
            "receipt_id": "receipt",
            "source_sha256": "1" * 64,
        },
        "gpu_device": GPU,
    }
    report = {
        "contract": contract,
        "tasks": [
            {
                "task": {
                    "case_path": "python/examples/dipole_in_vacuum_1D.py"
                },
                "archive": {
                    "path": "archives/042-dipole",
                    "archive_manifest_sha256": "a" * 64,
                    "receipt_id": "receipt",
                    "archive_complete": {
                        "path": "archives/042-dipole/ARCHIVE_COMPLETE",
                        "size_bytes": 1,
                        "sha256": "b" * 64,
                    },
                },
            }
        ],
    }
    return paired, report


class M3HostMatrixTests(unittest.TestCase):
    def test_schedule_is_four_cpu_only_non_cuda_units(self) -> None:
        tasks = HOST.build_tasks(fake_plan())
        self.assertEqual(len(tasks), 4)
        self.assertEqual([task.index for task in tasks], [1, 2, 3, 4])
        self.assertTrue(
            all(HOST.task_spec(task)["backend"] == "cpu" for task in tasks)
        )
        self.assertTrue(
            all(
                HOST.task_spec(task)["counts_toward_cuda_coverage"] is False
                for task in tasks
            )
        )

    def test_plot_command_is_bound_to_paired_producer(self) -> None:
        tasks = HOST.build_tasks(fake_plan())
        contract = {
            "python": "/python",
            "repo": "/repo",
            "producer": {
                "path": "/paired/archive",
                "archive_manifest_sha256": "a" * 64,
                "receipt_id": "receipt",
            },
        }
        regular = HOST.task_command(tasks[0], contract, pathlib.Path("/output"))
        plot = HOST.task_command(tasks[-1], contract, pathlib.Path("/output"))
        self.assertNotIn("--producer-archive", regular)
        self.assertEqual(
            plot[plot.index("--producer-archive") + 1], "/paired/archive"
        )
        self.assertEqual(
            plot[plot.index("--expected-receipt-id") + 1], "receipt"
        )

    def test_producer_identity_rejects_archive_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            plan = root / "plan"
            plan.mkdir()
            paired, report = paired_fixture(root, plan)
            identity = HOST.producer_identity(paired, report)
            self.assertEqual(identity["archive_manifest_sha256"], "a" * 64)
            escaped = root / "escape"
            escaped.mkdir()
            report["tasks"][0]["archive"]["path"] = "../escape"
            with self.assertRaisesRegex(Exception, "path is unsafe"):
                HOST.producer_identity(paired, report)

    def test_terminal_is_replayed_before_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            for name in ("report.json", "JOURNAL.json", "CHECKPOINT.json"):
                (output / name).write_text(f"fixture {name}\n")
            report = {
                "checkpoint": HOST.matrix.stable_file_record(
                    output / "CHECKPOINT.json", output, "checkpoint"
                )
            }
            observations = []

            def verify(_output, terminal_name="HOST_COMPLETE"):
                observations.append(
                    (
                        terminal_name,
                        (_output / "PENDING_HOST_COMPLETE").is_file(),
                        (_output / "HOST_COMPLETE").exists(),
                    )
                )
                return {"outcome": "PASS"}

            with mock.patch.object(HOST, "verify_complete", side_effect=verify):
                HOST.publish_terminal(output, report)
            self.assertEqual(observations, [("PENDING_HOST_COMPLETE", True, False)])
            self.assertTrue((output / "HOST_COMPLETE").is_file())

    def test_main_executes_all_four_units_before_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            plan_root = root / "plan"
            plan_root.mkdir()
            (root / "build" / "python").mkdir(parents=True)
            (root / "install").mkdir()
            paired, paired_report = paired_fixture(root, plan_root)
            output = root / "host-evidence"
            plan = fake_plan()
            tasks = HOST.build_tasks(plan)
            runtime = {
                "libmeep": {"path": str(root / "lib" / "libmeep.so")},
                "installed_environment": str(root / "installed"),
                "fontconfig_file": {"path": str(root / "fonts.conf")},
            }
            snapshot = {"available": True, "runtime_contract": runtime}
            hardware = {"fixture": "hardware"}
            process_count = 0

            def capacity(path):
                return {
                    "path": str(path.resolve()),
                    "free_bytes": 100 * 1024**3,
                    "total_bytes": 200 * 1024**3,
                    "free_inodes": 500_000,
                    "total_inodes": 1_000_000,
                }

            def run_observed(command, _environment, _root, log, *_args, **_kwargs):
                nonlocal process_count
                process_count += 1
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("fixture\n")
                return (
                    {
                        "command": command,
                        "command_pid": 1000 + process_count,
                        "returncode": 0,
                        "timed_out": False,
                        "output_limited": False,
                        "output_bytes": 0,
                        "peak_process_group_rss_bytes": 1,
                        "started_unix_seconds": 1.0,
                        "ended_unix_seconds": 2.0,
                        "wall_seconds": 1.0,
                        "log": {"fixture": True},
                    },
                    [],
                )

            def unit_output(task, *_args, **_kwargs):
                return {
                    "path": HOST.task_output_relative(task).as_posix(),
                    "complete": {
                        "path": f"unit-{task.index}/COMPLETE",
                        "size_bytes": 1,
                        "sha256": f"{task.index:064x}",
                    },
                    "report": {"fixture": task.index},
                }

            argv = [
                "--paired-output",
                str(paired),
                "--output",
                str(output),
            ]
            with mock.patch.object(
                HOST.paired_matrix, "verify_complete", return_value=paired_report
            ), mock.patch.object(
                HOST.m3_execution_plan, "load_release_plan", return_value=plan
            ), mock.patch.object(
                HOST.run_validation, "validation_snapshot", return_value=snapshot
            ), mock.patch.object(
                HOST, "_code_records", return_value=({}, {})
            ), mock.patch.object(
                HOST,
                "validate_contract",
                return_value=(ROOT, plan, tasks, GPU, paired_report),
            ), mock.patch.object(
                HOST.matrix, "hardware_snapshot", return_value=hardware
            ), mock.patch.object(
                HOST.matrix, "validate_hardware_evidence", return_value=None
            ), mock.patch.object(
                HOST, "capacity_snapshot", side_effect=capacity
            ), mock.patch.object(
                HOST.observed,
                "run_bounded_observed",
                side_effect=run_observed,
            ), mock.patch.object(
                HOST,
                "publish_telemetry",
                return_value={"path": "telemetry/replay.json"},
            ), mock.patch.object(
                HOST, "derive_unit_output", side_effect=unit_output
            ), mock.patch.object(
                HOST, "verify_complete", return_value={"outcome": "PASS"}
            ) as verifier:
                self.assertEqual(HOST.main(argv), 0)

            report = json.loads((output / "report.json").read_text())
            self.assertEqual(len(report["tasks"]), 4)
            self.assertEqual(process_count, 4)
            self.assertTrue((output / "HOST_COMPLETE").is_file())
            verifier.assert_called_once_with(output, "PENDING_HOST_COMPLETE")


if __name__ == "__main__":
    unittest.main()
