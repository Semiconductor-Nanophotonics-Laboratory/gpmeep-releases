#!/usr/bin/env python3
"""Contract tests for the five-unit M3 specialized matrix controller."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/run_m3_specialized_matrix.py"
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_m3_specialized_matrix", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)
GPU_A = "GPU-11111111-1111-1111-1111-111111111111"
GPU_B = "GPU-22222222-2222-2222-2222-222222222222"


def fake_plan(plan_path: pathlib.Path) -> dict:
    units = []
    drivers = MATRIX.specialized_plan.DRIVER_TOPOLOGIES
    for order, (unit_id, driver, target) in enumerate(
        MATRIX.specialized_plan.EXPECTED_ROWS, 1
    ):
        units.append(
            {
                "launch_order": order,
                "unit_id": unit_id,
                "driver": driver,
                "target_path": target,
                "coverage_family": f"family-{order}",
                "topologies": drivers[driver].split(","),
                "oracle_contract": f"oracle {order}",
                "timeout_seconds": 600,
            }
        )
    return {
        "schema": MATRIX.specialized_plan.SCHEMA,
        "input": {
            "path": str(plan_path),
            "size_bytes": plan_path.stat().st_size,
            "sha256": "1" * 64,
        },
        "units": units,
        "counts": {
                "physical_units": 5,
                "python_mpi_example_units": 4,
                "specialized_qualification_units": 1,
                "cpu8_gpu1_gpu2_units": 4,
        },
    }


def fake_paths(root: pathlib.Path, plan_root: pathlib.Path) -> dict[str, str]:
    return {
        "repo": str(ROOT),
        "plan_root": str(plan_root),
        "python": str(pathlib.Path(sys.executable).resolve()),
        "build_python": str(root / "build-python"),
        "install_prefix": str(root / "install"),
        "build_receipt": str(root / "receipt.json"),
        "mpiexec": str(pathlib.Path(sys.executable).resolve()),
        "manifest": str(ROOT / "scripts/python-validation/manifest.json"),
        "mpi_example_runner": str(
            ROOT / "scripts/python-validation/run_m3_mpi_example_case.py"
        ),
        "mpi_python_runner": str(ROOT / "scripts/run-mpi-python-validation.py"),
        "near2far_runner": str(
            ROOT / "scripts/run-near2far-mpi-qualification.py"
        ),
        "adjoint_runner": str(ROOT / "scripts/run-mpi-adjoint-benchmark.py"),
        "gpu_step_db": str(pathlib.Path(sys.executable).resolve()),
    }


class M3SpecializedMatrixTests(unittest.TestCase):
    def test_schedule_and_outer_timeout_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            plan_path = pathlib.Path(raw) / "plan.tsv"
            plan_path.write_text("fixture\n")
            tasks = MATRIX.build_tasks(fake_plan(plan_path))
        self.assertEqual(len(tasks), 5)
        self.assertEqual(
            [task["launch_order"] for task in tasks], list(range(1, 6))
        )
        example = tasks[1]
        specialized = tasks[0]
        self.assertEqual(MATRIX.task_outer_timeout(example), 2100.0)
        self.assertEqual(MATRIX.task_outer_timeout(specialized), 600.0)

    def test_task_commands_cover_correctness_drivers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            plan_path = root / "plan.tsv"
            plan_path.write_text("fixture\n")
            tasks = MATRIX.build_tasks(fake_plan(plan_path))
            contract = {
                "paths": fake_paths(root, root),
                "gpu_devices": [GPU_A, GPU_B],
                "gpu_ordinals": [0, 1],
            }
            commands = [
                MATRIX.task_command(task, contract, root) for task in tasks
            ]
        self.assertEqual(
            {task["driver"] for task in tasks},
            {
                "python-mpi-example-case",
                "run-mpi-python-validation",
            },
        )
        self.assertIn(f"{GPU_A},{GPU_B}", commands[1])
        self.assertIn("0,1", commands[0])

    def test_terminal_is_replayed_before_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            for name in ("report.json", "JOURNAL.json", "CHECKPOINT.json"):
                (output / name).write_text(f"fixture {name}\n")
            report = {
                "checkpoint": MATRIX.matrix.stable_file_record(
                    output / "CHECKPOINT.json", output, "checkpoint"
                )
            }
            observations = []

            def verify(_output, terminal_name="SPECIALIZED_COMPLETE"):
                observations.append(
                    (
                        terminal_name,
                        (_output / "PENDING_SPECIALIZED_COMPLETE").is_file(),
                        (_output / "SPECIALIZED_COMPLETE").exists(),
                    )
                )
                return {"outcome": "PASS"}

            with mock.patch.object(MATRIX, "verify_complete", side_effect=verify):
                MATRIX.publish_terminal(output, report)
            self.assertEqual(
                observations, [("PENDING_SPECIALIZED_COMPLETE", True, False)]
            )
            self.assertTrue((output / "SPECIALIZED_COMPLETE").is_file())

    def test_main_executes_five_units_and_publishes_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            plan_root = root / "plan"
            plan_root.mkdir()
            plan_path = plan_root / "plan.tsv"
            plan_path.write_text("fixture\n")
            (root / "build-python").mkdir()
            (root / "install").mkdir()
            receipt_path = root / "receipt.json"
            receipt_path.write_text("{}\n")
            output = root / "evidence"
            plan = fake_plan(plan_path)
            tasks = MATRIX.build_tasks(plan)
            paths = fake_paths(root, plan_root)
            receipt = {
                "receipt_id": "receipt",
                "build_input_id": "input",
                "artifact_set_id": "artifacts",
                "source_start": {"sha256": "1" * 64},
            }
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
                    },
                    [],
                )

            def replay(task, *_args):
                return {
                    "schema": MATRIX.specialized_replay.SCHEMA,
                    "unit_id": task["unit_id"],
                    "evidence": {"fixture": task["launch_order"]},
                }

            argv = [
                "--repo",
                str(ROOT),
                "--plan-root",
                str(plan_root),
                "--output",
                str(output),
                "--python",
                sys.executable,
                "--build-python",
                str(root / "build-python"),
                "--install-prefix",
                str(root / "install"),
                "--build-receipt",
                str(receipt_path),
                "--mpiexec",
                sys.executable,
                "--gpu-devices",
                f"{GPU_A},{GPU_B}",
            ]
            with mock.patch.object(
                MATRIX.specialized_plan, "load_release_plan", return_value=plan
            ), mock.patch.object(
                MATRIX, "_runtime_paths", return_value=paths
            ), mock.patch.object(
                MATRIX, "_code_records", return_value=({}, {})
            ), mock.patch.object(
                MATRIX.provenance, "verify_build_receipt", return_value=receipt
            ), mock.patch.object(
                MATRIX,
                "validate_contract",
                return_value=(ROOT, plan, tasks, (GPU_A, GPU_B)),
            ), mock.patch.object(
                MATRIX.matrix, "hardware_snapshot", return_value=hardware
            ), mock.patch.object(
                MATRIX.matrix, "validate_hardware_evidence", return_value=None
            ), mock.patch.object(
                MATRIX, "capacity_snapshot", side_effect=capacity
            ), mock.patch.object(
                MATRIX.observed, "run_bounded_observed", side_effect=run_observed
            ), mock.patch.object(
                MATRIX,
                "publish_telemetry",
                return_value={"path": "telemetry/replay.json"},
            ), mock.patch.object(
                MATRIX, "derive_task_replay", side_effect=replay
            ), mock.patch.object(
                MATRIX, "verify_complete", return_value={"outcome": "PASS"}
            ) as verifier:
                self.assertEqual(MATRIX.main(argv), 0)

            report = json.loads((output / "report.json").read_text())
            self.assertEqual(len(report["tasks"]), 5)
            self.assertEqual(process_count, 5)
            self.assertTrue((output / "SPECIALIZED_COMPLETE").is_file())
            verifier.assert_called_once_with(
                output, "PENDING_SPECIALIZED_COMPLETE"
            )


if __name__ == "__main__":
    unittest.main()
