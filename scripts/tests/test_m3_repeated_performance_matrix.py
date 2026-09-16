#!/usr/bin/env python3
"""Focused tests for the resumable M3 repeated-performance controller."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "scripts"
    / "python-validation"
    / "run_m3_repeated_performance_matrix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_m3_repeated_performance_matrix", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


GPU0 = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GPU1 = "GPU-11111111-2222-3333-4444-555555555555"


def release_plan() -> dict[str, object]:
    return {
        "schema": MATRIX.performance_plan.SCHEMA,
        "units": [
            {
                "unit_id": "performance-edge-emitter-3d",
                "driver": "repeated-mpi-example",
                "target": "python/examples/edge_emitter_3D.py",
                "topologies": ["cpu8", "cuda1", "cuda2"],
                "warmup_cycles": 1,
                "measured_cycles": 5,
                "timeout_seconds": 1000,
                "timing_contract": "sum-all-simulation-run-wall-seconds",
            },
            {
                "unit_id": "performance-metasurface-crossover",
                "driver": "repeated-mpi-example-auto",
                "target": "python/examples/metasurface_lens.py",
                "topologies": ["cpu8", "cpu1", "cuda1", "cuda2", "auto1"],
                "warmup_cycles": 1,
                "measured_cycles": 5,
                "timeout_seconds": 1000,
                "timing_contract": (
                    "run-index-18-small-and-19-large-wall-seconds"
                ),
            },
            {
                "unit_id": "performance-long-horizon-policy",
                "driver": "repeated-mpi-example-auto",
                "target": "python/examples/stochastic_emitter_line.py",
                "topologies": ["cpu8", "cpu1", "cuda1", "auto1"],
                "warmup_cycles": 1,
                "measured_cycles": 5,
                "timeout_seconds": 1000,
                "timing_contract": "sum-all-simulation-run-wall-seconds",
            },
        ],
    }


def primary_seconds(unit_id: str, topology: str) -> float:
    values = {
        "performance-edge-emitter-3d": {
            "cpu8": 10.0,
            "cuda1": 5.0,
            "cuda2": 4.0,
        },
        "performance-metasurface-crossover": {
            "cpu8": 10.0,
            "cpu1": 12.0,
            "cuda1": 5.0,
            "cuda2": 4.0,
            "auto1": 5.5,
        },
        "performance-long-horizon-policy": {
            "cpu8": 8.0,
            "cpu1": 10.0,
            "cuda1": 4.0,
            "auto1": 10.5,
        },
    }
    return values[unit_id][topology]


class M3RepeatedPerformanceMatrixTests(unittest.TestCase):
    def test_task_command_binds_the_exact_sample(self) -> None:
        task = MATRIX.performance_replay.build_tasks(release_plan())[0]
        contract = {
            "paths": {
                "python": "/env/python3.11",
                "sample_runner": "/repo/sample.py",
                "repo": "/repo",
                "manifest": "/repo/manifest.json",
                "build_python": "/repo/build/python",
                "install_prefix": "/repo/install",
                "build_receipt": "/repo/receipt.json",
                "mpiexec": "/env/mpirun",
            },
            "gpu_devices": [GPU0, GPU1],
        }
        command = MATRIX.task_command(task, contract, pathlib.Path("/evidence"))
        self.assertEqual(command[:2], ["/env/python3.11", "/repo/sample.py"])
        self.assertEqual(command[command.index("--topology") + 1], "cpu8")
        self.assertEqual(command[command.index("--sample-kind") + 1], "warmup")
        self.assertEqual(command[command.index("--cycle-index") + 1], "0")
        self.assertEqual(command[command.index("--gpu-devices") + 1], f"{GPU0},{GPU1}")

    def test_clean_environment_uses_private_runtime_directories(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            runtime = pathlib.Path(raw) / "runtime"
            environment = MATRIX.clean_environment(
                {
                    "paths": {"python": "/env/bin/python3.11"},
                    "gpu_devices": [GPU0, GPU1],
                },
                runtime,
            )
            self.assertTrue((runtime / "home").is_dir())
            self.assertTrue((runtime / "cache").is_dir())
            self.assertTrue((runtime / "tmp").is_dir())
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], f"{GPU0},{GPU1}")

    def test_observer_inventory_matches_the_topology_that_owns_work(self) -> None:
        tasks = MATRIX.performance_replay.build_tasks(release_plan())
        contract = {
            "paths": {"python": "/env/bin/python3.11"},
            "gpu_devices": [GPU0, GPU1],
            "limits": {"telemetry_interval_seconds": 1.0},
        }
        selected = {
            (task["unit_id"], task["topology"]): task for task in tasks
        }
        cases = (
            ("performance-edge-emitter-3d", "cpu8", (), 1),
            ("performance-edge-emitter-3d", "cuda1", (GPU0,), 2),
            ("performance-edge-emitter-3d", "cuda2", (GPU0, GPU1), 2),
            ("performance-metasurface-crossover", "auto1", (GPU0,), 2),
            ("performance-long-horizon-policy", "auto1", (), 1),
        )
        for unit, topology, expected_devices, observer_count in cases:
            with self.subTest(unit=unit, topology=topology):
                task = selected[(unit, topology)]
                self.assertEqual(
                    MATRIX.task_gpu_devices(task, contract), expected_devices
                )
                self.assertEqual(
                    len(
                        MATRIX.observer_specs(
                            task, 1234, pathlib.Path("/evidence"), contract
                        )
                    ),
                    observer_count,
                )

    def test_full_unit_derivation_replays_72_samples_and_54_pairs(self) -> None:
        plan = release_plan()
        tasks = MATRIX.performance_replay.build_tasks(plan)
        by_name = {MATRIX.performance_replay.task_name(task): task for task in tasks}

        def sample_report(root: pathlib.Path) -> dict[str, object]:
            task = by_name[root.name]
            primary = primary_seconds(task["unit_id"], task["topology"])
            timing = {
                "contract": task["timing_contract"],
                "primary_seconds": primary,
            }
            if task["unit_id"] == "performance-metasurface-crossover":
                small = 1.05 if task["topology"] == "auto1" else 1.0
                timing.update(
                    {"small_seconds": small, "large_seconds": primary}
                )
            return {
                "outcome": "PASS",
                "case_path": task["target"],
                "topology": task["topology"],
                "sample_kind": task["sample_kind"],
                "cycle_index": task["cycle_index"],
                "timing": timing,
                "lane": {
                    "stdout": {
                        "path": "stdout.log",
                        "size_bytes": 1,
                        "sha256": "0" * 64,
                    }
                },
            }

        cases = [
            {"path": unit["target"], "comparison": {}}
            for unit in plan["units"]
        ]
        completed = [{"task": task} for task in tasks]
        comparison = {"outcome": "PASS", "failure_count": 0}
        contract = {
            "plan": plan,
            "tasks": tasks,
            "paths": {"manifest": "/manifest", "repo": "/repo"},
        }
        with mock.patch.object(
            MATRIX.performance_sample,
            "verify_complete",
            side_effect=sample_report,
        ) as verifier, mock.patch.object(
            MATRIX.validation, "load_json", return_value={}
        ), mock.patch.object(
            MATRIX.validation, "materialize_cases", return_value=cases
        ), mock.patch.object(
            MATRIX.validation,
            "compare_json_metric_files",
            return_value=comparison,
        ) as comparator:
            result = MATRIX.derive_unit_results(
                pathlib.Path("/evidence"), contract, completed
            )
        self.assertEqual(verifier.call_count, 72)
        self.assertEqual(comparator.call_count, 54)
        self.assertEqual(set(result), set(MATRIX.performance_replay.REPEATED_UNIT_IDS))
        self.assertTrue(
            all(value["statistics"]["outcome"] == "PASS" for value in result.values())
        )

    def test_journal_is_exactly_derived_from_checkpoint_records(self) -> None:
        task = MATRIX.performance_replay.build_tasks(release_plan())[0]
        process = {"returncode": 0}
        replay = {"complete": {"sha256": "0" * 64}}
        record = {"task": task, "process": process, "sample_replay": replay}
        contract = {"sealed": True}
        journal = {
            "schema": MATRIX.JOURNAL_SCHEMA,
            "contract": contract,
            "events": [
                {"state": "controller-started", "unix_seconds": 1.0},
                {"state": "task-started", "task": task},
                {"state": "task-process-ended", "task": task, "process": process},
                {"state": "task-validated", "task": task, "sample_replay": replay},
            ],
        }
        self.assertEqual(
            MATRIX.expected_journal(journal, contract, [record]), journal["events"]
        )
        journal["events"][-1]["state"] = "tampered"
        with self.assertRaisesRegex(Exception, "checkpoint"):
            MATRIX.expected_journal(journal, contract, [record])

    def test_pending_terminal_is_replayed_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            (output / "report.json").write_text("{}", encoding="utf-8")
            (output / "JOURNAL.json").write_text("{}", encoding="utf-8")
            report = {"checkpoint": {"path": "CHECKPOINT.json"}}
            with mock.patch.object(MATRIX, "verify_complete") as replay:
                MATRIX.publish_terminal(output, report)
            replay.assert_called_once_with(output, "PENDING_PERFORMANCE_COMPLETE")
            self.assertFalse((output / "PENDING_PERFORMANCE_COMPLETE").exists())
            marker = json.loads((output / "PERFORMANCE_COMPLETE").read_text())
            self.assertEqual(marker["outcome"], "PASS")


if __name__ == "__main__":
    unittest.main()
