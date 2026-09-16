#!/usr/bin/env python3
"""Contract tests for the resumable M3 paired-unit controller."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "run_m3_paired_matrix.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_paired", SOURCE)
assert SPEC is not None and SPEC.loader is not None
PAIRED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PAIRED
SPEC.loader.exec_module(PAIRED)
GPU = "GPU-11111111-1111-1111-1111-111111111111"


def fake_plan():
    selected = []
    for index in range(1, 57):
        selected.append(
            {
                "launch_order": index,
                "unit_id": f"selected-{index:02d}",
                "section": "cluster-a",
                "case_path": f"python/examples/case-{index:02d}.py",
                "purpose": f"purpose {index}",
                "timeout_seconds": None,
                "release_role": (
                    "fail_fast_canary" if index <= 10 else "full_matrix"
                ),
                "covered_example": None,
            }
        )
    stronger = []
    for index in range(1, 15):
        stronger.append(
            {
                "launch_order": index,
                "unit_id": f"stronger-{index:02d}",
                "section": "stronger-tests",
                "case_path": f"python/tests/test_case_{index:02d}.py",
                "purpose": f"oracle {index}",
                "timeout_seconds": 1800,
                "release_role": "stronger_test",
                "covered_example": f"python/examples/covered-{index:02d}.py",
            }
        )
    return {
        "schema": PAIRED.m3_execution_plan.SCHEMA,
        "selected": selected,
        "stronger": stronger,
    }


def fake_contract():
    return {
        "paths": {
            "python": "/prefix/bin/python",
            "runner": "/repo/scripts/python-validation/run_validation.py",
            "repo": "/repo",
            "manifest": "/repo/scripts/python-validation/manifest.json",
            "build_python": "/repo/build/meep/python",
            "install_prefix": "/repo/install/meep",
        },
        "build_receipt": {"receipt_id": "receipt"},
        "gpu_device": GPU,
        "limits": {"telemetry_interval_seconds": 1.0},
    }


def passing_report(task):
    return {
        "exit_code": 0,
        "configuration": {
            "case_ids": [task.case_path],
            "backends": ["cpu", "cuda"],
            "tiers": ["full"],
            "performance_evidence": {
                "valid_for_speed_gate": False,
                "concurrency_detected": False,
            },
        },
        "results": [
            {
                "selected": True,
                "path": task.case_path,
                "outcome": "PASS",
                "runs": {
                    "cpu": {"outcome": "PASS"},
                    "cuda": {"outcome": "PASS"},
                },
                "comparison": {"outcome": "PASS"},
            }
        ],
    }


class M3PairedMatrixTests(unittest.TestCase):
    def test_schedule_is_exact_56_plus_14(self) -> None:
        tasks = PAIRED.build_tasks(fake_plan())
        self.assertEqual(len(tasks), 70)
        self.assertEqual(tasks[0].index, 1)
        self.assertEqual(tasks[0].release_role, "fail_fast_canary")
        self.assertEqual(tasks[55].release_role, "full_matrix")
        self.assertEqual(tasks[56].index, 57)
        self.assertEqual(tasks[56].release_role, "stronger_test")
        self.assertEqual(tasks[-1].index, 70)

    def test_task_command_selects_one_case_and_both_backends(self) -> None:
        task = PAIRED.build_tasks(fake_plan())[0]
        command = PAIRED.task_command(task, fake_contract(), pathlib.Path("/evidence"))
        self.assertEqual(command.count("--case"), 1)
        self.assertEqual(command[command.index("--case") + 1], task.case_path)
        self.assertEqual(command[command.index("--backend") + 1], "both")
        self.assertNotIn("--concurrent-gpu-work", command)

    def test_gpu_and_argument_bounds_are_fail_closed(self) -> None:
        self.assertEqual(PAIRED.normalized_gpu(GPU), GPU)
        with self.assertRaises(Exception):
            PAIRED.normalized_gpu("0")
        args = PAIRED.parse_args(
            [
                "--plan-root",
                "/plan",
                "--output",
                "/output",
                "--python",
                "/python",
                "--build-python",
                "/build-python",
                "--install-prefix",
                "/prefix",
                "--build-receipt",
                "/receipt",
                "--gpu-device",
                GPU,
                "--unit-timeout-seconds",
                "3599",
            ]
        )
        with self.assertRaisesRegex(Exception, "timeout"):
            PAIRED.validate_args(args)

    def test_output_cannot_overlap_repo_or_plan_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            plan = root / "plan"
            repo.mkdir()
            plan.mkdir()
            with self.assertRaisesRegex(Exception, "overlaps"):
                PAIRED.validate_output_isolation(repo / "evidence", repo, plan)
            with self.assertRaisesRegex(Exception, "overlaps"):
                PAIRED.validate_output_isolation(plan / "evidence", repo, plan)
            PAIRED.validate_output_isolation(root / "evidence", repo, plan)

    def test_contract_replays_code_plan_receipt_paths_and_gates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            plan_root = root / "plan"
            build_root = root / "build"
            build_python = build_root / "python"
            install = root / "install"
            plan_root.mkdir()
            build_python.mkdir(parents=True)
            install.mkdir()
            receipt_path = build_root / "build-provenance.json"
            receipt_path.write_text("fixture receipt\n")
            receipt = {
                "receipt_id": "receipt",
                "build_input_id": "build-input",
                "artifact_set_id": "artifacts",
                "source_start": {"sha256": "1" * 64},
            }
            args = PAIRED.argparse.Namespace(
                plan_root=plan_root,
                python=pathlib.Path(sys.executable),
                build_python=build_python,
                install_prefix=install,
                build_receipt=receipt_path,
                unit_timeout_seconds=24 * 3600.0,
                stdout_limit_mib=32,
                telemetry_interval_seconds=1.0,
            )
            plan = fake_plan()
            tasks = PAIRED.build_tasks(plan)
            stables, code = PAIRED._stable_code_records(ROOT)
            try:
                receipt_record = PAIRED._receipt_record(receipt_path, receipt)
                nvidia = PAIRED.absolute_file_record(
                    pathlib.Path("/usr/bin/nvidia-smi"), "nvidia-smi"
                )
                contract = PAIRED.build_contract(
                    args,
                    ROOT,
                    plan,
                    tasks,
                    code,
                    receipt_record,
                    GPU,
                    nvidia,
                )
                with mock.patch.object(
                    PAIRED.m3_execution_plan,
                    "load_release_plan",
                    return_value=plan,
                ), mock.patch.object(
                    PAIRED.provenance,
                    "verify_build_receipt",
                    return_value=receipt,
                ):
                    _repo, replayed_plan, replayed_tasks, replayed_gpu = (
                        PAIRED.validate_contract(contract)
                    )
                    self.assertEqual(replayed_plan, plan)
                    self.assertEqual(len(replayed_tasks), 70)
                    self.assertEqual(replayed_gpu, GPU)

                    weakened = json.loads(json.dumps(contract))
                    weakened["gates"]["minimum_gpu_free_memory_fraction"] = 0.0
                    with self.assertRaisesRegex(Exception, "sealed gates differ"):
                        PAIRED.validate_contract(weakened)

                    extra = json.loads(json.dumps(contract))
                    extra["unsealed"] = True
                    with self.assertRaisesRegex(Exception, "schema differs"):
                        PAIRED.validate_contract(extra)
            finally:
                for stable in stables.values():
                    stable.close(verify=False)

    def test_archive_record_requires_exact_selected_pass(self) -> None:
        task = PAIRED.build_tasks(fake_plan())[0]
        contract = fake_contract()
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            archive = output.joinpath(*PAIRED.task_archive_relative(task).parts)
            runner = archive.joinpath(*PAIRED.archive_validation.RUNNER_ROOT.parts)
            runner.mkdir(parents=True)
            (archive / PAIRED.archive_validation.COMPLETE_NAME).write_text("complete\n")
            (archive / PAIRED.archive_validation.MANIFEST_NAME).write_text("manifest\n")
            (runner / "report.json").write_text(json.dumps(passing_report(task)))
            complete = {
                "archive_manifest_sha256": "a" * 64,
                "report_sha256": "b" * 64,
                "build_receipt_id": "receipt",
            }
            with mock.patch.object(
                PAIRED.archive_validation, "verify_archive", return_value=complete
            ) as verifier:
                record = PAIRED.derive_archive_record(task, output, contract)
            verifier.assert_called_once_with(archive)
            self.assertEqual(record["receipt_id"], "receipt")

            failed = passing_report(task)
            failed["results"][0]["runs"]["cuda"]["outcome"] = "FAILED"
            (runner / "report.json").write_text(json.dumps(failed))
            with mock.patch.object(
                PAIRED.archive_validation, "verify_archive", return_value=complete
            ), self.assertRaisesRegex(Exception, "not an exact PASS"):
                PAIRED.derive_archive_record(task, output, contract)

    def test_terminal_is_verified_before_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            for name in ("report.json", "JOURNAL.json", "CHECKPOINT.json"):
                (output / name).write_text(f"fixture {name}\n")
            report = {
                "checkpoint": PAIRED.matrix.stable_file_record(
                    output / "CHECKPOINT.json", output, "checkpoint"
                )
            }
            observations = []

            def verify(_output, terminal_name="PAIRED_COMPLETE"):
                observations.append(
                    (
                        terminal_name,
                        (_output / "PENDING_PAIRED_COMPLETE").is_file(),
                        (_output / "PAIRED_COMPLETE").exists(),
                    )
                )
                return {"outcome": "PASS"}

            with mock.patch.object(PAIRED, "verify_complete", side_effect=verify):
                PAIRED.publish_terminal(output, report)
            self.assertEqual(
                observations, [("PENDING_PAIRED_COMPLETE", True, False)]
            )
            self.assertTrue((output / "PAIRED_COMPLETE").is_file())
            self.assertFalse((output / "PENDING_PAIRED_COMPLETE").exists())

    def test_journal_prefix_rejects_mutation(self) -> None:
        contract = {"fixture": True}
        record = {
            "task": {"index": 1},
            "process": {"returncode": 0},
            "archive": {"archive_complete": {"sha256": "a" * 64}},
        }
        events = [
            {"state": "controller-started", "unix_seconds": 1.0},
            {"state": "task-started", "task": record["task"]},
            {
                "state": "task-process-ended",
                "task": record["task"],
                "process": record["process"],
            },
            {
                "state": "task-validated",
                "task": record["task"],
                "archive_complete": record["archive"]["archive_complete"],
            },
        ]
        journal = {
            "schema": PAIRED.JOURNAL_SCHEMA,
            "contract": contract,
            "events": events,
        }
        self.assertEqual(PAIRED.expected_journal(journal, contract, [record]), events)
        journal["events"][-1]["state"] = "task-skipped"
        with self.assertRaisesRegex(Exception, "disagrees"):
            PAIRED.expected_journal(journal, contract, [record])

    def test_main_executes_all_70_units_before_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            plan_root = root / "plan"
            output = root / "evidence"
            build_root = root / "build"
            build_python = build_root / "python"
            install = root / "install"
            plan_root.mkdir()
            build_python.mkdir(parents=True)
            install.mkdir()
            receipt_path = build_root / "build-provenance.json"
            receipt_path.write_text("fixture receipt\n")
            plan = fake_plan()
            tasks = PAIRED.build_tasks(plan)
            receipt = {
                "receipt_id": "receipt",
                "build_input_id": "build-input",
                "artifact_set_id": "artifacts",
                "source_start": {"sha256": "1" * 64},
            }
            hardware = {"fixture": "hardware"}
            process_count = 0
            seal_count = 0

            def capacity(path):
                return {
                    "path": str(path.resolve()),
                    "free_bytes": 200 * 1024**3,
                    "total_bytes": 400 * 1024**3,
                    "free_inodes": 500_000,
                    "total_inodes": 1_000_000,
                }

            def run_observed(command, _environment, _root, log, *_args, **_kwargs):
                nonlocal process_count
                process_count += 1
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("fixture process\n")
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

            def seal(*_args, **_kwargs):
                nonlocal seal_count
                seal_count += 1
                return {"fixture": True}

            def archive_record(task, *_args, **_kwargs):
                return {
                    "path": PAIRED.task_archive_relative(task).as_posix(),
                    "archive_complete": {
                        "path": f"archive-{task.index}/ARCHIVE_COMPLETE",
                        "size_bytes": 1,
                        "sha256": f"{task.index:064x}",
                    },
                    "archive_manifest": {"fixture": task.index},
                    "runner_report": {"fixture": task.index},
                    "archive_manifest_sha256": f"{task.index:064x}",
                    "report_sha256": f"{task.index + 1:064x}",
                    "receipt_id": "receipt",
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
                str(build_python),
                "--install-prefix",
                str(install),
                "--build-receipt",
                str(receipt_path),
                "--gpu-device",
                GPU,
            ]
            with mock.patch.object(
                PAIRED.m3_execution_plan, "load_release_plan", return_value=plan
            ), mock.patch.object(
                PAIRED, "_stable_code_records", return_value=({}, {})
            ), mock.patch.object(
                PAIRED.provenance, "verify_build_receipt", return_value=receipt
            ), mock.patch.object(
                PAIRED, "validate_contract", return_value=(ROOT, plan, tasks, GPU)
            ), mock.patch.object(
                PAIRED.matrix, "hardware_snapshot", return_value=hardware
            ), mock.patch.object(
                PAIRED.matrix, "validate_hardware_evidence", return_value=None
            ), mock.patch.object(
                PAIRED, "capacity_snapshot", side_effect=capacity
            ), mock.patch.object(
                PAIRED.observed,
                "run_bounded_observed",
                side_effect=run_observed,
            ), mock.patch.object(
                PAIRED,
                "publish_telemetry",
                return_value={"path": "telemetry/replay.json"},
            ), mock.patch.object(
                PAIRED.archive_validation, "seal_archive", side_effect=seal
            ), mock.patch.object(
                PAIRED, "derive_archive_record", side_effect=archive_record
            ), mock.patch.object(
                PAIRED, "verify_complete", return_value={"outcome": "PASS"}
            ) as verifier:
                self.assertEqual(PAIRED.main(argv), 0)

            report = json.loads((output / "report.json").read_text())
            self.assertEqual(len(report["tasks"]), 70)
            self.assertEqual(process_count, 70)
            self.assertEqual(seal_count, 70)
            self.assertTrue((output / "PAIRED_COMPLETE").is_file())
            self.assertFalse((output / "PENDING_PAIRED_COMPLETE").exists())
            verifier.assert_called_once_with(output, "PENDING_PAIRED_COMPLETE")


if __name__ == "__main__":
    unittest.main()
