#!/usr/bin/env python3
"""Contract and schedule tests for the release AuNP hybrid controller."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKLOAD_DIR = ROOT / "scripts" / "user-workloads"
sys.path.insert(0, str(WORKLOAD_DIR))
SOURCE = WORKLOAD_DIR / "run_hybrid_aunp_matrix.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_hybrid_aunp", SOURCE)
assert SPEC is not None and SPEC.loader is not None
HYBRID = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HYBRID
SPEC.loader.exec_module(HYBRID)
GPU_ONE = "GPU-11111111-1111-1111-1111-111111111111"
GPU_TWO = "GPU-22222222-2222-2222-2222-222222222222"


def fake_build():
    return types.SimpleNamespace(
        python=pathlib.Path("/prefix/bin/python"),
        mpiexec=pathlib.Path("/prefix/bin/mpirun"),
        receipt_path=pathlib.Path("/receipt.json"),
        single_precision=True,
        cuda_enabled=True,
        receipt={
            "receipt_id": "receipt",
            "build_input_id": "build-input",
            "artifact_set_id": "artifacts",
        },
    )


class HybridAuNPContractTests(unittest.TestCase):
    def args(self, *extra):
        values = [
            "--archive",
            "/archive.tar.gz",
            "--output",
            "/output",
            "--fp32-python",
            "/python",
            "--fp32-mpiexec",
            "/mpirun",
            "--fp32-receipt",
            "/receipt",
            "--gpu-devices",
            "GPU-one,GPU-two",
        ]
        return HYBRID.parse_args(values + list(extra))

    def sealed_contract(self):
        build = fake_build()
        devices = (GPU_ONE, GPU_TWO)
        tasks = HYBRID.build_tasks(build, 8, devices)
        return {
            "input": {
                "path": "/archive.tar.gz",
                "size_bytes": 1,
                "sha256": HYBRID.AUNP_ARCHIVE_SHA256,
            },
            "evidence_code": {},
            "build": {
                "python": str(build.python),
                "mpiexec": str(build.mpiexec),
                "receipt_path": str(build.receipt_path),
                "receipt_id": "receipt",
                "build_input_id": "build-input",
                "artifact_set_id": "artifacts",
                "source_sha256": "source-snapshot",
            },
            "cpu_ranks": 8,
            "gpu_devices": list(devices),
            "performance_repeats": 2,
            "qualification_profile": HYBRID.aunp_qualification_profile(),
            "fixed_window_meep_time": HYBRID.performance_adapter.FIXED_WINDOW_MEEP_TIME,
            "minimum_timesteps_per_phase": (
                HYBRID.performance_adapter.MINIMUM_TIMESTEPS_PER_PHASE
            ),
            "limits": {
                "exact_timeout_seconds": 72 * 3600.0,
                "performance_timeout_seconds": 6 * 3600.0,
                "comparison_timeout_seconds": 12 * 3600.0,
                "stdout_limit_bytes": 128 * 1024**2,
                "telemetry_interval_seconds": 5.0,
            },
            "gates": HYBRID.release_gate_values(self.args()),
            "nvidia_smi": {},
            "tasks": [HYBRID.task_spec(task) for task in tasks],
            "exact_comparisons": HYBRID.exact_comparison_specs(),
        }

    def test_schedule_is_exact_and_alternating(self) -> None:
        tasks = HYBRID.build_tasks(
            fake_build(), 8, ("GPU-one", "GPU-two")
        )
        self.assertEqual(len(tasks), 10)
        self.assertEqual(
            [(task.role, task.kind) for task in tasks[:4]],
            [
                ("smoke-cuda-fp32-1g", "smoke"),
                ("smoke-cuda-fp32-2g", "smoke"),
                ("exact-cuda-fp32-1g", "exact"),
                ("exact-cuda-fp32-2g", "exact"),
            ],
        )
        self.assertEqual(
            [task.role for task in tasks[4:7]],
            [
                "performance-cpu-fp32-8r",
                "performance-cuda-fp32-1g",
                "performance-cuda-fp32-2g",
            ],
        )
        self.assertEqual(
            [task.role for task in tasks[7:10]],
            [
                "performance-cuda-fp32-2g",
                "performance-cuda-fp32-1g",
                "performance-cpu-fp32-8r",
            ],
        )
        self.assertTrue(all(task.counts_toward_performance for task in tasks[4:]))
        self.assertFalse(any(task.counts_toward_performance for task in tasks[:4]))

    def test_two_directional_exact_comparisons_are_mandatory(self) -> None:
        self.assertEqual(
            HYBRID.exact_comparison_specs(),
            [
                {
                    "name": "gpu1-reference",
                    "reference_role": "exact-cuda-fp32-1g",
                    "candidate_role": "exact-cuda-fp32-2g",
                    "candidate_class": "repeat-cuda-fp32",
                },
                {
                    "name": "gpu2-reference",
                    "reference_role": "exact-cuda-fp32-2g",
                    "candidate_role": "exact-cuda-fp32-1g",
                    "candidate_class": "repeat-cuda-fp32",
                },
            ],
        )

    def test_exact_comparison_replay_canonicalizes_integer_rank_keys(self) -> None:
        spec = HYBRID.exact_comparison_specs()[0]
        exact_paths = {
            "exact-cuda-fp32-1g": pathlib.Path("/exact-gpu1"),
            "exact-cuda-fp32-2g": pathlib.Path("/exact-gpu2"),
        }
        derived = {
            "schema": "gpmeep-aunp-r4000-comparison-v2",
            "reference_provenance": {"gpu_devices": {0: "gpu-one"}},
            "comparisons": [
                {
                    "label": "packaged-upstream-expected",
                    "outcome": "PASS",
                },
                {
                    "label": spec["candidate_role"],
                    "comparison_class": spec["candidate_class"],
                    "provenance": {
                        "gpu_devices": {0: "gpu-one", 1: "gpu-two"}
                    },
                    "outcome": "PASS",
                },
            ],
            "outcome": "PASS",
        }
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            directory = output / "comparisons" / spec["name"]
            directory.mkdir(parents=True)
            report_path = directory / "report.json"
            report_path.write_text(json.dumps(derived, sort_keys=True))
            complete_path = directory / "COMPLETE"
            complete_path.write_text("{}\n")
            build = fake_build()
            archive = pathlib.Path("/archive.tar.gz")
            process = {
                "command": HYBRID.comparison_command(
                    spec, build, archive, output, exact_paths
                )
            }
            record = {
                "spec": spec,
                "process": process,
                "complete": HYBRID.matrix.stable_file_record(
                    complete_path, output, "fixture COMPLETE"
                ),
                "report": HYBRID.matrix.stable_file_record(
                    report_path, output, "fixture report"
                ),
            }
            with mock.patch.object(
                HYBRID.matrix, "_verify_process_record"
            ), mock.patch.object(
                HYBRID, "derive_comparison_report", return_value=derived
            ):
                self.assertEqual(
                    HYBRID.validate_comparison(
                        output, record, spec, build, archive, exact_paths
                    ),
                    record,
                )

    def test_sealed_contract_schema_and_limits_are_fail_closed(self) -> None:
        sealed = self.sealed_contract()
        with mock.patch.object(
            HYBRID.matrix, "load_build", return_value=fake_build()
        ), mock.patch.object(
            HYBRID.matrix, "_source_identity", return_value="source-snapshot"
        ):
            _build, devices, tasks = HYBRID.reconstruct_contract_tasks(sealed)
            self.assertEqual(devices, (GPU_ONE, GPU_TWO))
            self.assertEqual(len(tasks), 10)

            extra = json.loads(json.dumps(sealed))
            extra["unsealed"] = True
            with self.assertRaisesRegex(Exception, "sealed contract is invalid"):
                HYBRID.reconstruct_contract_tasks(extra)

            malformed = json.loads(json.dumps(sealed))
            malformed["limits"]["stdout_limit_bytes"] += 1
            with self.assertRaisesRegex(Exception, "stdout limit is invalid"):
                HYBRID.reconstruct_contract_tasks(malformed)

            wrong_profile = json.loads(json.dumps(sealed))
            wrong_profile["qualification_profile"]["dft_minimum_run_time"] = 19.0
            with self.assertRaisesRegex(Exception, "TM-only qualification profile"):
                HYBRID.reconstruct_contract_tasks(wrong_profile)

            relative = json.loads(json.dumps(sealed))
            relative["input"]["path"] = "archive.tar.gz"
            with self.assertRaisesRegex(Exception, "input contract is invalid"):
                HYBRID.reconstruct_contract_tasks(relative)

    def test_terminal_publication_is_fail_closed_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            for name in ("report.json", "report.md", "JOURNAL.json"):
                (output / name).write_text(f"fixture {name}\n")

            with self.assertRaisesRegex(Exception, "performance gates failed"):
                HYBRID.publish_terminal(output, "FAIL")
            self.assertFalse((output / "PENDING_COMPLETE").exists())
            self.assertFalse((output / "COMPLETE").exists())

            with mock.patch.object(
                HYBRID,
                "verify_hybrid_complete",
                side_effect=HYBRID.WorkloadError("fixture replay failure"),
            ):
                with self.assertRaisesRegex(Exception, "fixture replay failure"):
                    HYBRID.publish_terminal(output, "PASS")
            self.assertTrue((output / "PENDING_COMPLETE").is_file())
            self.assertFalse((output / "COMPLETE").exists())

            (output / "PENDING_COMPLETE").unlink()
            observations = []

            def verify(_output, terminal_name="COMPLETE"):
                observations.append(
                    (
                        terminal_name,
                        (_output / "PENDING_COMPLETE").is_file(),
                        (_output / "COMPLETE").exists(),
                    )
                )
                return {"outcome": "PASS"}

            with mock.patch.object(
                HYBRID, "verify_hybrid_complete", side_effect=verify
            ):
                HYBRID.publish_terminal(output, "PASS")
            self.assertEqual(observations, [("PENDING_COMPLETE", True, False)])
            self.assertFalse((output / "PENDING_COMPLETE").exists())
            self.assertTrue((output / "COMPLETE").is_file())

    def test_journal_rejects_performance_before_exact_comparisons(self) -> None:
        contract = {"fixture": True}
        completed = [
            {
                "task": {"role": f"task-{index}"},
                "process": {"index": index},
                "complete": {"index": index},
            }
            for index in range(5)
        ]
        journal = {
            "schema": HYBRID.JOURNAL_SCHEMA,
            "contract": contract,
            "events": [{"state": "controller-started", "unix_seconds": 1.0}],
        }
        with self.assertRaisesRegex(
            Exception, "performance tasks precede exact comparisons"
        ):
            HYBRID.journal_prefix(journal, contract, completed, [])

    def test_smoke_and_performance_use_distinct_adapter_from_exact(self) -> None:
        tasks = HYBRID.build_tasks(
            fake_build(), 8, ("GPU-one", "GPU-two")
        )
        exact = HYBRID.task_command(tasks[2], pathlib.Path("/a"), pathlib.Path("/o"))
        smoke = HYBRID.task_command(tasks[0], pathlib.Path("/a"), pathlib.Path("/o"))
        performance = HYBRID.task_command(
            tasks[4], pathlib.Path("/a"), pathlib.Path("/o")
        )
        self.assertIn("run_aunp_workload.py", " ".join(exact))
        self.assertIn("run_aunp_performance_workload.py", " ".join(smoke))
        self.assertEqual(smoke[10], performance[10])
        self.assertNotIn("fixed-window", " ".join(performance))

    def test_release_repeats_and_gates_cannot_be_weakened(self) -> None:
        args = self.args()
        HYBRID.validate_args(args)
        with self.assertRaisesRegex(Exception, "exactly two"):
            HYBRID.validate_args(self.args("--performance-repeats", "3"))
        with self.assertRaisesRegex(Exception, "outside release bounds"):
            HYBRID.validate_args(self.args("--minimum-two-gpu-speedup", "1.9"))
        with self.assertRaisesRegex(Exception, "outside release bounds"):
            HYBRID.validate_args(
                self.args("--minimum-gpu-free-memory-fraction", "0.19")
            )

    def test_gpu_uuid_normalization_requires_two_distinct_devices(self) -> None:
        self.assertEqual(
            HYBRID.normalized_devices(f"{GPU_ONE},{GPU_TWO}"),
            (GPU_ONE, GPU_TWO),
        )
        with self.assertRaisesRegex(Exception, "two distinct"):
            HYBRID.normalized_devices(f"{GPU_ONE},{GPU_ONE}")
        with self.assertRaisesRegex(Exception, "invalid physical GPU UUID"):
            HYBRID.normalized_devices("GPU-one")

    def test_capacity_gate_is_task_sensitive_and_exact(self) -> None:
        gates = HYBRID.release_gate_values(self.args())
        snapshot = {
            "path": "/output",
            "free_bytes": 50 * 1024**3,
            "total_bytes": 100 * 1024**3,
            "free_inodes": 200_000,
            "total_inodes": 1_000_000,
        }
        HYBRID.validate_capacity(snapshot, gates, None)
        broken = dict(snapshot)
        broken["free_inodes"] = 99_999
        with self.assertRaisesRegex(Exception, "inode"):
            HYBRID.validate_capacity(broken, gates, None)
        broken = dict(snapshot)
        broken["extra"] = 1
        with self.assertRaisesRegex(Exception, "invalid"):
            HYBRID.validate_capacity(broken, gates, None)

    def test_task_paths_are_unique_for_every_repeat(self) -> None:
        tasks = HYBRID.build_tasks(
            fake_build(), 8, ("GPU-one", "GPU-two")
        )
        for function in (
            HYBRID.task_relative,
            HYBRID.task_runtime_relative,
            HYBRID.task_log_relative,
            HYBRID.task_telemetry_relative,
            HYBRID.task_replay_relative,
        ):
            values = [function(task).as_posix() for task in tasks]
            self.assertEqual(len(values), len(set(values)))

    def performance_records(self, gpu1=2.0, gpu2=1.0):
        tasks = HYBRID.build_tasks(
            fake_build(), 8, ("GPU-one", "GPU-two")
        )
        values = {
            "performance-cpu-fp32-8r": 4.0,
            "performance-cuda-fp32-1g": gpu1,
            "performance-cuda-fp32-2g": gpu2,
        }
        records = []
        for task in tasks:
            if not task.counts_toward_performance:
                continue
            records.append(
                {
                    "task": HYBRID.task_spec(task),
                    "phase_timestep_deltas": [2000, 2000],
                    "fdtd_wall_seconds": values[task.role],
                    "workload_end_to_end_seconds": values[task.role] * 1.1,
                }
            )
        return records

    def test_performance_report_gates_kernel_and_end_to_end_timings(self) -> None:
        gates = HYBRID.release_gate_values(self.args())
        performance, diagnostics = HYBRID.performance_report(
            self.performance_records(), 8, gates
        )
        self.assertEqual(HYBRID.performance_outcome(performance), "PASS")
        self.assertEqual(set(performance), {
            "fdtd_wall_seconds",
            "workload_end_to_end_seconds",
        })
        metric = performance["fdtd_wall_seconds"]["cpu-fp32-to-one-gpu"]
        self.assertEqual(metric["reference_samples_seconds"], [4.0, 4.0])
        self.assertEqual(metric["candidate_samples_seconds"], [2.0, 2.0])
        self.assertEqual(metric["reference_mean_seconds"], 4.0)
        self.assertEqual(metric["candidate_mean_seconds"], 2.0)
        self.assertEqual(metric["mean_speedup"], 2.0)
        self.assertNotIn("median_speedup", metric)
        self.assertNotIn("conservative_speedup", metric)
        self.assertIsNone(
            diagnostics["fdtd_wall_seconds"]["performance-cpu-fp32-8r"][
                "release_gate"
            ]
        )
        performance, _diagnostics = HYBRID.performance_report(
            self.performance_records(gpu1=3.0, gpu2=2.5), 8, gates
        )
        self.assertEqual(HYBRID.performance_outcome(performance), "FAIL")

    def test_two_run_dispersion_is_diagnostic_only(self) -> None:
        records = self.performance_records()
        cpu_records = [
            record
            for record in records
            if record["task"]["role"] == "performance-cpu-fp32-8r"
        ]
        cpu_records[0]["fdtd_wall_seconds"] = 4.0
        cpu_records[1]["fdtd_wall_seconds"] = 8.0
        cpu_records[0]["workload_end_to_end_seconds"] = 4.4
        cpu_records[1]["workload_end_to_end_seconds"] = 8.8
        performance, diagnostics = HYBRID.performance_report(
            records, 8, HYBRID.release_gate_values(self.args())
        )
        self.assertEqual(HYBRID.performance_outcome(performance), "PASS")
        self.assertGreater(
            diagnostics["fdtd_wall_seconds"]["performance-cpu-fp32-8r"][
                "coefficient_of_variation"
            ],
            0.15,
        )

    def test_performance_report_rejects_smoke_substitution_and_short_work(self) -> None:
        gates = HYBRID.release_gate_values(self.args())
        records = self.performance_records()
        records[0]["task"]["counts_toward_performance"] = False
        with self.assertRaisesRegex(Exception, "inventory"):
            HYBRID.performance_report(records, 8, gates)
        records = self.performance_records()
        records[0]["phase_timestep_deltas"][0] = 1999
        with self.assertRaisesRegex(Exception, "timestep"):
            HYBRID.performance_report(records, 8, gates)

    def test_observer_inventory_tracks_backend_and_selected_uuids(self) -> None:
        tasks = HYBRID.build_tasks(
            fake_build(), 8, (GPU_ONE, GPU_TWO)
        )
        contract = {"limits": {"telemetry_interval_seconds": 5.0}}
        gpu1 = HYBRID.observer_specs(tasks[0], 42, pathlib.Path("/out"), contract)
        gpu2 = HYBRID.observer_specs(tasks[1], 42, pathlib.Path("/out"), contract)
        cpu = HYBRID.observer_specs(tasks[4], 42, pathlib.Path("/out"), contract)
        self.assertEqual([item.label for item in gpu1], ["host", "gpu"])
        self.assertEqual([item.label for item in gpu2], ["host", "gpu"])
        self.assertEqual([item.label for item in cpu], ["host"])
        self.assertEqual(gpu1[1].command.count("--selected-uuid"), 1)
        self.assertEqual(gpu2[1].command.count("--selected-uuid"), 2)
        self.assertIn(GPU_ONE, gpu2[1].command)
        self.assertIn(GPU_TWO, gpu2[1].command)

    def test_clock_event_report_binds_disclosure_and_rejects_performance_activity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            schedule = HYBRID.build_tasks(
                fake_build(), 8, (GPU_ONE, GPU_TWO)
            )
            selected = (schedule[2], schedule[5])
            completed = []

            def replay_for(task, active):
                reasons = {
                    uuid: {
                        name: {
                            "active_sample_count": int(active),
                            "active_alive_sample_count": int(active),
                            "first_active_unix_seconds": 1.0 if active else None,
                            "last_active_unix_seconds": 1.0 if active else None,
                        }
                        for name in HYBRID.telemetry_replay.CLOCK_EVENT_REASON_METRICS
                    }
                    for uuid in task.lane.devices
                }
                return {
                    "status": "PASS",
                    "gpu": {
                        "clock_event_policy": (
                            "reject-active-during-alive-performance-samples"
                            if task.counts_toward_performance
                            else "disclose-active-reasons"
                        ),
                        "clock_event_reasons": reasons,
                        "process_coverage": {"outcome": "PASS"},
                    },
                }

            for index, task in enumerate(selected):
                path = output / f"replay-{index}.json"
                HYBRID.atomic_write_json(path, replay_for(task, index == 0))
                completed.append(
                    {
                        "task": HYBRID.task_spec(task),
                        "telemetry_replay": HYBRID.file_record(path, output),
                    }
                )

            report = HYBRID.clock_event_report(output, completed)
            self.assertEqual(report["performance_outcome"], "PASS")
            self.assertEqual(report["performance_active_alive_sample_count"], 0)
            self.assertGreater(report["nonperformance_active_alive_sample_count"], 0)

            exact_path = output / "replay-0.json"
            exact_path.write_bytes(exact_path.read_bytes() + b" ")
            with self.assertRaisesRegex(Exception, "size mismatch|digest mismatch"):
                HYBRID.clock_event_report(output, completed)

            HYBRID.atomic_write_json(
                output / "replay-0-restored.json", replay_for(selected[0], True)
            )
            completed[0]["telemetry_replay"] = HYBRID.file_record(
                output / "replay-0-restored.json", output
            )
            performance_path = output / "replay-1.json"
            performance_path.unlink()
            HYBRID.atomic_write_json(
                performance_path, replay_for(selected[1], True)
            )
            completed[1]["telemetry_replay"] = HYBRID.file_record(
                performance_path, output
            )
            with self.assertRaisesRegex(Exception, "measured performance"):
                HYBRID.clock_event_report(output, completed)

    def test_main_executes_complete_schedule_before_terminal_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            archive = root / "archive.tar.gz"
            archive.write_bytes(b"fixture archive\n")
            output = root / "evidence"
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            build = fake_build()
            hardware = {"fixture": "hardware"}
            process_index = 0

            def capacity(path):
                return {
                    "path": str(path.resolve()),
                    "free_bytes": 100 * 1024**3,
                    "total_bytes": 200 * 1024**3,
                    "free_inodes": 500_000,
                    "total_inodes": 1_000_000,
                }

            def run_observed(command, _env, _root, _log, *_args, **_kwargs):
                nonlocal process_index
                process_index += 1
                return (
                    {
                        "command": command,
                        "command_pid": 1000 + process_index,
                        "returncode": 0,
                        "timed_out": False,
                        "output_limited": False,
                        "output_bytes": 0,
                        "peak_process_group_rss_bytes": 1,
                        "started_unix_seconds": 1.0,
                        "ended_unix_seconds": 2.0,
                        "wall_seconds": 1.0,
                        "log": {"path": "fixture", "size_bytes": 0, "sha256": "0" * 64},
                    },
                    [],
                )

            def lane_evidence(task, _lane, _output, *, publish_replay):
                self.assertTrue(publish_replay)
                values = {
                    "performance-cpu-fp32-8r": 4.0,
                    "performance-cuda-fp32-1g": 2.0,
                    "performance-cuda-fp32-2g": 1.0,
                }
                timing = values.get(task.role, 1.0)
                return {
                    "fdtd_wall_seconds": timing,
                    "phase_wall_seconds": [timing / 2] * 2,
                    "phase_timestep_deltas": [2000] * 2,
                    "workload_end_to_end_seconds": timing * 1.1,
                    "provenance": {"fixture": True},
                    "complete": {"path": f"{task.role}/COMPLETE"},
                    "lane_replay": None,
                }

            def comparisons(
                _output,
                retained,
                _completed,
                journal,
                _build,
                _archive,
                _args,
                checkpoint_callback=None,
            ):
                for spec in HYBRID.exact_comparison_specs()[len(retained) :]:
                    record = {
                        "spec": spec,
                        "process": {"fixture": True},
                        "complete": {"path": f"{spec['name']}/COMPLETE"},
                        "report": {"path": f"{spec['name']}/report.json"},
                    }
                    retained.append(record)
                    journal["events"].extend(
                        [
                            {
                                "state": "comparison-process-ended",
                                "spec": spec,
                                "process": record["process"],
                            },
                            {
                                "state": "comparison-validated",
                                "spec": spec,
                                "complete": record["complete"],
                            },
                        ]
                    )
                    if checkpoint_callback:
                        checkpoint_callback()

            argv = [
                "--archive",
                str(archive),
                "--output",
                str(output),
                "--fp32-python",
                "/python",
                "--fp32-mpiexec",
                "/mpirun",
                "--fp32-receipt",
                "/receipt",
                "--cpu-ranks",
                "8",
                "--gpu-devices",
                f"{GPU_ONE},{GPU_TWO}",
            ]
            with mock.patch.object(
                HYBRID, "AUNP_ARCHIVE_SHA256", digest
            ), mock.patch.object(
                HYBRID, "verify_file", return_value=archive.resolve()
            ), mock.patch.object(
                HYBRID.matrix, "load_build", return_value=build
            ), mock.patch.object(
                HYBRID.matrix, "_source_identity", return_value="source-snapshot"
            ), mock.patch.object(
                HYBRID.matrix, "physical_core_count", return_value=8
            ), mock.patch.object(
                HYBRID.matrix, "hardware_snapshot", return_value=hardware
            ), mock.patch.object(
                HYBRID.matrix, "validate_hardware_evidence", return_value=None
            ), mock.patch.object(
                HYBRID.matrix, "clean_environment", return_value={"PATH": "/usr/bin:/bin"}
            ), mock.patch.object(
                HYBRID, "capacity_snapshot", side_effect=capacity
            ), mock.patch.object(
                HYBRID.observed,
                "run_bounded_observed",
                side_effect=run_observed,
            ), mock.patch.object(
                HYBRID, "derive_lane_evidence", side_effect=lane_evidence
            ), mock.patch.object(
                HYBRID,
                "publish_telemetry_replay",
                return_value={"path": "telemetry/replay.json"},
            ), mock.patch.object(
                HYBRID,
                "clock_event_report",
                return_value={
                    "schema": "gpmeep-aunp-clock-event-report-v1",
                    "performance_policy": "zero-active-clock-event-reasons",
                    "nonperformance_policy": "disclose-active-clock-event-reasons",
                    "performance_active_alive_sample_count": 0,
                    "nonperformance_active_alive_sample_count": 0,
                    "performance_outcome": "PASS",
                    "tasks": [],
                },
            ), mock.patch.object(
                HYBRID, "run_missing_comparisons", side_effect=comparisons
            ):
                terminal_observations = []

                def final_replay(_output, terminal_name="COMPLETE"):
                    terminal_observations.append(
                        {
                            "terminal_name": terminal_name,
                            "pending_exists": (_output / "PENDING_COMPLETE").is_file(),
                            "complete_exists": (_output / "COMPLETE").exists(),
                        }
                    )
                    return {"outcome": "PASS"}

                with mock.patch.object(
                    HYBRID, "verify_hybrid_complete", side_effect=final_replay
                ) as verifier:
                    self.assertEqual(HYBRID.main(argv), 0)

            report = json.loads((output / "report.json").read_text())
            self.assertEqual(len(report["tasks"]), 10)
            self.assertEqual(len(report["exact_comparisons"]), 2)
            self.assertEqual(report["outcome"], "PASS")
            self.assertEqual(process_index, 10)
            verifier.assert_called_once_with(output, "PENDING_COMPLETE")
            self.assertEqual(
                terminal_observations,
                [
                    {
                        "terminal_name": "PENDING_COMPLETE",
                        "pending_exists": True,
                        "complete_exists": False,
                    }
                ],
            )
            self.assertTrue((output / "COMPLETE").is_file())
            self.assertFalse((output / "PENDING_COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
