from __future__ import annotations

import importlib.util
import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "run-mpi-completion-ab.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_completion_ab_tested", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AB)


def sample(
    policy: str,
    kind: str,
    iteration: int,
    *,
    workload_seconds: float,
    process_seconds: float,
    gradient_delta: float = 0.0,
) -> dict:
    gradient = [1.0, 2.0 + gradient_delta]
    gradient_sha256 = AB.EVIDENCE.canonical_float64_sha256(gradient)
    environment = {
        "MEEP_GPU_MPI_COMPLETION": policy,
        "MEEP_GPU_BACKEND": "cuda",
    }
    statistics = {"multi_gpu": {"mpi_messages": 10}, "dispatch": {"calls": 4}}
    rank_runtime = {
        "python_executable": {"path": "/python", "sha256": "1" * 64},
        "loaded_libraries": {},
    }
    result = {
        "workload": {
            "resolution": 64,
            "run_time": 40.0,
            "cell_size": 32.0,
            "design_resolution": 20,
            "design_shape": [1, 2],
            "design_variables": 2,
            "yee_cells": 100,
            "weights_sha256": "2" * 64,
            "direction_sha256": "3" * 64,
        },
        "result": {
            "objective": 1.0,
            "gradient": gradient,
            "gradient_shape": [1, 2],
            "gradient_count": 2,
            "gradient_sha256": gradient_sha256,
        },
        "timing": {"workload_wall_seconds": workload_seconds},
        "distributed": {
            "ranks": [
                {
                    "process_environment": dict(environment),
                    "final_statistics": statistics,
                },
                {
                    "process_environment": dict(environment),
                    "final_statistics": statistics,
                },
            ]
        },
    }
    return {
        "completion_policy": policy,
        "sample_kind": kind,
        "iteration": iteration,
        "nonce": f"{policy}-{kind}-{iteration}",
        "lane": "cuda-multi",
        "ranks": 2,
        "capture_field_times": True,
        "process_seconds": process_seconds,
        "result": result,
        "validation": {
            "objective": 1.0,
            "workload_wall_seconds": workload_seconds,
            "device_identifiers": ["a" * 32, "b" * 32],
            "rank_runtimes": [rank_runtime, rank_runtime],
        },
    }


def matrix(*, speedup: float = 1.01, gradient_delta: float = 0.0) -> list[dict]:
    samples = []
    for policy, kind, iteration in AB._expected_matrix():
        some_seconds = 10.0 + 0.01 * iteration
        wall = some_seconds if policy == "waitsome" else some_seconds / speedup
        samples.append(
            sample(
                policy,
                kind,
                iteration,
                workload_seconds=wall,
                process_seconds=wall + 0.5,
                gradient_delta=gradient_delta if policy == "waitall" else 0.0,
            )
        )
    return samples


def add_fresh_mpi_sessions(samples: list[dict]) -> None:
    for sample_index, item in enumerate(samples):
        command = [
            "/python",
            "/repo/scripts/benchmark-adjoint.py",
            "--run-nonce",
            item["nonce"],
        ]
        item["result"]["producer"] = {"command": command}
        for rank_index, rank in enumerate(item["result"]["distributed"]["ranks"]):
            rank["process_environment"].update(
                {
                    "OMPI_ARGV": " ".join(command[1:]),
                    "OMPI_FILE_LOCATION": f"/tmp/ompi.{sample_index}/1/{rank_index}",
                    "PMIX_NAMESPACE": f"prterun-host-{sample_index}@1",
                    "PMIX_SERVER_TMPDIR": f"/tmp/ompi.{sample_index}",
                    "PMIX_SERVER_URI2": f"job-{sample_index};tcp4://127.0.0.1:{30000 + sample_index}",
                    "PMIX_SERVER_URI21": f"job-{sample_index};tcp4://127.0.0.1:{30000 + sample_index}",
                    "PMIX_SERVER_URI3": f"job-{sample_index};tcp4://127.0.0.1:{30000 + sample_index}",
                    "PMIX_SERVER_URI4": f"job-{sample_index};tcp4://127.0.0.1:{30000 + sample_index}",
                    "PMIX_SERVER_URI41": f"job-{sample_index};tcp4://127.0.0.1:{30000 + sample_index}",
                }
            )


class CompletionABTests(unittest.TestCase):
    def test_reverify_rejects_stored_validation_and_ledger_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            raw_output = root / "raw"
            raw_output.mkdir()
            repo = SCRIPT.parents[1]
            run_id = "run"
            receipt_path = root / "receipt.json"
            receipt_path.write_text("{}\n", encoding="utf-8")
            snapshot = {"sha256": "a" * 64}
            receipt = {"receipt_id": "receipt", "source_end": snapshot}
            samples = matrix(speedup=1.01)
            add_fresh_mpi_sessions(samples)
            for item in samples:
                label = (
                    f"{run_id}-{item['sample_kind']}-{item['iteration']}-"
                    f"{item['completion_policy']}"
                )
                item["result"]["producer"]["command"] = ["producer", "arg"]
                for rank in item["result"]["distributed"]["ranks"]:
                    rank["process_environment"]["OMPI_ARGV"] = "arg"
                result_path = raw_output / f"{label}.json"
                stdout_path = raw_output / f"{label}.stdout.log"
                stderr_path = raw_output / f"{label}.stderr.log"
                timing_path = raw_output / f"{label}.parent-timing.json"
                result_path.write_text(
                    json.dumps(item["result"], sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                result_sha256 = AB.EVIDENCE.sha256_file(result_path)
                stdout_path.write_text(
                    AB.RUNNER.PREFIX
                    + json.dumps(
                        {
                            "result_file": str(result_path.resolve()),
                            "sha256": result_sha256,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                stderr_path.write_text("", encoding="utf-8")
                item.update(
                    {
                        "completion_policy": item["completion_policy"],
                        "artifact_label": label,
                        "receipt_gate": {"pass": True},
                        "process_environment_gate": {"pass": True},
                        "lazy_import_gate": {"pass": True},
                        "material_gradient_gate": {"pass": True},
                        "result_file": str(result_path.resolve()),
                        "result_sha256": result_sha256,
                        "stdout_log": str(stdout_path.resolve()),
                        "stdout_sha256": AB.EVIDENCE.sha256_file(stdout_path),
                        "stderr_log": str(stderr_path.resolve()),
                        "stderr_sha256": AB.EVIDENCE.sha256_file(stderr_path),
                        "timing_file": str(timing_path.resolve()),
                        "command": ["command"],
                        "environment": {
                            "BASE": "1",
                            "MEEP_GPU_MPI_COMPLETION": item["completion_policy"],
                        },
                    }
                )
                item["validation"]["rank_driver"] = [{"pass": True}]
                timing_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "run_id": run_id,
                            "label": label,
                            "lane": "cuda-multi",
                            "sample_kind": item["sample_kind"],
                            "iteration": item["iteration"],
                            "nonce": item["nonce"],
                            "process_seconds": item["process_seconds"],
                            "command_sha256": AB.PROVENANCE.canonical_sha256(
                                item["command"]
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
                item["timing_sha256"] = AB.EVIDENCE.sha256_file(timing_path)
                self.assertEqual(set(item), AB.RUNNER.EXPECTED_SAMPLE_KEYS)
            ledger = AB._expected_controller_ledger(
                output=raw_output, run_id=run_id, samples=samples
            )
            fresh_by_nonce = {
                item["nonce"]: {
                    key: copy.deepcopy(value)
                    for key, value in item["validation"].items()
                    if key != "rank_driver"
                }
                for item in samples
            }
            def fresh_validation(raw, **_kwargs):
                match = next(
                    item for item in samples if item["result"] == raw
                )
                return copy.deepcopy(fresh_by_nonce[match["nonce"]])

            common = (
                mock.patch.object(
                    AB.PROVENANCE, "verify_build_receipt", return_value=receipt
                ),
                mock.patch.object(
                    AB.PROVENANCE, "source_snapshot", return_value=snapshot
                ),
                mock.patch.object(
                    AB.RUNNER,
                    "validate_build_receipt_contract",
                    return_value={"pass": True},
                ),
                mock.patch.object(AB.RUNNER, "_receipt_runtime", return_value={}),
                mock.patch.object(
                    AB.RUNNER, "_expected_sample_command", return_value=["command"]
                ),
                mock.patch.object(
                    AB.RUNNER,
                    "_child_environment",
                    return_value={"BASE": "1"},
                ),
                mock.patch.object(
                    AB.RUNNER,
                    "_expected_producer_command",
                    return_value=["producer", "arg"],
                ),
                mock.patch.object(
                    AB.RUNNER.COMPARATOR,
                    "validate_qualification_binding",
                    return_value=None,
                ),
                mock.patch.object(AB.RUNNER, "_fixed_workload_gate", return_value=None),
                mock.patch.object(
                    AB.RUNNER,
                    "_validate_lazy_import_contract",
                    return_value={"pass": True},
                ),
                mock.patch.object(
                    AB.RUNNER.COMPARATOR,
                    "validate_distributed_record",
                    side_effect=fresh_validation,
                ),
                mock.patch.object(
                    AB.RUNNER,
                    "_validate_cuda_rank_driver_evidence",
                    return_value=[{"pass": True}],
                ),
                mock.patch.object(
                    AB.RUNNER,
                    "_validate_receipt_rank_runtimes",
                    return_value={"pass": True},
                ),
                mock.patch.object(
                    AB.RUNNER,
                    "_validate_actual_process_environments",
                    return_value={"pass": True},
                ),
                mock.patch.object(
                    AB.RUNNER, "_validate_lane_process_environment", return_value=None
                ),
                mock.patch.object(
                    AB.RUNNER,
                    "_extract_material_gradient_stats",
                    return_value={"pass": True},
                ),
            )
            for patcher in common:
                patcher.start()
                self.addCleanup(patcher.stop)
            kwargs = {
                "expected_run_id": run_id,
                "repo": repo,
                "receipt_path": receipt_path,
                "expected_receipt_id": "receipt",
                "expected_snapshot": snapshot,
                "expected_producer_sha256": AB.EVIDENCE.sha256_file(
                    repo / "scripts" / "benchmark-adjoint.py"
                ),
                "samples": samples,
                "expected_controller_ledger": ledger,
            }
            verified = AB._reverify(**kwargs)
            self.assertTrue(verified["controller_ledger"]["pass"])
            samples[0]["validation"]["objective"] = 9.0
            with self.assertRaisesRegex(RuntimeError, "fresh-derived"):
                AB._reverify(**kwargs)
            samples[0]["validation"]["objective"] = 1.0
            forged_ledger = copy.deepcopy(ledger)
            next(iter(forged_ledger["measurements"].values()))[
                "process_seconds"
            ] += 1.0
            with self.assertRaisesRegex(RuntimeError, "controller publication ledger"):
                AB._reverify(**{**kwargs, "expected_controller_ledger": forged_ledger})

    def test_parent_timing_attestation_rejects_identity_and_time_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "timing.json"
            command = ["/python", "/producer.py", "--fixed"]
            item = sample(
                "waitsome",
                "measured",
                0,
                workload_seconds=10.0,
                process_seconds=10.5,
            )
            item["timing_file"] = str(path)
            record = {
                "schema_version": 1,
                "run_id": "run",
                "label": "run-measured-0-waitsome",
                "lane": "cuda-multi",
                "sample_kind": "measured",
                "iteration": 0,
                "nonce": item["nonce"],
                "process_seconds": item["process_seconds"],
                "command_sha256": AB.PROVENANCE.canonical_sha256(command),
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(
                AB._validate_parent_timing_attestation(
                    item,
                    expected_run_id="run",
                    label="run-measured-0-waitsome",
                    expected_command=command,
                ),
                record,
            )
            for field, value in (("process_seconds", 0.1), ("nonce", "forged")):
                changed = dict(record)
                changed[field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(field=field), self.assertRaisesRegex(
                    RuntimeError, "parent timing attestation changed"
                ):
                    AB._validate_parent_timing_attestation(
                        item,
                        expected_run_id="run",
                        label="run-measured-0-waitsome",
                        expected_command=command,
                    )

    def test_fixed_profile_and_interleaved_matrix(self):
        AB._assert_fixed_profile()
        samples = matrix()
        validated = AB.validate_sample_matrix(samples)
        self.assertTrue(validated["pass"])
        self.assertEqual(validated["actual"], AB._expected_matrix())
        measured = [row for row in AB._expected_matrix() if row[1] == "measured"]
        for policy in AB.PROFILE["policies"]:
            self.assertEqual(
                sum(
                    1
                    for iteration in range(
                        AB.PROFILE["measured_repeats_per_policy"]
                    )
                    if [row[0] for row in measured if row[2] == iteration][0]
                    == policy
                ),
                AB.PROFILE["measured_repeats_per_policy"] // 2,
            )

    def test_duplicate_or_reordered_sample_rejected(self):
        samples = matrix()
        samples[2], samples[3] = samples[3], samples[2]
        with self.assertRaisesRegex(RuntimeError, "not interleaved"):
            AB.validate_sample_matrix(samples)
        samples = matrix()
        samples[-1]["nonce"] = samples[0]["nonce"]
        with self.assertRaisesRegex(RuntimeError, "nonce was reused"):
            AB.validate_sample_matrix(samples)

    def test_valid_negative_result_is_complete_but_not_promoted(self):
        compared = AB.compare_samples(matrix(speedup=1.01))
        self.assertTrue(compared["evidence_gate"]["pass"])
        self.assertFalse(compared["promotion"]["pass"])
        self.assertEqual(
            compared["promotion"]["decision"], "retain-waitsome-default"
        )

    def test_material_result_requires_both_wall_metrics_and_majority(self):
        compared = AB.compare_samples(matrix(speedup=1.04))
        self.assertTrue(compared["promotion"]["pass"])
        self.assertEqual(compared["promotion"]["decision"], "promote-waitall")
        self.assertEqual(compared["promotion"]["waitall_faster_pairs"], 6)

    def test_unstratified_process_speedup_cannot_hide_one_failed_position(self):
        samples = matrix(speedup=1.04)
        for item in samples:
            if item["sample_kind"] != "measured":
                continue
            order = (
                AB.PROFILE["policies"]
                if item["iteration"] % 2 == 0
                else list(reversed(AB.PROFILE["policies"]))
            )
            position = "first" if item["completion_policy"] == order[0] else "second"
            item["process_seconds"] = {
                ("waitsome", "first"): 20.0,
                ("waitsome", "second"): 10.0,
                ("waitall", "first"): 20.0,
                ("waitall", "second"): 5.0,
            }[(item["completion_policy"], position)]
        compared = AB.compare_samples(samples)
        process = compared["performance"]["fresh_process_wall"]
        self.assertGreater(
            process[
                "unstratified_waitsome_over_waitall_median_speedup_diagnostic_only"
            ],
            AB.PROFILE["material_promotion_speedup"],
        )
        self.assertGreater(
            process["by_position"]["balanced_geometric_mean_speedup"],
            AB.PROFILE["material_promotion_speedup"],
        )
        self.assertLess(
            process["by_position"]["waitsome_over_waitall_speedup"]["first"],
            AB.PROFILE["material_promotion_speedup"],
        )
        self.assertTrue(process["position_effect_diagnostic"]["warning"])
        self.assertFalse(compared["promotion"]["both_process_position_strata_pass"])
        self.assertFalse(compared["promotion"]["pass"])

    def test_gradient_tamper_fails_evidence_gate(self):
        with self.assertRaisesRegex(RuntimeError, "gradient"):
            AB.compare_samples(matrix(speedup=1.04, gradient_delta=1.0))

    def test_fresh_mpi_session_identifiers_are_validated_then_projected(self):
        samples = matrix(speedup=1.01)
        add_fresh_mpi_sessions(samples)
        compared = AB.compare_samples(samples)
        self.assertTrue(compared["evidence_gate"]["pass"])
        self.assertFalse(compared["promotion"]["pass"])

    def test_unclassified_environment_change_still_fails_closed(self):
        samples = matrix(speedup=1.01)
        add_fresh_mpi_sessions(samples)
        samples[-1]["result"]["distributed"]["ranks"][0][
            "process_environment"
        ]["UNCLASSIFIED_RUNTIME_CONTROL"] = "injected"
        with self.assertRaisesRegex(RuntimeError, "stable_environment_except"):
            AB.compare_samples(samples)

    def test_mpi_argv_must_match_validated_producer_command(self):
        samples = matrix(speedup=1.01)
        add_fresh_mpi_sessions(samples)
        samples[-1]["result"]["distributed"]["ranks"][0][
            "process_environment"
        ]["OMPI_ARGV"] += " --injected"
        with self.assertRaisesRegex(RuntimeError, "OMPI_ARGV"):
            AB.compare_samples(samples)

    def test_mpi_argv_is_required_with_a_recorded_producer_command(self):
        samples = matrix(speedup=1.01)
        add_fresh_mpi_sessions(samples)
        samples[-1]["result"]["distributed"]["ranks"][0][
            "process_environment"
        ].pop("OMPI_ARGV")
        with self.assertRaisesRegex(RuntimeError, "OMPI_ARGV is absent"):
            AB.compare_samples(samples)

    def test_mpi_session_key_presence_must_remain_stable(self):
        samples = matrix(speedup=1.01)
        add_fresh_mpi_sessions(samples)
        samples[-1]["result"]["distributed"]["ranks"][0][
            "process_environment"
        ].pop("PMIX_SERVER_URI41")
        with self.assertRaisesRegex(RuntimeError, "stable_environment_except"):
            AB.compare_samples(samples)

    def test_reserved_session_sentinel_cannot_be_injected(self):
        samples = matrix(speedup=1.01)
        add_fresh_mpi_sessions(samples)
        samples[-1]["result"]["distributed"]["ranks"][0][
            "process_environment"
        ][AB.MPI_SESSION_PRESENCE_SENTINEL] = "forged"
        with self.assertRaisesRegex(RuntimeError, "reserved session sentinel"):
            AB.compare_samples(samples)

    def test_publisher_rejects_minimal_unverified_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            (output / "raw").mkdir()
            (output / "raw" / "sample.json").write_text("{}\n", encoding="utf-8")
            (output / "build-provenance.json").write_text("{}\n", encoding="utf-8")
            report = {
                "build_receipt": {"receipt_id": "r" * 64},
                "source_snapshot": {"sha256": "s" * 64},
            }
            with self.assertRaises(TypeError):
                AB._publish(
                    output=output,
                    run_id="run",
                    report=report,
                    markdown="# report\n",
                )
            with mock.patch.object(
                AB, "_reverify", return_value={"pass": True}
            ), self.assertRaisesRegex(
                RuntimeError, "live controller ledger|not freshly derived"
            ):
                publication_capability = AB.RUNNER._RunPublicationCapability(
                    output / "raw", "run"
                )
                publication_capability.seal()
                AB._publish(
                    output=output,
                    run_id="run",
                    report=report,
                    markdown="# report\n",
                    publication_capability=publication_capability,
                    repo=SCRIPT.parents[1],
                    receipt_path=output / "build-provenance.json",
                    expected_receipt_id="r" * 64,
                    expected_snapshot={"sha256": "s" * 64},
                    expected_producer_sha256="p" * 64,
                    expected_build_receipt=report["build_receipt"],
                    expected_integrity={"pass": True},
                    expected_controller_ledger={
                        "measurements": {},
                        "artifacts": {},
                    },
                    samples=matrix(),
                )
            self.assertFalse((output / "COMPLETE").exists())
            self.assertFalse((output / "report.json").exists())

    def test_nonempty_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            existing = output / "keep.txt"
            existing.write_text("user data\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "absent or empty"):
                AB._prepare_output(output, "run")
            self.assertEqual(existing.read_text(encoding="utf-8"), "user data\n")

    def test_raw_artifact_set_rejects_extra_same_iteration_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = pathlib.Path(temporary)
            samples = matrix()
            run_id = "run"
            for directory in (
                raw / "home" / run_id,
                raw / "matplotlib",
                raw / "pycache" / run_id,
                raw / "tmp",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (raw / "home" / run_id / ".gpmeep-empty-home").write_text(
                "gpmeep isolated MPI qualification home\n", encoding="utf-8"
            )
            for index, item in enumerate(samples):
                for key, suffix in (
                    ("result_file", ".json"),
                    ("stdout_log", ".stdout.log"),
                    ("stderr_log", ".stderr.log"),
                    ("timing_file", ".parent-timing.json"),
                ):
                    path = raw / f"{run_id}-sample-{index}{suffix}"
                    path.write_text("\n", encoding="utf-8")
                    item[key] = str(path)
            AB.validate_raw_artifact_set(raw, samples)
            (raw / "sample-0-copy.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicated"):
                AB.validate_raw_artifact_set(raw, samples)

    def test_raw_artifact_set_rejects_arbitrary_extension_and_subdirectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = pathlib.Path(temporary)
            samples = matrix()
            run_id = "run"
            for directory in (
                raw / "home" / run_id,
                raw / "matplotlib",
                raw / "pycache" / run_id,
                raw / "tmp",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (raw / "home" / run_id / ".gpmeep-empty-home").write_text(
                "gpmeep isolated MPI qualification home\n", encoding="utf-8"
            )
            for index, item in enumerate(samples):
                for key, suffix in (
                    ("result_file", ".json"),
                    ("stdout_log", ".stdout.log"),
                    ("stderr_log", ".stderr.log"),
                    ("timing_file", ".parent-timing.json"),
                ):
                    path = raw / f"{run_id}-sample-{index}{suffix}"
                    path.write_text("\n", encoding="utf-8")
                    item[key] = str(path)
            AB.validate_raw_artifact_set(raw, samples)
            (raw / "injected.bin").write_bytes(b"injected")
            with self.assertRaisesRegex(RuntimeError, "duplicated"):
                AB.validate_raw_artifact_set(raw, samples)
            (raw / "injected.bin").unlink()
            nested = raw / "unexpected"
            nested.mkdir()
            with self.assertRaisesRegex(RuntimeError, "duplicated"):
                AB.validate_raw_artifact_set(raw, samples)


if __name__ == "__main__":
    unittest.main()
