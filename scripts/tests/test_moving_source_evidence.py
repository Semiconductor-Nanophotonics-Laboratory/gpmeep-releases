import copy
import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "python-validation" / "sitecustomize.py"
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_moving_source_sitecustomize_test", MODULE_PATH
)
SITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SITE)

PRODUCER_PATH = ROOT / "scripts" / "benchmark-moving-source-mpi-development.py"
PRODUCER_SPEC = importlib.util.spec_from_file_location(
    "gpmeep_moving_source_development_producer_test", PRODUCER_PATH
)
PRODUCER = importlib.util.module_from_spec(PRODUCER_SPEC)
PRODUCER_SPEC.loader.exec_module(PRODUCER)


class TestMovingSourceEvidence(unittest.TestCase):
    def valid_contract(self):
        contract = {
            "environment": {
                "OMP_NUM_THREADS": "1",
                "workspace": "/tmp/검증 경로",
            },
            "requested_backend": "auto",
            "active_backend": "cpu",
        }
        contract["invocation_sha256"] = SITE._canonical_sha256(contract)
        return contract

    def valid_statistics(self, calls):
        return {
            "runtime": {
                "runtime_availability_probes": 0,
                "runtime_device_enumerations": 0,
                "runtime_device_selections": 0,
            },
            "dispatch": {
                "cpu_curl_calls": calls,
                "cpu_curl_points": 10 * calls,
                "cuda_curl_calls": 0,
                "cuda_curl_points": 0,
                "host_to_device_bytes": 0,
                "device_to_host_bytes": 0,
            },
            "resident": {
                "device_buffer_allocations": 0,
            },
            "field_updates": {
                "cpu_update_eh_calls": calls,
                "cpu_update_eh_points": 10 * calls,
                "cuda_update_eh_calls": 0,
                "cuda_update_eh_points": 0,
            },
            "polarizations": {
                "cpu_polarization_calls": 0,
                "cuda_polarization_calls": 0,
            },
            "sources": {
                "cpu_source_calls": calls,
                "cpu_source_points": 10 * calls,
                "cuda_source_calls": 0,
                "cuda_source_points": 0,
            },
            "boundaries": {
                "cpu_boundary_calls": calls,
                "cpu_boundary_points": 10 * calls,
                "cuda_boundary_calls": 0,
                "cuda_boundary_points": 0,
            },
            "dfts": {"cpu_dft_calls": 0, "cuda_dft_calls": 0},
            "dft_reductions": {
                "cpu_dft_reduction_calls": 0,
                "cuda_dft_reduction_calls": 0,
            },
            "dft_materializations": {
                "cpu_dft_array_materialization_calls": 0,
                "cuda_dft_array_materialization_calls": 0,
                "cpu_dft_output_calls": 0,
                "cuda_dft_output_calls": 0,
            },
            "eigenmode_overlaps": {
                "cpu_eigenmode_overlap_calls": 0,
                "cuda_eigenmode_overlap_calls": 0,
            },
            "ldos": {
                "cpu_ldos_reduction_calls": 0,
                "cuda_ldos_reduction_calls": 0,
            },
            "near2far": {
                "cpu_near2far_transform_calls": 0,
                "cuda_near2far_transform_calls": 0,
                "cpu_near2far_adjoint_calls": 0,
                "cuda_near2far_adjoint_calls": 0,
                "near2far_mpi_allreduce_calls": 0,
                "near2far_mpi_allreduce_bytes": 0,
            },
        }

    def valid_result(self):
        decision = {
            "requested_backend": "auto",
            "active_backend": "cpu",
            "backend_diagnostic": "automatic fields owner selected CPU",
            "process_active_backend": "cpu",
            "process_backend_diagnostic": (
                "automatic fields owner selected CPU"
            ),
            "selected_device": -1,
        }
        workload = SITE._expected_moving_source_workload(1)
        metric = {
            "shape": workload["expected_field_shape"],
            "sum": 0.0,
            "l1": 2.0,
            "l2": 1.0,
            "maximum_absolute": 0.5,
            "source_updates": workload["expected_source_updates"],
            "meep_time": workload["expected_meep_time"],
        }
        snapshots = [self.valid_statistics(1), self.valid_statistics(2)]
        return {
            "schema_version": 2,
            "resolution": 1,
            "repeat": 2,
            "solver_seconds": [1.0, 1.1],
            "solver_seconds_median": 1.05,
            "metrics": [copy.deepcopy(metric), metric],
            "backend_decisions": [copy.deepcopy(decision), decision],
            "workload_contract": workload,
            "statistics_by_repeat": snapshots,
            "gpu_statistics": copy.deepcopy(snapshots[-1]),
            "parallelism_contract": "serial-single-rank-diagnostic",
            "performance_claim_eligible": False,
        }

    def test_invocation_hash_rejects_contract_tampering(self):
        contract = self.valid_contract()
        self.assertTrue(SITE._valid_invocation_hash(contract))
        contract["environment"]["OMP_NUM_THREADS"] = "16"
        self.assertFalse(SITE._valid_invocation_hash(contract))

    def test_registered_digest_cannot_be_replaced_by_result_self_hash(self):
        saved_invocation = SITE._REGISTERED_INVOCATION_SHA256
        saved_result = SITE._REGISTERED_RESULT_SHA256
        SITE._REGISTERED_INVOCATION_SHA256 = None
        SITE._REGISTERED_RESULT_SHA256 = None
        try:
            original_contract = self.valid_contract()
            original_result = b'{"solver_seconds":[1.0]}\n'
            original_result_hash = SITE.hashlib.sha256(
                original_result
            ).hexdigest()
            SITE.register_gpmeep_benchmark_evidence(
                original_contract["invocation_sha256"],
                original_result_hash,
            )
            self.assertTrue(
                SITE._registered_evidence_matches(
                    original_contract["invocation_sha256"],
                    original_result_hash,
                )
            )
            tampered_contract = copy.deepcopy(original_contract)
            tampered_contract["environment"]["OMP_NUM_THREADS"] = "16"
            unsigned = dict(tampered_contract)
            unsigned.pop("invocation_sha256")
            tampered_contract["invocation_sha256"] = SITE._canonical_sha256(
                unsigned
            )
            tampered_result_hash = SITE.hashlib.sha256(
                b'{"solver_seconds":[0.01]}\n'
            ).hexdigest()
            self.assertTrue(SITE._valid_invocation_hash(tampered_contract))
            self.assertFalse(
                SITE._registered_evidence_matches(
                    tampered_contract["invocation_sha256"],
                    tampered_result_hash,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "registered twice"):
                SITE.register_gpmeep_benchmark_evidence(
                    tampered_contract["invocation_sha256"],
                    tampered_result_hash,
                )
        finally:
            SITE._REGISTERED_INVOCATION_SHA256 = saved_invocation
            SITE._REGISTERED_RESULT_SHA256 = saved_result

    def test_result_shape_rejects_repeat_and_decision_tampering(self):
        result = self.valid_result()
        self.assertTrue(SITE._valid_benchmark_shape(result, "auto", "cpu"))
        wrong_repeat = copy.deepcopy(result)
        wrong_repeat["repeat"] = 3
        self.assertFalse(
            SITE._valid_benchmark_shape(wrong_repeat, "auto", "cpu")
        )
        split_decision = copy.deepcopy(result)
        split_decision["backend_decisions"][1]["active_backend"] = "cuda"
        self.assertFalse(
            SITE._valid_benchmark_shape(split_decision, "auto", "cpu")
        )
        false_speed_claim = copy.deepcopy(result)
        false_speed_claim["performance_claim_eligible"] = True
        self.assertFalse(
            SITE._valid_benchmark_shape(false_speed_claim, "auto", "cpu")
        )

    def test_result_shape_rejects_self_consistent_early_and_nonfinite_runs(self):
        early = self.valid_result()
        early["workload_contract"]["expected_meep_time"] = 1.0
        early["workload_contract"]["expected_source_updates"] = 2
        for metric in early["metrics"]:
            metric["meep_time"] = 1.0
            metric["source_updates"] = 2
        self.assertFalse(SITE._valid_benchmark_shape(early, "auto", "cpu"))

        nonfinite = self.valid_result()
        nonfinite["solver_seconds"][1] = float("nan")
        self.assertFalse(
            SITE._valid_benchmark_shape(nonfinite, "auto", "cpu")
        )

        changed_repeat = self.valid_result()
        changed_repeat["metrics"][1]["l2"] += 1.0
        self.assertFalse(
            SITE._valid_benchmark_shape(changed_repeat, "auto", "cpu")
        )

    def test_every_repeat_must_advance_native_statistics(self):
        result = self.valid_result()
        self.assertTrue(
            SITE._valid_statistics_by_repeat(result, "auto", "cpu")
        )
        result["statistics_by_repeat"][1] = copy.deepcopy(
            result["statistics_by_repeat"][0]
        )
        result["gpu_statistics"] = copy.deepcopy(
            result["statistics_by_repeat"][1]
        )
        self.assertFalse(
            SITE._valid_statistics_by_repeat(result, "auto", "cpu")
        )

    def test_stale_complete_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            complete = pathlib.Path(directory) / "COMPLETE"
            self.assertTrue(SITE._completion_path_is_fresh(False, complete))
            complete.write_text("stale\n", encoding="utf-8")
            self.assertFalse(SITE._completion_path_is_fresh(False, complete))
            complete.unlink()
            self.assertFalse(SITE._completion_path_is_fresh(True, complete))


class TestMovingSourceDevelopmentProducerEvidence(unittest.TestCase):
    def git(self, directory, *arguments):
        return subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=directory,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_complete_snapshot_detects_tracked_and_untracked_content(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            self.git(repo, "init", "-q")
            self.git(repo, "config", "user.email", "test@example.invalid")
            self.git(repo, "config", "user.name", "Test")
            tracked = repo / "tracked.cpp"
            untracked = repo / "untracked.cpp"
            tracked.write_text("tracked one\n", encoding="utf-8")
            self.git(repo, "add", "tracked.cpp")
            self.git(repo, "commit", "-qm", "fixture")
            untracked.write_text("untracked one\n", encoding="utf-8")
            initial = PRODUCER.source_snapshot(repo)

            untracked.write_text("untracked two\n", encoding="utf-8")
            changed_untracked = PRODUCER.source_snapshot(repo)
            self.assertNotEqual(initial, changed_untracked)

            untracked.write_text("untracked one\n", encoding="utf-8")
            tracked.write_text("tracked two\n", encoding="utf-8")
            changed_tracked = PRODUCER.source_snapshot(repo)
            self.assertNotEqual(initial, changed_tracked)

    def test_source_change_during_run_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "changed during benchmark"):
            PRODUCER.require_unchanged_source(
                {"entries_sha256": "a"}, {"entries_sha256": "b"}
            )

    def test_early_dependency_failure_removes_stale_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "result.json"
            output.write_text("stale success\n", encoding="utf-8")
            saved_loader = PRODUCER.load_benchmark_dependencies
            PRODUCER.load_benchmark_dependencies = lambda: (_ for _ in ()).throw(
                ImportError("injected early import failure")
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "requires mpi4py"):
                    PRODUCER.main(types.SimpleNamespace(output=output))
            finally:
                PRODUCER.load_benchmark_dependencies = saved_loader
            self.assertFalse(output.exists())

    def test_cli_argument_failure_removes_stale_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "result.json"
            output.write_text("stale success\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PRODUCER_PATH),
                    "--output",
                    str(output),
                    "--backend",
                    "cpu",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())

    def test_physical_core_affinity_normalizes_smt_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for logical_cpu in (0, 8):
                topology = root / f"cpu{logical_cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text("0\n", encoding="utf-8")
                (topology / "core_id").write_text("3\n", encoding="utf-8")
            affinity = PRODUCER.physical_core_affinity((0, 8), root)
            self.assertEqual(len(affinity), 1)
            self.assertEqual(affinity[0]["package_id"], 0)
            self.assertEqual(affinity[0]["core_id"], 3)

    def test_mapped_libmeep_record_hashes_loaded_file(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            library = directory / "libmeep.so.38"
            library.write_bytes(b"current native library")
            maps = directory / "maps"
            maps.write_text(
                f"7f00-7f10 r-xp 00000000 00:00 1 {library}\n",
                encoding="utf-8",
            )
            record = PRODUCER.mapped_libmeep_record(maps)
            self.assertEqual(record["path"], str(library.resolve()))
            self.assertEqual(record["sha256"], PRODUCER.sha256_file(library))


if __name__ == "__main__":
    unittest.main()
