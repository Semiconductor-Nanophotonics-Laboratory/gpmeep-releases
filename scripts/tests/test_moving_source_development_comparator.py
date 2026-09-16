import copy
import contextlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "compare-moving-source-mpi-development.py"
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_moving_source_comparator_test", MODULE_PATH
)
COMPARATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARATOR)


def artifact(path, marker):
    return {"path": path, "size_bytes": 100, "sha256": marker * 64}


class TestMovingSourceDevelopmentComparator(unittest.TestCase):
    def runtime(self):
        return {
            "python": artifact("/runtime/python", "1"),
            "extension": artifact("/runtime/_meep.so", "2"),
            "mapped_libmeep": artifact("/runtime/libmeep.so", "3"),
            "platform": ["Linux", "node", "release", "version", "x86_64", "x86_64"],
            "cpu_identity": {"vendor_id": "test", "model name": "test CPU"},
        }

    def statistics(self, backend, ranks, scale=1):
        result = []
        for _ in range(ranks):
            value = {
                "runtime": {
                    "runtime_availability_probes": 1 if backend == "cuda" else 0,
                    "runtime_device_enumerations": 1 if backend == "cuda" else 0,
                    "runtime_device_selections": 1 if backend == "cuda" else 0,
                },
                "multi_gpu": {
                    "mpi_messages": 0,
                    "mpi_scalars": 0,
                    "cuda_aware_bytes": 0,
                    "pinned_staging_bytes": 0,
                    "pinned_device_to_host_bytes": 0,
                    "pinned_host_to_device_bytes": 0,
                },
            }
            for group, suffixes in COMPARATOR.NATIVE_GROUPS.items():
                value[group] = {}
                for candidate in ("cpu", "cuda"):
                    for suffix in suffixes:
                        active = candidate == backend and group in (
                            "dispatch", "field_updates", "sources", "boundaries"
                        )
                        value[group][f"{candidate}_{suffix}"] = scale if active else 0
            result.append(value)
        return result

    def record(self, backend):
        ranks = 8 if backend == "cpu" else 1
        threads = 1 if backend == "cpu" else 8
        resolution = 1
        workload = COMPARATOR.expected_workload_contract(resolution)
        metric = {
            "shape": workload["expected_field_shape"],
            "sum": 0.0,
            "l1": 10.0,
            "l2": 2.0,
            "maximum_absolute": 1.0,
            "source_updates": workload["expected_source_updates"],
            "meep_time": workload["expected_meep_time"],
        }
        owner = {
            "selected_backend": backend,
            "process_backend": backend,
            "diagnostic": "test",
        }
        execution = []
        for rank in range(ranks):
            first_core = rank if backend == "cpu" else 0
            affinity = [
                {"host": "node", "package_id": 0, "core_id": core}
                for core in range(first_core, first_core + threads)
            ]
            execution.append(
                {
                    "rank": rank,
                    "physical_core_affinity": affinity,
                    "selected_device": -1 if backend == "cpu" else 0,
                    "selected_device_identifier": "" if backend == "cpu" else "GPU-test",
                }
            )
        statistics_by_repeat = [
            self.statistics(backend, ranks, repeat_index + 1)
            for repeat_index in range(2)
        ]
        rank_samples = []
        for repeat_index in range(2):
            rank_samples.append(
                [
                    {
                        "rank": rank,
                        "metric": dict(metric),
                        "owner": dict(owner),
                        "physical_core_affinity": copy.deepcopy(
                            execution[rank]["physical_core_affinity"]
                        ),
                        "local_seconds": 1.0,
                        "statistics": copy.deepcopy(
                            statistics_by_repeat[repeat_index][rank]
                        ),
                    }
                    for rank in range(ranks)
                ]
            )
        source_entries = [
            {
                "path": "source.cpp",
                "kind": "file",
                "mode": 0o644,
                "size_bytes": 3,
                "sha256": "c" * 64,
            }
        ]
        source_payload = json.dumps(
            source_entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        source = {
            "head": "abc",
            "index_sha256": "a" * 64,
            "entry_count": 1,
            "entries_sha256": COMPARATOR.sha256_bytes(source_payload),
            "entries": source_entries,
        }
        runtime = self.runtime()
        return {
            "schema_version": 2,
            "development_only": True,
            "authoritative_release_evidence": False,
            "performance_claim_eligible": False,
            "backend": backend,
            "resolution": resolution,
            "repeat": 2,
            "mpi_ranks": ranks,
            "omp_threads_per_rank": threads,
            "total_host_thread_budget": ranks * threads,
            "solver_seconds": [2.0, 2.2] if backend == "cpu" else [1.0, 1.1],
            "metrics": [dict(metric), dict(metric)],
            "global_work_totals": [
                {
                    "field_sample_count": 1600,
                    "source_updates": workload["expected_source_updates"],
                    "meep_time": workload["expected_meep_time"],
                },
                {
                    "field_sample_count": 1600,
                    "source_updates": workload["expected_source_updates"],
                    "meep_time": workload["expected_meep_time"],
                },
            ],
            "owners": [dict(owner), dict(owner)],
            "rank_samples": rank_samples,
            "statistics_by_rank": copy.deepcopy(statistics_by_repeat[-1]),
            "execution_by_rank": execution,
            "source_state": {"start": copy.deepcopy(source), "end": source},
            "workload_contract": workload,
            "runtime": runtime,
            "runtime_by_rank": [copy.deepcopy(runtime) for _ in range(ranks)],
            "environment": {
                "omp_num_threads": str(threads),
                "gpu_mpi_transport": "auto",
                "allow_oversubscribe": "0",
            },
        }

    def invoke_main(self, cpu, cuda, output, cpu_path, cuda_path):
        cpu_path.write_text(json.dumps(cpu), encoding="utf-8")
        cuda_path.write_text(json.dumps(cuda), encoding="utf-8")
        return COMPARATOR.main(
            types.SimpleNamespace(cpu=cpu_path, cuda=cuda_path, output=output)
        )

    def test_valid_pair_has_two_x_speedup(self):
        cpu = COMPARATOR.validate_record(self.record("cpu"), "cpu")
        cuda = COMPARATOR.validate_record(self.record("cuda"), "cuda")
        self.assertEqual(
            cpu["timing"]["median_seconds"] / cuda["timing"]["median_seconds"],
            2.0,
        )

    def test_valid_main_writes_all_repeat_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            output = directory / "comparison.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.invoke_main(
                    self.record("cpu"),
                    self.record("cuda"),
                    output,
                    directory / "cpu.json",
                    directory / "cuda.json",
                )
            comparison = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(comparison["speedup_cpu_over_cuda"], 2.0)
            self.assertEqual(
                len(comparison["scalar_metric_differences_by_repeat"]), 2
            )
            self.assertFalse(comparison["authoritative_release_evidence"])
            self.assertFalse(comparison["performance_claim_eligible"])

    def test_source_mismatch_removes_stale_output(self):
        cpu = self.record("cpu")
        cuda = self.record("cuda")
        cuda["source_state"]["start"]["head"] = "other"
        cuda["source_state"]["end"]["head"] = "other"
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            output = directory / "comparison.json"
            output.write_text("stale success\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different complete source"):
                self.invoke_main(
                    cpu, cuda, output, directory / "cpu.json", directory / "cuda.json"
                )
            self.assertFalse(output.exists())

    def test_input_load_failure_removes_stale_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            output = directory / "comparison.json"
            output.write_text("stale success\n", encoding="utf-8")
            cpu = directory / "cpu.json"
            cuda = directory / "cuda.json"
            cpu.write_text("not JSON", encoding="utf-8")
            cuda.write_text("{}", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                COMPARATOR.main(
                    types.SimpleNamespace(cpu=cpu, cuda=cuda, output=output)
                )
            self.assertFalse(output.exists())

    def test_cli_argument_failure_removes_stale_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "comparison.json"
            output.write_text("stale success\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--output",
                    str(output),
                    "--cpu",
                    str(pathlib.Path(directory) / "cpu.json"),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())

    def test_rejects_false_release_authority(self):
        record = self.record("cpu")
        record["authoritative_release_evidence"] = True
        with self.assertRaisesRegex(ValueError, "release authority"):
            COMPARATOR.validate_record(record, "cpu")

    def test_rejects_incomplete_rank_samples(self):
        record = self.record("cpu")
        record["rank_samples"][0].pop()
        with self.assertRaisesRegex(ValueError, "rank sample count"):
            COMPARATOR.validate_record(record, "cpu")

    def test_rejects_empty_native_statistics(self):
        record = self.record("cuda")
        record["statistics_by_rank"] = [{}]
        with self.assertRaisesRegex(ValueError, "native counter"):
            COMPARATOR.validate_record(record, "cuda")

    def test_rejects_native_cpu_fallback_in_cuda_run(self):
        record = self.record("cuda")
        record["statistics_by_rank"][0]["dispatch"]["cpu_curl_calls"] = 1
        with self.assertRaisesRegex(ValueError, "used cpu fallback"):
            COMPARATOR.validate_record(record, "cuda")

    def test_allows_only_live_resident_gauge_to_drop_after_reset(self):
        record = self.record("cuda")
        for repeat in record["rank_samples"]:
            repeat[0]["statistics"]["resident"] = {
                "device_buffer_allocations": 7,
                "live_resident_device_buffers": 3,
            }
        record["statistics_by_rank"][0]["resident"] = {
            "device_buffer_allocations": 7,
            "live_resident_device_buffers": 0,
        }
        COMPARATOR.validate_record(record, "cuda")

        record["statistics_by_rank"][0]["resident"][
            "device_buffer_allocations"
        ] = 8
        with self.assertRaisesRegex(ValueError, "cumulative statistics"):
            COMPARATOR.validate_record(record, "cuda")

    def test_rejects_overlapping_cpu_physical_cores(self):
        record = self.record("cpu")
        duplicate = copy.deepcopy(record["execution_by_rank"][0]["physical_core_affinity"])
        record["execution_by_rank"][1]["physical_core_affinity"] = duplicate
        for samples in record["rank_samples"]:
            samples[1]["physical_core_affinity"] = copy.deepcopy(duplicate)
        with self.assertRaisesRegex(ValueError, "overlap"):
            COMPARATOR.validate_record(record, "cpu")

    def test_rejects_changed_affinity_during_repeat(self):
        record = self.record("cpu")
        record["rank_samples"][1][0]["physical_core_affinity"][0]["core_id"] = 99
        with self.assertRaisesRegex(ValueError, "affinity changed"):
            COMPARATOR.validate_record(record, "cpu")

    def test_rejects_wrong_native_library(self):
        cpu = self.record("cpu")
        cuda = self.record("cuda")
        cuda["runtime"]["mapped_libmeep"]["sha256"] = "4" * 64
        cuda["runtime_by_rank"][0] = copy.deepcopy(cuda["runtime"])
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            with self.assertRaisesRegex(ValueError, "different Python/native"):
                self.invoke_main(
                    cpu,
                    cuda,
                    directory / "comparison.json",
                    directory / "cpu.json",
                    directory / "cuda.json",
                )

    def test_rejects_different_physical_core_budget(self):
        cpu = self.record("cpu")
        cuda = self.record("cuda")
        for item in cuda["execution_by_rank"][0]["physical_core_affinity"]:
            item["core_id"] += 8
        cuda["rank_samples"] = [
            [
                {
                    **sample,
                    "physical_core_affinity": copy.deepcopy(
                        cuda["execution_by_rank"][0]["physical_core_affinity"]
                    ),
                }
                for sample in repeat
            ]
            for repeat in cuda["rank_samples"]
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            with self.assertRaisesRegex(ValueError, "different physical-core"):
                self.invoke_main(
                    cpu,
                    cuda,
                    directory / "comparison.json",
                    directory / "cpu.json",
                    directory / "cuda.json",
                )

    def test_rejects_nan_and_repeat_metric_tampering(self):
        nonfinite = self.record("cpu")
        nonfinite["metrics"][0]["l2"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            COMPARATOR.validate_record(nonfinite, "cpu")

        changed_repeat = self.record("cpu")
        changed_repeat["metrics"][1]["l2"] = 3.0
        changed_repeat["rank_samples"][1] = [
            {**sample, "metric": copy.deepcopy(changed_repeat["metrics"][1])}
            for sample in changed_repeat["rank_samples"][1]
        ]
        with self.assertRaisesRegex(ValueError, "repeat changed"):
            COMPARATOR.validate_record(changed_repeat, "cpu")

    def test_rejects_unexpected_end_time_and_update_count(self):
        wrong_time = self.record("cpu")
        for repeat_index, metric in enumerate(wrong_time["metrics"]):
            metric["meep_time"] = 5.0
            for sample in wrong_time["rank_samples"][repeat_index]:
                sample["metric"]["meep_time"] = 5.0
        with self.assertRaisesRegex(ValueError, "unexpected Meep end time"):
            COMPARATOR.validate_record(wrong_time, "cpu")

        wrong_updates = self.record("cpu")
        for repeat_index, metric in enumerate(wrong_updates["metrics"]):
            metric["source_updates"] = 21
            for sample in wrong_updates["rank_samples"][repeat_index]:
                sample["metric"]["source_updates"] = 21
        with self.assertRaisesRegex(ValueError, "unexpected moving-source update"):
            COMPARATOR.validate_record(wrong_updates, "cpu")

    def test_rejects_self_consistent_early_workload_contract(self):
        record = self.record("cpu")
        record["workload_contract"]["expected_meep_time"] = 5.0
        record["workload_contract"]["expected_source_updates"] = 21
        for repeat_index, metric in enumerate(record["metrics"]):
            metric["meep_time"] = 5.0
            metric["source_updates"] = 21
            record["global_work_totals"][repeat_index]["meep_time"] = 5.0
            record["global_work_totals"][repeat_index]["source_updates"] = 21
            for sample in record["rank_samples"][repeat_index]:
                sample["metric"] = copy.deepcopy(metric)
        with self.assertRaisesRegex(ValueError, "independently derived"):
            COMPARATOR.validate_record(record, "cpu")

    def test_rejects_source_change_during_run(self):
        record = self.record("cpu")
        record["source_state"]["end"]["entries_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "source changed"):
            COMPARATOR.validate_record(record, "cpu")

    def test_loader_rejects_nonfinite_json_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "bad.json"
            path.write_text('{"metric": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                COMPARATOR.load_record(path)
            path.write_text('{"metric": 1, "metric": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                COMPARATOR.load_record(path)


if __name__ == "__main__":
    unittest.main()
