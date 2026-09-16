import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gpmeep-self-check.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_self_check", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def release_manifest(commit="a" * 40):
    return copy.deepcopy({**MODULE.RELEASE_MANIFEST_FIELDS, "source_commit": commit})


def statistics(backend="cpu", *, pinned=False, waitsome=False):
    cpu = backend == "cpu"
    cuda = backend == "cuda"
    return {
        "dispatch": {
            "cpu_curl_points": 120 if cpu else 0,
            "cuda_curl_points": 120 if cuda else 0,
        },
        "field_updates": {
            "cpu_update_eh_points": 80 if cpu else 0,
            "cuda_update_eh_points": 80 if cuda else 0,
        },
        "multi_gpu": {
            "pinned_staging_bytes": 64 if pinned else 0,
            "pinned_device_to_host_bytes": 32 if pinned else 0,
            "pinned_host_to_device_bytes": 32 if pinned else 0,
        },
        "mpi_completion": {
            "mpi_waitsome_executions": 4 if waitsome else 0,
            "mpi_waitall_executions": 0,
        },
    }


def record(backend, rank=0, world_size=1, identifier="GPU-A"):
    return {
        "rank": rank,
        "world_size": world_size,
        "lane": "cpu" if backend == "cpu" else ("gpu2" if world_size == 2 else "gpu1"),
        "backend": backend,
        "requested_backend": backend,
        "active_backend": backend,
        "cuda_execution_selected": backend == "cuda",
        "selected_device_identifier": identifier if backend == "cuda" else None,
        "statistics": statistics(
            backend,
            pinned=world_size == 2,
            waitsome=world_size == 2,
        ),
        "field_shape": [2, 2],
        "field_values": [0.0, 1.0, -1.0, 0.5],
        "flux_values": [0.25, 0.5],
    }


class SelfCheckContractTests(unittest.TestCase):
    def test_release_manifest_binds_v103_to_accepted_m3_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory)
            path = prefix / MODULE.RELEASE_MANIFEST
            path.parent.mkdir(parents=True)
            value = release_manifest()
            path.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(MODULE.sys, "prefix", str(prefix)):
                with mock.patch.object(
                    MODULE, "validate_cuda_architecture_audit"
                ), mock.patch.object(MODULE, "validate_glibc_compatibility_audit"):
                    result = MODULE._release_manifest()
            self.assertEqual(result["performance_anchor"], MODULE.SEALED_M3_ANCHOR)

    def test_release_manifest_rejects_schema_policy_and_anchor_mutations(self):
        mutations = {
            "missing_python_import": lambda value: value.pop("python_import"),
            "unexpected_key": lambda value: value.update(unexpected_claim=True),
            "wrong_cuda_toolkit": lambda value: value.update(cuda_toolkit="0.0"),
            "wrong_cuda_policy": lambda value: value.update(
                cuda_architectures="1-real"
            ),
            "empty_validated_architectures": lambda value: value.update(
                runtime_validated_architectures=[]
            ),
            "boolean_as_integer": lambda value: value["performance_anchor"].update(
                qualification_eligible=1
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory)
            path = prefix / MODULE.RELEASE_MANIFEST
            path.parent.mkdir(parents=True)
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    value = release_manifest()
                    mutate(value)
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with mock.patch.object(MODULE.sys, "prefix", str(prefix)):
                        with self.assertRaises(MODULE.SelfCheckError):
                            MODULE._release_manifest()

    def test_release_manifest_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = pathlib.Path(directory)
            path = prefix / MODULE.RELEASE_MANIFEST
            path.parent.mkdir(parents=True)
            raw = json.dumps(release_manifest()).replace(
                '"qualification_eligible": true',
                '"qualification_eligible": 1, "qualification_eligible": true',
            )
            path.write_text(raw, encoding="utf-8")
            with mock.patch.object(MODULE.sys, "prefix", str(prefix)):
                with self.assertRaisesRegex(MODULE.SelfCheckError, "duplicate JSON"):
                    MODULE._release_manifest()

    def test_release_manifest_rejects_intermediate_parent_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            prefix = root / "prefix"
            outside = root / "outside"
            (prefix / "share").mkdir(parents=True)
            outside.mkdir()
            (outside / "release.json").write_text(
                json.dumps(release_manifest()), encoding="utf-8"
            )
            (prefix / "share/gpmeep").symlink_to(
                outside, target_is_directory=True
            )
            with mock.patch.object(MODULE.sys, "prefix", str(prefix)):
                with self.assertRaisesRegex(MODULE.SelfCheckError, "unreadable"):
                    MODULE._release_manifest()

    def test_parse_marker_requires_exactly_one_record(self):
        value = MODULE._parse_marker("noise\n" + MODULE.MARKER + '{"state":"PASS"}\n')
        self.assertEqual(value["state"], "PASS")
        with self.assertRaises(MODULE.SelfCheckError):
            MODULE._parse_marker("noise only")
        with self.assertRaises(MODULE.SelfCheckError):
            MODULE._parse_marker(MODULE.MARKER + "{}\n" + MODULE.MARKER + "{}")

    def test_cpu_and_gpu_contracts_reject_fallback(self):
        MODULE._validate_cpu_record(record("cpu"))
        MODULE._validate_cuda_record(record("cuda"))
        bad = record("cuda")
        bad["statistics"]["dispatch"]["cpu_curl_points"] = 1
        with self.assertRaisesRegex(MODULE.SelfCheckError, "CPU fallback"):
            MODULE._validate_cuda_record(bad)

    def test_two_gpu_contract_requires_distinct_devices_and_traffic(self):
        value = {
            "schema_version": 1,
            "lane": "gpu2",
            "rank_records": [
                record("cuda", 0, 2, "GPU-A"),
                record("cuda", 1, 2, "GPU-B"),
            ],
        }
        self.assertEqual(len(MODULE._validate_lane(value, "gpu2")), 2)
        value["rank_records"][1]["selected_device_identifier"] = "GPU-A"
        with self.assertRaisesRegex(MODULE.SelfCheckError, "distinct"):
            MODULE._validate_lane(value, "gpu2")

    def test_numerical_comparison_is_fail_closed(self):
        cpu = record("cpu")
        gpu = record("cuda")
        result = MODULE._compare_observables(cpu, gpu)
        self.assertEqual(result["field_violations"], 0)
        gpu["field_values"][1] += 0.1
        with self.assertRaisesRegex(MODULE.SelfCheckError, "numerical"):
            MODULE._compare_observables(cpu, gpu)

    def test_device_specification_rejects_empty_and_duplicate_entries(self):
        self.assertEqual(MODULE._normalized_visible_devices("GPU-A,1"), ["GPU-A", "1"])
        for value in ("0,", "0,0"):
            with self.assertRaises(MODULE.SelfCheckError):
                MODULE._normalized_visible_devices(value)

    def test_report_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            MODULE._write_report(path, {"state": "PASS"})
            with self.assertRaisesRegex(MODULE.SelfCheckError, "overwrite"):
                MODULE._write_report(path, {"state": "PASS"})

    def test_public_receipt_omits_full_observable_vectors(self):
        lane = record("cuda")
        lane.update({"elapsed_seconds": 1.25, "timesteps": 320})
        report = {
            "state": "PASS",
            "mode": "gpu1",
            "report_id": "receipt-id",
            "wall_seconds": 2.0,
            "valid_for_speed_gate": False,
            "comparisons": {"cpu_vs_gpu1": {"field_violations": 0}},
            "lanes": {
                "gpu1": {
                    "rank_records": [lane],
                }
            },
        }
        receipt = MODULE._public_receipt(report)
        self.assertEqual(receipt["lanes"]["gpu1"]["rank_count"], 1)
        self.assertNotIn("field_values", str(receipt))

    def test_gpu2_launcher_uses_distributed_assignment_not_auto_ordinal(self):
        completed = mock.Mock(returncode=0, stdout=MODULE.MARKER + "{}\n", stderr="")
        with mock.patch.object(
            MODULE.shutil, "which", return_value="/prefix/bin/mpiexec"
        ):
            with mock.patch.object(
                MODULE.subprocess, "run", return_value=completed
            ) as run:
                MODULE._run_worker("gpu2", ["0", "1"], 30)
        environment = run.call_args.kwargs["env"]
        self.assertNotIn("MEEP_GPU_DEVICE", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0,1")


if __name__ == "__main__":
    unittest.main()
