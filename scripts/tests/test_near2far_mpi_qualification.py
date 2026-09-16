from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "run-near2far-mpi-qualification.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_n2f_mpi", SCRIPT)
assert SPEC and SPEC.loader
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def write_record(
    path: pathlib.Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    backend: str = "cpu",
    targets: int = 2,
    frequencies: int = 1,
    repetitions: int = 3,
) -> None:
    local_chunks = 2
    local_points = 11 + rank
    cuda = backend == "cuda"
    lines = [
        f"schema={qualification.RECORD_SCHEMA}",
        f"backend={backend}",
        f"rank={rank}",
        f"world_size={world_size}",
        f"selected_device={rank if cuda else -1}",
        f"device_identifier={'GPU-' + str(rank) if cuda else 'cpu'}",
        f"local_chunk_count={local_chunks}",
        f"local_source_points={local_points}",
        f"target_count={targets}",
        f"frequency_count={frequencies}",
        f"repetitions={repetitions}",
        "live_buffers_before=7",
        "live_buffers_after=7",
        f"cpu_transform_calls={0 if cuda else repetitions * targets}",
        f"cuda_transform_calls={repetitions if cuda else 0}",
        f"cuda_submitted_chunks={repetitions * local_chunks if cuda else 0}",
        f"cuda_source_points={repetitions * local_points if cuda else 0}",
        f"cuda_output_points={repetitions * targets if cuda else 0}",
        f"cuda_frequencies={repetitions * frequencies if cuda else 0}",
        f"cuda_periodic_copies={repetitions if cuda else 0}",
        f"cuda_fast_precision_calls={repetitions if cuda else 0}",
        "cuda_mixed_precision_calls=0",
        "cuda_cancellation_retries=0",
        f"cuda_target_tiles={repetitions if cuda else 0}",
        f"cuda_frequency_tiles={repetitions if cuda else 0}",
        f"cuda_operation_tiles={repetitions if cuda else 0}",
        f"cuda_maximum_workspace_bytes={4096 if cuda else 0}",
        "cuda_descriptor_uploads=0",
        f"cuda_kernel_launches={2 * repetitions if cuda else 0}",
        f"cuda_result_device_to_host_bytes="
        f"{repetitions * targets * frequencies * 12 * 8 if cuda else 0}",
        f"cuda_condition_device_to_host_bytes="
        f"{repetitions * targets * frequencies * 8 if cuda else 0}",
        f"dft_device_to_host_bytes_avoided={8192 if cuda else 0}",
        f"mpi_allreduce_calls={repetitions}",
        f"mpi_allreduce_bytes="
        f"{repetitions * targets * frequencies * (13 if cuda else 12) * 8}",
        "cpu_curl_calls=0",
        "cuda_curl_calls=0",
    ]
    for repetition in range(repetitions):
        lines.append(f"elapsed_{repetition}={(0.01 + repetition * 0.001).hex()}")
    for index in range(targets * frequencies * 6):
        lines.append(f"field_{index}_real={(index * 0.125).hex()}")
        lines.append(f"field_{index}_imag={(-index * 0.0625).hex()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Near2FarMpiQualificationTest(unittest.TestCase):
    def test_loaded_dependency_provenance_ignores_only_aslr_addresses(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            actual = repo / "gpu-step-db"
            actual.write_bytes(b"\x7fELFfixture")
            names = (
                "libmeep.so.38",
                "libcudart.so.12",
                "libmpi.so.40",
                "libhdf5.so.310",
            )
            libraries = []
            for name in names:
                library = repo / name
                library.write_bytes(name.encode("ascii"))
                libraries.append(library)
            loader = repo / "ld-linux-x86-64.so.2"
            loader.write_bytes(b"dynamic-loader")
            stdout = "".join(
                f"{name} => {path} (0x0000{index + 1}000)\n"
                for index, (name, path) in enumerate(zip(names, libraries))
            )
            stdout += f"{loader} (0x00009999000)\n"
            stdout = stdout.encode("utf-8")
            first = {
                "command": ["ldd"],
                "exit_code": 0,
                "stdout": stdout,
                "stderr": b"",
                "timeout": False,
                "output_limit": None,
                "error": None,
            }
            second = dict(first)
            second["stdout"] = stdout.replace(b"0x0000", b"0x9999")
            alternate = repo / "libmeep-alt.so.38"
            alternate.write_bytes(libraries[0].read_bytes())
            wrong = dict(first)
            wrong["stdout"] = stdout.replace(
                str(libraries[0]).encode("utf-8"),
                str(alternate).encode("utf-8"),
            )
            records = []
            with mock.patch.object(
                qualification.provenance,
                "bounded_command",
                side_effect=[first, second, wrong, first],
            ):
                records.append(
                    qualification.loaded_dependency_records(actual, repo)
                )
                records.append(
                    qualification.loaded_dependency_records(actual, repo)
                )
                records.append(
                    qualification.loaded_dependency_records(actual, repo)
                )
                loader.write_bytes(b"changed-dynamic-loader")
                records.append(
                    qualification.loaded_dependency_records(actual, repo)
                )
        self.assertEqual(records[0], records[1])
        self.assertNotEqual(records[0], records[2])
        self.assertNotEqual(
            records[0]["dynamic_interpreter"],
            records[3]["dynamic_interpreter"],
        )

    def test_resolve_runtime_executable_requires_direct_elf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            wrapper = root / "gpu-step-db"
            wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
            wrapper.chmod(0o755)
            with self.assertRaisesRegex(
                qualification.QualificationError, "direct ELF"
            ):
                qualification.resolve_runtime_executable(wrapper)
            actual = root / "lt-gpu-step-db"
            actual.write_bytes(b"\x7fELFfixture")
            actual.chmod(0o755)
            expected = qualification.file_record(actual)
            runtime = qualification.resolve_runtime_executable(actual)
        self.assertEqual(runtime["actual_elf"], expected)
        self.assertEqual(runtime["execution_contract"], "direct-elf-v1")

    def test_release_provenance_requires_build_receipt_before_probing(self):
        with self.assertRaisesRegex(
            qualification.QualificationError, "requires --build-receipt"
        ):
            qualification.capture_runtime_provenance(
                pathlib.Path("missing"),
                pathlib.Path("."),
                None,
                "release",
            )

    def test_release_provenance_rejects_missing_relative_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            executable = repo / "gpu-step-db"
            executable.write_bytes(b"\x7fELFfixture")
            executable.chmod(0o755)
            with self.assertRaisesRegex(
                qualification.QualificationError,
                "build receipt verification failed",
            ):
                qualification.capture_runtime_provenance(
                    executable,
                    repo,
                    pathlib.Path("missing-receipt.json"),
                    "release",
                )

    def test_release_runtime_is_bound_to_receipt_elf_and_libmeep(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            executable = repo / "lt-gpu-step-db"
            executable.write_bytes(b"\x7fELFfixture")
            library = repo / "libmeep.so.38"
            library.write_bytes(b"libmeep")
            executable_record = qualification.file_record(executable)
            library_record = qualification.file_record(library)
            runtime = {"actual_elf": executable_record}
            dependencies = {
                "libraries": {"libmeep.so.38": library_record}
            }
            receipt = {
                "artifacts": {
                    "gpu_step_db_test": executable_record,
                    "libmeep": library_record,
                }
            }
            bound = qualification.require_release_runtime_binding(
                runtime, dependencies, receipt, repo
            )
            self.assertEqual(bound["gpu_step_db_test"], executable_record)
            alternate = repo / "alternate-gpu-step-db"
            alternate.write_bytes(executable.read_bytes())
            wrong_runtime = {
                "actual_elf": qualification.file_record(alternate)
            }
            with self.assertRaisesRegex(
                qualification.QualificationError, "Near2Far ELF"
            ):
                qualification.require_release_runtime_binding(
                    wrong_runtime, dependencies, receipt, repo
                )
            alternate_library = repo / "alternate-libmeep.so.38"
            alternate_library.write_bytes(library.read_bytes())
            wrong_dependencies = {
                "libraries": {
                    "libmeep.so.38": qualification.file_record(
                        alternate_library
                    )
                }
            }
            with self.assertRaisesRegex(
                qualification.QualificationError, "loaded libmeep"
            ):
                qualification.require_release_runtime_binding(
                    runtime, wrong_dependencies, receipt, repo
                )

    def test_release_runtime_accepts_relative_receipt_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            executable = repo / "build" / "lt-gpu-step-db"
            executable.parent.mkdir()
            executable.write_bytes(b"\x7fELFfixture")
            executable.chmod(0o755)
            library = repo / "install" / "libmeep.so.38"
            library.parent.mkdir()
            library.write_bytes(b"libmeep")

            runtime = {"actual_elf": qualification.file_record(executable)}
            dependencies = {
                "libraries": {
                    "libmeep.so.38": qualification.provenance.file_record(
                        library, repo
                    )
                }
            }
            receipt = {
                "artifacts": {
                    "gpu_step_db_test": (
                        qualification.provenance.file_record(executable, repo)
                    ),
                    "libmeep": qualification.provenance.file_record(
                        library, repo
                    ),
                }
            }

            bound = qualification.require_release_runtime_binding(
                runtime, dependencies, receipt, repo
            )
            self.assertEqual(
                bound["gpu_step_db_test"],
                receipt["artifacts"]["gpu_step_db_test"],
            )

            receipt["artifacts"]["gpu_step_db_test"] = dict(
                receipt["artifacts"]["gpu_step_db_test"],
                sha256="0" * 64,
            )
            with self.assertRaisesRegex(
                qualification.QualificationError, "Near2Far ELF"
            ):
                qualification.require_release_runtime_binding(
                    runtime, dependencies, receipt, repo
                )

    def test_release_receipt_requires_exact_build_kind_and_contract(self):
        valid = {
            "build_kind": qualification.RELEASE_BUILD_KIND,
            "configuration": {
                "qualification_contract": (
                    qualification.RELEASE_QUALIFICATION_CONTRACT
                )
            },
        }
        qualification.require_release_receipt_contract(valid)
        wrong_kind = dict(valid, build_kind="generic")
        with self.assertRaisesRegex(
            qualification.QualificationError, "wrong build kind"
        ):
            qualification.require_release_receipt_contract(wrong_kind)
        wrong_contract = dict(valid)
        wrong_contract["configuration"] = {
            "qualification_contract": "generic"
        }
        with self.assertRaisesRegex(
            qualification.QualificationError, "wrong qualification contract"
        ):
            qualification.require_release_receipt_contract(wrong_contract)

    def test_release_speed_gates_cannot_be_weakened(self):
        defaults = argparse.Namespace(
            qualification_tier="release",
            small_minimum_speedup=qualification.RELEASE_SMALL_MINIMUM_SPEEDUP,
            batch_minimum_speedup=qualification.RELEASE_BATCH_MINIMUM_SPEEDUP,
            batch_minimum_two_gpu_scaling=(
                qualification.RELEASE_BATCH_MINIMUM_TWO_GPU_SCALING
            ),
        )
        qualification.validate_release_speed_gates(defaults)
        defaults.batch_minimum_speedup = 0.1
        with self.assertRaisesRegex(
            qualification.QualificationError, "cannot be lower"
        ):
            qualification.validate_release_speed_gates(defaults)

    def test_gpu_inventory_is_bounded(self):
        completed = {
            "command": ["nvidia-smi"],
            "exit_code": 0,
            "stdout": (
                b"0, GPU-00000000-0000-0000-0000-000000000000, "
                b"RTX test, 999.1, 8.6, 24564, 0, 0\n"
                b"1, GPU-11111111-1111-1111-1111-111111111111, "
                b"RTX test, 999.1, 8.6, 24564, 1, 2\n"
            ),
            "stderr": b"",
            "timeout": False,
            "output_limit": None,
            "error": None,
        }
        with (
            mock.patch.object(qualification.shutil, "which", return_value="/nvidia-smi"),
            mock.patch.object(
                qualification.provenance,
                "bounded_command",
                return_value=completed,
            ) as bounded,
        ):
            devices = qualification.inventory_gpus()
        self.assertEqual(
            [item["uuid"] for item in devices],
            [
                "GPU-00000000-0000-0000-0000-000000000000",
                "GPU-11111111-1111-1111-1111-111111111111",
            ],
        )
        self.assertEqual(bounded.call_args.kwargs["timeout_seconds"], 30.0)
        self.assertEqual(bounded.call_args.kwargs["stdout_limit"], 1024 * 1024)

    def test_parse_record_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "rank.txt"
            write_record(path, backend="cuda", targets=3, frequencies=2)
            record = qualification.parse_record(path)
        self.assertEqual(record["backend"], "cuda")
        self.assertEqual(record["cuda_transform_calls"], 3)
        self.assertEqual(len(record["fields"]), 36)
        self.assertEqual(len(record["elapsed"]), 3)

    def test_parse_record_rejects_duplicate_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "rank.txt"
            write_record(path)
            path.write_text(
                path.read_text(encoding="utf-8") + "rank=0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                qualification.QualificationError, "duplicate evidence key"
            ):
                qualification.parse_record(path)

    def test_parse_record_rejects_dimensions_before_materializing_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "rank.txt"
            write_record(path)
            text = path.read_text(encoding="utf-8").replace(
                "repetitions=3",
                f"repetitions={qualification.MAX_EVIDENCE_REPETITIONS + 1}",
            )
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(
                qualification.QualificationError, "exceed verifier bounds"
            ):
                qualification.parse_record(path)

    def test_bounded_control_reader_rejects_fifo_and_oversized_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            fifo = root / "control.fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(
                qualification.QualificationError, "direct regular file"
            ):
                qualification._read_json_object(fifo, "fixture", 1024)
            oversized = root / "oversized.json"
            with oversized.open("wb") as handle:
                handle.truncate(1025)
            with self.assertRaisesRegex(
                qualification.QualificationError, "size bound"
            ):
                qualification._read_json_object(oversized, "fixture", 1024)
            empty = root / "empty.txt"
            empty.touch()
            false_size = qualification.file_record(empty)
            false_size["size_bytes"] = False
            with self.assertRaisesRegex(
                qualification.QualificationError, "file record is invalid"
            ):
                qualification._require_file_record_at(
                    false_size, empty, "empty fixture"
                )

    def test_release_raw_bounds_check_actual_files_before_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            raw = output / "raw"
            raw.mkdir()
            for index in range(qualification.RELEASE_RAW_FILE_COUNT):
                path = raw / f"record-{index}.txt"
                path.touch()
            manifest = qualification.provenance.tree_manifest(raw, output)
            false_counts = dict(manifest)
            false_counts["directory_count"] = False
            with self.assertRaisesRegex(
                qualification.QualificationError,
                "release raw manifest inventory differs",
            ):
                qualification._validate_release_raw_manifest_bounds(
                    raw, false_counts
                )
            with (raw / "record-0.txt").open("r+b") as handle:
                handle.truncate(qualification.MAX_EVIDENCE_FILE_BYTES + 1)
            with self.assertRaisesRegex(
                qualification.QualificationError,
                "bounded flat regular-file directory",
            ):
                qualification._validate_release_raw_manifest_bounds(raw, manifest)

    def test_release_raw_file_count_matches_execution_contract(self):
        self.assertEqual(68, qualification.RELEASE_RAW_FILE_COUNT)
        self.assertEqual(
            qualification.RELEASE_RAW_FILE_COUNT,
            qualification.RELEASE_MEASURED_SCENARIO_COUNT
            * sum(
                2 + ranks
                for ranks in qualification.RELEASE_MEASURED_LANE_RANK_COUNTS
            )
            + 2 * len(qualification.RELEASE_COLLECTIVE_PROBES)
            + 2,
        )

    def test_symmetric_error_reports_known_difference(self):
        error = qualification.symmetric_error(
            [[1.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 2.1]],
        )
        self.assertAlmostEqual(error["nrmse"], 0.1 / (5.0**0.5))
        self.assertAlmostEqual(error["normalized_max"], 0.05)

    def test_exact_json_equality_rejects_bool_int_and_int_float_aliases(self):
        self.assertTrue(
            qualification.exact_json_equal(
                {"a": [1, 2.0, False]}, {"a": [1, 2.0, False]}
            )
        )
        self.assertFalse(qualification.exact_json_equal(False, 0))
        self.assertFalse(qualification.exact_json_equal(1, 1.0))
        self.assertFalse(
            qualification.exact_json_equal({"a": [1]}, {"a": [True]})
        )

    def test_numeric_gate_compares_two_gpu_directly_to_cpu(self):
        cpu = [[1.0, 0.0]]
        one_gpu = [[1.0004, 0.0]]
        two_gpu = [[1.0008, 0.0]]
        with self.assertRaisesRegex(
            qualification.QualificationError, "2-GPU result differs from CPU"
        ):
            qualification.validate_numeric_agreement(
                "fixture", cpu, cpu, one_gpu, two_gpu
            )

    def test_gpu_uuid_gate_binds_one_and_two_gpu_lanes_to_inventory(self):
        devices = [
            {"index": "0", "uuid": "GPU-a"},
            {"index": "1", "uuid": "GPU-b"},
        ]
        one_gpu = {
            "records": [{"selected_device": 0, "device_identifier": "GPU-a"}]
        }
        two_gpu = {
            "records": [
                {"selected_device": 0, "device_identifier": "GPU-a"},
                {"selected_device": 1, "device_identifier": "GPU-b"},
            ]
        }
        qualification.validate_gpu_identity(
            "fixture", one_gpu, two_gpu, devices
        )
        one_gpu["records"][0]["device_identifier"] = "GPU-c"
        with self.assertRaisesRegex(
            qualification.QualificationError, "inventory-bound UUID"
        ):
            qualification.validate_gpu_identity(
                "fixture", one_gpu, two_gpu, devices
            )

    def test_gpu_inventory_must_be_identical_after_qualification(self):
        initial = [
            {"uuid": "GPU-a", "utilization_percent": "0"},
            {"uuid": "GPU-b", "utilization_percent": "0"},
        ]
        qualification.validate_terminal_gpu_inventory(initial, list(initial))
        changed = [dict(record) for record in initial]
        changed[1]["utilization_percent"] = "1"
        with self.assertRaisesRegex(
            qualification.QualificationError, "GPU inventory changed"
        ):
            qualification.validate_terminal_gpu_inventory(initial, changed)

    def test_release_inventory_rejects_opaque_gpu_and_impossible_cpu(self):
        opaque = [
            {
                "index": str(index),
                "uuid": f"device-{index}",
                "name": "GPU",
                "driver": "1",
                "compute_capability": "8.6",
                "memory_mib": "1",
                "memory_used_mib": "0",
                "utilization_percent": "0",
            }
            for index in range(2)
        ]
        with self.assertRaisesRegex(
            qualification.QualificationError, "malformed physical UUID"
        ):
            qualification.validate_gpu_inventory_records(opaque)
        coerced = [
            {
                "index": index == 1,
                "uuid": f"GPU-{index:032x}",
                "name": "GPU",
                "driver": "1",
                "compute_capability": "8.6",
                "memory_mib": True,
                "memory_used_mib": False,
                "utilization_percent": False,
            }
            for index in range(2)
        ]
        with self.assertRaisesRegex(
            qualification.QualificationError, "exact nvidia-smi strings"
        ):
            qualification.validate_gpu_inventory_records(coerced)
        impossible = {
            "model": "CPU",
            "logical_cpu_count": -5,
            "available_affinity_cpu_count": 2,
            "available_affinity_cpus": ["fake0", "fake1"],
        }
        with self.assertRaisesRegex(
            qualification.QualificationError, "cannot support"
        ):
            qualification.validate_cpu_inventory_record(impossible, 2)

    def test_evidence_bundle_seals_reports_and_complete_raw_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            output = repo / "evidence"
            raw = output / "raw"
            raw.mkdir(parents=True)
            raw_file = raw / "lane.stdout"
            raw_file.write_bytes(b"PASS\n")
            report = {
                "schema": qualification.SCHEMA,
                "state": "complete",
                "gpu_inventory": [{"uuid": "GPU-a"}, {"uuid": "GPU-b"}],
                "runtime_provenance": {
                    "qualification_tier": "development",
                    "build_receipt": None,
                },
                "scenarios": [],
                "raw_fixture": qualification.file_record(raw_file),
            }
            qualification.write_report(output, report, repo)
            manifest = qualification.verify_evidence_bundle(output, repo)
            complete = json.loads(
                (output / "COMPLETE").read_text(encoding="utf-8")
            )
            self.assertEqual(complete["bundle_id"], manifest["bundle_id"])
            self.assertEqual(manifest["raw_manifest"]["file_count"], 1)
            self.assertIsNone(manifest["build_receipt"])
            raw_file.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(
                qualification.QualificationError, "sealed raw evidence changed"
            ):
                qualification.verify_evidence_bundle(output, repo)

    def test_evidence_bundle_rejects_complete_seal_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            output = repo / "evidence"
            raw = output / "raw"
            raw.mkdir(parents=True)
            raw_file = raw / "lane.stdout"
            raw_file.write_bytes(b"PASS\n")
            report = {
                "schema": qualification.SCHEMA,
                "state": "complete",
                "gpu_inventory": [{"uuid": "GPU-a"}, {"uuid": "GPU-b"}],
                "runtime_provenance": {
                    "qualification_tier": "development",
                    "build_receipt": None,
                },
                "scenarios": [],
                "raw_fixture": qualification.file_record(raw_file),
            }
            qualification.write_report(output, report, repo)
            complete_path = output / "COMPLETE"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["bundle_id"] = "0" * 64
            complete_path.write_text(
                json.dumps(complete), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                qualification.QualificationError, "COMPLETE seal differs"
            ):
                qualification.verify_evidence_bundle(output, repo)

    def test_verification_cli_does_not_treat_generation_tier_as_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            output = repo / "evidence"
            raw = output / "raw"
            raw.mkdir(parents=True)
            raw_file = raw / "lane.stdout"
            raw_file.write_bytes(b"PASS\n")
            report = {
                "schema": qualification.SCHEMA,
                "state": "complete",
                "gpu_inventory": [{"uuid": "GPU-a"}, {"uuid": "GPU-b"}],
                "runtime_provenance": {
                    "qualification_tier": "development",
                    "build_receipt": None,
                },
                "scenarios": [],
                "raw_fixture": qualification.file_record(raw_file),
            }
            qualification.write_report(output, report, repo)
            process = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--verify-evidence",
                    str(output),
                    "--qualification-tier",
                    "release",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            self.assertEqual(process.returncode, 4)
            self.assertIn("--expected-qualification-tier", process.stderr)
            process = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--verify-evidence",
                    str(output),
                    "--expected-qualification-tier",
                    "release",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            self.assertEqual(process.returncode, 4)
            self.assertIn("requires a bundle ID and attestation", process.stderr)

    def test_evidence_manifest_rejects_extra_resealed_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            output = repo / "evidence"
            raw = output / "raw"
            raw.mkdir(parents=True)
            raw_file = raw / "lane.stdout"
            raw_file.write_bytes(b"PASS\n")
            report = {
                "schema": qualification.SCHEMA,
                "state": "complete",
                "gpu_inventory": [{"uuid": "GPU-a"}, {"uuid": "GPU-b"}],
                "runtime_provenance": {
                    "qualification_tier": "development",
                    "build_receipt": None,
                },
                "scenarios": [],
                "raw_fixture": qualification.file_record(raw_file),
            }
            qualification.write_report(output, report, repo)
            manifest_path = output / qualification.BUNDLE_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("bundle_id")
            manifest["bogus_certified_claim"] = True
            manifest["bundle_id"] = qualification.provenance.canonical_sha256(
                manifest
            )
            qualification.provenance.atomic_write_json(manifest_path, manifest)
            qualification.provenance.atomic_write_json(
                output / "COMPLETE",
                {
                    "schema": qualification.BUNDLE_SCHEMA,
                    "state": "complete",
                    "bundle_id": manifest["bundle_id"],
                    "evidence_manifest": qualification.provenance.file_record(
                        manifest_path, output
                    ),
                },
            )
            with self.assertRaisesRegex(
                qualification.QualificationError,
                "evidence manifest identity is invalid",
            ):
                qualification.verify_evidence_bundle(output, repo)

    def test_release_bundle_outer_seal_binds_receipt_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            output = repo / "evidence"
            raw = output / "raw"
            raw.mkdir(parents=True)
            raw_file = raw / "lane.stdout"
            raw_file.write_bytes(b"PASS\n")
            executable = repo / "lt-gpu-step-db"
            executable.write_bytes(b"\x7fELFfixture")
            library = repo / "libmeep.so.38"
            library.write_bytes(b"libmeep")
            receipt_path = repo / "build-provenance.json"
            receipt_path.write_text("{}\n", encoding="utf-8")
            executable_record = qualification.file_record(executable)
            library_record = qualification.file_record(library)
            verified_receipt = {
                "build_kind": qualification.RELEASE_BUILD_KIND,
                "receipt_id": "receipt-id",
                "build_input_id": "build-input-id",
                "artifact_set_id": "artifact-set-id",
                "git_head": "fixture-head",
                "git_status_porcelain": [],
                "source_end": {
                    "sha256": "fixture-source",
                    "source_manifest_sha256": "fixture-manifest",
                },
                "configuration": {
                    "qualification_contract": (
                        qualification.RELEASE_QUALIFICATION_CONTRACT
                    )
                },
                "artifacts": {
                    "gpu_step_db_test": executable_record,
                    "libmeep": library_record,
                },
            }
            report = {
                "schema": qualification.SCHEMA,
                "state": "complete",
                "gpu_inventory": [{"uuid": "GPU-a"}, {"uuid": "GPU-b"}],
                "runtime_provenance": {
                    "qualification_tier": "release",
                    "runtime": {"actual_elf": executable_record},
                    "loaded_dependencies": {
                        "libraries": {"libmeep.so.38": library_record}
                    },
                    "git_head": "fixture-head",
                    "git_status_porcelain": [],
                    "source_snapshot": {
                        "sha256": "fixture-source",
                        "source_manifest_sha256": "fixture-manifest",
                    },
                    "build_receipt": {
                        "file": qualification.file_record(receipt_path),
                        "receipt_id": "receipt-id",
                        "build_input_id": "build-input-id",
                        "artifact_set_id": "artifact-set-id",
                        "bound_artifacts": {
                            "gpu_step_db_test": executable_record,
                            "libmeep": library_record,
                        },
                    },
                },
                "scenarios": [],
                "raw_fixture": qualification.file_record(raw_file),
            }
            raw_bound_patch = mock.patch.object(
                qualification, "_validate_release_raw_manifest_bounds"
            )
            bounded_tree_patch = mock.patch.object(
                qualification,
                "_bounded_release_raw_tree_manifest",
                side_effect=lambda _raw, _repo, manifest: manifest,
            )
            raw_bound_patch.start()
            bounded_tree_patch.start()
            self.addCleanup(raw_bound_patch.stop)
            self.addCleanup(bounded_tree_patch.stop)
            with (
                mock.patch.object(
                    qualification.provenance,
                    "verify_build_receipt",
                    return_value=verified_receipt,
                ),
                mock.patch.object(qualification, "validate_report_semantics"),
            ):
                manifest = qualification.write_report(output, report, repo)
                with self.assertRaisesRegex(
                    qualification.QualificationError,
                    "external authority inputs",
                ):
                    qualification.verify_evidence_bundle(output, repo)
                attestation_path = repo / "release-attestation.json"
                qualification.write_release_attestation(
                    attestation_path, output, report, manifest
                )
                with self.assertRaisesRegex(
                    qualification.QualificationError,
                    "externally pinned bundle ID",
                ):
                    qualification.verify_evidence_bundle(
                        output,
                        repo,
                        expected_bundle_id="0" * 64,
                        release_attestation=attestation_path,
                    )
                manifest = qualification.verify_evidence_bundle(
                    output,
                    repo,
                    expected_bundle_id=manifest["bundle_id"],
                    release_attestation=attestation_path,
                )
                trusted_bundle_id = manifest["bundle_id"]
            self.assertEqual(
                manifest["build_receipt"],
                qualification.file_record(receipt_path),
            )
            receipt_path.write_text('{"tampered": true}\n', encoding="utf-8")
            with (
                mock.patch.object(qualification, "validate_report_semantics"),
                self.assertRaisesRegex(
                    qualification.QualificationError,
                    "sealed build receipt file changed",
                ),
            ):
                qualification.verify_evidence_bundle(
                    output,
                    repo,
                    expected_bundle_id=manifest["bundle_id"],
                    release_attestation=attestation_path,
                )
            receipt_path.write_text("{}\n", encoding="utf-8")
            downgraded_report = json.loads(
                (output / "report.json").read_text(encoding="utf-8")
            )
            downgraded_report["runtime_provenance"][
                "qualification_tier"
            ] = "development"
            (output / "report.json").write_text(
                json.dumps(downgraded_report) + "\n", encoding="utf-8"
            )
            (output / "report.md").write_text(
                qualification._report_markdown(downgraded_report),
                encoding="utf-8",
            )
            downgraded_manifest = dict(manifest)
            downgraded_manifest.pop("bundle_id")
            downgraded_manifest["qualification_tier"] = "development"
            downgraded_manifest["files"] = {
                "report_json": qualification.provenance.file_record(
                    output / "report.json", output
                ),
                "report_markdown": qualification.provenance.file_record(
                    output / "report.md", output
                ),
            }
            downgraded_bundle_id = qualification.provenance.canonical_sha256(
                downgraded_manifest
            )
            downgraded_manifest["bundle_id"] = downgraded_bundle_id
            qualification.provenance.atomic_write_json(
                output / qualification.BUNDLE_MANIFEST_NAME,
                downgraded_manifest,
            )
            qualification.provenance.atomic_write_json(
                output / "COMPLETE",
                {
                    "schema": qualification.BUNDLE_SCHEMA,
                    "state": "complete",
                    "bundle_id": downgraded_bundle_id,
                    "evidence_manifest": qualification.provenance.file_record(
                        output / qualification.BUNDLE_MANIFEST_NAME, output
                    ),
                },
            )
            with self.assertRaisesRegex(
                qualification.QualificationError,
                "externally pinned bundle ID",
            ):
                qualification.verify_evidence_bundle(
                    output,
                    repo,
                    expected_bundle_id=trusted_bundle_id,
                    release_attestation=attestation_path,
                )

    def test_release_bundle_rejects_resealed_empty_semantic_inventory(self):
        """Outer hashes are not authority for empty release claims."""
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            output = repo / "evidence"
            raw = output / "raw"
            raw.mkdir(parents=True)
            raw_file = raw / "lane.stdout"
            raw_file.write_bytes(b"PASS\n")
            report = {
                "schema": qualification.SCHEMA,
                "state": "complete",
                "gpu_inventory": [{"uuid": "GPU-a"}, {"uuid": "GPU-b"}],
                "runtime_provenance": {
                    "qualification_tier": "release",
                    "build_receipt": None,
                },
                "scenarios": [],
                "lanes": [],
                "collective_safety_probes": [],
                "raw_fixture": qualification.file_record(raw_file),
            }
            raw_bound_patch = mock.patch.object(
                qualification, "_validate_release_raw_manifest_bounds"
            )
            bounded_tree_patch = mock.patch.object(
                qualification,
                "_bounded_release_raw_tree_manifest",
                side_effect=lambda _raw, _repo, manifest: manifest,
            )
            raw_bound_patch.start()
            bounded_tree_patch.start()
            self.addCleanup(raw_bound_patch.stop)
            self.addCleanup(bounded_tree_patch.stop)
            # Emulate an attacker who rewrites the report and recomputes every
            # unhashed outer manifest/COMPLETE field.
            with (
                mock.patch.object(qualification, "validate_report_semantics"),
                mock.patch.object(qualification, "verify_evidence_bundle"),
            ):
                manifest = qualification.write_report(output, report, repo)
            with self.assertRaisesRegex(
                qualification.QualificationError,
                "release runtime-provenance inventory differs",
            ):
                qualification.verify_evidence_bundle(
                    output,
                    repo,
                    expected_bundle_id=manifest["bundle_id"],
                    release_attestation=repo / "missing-attestation.json",
                )

    def test_run_lane_validates_rank_local_cuda_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            executable = root / "gpu-step-db"
            executable.write_text("fixture\n", encoding="utf-8")
            executable.chmod(0o755)
            raw = root / "raw"
            raw.mkdir()

            def fake_bounded(
                command,
                *,
                cwd,
                environment,
                timeout_seconds,
                stdout_limit,
                stderr_limit,
            ):
                del cwd, timeout_seconds, stdout_limit, stderr_limit
                ranks = int(command[command.index("-np") + 1])
                prefix = pathlib.Path(
                    environment["MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_WRITE"]
                )
                for rank in range(ranks):
                    write_record(
                        pathlib.Path(f"{prefix}.rank-{rank}.txt"),
                        rank=rank,
                        world_size=ranks,
                        backend=environment[
                            "MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_BACKEND"
                        ],
                        targets=int(
                            environment["MEEP_GPU_TEST_NEAR2FAR_TARGET_COUNT"]
                        ),
                        frequencies=int(
                            environment["MEEP_GPU_TEST_NEAR2FAR_FREQUENCY_COUNT"]
                        ),
                        repetitions=int(
                            environment["MEEP_GPU_TEST_NEAR2FAR_REPETITIONS"]
                        ),
                    )
                return {
                    "command": command,
                    "exit_code": 0,
                    "stdout": b"PASS\n",
                    "stderr": b"",
                    "timeout": False,
                    "output_limit": None,
                    "error": None,
                }

            with mock.patch.object(
                qualification.provenance,
                "bounded_command",
                side_effect=fake_bounded,
            ):
                lane = qualification.run_lane(
                    executable,
                    raw,
                    "cuda-2",
                    "cuda",
                    2,
                    1,
                    2,
                    1,
                    3,
                )
        self.assertEqual(lane["ranks"], 2)
        self.assertEqual(len(lane["records"]), 2)
        self.assertEqual(lane["records"][1]["selected_device"], 1)

    def test_rank_local_failure_probe_requires_both_ranks_and_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            executable = root / "gpu-step-db"
            executable.write_text("fixture\n", encoding="utf-8")
            raw = root / "raw"
            raw.mkdir()
            diagnostic = (
                "gpmeep-near2far-abort-probe:rank=0,"
                "stage=before-rank-local-cuda-failure\n"
                "gpmeep-near2far-abort-probe:rank=1,"
                "stage=before-rank-local-cuda-failure\n"
                "rank-local failure during distributed near-to-far "
                "processing\n"
            )
            completed = {
                "command": ["mpirun"],
                "exit_code": 17,
                "stdout": b"",
                "stderr": diagnostic.encode("utf-8"),
                "timeout": False,
                "output_limit": None,
                "error": None,
            }
            with mock.patch.object(
                qualification.provenance,
                "bounded_command",
                return_value=completed,
            ):
                result = qualification.run_failure_abort_probe(
                    executable, raw
                )
        self.assertEqual(result["exit_code"], 17)

    def test_collective_safety_probes_require_clean_completion_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            executable = root / "gpu-step-db"
            executable.write_text("fixture\n", encoding="utf-8")
            raw = root / "raw"
            raw.mkdir()

            def fake_bounded(command, *, cwd, environment, timeout_seconds,
                             stdout_limit, stderr_limit):
                del command, cwd, timeout_seconds, stdout_limit, stderr_limit
                if environment.get(
                    "MEEP_GPU_TEST_NEAR2FAR_CROSS_RANK_CANCELLATION"
                ):
                    marker = (
                        "PASS: Near2Far collective gate detects cross-rank "
                        "cancellation\n"
                    )
                elif environment.get(
                    "MEEP_GPU_TEST_NEAR2FAR_LOWLEVEL_DISTRIBUTED"
                ):
                    marker = (
                        "PASS: distributed Near2Far lowlevel preserves "
                        "rank-local ownership and forces mixed CUDA before "
                        "explicit sum\n"
                    )
                elif environment.get("MEEP_GPU_TEST_NEAR2FAR_CHUNKLESS_RANK"):
                    marker = (
                        "PASS: chunkless rank enters the public 13-double "
                        "Near2Far collective and receives the nonempty rank "
                        "result\n"
                    )
                elif environment.get(
                    "MEEP_GPU_TEST_NEAR2FAR_METADATA_MISMATCH"
                ):
                    marker = (
                        "PASS: Near2Far "
                        f"{environment['MEEP_GPU_TEST_NEAR2FAR_METADATA_MISMATCH']} "
                        "metadata mismatch fails closed before result "
                        "collectives\n"
                    )
                else:
                    marker = (
                        "PASS: Near2Far "
                        f"{environment['MEEP_GPU_TEST_NEAR2FAR_SELECTION_API']} "
                        f"{environment['MEEP_GPU_TEST_NEAR2FAR_SELECTION_MISMATCH']} "
                        "mismatch fails closed on every MPI rank\n"
                    )
                return {
                    "exit_code": 0,
                    "stdout": marker.encode("utf-8"),
                    "stderr": b"",
                    "timeout": False,
                    "output_limit": None,
                    "error": None,
                }

            with mock.patch.object(
                qualification.provenance, "bounded_command",
                side_effect=fake_bounded,
            ):
                probes = [
                    qualification.run_collective_safety_probe(
                        executable, raw, "optout", selection_mismatch="optout",
                        api="batch",
                    ),
                    qualification.run_collective_safety_probe(
                        executable, raw, "backend", selection_mismatch="backend",
                        api="grid",
                    ),
                    qualification.run_collective_safety_probe(
                        executable, raw, "chunkless", chunkless_rank=True,
                    ),
                    qualification.run_collective_safety_probe(
                        executable, raw, "metadata",
                        metadata_mismatch="point-count",
                    ),
                    qualification.run_collective_safety_probe(
                        executable, raw, "cross-rank",
                        cross_rank_cancellation=True,
                    ),
                    qualification.run_collective_safety_probe(
                        executable, raw, "lowlevel",
                        distributed_lowlevel=True,
                    ),
                ]
        self.assertEqual(
            [probe["exit_code"] for probe in probes], [0, 0, 0, 0, 0, 0]
        )

    def test_collective_safety_probe_fails_closed_on_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            executable = root / "gpu-step-db"
            executable.write_text("fixture\n", encoding="utf-8")
            raw = root / "raw"
            raw.mkdir()
            timed_out = {
                "exit_code": -15,
                "stdout": b"",
                "stderr": b"",
                "timeout": True,
                "output_limit": None,
                "error": None,
            }
            with mock.patch.object(
                qualification.provenance, "bounded_command",
                return_value=timed_out,
            ):
                with self.assertRaisesRegex(
                    qualification.QualificationError,
                    "hung in mismatched collectives",
                ):
                    qualification.run_collective_safety_probe(
                        executable, raw, "timeout",
                        selection_mismatch="optout", timeout_seconds=0.25,
                    )

    def test_run_lane_fails_closed_on_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            executable = root / "gpu-step-db"
            executable.write_text("fixture\n", encoding="utf-8")
            raw = root / "raw"
            raw.mkdir()
            timed_out = {
                "command": ["mpirun"],
                "exit_code": -15,
                "stdout": b"",
                "stderr": b"",
                "timeout": True,
                "output_limit": None,
                "error": None,
            }
            with mock.patch.object(
                qualification.provenance,
                "bounded_command",
                return_value=timed_out,
            ):
                with self.assertRaisesRegex(
                    qualification.QualificationError, "exceeded 0.25s timeout"
                ):
                    qualification.run_lane(
                        executable, raw, "timeout", "cpu", 1, 1, 2, 1, 3,
                        timeout_seconds=0.25,
                    )

    def test_run_lane_fails_closed_on_output_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            executable = root / "gpu-step-db"
            executable.write_text("fixture\n", encoding="utf-8")
            raw = root / "raw"
            raw.mkdir()
            oversized = {
                "command": ["mpirun"],
                "exit_code": -15,
                "stdout": b"x" * 16,
                "stderr": b"",
                "timeout": False,
                "output_limit": "stdout",
                "error": None,
            }
            with mock.patch.object(
                qualification.provenance,
                "bounded_command",
                return_value=oversized,
            ):
                with self.assertRaisesRegex(
                    qualification.QualificationError,
                    "exceeded stdout output limit",
                ):
                    qualification.run_lane(
                        executable, raw, "oversized", "cpu", 1, 1, 2, 1, 3,
                        stdout_limit=16,
                    )


if __name__ == "__main__":
    unittest.main()
