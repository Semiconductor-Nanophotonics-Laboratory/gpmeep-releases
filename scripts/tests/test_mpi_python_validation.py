from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "run-mpi-python-validation.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("gpmeep_mpi_python_validation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestMpiPythonValidation(unittest.TestCase):
    receipt_id = "a" * 64
    nonce = "b" * 32

    @staticmethod
    def create_private_lane(output, lane):
        output.mkdir(mode=0o700)
        (output / lane).mkdir(mode=0o700)

    def metrics(self):
        values = [0.25] * 6400
        return {
            "timesteps": 320,
            "meep_time": 8.0,
            "shape": [80, 80],
            "field_values": values,
            "field_sha256": MODULE.canonical_field_sha256(values),
            "field_sum": 1600.0,
            "field_l1": 1600.0,
            "field_l2": 20.0,
            "field_maximum_absolute": 0.25,
            "field_weighted_checksum": 5120800.0,
            "flux_values": [0.1, 0.2, 0.3],
        }

    def statistics(self, backend, transport="pinned"):
        result = {}
        for phase, (group, stem) in MODULE.PHASES.items():
            result[group] = {
                f"cpu_{stem}_calls": 1 if backend == "cpu" and phase != "polarization" else 0,
                f"cpu_{stem}_points": 5 if backend == "cpu" and phase != "polarization" else 0,
                f"cuda_{stem}_calls": 1 if backend == "cuda" and phase != "polarization" else 0,
                f"cuda_{stem}_points": 5 if backend == "cuda" and phase != "polarization" else 0,
            }
        result["runtime"] = {
            "runtime_availability_probes": 1 if backend == "cuda" else 0,
            "runtime_device_enumerations": 1 if backend == "cuda" else 0,
            "runtime_device_selections": 1 if backend == "cuda" else 0,
        }
        cuda_aware = backend == "cuda" and transport == "cuda-aware"
        pinned = backend == "cuda" and transport == "pinned"
        result["multi_gpu"] = {
            "mpi_messages": 4 if backend == "cuda" else 0,
            "mpi_scalars": 16 if backend == "cuda" else 0,
            "cuda_aware_bytes": 64 if cuda_aware else 0,
            "pinned_staging_bytes": 64 if pinned else 0,
            "pinned_device_to_host_bytes": 32 if pinned else 0,
            "pinned_host_to_device_bytes": 32 if pinned else 0,
        }
        result["mpi_completion"] = {
            "mpi_waitsome_executions": 4 if pinned else 0,
            "mpi_waitall_executions": 4 if cuda_aware else 0,
        }
        return result

    @staticmethod
    def artifact(marker):
        return {"path": f"/runtime/{marker}", "size_bytes": 10, "sha256": marker * 64}

    def record(self, lane):
        backend = "cpu" if lane == "cpu-hidden" else "cuda"
        transport = "cuda-aware" if lane == "cuda-aware-waitall" else "pinned"
        completion = "waitall" if lane == "cuda-aware-waitall" else "waitsome"
        visible = "" if backend == "cpu" else "0,1"
        metrics = self.metrics()
        runtime = {
            "python": self.artifact("1"),
            "extension": self.artifact("2"),
            "libmeep": self.artifact("3"),
        }
        ranks = []
        for rank in range(2):
            ranks.append(
                {
                    "rank": rank,
                    "pid": 100 + rank,
                    "physical_affinity": [
                        {"host": "node", "package_id": 0, "core_id": rank}
                    ],
                    "active_backend": backend,
                    "requested_backend": backend,
                    "backend_diagnostic": "test",
                    "selected_device": -1 if backend == "cpu" else rank,
                    "selected_device_identifier": "" if backend == "cpu" else str(rank + 1) * 32,
                    "local_seconds": 1.0,
                    "statistics": self.statistics(backend, transport),
                    "startup_statistics": self.statistics("cpu"),
                    "metrics": copy.deepcopy(metrics),
                    "runtime": copy.deepcopy(runtime),
                    "environment": {
                        "CUDA_VISIBLE_DEVICES": visible,
                        "MEEP_GPU_BACKEND": backend,
                        "MEEP_GPU_STRICT": None if backend == "cpu" else "1",
                        "MEEP_GPU_ALLOW_OVERSUBSCRIBE": "0",
                        "MEEP_GPU_MPI_TRANSPORT": transport,
                        "MEEP_GPU_MPI_COMPLETION": completion,
                        "OMP_NUM_THREADS": "1",
                    },
                }
            )
        return {
            "schema_version": 1,
            "lane": lane,
            "receipt_id": self.receipt_id,
            "nonce": self.nonce,
            "mpi_ranks": 2,
            "workload": copy.deepcopy(MODULE.WORKLOAD),
            "metrics": metrics,
            "rank_records": ranks,
        }

    def validate(self, record, lane):
        return MODULE.validate_probe_record(
            record,
            lane=lane,
            receipt_id=self.receipt_id,
            nonce=self.nonce,
            visible_devices="" if lane == "cpu-hidden" else "0,1",
        )

    def test_valid_three_lane_matrix(self):
        for lane in MODULE.LANES:
            with self.subTest(lane=lane):
                self.assertTrue(self.validate(self.record(lane), lane)["pass"])

    def test_rejects_missing_rank_and_shortened_workload(self):
        missing = self.record("cpu-hidden")
        missing["rank_records"].pop()
        with self.assertRaisesRegex(MODULE.EvidenceError, "rank records"):
            self.validate(missing, "cpu-hidden")
        shortened = self.record("cpu-hidden")
        shortened["workload"]["expected_timesteps"] = 319
        shortened["metrics"]["timesteps"] = 319
        for rank in shortened["rank_records"]:
            rank["metrics"] = copy.deepcopy(shortened["metrics"])
        with self.assertRaisesRegex(MODULE.EvidenceError, "workload"):
            self.validate(shortened, "cpu-hidden")

    def test_cpu_hidden_rejects_runtime_probe_and_cuda_call(self):
        for mutation in ("probe", "call", "points"):
            record = self.record("cpu-hidden")
            if mutation == "probe":
                record["rank_records"][0]["statistics"]["runtime"][
                    "runtime_availability_probes"
                ] = 1
                message = "touched the CUDA runtime"
            elif mutation == "call":
                record["rank_records"][0]["statistics"]["dispatch"][
                    "cuda_curl_calls"
                ] = 1
                message = "cuda fallback"
            else:
                record["rank_records"][0]["statistics"]["dispatch"][
                    "cuda_curl_points"
                ] = 1
                message = "cuda fallback"
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(MODULE.EvidenceError, message):
                    self.validate(record, "cpu-hidden")

    def test_cuda_rejects_cpu_fallback_and_duplicate_uuid(self):
        fallback = self.record("cuda-pinned-waitsome")
        fallback["rank_records"][0]["statistics"]["dispatch"]["cpu_curl_calls"] = 1
        with self.assertRaisesRegex(MODULE.EvidenceError, "cpu fallback"):
            self.validate(fallback, "cuda-pinned-waitsome")
        duplicate = self.record("cuda-pinned-waitsome")
        duplicate["rank_records"][1]["selected_device_identifier"] = "1" * 32
        with self.assertRaisesRegex(MODULE.EvidenceError, "duplicate GPUs"):
            self.validate(duplicate, "cuda-pinned-waitsome")

    def test_rejects_transport_counter_and_completion_mismatch(self):
        pinned = self.record("cuda-pinned-waitsome")
        pinned["rank_records"][0]["statistics"]["multi_gpu"]["cuda_aware_bytes"] = 1
        with self.assertRaisesRegex(MODULE.EvidenceError, "pinned lane"):
            self.validate(pinned, "cuda-pinned-waitsome")
        aware = self.record("cuda-aware-waitall")
        aware["rank_records"][0]["environment"]["MEEP_GPU_MPI_COMPLETION"] = "waitsome"
        with self.assertRaisesRegex(MODULE.EvidenceError, "waitall environment"):
            self.validate(aware, "cuda-aware-waitall")
        wrong_branch = self.record("cuda-aware-waitall")
        wrong_branch["rank_records"][0]["statistics"]["mpi_completion"] = {
            "mpi_waitsome_executions": 1,
            "mpi_waitall_executions": 0,
        }
        with self.assertRaisesRegex(MODULE.EvidenceError, "MPI_Waitall"):
            self.validate(wrong_branch, "cuda-aware-waitall")

    def test_cpu_hidden_rejects_cuda_touch_before_reset(self):
        record = self.record("cpu-hidden")
        record["rank_records"][0]["startup_statistics"]["runtime"][
            "runtime_availability_probes"
        ] = 1
        with self.assertRaisesRegex(MODULE.EvidenceError, "import touched"):
            self.validate(record, "cpu-hidden")

    def test_pointwise_comparison_detects_tampering(self):
        cpu = self.record("cpu-hidden")["metrics"]
        cuda = copy.deepcopy(cpu)
        self.assertTrue(MODULE.compare_metrics(cpu, cuda)["pass"])
        cuda["field_values"][17] += 0.1
        cuda["field_sha256"] = MODULE.canonical_field_sha256(cuda["field_values"])
        values = cuda["field_values"]
        cuda.update(
            {
                "field_sum": math.fsum(values),
                "field_l1": math.fsum(abs(value) for value in values),
                "field_l2": math.sqrt(math.fsum(value * value for value in values)),
                "field_maximum_absolute": max(abs(value) for value in values),
                "field_weighted_checksum": math.fsum(
                    (index + 1) * value for index, value in enumerate(values)
                ),
            }
        )
        with self.assertRaisesRegex(MODULE.EvidenceError, "pointwise"):
            MODULE.compare_metrics(cpu, cuda)

    def test_loader_rejects_duplicate_and_nonfinite_json(self):
        for payload, message in (("{\"a\":1,\"a\":2}", "duplicate"),
                                 ("{\"a\":NaN}", "non-finite")):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(MODULE.EvidenceError, message):
                    MODULE.load_json_text(payload)

    def test_extract_probe_requires_one_record(self):
        encoded = MODULE.PROBE_PREFIX + json.dumps(self.record("cpu-hidden"))
        self.assertEqual(MODULE.extract_probe(encoded)["mpi_ranks"], 2)
        with self.assertRaisesRegex(MODULE.EvidenceError, "found 2"):
            MODULE.extract_probe(encoded + "\n" + encoded)

    def test_rank_device_bindings_join_rank_uuid_ordinal_and_pid(self):
        inventory = {
            4: {"uuid": "1" * 32},
            7: {"uuid": "2" * 32},
        }
        pinned = {"record": self.record("cuda-pinned-waitsome")}
        aware_record = self.record("cuda-aware-waitall")
        for rank in aware_record["rank_records"]:
            rank["pid"] += 10
        aware = {"record": aware_record}
        self.assertEqual(
            MODULE.validate_rank_device_bindings(
                [pinned, aware], [4, 7], inventory
            ),
            {100: 4, 101: 7, 110: 4, 111: 7},
        )
        swapped = copy.deepcopy(aware)
        swapped["record"]["rank_records"][0]["selected_device_identifier"] = (
            "2" * 32
        )
        with self.assertRaisesRegex(MODULE.EvidenceError, "UUID"):
            MODULE.validate_rank_device_bindings(
                [pinned, swapped], [4, 7], inventory
            )
        reused = copy.deepcopy(aware)
        reused["record"]["rank_records"][0]["pid"] = 100
        with self.assertRaisesRegex(MODULE.EvidenceError, "reused"):
            MODULE.validate_rank_device_bindings(
                [pinned, reused], [4, 7], inventory
            )

    def test_divide_telemetry_requires_backend_execution(self):
        records = []
        for rank in range(2):
            records.append({
                "schema_version": 1,
                "nonce": self.nonce,
                "world_rank": rank,
                "pid": 200 + rank,
                "expected_backend": "cuda",
                "active_backend": "cuda",
                "requested_backend": "cuda",
                "selected_device": rank,
                "selected_device_identifier": str(rank + 1) * 32,
                "startup_statistics": self.statistics("cpu"),
                "statistics": self.statistics("cuda", "pinned"),
            })
        self.assertTrue(MODULE.validate_divide_telemetry(
            records, backend="cuda", nonce=self.nonce
        )["pass"])
        records[0]["statistics"]["dispatch"]["cpu_curl_calls"] = 1
        with self.assertRaisesRegex(MODULE.EvidenceError, "cpu fallback"):
            MODULE.validate_divide_telemetry(
                records, backend="cuda", nonce=self.nonce
            )

    def test_runtime_artifacts_must_match_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            paths = {}
            for name in ("python", "extension", "libmeep"):
                path = repo / name
                path.write_bytes(name.encode())
                paths[name] = path
            runtime = {
                name: MODULE.file_record(path, repo)
                for name, path in paths.items()
            }
            receipt = {
                "toolchain": {"python": copy.deepcopy(runtime["python"])},
                "artifacts": {
                    "python_extension": copy.deepcopy(runtime["extension"]),
                    "libmeep": copy.deepcopy(runtime["libmeep"]),
                },
            }
            self.assertTrue(MODULE.validate_runtime_against_receipt(
                runtime, receipt, repo
            )["pass"])
            receipt["artifacts"]["libmeep"]["sha256"] = "f" * 64
            with self.assertRaisesRegex(MODULE.EvidenceError, "build receipt"):
                MODULE.validate_runtime_against_receipt(runtime, receipt, repo)

    def test_unittest_parser_is_exact_and_rejects_skip(self):
        identity = (
            "test_divide_parallel_processes "
            "(test_divide_mpi_processes.TestDivideParallelProcesses."
            "test_divide_parallel_processes)"
        )
        text = (
            "Using MPI version 3.1, 2 processes\n"
            + f"{identity} ... ok\nRan 1 test in 0.1s\n\nOK\n" * 2
        )
        self.assertTrue(MODULE.parse_unittest_log(text)["pass"])
        self.assertTrue(MODULE.parse_unittest_log(
            text + "OSC UCX Error: harmless external warning\n"
        )["pass"])
        with self.assertRaisesRegex(MODULE.EvidenceError, "skipped"):
            MODULE.parse_unittest_log(text + "skipped=1\n")

    def test_divide_telemetry_is_one_hashed_world_gather_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "telemetry.json"
            records = [{"world_rank": 0}, {"world_rank": 1}]
            path.write_text(
                json.dumps({"records": records}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            witness = {
                "schema_version": 1,
                "sha256": MODULE.sha256_file(path),
            }
            encoded = MODULE.DIVIDE_PREFIX + json.dumps(witness)
            self.assertEqual(
                MODULE.load_divide_telemetry(encoded, path),
                {"records": records, "sha256": witness["sha256"]},
            )
            with self.assertRaisesRegex(MODULE.EvidenceError, "witness"):
                MODULE.load_divide_telemetry(encoded + "\n" + encoded, path)
            path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.EvidenceError, "hash mismatch"):
                MODULE.load_divide_telemetry(encoded, path)

    def test_pmon_rejects_unrelated_or_missing_pid(self):
        text = "# gpu pid type sm mem enc dec command\n0 10 C 1 1 - - python\n1 11 C 1 1 - - python\n"
        self.assertTrue(
            MODULE.parse_pmon(text, expected_pid_to_gpu={10: 0, 11: 1})["pass"]
        )
        with self.assertRaisesRegex(MODULE.EvidenceError, "unrelated"):
            MODULE.parse_pmon(text + "0 99 C 1 1 - - other\n",
                              expected_pid_to_gpu={10: 0, 11: 1})
        with self.assertRaisesRegex(MODULE.EvidenceError, "missed"):
            MODULE.parse_pmon(text, expected_pid_to_gpu={10: 0, 11: 1, 12: 0})

    def test_nvidia_witness_environment_is_minimal(self):
        environment = MODULE.nvidia_environment()
        self.assertEqual(
            environment,
            {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertNotIn("LD_LIBRARY_PATH", environment)

    def test_nvidia_witness_identity_must_remain_stable(self):
        before = {
            "inventory": [{"index": 0, "uuid": "1" * 32}],
            "nvidia_smi": self.artifact("a"),
        }
        MODULE.validate_stable_gpu_witness(before, copy.deepcopy(before))
        changed = copy.deepcopy(before)
        changed["nvidia_smi"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(MODULE.EvidenceError, "identity changed"):
            MODULE.validate_stable_gpu_witness(before, changed)

    def test_clean_environment_redirects_font_and_matplotlib_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            fontconfig = repo / "build" / "qualification-fontconfig.conf"
            fontconfig.parent.mkdir(parents=True)
            fontconfig.write_text("<fontconfig/>\n", encoding="utf-8")
            receipt = {
                "build_dir": "build",
                "receipt_id": self.receipt_id,
                "configuration_files": {
                    "qualification_fontconfig": MODULE.file_record(
                        fontconfig, repo
                    )
                },
                "configuration": {"environment": {}},
            }
            output = repo / "evidence"
            self.create_private_lane(output, "cpu-hidden")
            previous_umask = os.umask(0o777)
            try:
                environment = MODULE.clean_environment(
                    repo,
                    output,
                    receipt,
                    "cpu-hidden",
                    "",
                    self.nonce,
                )
            finally:
                os.umask(previous_umask)
            self.assertEqual(
                environment["FONTCONFIG_FILE"],
                str(
                    output
                    / "cpu-hidden"
                    / "runtime"
                    / "config"
                    / "qualification-fontconfig.conf"
                ),
            )
            copied = pathlib.Path(environment["FONTCONFIG_FILE"])
            self.assertEqual(copied.read_bytes(), fontconfig.read_bytes())
            self.assertEqual(copied.stat().st_mode & 0o777, 0o400)
            for directory in (
                output / "cpu-hidden" / "runtime",
                output / "cpu-hidden" / "runtime" / "home",
                output / "cpu-hidden" / "runtime" / "cache",
                output / "cpu-hidden" / "runtime" / "config",
            ):
                self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            self.assertEqual(
                environment["MPLCONFIGDIR"],
                str(output / "cpu-hidden" / "runtime" / "config" / "matplotlib"),
            )
            self.assertEqual(
                environment["XDG_CACHE_HOME"],
                str(output / "cpu-hidden" / "runtime" / "cache"),
            )

    def test_clean_environment_requires_receipt_bound_fontconfig(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            receipt = {
                "build_dir": "build",
                "receipt_id": self.receipt_id,
                "configuration_files": {},
                "configuration": {"environment": {}},
            }
            with self.assertRaisesRegex(
                MODULE.EvidenceError, "no qualification Fontconfig record"
            ):
                MODULE.clean_environment(
                    repo,
                    repo / "evidence",
                    receipt,
                    "cpu-hidden",
                    "",
                    self.nonce,
                )

    def test_clean_environment_rejects_changed_or_symlinked_fontconfig(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            fontconfig = repo / "build" / "qualification-fontconfig.conf"
            fontconfig.parent.mkdir(parents=True)
            fontconfig.write_text("<fontconfig/>\n", encoding="utf-8")
            record = MODULE.file_record(fontconfig, repo)
            receipt = {
                "build_dir": "build",
                "receipt_id": self.receipt_id,
                "configuration_files": {
                    "qualification_fontconfig": record
                },
                "configuration": {"environment": {}},
            }
            fontconfig.write_text("changed\n", encoding="utf-8")
            changed_output = repo / "changed-evidence"
            self.create_private_lane(changed_output, "cpu-hidden")
            with self.assertRaisesRegex(
                MODULE.EvidenceError, "do not match the build receipt"
            ):
                MODULE.clean_environment(
                    repo,
                    changed_output,
                    receipt,
                    "cpu-hidden",
                    "",
                    self.nonce,
                )

            fontconfig.unlink()
            target = repo / "build" / "target.conf"
            target.write_text("<fontconfig/>\n", encoding="utf-8")
            fontconfig.symlink_to(target)
            symlink_output = repo / "symlink-evidence"
            self.create_private_lane(symlink_output, "cpu-hidden")
            with self.assertRaisesRegex(
                MODULE.EvidenceError, "cannot securely open"
            ):
                MODULE.clean_environment(
                    repo,
                    symlink_output,
                    receipt,
                    "cpu-hidden",
                    "",
                    self.nonce,
                )

    def test_clean_environment_rejects_symlinked_runtime_cache_or_config(self):
        for linked_name in ("cache", "config"):
            with self.subTest(linked_name=linked_name), tempfile.TemporaryDirectory() as directory:
                repo = pathlib.Path(directory)
                fontconfig = repo / "build" / "qualification-fontconfig.conf"
                fontconfig.parent.mkdir(parents=True)
                fontconfig.write_text("<fontconfig/>\n", encoding="utf-8")
                receipt = {
                    "build_dir": "build",
                    "receipt_id": self.receipt_id,
                    "configuration_files": {
                        "qualification_fontconfig": MODULE.file_record(
                            fontconfig, repo
                        )
                    },
                    "configuration": {"environment": {}},
                }
                output = repo / "evidence"
                self.create_private_lane(output, "cpu-hidden")
                runtime = output / "cpu-hidden" / "runtime"
                runtime.mkdir(mode=0o700)
                (runtime / "home").mkdir(mode=0o700)
                external = repo / "external"
                external.mkdir()
                (runtime / linked_name).symlink_to(external, target_is_directory=True)
                other_name = "config" if linked_name == "cache" else "cache"
                (runtime / other_name).mkdir(mode=0o700)
                with self.assertRaisesRegex(
                    MODULE.EvidenceError, "not a real directory"
                ):
                    MODULE.clean_environment(
                        repo,
                        output,
                        receipt,
                        "cpu-hidden",
                        "",
                        self.nonce,
                    )

    def test_prepare_output_does_not_touch_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "existing"
            output.mkdir()
            complete = output / "COMPLETE"
            complete.write_text("old authority\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.EvidenceError, "already exists"):
                MODULE.prepare_output(output, "new-run")
            self.assertEqual(complete.read_text(encoding="utf-8"), "old authority\n")

    def test_mark_failed_revokes_partial_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            for name in (
                "report.json", "report.md", "artifacts.sha256.json", "COMPLETE"
            ):
                (output / name).write_text("partial", encoding="utf-8")
            MODULE.mark_failed(output, "run-1", RuntimeError("boom"))
            for name in (
                "report.json", "report.md", "artifacts.sha256.json", "COMPLETE"
            ):
                self.assertFalse((output / name).exists())
            failed = MODULE.load_json_text(
                (output / "FAILED.json").read_text(encoding="utf-8")
            )
            state = MODULE.load_json_text(
                (output / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed, state)
            self.assertEqual(state["state"], "FAILED")

    def test_artifact_manifest_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            payload = repo / "raw.log"
            payload.write_text("original", encoding="utf-8")
            empty_log = repo / "empty-stderr.log"
            empty_log.write_bytes(b"")
            manifest = {
                "schema_version": 1,
                "root": str(repo),
                "file_count": 2,
                "files": [
                    MODULE.file_record(empty_log, repo),
                    MODULE.file_record(payload, repo),
                ],
            }
            MODULE.verify_artifact_manifest(
                manifest, repo, expected_root=repo
            )
            payload.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.EvidenceError, "changed"):
                MODULE.verify_artifact_manifest(
                    manifest, repo, expected_root=repo
                )

    def test_artifact_manifest_rejects_symlink_and_output_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            output = repo / "evidence"
            MODULE.prepare_output(output, "run-private")
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            payload = output / "raw.log"
            payload.write_text("raw", encoding="utf-8")
            manifest = MODULE.artifact_manifest(output, repo)
            external = repo / "external.log"
            external.write_text("external", encoding="utf-8")
            (output / "alias.log").symlink_to(external)
            with self.assertRaisesRegex(
                MODULE.EvidenceError, "contains a symlink"
            ):
                MODULE.verify_artifact_manifest(
                    manifest, repo, expected_root=output
                )
            with self.assertRaisesRegex(
                MODULE.EvidenceError, "contains a symlink"
            ):
                MODULE.artifact_manifest(output, repo)

    def test_artifact_manifest_rejects_split_raw_root(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            raw_root = repo / "raw-root"
            evidence = repo / "evidence"
            raw_root.mkdir(mode=0o700)
            evidence.mkdir(mode=0o700)
            (raw_root / "raw.log").write_text("raw", encoding="utf-8")
            manifest = MODULE.artifact_manifest(raw_root, repo)
            with self.assertRaisesRegex(
                MODULE.EvidenceError,
                "root differs from the evidence directory",
            ):
                MODULE.verify_artifact_manifest(
                    manifest, repo, expected_root=evidence
                )

    def test_complete_and_state_are_cryptographically_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            output = repo / "evidence"
            output.mkdir()
            raw = output / "raw.log"
            raw.write_text("raw", encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "root": str(output),
                "file_count": 1,
                "files": [MODULE.file_record(raw, repo)],
            }
            MODULE.atomic_write_json(output / "artifacts.sha256.json", manifest)
            markdown = output / "report.md"
            markdown.write_text("report", encoding="utf-8")
            manifest_record = MODULE.file_record(
                output / "artifacts.sha256.json", repo
            )
            report = {
                "state": "COMPLETE",
                "run_id": "run-1",
                "build_receipt": {"receipt_id": "a" * 64},
                "source_snapshot": {"sha256": "b" * 64},
                "artifact_manifest": manifest_record,
                "report_markdown": MODULE.file_record(markdown, repo),
            }
            MODULE.atomic_write_json(output / "report.json", report)
            marker = {
                "schema_version": 1,
                "state": "COMPLETE",
                "run_id": "run-1",
                "build_receipt_id": "a" * 64,
                "source_snapshot_sha256": "b" * 64,
                "report": MODULE.file_record(output / "report.json", repo),
                "report_markdown": MODULE.file_record(markdown, repo),
                "artifact_manifest": manifest_record,
            }
            MODULE.atomic_write_json(output / "COMPLETE", marker)
            MODULE.atomic_write_json(output / "state.json", marker)
            MODULE.verify_complete_publication(output, repo)
            (output / "FAILED.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.EvidenceError, "contradictory FAILED.json"
            ):
                MODULE.verify_complete_publication(output, repo)
            (output / "FAILED.json").unlink()
            marker["run_id"] = "tampered"
            MODULE.atomic_write_json(output / "state.json", marker)
            with self.assertRaisesRegex(MODULE.EvidenceError, "publication mismatch"):
                MODULE.verify_complete_publication(output, repo)


if __name__ == "__main__":
    unittest.main()
