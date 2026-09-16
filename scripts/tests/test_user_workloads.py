import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import pathlib
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKLOAD_DIR = ROOT / "scripts" / "user-workloads"
sys.path.insert(0, str(WORKLOAD_DIR))


def load_module(name):
    path = WORKLOAD_DIR / name
    module_name = f"gpmeep_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


COMMON = load_module("common.py")
sys.modules["common"] = COMMON
COMPARE = load_module("compare_user_workloads.py")
MATRIX = load_module("run_user_workload_matrix.py")
sys.modules["run_user_workload_matrix"] = MATRIX
HYBRID = load_module("run_hybrid_ters_matrix.py")


def phase_contract(run_count):
    return [
        {
            "stage": f"stage-{index}",
            "required_per_rank": list(COMMON.FDTD_PHASE_COUNTERS),
            "required_aggregate": [],
            "forbidden": [],
        }
        for index in range(run_count)
    ]


def statistics_snapshot(backend=None, value=0):
    result = {}
    for _phase, (group_name, stem) in COMMON.FDTD_PHASE_COUNTERS.items():
        result[group_name] = {
            f"{candidate}_{stem}_{field}": (
                value if candidate == backend else 0
            )
            for candidate in ("cpu", "cuda")
            for field in ("calls", "points")
        }
    result["multi_gpu"] = {name: 0 for name in COMMON.MULTI_GPU_COUNTERS}
    result["mpi_completion"] = {
        name: 0 for name in COMMON.MPI_COMPLETION_COUNTERS
    }
    return result


def strict_phase_record(index, backend, *, two_gpu=False):
    before = statistics_snapshot()
    after = statistics_snapshot(backend, 1)
    if two_gpu:
        after["multi_gpu"].update(
            {
                "mpi_messages": 2,
                "mpi_scalars": 10,
                "cuda_aware_bytes": 0,
                "pinned_staging_bytes": 40,
                "pinned_device_to_host_bytes": 20,
                "pinned_host_to_device_bytes": 20,
            }
        )
        after["mpi_completion"]["mpi_waitall_executions"] = 1
    delta = COMMON.statistics_delta(before, after)
    return {
        "run_index": index,
        "statistics_before": before,
        "statistics_after": after,
        "statistics_delta": delta,
        "phase_counters": COMMON.phase_counter_view(delta),
        "phase_calls": COMMON.phase_call_totals(delta),
    }


def original_ters_rank_outputs(root, rank=0):
    source_payload = b"fixture source\n"
    source = root / "fixture.py"
    if not source.exists():
        source.write_bytes(source_payload)
    result = (
        root
        / "work"
        / f"rank-{rank:05d}"
        / "results"
        / "fixture.py_00-00-00_20000101"
    )
    result.mkdir(parents=True, exist_ok=True)
    names = (
        "fixture_backup.py",
        "withtip_geo.png",
        "withtip_dft.csv",
        "withtip_dft.png",
        "withtip_dft.npy",
        "wotip_geo.png",
        "wotip_dft.csv",
        "wotip_dft.png",
        "wotip_dft.npy",
    )
    for name in names:
        payload = source_payload if name.endswith("_backup.py") else b"fixture\n"
        (result / name).write_bytes(payload)
    return (
        [COMMON.file_record(result / name, root) for name in names],
        hashlib.sha256(source_payload).hexdigest(),
        result,
    )


class UserWorkloadCommonTests(unittest.TestCase):
    def test_phase_contract_rejects_partial_cuda_and_cpu_fallback(self):
        contract = phase_contract(1)
        record = strict_phase_record(0, "cuda")
        COMMON.validate_phase_contract(
            [{"rank": 0, "records": [record]}], "cuda", contract
        )

        missing = strict_phase_record(0, "cuda")
        missing["statistics_after"]["dfts"]["cuda_dft_calls"] = 0
        missing["statistics_after"]["dfts"]["cuda_dft_points"] = 0
        missing["statistics_delta"] = COMMON.statistics_delta(
            missing["statistics_before"], missing["statistics_after"]
        )
        missing["phase_counters"] = COMMON.phase_counter_view(
            missing["statistics_delta"]
        )
        missing["phase_calls"] = COMMON.phase_call_totals(missing["statistics_delta"])
        with self.assertRaisesRegex(COMMON.WorkloadError, "lacks cuda dft work"):
            COMMON.validate_phase_contract(
                [{"rank": 0, "records": [missing]}], "cuda", contract
            )

        fallback = strict_phase_record(0, "cuda")
        fallback["statistics_after"]["sources"]["cpu_source_calls"] = 1
        fallback["statistics_after"]["sources"]["cpu_source_points"] = 1
        fallback["statistics_delta"] = COMMON.statistics_delta(
            fallback["statistics_before"], fallback["statistics_after"]
        )
        fallback["phase_counters"] = COMMON.phase_counter_view(
            fallback["statistics_delta"]
        )
        fallback["phase_calls"] = COMMON.phase_call_totals(fallback["statistics_delta"])
        with self.assertRaisesRegex(COMMON.WorkloadError, "fallback for source"):
            COMMON.validate_phase_contract(
                [{"rank": 0, "records": [fallback]}], "cuda", contract
            )

    def test_two_gpu_contract_requires_rank_work_pinned_traffic_and_completion(self):
        contract = phase_contract(1)
        ranks = [
            {"rank": rank, "records": [strict_phase_record(0, "cuda", two_gpu=True)]}
            for rank in range(2)
        ]
        COMMON.validate_phase_contract(ranks, "cuda", contract)

        no_traffic = [
            {"rank": rank, "records": [strict_phase_record(0, "cuda")]}
            for rank in range(2)
        ]
        with self.assertRaisesRegex(COMMON.WorkloadError, "two-GPU MPI"):
            COMMON.validate_phase_contract(no_traffic, "cuda", contract)

    def test_statistics_delta_rejects_schema_change_and_counter_reset(self):
        before = statistics_snapshot("cpu", 2)
        after = statistics_snapshot("cpu", 1)
        with self.assertRaisesRegex(COMMON.WorkloadError, "decreasing counter"):
            COMMON.statistics_delta(before, after)
        after = statistics_snapshot("cpu", 2)
        after["extra"] = {"counter": 0}
        with self.assertRaisesRegex(COMMON.WorkloadError, "schema changed"):
            COMMON.statistics_delta(before, after)

    def test_safe_member_path_rejects_escape_and_nonportable_names(self):
        for name in ("../escape", "/absolute", "a/../escape", "a\\b"):
            with self.subTest(name=name):
                with self.assertRaises(COMMON.WorkloadError):
                    COMMON._safe_member_path(name)

    def test_verified_extraction_rejects_links_and_binds_every_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = pathlib.Path(temporary)
            archive = temporary / "fixture.tar.gz"
            members = {
                "aunp_r4000_repro/one.txt": b"one\n",
                "aunp_r4000_repro/two.txt": b"two\n",
            }
            with tarfile.open(archive, "w:gz") as bundle:
                for name, payload in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    bundle.addfile(info, io.BytesIO(payload))
            archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            member_digests = {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in members.items()
            }
            with mock.patch.object(COMMON, "AUNP_ARCHIVE_SHA256", archive_digest), mock.patch.object(
                COMMON, "AUNP_MEMBER_SHA256", member_digests
            ):
                root = COMMON.extract_verified_aunp(archive, temporary / "out")
            self.assertEqual((root / "one.txt").read_bytes(), b"one\n")
            self.assertEqual((root / "two.txt").read_bytes(), b"two\n")

    def test_verified_extraction_rejects_special_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = pathlib.Path(temporary)
            archive = temporary / "fixture.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                link = tarfile.TarInfo("aunp_r4000_repro/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                bundle.addfile(link)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with mock.patch.object(COMMON, "AUNP_ARCHIVE_SHA256", digest), mock.patch.object(
                COMMON,
                "AUNP_MEMBER_SHA256",
                {"aunp_r4000_repro/link": hashlib.sha256(b"").hexdigest()},
            ):
                with self.assertRaisesRegex(COMMON.WorkloadError, "link or special"):
                    COMMON.extract_verified_aunp(archive, temporary / "out")

    def test_regular_file_rejects_terminal_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "target"
            target.write_text("payload", encoding="utf-8")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(COMMON.WorkloadError, "non-symlink"):
                COMMON.regular_file(link, "fixture")

    def test_runtime_receipt_accepts_build_or_installed_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary)
            paths = {}
            for name, payload in (
                ("python", b"python"),
                ("build_extension", b"build extension"),
                ("installed_extension", b"installed extension"),
                ("build_libmeep", b"build libmeep"),
                ("installed_libmeep", b"installed libmeep"),
            ):
                paths[name] = repo / name
                paths[name].write_bytes(payload)

            def record(name):
                return COMMON.absolute_file_record(paths[name], name)

            receipt = {
                "toolchain": {"python": record("python")},
                "artifacts": {
                    "python_extension": record("build_extension"),
                    "installed_python_extension": record("installed_extension"),
                    "libmeep": record("build_libmeep"),
                    "installed_libmeep": record("installed_libmeep"),
                },
            }
            COMMON.validate_runtime_against_receipt(
                {
                    "python": record("python"),
                    "extension": record("build_extension"),
                    "libmeep": record("build_libmeep"),
                },
                receipt,
                repo,
            )
            with self.assertRaisesRegex(COMMON.WorkloadError, "coherent"):
                COMMON.validate_runtime_against_receipt(
                    {
                        "python": record("python"),
                        "extension": record("build_extension"),
                        "libmeep": record("installed_libmeep"),
                    },
                    receipt,
                    repo,
                )
            COMMON.validate_runtime_against_receipt(
                {
                    "python": record("python"),
                    "extension": record("installed_extension"),
                    "libmeep": record("installed_libmeep"),
                },
                receipt,
                repo,
            )

    def test_stable_file_detects_in_place_mutation_and_name_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "artifact"
            path.write_bytes(b"original")
            with self.assertRaisesRegex(COMMON.WorkloadError, "changed while"):
                with COMMON.StableFile(path, "fixture"):
                    path.write_bytes(b"mutated!")

            path.write_bytes(b"original")
            replacement = root / "replacement"
            replacement.write_bytes(b"original")
            with self.assertRaisesRegex(
                COMMON.WorkloadError, "changed while|name was replaced"
            ):
                with COMMON.StableFile(path, "fixture"):
                    os.replace(replacement, path)

            path.write_bytes(b"original")
            replacement.write_bytes(b"original")
            backup = root / "backup"
            with self.assertRaisesRegex(
                COMMON.WorkloadError, "changed while|parent directory"
            ):
                with COMMON.StableFile(path, "fixture"):
                    os.replace(path, backup)
                    os.replace(replacement, path)
                    os.replace(path, replacement)
                    os.replace(backup, path)


class UserWorkloadMatrixTests(unittest.TestCase):
    @staticmethod
    def log_path(root, name):
        return root / "logs" / "fixture" / name

    def test_normalized_uuid_is_strict(self):
        value = "GPU-00112233-4455-6677-8899-aabbccddeeff"
        self.assertEqual(MATRIX._normalized_uuid(value), value)
        for invalid in ("0", "GPU-not-a-uuid", "00112233445566778899aabbccddeefg"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(COMMON.WorkloadError):
                    MATRIX._normalized_uuid(invalid)

    def test_clean_environment_isolates_fontconfig_from_build_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            prefix = root / "prefix&escaped"
            build = MATRIX.Build(
                name="fixture",
                python=prefix / "bin" / "python3.11",
                mpiexec=prefix / "bin" / "mpirun",
                receipt_path=root / "receipt.json",
                receipt={},
                pythonpath=root / "build" / "python",
                lib_directory=root / "build" / "src" / ".libs",
                single_precision=True,
            )
            lane = MATRIX.Lane("cuda-fp32-1g", build, "cuda", 1, ("GPU-0",))
            runtime = root / "runtime"
            environment = MATRIX.clean_environment(lane, runtime)

            config = runtime / "fontconfig.conf"
            self.assertEqual(environment["CONDA_PREFIX"], str(prefix))
            self.assertEqual(environment["FONTCONFIG_FILE"], str(config))
            self.assertEqual(environment["XDG_CACHE_HOME"], str(runtime / "cache"))
            self.assertTrue((runtime / "cache" / "fontconfig").is_dir())
            payload = config.read_text(encoding="utf-8")
            self.assertIn("prefix&amp;escaped/fonts", payload)
            self.assertIn('<cachedir prefix="xdg">fontconfig</cachedir>', payload)
            self.assertNotIn("var/cache/fontconfig", payload)

    def test_ters_timeout_covers_multi_day_exact_cpu_reference(self):
        self.assertGreaterEqual(MATRIX.DEFAULT_TIMEOUT["ters"], 7 * 24 * 3600)

    def test_lane_metrics_follow_tm_only_aunp_phase_contract(self):
        summary = {
            "phase_contract": phase_contract(2),
            "rank_records": [
                {
                    "rank": 0,
                    "process_wall_seconds": 9.0,
                    "records": [
                        {
                            "run_index": 0,
                            "timestep_delta": 100,
                            "meep_time": 20.0,
                            "wall_seconds": 2.5,
                        },
                        {
                            "run_index": 1,
                            "timestep_delta": 200,
                            "meep_time": 44.0,
                            "wall_seconds": 4.5,
                        },
                    ],
                }
            ],
        }
        metrics = MATRIX.derive_lane_metrics(summary, "aunp", "tm-only")
        self.assertEqual(metrics["phase_wall_seconds"], [2.5, 4.5])
        self.assertEqual(metrics["fdtd_wall_seconds"], 7.0)
        self.assertEqual(metrics["workload_end_to_end_seconds"], 9.0)
        self.assertEqual(
            metrics["timestep_contract"],
            [
                {"timestep_delta": 100, "meep_time": 20.0},
                {"timestep_delta": 200, "meep_time": 44.0},
            ],
        )

    def test_lane_metrics_reject_rank_count_that_differs_from_phase_contract(self):
        summary = {
            "phase_contract": phase_contract(2),
            "rank_records": [
                {
                    "rank": 0,
                    "process_wall_seconds": 1.0,
                    "records": [
                        {
                            "run_index": 0,
                            "timestep_delta": 1,
                            "meep_time": 1.0,
                            "wall_seconds": 1.0,
                        }
                    ],
                }
            ],
        }
        with self.assertRaisesRegex(
            COMMON.WorkloadError, "rank record count differs from phase contract"
        ):
            MATRIX.derive_lane_metrics(summary, "aunp", "partial")

    def test_comparison_specs_include_direct_cpu_and_every_repeat_lane(self):
        labels = ("cpu64", "cpu32", "gpu1", "gpu2")
        initial = MATRIX.comparison_specs(labels, 0)
        self.assertIn(("cpu32", "cpu-fp64-fp32"), initial[0][3])
        repeated = MATRIX.comparison_specs(labels, 1)
        repeat_specs = repeated[2:]
        self.assertEqual(len(repeat_specs), 4)
        self.assertEqual({spec[1] for spec in repeat_specs}, set(labels))
        self.assertTrue(all(spec[2] == 0 for spec in repeat_specs))

    def test_shared_fp32_receipt_contract_is_fail_closed(self):
        labels = ("cpu64", "cpu32", "gpu1", "gpu2")
        valid = {
            labels[0]: {"a" * 64},
            labels[1]: {"b" * 64},
            labels[2]: {"b" * 64},
            labels[3]: {"b" * 64},
        }
        MATRIX.validate_shared_fp32_receipts(valid, labels)
        invalid = {key: set(value) for key, value in valid.items()}
        invalid[labels[2]] = {"c" * 64}
        with self.assertRaisesRegex(COMMON.WorkloadError, "same exact build"):
            MATRIX.validate_shared_fp32_receipts(invalid, labels)

    def test_build_contract_is_explicit_and_fail_closed(self):
        valid = {
            "build_kind": "cpu-mpi-python-fp64",
            "configuration": {
                "qualification_contract": "gpmeep-cpu-mpi-python-fp64-v1",
                "configure_argv": [
                    "--disable-single",
                    "--disable-cuda",
                    "--with-mpi",
                    "--with-python",
                    "--without-scheme",
                ]
            },
        }
        configuration, single = MATRIX.validate_build_contract(
            "cpu-fp64",
            valid,
            "cpu-mpi-python-fp64",
            False,
            False,
        )
        self.assertFalse(single)
        self.assertIn("--disable-cuda", configuration)

        mutations = (
            ("build_kind", "generic", "build kind"),
            ("contract", "generic", "qualification contract"),
            ("flags", "--disable-single", "explicit FP32"),
            ("flags", "--disable-cuda", "explicit CUDA"),
            ("flags", "--with-mpi", "explicit MPI"),
            ("flags", "--without-scheme", "explicit Scheme"),
        )
        for mutation, value, diagnostic in mutations:
            candidate = {
                "build_kind": valid["build_kind"],
                "configuration": {
                    "qualification_contract": valid["configuration"][
                        "qualification_contract"
                    ],
                    "configure_argv": list(
                        valid["configuration"]["configure_argv"]
                    )
                },
            }
            if mutation == "build_kind":
                candidate["build_kind"] = value
            elif mutation == "contract":
                candidate["configuration"]["qualification_contract"] = value
            else:
                candidate["configuration"]["configure_argv"].remove(value)
            with self.subTest(value=value), self.assertRaisesRegex(
                COMMON.WorkloadError, diagnostic
            ):
                MATRIX.validate_build_contract(
                    "cpu-fp64",
                    candidate,
                    "cpu-mpi-python-fp64",
                    False,
                    False,
                )

        for duplicate in (
            "--disable-single",
            "--disable-cuda",
            "--with-mpi",
            "--with-python",
            "--without-scheme",
        ):
            candidate = {
                "build_kind": valid["build_kind"],
                "configuration": {
                    "qualification_contract": valid["configuration"][
                        "qualification_contract"
                    ],
                    "configure_argv": [
                        *valid["configuration"]["configure_argv"],
                        duplicate,
                    ]
                },
            }
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(
                COMMON.WorkloadError, "exactly one explicit"
            ):
                MATRIX.validate_build_contract(
                    "cpu-fp64",
                    candidate,
                    "cpu-mpi-python-fp64",
                    False,
                    False,
                )

        for override in (
            "--enable-single=no",
            "--disable-single=no",
            "--enable-cuda=no",
            "--disable-cuda=no",
            "--with-mpi=no",
            "--without-python=no",
            "--with-scheme=no",
        ):
            candidate = copy.deepcopy(valid)
            candidate["configuration"]["configure_argv"].append(override)
            with self.subTest(override=override), self.assertRaisesRegex(
                COMMON.WorkloadError, "noncanonical"
            ):
                MATRIX.validate_build_contract(
                    "cpu-fp64",
                    candidate,
                    "cpu-mpi-python-fp64",
                    False,
                    False,
                )

    def test_hardware_evidence_rejects_selected_gpu_process(self):
        uuid = "GPU-00112233-4455-6677-8899-aabbccddeeff"
        snapshot = {
            "captured_unix_seconds": 1.0,
            "hostname": "host",
            "kernel": "kernel",
            "cpu_model": "cpu",
            "visible_logical_cpus": [0, 1],
            "visible_physical_cores": 1,
            "cpu_governors": ["performance"],
            "load_average": [0.1, 0.2, 0.3],
            "memory_bytes": {"MemTotal": 1, "MemAvailable": 1},
            "gpus": [
                {
                    "index": 0,
                    "uuid": uuid,
                    "name": "gpu",
                    "driver_version": "1",
                    "compute_capability": "8.6",
                    "memory_total_mib": 1,
                }
            ],
            "selected_gpu_compute_processes": [],
        }
        evidence = {"before": dict(snapshot), "after": dict(snapshot)}
        MATRIX.validate_hardware_evidence(evidence, [uuid])
        evidence["after"] = {
            **snapshot,
            "selected_gpu_compute_processes": [f"{uuid}, 1, other, 1"],
        }
        with self.assertRaisesRegex(COMMON.WorkloadError, "not idle"):
            MATRIX.validate_hardware_evidence(evidence, [uuid])

    def test_release_ters_adapter_has_no_lifecycle_or_presentation_override(self):
        source = (WORKLOAD_DIR / "run_ters_workload.py").read_text(encoding="utf-8")
        for forbidden in (
            ".reset_meep(",
            "np.savetxt =",
            "Simulation.plot2D =",
            "plt.savefig =",
            "FixedDateTime",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_conservative_speedup_uses_worst_paired_repeat(self):
        result = MATRIX.conservative_speedup([10.0, 12.0], [4.0, 6.0])
        self.assertEqual(result["conservative_speedup"], 2.0)
        self.assertEqual(result["median_speedup"], 2.2)
        with self.assertRaises(COMMON.WorkloadError):
            MATRIX.conservative_speedup([1.0], [0.0])

    def test_release_classification_and_timing_stability_are_fail_closed(self):
        stable = MATRIX.timing_stability([10.0, 10.1, 9.9], 0.15)
        unstable = MATRIX.timing_stability([5.0, 10.0, 20.0], 0.15)
        self.assertEqual(stable["outcome"], "PASS")
        self.assertEqual(unstable["outcome"], "FAIL")
        passing = {"speed": {"outcome": "PASS"}}
        self.assertEqual(
            MATRIX.classify_outcome(True, passing, {"lane": stable}), "PASS"
        )
        self.assertEqual(
            MATRIX.classify_outcome(False, passing, {}), "DEVELOPMENT_ONLY"
        )
        self.assertEqual(
            MATRIX.classify_outcome(True, passing, {"lane": unstable}), "FAIL"
        )
        with self.assertRaisesRegex(COMMON.WorkloadError, "at least three"):
            MATRIX.timing_stability([1.0, 1.0], 0.15)

    def test_end_to_end_speed_gate_catches_hidden_overhead(self):
        fdtd = MATRIX.conservative_speedup([10.0, 10.0, 10.0], [4.0, 4.0, 4.0])
        end_to_end = MATRIX.conservative_speedup(
            [12.0, 12.0, 12.0], [14.0, 14.0, 14.0]
        )
        fdtd["outcome"] = "PASS" if fdtd["conservative_speedup"] >= 1.5 else "FAIL"
        end_to_end["outcome"] = (
            "PASS" if end_to_end["conservative_speedup"] >= 1.5 else "FAIL"
        )
        self.assertEqual(
            MATRIX.classify_outcome(
                True,
                {"fdtd": fdtd, "end-to-end": end_to_end},
                {"lane": MATRIX.timing_stability([1.0, 1.0, 1.0], 0.15)},
            ),
            "FAIL",
        )

    def test_rank_count_contaminated_end_to_end_is_diagnostic_not_release_gate(self):
        fdtd = MATRIX.conservative_speedup([10.0, 10.0, 10.0], [4.0, 4.0, 4.0])
        fdtd["minimum"] = 1.5
        fdtd["outcome"] = "PASS"
        diagnostic = MATRIX.conservative_speedup(
            [100.0, 100.0, 100.0], [120.0, 120.0, 120.0]
        )
        self.assertLess(diagnostic["conservative_speedup"], 1.0)
        self.assertEqual(
            MATRIX.classify_outcome(
                True,
                {"fdtd": fdtd},
                {"cpu:fdtd": MATRIX.timing_stability([10.0] * 3, 0.15)},
            ),
            "PASS",
        )

    def test_partial_failure_evidence_seals_lane_and_comparison_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            paths = (
                output / "lanes" / "lane" / "summary.json",
                output / "lanes" / "lane" / "COMPLETE",
                output / "comparisons" / "fp32" / "report.json",
                output / "comparisons" / "fp32" / "COMPLETE",
                output / "logs" / "lane.log",
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            records = MATRIX.collect_partial_evidence(output)
            self.assertEqual(
                {record["path"] for record in records},
                {path.relative_to(output).as_posix() for path in paths},
            )

    def test_checkpoint_contract_is_exact_and_invocation_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "input.py"
            source.write_bytes(b"fixture\n")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            receipt = {
                "receipt_id": "1" * 64,
                "build_input_id": "2" * 64,
                "artifact_set_id": "3" * 64,
                "source_start": {"sha256": "4" * 64},
            }
            build = MATRIX.Build(
                "fixture",
                root / "python",
                root / "mpiexec",
                root / "receipt.json",
                receipt,
                root / "pythonpath",
                root / "lib",
                True,
            )
            args = argparse.Namespace(
                workload="ters",
                repeats=3,
                comparison_timeout_seconds=10.0,
                stdout_limit_mib=2,
                minimum_one_gpu_speedup=1.5,
                minimum_two_gpu_speedup=2.0,
                minimum_multi_gpu_scaling=1.1,
                maximum_timing_cv=0.15,
            )
            contract = MATRIX.checkpoint_contract(
                args,
                source.resolve(),
                source_sha,
                {"controller": {"sha256": "5" * 64}},
                build,
                build,
                8,
                ("GPU-00112233-4455-6677-8899-aabbccddeeff",),
                20.0,
            )
            MATRIX.write_checkpoint(root, contract, {"hostname": "fixture"}, [], [])
            self.assertEqual(MATRIX.load_checkpoint(root, contract)["contract"], contract)
            changed = copy.deepcopy(contract)
            changed["repeats"] = 4
            with self.assertRaisesRegex(COMMON.WorkloadError, "invocation differs"):
                MATRIX.load_checkpoint(root, changed)

    def test_resume_journal_uses_only_validated_checkpoint_prefix(self):
        input_record = {"path": "/fixture", "size_bytes": 1, "sha256": "a" * 64}
        evidence = {"controller": {"path": "controller", "size_bytes": 1, "sha256": "b" * 64}}
        process = {"command": ["fixture"]}
        complete = {"path": "lane/COMPLETE", "size_bytes": 1, "sha256": "c" * 64}
        entry = {
            "lane": "cpu-fp64-8r",
            "run": {
                "repeat": 0,
                "process": process,
                "complete": complete,
            },
        }
        expected = [
            {"state": "controller-started", "unix_seconds": 1.0},
            {"state": "lane-started", "repeat": 0, "lane": entry["lane"]},
            {
                "state": "lane-process-ended",
                "repeat": 0,
                "lane": entry["lane"],
                "process": process,
            },
            {
                "state": "lane-validated",
                "repeat": 0,
                "lane": entry["lane"],
                "complete": complete,
            },
        ]
        journal = {
            "schema": "gpmeep-user-workload-matrix-journal-v2",
            "input": input_record,
            **evidence,
            "events": [
                *expected,
                {"state": "lane-started", "repeat": 0, "lane": "cpu-fp32-8r"},
            ],
        }
        self.assertEqual(
            MATRIX.validate_resume_journal_prefix(
                journal, input_record, evidence, [entry], []
            ),
            expected,
        )
        journal["events"][2]["lane"] = "tampered"
        with self.assertRaisesRegex(COMMON.WorkloadError, "disagrees"):
            MATRIX.validate_resume_journal_prefix(
                journal, input_record, evidence, [entry], []
            )

    def test_resume_archive_preserves_partial_attempt_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            COMMON.atomic_write_json(output / "JOURNAL.json", {"events": []})
            COMMON.atomic_write_json(output / "FAILED.json", {"error": "fixture"})
            partial = pathlib.PurePosixPath("lanes", "lane", "repeat-00")
            partial_file = output.joinpath(*partial.parts) / "partial.dat"
            partial_file.parent.mkdir(parents=True)
            partial_file.write_bytes(b"partial")
            checkpoint = output / "CHECKPOINT.json"
            checkpoint.write_bytes(b"checkpoint")
            attempt = MATRIX.archive_interrupted_attempt(output, [partial])
            self.assertEqual((attempt / partial / "partial.dat").read_bytes(), b"partial")
            self.assertTrue((attempt / "FAILED.json").is_file())
            self.assertTrue((attempt / "JOURNAL.json").is_file())
            self.assertFalse((output / "FAILED.json").exists())
            self.assertFalse(output.joinpath(*partial.parts).exists())
            self.assertEqual(checkpoint.read_bytes(), b"checkpoint")

    def test_controller_resume_skips_validated_lane_and_finishes_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "input.py"
            source.write_bytes(b"fixture input\n")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            output = root / "matrix"
            prefix = root / "prefix"
            (prefix / "bin").mkdir(parents=True)
            (prefix / "lib").mkdir()
            cpu_receipt = {
                "receipt_id": "1" * 64,
                "build_input_id": "2" * 64,
                "artifact_set_id": "3" * 64,
                "source_start": {"sha256": "4" * 64},
            }
            fp32_receipt = {
                "receipt_id": "5" * 64,
                "build_input_id": "6" * 64,
                "artifact_set_id": "7" * 64,
                "source_start": {"sha256": "4" * 64},
            }
            cpu_build = MATRIX.Build(
                "cpu-fp64",
                prefix / "bin" / "python",
                prefix / "bin" / "mpiexec",
                root / "cpu-receipt.json",
                cpu_receipt,
                root / "cpu-pythonpath",
                root / "cpu-lib",
                False,
            )
            fp32_build = MATRIX.Build(
                "shared-fp32",
                prefix / "bin" / "python",
                prefix / "bin" / "mpiexec",
                root / "fp32-receipt.json",
                fp32_receipt,
                root / "fp32-pythonpath",
                root / "fp32-lib",
                True,
            )
            gpu_uuids = (
                "GPU-00112233-4455-6677-8899-aabbccddeeff",
                "GPU-ffeeddcc-bbaa-9988-7766-554433221100",
            )
            hardware = {
                "captured_unix_seconds": 1.0,
                "hostname": "fixture-host",
                "kernel": "fixture-kernel",
                "cpu_model": "fixture-cpu",
                "visible_logical_cpus": [0, 1],
                "visible_physical_cores": 2,
                "cpu_governors": ["performance"],
                "load_average": [0.0, 0.0, 0.0],
                "memory_bytes": {},
                "gpus": [
                    {
                        "index": index,
                        "uuid": uuid,
                        "name": "fixture-gpu",
                        "driver_version": "1",
                        "compute_capability": "8.6",
                        "memory_total_mib": 1,
                    }
                    for index, uuid in enumerate(gpu_uuids)
                ],
                "selected_gpu_compute_processes": [],
            }
            failed_once = False
            successful_lane_outputs = []

            def fake_run_bounded(command, _environment, log_path, *_limits):
                nonlocal failed_once
                is_comparison = any("compare_user_workloads.py" in value for value in command)
                target = pathlib.Path(command[command.index("--output") + 1])
                if not is_comparison and len(successful_lane_outputs) == 1 and not failed_once:
                    failed_once = True
                    raise COMMON.WorkloadError("simulated interruption")
                target.mkdir(parents=True)
                if is_comparison:
                    COMMON.atomic_write_json(
                        target / "COMPLETE", {"outcome": "PASS"}
                    )
                else:
                    (target / "COMPLETE").write_bytes(b"lane complete\n")
                    successful_lane_outputs.append(target.relative_to(output).as_posix())
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_bytes(b"ok\n")
                return {
                    "command": command,
                    "command_pid": 1,
                    "returncode": 0,
                    "timed_out": False,
                    "output_limited": False,
                    "output_bytes": 3,
                    "peak_process_group_rss_bytes": 1,
                    "started_unix_seconds": 1.0,
                    "ended_unix_seconds": 2.0,
                    "wall_seconds": 1.0,
                    "log": COMMON.file_record(log_path, output),
                }

            def fake_lane_validation(lane, _output, _workload):
                walls = {
                    "cpu-fp64-2r": 12.0,
                    "cpu-fp32-2r": 10.0,
                    "cuda-fp32-1g": 4.0,
                    "cuda-fp32-2g": 3.0,
                }
                wall = walls[lane.label]
                return {
                    "fdtd_wall_seconds": wall,
                    "workload_end_to_end_seconds": wall,
                    "phase_wall_seconds": [wall / 2, wall / 2],
                    "timestep_contract": [
                        {"timestep_delta": 10, "meep_time": 1.0},
                        {"timestep_delta": 10, "meep_time": 1.0},
                    ],
                    "provenance": {"lane": lane.label},
                }

            argv = [
                "--workload",
                "ters",
                "--input",
                str(source),
                "--output",
                str(output),
                "--cpu-fp64-python",
                str(cpu_build.python),
                "--cpu-fp64-mpiexec",
                str(cpu_build.mpiexec),
                "--cpu-fp64-receipt",
                str(cpu_build.receipt_path),
                "--fp32-python",
                str(fp32_build.python),
                "--fp32-mpiexec",
                str(fp32_build.mpiexec),
                "--fp32-receipt",
                str(fp32_build.receipt_path),
                "--cpu-ranks",
                "2",
                "--gpu-devices",
                ",".join(gpu_uuids),
                "--timeout-seconds",
                "20",
            ]

            def fake_load_build(name, *_args, **_kwargs):
                return cpu_build if name == "cpu-fp64" else fp32_build

            with mock.patch.object(MATRIX, "TERS_SHA256", source_sha), mock.patch.object(
                MATRIX, "load_build", side_effect=fake_load_build
            ), mock.patch.object(
                MATRIX, "physical_core_count", return_value=2
            ), mock.patch.object(
                MATRIX, "hardware_snapshot", return_value=hardware
            ), mock.patch.object(
                MATRIX, "validate_hardware_evidence", return_value=None
            ), mock.patch.object(
                MATRIX, "run_bounded", side_effect=fake_run_bounded
            ), mock.patch.object(
                MATRIX, "validate_lane_output", side_effect=fake_lane_validation
            ), mock.patch.object(
                MATRIX, "verify_matrix_complete", return_value={"outcome": "PASS"}
            ):
                with self.assertRaisesRegex(COMMON.WorkloadError, "simulated interruption"):
                    MATRIX.main(argv)
                checkpoint = json.loads(
                    (output / "CHECKPOINT.json").read_text(encoding="utf-8")
                )
                self.assertEqual(len(checkpoint["completed_lanes"]), 1)
                first_output = successful_lane_outputs[0]
                self.assertEqual(MATRIX.main([*argv, "--resume"]), 0)

            self.assertEqual(len(successful_lane_outputs), 12)
            self.assertEqual(successful_lane_outputs.count(first_output), 1)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["resume_history"]), 1)
            self.assertTrue((output / "COMPLETE").is_file())

    def test_bounded_process_success_timeout_and_output_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            environment = dict(os.environ)
            success = MATRIX.run_bounded(
                [sys.executable, "-c", "print('ok')"],
                environment,
                self.log_path(root, "success.log"),
                5.0,
                1024,
            )
            self.assertEqual(success["returncode"], 0)
            self.assertFalse(success["timed_out"])
            self.assertFalse(success["output_limited"])

            timeout = MATRIX.run_bounded(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                environment,
                self.log_path(root, "timeout.log"),
                0.1,
                1024,
            )
            self.assertTrue(timeout["timed_out"])
            self.assertFalse(MATRIX.process_group_alive(timeout["command_pid"]))

            limited = MATRIX.run_bounded(
                [
                    sys.executable,
                    "-c",
                    "import os,time; os.write(1,b'x'*8192); time.sleep(30)",
                ],
                environment,
                self.log_path(root, "limited.log"),
                5.0,
                1024,
            )
            self.assertTrue(limited["output_limited"])
            self.assertEqual(limited["output_bytes"], 1024)
            self.assertEqual(self.log_path(root, "limited.log").stat().st_size, 1024)

    def test_timeout_kills_descendant_after_launcher_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            program = (
                "import subprocess,sys; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                "print(child.pid,flush=True)"
            )
            result = MATRIX.run_bounded(
                [sys.executable, "-c", program],
                dict(os.environ),
                self.log_path(root, "descendant.log"),
                0.1,
                1024,
            )
            self.assertTrue(result["timed_out"])
            self.assertFalse(MATRIX.process_group_alive(result["command_pid"]))

    def test_controller_sigterm_kills_active_process_group(self):
        import threading
        import time

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)

            def interrupt():
                time.sleep(0.15)
                os.kill(os.getpid(), MATRIX.signal.SIGTERM)

            sender = threading.Thread(target=interrupt)
            sender.start()
            with self.assertRaisesRegex(COMMON.WorkloadError, "received SIGTERM"):
                MATRIX.run_bounded(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    dict(os.environ),
                    self.log_path(root, "signal.log"),
                    5.0,
                    1024,
                )
            sender.join(timeout=2.0)
            self.assertFalse(sender.is_alive())

    def test_top_complete_replays_nested_evidence_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            repeats = MATRIX.RELEASE_REPEATS
            cpu_ranks = 2
            labels = (
                "cpu-fp64-2r",
                "cpu-fp32-2r",
                "cuda-fp32-1g",
                "cuda-fp32-2g",
            )
            roles = {
                labels[0]: ("cpu", False, 2, 12.0, 12.0),
                labels[1]: ("cpu", True, 2, 10.0, 10.0),
                labels[2]: ("cuda", True, 1, 4.0, 5.0),
                labels[3]: ("cuda", True, 2, 3.0, 4.0),
            }
            gpu_uuids = (
                "GPU-00112233-4455-6677-8899-aabbccddeeff",
                "GPU-ffeeddcc-bbaa-9988-7766-554433221100",
            )
            normalized_gpus = tuple(
                value.lower().removeprefix("gpu-").replace("-", "")
                for value in gpu_uuids
            )
            source_snapshot = "a" * 64
            lane_objects = {}
            provenances = {}
            lane_runs = {label: [] for label in labels}
            schedule = []
            events = [{"state": "controller-started", "unix_seconds": 1.0}]

            def process_record(log_relative, wall):
                log = output / log_relative
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_bytes(b"x")
                return {
                    "command": ["fixture"],
                    "command_pid": 123,
                    "returncode": 0,
                    "timed_out": False,
                    "output_limited": False,
                    "output_bytes": 1,
                    "peak_process_group_rss_bytes": 1024,
                    "started_unix_seconds": 1.0,
                    "ended_unix_seconds": 1.0 + wall,
                    "wall_seconds": wall,
                    "log": COMMON.file_record(log, output),
                }

            for repeat in range(repeats):
                order = labels if repeat % 2 == 0 else tuple(reversed(labels))
                for label in order:
                    backend, precision, mpi_size, fdtd_wall, process_wall = roles[label]
                    lane = output / "lanes" / label / f"repeat-{repeat:02d}"
                    lane.mkdir(parents=True)
                    COMMON.atomic_write_json(lane / "COMPLETE", {"fixture": True})
                    records = [
                        {
                            "run_index": index,
                            "timestep_delta": 10,
                            "meep_time": 1.0,
                            "wall_seconds": fdtd_wall / 2,
                        }
                        for index in range(2)
                    ]
                    summary = {
                        "expected_backend": backend,
                        "single_precision": precision,
                        "mpi_size": mpi_size,
                        "phase_contract": phase_contract(2),
                        "rank_records": [
                            {
                                "rank": rank,
                                "exact_end_to_end_seconds": process_wall,
                                "records": [dict(item) for item in records],
                            }
                            for rank in range(mpi_size)
                        ],
                    }
                    provenance = {
                        "receipt_id": (
                            "f" * 64 if label == labels[0] else "b" * 64
                        ),
                        "source_start": {"sha256": source_snapshot},
                        "gpu_devices": (
                            {}
                            if backend == "cpu"
                            else {
                                rank: normalized_gpus[rank]
                                for rank in range(mpi_size)
                            }
                        ),
                    }
                    lane_objects[lane.resolve()] = {"root": lane, "summary": summary}
                    provenances[lane.resolve()] = provenance
                    process = process_record(
                        f"logs/{label}/repeat-{repeat:02d}.log", process_wall
                    )
                    metrics = MATRIX.derive_lane_metrics(summary, "ters", label)
                    run = {
                        "repeat": repeat,
                        "output": lane.relative_to(output).as_posix(),
                        "process": process,
                        "hardware": {},
                        **metrics,
                        "provenance": provenance,
                        "complete": COMMON.file_record(lane / "COMPLETE", output),
                    }
                    lane_runs[label].append(run)
                    schedule.append({"repeat": repeat, "lane": label})
                    events.extend(
                        (
                            {"state": "lane-started", "repeat": repeat, "lane": label},
                            {
                                "state": "lane-process-ended",
                                "repeat": repeat,
                                "lane": label,
                                "process": process,
                            },
                            {
                                "state": "lane-validated",
                                "repeat": repeat,
                                "lane": label,
                                "complete": run["complete"],
                            },
                        )
                    )

            def comparison_report(args):
                return {
                    "schema": "fixture-comparison",
                    "reference": str(args.reference),
                    "candidates": [label for label, _, _ in args.candidate],
                    "outcome": "PASS",
                }

            comparisons = []
            for repeat in range(repeats):
                for (
                    precision,
                    reference_label,
                    reference_repeat,
                    candidate_specs,
                    relative,
                ) in MATRIX.comparison_specs(labels, repeat):
                    comparison = output.joinpath(*relative.parts)
                    comparison.mkdir(parents=True)
                    args = argparse.Namespace(
                        reference=(
                            output
                            / lane_runs[reference_label][reference_repeat]["output"]
                        ),
                        candidate=[
                            (
                                label,
                                comparison_class,
                                output / lane_runs[label][repeat]["output"],
                            )
                            for label, comparison_class in candidate_specs
                        ],
                    )
                    COMMON.atomic_write_json(
                        comparison / "report.json", comparison_report(args)
                    )
                    (comparison / "report.md").write_text("fixture\n", encoding="utf-8")
                    COMMON.atomic_write_json(
                        comparison / "COMPLETE",
                        {
                            "schema": "gpmeep-user-workload-comparison-complete-v2",
                            "outcome": "PASS",
                            "report": COMMON.file_record(
                                comparison / "report.json", comparison
                            ),
                            "markdown": COMMON.file_record(
                                comparison / "report.md", comparison
                            ),
                        },
                    )
                    process = process_record(
                        f"logs/comparisons/{precision}-repeat-{repeat:02d}.log", 1.0
                    )
                    record = {
                        "precision": precision,
                        "repeat": repeat,
                        "reference_repeat": reference_repeat,
                        "output": comparison.relative_to(output).as_posix(),
                        "process": process,
                        "complete": COMMON.file_record(
                            comparison / "COMPLETE", output
                        ),
                    }
                    comparisons.append(record)
                    events.extend(
                        (
                            {
                                "state": "comparison-process-ended",
                                "precision": precision,
                                "repeat": repeat,
                                "process": process,
                            },
                            {
                                "state": "comparison-validated",
                                "precision": precision,
                                "repeat": repeat,
                                "complete": record["complete"],
                            },
                        )
                    )
            evidence_code = {
                "controller": COMMON.file_record(
                    pathlib.Path(MATRIX.__file__), MATRIX.REPO
                ),
                "comparator": COMMON.file_record(
                    WORKLOAD_DIR / "compare_user_workloads.py", MATRIX.REPO
                ),
                "common": COMMON.file_record(
                    WORKLOAD_DIR / "common.py", MATRIX.REPO
                ),
                "ters_adapter": COMMON.file_record(
                    WORKLOAD_DIR / "run_ters_workload.py", MATRIX.REPO
                ),
                "aunp_adapter": COMMON.file_record(
                    WORKLOAD_DIR / "run_aunp_workload.py", MATRIX.REPO
                ),
                "aunp_exact_resume": COMMON.file_record(
                    WORKLOAD_DIR / "aunp_exact_resume.py", MATRIX.REPO
                ),
            }
            input_path = output / "input.py"
            input_path.write_text("fixture\n", encoding="utf-8")
            input_record = COMMON.absolute_file_record(input_path, "fixture input")
            fdtd_timings = {
                label: [run["fdtd_wall_seconds"] for run in lane_runs[label]]
                for label in labels
            }
            process_timings = {
                label: [run["workload_end_to_end_seconds"] for run in lane_runs[label]]
                for label in labels
            }
            specs = {
                "fdtd-cpu-fp32-to-1gpu": (fdtd_timings[labels[1]], fdtd_timings[labels[2]], 1.5),
                "fdtd-cpu-fp32-to-2gpu": (fdtd_timings[labels[1]], fdtd_timings[labels[3]], 2.0),
                "fdtd-one-gpu-to-two-gpu": (fdtd_timings[labels[2]], fdtd_timings[labels[3]], 1.1),
            }
            performance = {}
            for name, (reference, candidate, minimum) in specs.items():
                performance[name] = {
                    **MATRIX.conservative_speedup(reference, candidate),
                    "minimum": minimum,
                    "outcome": "PASS",
                }
            variability = {
                f"{label}:fdtd": MATRIX.timing_stability(fdtd_timings[label], 0.15)
                for label in labels
            }
            diagnostic_performance = {
                "workload-end-to-end-cpu-fp32-to-1gpu": MATRIX.conservative_speedup(
                    process_timings[labels[1]], process_timings[labels[2]]
                ),
                "workload-end-to-end-cpu-fp32-to-2gpu": MATRIX.conservative_speedup(
                    process_timings[labels[1]], process_timings[labels[3]]
                ),
                "workload-end-to-end-one-gpu-to-two-gpu": MATRIX.conservative_speedup(
                    process_timings[labels[2]], process_timings[labels[3]]
                ),
            }
            diagnostic_variability = {
                f"{label}:workload-end-to-end": MATRIX.timing_stability(
                    process_timings[label], 0.15
                )
                for label in labels
            }
            checkpoint_lanes = [
                {
                    "lane": scheduled["lane"],
                    "run": lane_runs[scheduled["lane"]][scheduled["repeat"]],
                }
                for scheduled in schedule
            ]
            checkpoint = {
                "schema": MATRIX.CHECKPOINT_SCHEMA,
                "contract": {
                    "workload": "ters",
                    "input": input_record,
                    "evidence_code": evidence_code,
                    "cpu_ranks": cpu_ranks,
                    "gpu_devices": list(gpu_uuids),
                    "repeats": repeats,
                    "release_eligible": True,
                },
                "hardware_before": {},
                "completed_lanes": checkpoint_lanes,
                "completed_comparisons": comparisons,
            }
            COMMON.atomic_write_json(output / "CHECKPOINT.json", checkpoint)
            report = {
                "schema": MATRIX.SCHEMA,
                "outcome": "PASS",
                "release_eligible": True,
                "repeats": repeats,
                "workload": "ters",
                "input": input_record,
                "evidence_code": evidence_code,
                "source_snapshot_sha256": source_snapshot,
                "cpu_physical_cores": cpu_ranks,
                "gpu_devices": list(gpu_uuids),
                "hardware": {"before": {}, "after": {}},
                "schedule": schedule,
                "workload_contract": lane_runs[labels[0]][0]["timestep_contract"],
                "lanes": lane_runs,
                "comparisons": comparisons,
                "recovery_checkpoint": COMMON.file_record(
                    output / "CHECKPOINT.json", output
                ),
                "resume_history": [],
                "performance": performance,
                "diagnostic_performance": diagnostic_performance,
                "timing_variability": variability,
                "diagnostic_timing_variability": diagnostic_variability,
                "diagnostic_timing_warning": "fixture warning",
            }
            COMMON.atomic_write_json(output / "report.json", report)
            (output / "report.md").write_text("fixture\n", encoding="utf-8")
            COMMON.atomic_write_json(
                output / "JOURNAL.json",
                {
                    "schema": "gpmeep-user-workload-matrix-journal-v2",
                    "input": input_record,
                    **evidence_code,
                    "events": events,
                },
            )
            COMMON.atomic_write_json(
                output / "COMPLETE",
                {
                    "schema": MATRIX.COMPLETE_SCHEMA,
                    "outcome": "PASS",
                    "report": COMMON.file_record(output / "report.json", output),
                    "markdown": COMMON.file_record(output / "report.md", output),
                    "journal": COMMON.file_record(output / "JOURNAL.json", output),
                },
            )
            with mock.patch.object(
                MATRIX.COMPARATOR,
                "load_lane",
                side_effect=lambda path, _schema: lane_objects[path.resolve()],
            ), mock.patch.object(
                MATRIX.COMPARATOR,
                "validate_lane_provenance",
                side_effect=lambda lane: provenances[lane["root"].resolve()],
            ), mock.patch.object(MATRIX.COMPARATOR, "_ters_files"), mock.patch.object(
                MATRIX.COMPARATOR, "compare_ters", side_effect=comparison_report
            ), mock.patch.object(
                MATRIX, "validate_hardware_evidence", return_value=None
            ), mock.patch.object(
                MATRIX, "TERS_SHA256", input_record["sha256"]
            ):
                self.assertEqual(
                    MATRIX.verify_matrix_complete(output)["outcome"], "PASS"
                )
                report["performance"]["fdtd-cpu-fp32-to-1gpu"][
                    "conservative_speedup"
                ] = 99.0
                COMMON.atomic_write_json(output / "report.json", report)
                COMMON.atomic_write_json(
                    output / "COMPLETE",
                    {
                        "schema": MATRIX.COMPLETE_SCHEMA,
                        "outcome": "PASS",
                        "report": COMMON.file_record(output / "report.json", output),
                        "markdown": COMMON.file_record(output / "report.md", output),
                        "journal": COMMON.file_record(output / "JOURNAL.json", output),
                    },
                )
                with self.assertRaisesRegex(COMMON.WorkloadError, "re-derived"):
                    MATRIX.verify_matrix_complete(output)


class HybridTersMatrixTests(unittest.TestCase):
    @staticmethod
    def fixture_build(root, name, receipt_digit, *, single_precision):
        return MATRIX.Build(
            name,
            root / "prefix" / "bin" / "python",
            root / "prefix" / "bin" / "mpiexec",
            root / f"{name}-receipt.json",
            {
                "receipt_id": receipt_digit * 64,
                "build_input_id": str((int(receipt_digit) + 1) % 10) * 64,
                "artifact_set_id": str((int(receipt_digit) + 2) % 10) * 64,
                "source_start": {"sha256": "a" * 64},
            },
            root / f"{name}-pythonpath",
            root / f"{name}-lib",
            single_precision,
        )

    def test_task_schedule_separates_exact_and_performance_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu = self.fixture_build(root, "cpu", "1", single_precision=False)
            fp32 = self.fixture_build(root, "fp32", "4", single_precision=True)
            devices = (
                "GPU-00112233-4455-6677-8899-aabbccddeeff",
                "GPU-ffeeddcc-bbaa-9988-7766-554433221100",
            )
            tasks = HYBRID.build_tasks(cpu, fp32, 8, devices, 5)
            self.assertEqual(len(tasks), 18)
            self.assertEqual(
                [task.measurement_mode for task in tasks[:3]], ["exact"] * 3
            )
            self.assertEqual(
                {task.role for task in tasks[:3]},
                {
                    "exact-cpu-fp64-8r",
                    "exact-cuda-fp32-1g",
                    "exact-cuda-fp32-2g",
                },
            )
            self.assertTrue(
                all(
                    task.measurement_mode == "fixed-meep-time-performance-window"
                    for task in tasks[3:]
                )
            )
            repeat_one = [task.role for task in tasks if task.repeat == 1]
            self.assertEqual(
                repeat_one,
                [
                    "performance-cuda-fp32-2g",
                    "performance-cuda-fp32-1g",
                    "performance-cpu-fp32-8r",
                ],
            )

    def test_lane_mode_cannot_confuse_window_with_exact_evidence(self):
        task = mock.Mock(measurement_mode="exact", role="exact")
        HYBRID.validate_lane_mode(
            {
                "measurement_mode": "exact",
                "performance_window_meep_time": None,
                "physics_overrides": [],
            },
            task,
            COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME,
        )
        with self.assertRaisesRegex(COMMON.WorkloadError, "wrong measurement mode"):
            HYBRID.validate_lane_mode(
                {
                    "measurement_mode": "fixed-meep-time-performance-window",
                    "performance_window_meep_time": (
                        COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
                    ),
                    "physics_overrides": ["adapted"],
                },
                task,
                COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME,
            )
        task.measurement_mode = "fixed-meep-time-performance-window"
        with self.assertRaisesRegex(COMMON.WorkloadError, "wrong performance window"):
            HYBRID.validate_lane_mode(
                {
                    "measurement_mode": task.measurement_mode,
                    "performance_window_meep_time": 0.1,
                    "physics_overrides": ["adapted"],
                },
                task,
                COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME,
            )

    def test_hybrid_rejects_subminimum_performance_window(self):
        self.assertEqual(HYBRID.DEFAULT_WINDOW_MEEP_TIME, 0.25)
        self.assertEqual(COMMON.MINIMUM_TERS_PERFORMANCE_TIMESTEPS_PER_PHASE, 3500)
        args = HYBRID.parse_args(
            [
                "--input", "input.py",
                "--output", "output",
                "--cpu-fp64-python", "cpu-python",
                "--cpu-fp64-mpiexec", "cpu-mpiexec",
                "--cpu-fp64-receipt", "cpu-receipt",
                "--fp32-python", "fp32-python",
                "--fp32-mpiexec", "fp32-mpiexec",
                "--fp32-receipt", "fp32-receipt",
                "--gpu-devices", "0" * 32 + "," + "1" * 32,
                "--performance-window-meep-time", "0.249999",
            ]
        )
        with self.assertRaisesRegex(COMMON.WorkloadError, "performance window"):
            HYBRID.validate_args(args)
        args.performance_window_meep_time = (
            COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
        )
        HYBRID.validate_args(args)

    def test_ters_window_crosses_first_automatic_dft_update(self):
        resolution = 7000
        fcen = 1 / 0.6328
        fmin = 1 / 0.8
        fmax = 1 / 0.4
        input_fwidth = 2 * max(fcen - fmin, fmax - fcen)
        source_width = 1 / input_fwidth
        source_fwidth = math.sqrt(-2 * math.log(1e-7)) / (
            math.pi * source_width
        )
        source_frequency_max = fcen + 0.5 * source_fwidth
        monitor_frequency_max = fcen
        dt = 0.5 / resolution
        decimation_factor = math.floor(
            1 / (dt * (source_frequency_max + monitor_frequency_max))
        )

        self.assertAlmostEqual(source_fwidth, 3.32436345415219, places=14)
        self.assertEqual(decimation_factor, 2902)
        short_steps = round(0.05 / dt)
        release_steps = round(
            COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME / dt
        )
        self.assertEqual(short_steps, 700)
        self.assertEqual(short_steps // decimation_factor, 0)
        self.assertEqual(release_steps, 3500)
        self.assertEqual(release_steps // decimation_factor, 1)
        self.assertEqual(release_steps - decimation_factor, 598)

    def test_release_performance_gates_cannot_be_resealed_weaker(self):
        valid = {
            "maximum_timing_cv": 0.15,
            "minimum_one_gpu_speedup": 1.5,
            "minimum_two_gpu_speedup": 2.0,
            "minimum_multi_gpu_scaling": 1.1,
        }
        HYBRID.validate_release_performance_gates(valid)
        for name, weakened in (
            ("maximum_timing_cv", 100.0),
            ("minimum_one_gpu_speedup", 0.0),
            ("minimum_two_gpu_speedup", 0.0),
            ("minimum_multi_gpu_scaling", 0.0),
        ):
            tampered = {**valid, name: weakened}
            with self.subTest(name=name), self.assertRaisesRegex(
                COMMON.WorkloadError, "performance gate"
            ):
                HYBRID.validate_release_performance_gates(tampered)

    def test_reconstructed_schedule_rejects_two_gpu_downgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cpu = self.fixture_build(root, "cpu", "1", single_precision=False)
            fp32 = self.fixture_build(root, "fp32", "4", single_precision=True)
            devices = (
                "GPU-00112233-4455-6677-8899-aabbccddeeff",
                "GPU-ffeeddcc-bbaa-9988-7766-554433221100",
            )
            tasks = HYBRID.build_tasks(cpu, fp32, 8, devices, 3)
            sealed = {
                "builds": {
                    "cpu_fp64": HYBRID.build_contract_record(cpu),
                    "shared_fp32": HYBRID.build_contract_record(fp32),
                },
                "cpu_ranks": 8,
                "gpu_devices": list(devices),
                "performance_repeats": 3,
                "tasks": [HYBRID.task_spec(task) for task in tasks],
            }

            def fake_load(name, *_args, **_kwargs):
                return cpu if name == "cpu-fp64" else fp32

            with mock.patch.object(MATRIX, "load_build", side_effect=fake_load):
                rebuilt = HYBRID.reconstruct_contract_tasks(sealed)
                self.assertEqual([HYBRID.task_spec(task) for task in rebuilt[3]], sealed["tasks"])
                sealed["tasks"][2]["ranks"] = 1
                sealed["tasks"][2]["devices"] = [devices[0]]
                with self.assertRaisesRegex(COMMON.WorkloadError, "not canonical"):
                    HYBRID.reconstruct_contract_tasks(sealed)

    def test_performance_report_requires_same_work_and_passes_strong_speedup(self):
        args = argparse.Namespace(
            performance_repeats=3,
            performance_window_meep_time=(
                COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
            ),
            maximum_timing_cv=0.15,
            minimum_one_gpu_speedup=1.5,
            minimum_two_gpu_speedup=2.0,
            minimum_multi_gpu_scaling=1.1,
        )
        completed = []
        roles = {
            "performance-cpu-fp32-8r": 10.0,
            "performance-cuda-fp32-1g": 4.0,
            "performance-cuda-fp32-2g": 2.0,
        }
        for repeat in range(3):
            for role, wall in roles.items():
                completed.append(
                    {
                        "task": {"role": role},
                        "fdtd_wall_seconds": wall,
                        "timestep_contract": [
                            {
                                "timestep_delta": 3500,
                                "meep_time": (
                                    COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
                                ),
                            },
                            {
                                "timestep_delta": 3500,
                                "meep_time": (
                                    COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
                                ),
                            },
                        ],
                    }
                )
        performance, stability = HYBRID.performance_report(args, completed, 8)
        self.assertTrue(all(value["outcome"] == "PASS" for value in performance.values()))
        self.assertTrue(all(value["outcome"] == "PASS" for value in stability.values()))
        completed[-1]["timestep_contract"][1]["timestep_delta"] = 3499
        with self.assertRaisesRegex(COMMON.WorkloadError, "minimum work contract"):
            HYBRID.performance_report(args, completed, 8)

    def test_hybrid_controller_resumes_after_exact_lane(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "prefix" / "bin").mkdir(parents=True)
            (root / "prefix" / "lib").mkdir()
            source = root / "input.py"
            source.write_bytes(b"hybrid fixture\n")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            output = root / "evidence"
            cpu = self.fixture_build(root, "cpu", "1", single_precision=False)
            fp32 = self.fixture_build(root, "fp32", "4", single_precision=True)
            devices = (
                "GPU-00112233-4455-6677-8899-aabbccddeeff",
                "GPU-ffeeddcc-bbaa-9988-7766-554433221100",
            )
            hardware = {
                "captured_unix_seconds": 1.0,
                "hostname": "fixture",
                "kernel": "fixture",
                "cpu_model": "fixture",
                "visible_logical_cpus": [0, 1],
                "visible_physical_cores": 2,
                "cpu_governors": ["performance"],
                "load_average": [0.0, 0.0, 0.0],
                "memory_bytes": {},
                "gpus": [
                    {
                        "index": index,
                        "uuid": uuid,
                        "name": "fixture",
                        "driver_version": "1",
                        "compute_capability": "8.6",
                        "memory_total_mib": 1,
                    }
                    for index, uuid in enumerate(devices)
                ],
                "selected_gpu_compute_processes": [],
            }
            failed_once = False
            successful_lanes = []
            lane_objects = {}
            lane_provenances = {}

            def fake_run(command, _environment, log_path, *_limits):
                nonlocal failed_once
                target = pathlib.Path(command[command.index("--output") + 1])
                comparison = any("compare_user_workloads.py" in item for item in command)
                if not comparison and len(successful_lanes) == 1 and not failed_once:
                    failed_once = True
                    raise COMMON.WorkloadError("hybrid interruption")
                target.mkdir(parents=True)
                if comparison:
                    COMMON.atomic_write_json(target / "COMPLETE", {"outcome": "PASS"})
                else:
                    (target / "COMPLETE").write_bytes(b"complete\n")
                    successful_lanes.append(target.relative_to(output).as_posix())
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_bytes(b"ok\n")
                return {
                    "command": command,
                    "command_pid": 1,
                    "returncode": 0,
                    "timed_out": False,
                    "output_limited": False,
                    "output_bytes": 3,
                    "peak_process_group_rss_bytes": 1,
                    "started_unix_seconds": 1.0,
                    "ended_unix_seconds": 2.0,
                    "wall_seconds": 1.0,
                    "log": COMMON.file_record(log_path, output),
                }

            def fake_validate(lane, lane_output, _workload):
                role = lane_output.parent.name
                performance = role.startswith("performance-")
                if "cpu" in role:
                    wall = 10.0 if performance else 12.0
                elif "1g" in role:
                    wall = 4.0
                else:
                    wall = 2.0
                mode = (
                    COMMON.TERS_PERFORMANCE_MEASUREMENT_MODE if performance else "exact"
                )
                timestep_contract = [
                    {
                        "timestep_delta": 3500 if performance else 70000,
                        "meep_time": (
                            COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
                            if performance
                            else 5.0
                        ),
                    }
                    for _ in range(2)
                ]
                rank_records = [
                    {
                        "rank": rank,
                        "exact_end_to_end_seconds": wall,
                        "records": [
                            {
                                **phase,
                                "run_index": index,
                                "wall_seconds": wall / 2,
                            }
                            for index, phase in enumerate(timestep_contract)
                        ],
                    }
                    for rank in range(lane.ranks)
                ]
                summary = {
                    "expected_backend": lane.backend,
                    "single_precision": lane.build.single_precision,
                    "mpi_size": lane.ranks,
                    "phase_contract": phase_contract(2),
                    "rank_records": rank_records,
                    "measurement_mode": mode,
                    "performance_window_meep_time": (
                        COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
                        if performance
                        else None
                    ),
                    "physics_overrides": (
                        [COMMON.TERS_PERFORMANCE_ADAPTATION] if performance else []
                    ),
                }
                provenance = {
                    "receipt_id": lane.build.receipt["receipt_id"],
                    "build_input_id": lane.build.receipt["build_input_id"],
                    "artifact_set_id": lane.build.receipt["artifact_set_id"],
                    "source_start": lane.build.receipt["source_start"],
                    "adapter_sha256": "b" * 64,
                    "common_sha256": "c" * 64,
                    "gpu_devices": {
                        rank: lane.devices[rank]
                        .lower()
                        .removeprefix("gpu-")
                        .replace("-", "")
                        for rank in range(lane.ranks)
                    }
                    if lane.backend == "cuda"
                    else {},
                    "physical_cores": [["fixture", 0, rank] for rank in range(lane.ranks)],
                }
                lane_objects[lane_output.resolve()] = {
                    "root": lane_output.resolve(),
                    "summary": summary,
                }
                lane_provenances[lane_output.resolve()] = provenance
                return {
                    "summary": summary,
                    "fdtd_wall_seconds": wall,
                    "workload_end_to_end_seconds": wall,
                    "phase_wall_seconds": [wall / 2, wall / 2],
                    "timestep_contract": timestep_contract,
                    "provenance": provenance,
                }

            argv = [
                "--input",
                str(source),
                "--output",
                str(output),
                "--cpu-fp64-python",
                str(cpu.python),
                "--cpu-fp64-mpiexec",
                str(cpu.mpiexec),
                "--cpu-fp64-receipt",
                str(cpu.receipt_path),
                "--fp32-python",
                str(fp32.python),
                "--fp32-mpiexec",
                str(fp32.mpiexec),
                "--fp32-receipt",
                str(fp32.receipt_path),
                "--cpu-ranks",
                "2",
                "--gpu-devices",
                ",".join(devices),
                "--performance-repeats",
                "3",
            ]

            def fake_load(name, *_args, **_kwargs):
                return cpu if name == "cpu-fp64" else fp32

            with mock.patch.object(HYBRID, "TERS_SHA256", source_sha), mock.patch.object(
                MATRIX, "load_build", side_effect=fake_load
            ), mock.patch.object(
                MATRIX, "physical_core_count", return_value=2
            ), mock.patch.object(
                MATRIX, "hardware_snapshot", return_value=hardware
            ), mock.patch.object(
                MATRIX, "validate_hardware_evidence", return_value=None
            ), mock.patch.object(
                MATRIX, "run_bounded", side_effect=fake_run
            ), mock.patch.object(
                MATRIX, "validate_lane_output", side_effect=fake_validate
            ), mock.patch.object(
                MATRIX.COMPARATOR,
                "load_lane",
                side_effect=lambda path, _schema: lane_objects[path.resolve()],
            ), mock.patch.object(
                MATRIX.COMPARATOR,
                "validate_lane_provenance",
                side_effect=lambda lane: lane_provenances[lane["root"].resolve()],
            ), mock.patch.object(
                MATRIX.COMPARATOR, "_ters_files", return_value=({}, {})
            ), mock.patch.object(
                MATRIX.COMPARATOR, "compare_ters", return_value={"outcome": "PASS"}
            ):
                with self.assertRaisesRegex(COMMON.WorkloadError, "hybrid interruption"):
                    HYBRID.main(argv)
                checkpoint = json.loads(
                    (output / "CHECKPOINT.json").read_text(encoding="utf-8")
                )
                self.assertEqual(len(checkpoint["completed_tasks"]), 1)
                first = successful_lanes[0]
                self.assertEqual(HYBRID.main([*argv, "--resume"]), 0)
                self.assertEqual(HYBRID.verify_hybrid_complete(output)["outcome"], "PASS")

                original_report = json.loads(
                    (output / "report.json").read_text(encoding="utf-8")
                )
                original_checkpoint = json.loads(
                    (output / "CHECKPOINT.json").read_text(encoding="utf-8")
                )
                original_journal = json.loads(
                    (output / "JOURNAL.json").read_text(encoding="utf-8")
                )

                def publish(report, checkpoint, journal):
                    COMMON.atomic_write_json(output / "CHECKPOINT.json", checkpoint)
                    report["recovery_checkpoint"] = COMMON.file_record(
                        output / "CHECKPOINT.json", output
                    )
                    COMMON.atomic_write_json(output / "report.json", report)
                    COMMON.atomic_write_json(output / "JOURNAL.json", journal)
                    COMMON.atomic_write_json(
                        output / "COMPLETE",
                        {
                            "schema": HYBRID.COMPLETE_SCHEMA,
                            "outcome": "PASS",
                            "report": COMMON.file_record(output / "report.json", output),
                            "markdown": COMMON.file_record(output / "report.md", output),
                            "journal": COMMON.file_record(output / "JOURNAL.json", output),
                        },
                    )

                duplicate_report = copy.deepcopy(original_report)
                duplicate_checkpoint = copy.deepcopy(original_checkpoint)
                duplicate_journal = copy.deepcopy(original_journal)
                source_record = next(
                    record
                    for record in duplicate_report["tasks"]
                    if record["task"]["role"] == "performance-cuda-fp32-2g"
                    and record["task"]["repeat"] == 0
                )
                duplicate_record = next(
                    record
                    for record in duplicate_report["tasks"]
                    if record["task"]["role"] == "performance-cuda-fp32-2g"
                    and record["task"]["repeat"] == 2
                )
                duplicate_record["output"] = source_record["output"]
                duplicate_record["complete"] = source_record["complete"]
                duplicate_checkpoint["completed_tasks"] = copy.deepcopy(
                    duplicate_report["tasks"]
                )
                publish(duplicate_report, duplicate_checkpoint, duplicate_journal)
                with self.assertRaisesRegex(COMMON.WorkloadError, "output path"):
                    HYBRID.verify_hybrid_complete(output)

                topology_report = copy.deepcopy(original_report)
                topology_checkpoint = copy.deepcopy(original_checkpoint)
                topology_journal = copy.deepcopy(original_journal)
                old_spec = copy.deepcopy(topology_checkpoint["contract"]["tasks"][2])
                changed_spec = copy.deepcopy(old_spec)
                changed_spec["ranks"] = 1
                changed_spec["devices"] = [devices[0]]
                topology_checkpoint["contract"]["tasks"][2] = changed_spec
                topology_checkpoint["completed_tasks"][2]["task"] = changed_spec
                topology_report["tasks"][2]["task"] = changed_spec
                topology_journal["contract"] = copy.deepcopy(
                    topology_checkpoint["contract"]
                )
                for event in topology_journal["events"]:
                    if event.get("task") == old_spec:
                        event["task"] = copy.deepcopy(changed_spec)
                publish(topology_report, topology_checkpoint, topology_journal)
                with self.assertRaisesRegex(COMMON.WorkloadError, "not canonical"):
                    HYBRID.verify_hybrid_complete(output)

                gate_report = copy.deepcopy(original_report)
                gate_checkpoint = copy.deepcopy(original_checkpoint)
                gate_journal = copy.deepcopy(original_journal)
                gate_checkpoint["contract"]["gates"] = {
                    "maximum_timing_cv": 100.0,
                    "minimum_one_gpu_speedup": 0.0,
                    "minimum_two_gpu_speedup": 0.0,
                    "minimum_multi_gpu_scaling": 0.0,
                }
                gate_journal["contract"] = copy.deepcopy(gate_checkpoint["contract"])
                publish(gate_report, gate_checkpoint, gate_journal)
                with self.assertRaisesRegex(COMMON.WorkloadError, "performance gate"):
                    HYBRID.verify_hybrid_complete(output)

                publish(
                    copy.deepcopy(original_report),
                    copy.deepcopy(original_checkpoint),
                    copy.deepcopy(original_journal),
                )
                self.assertEqual(HYBRID.verify_hybrid_complete(output)["outcome"], "PASS")

            self.assertEqual(len(successful_lanes), 12)
            self.assertEqual(successful_lanes.count(first), 1)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["outcome"], "PASS")
            self.assertEqual(len(report["resume_history"]), 1)
            self.assertTrue((output / "COMPLETE").is_file())


class UserWorkloadComparisonTests(unittest.TestCase):
    @staticmethod
    def run_record(index, backend="cpu"):
        before = statistics_snapshot()
        after = statistics_snapshot(backend, 1)
        delta = COMMON.statistics_delta(before, after)
        return {
            "run_index": index,
            "active_backend": backend,
            "requested_backend": backend,
            "phase_calls": COMMON.phase_call_totals(delta),
            "phase_counters": COMMON.phase_counter_view(delta),
            "statistics_before": before,
            "statistics_after": after,
            "statistics_delta": delta,
            "timestep_delta": 10,
            "wall_seconds": 1.0,
            "meep_time": 1.0,
        }

    @staticmethod
    def ters_fixture(scale=1.0):
        x = np.linspace(-0.02, 0.02, 81)
        y = np.linspace(0.035, 0.045, 81)
        xx, yy = np.meshgrid(x, y)
        field = scale * np.exp(-((xx / 0.008) ** 2 + ((yy - 0.04) / 0.002) ** 2))
        field = field.astype(np.complex64) * np.exp(1j * 0.1)
        return field, x, y

    @staticmethod
    def ters_coordinates():
        return {
            "x": np.array([-0.02, 0.0, 0.02], dtype=np.float64),
            "y": np.array([0.035, 0.045], dtype=np.float64),
            "z": np.array([0.0], dtype=np.float64),
            "w": np.array(
                [[0.25, 0.5], [0.75, 1.0], [1.25, 1.5]], dtype=np.float64
            ),
        }

    def test_ters_coordinate_weight_policies_are_explicit_and_fail_closed(self):
        expected = {
            "cpu-fp64-fp32": ("symmetric-relative", 5.0e-7),
            "cuda-fp64-fp32": ("symmetric-relative", 5.0e-7),
            "cuda-same-fp32": ("symmetric-relative", 5.0e-7),
            "repeat-cpu-fp64": ("byte-exact", 0.0),
            "repeat-cpu-fp32": ("byte-exact", 0.0),
            "repeat-cuda-fp32": ("byte-exact", 0.0),
        }
        self.assertEqual(set(COMPARE.COMPARISON_CLASSES), set(expected))
        for name, (policy, rtol) in expected.items():
            contract = COMPARE.COMPARISON_CLASSES[name]
            self.assertEqual(contract["coordinate_weight_policy"], policy)
            self.assertEqual(contract["coordinate_weight_rtol"], rtol)

        malformed = dict(
            COMPARE.COMPARISON_CLASSES["repeat-cpu-fp64"],
            coordinate_weight_policy="symmetric-relative",
            coordinate_weight_rtol=0.0,
        )
        with mock.patch.dict(
            COMPARE.COMPARISON_CLASSES,
            {"repeat-cpu-fp64": malformed},
        ), self.assertRaisesRegex(COMMON.WorkloadError, "invalid coordinate-weight"):
            values = self.ters_coordinates()
            COMPARE._compare_ters_coordinates(
                values, copy.deepcopy(values), "repeat-cpu-fp64"
            )

    def test_ters_cross_topology_weights_accept_fp32_roundoff_only(self):
        reference = self.ters_coordinates()
        candidate = copy.deepcopy(reference)
        candidate["w"][0, 0] = float(
            np.nextafter(
                np.float32(reference["w"][0, 0]),
                np.float32(np.inf),
            )
        )
        for comparison_class in (
            "cpu-fp64-fp32",
            "cuda-fp64-fp32",
            "cuda-same-fp32",
        ):
            with self.subTest(comparison_class=comparison_class):
                result = COMPARE._compare_ters_coordinates(
                    reference, candidate, comparison_class
                )
                weights = result["cubature_weights"]
                self.assertEqual(result["outcome"], "PASS")
                self.assertEqual(weights["different_values"], 1)
                self.assertLessEqual(
                    weights["maximum_symmetric_relative_difference"],
                    weights["relative_tolerance"],
                )
                self.assertEqual(weights["absolute_tolerance"], 0.0)
                reversed_result = COMPARE._compare_ters_coordinates(
                    candidate, reference, comparison_class
                )
                self.assertEqual(
                    reversed_result["cubature_weights"][
                        "maximum_symmetric_relative_difference"
                    ],
                    weights["maximum_symmetric_relative_difference"],
                )

        excessive = copy.deepcopy(reference)
        excessive["w"][0, 0] *= 1.0 + 1.0e-6
        with self.assertRaisesRegex(COMMON.WorkloadError, "symmetric relative"):
            COMPARE._compare_ters_coordinates(
                reference, excessive, "cuda-fp64-fp32"
            )

    def test_ters_spatial_axes_remain_raw_byte_exact(self):
        reference = self.ters_coordinates()
        self.assertFalse(
            COMPARE._array_bit_exact(
                np.array([0.0], dtype=np.float64),
                np.array([-0.0], dtype=np.float64),
            )
        )
        mutations = {}

        one_ulp = copy.deepcopy(reference)
        one_ulp["x"][0] = np.nextafter(one_ulp["x"][0], np.inf)
        mutations["one-ulp"] = one_ulp

        changed_dtype = copy.deepcopy(reference)
        changed_dtype["x"] = changed_dtype["x"].astype(np.float32)
        mutations["dtype"] = changed_dtype

        changed_shape = copy.deepcopy(reference)
        changed_shape["x"] = changed_shape["x"].reshape(1, -1)
        mutations["shape"] = changed_shape

        signed_zero = copy.deepcopy(reference)
        signed_zero["z"][0] = -0.0
        mutations["signed-zero"] = signed_zero

        for name, candidate in mutations.items():
            with self.subTest(mutation=name), self.assertRaises(
                COMMON.WorkloadError
            ):
                COMPARE._compare_ters_coordinates(
                    reference, candidate, "cuda-fp64-fp32"
                )

    def test_ters_cubature_weight_structure_and_zero_are_fail_closed(self):
        reference = self.ters_coordinates()
        mutations = {}

        zero_to_nonzero_reference = copy.deepcopy(reference)
        zero_to_nonzero_reference["w"][0, 0] = 0.0
        mutations["zero-to-nonzero"] = (zero_to_nonzero_reference, reference)

        negative = copy.deepcopy(reference)
        negative["w"][0, 0] *= -1.0
        mutations["negative"] = (reference, negative)

        negative_zero = copy.deepcopy(reference)
        negative_zero["w"][0, 0] = -0.0
        mutations["negative-zero"] = (reference, negative_zero)

        changed_shape = copy.deepcopy(reference)
        changed_shape["w"] = changed_shape["w"].reshape(-1)
        mutations["shape"] = (reference, changed_shape)

        changed_dtype = copy.deepcopy(reference)
        changed_dtype["w"] = changed_dtype["w"].astype(np.float32)
        mutations["dtype"] = (reference, changed_dtype)

        both_changed_dtype = copy.deepcopy(reference)
        both_changed_dtype["w"] = both_changed_dtype["w"].astype(np.float32)
        mutations["both-dtype"] = (
            both_changed_dtype,
            copy.deepcopy(both_changed_dtype),
        )

        nonfinite = copy.deepcopy(reference)
        nonfinite["w"][0, 0] = np.nan
        mutations["nonfinite"] = (reference, nonfinite)

        infinite = copy.deepcopy(reference)
        infinite["w"][0, 0] = np.inf
        mutations["infinite"] = (reference, infinite)

        all_zero = copy.deepcopy(reference)
        all_zero["w"].fill(0.0)
        mutations["all-zero"] = (all_zero, copy.deepcopy(all_zero))

        for name, (left, right) in mutations.items():
            with self.subTest(mutation=name), self.assertRaises(
                COMMON.WorkloadError
            ):
                COMPARE._compare_ters_coordinates(
                    left, right, "cuda-fp64-fp32"
                )

    def test_ters_repeat_weights_require_byte_identity(self):
        reference = self.ters_coordinates()
        for comparison_class in (
            "repeat-cpu-fp64",
            "repeat-cpu-fp32",
            "repeat-cuda-fp32",
        ):
            with self.subTest(comparison_class=comparison_class):
                result = COMPARE._compare_ters_coordinates(
                    reference, copy.deepcopy(reference), comparison_class
                )
                self.assertEqual(result["cubature_weights"]["policy"], "byte-exact")

                changed = copy.deepcopy(reference)
                changed["w"][0, 0] = np.nextafter(changed["w"][0, 0], np.inf)
                with self.assertRaisesRegex(COMMON.WorkloadError, "byte-exact"):
                    COMPARE._compare_ters_coordinates(
                        reference, changed, comparison_class
                    )

    def test_ters_full_vector_and_gap_roi_pass_and_fail(self):
        reference, x, y = self.ters_fixture()
        candidate, _, _ = self.ters_fixture(1.0001)
        result = COMPARE.compare_ters_field(
            reference,
            candidate,
            x,
            y,
            COMPARE.PAIR_TOLERANCES["same-fp32"],
        )
        self.assertEqual(result["outcome"], "PASS")
        bad, _, _ = self.ters_fixture(1.1)
        result = COMPARE.compare_ters_field(
            reference,
            bad,
            x,
            y,
            COMPARE.PAIR_TOLERANCES["same-fp32"],
        )
        self.assertEqual(result["outcome"], "FAIL")
        self.assertIn("nrmse", result["failures"])

    def test_dft_coordinate_metadata_accepts_multidimensional_weights(self):
        x = np.linspace(-0.02, 0.02, 81)
        y = np.linspace(0.035, 0.045, 41)
        values = {
            "x": x,
            "y": y,
            "z": np.array([0.0]),
            "w": np.ones((x.size, y.size)),
        }
        COMMON.validate_dft_coordinate_metadata(values, np.isfinite)

        flattened = dict(values, w=values["w"].reshape(-1))
        with self.assertRaisesRegex(COMMON.WorkloadError, "cubature weights"):
            COMMON.validate_dft_coordinate_metadata(flattened, np.isfinite)

        multidimensional_axis = dict(values, x=x.reshape(9, 9))
        with self.assertRaisesRegex(COMMON.WorkloadError, "axis x"):
            COMMON.validate_dft_coordinate_metadata(
                multidimensional_axis, np.isfinite
            )

        nonfinite = dict(values, w=values["w"].copy())
        nonfinite["w"][0, 0] = np.nan
        with self.assertRaisesRegex(COMMON.WorkloadError, "cubature weights"):
            COMMON.validate_dft_coordinate_metadata(nonfinite, np.isfinite)

    def test_ters_binds_both_monitor_coordinate_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            field, x, y = self.ters_fixture()
            _records, source_sha256, results = original_ters_rank_outputs(root)
            self.enterContext(mock.patch.object(COMPARE, "TERS_SHA256", source_sha256))
            np.save(results / "withtip_dft.npy", field)
            np.save(results / "wotip_dft.npy", field)
            rank_original_outputs = [
                COMMON.file_record(path, root)
                for path in sorted(results.iterdir())
                if path.is_file()
            ]
            values = {
                "x": x,
                "y": y,
                "z": np.array([0.0]),
                "w": np.ones((x.size, y.size)),
            }
            evidence = COMPARE._coordinate_evidence(values)
            coordinate_records = []
            evidence_directory = root / "evidence"
            evidence_directory.mkdir()
            for run_index in range(2):
                path = evidence_directory / f"dft_coordinates_run{run_index}.npz"
                np.savez(path, **values)
                coordinate_records.append(
                    {
                        **COMMON.file_record(path, root),
                        "run_index": run_index,
                        "axes": evidence,
                    }
                )
            summary = {
                "schema": COMPARE.TERS_SCHEMA,
                "single_precision": True,
                "expected_backend": "cpu",
                "mpi_size": 1,
                "rank_records": [
                    {
                        "rank": 0,
                        "exact_end_to_end_seconds": 2.0,
                        "original_result_root": results.relative_to(root).as_posix(),
                        "original_outputs": rank_original_outputs,
                        "records": [
                            {**self.run_record(index), "dft_coordinates": evidence}
                            for index in range(2)
                        ],
                    }
                ],
                "phase_contract": phase_contract(2),
                "source": COMMON.absolute_file_record(root / "fixture.py", "fixture"),
                "source_sha256": source_sha256,
                "presentation_suppressed": False,
                "physics_overrides": [],
                "arrays": [
                    COMMON.file_record(results / "withtip_dft.npy", root),
                    COMMON.file_record(results / "wotip_dft.npy", root),
                ],
                "coordinates": coordinate_records,
                "original_outputs": [
                    COMMON.file_record(path, root)
                    for path in sorted(results.iterdir())
                    if path.is_file()
                ],
            }
            COMMON.atomic_write_json(root / "summary.json", summary)
            COMMON.atomic_write_json(
                root / "COMPLETE",
                {
                    "schema": "gpmeep-ters-workload-complete-v2",
                    "summary": COMMON.file_record(root / "summary.json", root),
                },
            )
            lane = COMPARE.load_lane(root, COMPARE.TERS_SCHEMA)
            _, coordinates = COMPARE._ters_files(lane)
            self.assertTrue(np.array_equal(coordinates["x"], x))

            summary["measurement_mode"] = COMMON.TERS_PERFORMANCE_MEASUREMENT_MODE
            summary["performance_window_meep_time"] = 0.249999
            summary["physics_overrides"] = [COMMON.TERS_PERFORMANCE_ADAPTATION]
            COMMON.atomic_write_json(root / "summary.json", summary)
            COMMON.atomic_write_json(
                root / "COMPLETE",
                {
                    "schema": "gpmeep-ters-workload-complete-v2",
                    "summary": COMMON.file_record(root / "summary.json", root),
                },
            )
            lane = COMPARE.load_lane(root, COMPARE.TERS_SCHEMA)
            with self.assertRaisesRegex(
                COMMON.WorkloadError, "performance adaptation"
            ):
                COMPARE._ters_files(lane, allow_performance_window=True)

            summary["performance_window_meep_time"] = (
                COMMON.MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME
            )
            COMMON.atomic_write_json(root / "summary.json", summary)
            COMMON.atomic_write_json(
                root / "COMPLETE",
                {
                    "schema": "gpmeep-ters-workload-complete-v2",
                    "summary": COMMON.file_record(root / "summary.json", root),
                },
            )
            lane = COMPARE.load_lane(root, COMPARE.TERS_SCHEMA)
            COMPARE._ters_files(lane, allow_performance_window=True)
            csv_path = results / "withtip_dft.csv"
            csv_contents = csv_path.read_bytes()
            csv_path.write_bytes(csv_contents + b"tamper\n")
            with self.assertRaisesRegex(COMMON.WorkloadError, "mismatch"):
                COMPARE._ters_files(lane, allow_performance_window=True)
            csv_path.write_bytes(csv_contents)
            summary["measurement_mode"] = "exact"
            summary["performance_window_meep_time"] = None
            summary["physics_overrides"] = []

            changed = dict(values)
            changed["y"] = y + 1e-6
            changed_path = evidence_directory / "dft_coordinates_run1.npz"
            np.savez(changed_path, **changed)
            changed_evidence = COMPARE._coordinate_evidence(changed)
            summary["coordinates"][1] = {
                **COMMON.file_record(changed_path, root),
                "run_index": 1,
                "axes": changed_evidence,
            }
            summary["rank_records"][0]["records"][1][
                "dft_coordinates"
            ] = changed_evidence
            summary["original_outputs"] = [
                COMMON.file_record(path, root)
                for path in sorted(results.iterdir())
                if path.is_file()
            ]
            COMMON.atomic_write_json(root / "summary.json", summary)
            COMMON.atomic_write_json(
                root / "COMPLETE",
                {
                    "schema": "gpmeep-ters-workload-complete-v2",
                    "summary": COMMON.file_record(root / "summary.json", root),
                },
            )
            lane = COMPARE.load_lane(root, COMPARE.TERS_SCHEMA)
            with self.assertRaisesRegex(COMMON.WorkloadError, "different coordinates"):
                COMPARE._ters_files(lane)

    @staticmethod
    def aunp_fixture(lambda0=812.8962141934727):
        wavelength = np.arange(400.0, 1000.1, 2.0)
        frequency = 1000.0 / wavelength
        a0 = 0.0580880449989245
        slope = -0.0002627550254630512
        fwhm = 63.383195198829625
        peak = 0.6804447074336587
        absorption = a0 + slope * (wavelength - lambda0) + (peak - a0) / (
            1 + 4 * ((wavelength - lambda0) / fwhm) ** 2
        )
        reflectance = 1 - absorption
        incident = 0.5 + 0.1 * np.cos(wavelength / 100)
        bottom_fraction = np.full(301, 1e-5)
        reflected = reflectance * incident
        coefficient = np.sqrt(reflectance) * np.exp(1j * wavelength / 400)
        modes = np.column_stack((coefficient, np.ones(301, dtype=complex)))
        return {
            "wavelength_nm": wavelength,
            "frequency": frequency,
            "incident_flux": incident,
            "reflected_flux_monitor1": reflected,
            "reflected_flux_monitor2": reflected,
            "reflectance_monitor1": reflectance,
            "reflectance_monitor2": reflectance,
            "bottom_flux": -bottom_fraction * incident,
            "reference_bottom_flux": -incident,
            "bottom_residual_fraction": bottom_fraction,
            "absorptance_opaque": absorption,
            "absorptance_to_bottom": absorption - bottom_fraction,
            "mode_coefficients_monitor1": modes,
            "mode_coefficients_monitor2": modes,
            "reflection_coefficient_monitor1": coefficient,
            "reflection_coefficient_monitor2": coefficient,
        }

    def test_resonance_model_and_spectral_comparator(self):
        reference = self.aunp_fixture()
        fitted = COMPARE.fit_tm_resonance(reference)
        self.assertAlmostEqual(fitted["lambda0_nm"], 812.8962141934727, places=5)
        result = COMPARE.compare_spectra_pair(
            reference,
            {name: value.copy() for name, value in reference.items()},
            COMPARE.PAIR_TOLERANCES["same-fp32"],
        )
        self.assertEqual(result["outcome"], "PASS")
        changed = {name: value.copy() for name, value in reference.items()}
        changed["reflectance_monitor1"] += 1e-2
        changed["absorptance_opaque"] = 1 - changed["reflectance_monitor1"]
        changed["absorptance_to_bottom"] = (
            changed["absorptance_opaque"] - changed["bottom_residual_fraction"]
        )
        changed["reflected_flux_monitor1"] = (
            changed["reflectance_monitor1"] * changed["incident_flux"]
        )
        result = COMPARE.compare_spectra_pair(
            reference, changed, COMPARE.PAIR_TOLERANCES["same-fp32"]
        )
        self.assertEqual(result["outcome"], "FAIL")
        self.assertIn("reflectance_monitor1", result["failures"])

    def test_lane_loader_rejects_summary_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            original_outputs, source_sha256, results = original_ters_rank_outputs(root)
            summary = {
                "schema": COMPARE.TERS_SCHEMA,
                "single_precision": True,
                "expected_backend": "cpu",
                "mpi_size": 1,
                "rank_records": [
                    {
                        "rank": 0,
                        "exact_end_to_end_seconds": 2.0,
                        "original_result_root": results.relative_to(root).as_posix(),
                        "original_outputs": original_outputs,
                        "records": [self.run_record(0), self.run_record(1)],
                    }
                ],
                "phase_contract": phase_contract(2),
                "source": COMMON.absolute_file_record(root / "fixture.py", "fixture"),
                "source_sha256": source_sha256,
            }
            COMMON.atomic_write_json(root / "summary.json", summary)
            COMMON.atomic_write_json(
                root / "COMPLETE",
                {
                    "schema": "gpmeep-ters-workload-complete-v2",
                    "summary": COMMON.file_record(root / "summary.json", root),
                },
            )
            with self.assertRaisesRegex(COMMON.WorkloadError, "byte-exact"):
                COMPARE.load_lane(root, COMPARE.TERS_SCHEMA)
            self.enterContext(mock.patch.object(COMPARE, "TERS_SHA256", source_sha256))
            self.assertEqual(
                COMPARE.load_lane(root, COMPARE.TERS_SCHEMA)["summary"], summary
            )
            summary["rank_records"][0]["records"][1]["requested_backend"] = "cuda"
            COMMON.atomic_write_json(root / "summary.json", summary)
            COMMON.atomic_write_json(
                root / "COMPLETE",
                {
                    "schema": "gpmeep-ters-workload-complete-v2",
                    "summary": COMMON.file_record(root / "summary.json", root),
                },
            )
            with self.assertRaisesRegex(COMPARE.WorkloadError, "backend/request"):
                COMPARE.load_lane(root, COMPARE.TERS_SCHEMA)
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                COMPARE.WorkloadError, "size mismatch|digest mismatch"
            ):
                COMPARE.load_lane(root, COMPARE.TERS_SCHEMA)

    def test_precision_class_is_fail_closed(self):
        fp64 = {"summary": {"single_precision": False, "expected_backend": "cpu"}}
        fp32 = {"summary": {"single_precision": True, "expected_backend": "cuda"}}
        cpu_fp32 = {"summary": {"single_precision": True, "expected_backend": "cpu"}}
        COMPARE._validate_precision_pair(fp64, fp32, "cuda-fp64-fp32")
        COMPARE._validate_precision_pair(cpu_fp32, fp32, "cuda-same-fp32")
        COMPARE._validate_precision_pair(fp64, cpu_fp32, "cpu-fp64-fp32")
        COMPARE._validate_backend_roles(
            fp64, fp32, "candidate", "cuda-fp64-fp32"
        )
        COMPARE._validate_backend_roles(
            fp64, cpu_fp32, "cpu-candidate", "cpu-fp64-fp32"
        )
        with self.assertRaisesRegex(COMPARE.WorkloadError, "requires FP64"):
            COMPARE._validate_precision_pair(fp32, fp32, "cuda-fp64-fp32")
        with self.assertRaisesRegex(COMPARE.WorkloadError, "must be a cuda"):
            COMPARE._validate_backend_roles(
                fp64, cpu_fp32, "candidate", "cuda-fp64-fp32"
            )

    def test_chunked_npz_and_hdf5_scientific_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            reference_npz = root / "reference.npz"
            candidate_npz = root / "candidate.npz"
            vector = np.linspace(0.1, 1.0, 257).astype(np.float32)
            np.savez(
                reference_npz,
                wavelength_nm=np.arange(257),
                complex_field=vector.astype(np.complex64) * (1 + 0.2j),
            )
            np.savez(
                candidate_npz,
                wavelength_nm=np.arange(257),
                complex_field=vector.astype(np.complex64) * (1 + 0.20001j),
            )
            npz_result = COMPARE.compare_npz_file(
                reference_npz,
                candidate_npz,
                COMPARE.PAIR_TOLERANCES["same-fp32"],
            )
            self.assertEqual(npz_result["outcome"], "PASS")

            reference_h5 = root / "reference.h5"
            candidate_h5 = root / "candidate.h5"
            with h5py.File(reference_h5, "w") as handle:
                handle.create_dataset("field", data=vector.reshape(257, 1), chunks=(17, 1))
                handle.create_dataset("ex.r", data=vector)
                handle.create_dataset("ex.i", data=np.zeros_like(vector))
            with h5py.File(candidate_h5, "w") as handle:
                handle.create_dataset(
                    "field", data=(vector * 1.0001).reshape(257, 1), chunks=(31, 1)
                )
                handle.create_dataset("ex.r", data=vector)
                # A zero reference imaginary part must be judged as part of
                # the complex field, not assigned an infinite standalone NRMSE.
                handle.create_dataset("ex.i", data=vector * 1e-5)
            hdf5_result = COMPARE.compare_hdf5_file(
                reference_h5,
                candidate_h5,
                COMPARE.PAIR_TOLERANCES["same-fp32"],
            )
            self.assertEqual(hdf5_result["outcome"], "PASS")
            with h5py.File(candidate_h5, "r+") as handle:
                handle["field"][:] *= 1.1
            hdf5_result = COMPARE.compare_hdf5_file(
                reference_h5,
                candidate_h5,
                COMPARE.PAIR_TOLERANCES["same-fp32"],
            )
            self.assertEqual(hdf5_result["outcome"], "FAIL")

    def test_scientific_dtype_width_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            reference_npz = root / "reference.npz"
            candidate_npz = root / "candidate.npz"
            tolerance = COMPARE.PAIR_TOLERANCES["same-fp32"]
            mutations = (
                (
                    np.arange(8, dtype=np.float32),
                    np.arange(8, dtype=np.float64),
                ),
                (
                    np.arange(8, dtype=np.complex64),
                    np.arange(8, dtype=np.complex128),
                ),
                (
                    np.arange(8, dtype=np.int32),
                    np.arange(8, dtype=np.int64),
                ),
                (
                    np.arange(8, dtype=np.uint32),
                    np.arange(8, dtype=np.int32),
                ),
            )
            for reference, candidate in mutations:
                np.savez(reference_npz, field=reference)
                np.savez(candidate_npz, field=candidate)
                with self.assertRaisesRegex(
                    COMPARE.WorkloadError, "dtype|shape/type"
                ):
                    COMPARE.compare_npz_file(
                        reference_npz, candidate_npz, tolerance, "cuda-same-fp32"
                    )

            reference_h5 = root / "reference.h5"
            candidate_h5 = root / "candidate.h5"
            with h5py.File(reference_h5, "w") as handle:
                handle.create_dataset("field", data=np.arange(8, dtype=np.float32))
                handle.create_dataset("ex.r", data=np.arange(8, dtype=np.float32))
                handle.create_dataset("ex.i", data=np.arange(8, dtype=np.float32))
            with h5py.File(candidate_h5, "w") as handle:
                handle.create_dataset("field", data=np.arange(8, dtype=np.float64))
                handle.create_dataset("ex.r", data=np.arange(8, dtype=np.float32))
                handle.create_dataset("ex.i", data=np.arange(8, dtype=np.float64))
            with self.assertRaisesRegex(COMPARE.WorkloadError, "dtype"):
                COMPARE.compare_hdf5_file(
                    reference_h5, candidate_h5, tolerance, "cuda-same-fp32"
                )

    def test_aunp_dtype_policy_rejects_self_consistent_width_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            reference_root = root / "reference"
            candidate_root = root / "candidate"
            for lane_root in (reference_root, candidate_root):
                (lane_root / "TM_Ex").mkdir(parents=True)

            reference_npz = reference_root / "TM_Ex/reference_data.npz"
            candidate_npz = candidate_root / "TM_Ex/reference_data.npz"
            np.savez(reference_npz, frequency=np.arange(8, dtype=np.float32))
            np.savez(candidate_npz, frequency=np.arange(8, dtype=np.float32))
            with self.assertRaisesRegex(
                COMPARE.WorkloadError, "fixed policy"
            ):
                COMPARE.compare_npz_file(
                    {
                        "path": reference_npz,
                        "root": reference_root,
                        "record": COMMON.file_record(reference_npz, reference_root),
                    },
                    {
                        "path": candidate_npz,
                        "root": candidate_root,
                        "record": COMMON.file_record(candidate_npz, candidate_root),
                    },
                    COMPARE.PAIR_TOLERANCES["same-fp32"],
                    "cuda-same-fp32",
                )

            reference_h5 = reference_root / "TM_Ex/raw_complex_fields.h5"
            candidate_h5 = candidate_root / "TM_Ex/raw_complex_fields.h5"
            for path in (reference_h5, candidate_h5):
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("ex.r", data=np.arange(8, dtype=np.float64))
                    handle.create_dataset("ex.i", data=np.arange(8, dtype=np.float64))
            with self.assertRaisesRegex(
                COMPARE.WorkloadError, "fixed policy"
            ):
                COMPARE.compare_hdf5_file(
                    {
                        "path": reference_h5,
                        "root": reference_root,
                        "record": COMMON.file_record(reference_h5, reference_root),
                    },
                    {
                        "path": candidate_h5,
                        "root": candidate_root,
                        "record": COMMON.file_record(candidate_h5, candidate_root),
                    },
                    COMPARE.PAIR_TOLERANCES["same-fp32"],
                    "cuda-same-fp32",
                )

            with h5py.File(reference_h5, "w") as handle:
                handle.create_dataset("ex.r", data=np.arange(8, dtype=np.float64))
                handle.create_dataset("ex.i", data=np.arange(8, dtype=np.float64))
            with h5py.File(candidate_h5, "w") as handle:
                handle.create_dataset("ex.r", data=np.arange(8, dtype=np.float32))
                handle.create_dataset("ex.i", data=np.arange(8, dtype=np.float32))
            result = COMPARE.compare_hdf5_file(
                {
                    "path": reference_h5,
                    "root": reference_root,
                    "record": COMMON.file_record(reference_h5, reference_root),
                },
                {
                    "path": candidate_h5,
                    "root": candidate_root,
                    "record": COMMON.file_record(candidate_h5, candidate_root),
                },
                COMPARE.PAIR_TOLERANCES["fp64-fp32"],
                "cuda-fp64-fp32",
            )
            self.assertEqual(result["outcome"], "PASS")
            self.assertEqual(result["datasets"]["ex"]["dtypes"]["real"], {
                "reference": np.dtype(np.float64).str,
                "candidate": np.dtype(np.float32).str,
            })

    def test_hdf5_selection_respects_value_cap_for_wide_trailing_axis(self):
        shape = (2, COMPARE.CHUNK_VALUES + 17, 3)
        selections = list(COMPARE._hdf5_selections(shape))
        self.assertGreater(len(selections), 1)
        for selection in selections:
            count = np.prod([item.stop - item.start for item in selection])
            self.assertLessEqual(count, COMPARE.CHUNK_VALUES)

    @staticmethod
    def _write_aunp_lane(root):
        simulation = root / "simulation"
        simulation.mkdir(parents=True)
        np.savez(simulation / "values.npz", field=np.arange(8, dtype=np.float32))
        with h5py.File(simulation / "fields.h5", "w") as handle:
            handle.create_dataset("ex.r", data=np.arange(12, dtype=np.float32).reshape(3, 4))
        effective = dict(COMMON.AUNP_EXPECTED_DEFAULT_CONFIG)
        effective["output_dir"] = str(root / "outputs")
        effective["required_mpi_ranks"] = 1
        effective.update(COMMON.AUNP_PHYSICAL_OVERRIDES)
        engine_digest = "e" * 64
        COMMON.atomic_write_json(
            simulation / "simulation_config.json",
            {
                "schema_version": 2,
                "config": effective,
                "mpi_processes": 1,
                "meep_version": "fixture",
                "script_sha256": engine_digest,
                "material_fingerprint": {"fixture": True},
                "estimated_raw_dft_storage": {"bytes": 1},
            },
        )
        files = []
        for path in sorted(simulation.iterdir()):
            record = COMMON.file_record(path, simulation)
            if path.suffix == ".h5":
                with h5py.File(path, "r") as handle:
                    record["hdf5_datasets"] = {
                        name: {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
                        for name, dataset in handle.items()
                    }
            files.append(record)
        COMMON.atomic_write_json(
            simulation / "output_manifest.json",
            {
                "file_count_excluding_manifest": len(files),
                "total_bytes_excluding_manifest": sum(item["size_bytes"] for item in files),
                "files": files,
            },
        )
        summary = {
            "schema": COMPARE.AUNP_SCHEMA,
            "single_precision": True,
            "expected_backend": "cpu",
            "mpi_size": 1,
            "rank_records": [
                {
                    "rank": 0,
                    "process_wall_seconds": 4.0,
                    "records": [
                        UserWorkloadComparisonTests.run_record(index)
                        for index in range(2)
                    ],
                }
            ],
            "phase_contract": phase_contract(2),
            "simulation_root": "simulation",
            "output_manifest": COMMON.file_record(
                simulation / "output_manifest.json", root
            ),
            "simulation_config": COMMON.file_record(
                simulation / "simulation_config.json", root
            ),
            "engine": {"sha256": engine_digest},
            "default_config": COMMON.AUNP_EXPECTED_DEFAULT_CONFIG,
            "effective_config": effective,
            "qualification_profile": COMMON.aunp_qualification_profile(),
            "physical_overrides": dict(COMMON.AUNP_PHYSICAL_OVERRIDES),
            "execution_policy_overrides": {
                "required_mpi_ranks": 1,
                "output_dir": effective["output_dir"],
                "memory_wait_policy": "single-snapshot-fail-closed",
            },
        }
        COMMON.atomic_write_json(root / "summary.json", summary)
        COMMON.atomic_write_json(
            root / "COMPLETE",
            {
                "schema": "gpmeep-aunp-r4000-workload-complete-v4",
                "summary": COMMON.file_record(root / "summary.json", root),
            },
        )

    def test_aunp_output_manifest_replays_full_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self._write_aunp_lane(root)
            lane = COMPARE.load_lane(root, COMPARE.AUNP_SCHEMA)
            manifest = COMPARE.load_aunp_output_manifest(lane)
            self.assertEqual(
                set(manifest),
                {"fields.h5", "simulation_config.json", "values.npz"},
            )
            normalized = COMPARE.validate_aunp_simulation_config(lane, manifest)
            self.assertEqual(normalized["config"]["resolution"], 4000)
            (root / "simulation" / "unrecorded.txt").write_text(
                "not sealed", encoding="utf-8"
            )
            with self.assertRaisesRegex(COMPARE.WorkloadError, "inventory differs"):
                COMPARE.load_aunp_output_manifest(lane)

    def test_aunp_config_rejects_physical_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self._write_aunp_lane(root)
            lane = COMPARE.load_lane(root, COMPARE.AUNP_SCHEMA)
            manifest = COMPARE.load_aunp_output_manifest(lane)
            lane["summary"]["effective_config"]["gap_thickness"] = 0.003
            with self.assertRaisesRegex(COMMON.WorkloadError, "forbidden keys"):
                COMPARE.validate_aunp_simulation_config(lane, manifest)


if __name__ == "__main__":
    unittest.main()
