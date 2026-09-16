#!/usr/bin/env python3
"""Focused tests for the fixed-work AuNP performance adapter."""

from __future__ import annotations

import copy
import dataclasses
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


def load_module(name: str):
    path = WORKLOAD_DIR / name
    module_name = f"gpmeep_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMMON = load_module("common.py")
sys.modules["common"] = COMMON
PERFORMANCE = load_module("run_aunp_performance_workload.py")


def zero_statistics():
    values = {}
    for group, stem in COMMON.FDTD_PHASE_COUNTERS.values():
        values[group] = {
            f"{backend}_{stem}_{field}": 0
            for backend in ("cpu", "cuda")
            for field in ("calls", "points")
        }
    values["multi_gpu"] = {name: 0 for name in COMMON.MULTI_GPU_COUNTERS}
    values["mpi_completion"] = {name: 0 for name in COMMON.MPI_COMPLETION_COUNTERS}
    return values


class AuNPPerformanceWorkloadTests(unittest.TestCase):
    def write(self, root: pathlib.Path, relative: str, payload: bytes):
        path = root.joinpath(*pathlib.PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_contract_has_two_tm_full_monitor_fixed_work_phases(self) -> None:
        self.assertEqual(PERFORMANCE.FIXED_WINDOW_MEEP_TIME, 0.25)
        self.assertEqual(PERFORMANCE.MINIMUM_TIMESTEPS_PER_PHASE, 2000)
        self.assertEqual(len(PERFORMANCE.PHASE_SPECS), 2)
        self.assertEqual(
            [item[3] for item in PERFORMANCE.PHASE_SPECS],
            [
                "analysis_reference",
                "analysis_structure",
            ],
        )
        self.assertEqual(
            [item[1] for item in PERFORMANCE.PHASE_SPECS],
            ["physical_ex", "physical_ex"],
        )

    def test_cli_does_not_allow_a_physics_or_window_override(self) -> None:
        with self.assertRaises(SystemExit):
            PERFORMANCE.parse_args(
                [
                    "--archive",
                    "archive",
                    "--output",
                    "output",
                    "--build-receipt",
                    "receipt",
                    "--expected-backend",
                    "cpu",
                    "--fixed-window",
                    "0.01",
                ]
            )

    def test_package_provenance_is_mutually_exclusive_and_identity_bound(self) -> None:
        parsed = PERFORMANCE.parse_args(
            [
                "--archive",
                "archive",
                "--output",
                "output",
                "--package-provenance",
                "provenance.json",
                "--expected-source-commit",
                "1" * 40,
                "--expected-package-sha256",
                "2" * 64,
                "--expected-backend",
                "cuda",
            ]
        )
        self.assertIsNone(parsed.build_receipt)
        self.assertEqual(parsed.package_provenance, pathlib.Path("provenance.json"))
        with self.assertRaises(SystemExit):
            PERFORMANCE.parse_args(
                [
                    "--archive",
                    "archive",
                    "--output",
                    "output",
                    "--build-receipt",
                    "receipt.json",
                    "--package-provenance",
                    "provenance.json",
                    "--expected-backend",
                    "cuda",
                ]
            )

    def test_installed_package_attestation_replays_files_and_report_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            prefix = root / "prefix"
            conda_record_path = self.write(
                prefix,
                "conda-meta/gpmeep-1.0.3-fixture.json",
                b"placeholder",
            )
            manifest_path = self.write(
                prefix,
                "share/gpmeep/release.json",
                b"placeholder",
            )
            runtime_paths = {
                name: self.write(prefix, relative, payload)
                for name, relative, payload in (
                    ("python", "bin/python3.11", b"python\n"),
                    (
                        "module",
                        "lib/python3.11/site-packages/meep/__init__.py",
                        b"module\n",
                    ),
                    (
                        "extension",
                        "lib/python3.11/site-packages/meep/_meep.so",
                        b"extension\n",
                    ),
                    ("libmeep", "lib/libmeep.so", b"libmeep\n"),
                )
            }
            source_commit = "1" * 40
            package_sha256 = "2" * 64
            conda_record_path.write_text(
                json.dumps(
                    {
                        "name": "gpmeep",
                        "version": COMMON.GPMEEP_DISTRIBUTION_VERSION,
                        "sha256": package_sha256,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            manifest = copy.deepcopy(COMMON.GPMEEP_RELEASE_MANIFEST_FIELDS)
            manifest["source_commit"] = source_commit
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            report = {
                "schema_version": 1,
                "state": "PASS",
                "prefix": str(prefix),
                "conda_record": COMMON.absolute_file_record(
                    conda_record_path, "fixture Conda record"
                ),
                "package_url": "file:///fixture.conda",
                "package_sha256": package_sha256,
                "ownership": {
                    "owned_paths": 4,
                    "regular_files": 4,
                    "symlinks": 0,
                    "content_hashes_verified": 4,
                    "content_sizes_verified": 4,
                    "prefix_rewritten_sizes": 0,
                },
                "manifest": {
                    **manifest,
                    "file": COMMON.absolute_file_record(
                        manifest_path, "fixture manifest"
                    ),
                },
                "runtime": {
                    **{
                        name: COMMON.absolute_file_record(path, f"fixture {name}")
                        for name, path in runtime_paths.items()
                    },
                    "meep_version": "1.35.0-beta",
                    "cuda_compiled": True,
                    "mpi_enabled": True,
                    "single_precision": True,
                },
            }
            report["report_id"] = COMMON._canonical_sha256(report)
            attestation = root / "provenance.json"
            attestation.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            replayed = COMMON.validate_package_provenance_attestation(
                attestation, source_commit, package_sha256
            )
            self.assertEqual(replayed["report_id"], report["report_id"])
            runtime_paths["extension"].write_bytes(b"mutated\n")
            with self.assertRaisesRegex(COMMON.WorkloadError, "extension .*mismatch"):
                COMMON.validate_package_provenance_attestation(
                    attestation, source_commit, package_sha256
                )
            runtime_paths["extension"].write_bytes(b"extension\n")
            manifest_path.write_text(
                json.dumps(manifest).replace(
                    '"qualification_eligible": true',
                    '"qualification_eligible": 1, "qualification_eligible": true',
                ),
                encoding="utf-8",
            )
            report["manifest"]["file"] = COMMON.absolute_file_record(
                manifest_path, "duplicate-key fixture manifest"
            )
            report.pop("report_id")
            report["report_id"] = COMMON._canonical_sha256(report)
            attestation.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(COMMON.WorkloadError, "duplicate JSON"):
                COMMON.validate_package_provenance_attestation(
                    attestation, source_commit, package_sha256
                )

    def test_main_runs_two_tm_phases_and_terminal_replay_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            output = root / "lane"
            archive = self.write(root, "archive.tar.gz", b"archive\n")
            receipt = self.write(root, "receipt.json", b"receipt\n")
            fake_meep_path = self.write(root, "fake_meep.py", b"fixture\n")

            @dataclasses.dataclass
            class FakeConfig:
                run_mode: str = "simulate"
                output_dir: object = "outputs"
                required_mpi_ranks: int = 64
                reference_memory_gib: float = 16.0
                structure_memory_gib: float = 96.0
                dft_minimum_run_time: float = 150.0
                dft_decay_tolerance: float = 1e-5

                def validate(self):
                    return None

            class FakeGpu:
                active_backend = "cpu"
                requested_backend = "cpu"
                selected_device = -1
                selected_device_identifier = ""

                def __init__(self):
                    self.values = zero_statistics()

                def reset_statistics(self):
                    self.values = zero_statistics()

                def statistics(self):
                    return copy.deepcopy(self.values)

            fake_gpu = FakeGpu()

            class FakeSimulation:
                def __init__(self, structure):
                    self.structure = structure
                    self.fields = types.SimpleNamespace(t=0)
                    self.loaded_flux_count = 0
                    self.reset = False

                def run(self, *, until):
                    self_outer.assertEqual(until, PERFORMANCE.FIXED_WINDOW_MEEP_TIME)
                    phases = {"curl", "update_eh", "source", "boundary", "dft"}
                    if self.structure:
                        phases.add("polarization")
                    for phase in phases:
                        group, stem = COMMON.FDTD_PHASE_COUNTERS[phase]
                        fake_gpu.values[group][f"cpu_{stem}_calls"] += 1
                        fake_gpu.values[group][f"cpu_{stem}_points"] += 8
                    self.fields.t += PERFORMANCE.MINIMUM_TIMESTEPS_PER_PHASE

                def timestep(self):
                    return self.fields.t

                def meep_time(self):
                    return PERFORMANCE.FIXED_WINDOW_MEEP_TIME

                def get_flux_data(self, monitor):
                    return {"monitor": monitor.name, "fixture": True}

                def load_minus_flux_data(self, _monitor, _data):
                    self.loaded_flux_count += 1

                def reset_meep(self):
                    self.reset = True

            class FakeGpuModule(types.ModuleType):
                pass

            fake_mp = FakeGpuModule("meep")
            fake_mp.__file__ = str(fake_meep_path)
            fake_mp.__version__ = "fixture"
            fake_mp.gpu = fake_gpu
            fake_mp.is_single_precision = lambda: True

            class FakeComm:
                def Get_rank(self):
                    return 0

                def Get_size(self):
                    return 1

                def bcast(self, value, root=0):
                    return value

                def gather(self, value, root=0):
                    return [value]

                def Barrier(self):
                    return None

            mpi4py_module = types.ModuleType("mpi4py")
            mpi4py_module.MPI = types.SimpleNamespace(COMM_WORLD=FakeComm())

            class FakeEngine:
                AppConfig = FakeConfig

                def __init__(self):
                    self.calls = []
                    self.simulations = []

                def serializable_config(self, cfg):
                    value = dict(COMMON.AUNP_EXPECTED_DEFAULT_CONFIG)
                    value["output_dir"] = str(cfg.output_dir)
                    value["required_mpi_ranks"] = cfg.required_mpi_ranks
                    value["dft_minimum_run_time"] = cfg.dft_minimum_run_time
                    value["dft_decay_tolerance"] = cfg.dft_decay_tolerance
                    return value

                def validate_runtime_resources(self, _cfg):
                    return None

                def memory_admission_snapshot(self):
                    return {"effective_available_gib": 512.0}

                def build_simulation(
                    self,
                    _cfg,
                    polarization,
                    *,
                    include_structure,
                    field_monitor_kind,
                ):
                    self.calls.append(
                        (polarization, include_structure, field_monitor_kind)
                    )
                    simulation = FakeSimulation(include_structure)
                    self.simulations.append(simulation)
                    monitors = types.SimpleNamespace(
                        monitor1=types.SimpleNamespace(name="monitor1"),
                        monitor2=types.SimpleNamespace(name="monitor2"),
                    )
                    return simulation, monitors

            engine = FakeEngine()
            self_outer = self
            archive_digest = hashlib.sha256(b"archive\n").hexdigest()
            engine_digest = hashlib.sha256(b"engine\n").hexdigest()
            engine_member = "aunp_r4000_repro/fdtd/meep/aunp_periodic_fdtd.py"

            def extract(_archive, destination):
                engine_path = self.write(
                    destination,
                    "aunp_r4000_repro/fdtd/meep/aunp_periodic_fdtd.py",
                    b"engine\n",
                )
                self.assertTrue(engine_path.is_file())
                return destination / "aunp_r4000_repro"

            argv = [
                "--archive",
                str(archive),
                "--output",
                str(output),
                "--build-receipt",
                str(receipt),
                "--expected-backend",
                "cpu",
            ]
            with mock.patch.dict(
                sys.modules, {"meep": fake_mp, "mpi4py": mpi4py_module}
            ), mock.patch.object(
                PERFORMANCE, "AUNP_ARCHIVE_SHA256", archive_digest
            ), mock.patch.object(
                PERFORMANCE, "AUNP_MEMBER_SHA256", {engine_member: engine_digest}
            ), mock.patch.object(
                PERFORMANCE,
                "verify_file",
                side_effect=lambda path, *_args: pathlib.Path(path),
            ), mock.patch.object(
                PERFORMANCE,
                "verify_build_receipt_collective",
                return_value=({"receipt_id": "fixture-receipt"}, {}),
            ), mock.patch.object(
                PERFORMANCE, "extract_verified_aunp", side_effect=extract
            ), mock.patch.object(
                PERFORMANCE, "_load_engine", return_value=engine
            ), mock.patch.object(
                PERFORMANCE,
                "physical_core_affinity",
                return_value=[{"fixture": True}],
            ):
                self.assertEqual(PERFORMANCE.main(argv), 0)

            self.assertEqual(
                engine.calls,
                [(item[1], item[2], item[3]) for item in PERFORMANCE.PHASE_SPECS],
            )
            self.assertTrue(all(simulation.reset for simulation in engine.simulations))
            self.assertEqual(
                [simulation.loaded_flux_count for simulation in engine.simulations],
                [0, 2],
            )
            with mock.patch.object(
                PERFORMANCE, "AUNP_ARCHIVE_SHA256", archive_digest
            ), mock.patch.object(
                PERFORMANCE, "AUNP_MEMBER_SHA256", {engine_member: engine_digest}
            ):
                summary = PERFORMANCE.validate_performance_output(
                    output, expected_backend="cpu", expected_mpi_size=1
                )
                self.assertEqual(summary["scientific_output_claims"], [])
                self.assertEqual(
                    [
                        record["timestep_delta"]
                        for record in summary["rank_records"][0]["records"]
                    ],
                    [PERFORMANCE.MINIMUM_TIMESTEPS_PER_PHASE]
                    * len(PERFORMANCE.PHASE_SPECS),
                )

                original_summary = copy.deepcopy(summary)

                def publish(value):
                    COMMON.atomic_write_json(output / "summary.json", value)
                    COMMON.atomic_write_json(
                        output / "COMPLETE",
                        {
                            "schema": PERFORMANCE.COMPLETE_SCHEMA,
                            "summary": COMMON.file_record(
                                output / "summary.json", output
                            ),
                        },
                    )

                changed = copy.deepcopy(original_summary)
                changed["fixed_window_meep_time"] = 0.01
                publish(changed)
                with self.assertRaisesRegex(
                    COMMON.WorkloadError, "adaptation contract"
                ):
                    PERFORMANCE.validate_performance_output(output)

                changed = copy.deepcopy(original_summary)
                changed["effective_config"]["gap_thickness"] = 0.003
                publish(changed)
                with self.assertRaisesRegex(
                    COMMON.WorkloadError, "physical configuration"
                ):
                    PERFORMANCE.validate_performance_output(output)

                changed = copy.deepcopy(original_summary)
                changed["effective_config"]["dft_minimum_run_time"] = 19.0
                publish(changed)
                with self.assertRaisesRegex(COMMON.WorkloadError, "stop-condition"):
                    PERFORMANCE.validate_performance_output(output)

                changed = copy.deepcopy(original_summary)
                changed["fdtd_wall_seconds"] += 1.0
                publish(changed)
                with self.assertRaisesRegex(
                    COMMON.WorkloadError, "timing was not re-derived"
                ):
                    PERFORMANCE.validate_performance_output(output)

                publish(original_summary)
                self.write(output, "spectra.npz", b"forbidden\n")
                with self.assertRaisesRegex(
                    COMMON.WorkloadError, "output inventory differs"
                ):
                    PERFORMANCE.validate_performance_output(output)


if __name__ == "__main__":
    unittest.main()
