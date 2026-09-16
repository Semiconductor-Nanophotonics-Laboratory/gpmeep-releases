#!/usr/bin/env python3
"""Mutation tests for exact AuNP stage recovery state."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import hashlib
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
RESUME = load_module("aunp_exact_resume.py")
sys.modules["aunp_exact_resume"] = RESUME
ADAPTER = load_module("run_aunp_workload.py")
COMPARE = load_module("compare_user_workloads.py")


class AuNPExactResumeTests(unittest.TestCase):
    def write(self, root: pathlib.Path, relative: str, payload: bytes = b"fixture\n"):
        path = root.joinpath(*pathlib.PurePosixPath(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def populate_stage(self, root: pathlib.Path, stage: str) -> None:
        for relative in RESUME.stage_relative_paths(stage):
            self.write(root, relative, relative.encode() + b"\n")

    def initialize(self, state: pathlib.Path, contract: dict):
        digest = RESUME.sha256_bytes(RESUME.canonical_json_bytes(contract))
        RESUME.append_event(
            state / "events", digest, "initialized", {"contract": contract}
        )
        return digest

    def test_stage_file_contract_is_exact(self) -> None:
        self.assertEqual(len(RESUME.REFERENCE_REQUIRED_BASENAMES), 10)
        self.assertEqual(len(RESUME.STRUCTURE_REQUIRED_BASENAMES), 13)
        self.assertEqual(len(RESUME.POSTPROCESS_REQUIRED_ROOT_BASENAMES), 2)
        self.assertEqual(
            RESUME.STAGE_ORDER,
            ("TM_Ex-reference", "TM_Ex-structure", "postprocessing"),
        )

    def test_manifest_replay_rejects_changed_prior_byte(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            path = self.write(root, "TM_Ex/reference_complete.json", b"sealed\n")
            manifest = RESUME.make_manifest(root)
            path.write_bytes(b"mutate\n")
            with self.assertRaisesRegex(COMMON.WorkloadError, "digest mismatch"):
                RESUME.replay_manifest(root, manifest, "prior")

    def test_manifest_replay_rejects_extra_file_when_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            self.write(root, "one.dat")
            manifest = RESUME.make_manifest(root)
            self.write(root, "extra.dat")
            with self.assertRaisesRegex(COMMON.WorkloadError, "inventory differs"):
                RESUME.replay_manifest(root, manifest, "final", exact_inventory=True)

    def test_inventory_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            target = self.write(root, "target.dat")
            (root / "linked.dat").symlink_to(target)
            with self.assertRaisesRegex(COMMON.WorkloadError, "non-regular file"):
                RESUME.inventory_files(root)

    def test_engine_marker_alone_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            simulation = output / "outputs" / "simulation" / "baseline_r4000"
            base = self.write(simulation, "simulation_config.json", b"base\n")
            marker = self.write(
                simulation, "TM_Ex/reference_complete.json", b"unsealed\n"
            )
            result = RESUME.quarantine_unsealed_files(
                simulation,
                {"simulation_config.json"},
                output / "attempt-quarantine",
                1,
                "unsealed reference stage",
            )
            self.assertIsNotNone(result)
            self.assertTrue(base.is_file())
            self.assertFalse(marker.exists())
            moved = output / result["root"] / "TM_Ex" / "reference_complete.json"
            self.assertEqual(moved.read_bytes(), b"unsealed\n")

    def test_structure_retry_preserves_exact_reference_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            simulation = output / "outputs" / "simulation" / "baseline_r4000"
            self.populate_stage(simulation, "TM_Ex-reference")
            reference = RESUME.make_manifest(
                simulation, RESUME.stage_relative_paths("TM_Ex-reference")
            )
            self.write(simulation, "TM_Ex/structure_complete.json", b"marker\n")
            self.write(simulation, "TM_Ex/raw_complex_fields.h5", b"partial\n")
            allowed = RESUME.replay_manifest(simulation, reference, "reference")
            result = RESUME.quarantine_unsealed_files(
                simulation,
                allowed,
                output / "attempt-quarantine",
                2,
                "unsealed structure stage",
            )
            RESUME.replay_manifest(simulation, reference, "reference")
            self.assertEqual(
                set(RESUME.inventory_files(simulation)),
                set(RESUME.stage_relative_paths("TM_Ex-reference")),
            )
            moved = set(RESUME.manifest_paths(result["manifest"], "quarantine"))
            self.assertEqual(
                moved,
                {"TM_Ex/raw_complex_fields.h5", "TM_Ex/structure_complete.json"},
            )

    def test_fdtd_seal_rejects_unowned_extra(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            self.populate_stage(root, "TM_Ex-reference")
            self.write(root, "TM_Ex/unowned.dat")
            with self.assertRaisesRegex(COMMON.WorkloadError, "file ownership differs"):
                RESUME.seal_fdtd_stage(root, "TM_Ex-reference", [])

    def test_reference_and_structure_seal_share_directory_safely(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            self.populate_stage(root, "TM_Ex-reference")
            reference = RESUME.seal_fdtd_stage(root, "TM_Ex-reference", [])
            self.populate_stage(root, "TM_Ex-structure")
            structure = RESUME.seal_fdtd_stage(root, "TM_Ex-structure", [reference])
            self.assertEqual(len(reference), 10)
            self.assertEqual(len(structure), 13)
            RESUME.replay_manifest(root, reference, "reference")
            RESUME.replay_manifest(root, structure, "structure")

    def test_each_exact_boundary_preserves_prior_stages_and_quarantines_next(self) -> None:
        for completed_count in range(len(RESUME.STAGE_ORDER)):
            with self.subTest(completed_count=completed_count), tempfile.TemporaryDirectory() as raw:
                output = pathlib.Path(raw)
                simulation = output / "simulation"
                state = output / "state"
                self.write(simulation, "simulation_config.json", b"base\n")
                self.write(simulation, "run_environment.json", b"environment\n")
                base_manifest = RESUME.make_manifest(
                    simulation, RESUME.BASE_REQUIRED_PATHS
                )
                contract = {"boundary_fixture": completed_count}
                digest = self.initialize(state, contract)
                RESUME.append_event(
                    state / "events",
                    digest,
                    "attempt-started",
                    {"attempt_index": 0, "started_utc": "2000-01-01T00:00:00Z"},
                )
                RESUME.append_event(
                    state / "events",
                    digest,
                    "base-sealed",
                    {"manifest": base_manifest},
                )
                completed_manifests = []
                for stage_index, stage in enumerate(
                    RESUME.STAGE_ORDER[:completed_count]
                ):
                    self.assertNotEqual(stage, "postprocessing")
                    self.populate_stage(simulation, stage)
                    manifest = RESUME.seal_fdtd_stage(
                        simulation, stage, completed_manifests
                    )
                    completed_manifests.append(manifest)
                    RESUME.append_event(
                        state / "events",
                        digest,
                        "stage-sealed",
                        {
                            "stage": stage,
                            "attempt_index": 0,
                            "manifest": manifest,
                            "rank_records": [{"rank": 0, "run_index": stage_index}],
                            "wall_seconds": 1.0,
                        },
                    )
                checkpoint = RESUME.publish_checkpoint(
                    state / "CHECKPOINT.json", contract, state / "events"
                )
                next_stage = RESUME.STAGE_ORDER[completed_count]
                partial = RESUME.stage_relative_paths(next_stage)[0]
                self.write(simulation, partial, b"unsealed next stage\n")
                allowed = RESUME.replay_completed_output(simulation, checkpoint)
                quarantine = RESUME.quarantine_unsealed_files(
                    simulation,
                    allowed,
                    output / "attempt-quarantine",
                    0,
                    f"partial {next_stage}",
                )
                self.assertIsNotNone(quarantine)
                self.assertIn(
                    partial,
                    RESUME.manifest_paths(quarantine["manifest"], "quarantine"),
                )
                self.assertNotIn(partial, RESUME.inventory_files(simulation))
                RESUME.replay_completed_output(simulation, checkpoint)

    def test_hash_chained_journal_rejects_gap(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = pathlib.Path(raw)
            contract = {"archive_sha256": "a" * 64}
            digest = self.initialize(state, contract)
            RESUME.append_event(
                state / "events",
                digest,
                "attempt-started",
                {"attempt_index": 0, "started_utc": "2000-01-01T00:00:00Z"},
            )
            (state / "events" / "00000000.json").rename(
                state / "events" / "00000002.json"
            )
            with self.assertRaisesRegex(COMMON.WorkloadError, "not contiguous"):
                RESUME.load_events(state / "events")

    def test_checkpoint_heals_only_durable_trailing_event(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = pathlib.Path(raw)
            contract = {"archive_sha256": "b" * 64}
            digest = self.initialize(state, contract)
            checkpoint = RESUME.publish_checkpoint(
                state / "CHECKPOINT.json", contract, state / "events"
            )
            self.assertEqual(checkpoint["journal_event_count"], 1)
            RESUME.append_event(
                state / "events",
                digest,
                "attempt-started",
                {"attempt_index": 0, "started_utc": "2000-01-01T00:00:00Z"},
            )
            healed = RESUME.replay_checkpoint(
                state / "CHECKPOINT.json", contract, state / "events"
            )
            self.assertEqual(healed["journal_event_count"], 2)
            self.assertEqual(len(healed["attempts"]), 1)

    def test_checkpoint_rejects_nonprefix_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state = pathlib.Path(raw)
            contract = {"archive_sha256": "c" * 64}
            self.initialize(state, contract)
            RESUME.publish_checkpoint(
                state / "CHECKPOINT.json", contract, state / "events"
            )
            value = json.loads((state / "CHECKPOINT.json").read_text())
            value["contract"] = {"archive_sha256": "d" * 64}
            (state / "CHECKPOINT.json").write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            )
            with self.assertRaisesRegex(COMMON.WorkloadError, "disagrees"):
                RESUME.replay_checkpoint(
                    state / "CHECKPOINT.json", contract, state / "events"
                )

    def test_postprocess_manifest_is_independently_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            self.write(root, "postprocess_complete.json", b"complete\n")
            manifest_file = self.write(root, "output_manifest.json", b"engine\n")
            self.write(root, "TM_Ex/derived.npz")
            manifest = RESUME.seal_postprocessing(root)
            manifest_file.write_bytes(b"tamper\n")
            with self.assertRaisesRegex(COMMON.WorkloadError, "digest mismatch"):
                RESUME.replay_manifest(root, manifest, "post", exact_inventory=True)

    def test_existing_extracted_input_is_replayed_byte_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            payload = b"sealed input\n"
            self.write(output / "input", "pkg/file.dat", payload)
            members = {"pkg/file.dat": hashlib.sha256(payload).hexdigest()}
            with mock.patch.object(ADAPTER, "AUNP_MEMBER_SHA256", members):
                package, reused = ADAPTER._prepare_package(
                    output / "unused-archive.tar.gz", output
                )
            self.assertTrue(reused)
            self.assertEqual(package, output / "input" / "aunp_r4000_repro")

    def test_partial_input_with_durable_state_is_not_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            self.write(output / "input", "pkg/file.dat", b"changed\n")
            self.write(
                output / ADAPTER.STAGE_STATE_DIRECTORY / ADAPTER.STAGE_EVENT_DIRECTORY,
                "00000000.json",
            )
            members = {"pkg/file.dat": hashlib.sha256(b"sealed\n").hexdigest()}
            with mock.patch.object(ADAPTER, "AUNP_MEMBER_SHA256", members):
                with self.assertRaises(COMMON.WorkloadError):
                    ADAPTER._prepare_package(output / "unused.tar.gz", output)

    def test_output_input_symlink_is_rejected_before_archive_replay(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            output = root / "lane"
            external = root / "external"
            external.mkdir()
            output.mkdir()
            (output / "input").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(COMMON.WorkloadError, "contains a symlink"):
                ADAPTER._prepare_package(root / "unused.tar.gz", output)

    def test_unsealed_partial_input_is_quarantined_before_reextract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            self.write(output / "input", "partial.dat", b"partial\n")
            payload = b"sealed\n"
            members = {"pkg/file.dat": hashlib.sha256(payload).hexdigest()}

            def extract(_archive, destination):
                self.write(destination, "pkg/file.dat", payload)
                return destination / "aunp_r4000_repro"

            with mock.patch.object(
                ADAPTER, "AUNP_MEMBER_SHA256", members
            ), mock.patch.object(ADAPTER, "extract_verified_aunp", side_effect=extract):
                package, reused = ADAPTER._prepare_package(
                    output / "unused.tar.gz", output
                )
            self.assertFalse(reused)
            self.assertEqual(package, output / "input" / "aunp_r4000_repro")
            self.assertEqual(
                (
                    output
                    / "attempt-quarantine"
                    / "bootstrap-input-000"
                    / "partial.dat"
                ).read_bytes(),
                b"partial\n",
            )

    def test_engine_import_cannot_create_input_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            module = self.write(root, "engine.py", b"VALUE = 1\n")
            previous = sys.dont_write_bytecode
            try:
                loaded = ADAPTER._load_engine(module)
                self.assertEqual(loaded.VALUE, 1)
                self.assertFalse((root / "__pycache__").exists())
            finally:
                sys.dont_write_bytecode = previous

    def test_adapter_resumes_marker_only_interruption_and_is_idempotent(self) -> None:
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
                simulation_name: str = "baseline_r4000"
                dft_minimum_run_time: float = 150.0
                dft_decay_tolerance: float = 1e-5

                def validate(self):
                    return None

            def zero_statistics():
                values = {}
                for group, stem in COMMON.FDTD_PHASE_COUNTERS.values():
                    values[group] = {
                        f"{backend}_{stem}_{field}": 0
                        for backend in ("cpu", "cuda")
                        for field in ("calls", "points")
                    }
                values["multi_gpu"] = {
                    name: 0 for name in COMMON.MULTI_GPU_COUNTERS
                }
                values["mpi_completion"] = {
                    name: 0 for name in COMMON.MPI_COMPLETION_COUNTERS
                }
                return values

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
                def __init__(self, structure=False):
                    self.structure = structure
                    self.fields = types.SimpleNamespace(t=0)

                def run(self, *_args, **_kwargs):
                    phases = {"curl", "update_eh", "source", "boundary", "dft"}
                    if self.structure:
                        phases.add("polarization")
                    for phase in phases:
                        group, stem = COMMON.FDTD_PHASE_COUNTERS[phase]
                        fake_gpu.values[group][f"cpu_{stem}_calls"] += 1
                        fake_gpu.values[group][f"cpu_{stem}_points"] += 4
                    self.fields.t += 20

                def timestep(self):
                    return self.fields.t

                def meep_time(self):
                    return self.fields.t / 10.0

            fake_mp = types.ModuleType("meep")
            fake_mp.__file__ = str(fake_meep_path)
            fake_mp.__version__ = "fixture"
            fake_mp.Simulation = FakeSimulation
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

                def allgather(self, value):
                    return [value]

                def Barrier(self):
                    return None

            fake_comm = FakeComm()
            fake_mpi = types.SimpleNamespace(COMM_WORLD=fake_comm)
            mpi4py_module = types.ModuleType("mpi4py")
            mpi4py_module.MPI = fake_mpi

            class FakeEngine:
                def __init__(self, engine_path: pathlib.Path, fail_reference=False):
                    self.engine_path = engine_path
                    self.fail_reference = fail_reference
                    self.AppConfig = FakeConfig
                    self.wait_for_stage_memory = lambda *_args: None
                    self.acquire_global_run_lock = lambda _cfg: None
                    self.release_global_run_lock = lambda _handle: None

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

                def simulation_root(self, cfg):
                    return pathlib.Path(cfg.output_dir) / "simulation" / cfg.simulation_name

                def polarization_directory_name(self, polarization):
                    return {"physical_ey": "TE_Ey", "physical_ex": "TM_Ex"}[
                        polarization
                    ]

                def render_simulation_stage_previews(self, _cfg, simulation_root):
                    raise AssertionError("TM-only adapter invoked archive previews")

                def run_reference_stage(self, _cfg, polarization, stage_directory):
                    stage_directory.mkdir(parents=True, exist_ok=True)
                    if self.fail_reference:
                        self.fail_reference = False
                        self_outer.write(
                            stage_directory, "reference_complete.json", b"marker-only\n"
                        )
                        raise RuntimeError("fixture interruption after engine marker")
                    simulation = FakeSimulation(structure=False)
                    simulation.run()
                    for name in RESUME.REFERENCE_REQUIRED_BASENAMES:
                        self_outer.write(stage_directory, name, name.encode() + b"\n")

                def run_structure_stage(self, _cfg, polarization, stage_directory):
                    simulation = FakeSimulation(structure=True)
                    simulation.run()
                    for name in RESUME.STRUCTURE_REQUIRED_BASENAMES:
                        self_outer.write(stage_directory, name, name.encode() + b"\n")

                def postprocess_simulation(self, _cfg, simulation_root):
                    self_outer.write(
                        simulation_root, "postprocess_complete.json", b"complete\n"
                    )
                    self_outer.write(simulation_root, "output_manifest.json", b"{}\n")
                    return simulation_root

                def validate_dft_h5_components(self, *_args):
                    return None

                def field_wavelengths_nm(self, _cfg):
                    return [700.0]

                def dense_analysis_wavelengths_nm(self, _cfg):
                    return [700.0]

                def plot_field_maps_and_metrics(self, *_args):
                    return None

                def plot_dense_line_heatmaps(self, *_args):
                    return None

                def plot_dense_probe_spectra(self, *_args):
                    return None

                def write_json_master(self, path, value):
                    COMMON.atomic_write_json(path, value)

                def write_output_manifest(self, simulation_root):
                    COMMON.atomic_write_json(
                        simulation_root / "output_manifest.json", {"files": []}
                    )
                    return simulation_root / "output_manifest.json"

                def run_simulations(self, cfg):
                    lock = self.acquire_global_run_lock(cfg)
                    try:
                        simulation_root = self.simulation_root(cfg)
                        simulation_root.mkdir(parents=True, exist_ok=True)
                        config = simulation_root / "simulation_config.json"
                        if not config.exists():
                            COMMON.atomic_write_json(
                                config,
                                {
                                    "config": self.serializable_config(cfg),
                                    "mpi_processes": 1,
                                    "script_sha256": COMMON.sha256_file(self.engine_path),
                                },
                            )
                            COMMON.atomic_write_json(
                                simulation_root / "run_environment.json", {"fixture": True}
                            )
                        self.wait_for_stage_memory(cfg, "preview", 1.0)
                        self.render_simulation_stage_previews(cfg, simulation_root)
                        for polarization in ("physical_ey", "physical_ex"):
                            stage_directory = simulation_root / self.polarization_directory_name(
                                polarization
                            )
                            self.run_reference_stage(cfg, polarization, stage_directory)
                            self.run_structure_stage(cfg, polarization, stage_directory)
                        self.postprocess_simulation(cfg, simulation_root)
                        return simulation_root
                    finally:
                        self.release_global_run_lock(lock)

            self_outer = self

            def prepare(_archive, lane_output):
                package = lane_output / "input" / "aunp_r4000_repro"
                engine_path = package / "fdtd" / "meep" / "aunp_periodic_fdtd.py"
                if not engine_path.exists():
                    self.write(package, "fdtd/meep/aunp_periodic_fdtd.py", b"engine\n")
                return package, engine_path.exists()

            prepare(archive, output)
            engine_path = (
                output
                / "input"
                / "aunp_r4000_repro"
                / "fdtd"
                / "meep"
                / "aunp_periodic_fdtd.py"
            )
            engines = [
                FakeEngine(engine_path, fail_reference=True),
                FakeEngine(engine_path),
                FakeEngine(engine_path),
            ]
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
                ADAPTER, "verify_file", side_effect=lambda path, *_args: pathlib.Path(path)
            ), mock.patch.object(
                ADAPTER,
                "verify_build_receipt_collective",
                return_value=({"receipt_id": "fixture-receipt"}, {}),
            ), mock.patch.object(
                ADAPTER, "_prepare_package", side_effect=prepare
            ), mock.patch.object(
                ADAPTER, "_load_engine", side_effect=engines
            ), mock.patch.object(
                ADAPTER, "physical_core_affinity", return_value=[{"fixture": True}]
            ):
                with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                    ADAPTER.main(argv)
                self.assertEqual(ADAPTER.main(argv), 0)
                self.assertEqual(ADAPTER.main(argv), 0)

            checkpoint = json.loads(
                (
                    output
                    / ADAPTER.STAGE_STATE_DIRECTORY
                    / ADAPTER.STAGE_CHECKPOINT_NAME
                ).read_text()
            )
            self.assertEqual(
                [item["stage"] for item in checkpoint["completed_stages"]],
                list(RESUME.STAGE_ORDER),
            )
            self.assertEqual(len(checkpoint["attempts"]), 2)
            self.assertIsNotNone(checkpoint["publication"])
            self.assertEqual(
                RESUME.manifest_paths(checkpoint["base_manifest"], "test base"),
                set(RESUME.BASE_REQUIRED_PATHS),
            )
            output_inventory = set(
                RESUME.inventory_files(
                    output / "outputs" / "simulation" / "baseline_r4000"
                )
            )
            self.assertFalse(
                any(
                    "preflight" in pathlib.PurePosixPath(path).parts
                    or "TE_Ey" in pathlib.PurePosixPath(path).parts
                    or "physical_ey" in path
                    for path in output_inventory
                )
            )
            lane = COMPARE.load_lane(output, COMPARE.AUNP_SCHEMA)
            self.assertEqual(
                lane["summary"]["qualification_profile"],
                COMMON.aunp_qualification_profile(),
            )
            self.assertEqual(
                lane["summary"]["physical_overrides"],
                COMMON.AUNP_PHYSICAL_OVERRIDES,
            )
            self.assertEqual(
                lane["summary"]["effective_config"]["dft_minimum_run_time"],
                20.0,
            )
            self.assertEqual(
                lane["summary"]["effective_config"]["dft_decay_tolerance"],
                5e-8,
            )
            replay = COMPARE.validate_aunp_exact_stage_resume(lane)
            self.assertEqual(replay["attempt_count"], 2)
            self.assertEqual(replay["quarantine_count"], 1)
            quarantined_marker = (
                output
                / "attempt-quarantine"
                / "attempt-001"
                / "TM_Ex"
                / "reference_complete.json"
            )
            self.assertEqual(quarantined_marker.read_bytes(), b"marker-only\n")
            checkpoint_path = (
                output
                / ADAPTER.STAGE_STATE_DIRECTORY
                / ADAPTER.STAGE_CHECKPOINT_NAME
            )
            tampered = json.loads(checkpoint_path.read_text())
            tampered["journal_terminal_sha256"] = "0" * 64
            checkpoint_path.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n"
            )
            with self.assertRaisesRegex(COMMON.WorkloadError, "disagrees"):
                COMPARE.validate_aunp_exact_stage_resume(lane)

    def test_tm_only_base_rejects_sealed_te_preview(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            simulation = root / "simulation"
            state = root / "state"
            for relative in RESUME.BASE_REQUIRED_PATHS:
                self.write(simulation, relative)
            self.write(simulation, "preflight/01_TE_Ey_reference.png", b"TE\n")
            contract = {"qualification_profile": COMMON.aunp_qualification_profile()}
            digest = self.initialize(state, contract)
            RESUME.append_event(
                state / "events",
                digest,
                "attempt-started",
                {"attempt_index": 0, "started_utc": "2000-01-01T00:00:00Z"},
            )
            RESUME.append_event(
                state / "events",
                digest,
                "base-sealed",
                {"manifest": RESUME.make_manifest(simulation)},
            )
            with self.assertRaisesRegex(COMMON.WorkloadError, "TM-only base"):
                RESUME.publish_checkpoint(
                    state / "CHECKPOINT.json", contract, state / "events"
                )


if __name__ == "__main__":
    unittest.main()
