#!/usr/bin/env python3
"""Focused tests for one sealed M3 performance sample."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "run_m3_performance_sample.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_performance_sample", SOURCE)
assert SPEC is not None and SPEC.loader is not None
SAMPLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAMPLE
SPEC.loader.exec_module(SAMPLE)


GPU0 = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GPU1 = "GPU-11111111-2222-3333-4444-555555555555"


def audited_run(backend: str, seconds: float) -> dict[str, object]:
    other = "cuda" if backend == "cpu" else "cpu"
    return {
        "run_wall_seconds": seconds,
        "active_backend": backend,
        "run_phase_calls": {backend: 1, other: 0},
    }


class M3PerformanceSampleTests(unittest.TestCase):
    def test_sample_identity_is_canonical(self) -> None:
        SAMPLE.validate_sample_identity("warmup", 0)
        SAMPLE.validate_sample_identity("measured", 1)
        for kind, cycle in (("warmup", 1), ("measured", 0), ("other", 1)):
            with self.subTest(kind=kind, cycle=cycle):
                with self.assertRaisesRegex(Exception, "sample"):
                    SAMPLE.validate_sample_identity(kind, cycle)

    def test_topology_selection_binds_exact_gpu_inventory(self) -> None:
        devices = SAMPLE.normalized_devices(f"{GPU0},{GPU1}")
        cpu, requested, policy = SAMPLE.select_lane(
            "cpu8", devices, "python/examples/edge_emitter_3D.py"
        )
        self.assertEqual((cpu.backend, cpu.ranks, cpu.devices), ("cpu", 8, ()))
        self.assertEqual(requested, "cpu")
        self.assertIsNone(policy["active_backends"])
        cpu1, requested, _ = SAMPLE.select_lane(
            "cpu1", devices, "python/examples/metasurface_lens.py"
        )
        self.assertEqual((cpu1.backend, cpu1.ranks, cpu1.devices), ("cpu", 1, ()))
        self.assertEqual(requested, "cpu")
        cuda2, requested, _ = SAMPLE.select_lane(
            "cuda2", devices, "python/examples/edge_emitter_3D.py"
        )
        self.assertEqual((cuda2.backend, cuda2.ranks), ("cuda", 2))
        self.assertEqual(cuda2.devices, devices)
        self.assertEqual(requested, "cuda")
        automatic, requested, policy = SAMPLE.select_lane(
            "auto1", devices, "python/examples/metasurface_lens.py"
        )
        self.assertEqual(requested, "auto")
        self.assertEqual(automatic.devices, devices[:1])
        self.assertEqual(automatic.backend, "cuda")
        self.assertEqual(policy["active_backends"][-1], "cuda")
        with self.assertRaisesRegex(Exception, "distinct GPUs"):
            SAMPLE.normalized_devices(f"{GPU0},{GPU0.lower()}")

    def test_automatic_environment_removes_strict_cuda_marker(self) -> None:
        lane = SAMPLE.mpi_case.Lane("auto", "cuda", 1, (GPU0,))
        with mock.patch.object(
            SAMPLE.mpi_case,
            "clean_environment",
            return_value={
                "MEEP_GPU_BACKEND": "cuda",
                "GPMEEP_VALIDATION_STRICT_CUDA": "1",
            },
        ):
            environment = SAMPLE.sample_environment(
                lane,
                "auto",
                python=pathlib.Path("/python"),
                build_python=pathlib.Path("/build"),
                runtime_contract={},
                run_nonce="nonce",
            )
        self.assertEqual(environment["MEEP_GPU_BACKEND"], "auto")
        self.assertEqual(
            environment["GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND"],
            "auto",
        )
        self.assertNotIn("GPMEEP_VALIDATION_STRICT_CUDA", environment)

    def test_timing_contracts_are_rederived_from_every_run(self) -> None:
        summed = SAMPLE.derive_timing(
            "python/examples/edge_emitter_3D.py",
            [audited_run("cpu", 1.0), audited_run("cpu", 2.0)],
        )
        self.assertEqual(summed["primary_seconds"], 3.0)
        runs = [audited_run("cpu", float(index + 1)) for index in range(20)]
        metasurface = SAMPLE.derive_timing(
            "python/examples/metasurface_lens.py", runs
        )
        self.assertEqual(metasurface["small_seconds"], 19.0)
        self.assertEqual(metasurface["large_seconds"], 20.0)
        self.assertEqual(metasurface["primary_seconds"], 20.0)
        with self.assertRaisesRegex(Exception, "exactly 20"):
            SAMPLE.derive_timing("python/examples/metasurface_lens.py", runs[:-1])
        with self.assertRaisesRegex(Exception, "not sealed"):
            SAMPLE.derive_timing("python/examples/other.py", runs)

    def test_forced_policy_requires_whole_owner_dispatch(self) -> None:
        lane = SAMPLE.mpi_case.Lane("cpu", "cpu", 8, ())
        policy = {
            "schema": "forced-v1",
            "classification": "forced",
            "active_backends": None,
        }
        result = SAMPLE.validate_policy(
            [audited_run("cpu", 1.0)], "cpu", lane, policy
        )
        self.assertEqual(result["active_backends"], ["cpu"])
        with self.assertRaisesRegex(Exception, "owner decisions"):
            SAMPLE.validate_policy(
                [audited_run("cuda", 1.0)], "cpu", lane, policy
            )

    def test_process_schema_rejects_boolean_sizes(self) -> None:
        valid = {
            "exit_code": 0,
            "timeout": False,
            "output_limit": None,
            "spawn_error": None,
            "stdout_size_bytes": 10,
            "stderr_size_bytes": 0,
            "duration_seconds": 1.0,
        }
        SAMPLE._validate_process(valid)
        invalid = dict(valid, stdout_size_bytes=False)
        with self.assertRaisesRegex(Exception, "process record"):
            SAMPLE._validate_process(invalid)

    def test_pending_terminal_is_replayed_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            (output / "report.json").write_text("{}", encoding="utf-8")
            report = {
                "case_path": "fixture.py",
                "topology": "cpu8",
                "sample_kind": "warmup",
                "cycle_index": 0,
                "receipt_id": "receipt",
            }
            with mock.patch.object(SAMPLE, "verify_complete") as replay:
                SAMPLE.publish_terminal(output, report)
            replay.assert_called_once_with(output, "PENDING_COMPLETE")
            self.assertFalse((output / "PENDING_COMPLETE").exists())
            marker = json.loads((output / "COMPLETE").read_text())
            self.assertEqual(marker["sample_kind"], "warmup")


if __name__ == "__main__":
    unittest.main()
