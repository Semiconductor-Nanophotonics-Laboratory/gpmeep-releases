#!/usr/bin/env python3
"""Focused tests for the M3 automatic-backend example runner."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "run_m3_auto_example_case.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_auto_example_case", SOURCE)
assert SPEC is not None and SPEC.loader is not None
CASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CASE
SPEC.loader.exec_module(CASE)


GPU = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def run_record(
    backend: str,
    wall_seconds: float = 0.25,
    requested_backend: str = "auto",
) -> dict[str, object]:
    other = "cuda" if backend == "cpu" else "cpu"
    return {
        "timestep": 10,
        "timestep_delta": 10,
        "meep_time": 1.0,
        "run_wall_seconds": wall_seconds,
        "field_energy": 2.0,
        "dft_norm": 3.0,
        "active_backend": backend,
        "requested_backend": requested_backend,
        "execution_diagnostic": "fixture",
        "runtime_counters": {
            "runtime_availability_probes": 0,
            "runtime_device_enumerations": 0,
            "runtime_device_selections": 0,
        },
        "run_phase_calls": {backend: 2, other: 0},
        "measurement_phase_calls": {"cpu": 0, "cuda": 0},
    }


class M3AutoExampleCaseTests(unittest.TestCase):
    def test_v1_auto_routes_all_have_sealed_policies(self) -> None:
        coverage = json.loads(
            (
                ROOT
                / "scripts/python-validation/v1_feature_coverage.json"
            ).read_text(encoding="utf-8")
        )
        routed = {
            route["case_id"]
            for route in coverage["routes"]
            if "auto" in route["topologies"]
        }
        self.assertEqual(
            routed,
            {
                "python/examples/antenna_pec_ground_plane_1D.py",
                "python/examples/cherenkov-radiation.py",
                "python/examples/edge_emitter_3D.py",
                "python/examples/stochastic_emitter_line.py",
            },
        )
        self.assertTrue(routed <= set(CASE.POLICIES))

    def test_policy_inventory_is_exact_and_returned_as_a_clone(self) -> None:
        self.assertEqual(
            set(CASE.POLICIES),
            {
                "python/examples/antenna_pec_ground_plane_1D.py",
                "python/examples/cherenkov-radiation.py",
                "python/examples/edge_emitter_3D.py",
                "python/examples/metasurface_lens.py",
                "python/examples/stochastic_emitter_line.py",
            },
        )
        antenna = CASE.policy_for(
            "python/examples/antenna_pec_ground_plane_1D.py"
        )
        self.assertEqual(antenna["active_backends"], ["cpu"] * 8)
        self.assertEqual(
            antenna["classification"],
            "launch-dominated-small-domain-cpu-policy",
        )
        cherenkov = CASE.policy_for("python/examples/cherenkov-radiation.py")
        self.assertEqual(cherenkov["active_backends"], ["cpu"] * 4)
        self.assertEqual(
            cherenkov["classification"],
            "long-horizon-small-domain-cpu-policy",
        )
        edge = CASE.policy_for("python/examples/edge_emitter_3D.py")
        self.assertEqual(edge["active_backends"], ["cpu"] * 4)
        self.assertEqual(
            edge["classification"],
            "production-crossover-bounded-validation-cpu-policy",
        )
        metasurface = CASE.policy_for("python/examples/metasurface_lens.py")
        self.assertEqual(len(metasurface["active_backends"]), 20)
        self.assertEqual(metasurface["active_backends"][:19], ["cpu"] * 19)
        self.assertEqual(metasurface["active_backends"][-1], "cuda")
        metasurface["active_backends"][0] = "cuda"
        self.assertEqual(
            CASE.policy_for("python/examples/metasurface_lens.py")[
                "active_backends"
            ][0],
            "cpu",
        )
        stochastic = CASE.policy_for(
            "python/examples/stochastic_emitter_line.py"
        )
        self.assertEqual(stochastic["active_backends"], ["cpu"] * 60)
        with self.assertRaisesRegex(Exception, "no sealed case"):
            CASE.policy_for("python/examples/unsealed.py")

    def test_audit_reader_and_policy_validation_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            stdout = pathlib.Path(raw) / "stdout.log"
            runs = [run_record("cpu"), run_record("cuda", 0.5)]
            stdout.write_text(
                "noise\n"
                + CASE.oracle.AUDIT_PREFIX
                + json.dumps({"runs": runs})
                + "\n",
                encoding="utf-8",
            )
            parsed = CASE.audit_runs(stdout)
            result = CASE.validate_policy_runs(
                parsed,
                {
                    "schema": "fixture-v1",
                    "classification": "fixture",
                    "active_backends": ["cpu", "cuda"],
                },
            )
            self.assertEqual(result["cpu_run_count"], 1)
            self.assertEqual(result["cuda_run_count"], 1)
            self.assertEqual(result["run_wall_seconds"], [0.25, 0.5])

            invalid = [run_record("cpu")]
            invalid[0]["run_phase_calls"] = {"cpu": 1, "cuda": False}
            with self.assertRaisesRegex(Exception, "whole-owner"):
                CASE.validate_policy_runs(
                    invalid,
                    {
                        "schema": "fixture-v1",
                        "classification": "fixture",
                        "active_backends": ["cpu"],
                    },
                )
            stdout.write_text(
                CASE.oracle.AUDIT_PREFIX
                + json.dumps({"runs": [run_record("cpu", 0.0)]})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Exception, "audit differs"):
                CASE.audit_runs(stdout)

            stdout.write_text(
                CASE.oracle.AUDIT_PREFIX
                + json.dumps({"runs": [run_record("cpu", 0.25, "cpu")]})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(CASE.audit_runs(stdout, "cpu")[0]["timestep"], 10)
            with self.assertRaisesRegex(Exception, "requested backend"):
                CASE.audit_runs(stdout, "bogus")

    def test_rank_bundle_enforces_automatic_cpu_runtime_neutrality(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw) / "lane"
            for name in ("identity", "statistics", "work"):
                (root / name).mkdir(parents=True)
            (root / "work" / "rank-00000").mkdir()
            command = ["/opt/python3.11", "/repo/example.py"]
            identity = {
                "schema": CASE.rank_launcher.SCHEMA,
                "rank": 0,
                "world_size": 1,
                "local_rank": 0,
                "pid": 7000,
                "executable": command[0],
                "command": command,
                "run_nonce": "nonce",
                "requested_backend": "auto",
                "visible_devices": GPU,
                "statistics": str(root / "statistics/rank-00000.json"),
                "work": str(root / "work/rank-00000"),
            }
            runtime = {
                "receipt_id": "receipt",
                **{
                    name: {"path": f"/runtime/{name}", "sha256": name}
                    for name in (
                        "python_executable",
                        "meep_module",
                        "extension",
                        "libmeep",
                    )
                },
            }
            statistics = {
                "pid": 7000,
                "capture_status": "captured",
                "gpu_api_available": True,
                "run_nonce": "nonce",
                "build_receipt_id": "receipt",
                "single_precision": True,
                "with_mpi": True,
                "gpu_compiled": True,
                "strict_cuda_marker": False,
                "requested_backend": "auto",
                "active_backend": "cpu",
                "selected_device_identifier": "",
                "statistics": {
                    "runtime": {
                        "runtime_availability_probes": 0,
                        "runtime_device_enumerations": 0,
                        "runtime_device_selections": 0,
                    }
                },
                **runtime,
            }
            (root / "identity/rank-00000.json").write_text(
                json.dumps(identity), encoding="utf-8"
            )
            stats_path = root / "statistics/rank-00000.json"
            stats_path.write_text(json.dumps(statistics), encoding="utf-8")
            (root / "stdout.log").write_text(
                CASE.oracle.AUDIT_PREFIX
                + json.dumps({"runs": [run_record("cpu")]})
                + "\n",
                encoding="utf-8",
            )
            policy = {
                "schema": "fixture-v1",
                "classification": "fixture",
                "active_backends": ["cpu"],
            }
            records, result = CASE.validate_rank_bundle(
                root, command, runtime, "nonce", GPU, policy
            )
            self.assertEqual(records[0]["backend_contract"], "PASS")
            self.assertEqual(result["cpu_run_count"], 1)

            statistics["statistics"]["runtime"][
                "runtime_availability_probes"
            ] = 1
            stats_path.write_text(json.dumps(statistics), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "touched the CUDA runtime"):
                CASE.validate_rank_bundle(
                    root, command, runtime, "nonce", GPU, policy
                )

    def test_pending_terminal_is_replayed_before_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            (output / "report.json").write_text("{}", encoding="utf-8")
            report = {"case_path": "fixture.py", "receipt_id": "receipt"}
            with mock.patch.object(CASE, "verify_complete") as replay:
                CASE.publish_terminal(output, report)
            replay.assert_called_once_with(output, "PENDING_COMPLETE")
            self.assertFalse((output / "PENDING_COMPLETE").exists())
            marker = json.loads((output / "COMPLETE").read_text())
            self.assertEqual(marker["outcome"], "PASS")

            (output / "COMPLETE").unlink()
            with mock.patch.object(
                CASE, "verify_complete", side_effect=RuntimeError("reject")
            ):
                with self.assertRaisesRegex(RuntimeError, "reject"):
                    CASE.publish_terminal(output, report)
            self.assertTrue((output / "PENDING_COMPLETE").is_file())
            self.assertFalse((output / "COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
