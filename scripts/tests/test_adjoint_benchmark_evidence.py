from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gpmeep_benchmark_evidence import (  # noqa: E402
    PHASE_COUNTERS,
    canonical_float64_sha256,
    compare_gradient_vectors,
    compare_workloads,
    sha256_file,
    validate_backend_and_paths,
)


def load_runner():
    path = SCRIPTS / "run-adjoint-benchmark.py"
    spec = importlib.util.spec_from_file_location("gpmeep_adjoint_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import adjoint benchmark runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MaterialGradientStatsTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner()

    def record(self, **overrides):
        values = {
            "points": 10,
            "forced_legacy_points": 0,
            "analytic_points": 10,
            "edge_legacy_points": 0,
            "unsupported_points": 0,
            "requests": 40,
            "analytic_requests": 40,
            "legacy_requests": 0,
        }
        values.update(overrides)
        return "material-gradient-stats: " + " ".join(
            f"{key}={value}" for key, value in values.items()
        )

    def test_extracts_and_gates_analytic_required(self):
        stats = self.runner.extract_material_gradient_stats(self.record())
        self.assertEqual(
            self.runner.validate_material_gradient_stats(
                stats, "analytic-required"
            ),
            [],
        )

    def test_rejects_bad_accounting_and_silent_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "accounting"):
            self.runner.extract_material_gradient_stats(
                self.record(legacy_requests=1)
            )
        stats = self.runner.extract_material_gradient_stats(
            self.record(
                analytic_points=9,
                unsupported_points=1,
                analytic_requests=36,
                legacy_requests=4,
            )
        )
        self.assertTrue(
            self.runner.validate_material_gradient_stats(
                stats, "analytic-required"
            )
        )

    def test_legacy_requires_all_requests_on_legacy_path(self):
        stats = self.runner.extract_material_gradient_stats(
            self.record(
                forced_legacy_points=10,
                analytic_points=0,
                analytic_requests=0,
                legacy_requests=40,
            )
        )
        self.assertEqual(
            self.runner.validate_material_gradient_stats(stats, "legacy"), []
        )


def counters(active_backend: str, points_offset: int = 0):
    result = {"cpu": {}, "cuda": {}}
    for index, operation in enumerate(PHASE_COUNTERS):
        for backend in ("cpu", "cuda"):
            active = backend == active_backend and operation != "polarization"
            result[backend][operation] = {
                "calls": 2 + index if active else 0,
                "points": 100 + index + points_offset if active else 0,
            }
    return result


def synthetic_result(
    backend: str,
    gradient: list[float],
    runtime: dict,
):
    return {
        "schema_version": 3,
        "backend": {
            "requested": backend,
            "active": backend,
            "compiled": True,
            "runtime_available": True,
            "single_precision": True,
            "strict_cuda_marker": backend == "cuda",
        },
        "runtime": runtime,
        "workload": {
            "resolution": 64,
            "run_time": 40.0,
            "cell_size": 8.0,
            "design_resolution": 20,
            "weights_sha256": "weights",
            "direction_sha256": "direction",
            "design_shape": [41, 41],
            "design_variables": 1681,
            "yee_cells": 262144,
            "phases": {
                "forward": {
                    "timesteps": 101,
                    "end_time": 51.671875,
                    "backend_counters": counters(backend),
                },
                "adjoint": {
                    "timesteps": 103,
                    "end_time": 155.0,
                    "backend_counters": counters(backend, 10),
                },
                "gradient": {
                    "backend_counters": {
                        "cpu": {
                            operation: {"calls": 0, "points": 0}
                            for operation in PHASE_COUNTERS
                        },
                        "cuda": {
                            operation: {"calls": 0, "points": 0}
                            for operation in PHASE_COUNTERS
                        },
                    }
                },
            },
        },
        "result": {
            "gradient": gradient,
            "gradient_shape": [len(gradient)],
            "gradient_count": len(gradient),
            "gradient_sha256": canonical_float64_sha256(gradient),
        },
    }


class GradientGateTests(unittest.TestCase):
    def test_compares_all_1681_values(self):
        cpu_gradient = [0.25] * 1681
        cpu_gradient[0] = 1.0
        cpu_gradient[1] = 2.0
        cuda_gradient = list(cpu_gradient)
        cuda_gradient[0], cuda_gradient[1] = (
            cuda_gradient[1],
            cuda_gradient[0],
        )
        cpu = synthetic_result("cpu", cpu_gradient, {})
        cuda = synthetic_result("cuda", cuda_gradient, {})

        comparison = compare_gradient_vectors(
            cpu, cuda, atol=1e-6, rtol=2e-3, expected_count=1681
        )

        self.assertFalse(comparison["pass"])
        self.assertEqual(comparison["compared_count"], 1681)
        self.assertIn(comparison["max_absolute_error_index"], (0, 1))

    def test_matching_full_vector_passes(self):
        gradient = [index / 1681 for index in range(1681)]
        cpu = synthetic_result("cpu", gradient, {})
        cuda = synthetic_result("cuda", gradient, {})
        comparison = compare_gradient_vectors(
            cpu, cuda, atol=1e-6, rtol=2e-3, expected_count=1681
        )
        self.assertTrue(comparison["pass"])
        self.assertEqual(comparison["relative_l2_error"], 0)

    def test_count_and_shape_are_independent_release_gates(self):
        gradient = [1.0] * 1681
        cpu = synthetic_result("cpu", gradient, {})
        cuda = synthetic_result("cuda", gradient, {})
        cpu["result"]["gradient_shape"] = [41, 41]
        comparison = compare_gradient_vectors(
            cpu, cuda, atol=1e-6, rtol=2e-3, expected_count=1681
        )
        self.assertFalse(comparison["pass"])
        self.assertFalse(comparison["structural_pass"])
        self.assertTrue(comparison["component_pass"])

        cpu = synthetic_result("cpu", gradient[:-1], {})
        cuda = synthetic_result("cuda", gradient[:-1], {})
        comparison = compare_gradient_vectors(
            cpu, cuda, atol=1e-6, rtol=2e-3, expected_count=1681
        )
        self.assertFalse(comparison["pass"])
        self.assertEqual(comparison["compared_count"], 1680)

    def test_invalid_hash_and_nonfinite_values_fail_closed(self):
        result = synthetic_result("cpu", [1.0] * 1681, {})
        result["result"]["gradient_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            compare_gradient_vectors(
                result,
                synthetic_result("cuda", [1.0] * 1681, {}),
                atol=1e-6,
                rtol=2e-3,
                expected_count=1681,
            )
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            canonical_float64_sha256([float("nan")])


class WorkloadGateTests(unittest.TestCase):
    def setUp(self):
        self.cpu = synthetic_result("cpu", [1.0] * 1681, {})
        self.cuda = synthetic_result("cuda", [1.0] * 1681, {})

    def test_equal_timestep_and_point_workloads_pass(self):
        self.assertTrue(compare_workloads(self.cpu, self.cuda)["pass"])

    def test_one_timestep_or_point_difference_fails(self):
        self.cuda["workload"]["phases"]["adjoint"]["timesteps"] += 1
        self.assertFalse(compare_workloads(self.cpu, self.cuda)["pass"])
        self.cuda["workload"]["phases"]["adjoint"]["timesteps"] -= 1
        self.cuda["workload"]["phases"]["forward"]["backend_counters"][
            "cuda"
        ]["curl"]["points"] += 1
        self.assertFalse(compare_workloads(self.cpu, self.cuda)["pass"])

    def test_one_phase_end_time_difference_fails(self):
        self.cuda["workload"]["phases"]["forward"]["end_time"] += 0.125
        comparison = compare_workloads(self.cpu, self.cuda)
        self.assertFalse(comparison["pass"])
        failed = [
            item
            for item in comparison["comparisons"]
            if item["metric"] == "end_time" and not item["pass"]
        ]
        self.assertEqual(len(failed), 1)


class BackendPathGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.temporary.name)
        self.python = root / "python"
        self.meep_module = root / "build" / "python" / "meep" / "__init__.py"
        self.extension = self.meep_module.parent / "_meep.so"
        self.libmeep = root / "build" / "src" / ".libs" / "libmeep.so.38.0.0"
        for path, value in (
            (self.python, "python"),
            (self.meep_module, "meep"),
            (self.extension, "extension"),
            (self.libmeep, "libmeep"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        self.runtime = {
            "python_executable": {
                "path": str(self.python),
                "sha256": sha256_file(self.python),
            },
            "meep_module": {
                "path": str(self.meep_module),
                "sha256": sha256_file(self.meep_module),
            },
            "extension": {
                "path": str(self.extension),
                "sha256": sha256_file(self.extension),
            },
            "libmeep": {
                "path": str(self.libmeep),
                "sha256": sha256_file(self.libmeep),
            },
        }

    def tearDown(self):
        self.temporary.cleanup()

    def gate(self, result):
        return validate_backend_and_paths(
            result,
            expected_backend="cuda",
            expected_python=self.python,
            expected_python_sha256=sha256_file(self.python),
            expected_meep_module=self.meep_module,
            expected_meep_module_sha256=sha256_file(self.meep_module),
            expected_extension=self.extension,
            expected_extension_sha256=sha256_file(self.extension),
            expected_libmeep=self.libmeep,
            expected_libmeep_sha256=sha256_file(self.libmeep),
        )

    def test_exact_cuda_paths_and_exclusive_phases_pass(self):
        result = synthetic_result("cuda", [1.0] * 1681, self.runtime)
        self.assertEqual(self.gate(result), {"pass": True, "problems": []})

    def test_wrong_hash_and_cpu_fallback_fail(self):
        result = synthetic_result("cuda", [1.0] * 1681, self.runtime)
        result["runtime"]["extension"]["sha256"] = "f" * 64
        result["workload"]["phases"]["forward"]["backend_counters"]["cpu"][
            "curl"
        ] = {"calls": 1, "points": 10}
        gate = self.gate(result)
        self.assertFalse(gate["pass"])
        self.assertTrue(any("SHA-256" in problem for problem in gate["problems"]))
        self.assertTrue(any("fallback" in problem for problem in gate["problems"]))

    def test_partial_cuda_coverage_fails(self):
        result = synthetic_result("cuda", [1.0] * 1681, self.runtime)
        result["workload"]["phases"]["adjoint"]["backend_counters"]["cuda"][
            "dft"
        ] = {"calls": 0, "points": 0}
        gate = self.gate(result)
        self.assertFalse(gate["pass"])
        self.assertTrue(any("adjoint" in problem for problem in gate["problems"]))


class StaleArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_start_and_failure_remove_stale_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            (output / "report.json").write_text('{"pass": true}\n')
            (output / "report.md").write_text("old\n")
            (output / "COMPLETE").write_text("old\n")

            self.runner.prepare_output(output, "new-run")
            self.assertFalse((output / "report.json").exists())
            self.assertFalse((output / "COMPLETE").exists())
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(state["state"], "RUNNING")

            self.runner.mark_failed(output, "new-run", RuntimeError("boom"))
            self.assertFalse((output / "report.json").exists())
            self.assertFalse((output / "COMPLETE").exists())
            failure = json.loads((output / "FAILED.json").read_text())
            self.assertEqual(failure["state"], "FAILED")
            self.assertEqual(failure["run_id"], "new-run")

    def test_complete_marker_binds_atomic_report_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            self.runner.prepare_output(output, "complete-run")
            report = {
                "schema_version": 3,
                "state": "COMPLETE",
                "run_id": "complete-run",
                "gate": {"pass": True},
            }
            self.runner.publish_complete(output, "complete-run", report, "ok\n")
            marker = json.loads((output / "COMPLETE").read_text())
            self.assertEqual(marker["state"], "COMPLETE")
            self.assertEqual(marker["report_sha256"], sha256_file(output / "report.json"))
            self.assertFalse((output / "FAILED.json").exists())

    def test_complete_marker_rejects_failed_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            self.runner.prepare_output(output, "failed-gate")
            report = {
                "schema_version": 3,
                "state": "COMPLETE",
                "run_id": "failed-gate",
                "gate": {"pass": False},
            }
            with self.assertRaisesRegex(RuntimeError, "unqualified"):
                self.runner.publish_complete(output, "failed-gate", report, "bad\n")
            self.assertFalse((output / "COMPLETE").exists())

    def test_missing_build_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runtime_paths = {"build_root": root / "build"}
            with self.assertRaisesRegex(RuntimeError, "required build provenance"):
                self.runner.verify_required_build_receipt(root, runtime_paths)


class CleanEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_inherited_backend_python_omp_and_cuda_tuning_are_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = SimpleNamespace(
                build_python=root / "build-python",
                output=root / "output",
                omp_threads=3,
                material_gradient_path="analytic-required",
            )
            inherited = {
                "PYTHONPATH": "/contaminating/package",
                "MEEP_GPU_BACKEND": "auto",
                "MEEP_GPU_STRICT": "1",
                "MEEP_MATERIAL_GRADIENT_PATH": "legacy",
                "MEEP_MATERIAL_GRADIENT_STATS": "stale",
                "GPMEEP_VALIDATION_STRICT_CUDA": "1",
                "OMP_NUM_THREADS": "99",
                "OMP_WAIT_POLICY": "ACTIVE",
                "CUDA_LAUNCH_BLOCKING": "1",
                "CUDA_VISIBLE_DEVICES": "2",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }
            with mock.patch.dict("os.environ", inherited, clear=True):
                env = self.runner.clean_child_environment(
                    args,
                    root,
                    "cpu",
                    root / "stats.json",
                    "synthetic-nonce",
                    "a" * 64,
                )
            self.assertNotIn("/contaminating/package", env["PYTHONPATH"])
            self.assertEqual(env["MEEP_GPU_BACKEND"], "cpu")
            self.assertNotIn("MEEP_GPU_STRICT", env)
            self.assertEqual(
                env["MEEP_MATERIAL_GRADIENT_PATH"], "analytic-required"
            )
            self.assertEqual(env["MEEP_MATERIAL_GRADIENT_STATS"], "1")
            self.assertNotIn("GPMEEP_VALIDATION_STRICT_CUDA", env)
            self.assertEqual(env["OMP_NUM_THREADS"], "3")
            self.assertEqual(env["OMP_DYNAMIC"], "FALSE")
            self.assertNotIn("OMP_WAIT_POLICY", env)
            self.assertNotIn("CUDA_LAUNCH_BLOCKING", env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertEqual(
                env["GPMEEP_VALIDATION_RUN_NONCE"], "synthetic-nonce"
            )
            self.assertEqual(
                env["GPMEEP_VALIDATION_BUILD_RECEIPT_ID"], "a" * 64
            )


if __name__ == "__main__":
    unittest.main()
