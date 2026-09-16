from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gpmeep_benchmark_evidence import sha256_file  # noqa: E402


def load_comparator():
    path = SCRIPTS / "compare-material-gradient-benchmarks.py"
    spec = importlib.util.spec_from_file_location(
        "gpmeep_material_gradient_comparator", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import MaterialGrid comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_report(*, repeats: int = 2, warmups: int = 1):
    runs = []
    for backend, scale in (("cpu", 2.0), ("cuda", 1.0)):
        for iteration in range(warmups):
            runs.append(
                {
                    "backend": backend,
                    "iteration": iteration,
                    "warmup": True,
                    "process_seconds": scale + 0.5,
                    "result": {
                        "timing": {
                            "elapsed_seconds": scale,
                            "gradient_seconds": 0.1,
                            "forward_seconds": 0.5 * scale,
                            "adjoint_seconds": 0.4 * scale,
                        }
                    },
                }
            )
        for iteration in range(repeats):
            elapsed = scale + 0.1 * iteration
            runs.append(
                {
                    "backend": backend,
                    "iteration": iteration,
                    "warmup": False,
                    "process_seconds": elapsed + 0.5,
                    "result": {
                        "timing": {
                            "elapsed_seconds": elapsed,
                            "gradient_seconds": 0.1,
                            "forward_seconds": 0.5 * elapsed,
                            "adjoint_seconds": 0.4 * elapsed,
                        }
                    },
                }
            )
    cpu = [2.0, 2.1][:repeats]
    cuda = [1.0, 1.1][:repeats]
    return {
        "schema_version": 3,
        "state": "COMPLETE",
        "run_id": "synthetic",
        "gate": {"pass": True},
        "workload": {
            "repeats": repeats,
            "warmups": warmups,
            "material_gradient_path": "legacy",
        },
        "timing": {
            "cpu_seconds": cpu,
            "cuda_seconds": cuda,
            "cpu_median_seconds": 2.05 if repeats == 2 else 2.0,
            "cuda_median_seconds": 1.05 if repeats == 2 else 1.0,
            "speedup_cpu_over_cuda": (
                (2.05 / 1.05) if repeats == 2 else 2.0
            ),
        },
        "runs": runs,
    }


class ComparatorEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparator = load_comparator()

    def test_run_matrix_and_aggregate_timing_are_recomputed(self):
        report = synthetic_report()
        self.comparator.validate_run_matrix(report)
        self.comparator.validate_standalone_timing(report)

        report["workload"]["repeats"] = 3
        with self.assertRaisesRegex(RuntimeError, "measured-run count"):
            self.comparator.validate_run_matrix(report)

    def test_stale_standalone_speedup_is_rejected(self):
        report = synthetic_report()
        report["timing"]["speedup_cpu_over_cuda"] = 99
        with self.assertRaisesRegex(RuntimeError, "stale or invalid"):
            self.comparator.validate_standalone_timing(report)

    def test_conservative_improvement_uses_worst_samples(self):
        self.assertEqual(
            self.comparator.conservative_improvement(
                [12.0, 10.0, 11.0], [4.0, 5.0, 3.0]
            ),
            2.0,
        )
        with self.assertRaisesRegex(RuntimeError, "positive and finite"):
            self.comparator.conservative_improvement([1.0], [0.0])

    def test_load_report_requires_hash_bound_complete_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "report.json"
            report = synthetic_report(repeats=1)
            path.write_text(json.dumps(report), encoding="utf-8")
            marker = {
                "state": "COMPLETE",
                "run_id": report["run_id"],
                "report": str(path.resolve()),
                "report_sha256": sha256_file(path),
            }
            (root / "COMPLETE").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            loaded = self.comparator.load_report(path.resolve(), "legacy")
            self.assertEqual(loaded["run_id"], "synthetic")

            marker["report_sha256"] = "0" * 64
            (root / "COMPLETE").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "does not authenticate"):
                self.comparator.load_report(path.resolve(), "legacy")

    def test_failure_lifecycle_never_leaves_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            (output / "COMPLETE").write_text("stale", encoding="utf-8")
            self.comparator.prepare_output(output, "new")
            self.assertFalse((output / "COMPLETE").exists())
            self.comparator.mark_failed(output, "new", RuntimeError("bad gate"))
            self.assertFalse((output / "COMPLETE").exists())
            failure = json.loads((output / "FAILED.json").read_text())
            self.assertEqual(failure["state"], "FAILED")

    def test_publish_rejects_failed_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            report = {
                "schema_version": 1,
                "state": "COMPLETE",
                "run_id": "bad",
                "gate": {"pass": False},
            }
            with self.assertRaisesRegex(RuntimeError, "unqualified"):
                self.comparator.publish_complete(output, "bad", report, "bad\n")
            self.assertFalse((output / "COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
