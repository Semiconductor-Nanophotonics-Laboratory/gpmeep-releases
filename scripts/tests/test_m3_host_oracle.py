#!/usr/bin/env python3
"""Focused tests for the four CPU-only M3 host compatibility oracles."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "run_m3_host_oracle.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_host_oracle", SOURCE)
assert SPEC is not None and SPEC.loader is not None
HOST = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HOST
SPEC.loader.exec_module(HOST)


def eps_metrics():
    return {
        "schema_version": 1,
        "frequency_count": 96,
        "initial_error": 1.0,
        "final_error": 1e-20,
        "improvement_ratio": 1e20,
        "fit_relative_l2": 1e-10,
        "gradient_relative_error": 1e-9,
        "material_relative_l2": 1e-14,
        "parameter_relative_l2": 1e-9,
        "fitted_parameters": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }


class M3HostOracleTests(unittest.TestCase):
    def test_bounded_text_and_metric_parser_are_fail_closed(self) -> None:
        stream = HOST.BoundedText("fixture", limit_bytes=4)
        self.assertEqual(stream.write("test"), 4)
        with self.assertRaisesRegex(Exception, "output limit"):
            stream.write("x")
        value = HOST._prefixed_json("prefix:{\"x\":1}\n", "prefix:", "fixture")
        self.assertEqual(value, {"x": 1})
        with self.assertRaisesRegex(Exception, "repeats key"):
            HOST._prefixed_json(
                "prefix:{\"x\":1,\"x\":2}\n", "prefix:", "fixture"
            )
        with self.assertRaisesRegex(Exception, "nonfinite"):
            HOST._prefixed_json("prefix:{\"x\":NaN}\n", "prefix:", "fixture")

    def test_eps_fit_requires_complete_thresholded_metrics(self) -> None:
        target = pathlib.Path("/target.py")
        stream = HOST.BoundedText("eps")

        def run_path(*_args, **_kwargs):
            print(HOST.METRIC_PREFIX + json.dumps(eps_metrics()))
            return {}

        with mock.patch.object(HOST.runpy, "run_path", side_effect=run_path):
            self.assertEqual(HOST.run_eps_fit(target, stream), eps_metrics())

        failed = eps_metrics()
        failed["gradient_relative_error"] = 1e-3
        stream = HOST.BoundedText("eps")

        def failed_path(*_args, **_kwargs):
            print(HOST.METRIC_PREFIX + json.dumps(failed))
            return {}

        with mock.patch.object(
            HOST.runpy, "run_path", side_effect=failed_path
        ), self.assertRaisesRegex(Exception, "thresholds failed"):
            HOST.run_eps_fit(target, stream)

    def test_mpb_suite_requires_47_tests_and_zero_skip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = pathlib.Path(raw) / "test_fixture.py"
            methods = "\n".join(
                f"    def test_{index:02d}(self): self.assertTrue(True)"
                for index in range(47)
            )
            target.write_text("import unittest\nclass T(unittest.TestCase):\n" + methods + "\n")
            result = HOST.run_mpb_suite(target, HOST.BoundedText("mpb"))
            self.assertEqual(result["tests_run"], 47)
            self.assertEqual(result["skipped"], 0)

    def test_parallel_waveguide_oracle_requires_full_distinct_arrays(self) -> None:
        separation = np.arange(0.025, 1.075, 0.05)
        namespace = {
            "ss": separation,
            "f_odd": np.linspace(0.2, 0.3, 21),
            "vg_odd": np.linspace(0.4, 0.5, 21),
            "f_even": np.linspace(0.21, 0.33, 21),
            "vg_even": np.linspace(0.45, 0.56, 21),
            "force_odd": np.linspace(-0.5, -0.1, 20),
            "force_even": np.linspace(0.1, 0.4, 20),
        }
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            HOST.runpy, "run_path", return_value=namespace
        ):
            result = HOST.run_parallel_waveguides(
                pathlib.Path("/target.py"),
                pathlib.Path(raw),
                HOST.BoundedText("parallel"),
            )
        self.assertEqual(result["separation"]["shape"], [21])
        self.assertEqual(result["force_even"]["shape"], [20])

        broken = dict(namespace)
        broken["f_even"] = broken["f_odd"].copy()
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            HOST.runpy, "run_path", return_value=broken
        ), self.assertRaisesRegex(Exception, "finite-array oracle failed"):
            HOST.run_parallel_waveguides(
                pathlib.Path("/target.py"),
                pathlib.Path(raw),
                HOST.BoundedText("parallel"),
            )

    def test_result_array_rejects_missing_imaginary_evidence(self) -> None:
        record = {
            "count": 2,
            "real": [1.0, 2.0],
            "imag": [0.0, 0.0],
            "l2": 1.0,
            "max_abs": 2.0,
            "weighted_real": 5.0,
            "weighted_imag": 0.0,
            "shape": [1, 2],
        }
        metrics = {"example_results": {"value": record}}
        np.testing.assert_array_equal(
            HOST._result_array(metrics, "value", (1, 2)), [[1.0, 2.0]]
        )
        del record["imag"]
        with self.assertRaisesRegex(Exception, "shape differs"):
            HOST._result_array(metrics, "value", (1, 2))

    def test_plotter_reconstructs_npz_and_requires_two_pngs(self) -> None:
        radial = np.linspace(0.1, 1.0, 56).reshape(7, 8)
        result_record = {
            "count": 56,
            "real": radial.ravel().tolist(),
            "imag": [0.0] * 56,
            "l2": float(np.linalg.norm(radial)),
            "max_abs": 1.0,
            "weighted_real": 1.0,
            "weighted_imag": 0.0,
            "shape": [7, 8],
        }
        metrics = {"example_results": {"dipole_1d_radial_flux": result_record}}
        complete = {
            "archive_manifest_sha256": "a" * 64,
            "build_receipt_id": "receipt",
        }

        def render(*_args, **_kwargs):
            print("relative error in radiation pattern (φ = 0):, 0.25")
            payload = b"\x89PNG\r\n\x1a\n" + b"x" * 2048
            pathlib.Path("dipole_radiation_pattern_phi0.png").write_bytes(payload)
            pathlib.Path("dipole_radiation_pattern_3D.png").write_bytes(payload)
            return {}

        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            HOST, "_producer_metrics", return_value=(metrics, complete)
        ), mock.patch.object(
            HOST.runpy, "run_path", side_effect=render
        ), mock.patch.object(
            HOST.archive_validation, "verify_archive", return_value=complete
        ):
            output = pathlib.Path(raw)
            result = HOST.run_plotter(
                pathlib.Path("/target.py"),
                output,
                HOST.BoundedText("plot"),
                pathlib.Path("/archive"),
                "a" * 64,
                "receipt",
            )
            with np.load(output / "dipole_radiation_pattern.npz") as data:
                np.testing.assert_allclose(data["radial_flux"], radial)
        self.assertEqual(result["relative_error"], 0.25)
        self.assertEqual(len(result["images"]), 2)

    def test_main_replays_pending_before_host_complete(self) -> None:
        metrics = eps_metrics()

        def run_eps(_target, stream):
            stream.write(HOST.METRIC_PREFIX + json.dumps(metrics) + "\n")
            return metrics

        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            HOST, "run_eps_fit", side_effect=run_eps
        ), mock.patch.object(
            HOST, "validate_complete", wraps=HOST.validate_complete
        ) as verifier:
            output = pathlib.Path(raw) / "evidence"
            self.assertEqual(
                HOST.main(
                    [
                        "--repo",
                        str(ROOT),
                        "--output",
                        str(output),
                        "--unit-id",
                        "host-eps-fit-lorentzian",
                    ]
                ),
                0,
            )
            self.assertTrue((output / "COMPLETE").is_file())
            self.assertFalse((output / "PENDING_COMPLETE").exists())
            verifier.assert_called_once()
            self.assertEqual(
                verifier.call_args.kwargs["terminal_name"], "PENDING_COMPLETE"
            )

    def test_parallel_complete_rederives_full_array_records(self) -> None:
        separation = np.arange(0.025, 1.075, 0.05)
        arrays = {
            "separation": separation,
            "f_odd": np.linspace(0.2, 0.3, 21),
            "vg_odd": np.linspace(0.4, 0.5, 21),
            "f_even": np.linspace(0.21, 0.33, 21),
            "vg_even": np.linspace(0.45, 0.56, 21),
            "force_odd": np.linspace(-0.5, -0.1, 20),
            "force_even": np.linspace(0.1, 0.4, 20),
        }
        result = {name: HOST._array_record(value) for name, value in arrays.items()}
        unit_id = "host-parallel-wvgs-mpb"
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            (output / "stdout.log").write_text("fixture\n")
            target = ROOT / HOST.ORACLES[unit_id]["target"]
            report = {
                "schema": HOST.SCHEMA,
                "outcome": "PASS",
                "unit_id": unit_id,
                "driver": HOST.ORACLES[unit_id]["driver"],
                "target": HOST.file_record(target, ROOT),
                "adapter": HOST.file_record(SOURCE, ROOT),
                "result": result,
                "stdout": HOST.file_record(output / "stdout.log", output),
            }
            HOST.atomic_write_json(output / "report.json", report)
            HOST.atomic_write_json(
                output / "COMPLETE",
                {
                    "schema": HOST.COMPLETE_SCHEMA,
                    "outcome": "PASS",
                    "report": HOST.file_record(output / "report.json", output),
                },
            )
            validated = HOST.validate_complete(output, ROOT, unit_id)
            self.assertEqual(validated["result"], result)

            report["result"]["f_odd"]["values"][0] += 0.1
            HOST.atomic_write_json(output / "report.json", report)
            HOST.atomic_write_json(
                output / "COMPLETE",
                {
                    "schema": HOST.COMPLETE_SCHEMA,
                    "outcome": "PASS",
                    "report": HOST.file_record(output / "report.json", output),
                },
            )
            with self.assertRaisesRegex(Exception, "not re-derived"):
                HOST.validate_complete(output, ROOT, unit_id)


if __name__ == "__main__":
    unittest.main()
