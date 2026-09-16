from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKLOADS = ROOT / "scripts/user-workloads"
sys.path.insert(0, str(WORKLOADS))
SPEC = importlib.util.spec_from_file_location(
    "gpmeep_test_installed_aunp_matrix",
    WORKLOADS / "run_installed_aunp_performance_matrix.py",
)
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
assert SPEC.loader is not None
SPEC.loader.exec_module(MATRIX)


def tasks(
    *,
    cpu: tuple[float, float] = (164.0, 178.0),
    gpu1: tuple[float, float] = (16.7, 41.0),
    gpu2: tuple[float, float] = (11.0, 30.0),
):
    values = {"cpu8": cpu, "gpu1": gpu1, "gpu2": gpu2}
    result = []
    for repeat in range(MATRIX.REPEATS):
        for lane in MATRIX.LANES:
            fdtd, end_to_end = values[lane["role"]]
            result.append(
                {
                    "repeat": repeat,
                    "role": lane["role"],
                    "fdtd_wall_seconds": fdtd * (1.0 + repeat * 0.01),
                    "workload_end_to_end_seconds": end_to_end * (1.0 + repeat * 0.01),
                }
            )
    return result


class InstalledAuNPPerformanceMatrixTests(unittest.TestCase):
    def test_anchor_is_the_accepted_adc67c3_terminal(self):
        anchor = MATRIX.SEALED_M3_ANCHOR
        self.assertEqual(
            anchor["source_commit"],
            "adc67c3466774bbdb1875d996e4e0e50aeb7a845",
        )
        self.assertTrue(anchor["qualification_eligible"])
        self.assertEqual(
            anchor["report_sha256"],
            "50d2d8539000c76fb554318e2aa641577342e0f069ea5078d6f5ce596c5c36dc",
        )
        self.assertEqual(
            anchor["means_seconds"]["gpu2"]["fdtd"],
            10.951151260989718,
        )

    def test_two_sample_matrix_rederives_speedups_and_anchor_gate(self):
        derived = MATRIX.derive_performance(tasks())
        self.assertGreater(
            derived["comparisons"]["cpu8-to-gpu2"]["fdtd"]["mean_speedup"],
            14.0,
        )
        self.assertEqual(
            derived["sealed_anchor_replay"]["gpu1"]["fdtd"]["outcome"],
            "PASS",
        )

    def test_inventory_speed_and_regression_fail_closed(self):
        with self.assertRaisesRegex(MATRIX.MatrixError, "order or inventory"):
            MATRIX.derive_performance(tasks()[:-1])
        with self.assertRaisesRegex(MATRIX.MatrixError, "below"):
            MATRIX.derive_performance(tasks(gpu1=(120.0, 140.0)))
        with self.assertRaisesRegex(MATRIX.MatrixError, "slowdown"):
            MATRIX.derive_performance(
                tasks(cpu=(210.0, 230.0), gpu1=(20.0, 48.0), gpu2=(13.0, 35.0))
            )

    def test_cli_has_no_repeat_or_physics_override(self):
        with self.assertRaises(SystemExit):
            MATRIX.parse_args(
                [
                    "run",
                    "--archive",
                    "archive",
                    "--output",
                    "/tmp/output",
                    "--prefix",
                    "/tmp/prefix",
                    "--package-provenance",
                    "report",
                    "--source-commit",
                    "1" * 40,
                    "--package-sha256",
                    "2" * 64,
                    "--repeats",
                    "1",
                ]
            )

    def test_identity_is_lowercase_hex_not_only_the_expected_length(self):
        self.assertTrue(MATRIX._lower_hex("a" * 40, 40))
        self.assertFalse(MATRIX._lower_hex("g" * 40, 40))
        self.assertFalse(MATRIX._lower_hex("A" * 40, 40))

    def test_relative_record_is_bound_to_the_expected_path(self):
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            expected = root / "expected.txt"
            alternate = root / "alternate.txt"
            expected.write_text("same\n", encoding="utf-8")
            alternate.write_text("same\n", encoding="utf-8")
            record = MATRIX.common.file_record(expected, root)
            MATRIX._verified_relative_record(
                root, expected, record, "expected.txt", "test"
            )
            record["path"] = "alternate.txt"
            with self.assertRaisesRegex(MATRIX.MatrixError, "path differs"):
                MATRIX._verified_relative_record(
                    root, expected, record, "expected.txt", "test"
                )

    def test_validator_rejects_duplicate_keys_in_complete_and_report(self):
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw).resolve()
            report_path = output / "report.json"
            complete_path = output / "COMPLETE"

            complete_path.write_text(
                '{"schema":"x","schema":"x","outcome":"PASS","report":{}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MATRIX.MatrixError, "invalid JSON"):
                MATRIX.validate(output)

            report_path.write_text('{"schema":"x","schema":"x"}', encoding="utf-8")
            complete_path.write_text(
                json.dumps(
                    {
                        "schema": MATRIX.COMPLETE_SCHEMA,
                        "outcome": "PASS",
                        "report": MATRIX.common.file_record(report_path, output),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MATRIX.MatrixError, "invalid JSON"):
                MATRIX.validate(output)


if __name__ == "__main__":
    unittest.main()
