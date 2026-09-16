#!/usr/bin/env python3
"""Tests for M3 specialized raw-evidence replay routing."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "m3_specialized_replay.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_specialized_replay", SOURCE)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


class M3SpecializedReplayTests(unittest.TestCase):
    def unit(self, driver: str) -> dict[str, object]:
        return {
            "launch_order": 2,
            "unit_id": "specialized-case",
            "driver": driver,
            "target_path": "python/examples/case.py",
        }

    def test_task_paths_are_stable(self) -> None:
        unit = self.unit("python-mpi-example-case")
        output = pathlib.Path("/evidence")
        self.assertEqual(
            REPLAY.task_raw(output, unit),
            output / "raw/02-specialized-case",
        )
        self.assertEqual(
            REPLAY.task_attestation(output, unit),
            output / "attestations/02-specialized-case.json",
        )

    def test_driver_routing_and_receipt_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            (output / "raw/02-specialized-case").mkdir(parents=True)
            contract = {
                "paths": {"repo": str(output), "build_receipt": str(output / "receipt")},
                "build_receipt": {"receipt_id": "r" * 64},
            }
            (output / "receipt").write_text("{}")
            expected = {
                "driver": "python-mpi-example-case",
                "case_path": "python/examples/case.py",
                "receipt_id": "r" * 64,
            }
            with mock.patch.object(REPLAY, "_python_example", return_value=expected):
                value = REPLAY.verify_task(
                    self.unit("python-mpi-example-case"), output, contract
                )
            self.assertEqual(value["schema"], REPLAY.SCHEMA)
            self.assertEqual(value["evidence"], expected)

            changed = dict(expected, receipt_id="x" * 64)
            with mock.patch.object(REPLAY, "_python_example", return_value=changed):
                with self.assertRaisesRegex(Exception, "receipt identity differs"):
                    REPLAY.verify_task(
                        self.unit("python-mpi-example-case"), output, contract
                    )

    def test_unknown_driver_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw)
            (output / "raw/02-specialized-case").mkdir(parents=True)
            receipt = output / "receipt"
            receipt.write_text("{}")
            contract = {
                "paths": {"repo": str(output), "build_receipt": str(receipt)},
                "build_receipt": {"receipt_id": "r" * 64},
            }
            with self.assertRaisesRegex(Exception, "unsupported"):
                REPLAY.verify_task(self.unit("unknown"), output, contract)


if __name__ == "__main__":
    unittest.main()
