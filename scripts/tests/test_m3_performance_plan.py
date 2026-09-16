#!/usr/bin/env python3
"""Tests for the frozen M3 release-performance plan loader."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/m3_performance_plan.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_performance_plan", SOURCE)
assert SPEC is not None and SPEC.loader is not None
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)


def rows() -> bytes:
    timings = (
        "controller-rederived-fixed-window-fdtd-wall",
        "controller-rederived-fixed-work-phase-wall",
        "sum-all-simulation-run-wall-seconds",
        "run-index-18-small-and-19-large-wall-seconds",
        "rederived-native-batch-lane-median",
        "rederived-forward-adjoint-trajectory-wall",
        "sum-all-simulation-run-wall-seconds",
    )
    lines = ["\t".join(PLAN.FIELDS)]
    for order, ((unit_id, driver, target, topologies), timing) in enumerate(
        zip(PLAN.EXPECTED_ROWS, timings, strict=True), 1
    ):
        warmups = "0" if driver.startswith("external-m") else "1"
        measured = "7" if unit_id == "performance-near2far-transform" else "5"
        lines.append(
            "\t".join(
                (
                    str(order),
                    unit_id,
                    driver,
                    target,
                    topologies,
                    warmups,
                    measured,
                    timing,
                    f"acceptance-{order}",
                    "1800",
                )
            )
        )
    return ("\n".join(lines) + "\n").encode()


class M3PerformancePlanTests(unittest.TestCase):
    def test_exact_inventory_and_repetition_contract(self) -> None:
        units = PLAN.parse_plan(rows())
        self.assertEqual(len(units), 7)
        self.assertEqual(units[4].measured_cycles, 7)
        self.assertTrue(
            all(
                unit.measured_cycles >= 5
                for unit in units
                if unit.unit_id != "performance-long-horizon-policy"
            )
        )
        self.assertEqual(
            units[-1].topologies, ("cpu8", "cpu1", "cuda1", "auto1")
        )

    def test_identity_topology_and_repetition_mutations_fail(self) -> None:
        mutations = (
            (b"performance-m1-ters", b"performance-other", "identity"),
            (b"cpu8,cuda1,cuda2", b"cpu8,cuda1", "identity"),
            (b"\t5\tcontroller-rederived", b"\t4\tcontroller-rederived", "repetition"),
        )
        for old, new, pattern in mutations:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(Exception, pattern):
                    PLAN.parse_plan(rows().replace(old, new, 1))

    def test_noncanonical_encoding_and_timeout_fail(self) -> None:
        with self.assertRaisesRegex(Exception, "noncanonical"):
            PLAN.parse_plan(rows().replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(Exception, "outside"):
            PLAN.parse_plan(rows().replace(b"\t1800\n", b"\t60\n", 1))

    def test_loaded_plan_is_json_round_trip_stable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            plan_root = pathlib.Path(raw).resolve()
            value = rows()
            (plan_root / PLAN.PLAN_NAME).write_bytes(value)
            with mock.patch.object(
                PLAN, "PLAN_SHA256", hashlib.sha256(value).hexdigest()
            ):
                plan = PLAN.load_release_plan(plan_root, ROOT)
        self.assertEqual(json.loads(json.dumps(plan)), plan)


if __name__ == "__main__":
    unittest.main()
