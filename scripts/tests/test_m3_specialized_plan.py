#!/usr/bin/env python3
"""Tests for the frozen M3 distributed-feature plan loader."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "m3_specialized_plan.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_specialized_plan", SOURCE)
assert SPEC is not None and SPEC.loader is not None
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)


def rows() -> bytes:
    lines = ["\t".join(PLAN.FIELDS)]
    for index, (unit_id, driver, target) in enumerate(PLAN.EXPECTED_ROWS, 1):
        lines.append(
            "\t".join(
                (
                    str(index),
                    unit_id,
                    driver,
                    target,
                    f"coverage-{index}",
                    PLAN.DRIVER_TOPOLOGIES[driver],
                    f"complete numeric oracle {index}",
                    "1800",
                )
            )
        )
    return ("\n".join(lines) + "\n").encode()


class M3SpecializedPlanTests(unittest.TestCase):
    def test_exact_five_unit_inventory(self) -> None:
        units = PLAN.parse_plan(rows())
        self.assertEqual(len(units), 5)
        self.assertEqual(
            [unit.driver for unit in units].count("python-mpi-example-case"), 4
        )
        self.assertEqual(units[-1].timeout_seconds, 1800)

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

    def test_identity_topology_and_order_mutations_fail(self) -> None:
        mutations = (
            (b"specialized-mpi-python-core", b"specialized-other", "identity"),
            (b"cpu2,cuda2-pinned,cuda2-aware", b"cpu8,cuda1,cuda2", "topology"),
            (b"\n2\tspecialized-differential", b"\n3\tspecialized-differential", "order"),
        )
        for old, new, pattern in mutations:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(Exception, pattern):
                    PLAN.parse_plan(rows().replace(old, new, 1))

    def test_noncanonical_text_and_path_escape_fail(self) -> None:
        with self.assertRaisesRegex(Exception, "noncanonical text"):
            PLAN.parse_plan(rows().replace(b"\n", b"\r\n"))
        changed = rows().replace(
            b"scripts/mpi-python-validation-probe.py", b"../probe.py", 1
        )
        with self.assertRaisesRegex(Exception, "identity differs"):
            PLAN.parse_plan(changed)

    def test_timeout_and_duplicate_coverage_fail(self) -> None:
        with self.assertRaisesRegex(Exception, "outside"):
            PLAN.parse_plan(rows().replace(b"\t1800\n", b"\t60\n", 1))
        changed = rows().replace(b"coverage-2", b"coverage-1", 1)
        with self.assertRaisesRegex(Exception, "not unique"):
            PLAN.parse_plan(changed)


if __name__ == "__main__":
    unittest.main()
