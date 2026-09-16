#!/usr/bin/env python3
"""Tests for the strict M3 execution-plan loader."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "python-validation" / "m3_execution_plan.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_execution_plan", SOURCE)
assert SPEC is not None and SPEC.loader is not None
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)


def selected_rows(count: int = 11) -> bytes:
    lines = ["\t".join(PLAN.SELECTED_FIELDS)]
    for order in range(1, count + 1):
        role = "fail_fast_canary" if order <= 10 else "full_matrix"
        lines.append(
            "\t".join(
                (
                    str(order),
                    f"unit-{order}",
                    "cluster-a",
                    f"python/examples/case-{order}.py",
                    "cpu,cuda",
                    f"purpose {order}",
                    role,
                )
            )
        )
    return ("\n".join(lines) + "\n").encode()


def stronger_rows() -> bytes:
    return (
        "\t".join(PLAN.STRONGER_FIELDS)
        + "\n1\tstronger-one\tpython/examples/covered.py\t"
        "python/tests/test_case.py\tcpu,cuda\tpython-validation-exact-case\t"
        "numeric oracle\t1800\n"
    ).encode()


def host_rows() -> bytes:
    return (
        "\t".join(PLAN.HOST_FIELDS)
        + "\n1\thost-one\tpython-unittest-file\tpython/tests/test_mpb.py\t"
        "cpu\tpython/examples/mpb_a.py;python/examples/mpb_b.py\t"
        "numeric oracle\t7200\n"
    ).encode()


class M3ExecutionPlanTests(unittest.TestCase):
    def test_selected_parser_preserves_canary_boundary(self) -> None:
        units = PLAN.parse_selected(selected_rows())
        self.assertEqual(len(units), 11)
        self.assertEqual(units[9].release_role, "fail_fast_canary")
        self.assertEqual(units[10].release_role, "full_matrix")

    def test_selected_parser_rejects_backend_and_order_mutation(self) -> None:
        changed = selected_rows().replace(b"cpu,cuda", b"cuda", 1)
        with self.assertRaisesRegex(Exception, "backend topology differs"):
            PLAN.parse_selected(changed)
        changed = selected_rows().replace(b"\n2\tunit-2", b"\n3\tunit-2", 1)
        with self.assertRaisesRegex(Exception, "order is not contiguous"):
            PLAN.parse_selected(changed)

    def test_selected_parser_rejects_canary_downgrade(self) -> None:
        changed = selected_rows().replace(b"fail_fast_canary", b"full_matrix", 1)
        with self.assertRaisesRegex(Exception, "canary partition differs"):
            PLAN.parse_selected(changed)

    def test_header_extra_field_and_noncanonical_text_are_rejected(self) -> None:
        changed = selected_rows().replace(b"release_role\n", b"release_role\textra\n")
        with self.assertRaisesRegex(Exception, "noncanonical text"):
            PLAN.parse_selected(changed)
        with self.assertRaisesRegex(Exception, "noncanonical text"):
            PLAN.parse_selected(selected_rows().replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(Exception, "noncanonical text"):
            PLAN.parse_selected(selected_rows()[:-1])
        changed = selected_rows().replace(b"purpose 1", b'"purpose\n1"', 1)
        with self.assertRaisesRegex(Exception, "noncanonical text"):
            PLAN.parse_selected(changed)

    def test_path_traversal_and_unsafe_unit_id_are_rejected(self) -> None:
        changed = selected_rows().replace(
            b"python/examples/case-1.py", b"../case-1.py", 1
        )
        with self.assertRaisesRegex(Exception, "safe repository-relative path"):
            PLAN.parse_selected(changed)
        changed = selected_rows().replace(b"unit-1", b"Unit_1", 1)
        with self.assertRaisesRegex(Exception, "safe unit ID"):
            PLAN.parse_selected(changed)

    def test_stronger_parser_is_exact(self) -> None:
        units = PLAN.parse_stronger(stronger_rows())
        self.assertEqual(units[0].timeout_seconds, 1800)
        changed = stronger_rows().replace(
            b"python-validation-exact-case", b"python-unittest-file"
        )
        with self.assertRaisesRegex(Exception, "driver differs"):
            PLAN.parse_stronger(changed)

    def test_host_parser_is_exact_and_splits_coverage(self) -> None:
        units = PLAN.parse_host(host_rows())
        self.assertEqual(
            units[0].covers_examples,
            ("python/examples/mpb_a.py", "python/examples/mpb_b.py"),
        )
        changed = host_rows().replace(b"\tcpu\t", b"\tcuda\t")
        with self.assertRaisesRegex(Exception, "execution policy differs"):
            PLAN.parse_host(changed)


if __name__ == "__main__":
    unittest.main()
