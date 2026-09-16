#!/usr/bin/env python3
"""Tests for the closed-world M3 feature evidence plan."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/m3_feature_plan.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_feature_plan", SOURCE)
assert SPEC is not None and SPEC.loader is not None
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)


def feature_ids() -> list[tuple[str, str]]:
    values = [
        (f"solver.feature_{index:02d}", "solver_physics") for index in range(19)
    ]
    values.extend(
        (f"monitor.feature_{index:02d}", "monitor_transform_adjoint")
        for index in range(13)
    )
    values.extend(
        (
            ("distributed.cuda_aware_mpi", "distributed_policy"),
            ("policy.long_horizon_small", "distributed_policy"),
        )
    )
    values.extend(
        (f"distributed.feature_{index:02d}", "distributed_policy")
        for index in range(8)
    )
    return values


def requirement_rows() -> bytes:
    lines = ["\t".join(PLAN.REQUIREMENT_FIELDS)]
    for index, (feature_id, family) in enumerate(feature_ids()):
        lines.append(
            "\t".join(
                (
                    feature_id,
                    family,
                    f"feature {index}",
                    f"oracle {index}",
                    f"required evidence {index}",
                    f"topology {index}",
                )
            )
        )
    return ("\n".join(lines) + "\n").encode()


def evidence_rows() -> bytes:
    lines = ["\t".join(PLAN.EVIDENCE_FIELDS)]
    for index, (feature_id, _family) in enumerate(feature_ids()):
        lines.append(
            "\t".join(
                (
                    feature_id,
                    f"python/examples/primary_{index}.py",
                    f"tests/native_{index}.cpp",
                    f"counter or gate {index}",
                    f"release topology {index}",
                    PLAN.STATUS_BY_FEATURE.get(feature_id, "planned"),
                )
            )
        )
    return ("\n".join(lines) + "\n").encode()


class M3FeaturePlanTests(unittest.TestCase):
    def test_exact_42_feature_partition_and_statuses(self) -> None:
        features = PLAN.parse_contract(requirement_rows(), evidence_rows())
        self.assertEqual(len(features), 42)
        self.assertEqual(
            features[32]["status"], "planned_capability_gated"
        )
        self.assertEqual(
            features[33]["status"], "planned_policy_gap_allowed"
        )
        self.assertEqual(
            features[0]["primary_current_receipt_evidence"],
            ["python/examples/primary_0.py"],
        )

    def test_id_order_family_and_status_mutations_fail(self) -> None:
        evidence = evidence_rows().replace(
            b"solver.feature_00", b"solver.feature_99", 1
        )
        with self.assertRaisesRegex(Exception, "ID inventory"):
            PLAN.parse_contract(requirement_rows(), evidence)
        requirements = requirement_rows().replace(
            b"solver_physics", b"distributed_policy", 1
        )
        with self.assertRaisesRegex(Exception, "family partition"):
            PLAN.parse_contract(requirements, evidence_rows())
        evidence = evidence_rows().replace(
            b"planned_capability_gated", b"planned", 1
        )
        with self.assertRaisesRegex(Exception, "row differs"):
            PLAN.parse_contract(requirement_rows(), evidence)

    def test_noncanonical_and_unsafe_evidence_fail(self) -> None:
        with self.assertRaisesRegex(Exception, "noncanonical"):
            PLAN.parse_contract(
                requirement_rows().replace(b"\n", b"\r\n"), evidence_rows()
            )
        evidence = evidence_rows().replace(
            b"python/examples/primary_0.py", b"../primary.py", 1
        )
        with self.assertRaisesRegex(Exception, "path differs"):
            PLAN.parse_contract(requirement_rows(), evidence)
        evidence = evidence_rows().replace(
            b"python/examples/primary_0.py",
            b"milestone:unsealed",
            1,
        )
        with self.assertRaisesRegex(Exception, "milestone differs"):
            PLAN.parse_contract(requirement_rows(), evidence)

    def test_execution_routes_include_all_matching_evidence_families(self) -> None:
        execution = {
            "selected": [{"case_path": "python/a.py", "unit_id": "paired-a"}],
            "stronger": [],
            "host": [
                {"covers_examples": ["python/h.py"], "unit_id": "host-h"}
            ],
        }
        specialized = {
            "units": [
                {
                    "target_path": "tests/gpu-step-db.cpp",
                    "driver": "run-mpi-python-validation",
                    "unit_id": "specialized-a",
                }
            ]
        }
        routes = PLAN._execution_routes(execution, specialized)
        self.assertEqual(
            routes["python/a.py"],
            [{"kind": "paired", "unit_id": "paired-a"}],
        )
        self.assertEqual(
            routes["scripts/run-mpi-python-validation.py"],
            [{"kind": "specialized", "unit_id": "specialized-a"}],
        )


if __name__ == "__main__":
    unittest.main()
