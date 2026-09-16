from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "scripts/python-validation/v1_feature_coverage.py"
MANIFEST_PATH = REPO / "scripts/python-validation/v1_feature_coverage.json"
RUNNER_MANIFEST = REPO / "scripts/python-validation/manifest.json"

SPEC = importlib.util.spec_from_file_location("v1_feature_coverage", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
COVERAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COVERAGE)


class TestV1FeatureCoverage(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def validate(self, plan=None, *, evidence_root=None):
        return COVERAGE.validate_plan(
            self.plan if plan is None else plan,
            repo=REPO,
            runner_manifest=RUNNER_MANIFEST,
            evidence_root=evidence_root,
        )

    def route(self, plan, route_id):
        return next(route for route in plan["routes"] if route["id"] == route_id)

    def feature(self, plan, feature_id):
        return next(
            feature for feature in plan["features"] if feature["id"] == feature_id
        )

    def test_checked_in_contract_is_exact_and_closed(self) -> None:
        report = self.validate()
        self.assertEqual(report["outcome"], "PASS")
        self.assertEqual(report["feature_count"], 42)
        self.assertEqual(report["route_count"], 32)
        self.assertEqual(
            report["route_counts_by_tier"], {"0": 1, "1": 8, "2": 18, "3": 5}
        )
        self.assertEqual(report["uncovered_features"], [])
        self.assertEqual(report["policy_counts"]["ordinary"], 40)

    def test_missing_feature_is_rejected(self) -> None:
        changed = copy.deepcopy(self.plan)
        changed["features"].pop()
        with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "exactly 42"):
            self.validate(changed)

    def test_uncovered_feature_is_rejected(self) -> None:
        changed = copy.deepcopy(self.plan)
        straight = self.route(changed, "case:straight-waveguide")
        straight["features"].remove("solver.real_cartesian")
        with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "real_cartesian.*requires"):
            self.validate(changed)

    def test_minimum_distinct_route_count_is_rejected(self) -> None:
        changed = copy.deepcopy(self.plan)
        self.route(changed, "case:cyl-off-axis")["features"].remove(
            "solver.dimensions"
        )
        self.route(changed, "case:cyl-on-axis")["features"].remove(
            "solver.dimensions"
        )
        with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "dimensions.*requires 4"):
            self.validate(changed)

    def test_unknown_feature_assignment_is_rejected(self) -> None:
        changed = copy.deepcopy(self.plan)
        self.route(changed, "case:straight-waveguide")["features"].append(
            "solver.not_real"
        )
        with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "unknown values"):
            self.validate(changed)

    def test_nonregistered_runner_case_is_rejected(self) -> None:
        changed = copy.deepcopy(self.plan)
        self.route(changed, "case:straight-waveguide")["case_id"] = (
            "python/examples/not-a-real-example.py"
        )
        with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "absent"):
            self.validate(changed)

    def test_gpu2_feature_without_gpu2_route_is_rejected(self) -> None:
        changed = copy.deepcopy(self.plan)
        route = self.route(changed, "specialized:mpi-python")
        route["topologies"].remove("gpu2")
        with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "lacks gpu2"):
            self.validate(changed)

    def test_adjoint_route_without_directional_derivative_is_rejected(self) -> None:
        changed = copy.deepcopy(self.plan)
        route = self.route(changed, "case:mode-converter")
        route["evidence"].remove("directional_derivative")
        with self.assertRaisesRegex(
            COVERAGE.CoveragePlanError, "adjoint_cartesian.*directional"
        ):
            self.validate(changed)

    def test_auto_policy_without_auto_topology_is_rejected(self) -> None:
        changed = copy.deepcopy(self.plan)
        route = self.route(changed, "case:antenna-pec-1d")
        route["topologies"].remove("auto")
        with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "auto_small.*auto"):
            self.validate(changed)

    def test_special_policy_cannot_be_silently_promoted(self) -> None:
        changed = copy.deepcopy(self.plan)
        self.feature(changed, "distributed.cuda_aware_mpi")["policy"] = "ordinary"
        with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "contract differs"):
            self.validate(changed)

    def test_route_deduplication_cap_is_enforced(self) -> None:
        changed = copy.deepcopy(self.plan)
        changed["maximum_route_count"] = 31
        with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "deduplication cap"):
            self.validate(changed)

    def test_route_mapping_cannot_change_without_frozen_digest_update(self) -> None:
        changed = copy.deepcopy(self.plan)
        route = self.route(changed, "case:straight-waveguide")
        route["features"].remove("monitor.dft_accumulation")
        # The feature still has its declared three routes only if another route is
        # substituted.  This mutation keeps the broad union closed but changes the
        # reviewed route semantics and must therefore fail its frozen digest.
        self.route(changed, "case:refl-quartz")["features"].append(
            "monitor.dft_accumulation"
        )
        with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "route contract digest"):
            self.validate(changed)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "duplicate.json"
            path.write_text('{"schema": 1, "schema": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "duplicate JSON key"):
                COVERAGE._load_json(path)

    def test_anchor_hash_mismatch_is_rejected_when_evidence_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            terminal = root / "current-scope-aunp-tm-adc67c3"
            terminal.mkdir()
            (terminal / "COMPLETE").write_text(
                json.dumps(
                    {
                        "outcome": "PASS",
                        "report": {"sha256": "0" * 64},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (terminal / "report.json").write_text(
                '{"outcome": "PASS"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(COVERAGE.CoveragePlanError, "COMPLETE SHA"):
                self.validate(evidence_root=root)


if __name__ == "__main__":
    unittest.main()
