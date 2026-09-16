#!/usr/bin/env python3
"""Contract tests for the user-revised release aggregator."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/run_current_scope_release.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_current_scope_release", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RELEASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RELEASE
SPEC.loader.exec_module(RELEASE)

GPU0 = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GPU1 = "GPU-11111111-2222-3333-4444-555555555555"


def performance_report() -> dict:
    roles = {
        "cpu-fp32-to-one-gpu": ("cpu8", "gpu1", [4.0, 6.0], [2.0, 3.0], 1.5),
        "cpu-fp32-to-two-gpu": ("cpu8", "gpu2", [4.0, 6.0], [1.5, 2.5], 2.0),
        "one-gpu-to-two-gpu": ("gpu1", "gpu2", [2.0, 3.0], [1.5, 2.5], 1.1),
    }
    timing = {}
    for name, (reference_role, candidate_role, reference, candidate, minimum) in roles.items():
        reference_mean = sum(reference) / 2
        candidate_mean = sum(candidate) / 2
        timing[name] = {
            "reference_role": reference_role,
            "candidate_role": candidate_role,
            "reference_samples_seconds": reference,
            "candidate_samples_seconds": candidate,
            "reference_mean_seconds": reference_mean,
            "candidate_mean_seconds": candidate_mean,
            "mean_speedup": reference_mean / candidate_mean,
            "minimum": minimum,
            "outcome": "PASS",
        }
    diagnostics = {
        role: {
            "sample_count": 2,
            "samples_seconds": samples,
            "arithmetic_mean_seconds": sum(samples) / 2,
            "population_stddev_seconds": 1.0,
            "coefficient_of_variation": 0.2,
            "release_gate": None,
        }
        for role, samples in {
            "performance-cpu-fp32-8r": [4.0, 6.0],
            "performance-cuda-fp32-1g": [2.0, 3.0],
            "performance-cuda-fp32-2g": [1.5, 2.5],
        }.items()
    }
    return {
        "outcome": "PASS",
        "performance_repeats": 2,
        "performance": {
            "fdtd_wall_seconds": timing,
            "workload_end_to_end_seconds": timing,
        },
        "timing_diagnostics": {
            "fdtd_wall_seconds": diagnostics,
            "workload_end_to_end_seconds": diagnostics,
        },
    }


def component_report(kind: str, count: int, receipt: str, source: str) -> dict:
    contract = {
        "build_receipt": {"receipt_id": receipt, "source_sha256": source},
        "gpu_devices": [GPU0, GPU1],
        "gpu_device": GPU0,
    }
    return {
        "outcome": "PASS",
        "contract": contract,
        "tasks": [
            {"task": {"unit_id": f"{kind}-{index:02d}"}}
            for index in range(1, count + 1)
        ],
    }


class CurrentScopeReleaseTests(unittest.TestCase):
    def test_aunp_performance_is_exactly_two_run_arithmetic_mean(self) -> None:
        value = performance_report()
        result = RELEASE.validate_aunp_performance(value)
        self.assertEqual(result["measured_runs_per_topology"], 2)
        self.assertEqual(result["aggregation"], "arithmetic_mean")
        self.assertEqual(result["dispersion_role"], "diagnostic_only")

        value["performance"]["fdtd_wall_seconds"]["cpu-fp32-to-one-gpu"][
            "reference_samples_seconds"
        ].append(8.0)
        with self.assertRaisesRegex(Exception, "exactly two"):
            RELEASE.validate_aunp_performance(value)

    def test_median_or_conservative_speedup_cannot_enter_release_metric(self) -> None:
        value = performance_report()
        metric = value["performance"]["fdtd_wall_seconds"][
            "cpu-fp32-to-one-gpu"
        ]
        metric["median_speedup"] = metric["mean_speedup"]
        with self.assertRaisesRegex(Exception, "schema differs"):
            RELEASE.validate_aunp_performance(value)

    def test_feature_closure_requires_a_passing_route_for_every_feature(self) -> None:
        features = []
        for index in range(42):
            features.append(
                {
                    "feature_id": f"family.feature_{index}",
                    "family": "family",
                    "status": "planned",
                    "primary_routes": [
                        {
                            "evidence": f"python/example-{index}.py",
                            "routes": [
                                {"kind": "paired", "unit_id": "paired-01"}
                            ],
                        }
                    ],
                }
            )
        closure = RELEASE.feature_closure(
            {"features": features}, {"paired": {"paired-01"}}
        )
        self.assertEqual(len(closure), 42)
        features[-1]["primary_routes"][0]["routes"][0]["unit_id"] = "missing"
        with self.assertRaisesRegex(Exception, "did not pass"):
            RELEASE.feature_closure(
                {"features": features}, {"paired": {"paired-01"}}
            )

    def test_build_closure_requires_identical_receipt_source_and_gpu_identity(self) -> None:
        source = "1" * 64
        aunp = {
            "build": {"receipt_id": "receipt"},
            "source_snapshot_sha256": source,
            "gpu_devices": [GPU0, GPU1],
            "cpu_physical_cores": 8,
        }
        components = {
            name: component_report(name, count, "receipt", source)
            for name, count in RELEASE.EXPECTED_COMPONENT_COUNTS.items()
        }
        closure = RELEASE._build_hardware_closure(aunp, components)
        self.assertTrue(closure["shared_identical_fp32_build"])
        components["specialized"]["contract"]["gpu_devices"].reverse()
        with self.assertRaisesRegex(Exception, "GPU identity"):
            RELEASE._build_hardware_closure(aunp, components)

    def test_derive_has_only_aunp_performance_and_79_example_units(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw).resolve()
            repo = root / "repo"
            repo.mkdir()
            roots = {}
            for name in ("aunp", "paired", "host", "specialized"):
                roots[name] = root / name
                roots[name].mkdir()
            plan_receipt = root / "plan.json"
            plan_receipt.write_text("{}\n")
            audit_receipt = root / "audit.json"
            audit_receipt.write_text("{}\n")
            deferral = root / "deferral.json"
            deferral.write_text("{}\n")
            source = "1" * 64
            aunp = performance_report()
            aunp.update(
                {
                    "build": {"receipt_id": "receipt"},
                    "source_snapshot_sha256": source,
                    "gpu_devices": [GPU0, GPU1],
                    "cpu_physical_cores": 8,
                    "exact_comparisons": [{}, {}],
                }
            )
            components = {
                name: component_report(name, count, "receipt", source)
                for name, count in RELEASE.EXPECTED_COMPONENT_COUNTS.items()
            }
            first_unit = components["paired"]["tasks"][0]["task"]["unit_id"]
            features = [
                {
                    "feature_id": f"family.feature_{index}",
                    "family": "family",
                    "status": "planned",
                    "primary_routes": [
                        {
                            "evidence": f"python/example-{index}.py",
                            "routes": [{"kind": "paired", "unit_id": first_unit}],
                        }
                    ],
                }
                for index in range(42)
            ]
            plan = {
                "repository": str(repo),
                "git_commit": "a" * 40,
                "git_status_porcelain": "",
                "plan": {"features": features},
            }
            contract = {
                "repo": str(repo),
                "plan_replay": str(plan_receipt),
                "hor_deferral": str(deferral),
                "aunp_output": str(roots["aunp"]),
                "aunp_audit": str(audit_receipt),
                "paired_output": str(roots["paired"]),
                "host_output": str(roots["host"]),
                "specialized_output": str(roots["specialized"]),
            }
            with mock.patch.object(
                RELEASE.plan_replay_seal, "verify", return_value=plan
            ), mock.patch.object(
                RELEASE, "validate_hor_deferral", return_value={"outcome": "deferred"}
            ), mock.patch.object(
                RELEASE.aunp_matrix, "verify_hybrid_complete", return_value=aunp
            ), mock.patch.object(
                RELEASE.paired_matrix, "verify_complete", return_value=components["paired"]
            ), mock.patch.object(
                RELEASE.host_matrix, "verify_complete", return_value=components["host"]
            ), mock.patch.object(
                RELEASE.specialized_matrix,
                "verify_complete",
                return_value=components["specialized"],
            ), mock.patch.object(
                RELEASE, "_audit_summary", return_value={"outcome": "PASS"}
            ), mock.patch.object(
                RELEASE, "_component_terminal", return_value={"path": "terminal"}
            ):
                report = RELEASE.derive(contract)
            self.assertEqual(report["official_meep_examples"]["total_physical_units"], 79)
            self.assertFalse(
                report["official_meep_examples"]["performance_release_gate"]
            )
            self.assertEqual(set(report["aunp"]["performance"]["metrics"]), RELEASE.TIMING_FIELDS)


if __name__ == "__main__":
    unittest.main()
