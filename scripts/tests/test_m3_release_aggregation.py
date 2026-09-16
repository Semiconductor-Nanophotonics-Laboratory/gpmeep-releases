#!/usr/bin/env python3
"""Focused fail-closed tests for the final M1--M3 evidence aggregator."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/python-validation/run_m3_release_aggregation.py"
SPEC = importlib.util.spec_from_file_location("gpmeep_m3_release_aggregation", SOURCE)
assert SPEC is not None and SPEC.loader is not None
AGGREGATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AGGREGATE
SPEC.loader.exec_module(AGGREGATE)


GPU0 = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GPU1 = "GPU-11111111-2222-3333-4444-555555555555"
RECEIPT = "a" * 64
CPU_RECEIPT = "b" * 64
SOURCE_SHA = "c" * 64
M1_RECEIPT = "d" * 64
M1_SOURCE_SHA = "f" * 64


def receipt() -> dict[str, str]:
    return {
        "receipt_id": RECEIPT,
        "source_sha256": SOURCE_SHA,
        "receipt_path": "/fixture/receipt.json",
    }


def source_identity(receipt_id: str, source_sha256: str) -> dict[str, object]:
    return {
        "schema": AGGREGATE.source_equivalence.SCHEMA,
        "policy": {
            "excluded_paths": sorted(
                AGGREGATE.source_equivalence.EXCLUDED_PATHS
            ),
            "meaning": (
                "all receipt-bound source files outside the exact reviewed "
                "M1-to-final validation-harness delta"
            ),
        },
        "receipt_id": receipt_id,
        "full_source_sha256": source_sha256,
        "retained_file_count": 100,
        "excluded_file_count": 10,
        "sha256": "e" * 64,
    }


def source_bridge() -> dict[str, object]:
    return {
        "schema": "gpmeep-m1-final-source-bridge-v1",
        "outcome": "PASS",
        "first": source_identity(M1_RECEIPT, M1_SOURCE_SHA),
        "second": source_identity(RECEIPT, SOURCE_SHA),
        "delta": {
            "added_paths": ["scripts/new.py"],
            "removed_paths": [],
            "modified_paths": [],
            "changed_path_count": 1,
            "allowlisted_path_count": len(
                AGGREGATE.source_equivalence.EXCLUDED_PATHS
            ),
            "unexpected_paths": [],
        },
        "retained_source": {"file_count": 100, "sha256": "e" * 64},
    }


def m2_comparison_fixture() -> dict[str, object]:
    return {
        "schema": "gpmeep-aunp-r4000-comparison-v2",
        "comparisons": [
            {"label": "packaged-upstream-expected", "outcome": "PASS"},
            {"label": "cuda-fp32-1g", "outcome": "PASS"},
        ],
        "outcome": "PASS",
    }


def paired_runner_fixture() -> dict[str, object]:
    return {
        "configuration": {
            "performance_evidence": {"valid_for_speed_gate": False}
        },
        "results": [
            {
                "selected": True,
                "path": "python/examples/ordinary.py",
                "outcome": "PASS",
                "runs": {
                    "cpu": {"duration_seconds": 2.0},
                    "cuda": {"duration_seconds": 1.0},
                },
                "comparison": {
                    "mode": "embedded_oracle",
                    "cpu_seconds": 2.0,
                    "cuda_seconds": 1.0,
                    "speedup_cpu_over_cuda": 2.0,
                    "outcome": "PASS",
                },
            }
        ],
    }


def fake_reports(repo: pathlib.Path, plan_root: pathlib.Path) -> dict[str, object]:
    paths = {"repo": str(repo), "plan_root": str(plan_root)}
    repeated = {
        "performance-edge-emitter-3d": {
            "statistics": {"outcome": "PASS"},
            "sample_count": 18,
            "numerical_comparisons": {
                "pair": {"outcome": "PASS", "failure_count": 0}
            },
        },
        "performance-metasurface-crossover": {
            "statistics": {"outcome": "PASS"},
            "sample_count": 30,
            "numerical_comparisons": {
                "pair": {"outcome": "PASS", "failure_count": 0}
            },
        },
        "performance-long-horizon-policy": {
            "statistics": {
                "outcome": "PASS",
                "automatic_policy": {
                    "forced-cuda-diagnostic": {
                        "missed_speedup_m4_gap": True
                    }
                },
            },
            "sample_count": 24,
            "numerical_comparisons": {
                "pair": {"outcome": "PASS", "failure_count": 0}
            },
        },
    }
    specialized_tasks = [
        {
            "task": {"unit_id": "specialized-mpi-python-core"},
            "task_replay": {
                "evidence": {
                    "driver": "run-mpi-python-validation",
                    "comparison_names": ["cpu_vs_cuda_aware"],
                }
            },
        },
        {
            "task": {"unit_id": "specialized-near2far-mpi-qualification"},
            "task_replay": {
                "evidence": {
                    "driver": "run-near2far-mpi-qualification",
                    "performance_scenarios": [
                        {"name": "single-point"},
                        {"name": "batch"},
                    ],
                }
            },
        },
        {
            "task": {"unit_id": "specialized-mpi-adjoint-qualification"},
            "task_replay": {
                "evidence": {
                    "driver": "run-mpi-adjoint-benchmark",
                    "headline_metric": "workload_wall",
                    "raw_samples": 15,
                    "cpu_over_cuda_single_median_speedup": 2.0,
                    "cpu_over_cuda_multi_median_speedup": 3.0,
                    "cuda_single_over_cuda_multi_median_speedup": 1.5,
                }
            },
        },
    ]
    return {
        "m1": {
            "outcome": "PASS",
            "source_snapshot_sha256": M1_SOURCE_SHA,
            "gpu_devices": [GPU0, GPU1],
            "tasks": [
                {
                    "task": {
                        "backend": "cpu",
                        "build_receipt_id": CPU_RECEIPT,
                    }
                },
                {
                    "task": {
                        "backend": "cuda",
                        "build_receipt_id": M1_RECEIPT,
                    }
                },
            ],
            "performance": {"gpu1": {"outcome": "PASS"}},
            "timing_stability": {"cpu8": {"outcome": "PASS"}},
            "exact_comparison": {
                "schema": "gpmeep-ters-comparison-v1",
                "comparisons": [
                    {
                        "label": label,
                        "comparison_class": "cuda-fp64-fp32",
                        "tolerance": {
                            "field_nrmse": 1.0e-3,
                            "field_peak_linf": 2.0e-3,
                            "field_energy_relative": 3.0e-3,
                            "field_peak_relative": 4.0e-3,
                            "phase_rms": 5.0e-3,
                        },
                        "fields": {
                            "Ez": {
                                "nrmse": 1.0e-5,
                                "peak_normalized_linf": 2.0e-5,
                                "energy_relative_error": 3.0e-5,
                                "peak_relative_error": 4.0e-5,
                                "masked_phase_rms_rad": 5.0e-5,
                                "outcome": "PASS",
                            }
                        },
                        "gap_enhancement": {"outcome": "PASS"},
                        "outcome": "PASS",
                    }
                    for label in ("cuda-fp32-1g", "cuda-fp32-2g")
                ],
                "outcome": "PASS",
            },
            "shared_fp32_source_identity": source_identity(
                M1_RECEIPT, M1_SOURCE_SHA
            ),
            "shared_fp32_receipt": {
                "path": "/fixture/m1-receipt.json",
                "size_bytes": 10,
                "sha256": "9" * 64,
            },
        },
        "m2": {
            "outcome": "PASS",
            "source_snapshot_sha256": SOURCE_SHA,
            "build": receipt(),
            "gpu_devices": [GPU0, GPU1],
            "tasks": [],
            "exact_comparisons": [
                {
                    "spec": {"name": "cpu-fp64-vs-cuda-fp32-1g"},
                    "report": {"path": "m2-comparison-1.json"},
                },
                {
                    "spec": {"name": "cuda-fp32-1g-vs-cuda-fp32-2g"},
                    "report": {"path": "m2-comparison-2.json"},
                },
            ],
            "performance": {"gpu1": {"outcome": "PASS"}},
            "timing_stability": {"cpu8": {"outcome": "PASS"}},
        },
        "paired": {
            "outcome": "PASS",
            "contract": {
                "paths": paths,
                "build_receipt": receipt(),
                "gpu_device": GPU0,
            },
            "tasks": [
                {
                    "task": {
                        "unit_id": (
                            "paired-ordinary"
                            if index == 0
                            else f"paired-fixture-{index:02d}"
                        ),
                        "case_path": "python/examples/ordinary.py",
                        "release_role": "full_matrix",
                    },
                    "archive": {
                        "runner_report": {
                            "path": f"paired-runner-{index:02d}.json"
                        }
                    },
                }
                for index in range(70)
            ],
        },
        "host": {
            "outcome": "PASS",
            "contract": {
                "repo": str(repo),
                "plan_root": str(plan_root),
                "build_receipt": receipt(),
                "gpu_device": GPU0,
            },
            "tasks": [{"task": {"unit_id": "host-ordinary"}}],
        },
        "specialized": {
            "outcome": "PASS",
            "contract": {
                "paths": paths,
                "build_receipt": receipt(),
                "gpu_devices": [GPU0, GPU1],
            },
            "tasks": specialized_tasks,
        },
        "performance": {
            "outcome": "PASS",
            "contract": {
                "paths": paths,
                "build_receipt": receipt(),
                "gpu_devices": [GPU0, GPU1],
            },
            "tasks": [],
            "unit_results": repeated,
        },
    }


def fake_feature_plan() -> dict[str, object]:
    features = []
    for index in range(42):
        if index == 39:
            feature_id = "distributed.cuda_aware_mpi"
            status = "planned_capability_gated"
            routes = [
                {
                    "evidence": "scripts/run-mpi-python-validation.py",
                    "routes": [
                        {
                            "kind": "specialized",
                            "unit_id": "specialized-mpi-python-core",
                        }
                    ],
                }
            ]
        elif index == 40:
            feature_id = "policy.long_horizon_small"
            status = "planned_policy_gap_allowed"
            routes = [
                {
                    "evidence": "python/examples/stochastic_emitter_line.py",
                    "routes": [
                        {
                            "kind": "performance",
                            "unit_id": "performance-long-horizon-policy",
                        }
                    ],
                }
            ]
        elif index == 41:
            feature_id = "evidence.resume_durability"
            status = "planned"
            routes = [
                {
                    "evidence": "milestone:M3-example-matrix",
                    "routes": [
                        {
                            "kind": "milestone",
                            "unit_id": "milestone:M3-example-matrix",
                        }
                    ],
                }
            ]
        else:
            feature_id = f"solver.feature_{index:02d}"
            status = "planned"
            routes = [
                {
                    "evidence": "python/examples/ordinary.py",
                    "routes": [
                        {"kind": "paired", "unit_id": "paired-ordinary"}
                    ],
                }
            ]
        features.append(
            {
                "feature_id": feature_id,
                "family": "solver_physics",
                "status": status,
                "primary_routes": routes,
                "primary_current_receipt_evidence": [routes[0]["evidence"]],
                "independent_or_native_evidence": ["tests/native.cpp"],
            }
        )
    return {"schema": "fixture", "features": features, "counts": {"features": 42}}


class M3ReleaseAggregationTests(unittest.TestCase):
    def contract(
        self, root: pathlib.Path
    ) -> tuple[dict[str, object], dict[str, pathlib.Path]]:
        repo = root / "repo"
        plan_root = root / "plan"
        repo.mkdir()
        plan_root.mkdir()
        plan_replay = root / "plan-replay.json"
        plan_replay.write_text("{}\n", encoding="utf-8")
        audits = {}
        for name in ("m1", "m2"):
            path = root / f"{name}-audit.json"
            path.write_text("{}\n", encoding="utf-8")
            audits[name] = str(path)
        components = {}
        for name in AGGREGATE.COMPONENT_TERMINALS:
            path = root / name
            path.mkdir()
            components[name] = path
        return (
            {
                "schema": AGGREGATE.SCHEMA,
                "repo": str(repo),
                "m1_replay_repo": str(repo),
                "plan_root": str(plan_root),
                "plan_replay": str(plan_replay),
                "audits": audits,
                "components": {name: str(path) for name, path in components.items()},
            },
            components,
        )

    def derive(
        self,
        contract: dict[str, object],
        reports: dict[str, object],
        plan_replay_override: dict[str, object] | None = None,
    ):
        frozen_plan = fake_feature_plan()
        sealed_plan = plan_replay_override or {
            "repository": contract["repo"],
            "plan_root": contract["plan_root"],
            "git_commit": "a" * 40,
            "plan": frozen_plan,
        }

        def audit_report(path):
            name = pathlib.Path(path).name.split("-", 1)[0]
            milestone = name.upper()
            source_repo = (
                contract["m1_replay_repo"] if name == "m1" else contract["repo"]
            )
            return {
                "outcome": "PASS",
                "milestone": milestone,
                "auditor": {"task_name": f"{name}_adversarial"},
                "source": {"repo": source_repo},
                "evidence": {
                    "root": contract["components"][name],
                    "terminal_name": "COMPLETE",
                },
                "finding_counts": {severity: 0 for severity in AGGREGATE.audit_seal.SEVERITIES},
                "unresolved_counts": {
                    severity: 0 for severity in AGGREGATE.audit_seal.SEVERITIES
                },
            }

        def load_stable_json_record(root, record, label):
            if str(record.get("path", "")).startswith("m2-comparison-"):
                return m2_comparison_fixture()
            if str(record.get("path", "")).startswith("paired-runner-"):
                return paired_runner_fixture()
            raise AssertionError(f"unexpected stable JSON record: {label}")

        with mock.patch.object(
            AGGREGATE, "_component_reports", return_value=reports
        ), mock.patch.object(
            AGGREGATE.feature_plan,
            "load_release_plan",
            return_value=frozen_plan,
        ), mock.patch.object(
            AGGREGATE.plan_replay_seal,
            "verify",
            return_value=sealed_plan,
        ), mock.patch.object(
            AGGREGATE.audit_seal,
            "verify",
            side_effect=audit_report,
        ), mock.patch.object(
            AGGREGATE, "_source_records", return_value={}
        ), mock.patch.object(
            AGGREGATE, "_component_summary", return_value={}
        ), mock.patch.object(
            AGGREGATE.source_equivalence,
            "compare",
            return_value=source_bridge(),
        ), mock.patch.object(
            AGGREGATE,
            "absolute_file_record",
            return_value=reports["m1"]["shared_fp32_receipt"],
        ), mock.patch.object(
            AGGREGATE.matrix,
            "stable_file_record",
            return_value={"path": "fixture", "size_bytes": 1, "sha256": "0" * 64},
        ), mock.patch.object(
            AGGREGATE.matrix,
            "load_stable_json_record",
            side_effect=load_stable_json_record,
        ):
            return AGGREGATE.derive_report(contract)

    def test_derives_42_features_and_seven_performance_units(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract, _components = self.contract(pathlib.Path(raw).resolve())
            reports = fake_reports(
                pathlib.Path(contract["repo"]), pathlib.Path(contract["plan_root"])
            )
            report = self.derive(contract, reports)
            markdown = AGGREGATE.render_markdown(report)
        self.assertEqual(report["feature_summary"]["feature_count"], 42)
        self.assertEqual(len(report["performance"]), 7)
        self.assertEqual(
            report["build_and_hardware_closure"]["receipt_id"], RECEIPT
        )
        self.assertEqual(
            report["build_and_hardware_closure"]["m1_fp32_receipt_id"],
            M1_RECEIPT,
        )
        by_id = {row["feature_id"]: row for row in report["features"]}
        self.assertEqual(
            by_id["distributed.cuda_aware_mpi"]["outcome"], "PASS_CAPABILITY"
        )
        self.assertEqual(
            by_id["policy.long_horizon_small"]["outcome"],
            "PASS_POLICY_GAP_RETAINED",
        )
        self.assertIn("# gpmeep M1--M3 release qualification", markdown)
        self.assertIn("performance-materialgrid-adjoint", markdown)
        self.assertIn("PASS_POLICY_GAP_RETAINED", markdown)
        self.assertIn("Feature-by-feature execution evidence", markdown)
        self.assertIn("M1 TERS exact CPU FP64 vs CUDA FP32 error", markdown)
        self.assertIn("1.000000e-05 / 1.000000e-03", markdown)
        self.assertIn("M2 AuNP exact GPU/oracle and one-/two-GPU error", markdown)
        self.assertIn("Deduplicated Python example CPU/CUDA correctness", markdown)
        self.assertIn("Adversarial milestone audits", markdown)
        self.assertIn("Requested-scope closure", markdown)
        self.assertEqual(report["requirement_closure"]["achieved_count"], 11)
        self.assertEqual(report["requirement_closure"]["deferred_count"], 3)
        for feature in report["features"]:
            self.assertIn(f"`{feature['feature_id']}`", markdown)

    def test_receipt_and_gpu_mutations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract, _components = self.contract(pathlib.Path(raw).resolve())
            repo = pathlib.Path(contract["repo"])
            plan_root = pathlib.Path(contract["plan_root"])
            reports = fake_reports(repo, plan_root)
            reports["specialized"]["contract"]["build_receipt"]["receipt_id"] = "d" * 64
            with self.assertRaisesRegex(Exception, "receipt closure"):
                self.derive(contract, reports)
            reports = fake_reports(repo, plan_root)
            reports["performance"]["contract"]["gpu_devices"] = [GPU1, GPU0]
            with self.assertRaisesRegex(Exception, "GPU identity closure"):
                self.derive(contract, reports)

    def test_missing_cuda_aware_capability_lane_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract, _components = self.contract(pathlib.Path(raw).resolve())
            reports = fake_reports(
                pathlib.Path(contract["repo"]), pathlib.Path(contract["plan_root"])
            )
            reports["specialized"]["tasks"][0]["task_replay"]["evidence"][
                "comparison_names"
            ] = []
            with self.assertRaisesRegex(Exception, "capability lane"):
                self.derive(contract, reports)

    def test_correctness_detail_mutations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract, _components = self.contract(pathlib.Path(raw).resolve())
            repo = pathlib.Path(contract["repo"])
            plan_root = pathlib.Path(contract["plan_root"])

            reports = fake_reports(repo, plan_root)
            reports["m2"]["exact_comparisons"][0].pop("spec")
            with self.assertRaisesRegex(Exception, "M2 exact comparison record"):
                self.derive(contract, reports)

            reports = fake_reports(repo, plan_root)
            reports["paired"]["tasks"][0]["archive"].pop("runner_report")
            with self.assertRaisesRegex(Exception, "paired correctness record"):
                self.derive(contract, reports)

            reports = fake_reports(repo, plan_root)
            reports["paired"]["tasks"].pop()
            with self.assertRaisesRegex(Exception, "requirement closure"):
                self.derive(contract, reports)

            reports = fake_reports(repo, plan_root)
            reports["performance"]["unit_results"][
                "performance-edge-emitter-3d"
            ]["numerical_comparisons"]["pair"]["outcome"] = "FAIL"
            with self.assertRaisesRegex(Exception, "repeated correctness"):
                self.derive(contract, reports)

    def test_final_plan_replay_must_match_roots_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract, _components = self.contract(pathlib.Path(raw).resolve())
            reports = fake_reports(
                pathlib.Path(contract["repo"]), pathlib.Path(contract["plan_root"])
            )
            wrong_root = {
                "repository": "/wrong-repository",
                "plan_root": contract["plan_root"],
                "plan": fake_feature_plan(),
            }
            with self.assertRaisesRegex(Exception, "plan-replay roots"):
                self.derive(contract, reports, wrong_root)
            wrong_plan = {
                "repository": contract["repo"],
                "plan_root": contract["plan_root"],
                "plan": {"features": []},
            }
            with self.assertRaisesRegex(Exception, "plan replay differs"):
                self.derive(contract, reports, wrong_plan)

    def test_adversarial_audit_roots_and_material_closure_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            contract, components = self.contract(pathlib.Path(raw).resolve())

            def audit(name, *, root=None, medium_open=0):
                return {
                    "outcome": "PASS",
                    "milestone": name.upper(),
                    "source": {
                        "repo": (
                            contract["m1_replay_repo"]
                            if name == "m1"
                            else contract["repo"]
                        )
                    },
                    "evidence": {
                        "root": root or str(components[name]),
                        "terminal_name": "COMPLETE",
                    },
                    "unresolved_counts": {
                        "critical": 0,
                        "high": 0,
                        "medium": medium_open,
                    },
                }

            audit_paths = {
                name: pathlib.Path(path) for name, path in contract["audits"].items()
            }
            with mock.patch.object(
                AGGREGATE.audit_seal,
                "verify",
                side_effect=[audit("m1", root="/wrong"), audit("m2")],
            ), self.assertRaisesRegex(Exception, "m1 adversarial audit"):
                AGGREGATE._audit_reports(
                    audit_paths,
                    pathlib.Path(contract["repo"]),
                    pathlib.Path(contract["m1_replay_repo"]),
                    components,
                )
            with mock.patch.object(
                AGGREGATE.audit_seal,
                "verify",
                side_effect=[audit("m1", medium_open=1), audit("m2")],
            ), self.assertRaisesRegex(Exception, "m1 adversarial audit"):
                AGGREGATE._audit_reports(
                    audit_paths,
                    pathlib.Path(contract["repo"]),
                    pathlib.Path(contract["m1_replay_repo"]),
                    components,
                )

    def test_publication_replays_pending_before_atomic_rename(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw).resolve()
            (output / "report.json").write_text("{}\n")
            (output / "report.md").write_text("fixture\n")
            observations = []

            def verify(path, terminal_name="M3_COMPLETE"):
                observations.append(
                    (
                        terminal_name,
                        (path / "PENDING_M3_COMPLETE").is_file(),
                        (path / "M3_COMPLETE").exists(),
                    )
                )
                return {}

            with mock.patch.object(AGGREGATE, "verify_complete", side_effect=verify):
                AGGREGATE.publish_terminal(output)
            self.assertEqual(observations, [("PENDING_M3_COMPLETE", True, False)])
            self.assertTrue((output / "M3_COMPLETE").is_file())

    def test_terminal_full_rederivation_rejects_changed_report(self) -> None:
        keys = {
            "schema",
            "outcome",
            "contract",
            "build_and_hardware_closure",
            "components",
            "adversarial_audits",
            "requirement_closure",
            "feature_plan",
            "feature_summary",
            "features",
            "source_evidence",
            "performance",
            "correctness",
            "evidence_code",
        }
        report = {name: {} for name in keys}
        report.update({"schema": AGGREGATE.SCHEMA, "outcome": "PASS", "contract": {}})
        with tempfile.TemporaryDirectory() as raw:
            output = pathlib.Path(raw).resolve()
            AGGREGATE.atomic_write_json(output / "report.json", report)
            AGGREGATE.atomic_write_text(output / "report.md", "fixture\n")
            AGGREGATE.atomic_write_json(
                output / "M3_COMPLETE", AGGREGATE._terminal_payload(output)
            )
            with mock.patch.object(
                AGGREGATE, "derive_report", return_value=report
            ), mock.patch.object(
                AGGREGATE, "render_markdown", return_value="fixture\n"
            ):
                self.assertEqual(AGGREGATE.verify_complete(output), report)
            AGGREGATE.atomic_write_text(output / "report.md", "tampered\n")
            AGGREGATE.atomic_write_json(
                output / "M3_COMPLETE", AGGREGATE._terminal_payload(output)
            )
            with mock.patch.object(
                AGGREGATE, "derive_report", return_value=report
            ), mock.patch.object(
                AGGREGATE, "render_markdown", return_value="fixture\n"
            ):
                with self.assertRaisesRegex(Exception, "not fully re-derived"):
                    AGGREGATE.verify_complete(output)
            AGGREGATE.atomic_write_text(output / "report.md", "fixture\n")
            report["outcome"] = "FAIL"
            AGGREGATE.atomic_write_json(output / "report.json", report)
            AGGREGATE.atomic_write_json(
                output / "M3_COMPLETE", AGGREGATE._terminal_payload(output)
            )
            with mock.patch.object(
                AGGREGATE, "derive_report", return_value=report
            ), mock.patch.object(
                AGGREGATE, "render_markdown", return_value="fixture\n"
            ):
                with self.assertRaisesRegex(Exception, "not fully re-derived"):
                    AGGREGATE.verify_complete(output)


if __name__ == "__main__":
    unittest.main()
