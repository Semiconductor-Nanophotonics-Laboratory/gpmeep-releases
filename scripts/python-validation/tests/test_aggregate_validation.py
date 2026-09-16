#!/usr/bin/env python3
"""Isolated fail-closed tests for batched validation aggregation."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
from collections import Counter
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "aggregate_validation.py"
SPEC = importlib.util.spec_from_file_location("aggregate_validation_for_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class Fixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.repo.mkdir()
        self.manifest = self.repo / "manifest.json"
        self.manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
        artifact_dir = self.repo / "native"
        artifact_dir.mkdir()
        self.extension = artifact_dir / "_meep.so"
        self.libmeep = artifact_dir / "libmeep.so"
        self.extension.write_bytes(b"python-extension\n")
        self.libmeep.write_bytes(b"libmeep\n")
        self.source = {
            "algorithm": "synthetic-source-v1",
            "file_count": 2,
            "missing_paths": [],
            "source_manifest_schema_version": 1,
            "source_manifest_sha256": "4" * 64,
            "files": [],
            "sha256": "3" * 64,
        }
        self.receipt = {
            "schema_version": 1,
            "state": "complete",
            "source_unchanged": True,
            "source_start": self.source,
            "source_end": self.source,
            "receipt_id": "a" * 64,
            "build_input_id": "b" * 64,
            "artifact_set_id": "c" * 64,
            "artifacts": {
                "python_extension": self._receipt_record(self.extension),
                "libmeep": self._receipt_record(self.libmeep),
            },
        }
        self.receipt_path = self.repo / "build-provenance.json"
        self._write_json(self.receipt_path, self.receipt)
        self.plan_path = self.root / "plan.json"
        self.batch_assignments = {
            "batch-a": [("case/a", "cpu")],
            "batch-b": [("case/b", "cuda")],
        }
        self.manifest_requirements = (
            {
                "case/a": {
                    "id": "case/a",
                    "disposition": "run",
                    "compute_scope": "fdtd_cuda",
                    "gpu_contract": "cuda_dispatch",
                },
                "case/b": {
                    "id": "case/b",
                    "disposition": "run",
                    "compute_scope": "host_only",
                    "gpu_contract": "none",
                },
                "covered-case": {
                    "id": "covered-case",
                    "disposition": "covered_by_test",
                    "compute_scope": "coverage_only",
                },
                "mpi-gate-1": {
                    "id": "mpi-gate-1",
                    "disposition": "mpi_only",
                    "compute_scope": "mpi_fdtd_cuda",
                },
            },
            ["covered-case"],
            ["mpi-gate-1"],
        )
        with mock.patch.object(
            aggregate, "verify_build_receipt", return_value=self.receipt
        ), mock.patch.object(
            aggregate,
            "_manifest_plan_requirements",
            return_value=self.manifest_requirements,
        ):
            self.plan = aggregate.seal_plan(
                output_path=self.plan_path,
                repo=self.repo,
                receipt_path=self.receipt_path,
                manifest_path=self.manifest,
                native_artifact_names=["python_extension", "libmeep"],
                batch_assignments=self.batch_assignments,
                external_gate_ids=["external-gate-1"],
                unresolved_covered_by=["covered-case"],
                unresolved_mpi_gate_semantics=["mpi-gate-1"],
                created_at_utc="2026-01-01T00:00:00Z",
            )
        self.batch_dirs: dict[str, pathlib.Path] = {}
        for batch_id, assignments in self.batch_assignments.items():
            directory = self.root / batch_id
            self.write_runner(directory, assignments)
            self.batch_dirs[batch_id] = directory

    def _receipt_record(self, path: pathlib.Path) -> dict[str, object]:
        data = path.read_bytes()
        return {
            "path": path.relative_to(self.repo).as_posix(),
            "size_bytes": len(data),
            "sha256": sha(data),
        }

    @staticmethod
    def _write_json(path: pathlib.Path, value: object, *, allow_nan: bool = False) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n",
            encoding="utf-8",
        )

    def _runtime_contract(self) -> dict[str, object]:
        return {
            "python_executable": {},
            "meep_module": {},
            "receipt_id": self.plan["identity"]["build_receipt"]["receipt_id"],
            "extension": {
                **self.plan["identity"]["native_artifacts"][1],
                "path": str(self.extension.resolve()),
            },
            "libmeep": {
                **self.plan["identity"]["native_artifacts"][0],
                "path": str(self.libmeep.resolve()),
            },
            "fontconfig_file": {},
            "build_home": str((self.root / "build-home").resolve()),
            "qualification_home": str((self.root / "qualification-home").resolve()),
            "installed_environment": str((self.root / "installed").resolve()),
        }

    def _endpoint(self) -> dict[str, object]:
        source = self.plan["identity"]["source_snapshot"]
        return {
            "captured_at_utc": "2026-01-02T00:00:00Z",
            "available": True,
            "source_snapshot": {
                "available": True,
                "algorithm": source["algorithm"],
                "file_count": source["file_count"],
                "sha256": source["sha256"],
            },
            "build_extension": {},
            "build_receipt": {
                "available": True,
                "path": str(self.receipt_path.resolve()),
                "sha256": self.plan["identity"]["build_receipt"]["sha256"],
                "receipt_id": self.receipt["receipt_id"],
                "build_input_id": self.receipt["build_input_id"],
                "artifact_set_id": self.receipt["artifact_set_id"],
            },
            "runtime_contract": self._runtime_contract(),
            "problems": [],
        }

    def _run(
        self,
        directory: pathlib.Path,
        case_id: str,
        backend: str,
        *,
        duration: float = 1.25,
    ) -> dict[str, object]:
        safe_case = aggregate.archive_contract.runner_contract.case_directory_id(case_id)
        run_root = directory / "runs" / safe_case / backend
        run_root.mkdir(parents=True, exist_ok=True)
        members: dict[str, dict[str, object]] = {}
        filenames = {
            "stdout": "stdout.log",
            "stderr": "stderr.log",
            "statistics": "gpu-statistics.json",
        }
        for evidence_name, filename in filenames.items():
            path = run_root / filename
            if evidence_name == "statistics":
                path.write_bytes(b"{}\n")
            else:
                path.write_bytes(f"{case_id}:{backend}:{evidence_name}\n".encode())
            data = path.read_bytes()
            members[evidence_name] = {
                "available": True,
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": len(data),
                "sha256": sha(data),
            }
        environment = {"MEEP_GPU_BACKEND": backend}
        return {
            "backend": backend,
            "command": ["python", "-m", "unittest"],
            "cwd": f"runs/{safe_case}/{backend}/work",
            "evidence_case_directory": safe_case,
            "environment_contract": {
                "MEEP_GPU_BACKEND": backend,
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "CUDA_CACHE_DISABLE": "1",
                "JAX_PLATFORMS": "cpu",
                "HOME": str((run_root / "runtime" / ("d" * 32) / "home").resolve()),
                "MPLCONFIGDIR": str(
                    (run_root / "runtime" / ("d" * 32) / "matplotlib").resolve()
                ),
                "XDG_CACHE_HOME": str(
                    (run_root / "runtime" / ("d" * 32) / "cache").resolve()
                ),
                "XDG_CONFIG_HOME": str(
                    (run_root / "runtime" / ("d" * 32) / "config").resolve()
                ),
                "FONTCONFIG_FILE": str((self.root / "fonts.conf").resolve()),
                "strict_cuda": backend == "cuda",
                "keys": sorted(environment),
                "sha256": aggregate.archive_contract.runner_contract.environment_sha256(
                    environment
                ),
                "environment": environment,
            },
            "started_at_utc": "2026-01-02T00:00:00Z",
            "run_nonce": "d" * 32,
            "duration_seconds": duration,
            "exit_code": 0,
            "timeout": False,
            "output_limit": None,
            "output_sizes": {
                "stdout": members["stdout"]["size_bytes"],
                "stderr": members["stderr"]["size_bytes"],
                "combined": (
                    members["stdout"]["size_bytes"]
                    + members["stderr"]["size_bytes"]
                ),
            },
            "output_limits": {"stdout": 1, "stderr": 1, "combined": 2},
            "unittest_skips": 0,
            "unittest_skip_details": [],
            "allowed_unittest_skips": 0,
            "unittest_skip_policy_problems": [],
            "unittest_reported_test_counts": [1],
            "unittest_test_count": 1,
            "unittest_test_identities": ["fixture.Case.test_case"],
            "expected_unittest_test_count": 1,
            "expected_unittest_test_identities": ["fixture.Case.test_case"],
            "unittest_test_contract_ok": True,
            "unittest_test_contract_problems": [],
            "unittest_terminal_summaries": ["Ran 1 test"],
            "stdout_log": members["stdout"]["path"],
            "stderr_log": members["stderr"]["path"],
            "statistics_file": members["statistics"]["path"],
            "evidence_files": members,
            "statistics": {},
            "backend_contract_ok": True,
            "backend_contract_problems": [],
            "outcome": "PASS",
        }

    def _result(
        self,
        directory: pathlib.Path,
        case_id: str,
        backend: str,
        *,
        duration: float = 1.25,
    ) -> dict[str, object]:
        result = {
            "id": case_id,
            "path": f"python/tests/{case_id.replace('/', '_')}.py",
            "kind": "unittest",
            "tier": ["full"],
            "disposition": "run",
            "reason": "synthetic aggregation fixture",
            "milestone": "test",
            "selected": True,
            "outcome": "SINGLE_BACKEND_ONLY",
            "runs": {backend: self._run(directory, case_id, backend, duration=duration)},
            "comparison": {
                "mode": "embedded_oracle",
                "outcome": "SINGLE_BACKEND_ONLY",
                "reason": "both CPU and CUDA are required",
            },
        }
        if "compute_scope" in aggregate.RESULT_KEYS:
            result["compute_scope"] = (
                "host_only" if case_id == "case/b" else "fdtd_cuda"
            )
        return result

    def write_runner(
        self,
        directory: pathlib.Path,
        assignments: list[tuple[str, str]],
        *,
        duration: float = 1.25,
        generated_at: str = "2026-01-02T00:00:00Z",
        mutate_report=None,
        allow_nan: bool = False,
    ) -> dict[str, object]:
        directory.mkdir(parents=True, exist_ok=True)
        results = [
            self._result(directory, case_id, backend, duration=duration)
            for case_id, backend in assignments
        ]
        source = self.plan["identity"]["source_snapshot"]
        endpoint = self._endpoint()
        report: dict[str, object] = {
            "schema_version": aggregate.RUNNER_SCHEMA_VERSION,
            "generated_at_utc": generated_at,
            "configuration": {
                "tiers": [],
                "case_ids": sorted(case_id for case_id, _ in assignments),
                "backends": sorted(set(backend for _, backend in assignments)),
                "performance_evidence": {
                    "valid_for_speed_gate": False,
                    "concurrency_detected": False,
                    "invalid_reasons": ["single sample"],
                },
            },
            "provenance": {
                "captured_at_utc": "2026-01-02T00:00:00Z",
                "hostname": "fixture",
                "platform": "fixture",
                "machine": "x86_64",
                "python_executable": "/usr/bin/python3",
                "python_version": "3",
                "repository": str(self.repo.resolve()),
                "git_head": "0" * 40,
                "git_status": "",
                "manifest_sha256": self.plan["identity"]["manifest"]["sha256"],
                "manifest": str(self.manifest.resolve()),
                "source_snapshot": {
                    "available": True,
                    "algorithm": source["algorithm"],
                    "file_count": source["file_count"],
                    "sha256": source["sha256"],
                },
                "build_python": str((self.root / "build-python").resolve()),
                "build_python_exists": True,
                "build_extensions": [],
                "build_package_probe": {},
                "install_prefix": None,
                "install_extensions": [],
                "install_package_probe": {},
                "source_files": [],
                "environment_selection": {},
                "validation_window": {
                    "start": endpoint,
                    "end": copy.deepcopy(endpoint),
                    "unchanged": True,
                    "problems": [],
                },
            },
            "inventory": {
                "total": len(results),
                "selected": len(results),
                "scheme_scope": "excluded by project requirement",
            },
            "summary": dict(
                sorted(Counter(result["outcome"] for result in results).items())
            ),
            "results": results,
            "exit_code": 0,
            "exit_code_meaning": {
                "0": "pass",
                "1": "execution failed",
                "2": "manifest invalid",
                "3": "comparison failed",
                "4": "blocked",
                "5": "backend contract",
                "6": "evidence integrity",
            },
        }
        if mutate_report is not None:
            mutate_report(report)
        report_path = directory / "report.json"
        self._write_json(report_path, report, allow_nan=allow_nan)
        markdown = directory / "report.md"
        markdown.write_text("# synthetic runner report\n", encoding="utf-8")
        report_data = report_path.read_bytes()
        markdown_data = markdown.read_bytes()
        complete = {
            "schema_version": aggregate.RUNNER_SCHEMA_VERSION,
            "completed_at_utc": generated_at,
            "exit_code": report["exit_code"],
            "report_sha256": sha(report_data),
            "report_size_bytes": len(report_data),
            "markdown_sha256": sha(markdown_data),
            "markdown_size_bytes": len(markdown_data),
        }
        self._write_json(directory / "COMPLETE", complete)
        return report

    def aggregate(self, output: pathlib.Path | None = None) -> dict[str, object]:
        output = output or self.root / "aggregate"
        with mock.patch.object(
            aggregate, "verify_build_receipt", return_value=self.receipt
        ), mock.patch.object(
            aggregate,
            "_manifest_plan_requirements",
            return_value=self.manifest_requirements,
        ), mock.patch.object(
            aggregate.archive_contract, "_replay_report_semantics", return_value=None
        ):
            return aggregate.aggregate_to_directory(
                output_dir=output,
                repo=self.repo,
                plan_path=self.plan_path,
                batch_directories=self.batch_dirs,
            )


class AggregateValidationTests(unittest.TestCase):
    def test_real_archive_fixture_passes_authoritative_semantic_replay(self) -> None:
        archive_test_path = pathlib.Path(__file__).with_name(
            "test_archive_validation.py"
        )
        archive_spec = importlib.util.spec_from_file_location(
            "archive_validation_fixture_for_aggregate_test", archive_test_path
        )
        assert archive_spec is not None and archive_spec.loader is not None
        archive_test = importlib.util.module_from_spec(archive_spec)
        archive_spec.loader.exec_module(archive_test)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = archive_test.ArchiveFixture(pathlib.Path(temporary))
            for run in fixture.report["results"][0]["runs"].values():
                for evidence_name, record in run["evidence_files"].items():
                    relative = pathlib.Path(record["path"]).resolve().relative_to(
                        fixture.output.resolve()
                    ).as_posix()
                    record["path"] = relative
                    direct_key = {
                        "stdout": "stdout_log",
                        "stderr": "stderr_log",
                        "statistics": "statistics_file",
                    }[evidence_name]
                    run[direct_key] = relative
            comparison = fixture.report["results"][0]["comparison"]
            comparison["raw_metric_evidence"] = {
                "cpu_stdout": fixture.report["results"][0]["runs"]["cpu"][
                    "evidence_files"
                ]["stdout"],
                "cuda_stdout": fixture.report["results"][0]["runs"]["cuda"][
                    "evidence_files"
                ]["stdout"],
            }
            fixture.reseal_report()
            plan_path = pathlib.Path(temporary) / "aggregate-plan.json"
            aggregate.seal_plan(
                output_path=plan_path,
                repo=fixture.repo,
                receipt_path=fixture.receipt,
                manifest_path=(
                    fixture.repo / "scripts/python-validation/manifest.json"
                ),
                native_artifact_names=["python_extension", "libmeep"],
                batch_assignments={
                    "full-replay": [
                        ("cases/example.py", "cpu"),
                        ("cases/example.py", "cuda"),
                    ]
                },
                created_at_utc="1999-12-31T23:59:59Z",
            )
            output = pathlib.Path(temporary) / "aggregate-output"
            report = aggregate.aggregate_to_directory(
                output_dir=output,
                repo=fixture.repo,
                plan_path=plan_path,
                batch_directories={"full-replay": fixture.output},
            )
            self.assertEqual(report["status"], "ASSIGNED_RUNS_PASS")
            self.assertEqual(report["coverage"]["gpu_coverage_assignment_count"], 1)
            self.assertEqual(
                report["coverage"]["gpu_performance_assignment_count"], 2
            )
            self.assertTrue((output / "COMPLETE").is_file())

    def test_success_binds_plan_report_and_explicit_unresolved_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            output = fixture.root / "aggregate"
            report = fixture.aggregate(output)
            self.assertEqual(
                report["status"], "ASSIGNED_RUNS_PASS_WITH_UNRESOLVED_INPUTS"
            )
            self.assertEqual(report["coverage"]["expected_assignment_count"], 2)
            self.assertEqual(report["coverage"]["observed_assignment_count"], 2)
            self.assertEqual(report["coverage"]["gpu_coverage_assignment_count"], 0)
            self.assertEqual(report["coverage"]["gpu_performance_assignment_count"], 0)
            self.assertEqual(report["coverage"]["host_only_assignment_count"], 1)
            host_cuda = next(
                timing
                for batch in report["batches"]
                for timing in batch["timings"]
                if timing["case_id"] == "case/b"
            )
            self.assertEqual(host_cuda["backend"], "cuda")
            self.assertEqual(host_cuda["compute_scope"], "host_only")
            self.assertFalse(host_cuda["counts_toward_gpu_coverage"])
            self.assertFalse(host_cuda["counts_toward_gpu_performance"])
            self.assertEqual(
                fixture.plan["unresolved_inputs"]["covered_by"], ["covered-case"]
            )
            self.assertEqual(
                fixture.plan["unresolved_inputs"]["mpi_gate_semantics"],
                ["mpi-gate-1"],
            )
            self.assertEqual(report["unresolved_inputs"]["covered_by"], ["covered-case"])
            self.assertEqual(report["unresolved_inputs"]["mpi_gate_semantics"], ["mpi-gate-1"])
            complete = json.loads((output / "COMPLETE").read_text(encoding="utf-8"))
            report_raw = (output / "report.json").read_bytes()
            self.assertEqual(complete["report_sha256"], sha(report_raw))
            self.assertEqual(complete["plan_id"], fixture.plan["plan_id"])
            self.assertEqual(complete["plan_sha256"], sha(fixture.plan_path.read_bytes()))
            with self.assertRaisesRegex(aggregate.AggregationError, "already exists"):
                fixture.aggregate(output)

    def test_recomputed_plan_id_cannot_forge_manifest_policy_or_unresolved_ids(self) -> None:
        def reseal(fixture: Fixture) -> None:
            unsigned = copy.deepcopy(fixture.plan)
            unsigned.pop("plan_id")
            fixture.plan = {**unsigned, "plan_id": aggregate.canonical_sha256(unsigned)}
            fixture._write_json(fixture.plan_path, fixture.plan)

        def forge_assignment_policy(fixture: Fixture) -> None:
            assignment = next(
                assignment
                for batch in fixture.plan["batches"]
                for assignment in batch["assignments"]
                if assignment["case_id"] == "case/b"
            )
            assignment.update(
                {
                    "compute_scope": "fdtd_cuda",
                    "gpu_contract": "cuda_dispatch",
                    "counts_toward_gpu_coverage": True,
                }
            )
            reseal(fixture)

        def forge_unresolved_inputs(fixture: Fixture) -> None:
            fixture.plan["unresolved_inputs"] = {
                "covered_by": [],
                "mpi_gate_semantics": [],
            }
            reseal(fixture)

        for name, mutation, message in (
            ("assignment-policy", forge_assignment_policy, "assignment policy"),
            ("unresolved-inputs", forge_unresolved_inputs, "unresolved inputs"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(pathlib.Path(temporary))
                mutation(fixture)
                with self.assertRaisesRegex(aggregate.AggregationError, message):
                    fixture.aggregate()

    def test_authoritative_semantic_replay_failure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            replay_error = aggregate.archive_contract.ArchiveError(
                "strict CUDA statistics contract failed"
            )
            with mock.patch.object(
                aggregate, "verify_build_receipt", return_value=fixture.receipt
            ), mock.patch.object(
                aggregate,
                "_manifest_plan_requirements",
                return_value=fixture.manifest_requirements,
            ), mock.patch.object(
                aggregate.archive_contract,
                "_replay_report_semantics",
                side_effect=replay_error,
            ) as replay, self.assertRaisesRegex(
                aggregate.AggregationError, "strict CUDA statistics"
            ):
                aggregate.build_aggregate(
                    repo=fixture.repo,
                    plan_path=fixture.plan_path,
                    batch_directories=fixture.batch_dirs,
                )
            self.assertEqual(replay.call_count, 1)

    def test_run_start_outside_validation_window_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))

            def mutate(report):
                report["results"][0]["runs"]["cpu"]["started_at_utc"] = (
                    "2026-01-03T00:00:00Z"
                )

            fixture.write_runner(
                fixture.batch_dirs["batch-a"],
                fixture.batch_assignments["batch-a"],
                mutate_report=mutate,
            )
            with mock.patch.object(
                aggregate, "verify_build_receipt", return_value=fixture.receipt
            ), mock.patch.object(
                aggregate,
                "_manifest_plan_requirements",
                return_value=fixture.manifest_requirements,
            ), mock.patch.object(
                aggregate.archive_contract,
                "_replay_report_semantics",
                return_value=None,
            ), self.assertRaisesRegex(aggregate.AggregationError, "outside validation"):
                aggregate.build_aggregate(
                    repo=fixture.repo,
                    plan_path=fixture.plan_path,
                    batch_directories=fixture.batch_dirs,
                )

    def test_pre_complete_reread_detects_raw_evidence_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            output = fixture.root / "aggregate-race"
            original_write = aggregate._atomic_write_json_noclobber
            changed = False

            def mutate_after_report(path, value):
                nonlocal changed
                snapshot = original_write(path, value)
                if path == output.resolve() / "report.json":
                    raw = next(
                        (fixture.batch_dirs["batch-a"] / "runs").rglob("stdout.log")
                    )
                    raw.write_text("changed after aggregate report\n", encoding="utf-8")
                    changed = True
                return snapshot

            with mock.patch.object(
                aggregate,
                "_atomic_write_json_noclobber",
                side_effect=mutate_after_report,
            ), self.assertRaisesRegex(aggregate.AggregationError, "before COMPLETE"):
                fixture.aggregate(output)
            self.assertTrue(changed)
            self.assertTrue((output / "report.json").is_file())
            self.assertFalse((output / "COMPLETE").exists())

    def test_plan_rejects_duplicate_assignments_gates_and_unobservable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            common = {
                "repo": fixture.repo,
                "receipt_path": fixture.receipt_path,
                "manifest_path": fixture.manifest,
                "native_artifact_names": ["python_extension"],
                "created_at_utc": "2026-01-01T00:00:00Z",
            }
            with mock.patch.object(
                aggregate, "verify_build_receipt", return_value=fixture.receipt
            ), mock.patch.object(
                aggregate,
                "_manifest_plan_requirements",
                return_value=fixture.manifest_requirements,
            ):
                with self.assertRaisesRegex(aggregate.AggregationError, "multiple batches"):
                    aggregate.build_plan(
                        **common,
                        batch_assignments={"a": [("case", "cpu")], "b": [("case", "cpu")]},
                    )
                with self.assertRaisesRegex(aggregate.AggregationError, "case appears"):
                    aggregate.build_plan(
                        **common,
                        batch_assignments={
                            "a": [("case", "cpu")],
                            "b": [("case", "cuda")],
                        },
                    )
                with self.assertRaisesRegex(aggregate.AggregationError, "must be unique"):
                    aggregate.build_plan(
                        **common,
                        batch_assignments={"a": [("case/a", "cpu")]},
                        external_gate_ids=["gate", "gate"],
                    )
                with self.assertRaisesRegex(aggregate.AggregationError, "not observable"):
                    aggregate.build_plan(
                        **{**common, "native_artifact_names": ["native-test"]},
                        batch_assignments={"a": [("case/a", "cpu")]},
                    )
                with self.assertRaisesRegex(aggregate.AggregationError, "no materialized"):
                    aggregate.build_plan(
                        **common,
                        batch_assignments={"a": [("not-in-manifest", "cpu")]},
                    )
                with self.assertRaisesRegex(aggregate.AggregationError, "differ from"):
                    aggregate.build_plan(
                        **common,
                        batch_assignments={"a": [("case/a", "cpu")]},
                        unresolved_covered_by=["caller-omitted-real-ids"],
                    )

    def test_plan_and_output_are_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            with mock.patch.object(
                aggregate, "verify_build_receipt", return_value=fixture.receipt
            ), mock.patch.object(
                aggregate,
                "_manifest_plan_requirements",
                return_value=fixture.manifest_requirements,
            ):
                with self.assertRaisesRegex(aggregate.AggregationError, "already exists"):
                    aggregate.seal_plan(
                        output_path=fixture.plan_path,
                        repo=fixture.repo,
                        receipt_path=fixture.receipt_path,
                        manifest_path=fixture.manifest,
                        native_artifact_names=["python_extension"],
                        batch_assignments={"a": [("case/a", "cpu")]},
                        created_at_utc="2026-01-01T00:00:00Z",
                    )

    def test_missing_extra_and_reused_batch_directories_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            variants = [
                {"batch-a": fixture.batch_dirs["batch-a"]},
                {**fixture.batch_dirs, "extra": fixture.root / "extra"},
                {
                    "batch-a": fixture.batch_dirs["batch-a"],
                    "batch-b": fixture.batch_dirs["batch-a"],
                },
            ]
            for index, directories in enumerate(variants):
                with self.subTest(index=index), mock.patch.object(
                    aggregate, "verify_build_receipt", return_value=fixture.receipt
                ), mock.patch.object(
                    aggregate.archive_contract,
                    "_replay_report_semantics",
                    return_value=None,
                ):
                    with self.assertRaises(aggregate.AggregationError):
                        aggregate.build_aggregate(
                            repo=fixture.repo,
                            plan_path=fixture.plan_path,
                            batch_directories=directories,
                        )

    def test_missing_extra_and_duplicate_cases_are_rejected(self) -> None:
        mutations = {
            "missing": lambda report: report["results"][0].__setitem__("runs", {}),
            "extra": lambda report: report["results"][0].__setitem__("id", "extra-case"),
            "duplicate": lambda report: (
                report["results"].append(copy.deepcopy(report["results"][0])),
                report["inventory"].__setitem__("total", 2),
                report["inventory"].__setitem__("selected", 2),
                report.__setitem__("summary", {"PASS": 2}),
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(pathlib.Path(temporary))
                directory = fixture.batch_dirs["batch-a"]
                fixture.write_runner(
                    directory,
                    fixture.batch_assignments["batch-a"],
                    mutate_report=mutation,
                )
                with mock.patch.object(
                    aggregate, "verify_build_receipt", return_value=fixture.receipt
                ), mock.patch.object(
                    aggregate.archive_contract,
                    "_replay_report_semantics",
                    return_value=None,
                ), self.assertRaises(aggregate.AggregationError):
                    aggregate.build_aggregate(
                        repo=fixture.repo,
                        plan_path=fixture.plan_path,
                        batch_directories=fixture.batch_dirs,
                    )

    def test_report_identity_mismatches_are_rejected(self) -> None:
        mutations = {
            "manifest": lambda report: report["provenance"].__setitem__(
                "manifest_sha256", "0" * 64
            ),
            "source": lambda report: report["provenance"]["source_snapshot"].__setitem__(
                "sha256", "0" * 64
            ),
            "receipt": lambda report: report["provenance"]["validation_window"]["start"][
                "build_receipt"
            ].__setitem__("sha256", "0" * 64),
            "artifact": lambda report: report["provenance"]["validation_window"]["end"][
                "runtime_contract"
            ]["extension"].__setitem__("sha256", "0" * 64),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(pathlib.Path(temporary))
                fixture.write_runner(
                    fixture.batch_dirs["batch-a"],
                    fixture.batch_assignments["batch-a"],
                    mutate_report=mutation,
                )
                with mock.patch.object(
                    aggregate, "verify_build_receipt", return_value=fixture.receipt
                ), mock.patch.object(
                    aggregate.archive_contract,
                    "_replay_report_semantics",
                    return_value=None,
                ), self.assertRaises(aggregate.AggregationError):
                    aggregate.build_aggregate(
                        repo=fixture.repo,
                        plan_path=fixture.plan_path,
                        batch_directories=fixture.batch_dirs,
                    )

    def test_stale_current_receipt_manifest_and_artifact_are_rejected(self) -> None:
        mutations = {
            "receipt": lambda fixture: fixture.receipt_path.write_text("{}\n", encoding="utf-8"),
            "manifest": lambda fixture: fixture.manifest.write_text(
                '{"schema_version":2}\n', encoding="utf-8"
            ),
            "artifact": lambda fixture: fixture.extension.write_bytes(b"changed\n"),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(pathlib.Path(temporary))
                mutation(fixture)
                with mock.patch.object(
                    aggregate, "verify_build_receipt", return_value=fixture.receipt
                ), mock.patch.object(
                    aggregate.archive_contract,
                    "_replay_report_semantics",
                    return_value=None,
                ), self.assertRaises(aggregate.AggregationError):
                    aggregate.build_aggregate(
                        repo=fixture.repo,
                        plan_path=fixture.plan_path,
                        batch_directories=fixture.batch_dirs,
                    )

    def test_failed_integrity_invalid_stale_and_nonfinite_reports_are_rejected(self) -> None:
        def failed(report):
            report["results"][0]["runs"]["cpu"]["outcome"] = "PROCESS_FAILED"

        def invalid_window(report):
            report["provenance"]["validation_window"]["unchanged"] = False
            report["provenance"]["validation_window"]["problems"] = ["changed"]

        def nonfinite(report):
            report["results"][0]["runs"]["cpu"]["duration_seconds"] = float("nan")

        def huge_timing(report):
            report["results"][0]["runs"]["cpu"]["duration_seconds"] = 10**1000

        def passport(report):
            report["results"][0]["runs"]["cpu"]["outcome"] = "PASSPORT"

        def comparison_mismatch(report):
            report["results"][0]["outcome"] = "MISMATCH"
            report["results"][0]["comparison"]["outcome"] = "MISMATCH"
            report["summary"] = {"MISMATCH": 1}

        variants = {
            "failed": (failed, False, "2026-01-02T00:00:00Z"),
            "invalid-window": (invalid_window, False, "2026-01-02T00:00:00Z"),
            "nonfinite": (nonfinite, True, "2026-01-02T00:00:00Z"),
            "huge-timing": (huge_timing, False, "2026-01-02T00:00:00Z"),
            "passport": (passport, False, "2026-01-02T00:00:00Z"),
            "comparison-mismatch": (
                comparison_mismatch,
                False,
                "2026-01-02T00:00:00Z",
            ),
            "stale": (None, False, "2025-01-01T00:00:00Z"),
            "future": (None, False, "2099-01-01T00:00:00Z"),
        }
        for name, (mutation, allow_nan, generated) in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(pathlib.Path(temporary))
                fixture.write_runner(
                    fixture.batch_dirs["batch-a"],
                    fixture.batch_assignments["batch-a"],
                    mutate_report=mutation,
                    allow_nan=allow_nan,
                    generated_at=generated,
                )
                with mock.patch.object(
                    aggregate, "verify_build_receipt", return_value=fixture.receipt
                ), mock.patch.object(
                    aggregate.archive_contract,
                    "_replay_report_semantics",
                    return_value=None,
                ), self.assertRaises(aggregate.AggregationError):
                    aggregate.build_aggregate(
                        repo=fixture.repo,
                        plan_path=fixture.plan_path,
                        batch_directories=fixture.batch_dirs,
                    )

    def test_external_gate_identifiers_alone_keep_status_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(pathlib.Path(temporary))
            no_manifest_unresolved = (
                {
                    "case/a": {
                        "id": "case/a",
                        "disposition": "run",
                        "compute_scope": "fdtd_cuda",
                        "gpu_contract": "cuda_dispatch",
                    },
                    "case/b": {
                        "id": "case/b",
                        "disposition": "run",
                        "compute_scope": "host_only",
                        "gpu_contract": "none",
                    },
                },
                [],
                [],
            )
            with mock.patch.object(
                aggregate, "verify_build_receipt", return_value=fixture.receipt
            ), mock.patch.object(
                aggregate,
                "_manifest_plan_requirements",
                return_value=no_manifest_unresolved,
            ):
                fixture.plan = aggregate.build_plan(
                    repo=fixture.repo,
                    receipt_path=fixture.receipt_path,
                    manifest_path=fixture.manifest,
                    native_artifact_names=["python_extension", "libmeep"],
                    batch_assignments=fixture.batch_assignments,
                    external_gate_ids=["external-only"],
                    unresolved_covered_by=[],
                    unresolved_mpi_gate_semantics=[],
                    created_at_utc="2026-01-01T00:00:00Z",
                )
            fixture._write_json(fixture.plan_path, fixture.plan)
            for batch_id, assignments in fixture.batch_assignments.items():
                fixture.write_runner(fixture.batch_dirs[batch_id], assignments)
            fixture.manifest_requirements = no_manifest_unresolved
            report = fixture.aggregate()
            self.assertEqual(
                report["status"], "ASSIGNED_RUNS_PASS_WITH_UNRESOLVED_INPUTS"
            )
            self.assertEqual(report["unresolved_inputs"]["covered_by"], [])
            self.assertEqual(report["unresolved_inputs"]["mpi_gate_semantics"], [])

    def test_complete_raw_evidence_and_plan_tampering_are_rejected(self) -> None:
        mutations = {
            "complete": lambda fixture: fixture._write_json(
                fixture.batch_dirs["batch-a"] / "COMPLETE",
                {
                    **json.loads(
                        (fixture.batch_dirs["batch-a"] / "COMPLETE").read_text(
                            encoding="utf-8"
                        )
                    ),
                    "report_sha256": "0" * 64,
                },
            ),
            "raw-evidence": lambda fixture: next(
                (fixture.batch_dirs["batch-a"] / "runs").rglob("stdout.log")
            ).write_text("tampered\n", encoding="utf-8"),
            "plan": lambda fixture: fixture._write_json(
                fixture.plan_path,
                {
                    **json.loads(fixture.plan_path.read_text(encoding="utf-8")),
                    "external_gate_ids": ["tampered-gate"],
                },
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(pathlib.Path(temporary))
                mutation(fixture)
                with mock.patch.object(
                    aggregate, "verify_build_receipt", return_value=fixture.receipt
                ), mock.patch.object(
                    aggregate.archive_contract,
                    "_replay_report_semantics",
                    return_value=None,
                ), self.assertRaises(aggregate.AggregationError):
                    aggregate.build_aggregate(
                        repo=fixture.repo,
                        plan_path=fixture.plan_path,
                        batch_directories=fixture.batch_dirs,
                    )


if __name__ == "__main__":
    unittest.main()
