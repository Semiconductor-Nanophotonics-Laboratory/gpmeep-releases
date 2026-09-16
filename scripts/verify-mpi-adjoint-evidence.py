#!/usr/bin/env python3
"""Read-only consumer verifier for published gpmeep MPI-adjoint evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import stat
import sys
import types
from typing import Any


def _load_source_module(name: str, path: pathlib.Path) -> types.ModuleType:
    resolved = path.resolve(strict=True)
    module = types.ModuleType(name)
    module.__file__ = str(resolved)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(resolved.read_bytes(), str(resolved), "exec"), module.__dict__)
    return module


def _canonical_file(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is absent") from error
    if (
        lexical != resolved
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise RuntimeError(f"{label} must be a canonical regular file")
    return resolved


def _canonical_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is absent") from error
    if (
        lexical != resolved
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise RuntimeError(f"{label} must be a canonical directory")
    return resolved


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script = pathlib.Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Replay a COMPLETE gpmeep MPI-adjoint publication from its raw "
            "files using the exact Python interpreter bound into its build receipt."
        )
    )
    parser.add_argument("--repo", type=pathlib.Path, default=script.parent.parent)
    parser.add_argument("--build-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--receipt-interpreter-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _receipt_path(record: Any, repo: pathlib.Path, label: str) -> pathlib.Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise RuntimeError(f"build receipt {label} record is absent")
    path = pathlib.Path(record["path"])
    return (path if path.is_absolute() else repo / path).resolve(strict=True)


def _ensure_receipt_interpreter(
    args: argparse.Namespace,
    receipt: dict[str, Any],
    repo: pathlib.Path,
) -> pathlib.Path:
    python = _receipt_path(receipt.get("toolchain", {}).get("python"), repo, "Python")
    record = receipt["toolchain"]["python"]
    if _sha256(python) != record.get("sha256"):
        raise RuntimeError("receipt-bound Python executable changed")
    current = pathlib.Path(sys.executable).resolve(strict=True)
    if current == python:
        return python
    if args.receipt_interpreter_child:
        raise RuntimeError("receipt-bound Python re-exec selected the wrong executable")
    command = [
        str(python),
        "-I",
        "-B",
        str(pathlib.Path(__file__).resolve()),
        "--repo",
        str(repo),
        "--build-receipt",
        str(args.build_receipt),
        "--output",
        str(args.output),
        "--receipt-interpreter-child",
    ]
    environment = {
        "HOME": os.environ.get("HOME", str(repo)),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    os.execve(str(python), command, environment)
    raise AssertionError("unreachable")


def _validate_prepublication_inventory(
    runner: types.ModuleType,
    output: pathlib.Path,
    run_id: str,
    inventory: Any,
) -> None:
    if (
        not isinstance(inventory, dict)
        or set(inventory) != {"pass", "epoch", "manifest", "sha256"}
        or inventory.get("pass") is not True
        or inventory.get("epoch") != "prepublication-running"
        or not isinstance(inventory.get("manifest"), dict)
        or runner.canonical_sha256(inventory["manifest"]) != inventory.get("sha256")
    ):
        raise RuntimeError("prepublication inventory is invalid")
    manifest = inventory["manifest"]
    if (
        set(manifest) != {"files", "runtime_directories"}
        or not isinstance(manifest.get("files"), list)
        or not isinstance(manifest.get("runtime_directories"), list)
        or manifest.get("runtime_directories")
        != sorted(set(manifest.get("runtime_directories", [])))
        or any(
            not isinstance(name, str)
            or name not in {"cache", "home", "matplotlib", "pycache", "tmp"}
            for name in manifest.get("runtime_directories", [])
        )
    ):
        raise RuntimeError("prepublication manifest schema is invalid")
    expected_running = (
        json.dumps(
            {"schema_version": 1, "state": "RUNNING", "run_id": run_id},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    expected_running_sha = hashlib.sha256(expected_running).hexdigest()
    seen: set[str] = set()
    for record in manifest["files"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256"}
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise RuntimeError("prepublication file record is invalid")
        path = pathlib.Path(record["path"])
        if path.parent != output or path.name in seen:
            raise RuntimeError("prepublication file ownership is invalid")
        seen.add(path.name)
        if path.name == "state.json":
            if record["sha256"] != expected_running_sha:
                raise RuntimeError("prepublication RUNNING state digest is invalid")
            continue
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or _sha256(path) != record["sha256"]
        ):
            raise RuntimeError("prepublication immutable file changed")
    if "state.json" not in seen:
        raise RuntimeError("prepublication RUNNING state record is absent")
    if [pathlib.Path(record["path"]).name for record in manifest["files"]] != sorted(
        seen
    ):
        raise RuntimeError("prepublication file records are not canonical")


def _expected_integrity_sample_records(
    runner: types.ModuleType, samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        summary = runner.summarize_samples([sample])[0]
        records.append(
            {
                "label": (
                    f"{sample['lane']}-{sample['sample_kind']}-{sample['iteration']}"
                ),
                "nonce": sample["nonce"],
                "process_seconds": sample["process_seconds"],
                "summary_sha256": runner.canonical_sha256(summary),
                "files": {
                    path_name: {
                        "path": sample[path_name],
                        "sha256": sample[digest_name],
                    }
                    for path_name, digest_name in (
                        ("result_file", "result_sha256"),
                        ("stdout_log", "stdout_sha256"),
                        ("stderr_log", "stderr_sha256"),
                        ("timing_file", "timing_sha256"),
                    )
                },
            }
        )
    return records


def _expected_integrity_fd_logs(fd: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        path_name: {"path": fd[path_name], "sha256": fd[digest_name]}
        for path_name, digest_name in (
            ("stdout_log", "stdout_sha256"),
            ("stderr_log", "stderr_sha256"),
            ("timing_file", "timing_sha256"),
        )
    }


def _validate_prepublication_closure(
    output: pathlib.Path,
    inventory: dict[str, Any],
    sample_records: list[dict[str, Any]],
    fd_logs: dict[str, dict[str, str]],
) -> None:
    expected = {
        (pathlib.Path(record["path"]).name, record["sha256"])
        for sample in sample_records
        for record in sample["files"].values()
    }
    expected.update(
        (pathlib.Path(record["path"]).name, record["sha256"])
        for record in fd_logs.values()
    )
    manifest = inventory["manifest"]
    running = next(
        record for record in manifest["files"] if record["path"] == str(output / "state.json")
    )
    expected.add(("state.json", running["sha256"]))
    observed = {
        (pathlib.Path(record["path"]).name, record["sha256"])
        for record in manifest["files"]
    }
    if observed != expected or len(observed) != len(manifest["files"]):
        raise RuntimeError("prepublication manifest is not the exact raw closure")


def _validate_postprocessor(
    runner: types.ModuleType,
    postprocessor: Any,
    *,
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    receipt: dict[str, Any],
    output: pathlib.Path,
    interpreter: pathlib.Path,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "git_head",
        "git_status_porcelain",
        "source_snapshot",
        "source_sha256",
        "python",
        "platform",
        "nvidia_driver",
        "nvidia_topology",
        "driver_device_binding",
        "command",
    }
    if not isinstance(postprocessor, dict) or set(postprocessor) != expected_keys:
        raise RuntimeError("published postprocessor schema is invalid")
    expected_python = {
        "path": str(interpreter),
        "sha256": _sha256(interpreter),
        "version": sys.version,
    }
    if (
        postprocessor.get("git_head")
        != runner.git_output(repo, "rev-parse", "HEAD").strip()
        or postprocessor.get("git_status_porcelain")
        != runner.git_output(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        or postprocessor.get("source_snapshot") != receipt["source_end"]
        or postprocessor.get("source_sha256")
        != {name: _sha256(repo / name) for name in runner.RUNNER_SOURCES}
        or postprocessor.get("python") != expected_python
        or postprocessor.get("platform") != platform.platform()
    ):
        raise RuntimeError("published postprocessor source binding changed")
    command = postprocessor.get("command")
    if (
        not isinstance(command, list)
        or len(command) < 2
        or command[0] != str(interpreter)
        or command[1] != str((repo / "scripts" / "run-mpi-adjoint-benchmark.py").resolve())
        or any(not isinstance(item, str) for item in command)
    ):
        raise RuntimeError("published postprocessor command is invalid")
    try:
        parsed = runner.parse_args(command[2:])
    except SystemExit as error:
        raise RuntimeError(
            "published postprocessor command argument parsing escaped verification"
        ) from error
    recorded_receipt = parsed.build_receipt
    if not recorded_receipt.is_absolute():
        recorded_receipt = repo / recorded_receipt
    if (
        recorded_receipt.resolve(strict=True) != receipt_path
        or parsed.output.resolve(strict=True) != output
        or parsed.profile != runner.PROFILE_ID
    ):
        raise RuntimeError("published postprocessor command binding changed")
    driver_binding = runner.validate_driver_device_binding(
        postprocessor.get("nvidia_driver"),
        comparison.get("devices", {}).get("identifiers", [])[:1],
        postprocessor.get("nvidia_topology"),
        repo=repo,
    )
    driver_binding["rank_local"] = comparison.get("devices", {}).get(
        "rank_local_driver_by_lane_and_rank"
    )
    if postprocessor.get("driver_device_binding") != driver_binding:
        raise RuntimeError("published postprocessor driver binding changed")
    return driver_binding


def _reconstruct_samples(
    runner: types.ModuleType,
    report: dict[str, Any],
    receipt: dict[str, Any],
    repo: pathlib.Path,
    output: pathlib.Path,
) -> list[dict[str, Any]]:
    runtime = runner._receipt_runtime(receipt, repo)
    snapshot_sha256 = receipt["source_end"]["sha256"]
    producer_sha256 = runner.sha256_file(repo / "scripts" / "benchmark-adjoint.py")
    run_id = report["run_id"]
    samples: list[dict[str, Any]] = []
    for summary in report.get("samples", []):
        if not isinstance(summary, dict):
            raise RuntimeError("published sample summary is invalid")
        raw_schema = summary.get("raw_result_schema")
        raw_gradient_sha256 = summary.get("raw_gradient_sha256")
        base = {
            key: value
            for key, value in summary.items()
            if key not in {"raw_result_schema", "raw_gradient_sha256"}
        }
        lane = base.get("lane")
        sample_kind = base.get("sample_kind")
        iteration = base.get("iteration")
        ranks = runner._lane_ranks(lane)
        artifact_label = f"{run_id}-{lane}-{sample_kind}-{iteration}"
        if (
            base.get("artifact_label") != artifact_label
            or base.get("ranks") != ranks
            or base.get("capture_field_times") is not False
            or base.get("completion_policy") != "waitsome"
        ):
            raise RuntimeError("published sample identity is invalid")
        paths: dict[str, pathlib.Path] = {}
        for path_name, digest_name, suffix in (
            ("result_file", "result_sha256", ".json"),
            ("stdout_log", "stdout_sha256", ".stdout.log"),
            ("stderr_log", "stderr_sha256", ".stderr.log"),
            ("timing_file", "timing_sha256", ".parent-timing.json"),
        ):
            path = runner._require_owned_regular_file(
                base.get(path_name),
                output=output,
                expected_name=f"{artifact_label}{suffix}",
                label=f"published sample {path_name}",
            )
            if _sha256(path) != base.get(digest_name):
                raise RuntimeError("published sample artifact changed")
            paths[path_name] = path
        raw = json.loads(paths["result_file"].read_text(encoding="utf-8"))
        sample = dict(base)
        sample["result"] = raw
        if set(sample) != runner.EXPECTED_SAMPLE_KEYS:
            raise RuntimeError("published sample exact schema is invalid")
        if (
            raw.get("schema_version") != raw_schema
            or runner.canonical_float64_sha256(raw["result"]["gradient"])
            != raw_gradient_sha256
            or runner.summarize_samples([sample])[0] != summary
        ):
            raise RuntimeError("published raw sample and summary disagree")

        expected_command = runner._expected_sample_command(
            repo=repo,
            runtime=runtime,
            receipt_id=receipt["receipt_id"],
            snapshot_sha256=snapshot_sha256,
            producer_sha256=producer_sha256,
            run_id=run_id,
            nonce=sample["nonce"],
            lane=lane,
            sample_kind=sample_kind,
            iteration=iteration,
            result_path=paths["result_file"],
            ranks=ranks,
            capture_field_times=False,
        )
        expected_environment = runner._child_environment(
            runtime,
            lane=lane,
            output=output,
            pycache_namespace=run_id,
            prepare_filesystem=False,
        )
        expected_environment["MEEP_GPU_MPI_COMPLETION"] = "waitsome"
        expected_environment = dict(sorted(expected_environment.items()))
        if sample["command"] != expected_command or sample["environment"] != expected_environment:
            raise RuntimeError("published sample launch command or environment changed")
        timing = json.loads(paths["timing_file"].read_text(encoding="utf-8"))
        if timing != {
            "schema_version": 1,
            "run_id": run_id,
            "label": artifact_label,
            "lane": lane,
            "sample_kind": sample_kind,
            "iteration": iteration,
            "nonce": sample["nonce"],
            "process_seconds": sample["process_seconds"],
            "command_sha256": runner.canonical_sha256(expected_command),
        }:
            raise RuntimeError("published parent timing record changed")
        pointer = runner._extract_pointer(paths["stdout_log"].read_text(encoding="utf-8"))
        if (
            pathlib.Path(str(pointer.get("result_file", ""))).resolve()
            != paths["result_file"]
            or pointer.get("sha256") != sample["result_sha256"]
        ):
            raise RuntimeError("published raw-result pointer changed")
        runner.COMPARATOR.validate_qualification_binding(
            raw,
            expected_nonce=sample["nonce"],
            expected_run_id=run_id,
            expected_receipt_id=receipt["receipt_id"],
            expected_source_snapshot_sha256=snapshot_sha256,
            expected_producer_sha256=producer_sha256,
            expected_sample_kind=sample_kind,
            expected_sample_lane=lane,
            expected_sample_iteration=iteration,
            label=artifact_label,
        )
        expected_producer_command = runner._expected_producer_command(
            repo=repo,
            runtime=runtime,
            receipt_id=receipt["receipt_id"],
            snapshot_sha256=snapshot_sha256,
            producer_sha256=producer_sha256,
            run_id=run_id,
            nonce=sample["nonce"],
            lane=lane,
            sample_kind=sample_kind,
            iteration=iteration,
            result_path=paths["result_file"],
            ranks=ranks,
            capture_field_times=False,
        )
        if raw.get("producer", {}).get("command") != expected_producer_command:
            raise RuntimeError("published raw producer command changed")
        runner._fixed_workload_gate(raw, lane=lane)
        lazy = runner._validate_lazy_import_contract(raw, label=artifact_label)
        if lane == runner.CPU_ORACLE_LANE:
            validation = runner.validate_cpu_oracle(raw)
        elif lane == runner.CPU_BENCHMARK_LANE:
            validation = runner.validate_cpu_benchmark_record(
                raw, label=artifact_label
            )
        elif lane in runner.CUDA_LANES:
            validation = runner.COMPARATOR.validate_distributed_record(
                raw, expected_world_size=ranks, label=artifact_label
            )
            validation["rank_driver"] = runner._validate_cuda_rank_driver_evidence(
                raw, validation, label=artifact_label
            )
        receipt_gate = runner._validate_receipt_rank_runtimes(receipt, validation)
        process_environment = runner._validate_actual_process_environments(
            raw, expected_environment, label=artifact_label
        )
        runner._validate_lane_process_environment(
            raw,
            expected_environment,
            lane=lane,
            label=artifact_label,
        )
        material_gradient = runner._extract_material_gradient_stats(
            paths["stdout_log"].read_text(encoding="utf-8"), lane=lane
        )
        if (
            lane != runner.CPU_ORACLE_LANE
            and sample["process_seconds"]
            < runner._finite_positive(
                validation.get("workload_wall_seconds"),
                f"{artifact_label}.validated workload wall",
            )
        ):
            raise RuntimeError("published parent process time is shorter than workload")
        if (
            sample["validation"] != validation
            or sample["receipt_gate"] != receipt_gate
            or sample["process_environment_gate"] != process_environment
            or sample["lazy_import_gate"] != lazy
            or sample["material_gradient_gate"] != material_gradient
        ):
            raise RuntimeError("published sample derived validation changed")
        samples.append(sample)
    runner.validate_sample_matrix(samples)
    return samples


def _validate_fd_oracle(
    runner: types.ModuleType,
    report: dict[str, Any],
    receipt: dict[str, Any],
    repo: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, Any]:
    fd = report.get("finite_difference_oracle")
    if not isinstance(fd, dict):
        raise RuntimeError("finite-difference oracle is absent")
    runner.validate_fd_oracle_summary(fd)
    run_id = report["run_id"]
    runtime = runner._receipt_runtime(receipt, repo)
    test_path = (repo / "python" / "tests" / "test_adjoint_default_material_grid.py").resolve()
    if (
        fd.get("receipt_id") != receipt["receipt_id"]
        or fd.get("source_snapshot_sha256") != receipt["source_end"]["sha256"]
        or fd.get("test_file") != str(test_path)
        or fd.get("test_sha256") != _sha256(test_path)
        or fd.get("methods") != list(runner.FD_TEST_METHODS)
    ):
        raise RuntimeError("finite-difference oracle binding changed")
    label = f"{run_id}-directional-fd-oracle"
    paths: dict[str, pathlib.Path] = {}
    for path_name, digest_name, suffix in (
        ("stdout_log", "stdout_sha256", ".stdout.log"),
        ("stderr_log", "stderr_sha256", ".stderr.log"),
        ("timing_file", "timing_sha256", ".parent-timing.json"),
    ):
        path = runner._require_owned_regular_file(
            fd.get(path_name),
            output=output,
            expected_name=f"{label}{suffix}",
            label=f"finite-difference oracle {path_name}",
        )
        if _sha256(path) != fd.get(digest_name):
            raise RuntimeError("finite-difference oracle artifact changed")
        paths[path_name] = path
    command = [str(runtime["python"]), str(test_path), "-v"]
    environment = dict(
        sorted(
            runner._child_environment(
                runtime,
                lane="cpu-legacy-oracle",
                output=output,
                pycache_namespace=run_id,
                prepare_filesystem=False,
            ).items()
        )
    )
    timing = json.loads(paths["timing_file"].read_text(encoding="utf-8"))
    if (
        fd.get("command") != command
        or fd.get("environment") != environment
        or timing
        != {
            "schema_version": 1,
            "run_id": run_id,
            "label": label,
            "process_seconds": fd.get("process_seconds"),
            "started_at_utc": fd.get("started_at_utc"),
            "completed_at_utc": fd.get("completed_at_utc"),
            "command_sha256": runner.canonical_sha256(command),
            "environment_sha256": runner.canonical_sha256(environment),
        }
        or fd.get("direction_records")
        != runner._parse_fd_oracle_output(
            paths["stdout_log"].read_text(encoding="utf-8"),
            paths["stderr_log"].read_text(encoding="utf-8"),
        )
    ):
        raise RuntimeError("finite-difference oracle derived evidence changed")
    return fd


def verify(args: argparse.Namespace) -> dict[str, Any]:
    repo = _canonical_directory(args.repo, "repository")
    receipt_arg = args.build_receipt
    if not receipt_arg.is_absolute():
        receipt_arg = repo / receipt_arg
    receipt_path = _canonical_file(receipt_arg, "build receipt")
    output = _canonical_directory(args.output, "evidence output")
    provenance = _load_source_module(
        "gpmeep_public_verifier_provenance", repo / "scripts" / "gpmeep_provenance.py"
    )
    receipt = provenance.verify_build_receipt(receipt_path, repo, verify_source=True)
    interpreter = _ensure_receipt_interpreter(args, receipt, repo)
    # Re-read the receipt after any re-exec so all semantic work is performed
    # by the interpreter whose path and bytes are sealed in that receipt.
    receipt = provenance.verify_build_receipt(receipt_path, repo, verify_source=True)
    runner = _load_source_module(
        "gpmeep_public_verifier_runner", repo / "scripts" / "run-mpi-adjoint-benchmark.py"
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    candidate_final_epoch = (
        report.get("final_publish_reverification")
        if isinstance(report, dict)
        else None
    )
    candidate_prepublication = (
        candidate_final_epoch.get("output_inventory")
        if isinstance(candidate_final_epoch, dict)
        else None
    )
    marker = runner.verify_terminal_publication(
        output,
        expected_prepublication_inventory=candidate_prepublication,
        expected_run_id=(report.get("run_id") if isinstance(report, dict) else None),
    )
    run_id = marker["run_id"]
    expected_report_keys = {
        "schema_version",
        "state",
        "run_id",
        "profile",
        "build_receipt",
        "finite_difference_oracle",
        "host_audit",
        "integrity_reverification",
        "sample_matrix",
        "samples",
        "comparison",
        "gate",
        "postprocessor",
        "final_publish_reverification",
        "report_markdown",
    }
    if (
        not isinstance(report, dict)
        or set(report) != expected_report_keys
        or report.get("schema_version") != 1
        or report.get("state") != "COMPLETE"
        or report.get("run_id") != run_id
        or report.get("profile") != dict(runner.PROFILE)
        or report.get("gate", {}).get("pass") is not True
        or any(value is not True for value in report.get("gate", {}).values())
    ):
        raise RuntimeError("published COMPLETE report schema or gate is invalid")
    contract = runner.validate_build_receipt_contract(receipt, repo)
    expected_receipt_record = runner.build_receipt_report_record(
        receipt_path, receipt, contract
    )
    if report.get("build_receipt") != expected_receipt_record:
        raise RuntimeError("published report build receipt binding changed")
    epochs = [
        report.get("integrity_reverification"),
        report.get("final_publish_reverification"),
    ]
    integrity_keys = {
        "pass",
        "verified_at_utc",
        "build_receipt",
        "receipt_id",
        "receipt_sha256",
        "source_snapshot",
        "contract",
        "samples",
        "finite_difference_logs",
        "controller_ledger",
        "output_inventory",
    }
    for index, epoch in enumerate(epochs):
        if (
            not isinstance(epoch, dict)
            or set(epoch)
            != (integrity_keys | ({"nvidia_driver"} if index == 1 else set()))
            or epoch.get("pass") is not True
            or not isinstance(epoch.get("verified_at_utc"), str)
            or not epoch["verified_at_utc"]
            or epoch.get("build_receipt") != expected_receipt_record
            or epoch.get("receipt_id") != receipt["receipt_id"]
            or epoch.get("receipt_sha256") != _sha256(receipt_path)
            or epoch.get("source_snapshot") != receipt["source_end"]
            or epoch.get("contract") != contract
        ):
            raise RuntimeError("published integrity epoch binding changed")
        _validate_prepublication_inventory(
            runner, output, run_id, epoch.get("output_inventory")
        )
    prepublication = epochs[1]["output_inventory"]
    if (
        epochs[0]["output_inventory"] != prepublication
        or marker["prepublication_output_inventory_sha256"]
        != prepublication["sha256"]
    ):
        raise RuntimeError("prepublication inventory epochs disagree")
    samples = _reconstruct_samples(runner, report, receipt, repo, output)
    fd = _validate_fd_oracle(runner, report, receipt, repo, output)
    expected_sample_records = _expected_integrity_sample_records(runner, samples)
    expected_fd_logs = _expected_integrity_fd_logs(fd)
    for epoch in epochs:
        if (
            epoch.get("samples") != expected_sample_records
            or epoch.get("finite_difference_logs") != expected_fd_logs
        ):
            raise RuntimeError("published integrity raw-file records changed")
    _validate_prepublication_closure(
        output, prepublication, expected_sample_records, expected_fd_logs
    )
    expected_ledger = runner._expected_controller_ledger(
        output=output, run_id=run_id, samples=samples, fd_oracle=fd
    )
    ledger_record = {
        "pass": True,
        "entries": len(expected_ledger["measurements"]),
        "sha256": runner.canonical_sha256(expected_ledger),
    }
    if any(epoch.get("controller_ledger") != ledger_record for epoch in epochs):
        raise RuntimeError("published controller ledger changed")
    recomputed_matrix = runner.validate_sample_matrix(samples)
    recomputed_comparison = runner.compare_measured_samples(samples, fd_oracle=fd)
    runner.bind_host_audit(
        recomputed_comparison,
        report.get("host_audit"),
        samples=samples,
        repo=repo,
        verify_live=True,
    )
    driver_binding = _validate_postprocessor(
        runner,
        report.get("postprocessor"),
        repo=repo,
        receipt_path=receipt_path,
        receipt=receipt,
        output=output,
        interpreter=interpreter,
        comparison=recomputed_comparison,
    )
    if epochs[1].get("nvidia_driver") != driver_binding:
        raise RuntimeError("final integrity driver binding changed")
    recomputed_comparison["gate"]["driver_runtime_device_binding"] = True
    recomputed_comparison["gate"]["pass"] = all(
        value
        for name, value in recomputed_comparison["gate"].items()
        if name != "pass"
    )
    if (
        report.get("sample_matrix") != recomputed_matrix
        or report.get("comparison") != recomputed_comparison
        or report.get("gate") != recomputed_comparison["gate"]
        or report.get("report_markdown")
        != {
            "path": str(output / "report.md"),
            "sha256": _sha256(output / "report.md"),
        }
        or (output / "report.md").read_text(encoding="utf-8")
        != runner.render_complete_markdown(recomputed_comparison, receipt["receipt_id"])
    ):
        raise RuntimeError("published numerical/performance report changed")
    performance = recomputed_comparison["performance"]
    comparisons = recomputed_comparison["comparisons"]
    return {
        "schema_version": 1,
        "state": "VERIFIED",
        "run_id": run_id,
        "receipt_id": receipt["receipt_id"],
        "receipt_python": str(interpreter),
        "raw_samples": len(samples),
        "terminal_files": len(
            marker["terminal_output_inventory"]["manifest"]["top_level_files"]
        ),
        "headline_metric": performance["headline_metric"],
        "cpu_over_cuda_single_median_speedup": performance["workload_wall"]
        ["comparisons"]["cpu_over_cuda_single"]["median_speedup"],
        "cpu_over_cuda_multi_median_speedup": performance["workload_wall"]
        ["comparisons"]["cpu_over_cuda_multi"]["median_speedup"],
        "cuda_single_over_cuda_multi_median_speedup": performance["workload_wall"]
        ["comparisons"]["cuda_single_over_cuda_multi"]["median_speedup"],
        "cpu_over_cuda_multi_process_median_speedup": performance[
            "fresh_process_wall"
        ]["comparisons"]["cpu_over_cuda_multi"]["median_speedup"],
        "maximum_gradient_absolute_error": max(
            item["gradient"]["max_absolute_error"] for item in comparisons
        ),
        "maximum_gradient_relative_l2_error": max(
            item["gradient"]["relative_l2_error"] for item in comparisons
        ),
        "maximum_objective_absolute_error": max(
            item["objective_absolute_error"] for item in comparisons
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = verify(args)
    except (Exception, SystemExit) as error:
        print(f"gpmeep MPI-adjoint evidence verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
