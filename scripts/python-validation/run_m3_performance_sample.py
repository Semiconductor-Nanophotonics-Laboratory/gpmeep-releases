#!/usr/bin/env python3
"""Run and seal one M3 repeated-performance example/topology sample."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
import uuid
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DEFAULT = SCRIPT_DIR.parents[1]
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import gpmeep_provenance as provenance  # noqa: E402
import run_m3_auto_example_case as auto_case  # noqa: E402
import run_m3_mpi_example_case as mpi_case  # noqa: E402
import run_validation as validation  # noqa: E402
from common import (  # noqa: E402
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    file_record,
)


SCHEMA = "gpmeep-m3-performance-sample-v1"
COMPLETE_SCHEMA = "gpmeep-m3-performance-sample-complete-v1"
TOPOLOGIES = ("cpu8", "cpu1", "cuda1", "cuda2", "auto1")
SAMPLE_KINDS = ("warmup", "measured")
CASE_TIMING = {
    "python/examples/edge_emitter_3D.py": "sum-all-simulation-run-wall-seconds",
    "python/examples/metasurface_lens.py": (
        "run-index-18-small-and-19-large-wall-seconds"
    ),
    "python/examples/stochastic_emitter_line.py": (
        "sum-all-simulation-run-wall-seconds"
    ),
}
REPORT_KEYS = {
    "schema",
    "outcome",
    "case_path",
    "topology",
    "sample_kind",
    "cycle_index",
    "gpu_devices",
    "receipt_id",
    "paths",
    "runtime_snapshot",
    "lane",
    "timing",
    "evidence_code",
}
LANE_KEYS = {
    "spec",
    "requested_backend",
    "run_nonce",
    "command",
    "environment_sha256",
    "process",
    "stdout",
    "stderr",
    "rank_records",
    "tree_manifest",
    "policy_result",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--manifest", type=pathlib.Path, default=SCRIPT_DIR / "manifest.json"
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--topology", required=True, choices=TOPOLOGIES)
    parser.add_argument("--sample-kind", required=True, choices=SAMPLE_KINDS)
    parser.add_argument("--cycle-index", required=True, type=int)
    parser.add_argument("--python", required=True, type=pathlib.Path)
    parser.add_argument("--build-python", required=True, type=pathlib.Path)
    parser.add_argument("--install-prefix", required=True, type=pathlib.Path)
    parser.add_argument("--build-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--mpiexec", required=True, type=pathlib.Path)
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def validate_sample_identity(kind: str, cycle: int) -> None:
    if kind not in SAMPLE_KINDS or type(cycle) is not int:
        raise WorkloadError("M3 performance sample identity differs")
    if (kind == "warmup" and cycle != 0) or (kind == "measured" and cycle < 1):
        raise WorkloadError("M3 performance sample cycle differs")


def normalized_devices(value: str) -> tuple[str, str]:
    devices = tuple(
        mpi_case.normalized_gpu(item) for item in value.split(",") if item
    )
    identities = {mpi_case._identity(item) for item in devices}
    if len(devices) != 2 or len(identities) != 2:
        raise WorkloadError("M3 performance sample requires two distinct GPUs")
    return devices  # type: ignore[return-value]


def select_lane(
    topology: str,
    devices: tuple[str, str],
    case_path: str,
) -> tuple[mpi_case.Lane, str, dict[str, Any]]:
    strict_lanes = dict(
        zip(
            ("cpu8", "cuda1", "cuda2"),
            mpi_case.build_lanes(8, devices),
            strict=True,
        )
    )
    strict_lanes["cpu1"] = mpi_case.Lane("cpu-fp32-1r", "cpu", 1, ())
    if topology in strict_lanes:
        lane = strict_lanes[topology]
        policy = {
            "schema": "gpmeep-m3-forced-backend-policy-v1",
            "classification": "forced-backend-performance-sample",
            "active_backends": None,
        }
        return lane, lane.backend, policy
    if topology != "auto1":
        raise WorkloadError("M3 performance sample topology differs")
    policy = auto_case.policy_for(case_path)
    lane = mpi_case.Lane(
        auto_case.LANE_NAME,
        policy["active_backends"][-1],
        1,
        devices[:1],
    )
    return lane, "auto", policy


def sample_environment(
    lane: mpi_case.Lane,
    requested_backend: str,
    *,
    python: pathlib.Path,
    build_python: pathlib.Path,
    runtime_contract: dict[str, Any],
    run_nonce: str,
) -> dict[str, str]:
    environment = mpi_case.clean_environment(
        lane,
        python=python,
        build_python=build_python,
        runtime_contract=runtime_contract,
        run_nonce=run_nonce,
    )
    if requested_backend == "auto":
        environment.update(
            {
                "MEEP_GPU_BACKEND": "auto",
                "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND": "auto",
            }
        )
        environment.pop("GPMEEP_VALIDATION_STRICT_CUDA", None)
    return environment


def derive_timing(
    case_path: str,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        contract = CASE_TIMING[case_path]
    except KeyError as exc:
        raise WorkloadError(
            f"M3 performance sample case is not sealed: {case_path}"
        ) from exc
    values = [float(record["run_wall_seconds"]) for record in runs]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise WorkloadError("M3 performance sample timing values differ")
    if contract == "sum-all-simulation-run-wall-seconds":
        return {
            "contract": contract,
            "run_count": len(values),
            "run_wall_seconds": values,
            "primary_seconds": sum(values),
        }
    if contract == "run-index-18-small-and-19-large-wall-seconds":
        if len(values) != 20:
            raise WorkloadError("M3 metasurface timing requires exactly 20 runs")
        return {
            "contract": contract,
            "run_count": len(values),
            "run_wall_seconds": values,
            "small_run_index": 18,
            "small_seconds": values[18],
            "large_run_index": 19,
            "large_seconds": values[19],
            "primary_seconds": values[19],
        }
    raise WorkloadError("M3 performance sample timing contract differs")


def validate_policy(
    runs: list[dict[str, Any]],
    requested_backend: str,
    lane: mpi_case.Lane,
    policy: dict[str, Any],
) -> dict[str, Any]:
    if requested_backend == "auto":
        return auto_case.validate_policy_runs(runs, policy)
    forced = dict(policy)
    forced["active_backends"] = [lane.backend] * len(runs)
    return auto_case.validate_policy_runs(runs, forced)


def _runtime_paths(args: argparse.Namespace) -> dict[str, pathlib.Path]:
    return mpi_case._runtime_paths(args)


def _code_records(paths: dict[str, pathlib.Path]) -> dict[str, dict[str, Any]]:
    return {
        "sample_runner": absolute_file_record(
            pathlib.Path(__file__).resolve(), "sample runner"
        ),
        "automatic_common": absolute_file_record(
            SCRIPT_DIR / "run_m3_auto_example_case.py", "automatic common"
        ),
        "mpi_example_common": absolute_file_record(
            SCRIPT_DIR / "run_m3_mpi_example_case.py", "MPI example common"
        ),
        "rank_launcher": absolute_file_record(paths["launcher"], "rank launcher"),
        "example_oracle": absolute_file_record(
            SCRIPT_DIR / "run_example_oracle.py", "example oracle"
        ),
        "validation_runner": absolute_file_record(
            SCRIPT_DIR / "run_validation.py", "validation runner"
        ),
        "manifest": absolute_file_record(paths["manifest"], "manifest"),
    }


def _terminal(output: pathlib.Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": file_record(output / "report.json", output),
        "case_path": report["case_path"],
        "topology": report["topology"],
        "sample_kind": report["sample_kind"],
        "cycle_index": report["cycle_index"],
        "receipt_id": report["receipt_id"],
    }


def publish_terminal(output: pathlib.Path, report: dict[str, Any]) -> None:
    atomic_write_json(output / "PENDING_COMPLETE", _terminal(output, report))
    verify_complete(output, "PENDING_COMPLETE")
    os.replace(output / "PENDING_COMPLETE", output / "COMPLETE")


def _validate_process(process: Any) -> None:
    if (
        not isinstance(process, dict)
        or set(process)
        != {
            "exit_code",
            "timeout",
            "output_limit",
            "spawn_error",
            "stdout_size_bytes",
            "stderr_size_bytes",
            "duration_seconds",
        }
        or process.get("exit_code") != 0
        or process.get("timeout") is not False
        or process.get("output_limit") is not None
        or process.get("spawn_error") is not None
        or isinstance(process.get("duration_seconds"), bool)
        or not isinstance(process.get("duration_seconds"), (int, float))
        or not math.isfinite(float(process["duration_seconds"]))
        or process["duration_seconds"] <= 0
        or type(process.get("stdout_size_bytes")) is not int
        or process["stdout_size_bytes"] < 0
        or type(process.get("stderr_size_bytes")) is not int
        or process["stderr_size_bytes"] < 0
    ):
        raise WorkloadError("M3 performance sample process record differs")


def verify_complete(
    output: pathlib.Path, terminal_name: str = "COMPLETE"
) -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal_name not in {"COMPLETE", "PENDING_COMPLETE"}:
        raise WorkloadError("M3 performance sample terminal filename differs")
    terminal = auto_case._load_object(
        output / terminal_name, "M3 performance sample COMPLETE"
    )
    report = auto_case._load_object(
        output / "report.json", "M3 performance sample report"
    )
    if (
        set(terminal)
        != {
            "schema",
            "outcome",
            "report",
            "case_path",
            "topology",
            "sample_kind",
            "cycle_index",
            "receipt_id",
        }
        or terminal.get("schema") != COMPLETE_SCHEMA
        or terminal.get("outcome") != "PASS"
        or not isinstance(report, dict)
        or set(report) != REPORT_KEYS
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
        or terminal != _terminal(output, report)
    ):
        raise WorkloadError("M3 performance sample terminal binding differs")
    validate_sample_identity(report["sample_kind"], report["cycle_index"])
    if report["topology"] not in TOPOLOGIES:
        raise WorkloadError("M3 performance sample retained topology differs")
    paths = {
        name: pathlib.Path(value).resolve(strict=True)
        for name, value in report["paths"].items()
    }
    namespace = argparse.Namespace(
        repo=paths["repo"],
        manifest=paths["manifest"],
        python=paths["python"],
        build_python=paths["build_python"],
        install_prefix=paths["install_prefix"],
        build_receipt=paths["build_receipt"],
        mpiexec=paths["mpiexec"],
    )
    if paths != _runtime_paths(namespace):
        raise WorkloadError("M3 performance sample runtime paths differ")
    if report["evidence_code"] != _code_records(paths):
        raise WorkloadError("M3 performance sample evidence-code identity differs")
    case = mpi_case._case_from_manifest(
        paths["repo"], paths["manifest"], report["case_path"]
    )
    if case["path"] not in CASE_TIMING:
        raise WorkloadError("M3 performance sample retained case differs")
    snapshot = validation.validation_snapshot(
        paths["repo"], paths["build_python"], paths["python"]
    )
    unchanged, _problems = validation.compare_validation_snapshots(
        report["runtime_snapshot"], snapshot
    )
    if not snapshot.get("available") or not unchanged:
        raise WorkloadError("M3 performance sample live snapshot differs")
    if snapshot["runtime_contract"].get("receipt_id") != report["receipt_id"]:
        raise WorkloadError("M3 performance sample receipt identity differs")
    devices = normalized_devices(",".join(report["gpu_devices"]))
    if list(devices) != report["gpu_devices"]:
        raise WorkloadError("M3 performance sample GPU inventory differs")
    lane, requested, policy = select_lane(
        report["topology"], devices, case["path"]
    )
    retained = report["lane"]
    if (
        not isinstance(retained, dict)
        or set(retained) != LANE_KEYS
        or retained["spec"] != mpi_case.lane_spec(lane)
        or retained["requested_backend"] != requested
    ):
        raise WorkloadError("M3 performance sample lane schema differs")
    nonce = retained["run_nonce"]
    if (
        not isinstance(nonce, str)
        or len(nonce) != 32
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise WorkloadError("M3 performance sample nonce differs")
    root = output / "lane"
    case_command = validation.case_command(case, paths["python"], paths["repo"])
    command = mpi_case.lane_command(
        lane,
        mpiexec=paths["mpiexec"],
        python=paths["python"],
        launcher=paths["launcher"],
        evidence_root=root,
        case_command=case_command,
    )
    if retained["command"] != command:
        raise WorkloadError("M3 performance sample command differs")
    environment = sample_environment(
        lane,
        requested,
        python=paths["python"],
        build_python=paths["build_python"],
        runtime_contract=snapshot["runtime_contract"],
        run_nonce=nonce,
    )
    if retained["environment_sha256"] != validation.environment_sha256(environment):
        raise WorkloadError("M3 performance sample environment differs")
    _validate_process(retained["process"])
    if requested == "auto":
        rank_records, policy_result = auto_case.validate_rank_bundle(
            root,
            case_command,
            snapshot["runtime_contract"],
            nonce,
            devices[0],
            policy,
        )
    else:
        rank_records = mpi_case.validate_rank_bundle(
            lane,
            root,
            case,
            snapshot["runtime_contract"],
            nonce,
            case_command,
        )
        runs = auto_case.audit_runs(root / "stdout.log", requested)
        policy_result = validate_policy(runs, requested, lane, policy)
    if retained["rank_records"] != rank_records:
        raise WorkloadError("M3 performance sample rank replay differs")
    if requested == "auto":
        runs = auto_case.audit_runs(root / "stdout.log", requested)
    if retained["policy_result"] != policy_result:
        raise WorkloadError("M3 performance sample policy replay differs")
    if report["timing"] != derive_timing(case["path"], runs):
        raise WorkloadError("M3 performance sample timing replay differs")
    for name in ("stdout", "stderr"):
        if retained[name] != file_record(root / f"{name}.log", root):
            raise WorkloadError(f"M3 performance sample {name} changed")
    if (
        retained["process"]["stdout_size_bytes"]
        != retained["stdout"]["size_bytes"]
        or retained["process"]["stderr_size_bytes"]
        != retained["stderr"]["size_bytes"]
    ):
        raise WorkloadError("M3 performance sample output sizes differ")
    tree_path = output / "tree-manifest.json"
    if (
        retained["tree_manifest"] != file_record(tree_path, output)
        or auto_case._load_object(tree_path, "M3 performance sample tree")
        != mpi_case.derive_tree(root)
    ):
        raise WorkloadError("M3 performance sample raw tree differs")
    validation.extract_json_metrics(
        (root / "stdout.log").read_text(encoding="utf-8"), case["comparison"]
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_sample_identity(args.sample_kind, args.cycle_index)
    if not 60 <= args.timeout_seconds <= 48 * 3600:
        raise WorkloadError("M3 performance sample timeout must be in [60s,48h]")
    output = pathlib.Path(os.path.abspath(args.output))
    if output.exists() or output.is_symlink():
        raise WorkloadError("M3 performance sample output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    try:
        paths = _runtime_paths(args)
        if output == paths["repo"] or paths["repo"] in output.parents:
            raise WorkloadError("M3 performance sample output overlaps source")
        if paths["build_python"].parent / "build-provenance.json" != paths[
            "build_receipt"
        ]:
            raise WorkloadError("M3 performance sample build paths differ")
        receipt = provenance.verify_build_receipt(
            paths["build_receipt"], paths["repo"]
        )
        if receipt.get("build_kind") != validation.EXPECTED_BUILD_KIND:
            raise WorkloadError("M3 performance sample build kind differs")
        case = mpi_case._case_from_manifest(
            paths["repo"], paths["manifest"], args.case
        )
        if case["path"] not in CASE_TIMING:
            raise WorkloadError("M3 performance sample case is not sealed")
        snapshot = validation.validation_snapshot(
            paths["repo"], paths["build_python"], paths["python"]
        )
        if not snapshot.get("available"):
            raise WorkloadError(
                "M3 performance sample runtime snapshot failed: "
                + "; ".join(snapshot["problems"])
            )
        if snapshot["runtime_contract"].get("receipt_id") != receipt["receipt_id"]:
            raise WorkloadError("M3 performance sample receipt differs")
        devices = normalized_devices(args.gpu_devices)
        lane, requested, policy = select_lane(args.topology, devices, case["path"])
        root = output / "lane"
        for name in ("statistics", "identity", "work"):
            (root / name).mkdir(parents=True, mode=0o700, exist_ok=False)
        nonce = uuid.uuid4().hex
        environment = sample_environment(
            lane,
            requested,
            python=paths["python"],
            build_python=paths["build_python"],
            runtime_contract=snapshot["runtime_contract"],
            run_nonce=nonce,
        )
        case_command = validation.case_command(case, paths["python"], paths["repo"])
        command = mpi_case.lane_command(
            lane,
            mpiexec=paths["mpiexec"],
            python=paths["python"],
            launcher=paths["launcher"],
            evidence_root=root,
            case_command=case_command,
        )
        started = time.monotonic()
        process = validation.run_bounded_process(
            command,
            cwd=paths["repo"],
            env=environment,
            timeout_seconds=args.timeout_seconds,
            stdout_file=root / "stdout.log",
            stderr_file=root / "stderr.log",
            stdout_limit=mpi_case.MAX_STDOUT_BYTES,
            stderr_limit=mpi_case.MAX_STDERR_BYTES,
            combined_limit=mpi_case.MAX_STDOUT_BYTES + mpi_case.MAX_STDERR_BYTES,
        )
        process["duration_seconds"] = time.monotonic() - started
        if (
            process["exit_code"] != 0
            or process["timeout"]
            or process["output_limit"] is not None
            or process["spawn_error"] is not None
            or not validation.is_strict_utf8_file(root / "stdout.log")
            or not validation.is_strict_utf8_file(root / "stderr.log")
        ):
            raise WorkloadError("M3 performance sample process failed")
        validation.extract_json_metrics(
            (root / "stdout.log").read_text(encoding="utf-8"),
            case["comparison"],
        )
        if requested == "auto":
            rank_records, policy_result = auto_case.validate_rank_bundle(
                root,
                case_command,
                snapshot["runtime_contract"],
                nonce,
                devices[0],
                policy,
            )
        else:
            rank_records = mpi_case.validate_rank_bundle(
                lane,
                root,
                case,
                snapshot["runtime_contract"],
                nonce,
                case_command,
            )
            runs = auto_case.audit_runs(root / "stdout.log", requested)
            policy_result = validate_policy(runs, requested, lane, policy)
        if requested == "auto":
            runs = auto_case.audit_runs(root / "stdout.log", requested)
        timing = derive_timing(case["path"], runs)
        atomic_write_json(output / "tree-manifest.json", mpi_case.derive_tree(root))
        lane_record = {
            "spec": mpi_case.lane_spec(lane),
            "requested_backend": requested,
            "run_nonce": nonce,
            "command": command,
            "environment_sha256": validation.environment_sha256(environment),
            "process": process,
            "stdout": file_record(root / "stdout.log", root),
            "stderr": file_record(root / "stderr.log", root),
            "rank_records": rank_records,
            "tree_manifest": file_record(output / "tree-manifest.json", output),
            "policy_result": policy_result,
        }
        ending = validation.validation_snapshot(
            paths["repo"], paths["build_python"], paths["python"]
        )
        unchanged, problems = validation.compare_validation_snapshots(
            snapshot, ending
        )
        if not unchanged:
            raise WorkloadError(
                "M3 performance sample source/build changed: "
                + "; ".join(problems)
            )
        report = {
            "schema": SCHEMA,
            "outcome": "PASS",
            "case_path": case["path"],
            "topology": args.topology,
            "sample_kind": args.sample_kind,
            "cycle_index": args.cycle_index,
            "gpu_devices": list(devices),
            "receipt_id": receipt["receipt_id"],
            "paths": {name: str(path) for name, path in paths.items()},
            "runtime_snapshot": snapshot,
            "lane": lane_record,
            "timing": timing,
            "evidence_code": _code_records(paths),
        }
        atomic_write_json(output / "report.json", report)
        publish_terminal(output, report)
        return 0
    except Exception as exc:
        atomic_write_json(
            output / "FAILED.json",
            {
                "schema": SCHEMA,
                "outcome": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_unix_seconds": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, WorkloadError, ValueError, json.JSONDecodeError) as error:
        print(f"M3 performance sample error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
