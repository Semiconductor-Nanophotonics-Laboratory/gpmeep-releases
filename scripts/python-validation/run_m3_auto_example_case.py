#!/usr/bin/env python3
"""Run one manifest example under a sealed single-rank automatic policy."""

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
import m3_mpi_rank_launcher as rank_launcher  # noqa: E402
import run_example_oracle as oracle  # noqa: E402
import run_m3_mpi_example_case as mpi_example  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
import run_validation as validation  # noqa: E402
from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    file_record,
)


SCHEMA = "gpmeep-m3-auto-example-case-v1"
COMPLETE_SCHEMA = "gpmeep-m3-auto-example-case-complete-v1"
LANE_NAME = "auto-fp32-1g"
MAX_JSON_BYTES = 512 * 1024**2
REPORT_KEYS = {
    "schema",
    "outcome",
    "case_path",
    "gpu_device",
    "receipt_id",
    "paths",
    "runtime_snapshot",
    "policy",
    "lane",
    "performance_evidence",
    "evidence_code",
}
POLICIES = {
    "python/examples/antenna_pec_ground_plane_1D.py": {
        "schema": "gpmeep-m3-auto-policy-v1",
        "active_backends": ["cpu"] * 8,
        "classification": "launch-dominated-small-domain-cpu-policy",
    },
    "python/examples/cherenkov-radiation.py": {
        "schema": "gpmeep-m3-auto-policy-v1",
        "active_backends": ["cpu"] * 4,
        "classification": "long-horizon-small-domain-cpu-policy",
    },
    "python/examples/edge_emitter_3D.py": {
        "schema": "gpmeep-m3-auto-policy-v1",
        "active_backends": ["cpu"] * 4,
        "classification": "production-crossover-bounded-validation-cpu-policy",
    },
    "python/examples/metasurface_lens.py": {
        "schema": "gpmeep-m3-auto-policy-v1",
        "active_backends": [*["cpu"] * 19, "cuda"],
        "classification": "small-cpu-large-cuda-crossover",
    },
    "python/examples/stochastic_emitter_line.py": {
        "schema": "gpmeep-m3-auto-policy-v1",
        "active_backends": ["cpu"] * 60,
        "classification": "long-horizon-small-domain-cpu-policy",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--manifest", type=pathlib.Path, default=SCRIPT_DIR / "manifest.json"
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--python", type=pathlib.Path, required=True)
    parser.add_argument("--build-python", type=pathlib.Path, required=True)
    parser.add_argument("--install-prefix", type=pathlib.Path, required=True)
    parser.add_argument("--build-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--mpiexec", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-device", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def policy_for(case_path: str) -> dict[str, Any]:
    try:
        policy = POLICIES[case_path]
    except KeyError as exc:
        raise WorkloadError(
            f"M3 automatic policy has no sealed case: {case_path}"
        ) from exc
    return json.loads(json.dumps(policy))


def _load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        if stable.initial_stat.st_size > MAX_JSON_BYTES:
            raise WorkloadError(f"{label} exceeds its JSON bound")
        with stable.file_object() as handle:
            try:
                value = json.load(
                    handle, object_pairs_hook=validation.unique_json_object
                )
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                validation.ManifestError,
            ) as exc:
                raise WorkloadError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} is not a JSON object")
    return value


def _prefixed_object(path: pathlib.Path, prefix: str, label: str) -> dict[str, Any]:
    encoded_prefix = prefix.encode("utf-8")
    with StableFile(path, label) as stable:
        if stable.initial_stat.st_size > mpi_example.MAX_STDOUT_BYTES:
            raise WorkloadError(f"{label} exceeds its text bound")
        with stable.file_object() as handle:
            lines = [line for line in handle if line.startswith(encoded_prefix)]
    if len(lines) != 1:
        raise WorkloadError(f"{label} must contain exactly one {prefix!r} record")
    try:
        value = json.loads(
            lines[0][len(encoded_prefix) :],
            object_pairs_hook=validation.unique_json_object,
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        validation.ManifestError,
    ) as exc:
        raise WorkloadError(f"{label} prefixed record is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkloadError(f"{label} prefixed record is not an object")
    return value


def audit_runs(
    stdout: pathlib.Path, requested_backend: str = "auto"
) -> list[dict[str, Any]]:
    if requested_backend not in {"cpu", "cuda", "auto"}:
        raise WorkloadError("M3 run audit requested backend differs")
    audit = _prefixed_object(stdout, oracle.AUDIT_PREFIX, "M3 auto audit")
    if set(audit) != {"runs"} or not isinstance(audit["runs"], list):
        raise WorkloadError("M3 auto audit schema differs")
    runs = audit["runs"]
    required = {
        "timestep",
        "timestep_delta",
        "meep_time",
        "run_wall_seconds",
        "field_energy",
        "dft_norm",
        "active_backend",
        "requested_backend",
        "execution_diagnostic",
        "runtime_counters",
        "run_phase_calls",
        "measurement_phase_calls",
    }
    for record in runs:
        if (
            not isinstance(record, dict)
            or set(record) != required
            or type(record["timestep"]) is not int
            or record["timestep"] < 0
            or type(record["timestep_delta"]) is not int
            or record["timestep_delta"] <= 0
            or isinstance(record["run_wall_seconds"], bool)
            or not isinstance(record["run_wall_seconds"], (int, float))
            or not 0 < float(record["run_wall_seconds"]) < 7 * 24 * 3600
            or any(
                isinstance(record[name], bool)
                or not isinstance(record[name], (int, float))
                or not math.isfinite(float(record[name]))
                for name in ("meep_time", "field_energy", "dft_norm")
            )
            or float(record["meep_time"]) < 0
            or record["requested_backend"] != requested_backend
            or not isinstance(record["execution_diagnostic"], str)
            or not record["execution_diagnostic"]
        ):
            raise WorkloadError("M3 auto run audit differs")
        runtime = record["runtime_counters"]
        expected_runtime_keys = {
            "runtime_availability_probes",
            "runtime_device_enumerations",
            "runtime_device_selections",
        }
        if (
            not isinstance(runtime, dict)
            or set(runtime) != expected_runtime_keys
            or any(type(value) is not int or value < 0 for value in runtime.values())
        ):
            raise WorkloadError("M3 auto run runtime audit differs")
        for name in ("run_phase_calls", "measurement_phase_calls"):
            calls = record[name]
            if (
                not isinstance(calls, dict)
                or set(calls) != {"cpu", "cuda"}
                or any(type(value) is not int or value < 0 for value in calls.values())
            ):
                raise WorkloadError(f"M3 auto {name} audit differs")
    return runs


def validate_policy_runs(
    runs: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    expected = policy["active_backends"]
    actual = [record["active_backend"] for record in runs]
    if actual != expected:
        raise WorkloadError(
            f"M3 automatic owner decisions differ: expected {expected}, got {actual}"
        )
    for index, record in enumerate(runs):
        active = actual[index]
        if active not in {"cpu", "cuda"}:
            raise WorkloadError("M3 auto selected an unsupported backend")
        other = "cuda" if active == "cpu" else "cpu"
        calls = record["run_phase_calls"]
        if (
            not isinstance(calls, dict)
            or set(calls) != {"cpu", "cuda"}
            or type(calls[active]) is not int
            or type(calls[other]) is not int
            or calls[active] <= 0
            or calls[other] != 0
        ):
            raise WorkloadError(
                f"M3 auto run {index} violated whole-owner dispatch"
            )
    return {
        "schema": policy["schema"],
        "classification": policy["classification"],
        "run_count": len(runs),
        "active_backends": actual,
        "cpu_run_count": actual.count("cpu"),
        "cuda_run_count": actual.count("cuda"),
        "run_wall_seconds": [float(record["run_wall_seconds"]) for record in runs],
    }


def _runtime_identity_problems(
    stats: dict[str, Any], runtime: dict[str, Any]
) -> list[str]:
    problems = []
    for name in ("python_executable", "meep_module", "extension", "libmeep"):
        expected = runtime.get(name)
        actual = stats.get(name)
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            problems.append(f"{name} runtime evidence is absent")
            continue
        if pathlib.Path(str(actual.get("path", ""))).resolve() != pathlib.Path(
            str(expected.get("path", ""))
        ).resolve() or actual.get("sha256") != expected.get("sha256"):
            problems.append(f"{name} runtime identity differs")
    return problems


def validate_rank_bundle(
    root: pathlib.Path,
    case_command: list[str],
    runtime: dict[str, Any],
    nonce: str,
    gpu: str,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity_path = root / "identity/rank-00000.json"
    statistics_path = root / "statistics/rank-00000.json"
    if sorted(path.name for path in (root / "identity").glob("rank-*.json")) != [
        identity_path.name
    ] or sorted(
        path.name for path in (root / "statistics").glob("rank-*.json")
    ) != [statistics_path.name]:
        raise WorkloadError("M3 auto rank evidence inventory differs")
    identity = _load_object(identity_path, "M3 auto rank identity")
    stats = _load_object(statistics_path, "M3 auto rank statistics")
    expected_identity_keys = {
        "schema",
        "rank",
        "world_size",
        "local_rank",
        "pid",
        "executable",
        "command",
        "run_nonce",
        "requested_backend",
        "visible_devices",
        "statistics",
        "work",
    }
    if (
        set(identity) != expected_identity_keys
        or identity.get("schema") != rank_launcher.SCHEMA
        or identity.get("rank") != 0
        or identity.get("world_size") != 1
        or identity.get("local_rank") != 0
        or type(identity.get("pid")) is not int
        or identity["pid"] <= 0
        or identity.get("command") != case_command
        or identity.get("executable") != case_command[0]
        or identity.get("run_nonce") != nonce
        or identity.get("requested_backend") != "auto"
        or identity.get("visible_devices") != gpu
        or identity.get("statistics") != str(statistics_path)
        or identity.get("work") != str(root / "work/rank-00000")
        or stats.get("pid") != identity["pid"]
    ):
        raise WorkloadError("M3 auto rank identity binding differs")
    expected_final = policy["active_backends"][-1]
    problems = _runtime_identity_problems(stats, runtime)
    if stats.get("capture_status") != "captured":
        problems.append("statistics capture did not complete")
    if stats.get("gpu_api_available") is not True:
        problems.append("GPU API is unavailable")
    if stats.get("run_nonce") != nonce or stats.get("build_receipt_id") != runtime.get(
        "receipt_id"
    ):
        problems.append("nonce or receipt identity differs")
    if (
        stats.get("single_precision") is not True
        or stats.get("with_mpi") is not True
        or stats.get("gpu_compiled") is not True
        or stats.get("strict_cuda_marker") is not False
        or stats.get("requested_backend") != "auto"
        or stats.get("active_backend") != expected_final
        or not isinstance(stats.get("statistics"), dict)
    ):
        problems.append("automatic backend runtime contract differs")
    else:
        problems.extend(validation.validate_phase_counters(stats))
    runs = audit_runs(root / "stdout.log")
    policy_result = validate_policy_runs(runs, policy)
    counters = stats.get("statistics", {})
    runtime_counters = counters.get("runtime") if isinstance(counters, dict) else None
    expected_runtime_keys = {
        "runtime_availability_probes",
        "runtime_device_enumerations",
        "runtime_device_selections",
    }
    if (
        not isinstance(runtime_counters, dict)
        or set(runtime_counters) != expected_runtime_keys
        or any(
            type(value) is not int or value < 0
            for value in runtime_counters.values()
        )
    ):
        problems.append("automatic backend runtime counters differ")
    if policy_result["cuda_run_count"] == 0:
        if validation.flatten_phase_counters(stats, "cuda_", strict_only=True) != 0:
            problems.append("automatic CPU policy recorded CUDA phase work")
        if runtime_counters != {
            "runtime_availability_probes": 0,
            "runtime_device_enumerations": 0,
            "runtime_device_selections": 0,
        }:
            problems.append("automatic CPU policy touched the CUDA runtime")
    else:
        selected = stats.get("selected_device_identifier")
        if (
            not isinstance(selected, str)
            or mpi_example._identity(selected) != mpi_example._identity(gpu)
            or not isinstance(runtime_counters, dict)
            or runtime_counters.get("runtime_availability_probes", 0) <= 0
            or runtime_counters.get("runtime_device_enumerations", 0) <= 0
            or runtime_counters.get("runtime_device_selections", 0) <= 0
            or validation.flatten_phase_counters(stats, "cuda_", strict_only=True)
            <= 0
            or validation.flatten_phase_counters(stats, "cpu_", strict_only=True)
            <= 0
        ):
            problems.append("automatic mixed policy GPU binding differs")
    if problems:
        raise WorkloadError("M3 auto backend contract failed: " + "; ".join(problems))
    return (
        [
            {
                "rank": 0,
                "identity": file_record(identity_path, root),
                "statistics": file_record(statistics_path, root),
                "pid": identity["pid"],
                "selected_device_identifier": stats.get(
                    "selected_device_identifier"
                ),
                "backend_contract": "PASS",
            }
        ],
        policy_result,
    )


def _terminal(output: pathlib.Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": file_record(output / "report.json", output),
        "case_path": report["case_path"],
        "receipt_id": report["receipt_id"],
    }


def publish_terminal(output: pathlib.Path, report: dict[str, Any]) -> None:
    atomic_write_json(output / "PENDING_COMPLETE", _terminal(output, report))
    verify_complete(output, "PENDING_COMPLETE")
    os.replace(output / "PENDING_COMPLETE", output / "COMPLETE")


def verify_complete(output: pathlib.Path, terminal: str = "COMPLETE") -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal not in {"COMPLETE", "PENDING_COMPLETE"}:
        raise WorkloadError("M3 auto terminal filename differs")
    marker = _load_object(output / terminal, "M3 auto COMPLETE")
    report = _load_object(output / "report.json", "M3 auto report")
    if (
        set(marker) != {"schema", "outcome", "report", "case_path", "receipt_id"}
        or set(report) != REPORT_KEYS
        or marker.get("schema") != COMPLETE_SCHEMA
        or marker.get("outcome") != "PASS"
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
        or marker != _terminal(output, report)
    ):
        raise WorkloadError("M3 auto terminal report binding differs")
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
    if paths != mpi_example._runtime_paths(namespace):
        raise WorkloadError("M3 auto retained runtime paths are not authoritative")
    expected_code = {
        "controller": absolute_file_record(pathlib.Path(__file__).resolve(), "controller"),
        "rank_launcher": absolute_file_record(paths["launcher"], "rank launcher"),
        "mpi_example_common": absolute_file_record(
            SCRIPT_DIR / "run_m3_mpi_example_case.py", "MPI example common"
        ),
        "validation_runner": absolute_file_record(
            SCRIPT_DIR / "run_validation.py", "validation runner"
        ),
        "example_oracle": absolute_file_record(
            SCRIPT_DIR / "run_example_oracle.py", "example oracle"
        ),
        "manifest": absolute_file_record(paths["manifest"], "manifest"),
    }
    if report["evidence_code"] != expected_code:
        raise WorkloadError("M3 auto evidence-code identity differs")
    case = mpi_example._case_from_manifest(
        paths["repo"], paths["manifest"], report["case_path"]
    )
    policy = policy_for(case["path"])
    if report["policy"] != policy:
        raise WorkloadError("M3 auto policy was not re-derived")
    snapshot = validation.validation_snapshot(
        paths["repo"], paths["build_python"], paths["python"]
    )
    unchanged, _problems = validation.compare_validation_snapshots(
        report["runtime_snapshot"], snapshot
    )
    if not snapshot.get("available") or not unchanged:
        raise WorkloadError("M3 auto live source/build snapshot differs")
    if snapshot["runtime_contract"].get("receipt_id") != report["receipt_id"]:
        raise WorkloadError("M3 auto live receipt differs")
    lane = report["lane"]
    expected_lane_keys = {
        "name",
        "requested_backend",
        "expected_final_backend",
        "ranks",
        "devices",
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
    if (
        not isinstance(lane, dict)
        or set(lane) != expected_lane_keys
        or lane["name"] != LANE_NAME
        or lane["requested_backend"] != "auto"
        or lane["expected_final_backend"] != policy["active_backends"][-1]
        or lane["ranks"] != 1
        or lane["devices"] != [report["gpu_device"]]
    ):
        raise WorkloadError("M3 auto retained lane schema differs")
    root = output / "lane"
    case_command = validation.case_command(case, paths["python"], paths["repo"])
    expected_lane = mpi_example.Lane(
        LANE_NAME, policy["active_backends"][-1], 1, (report["gpu_device"],)
    )
    if lane["command"] != mpi_example.lane_command(
        expected_lane,
        mpiexec=paths["mpiexec"],
        python=paths["python"],
        launcher=paths["launcher"],
        evidence_root=root,
        case_command=case_command,
    ):
        raise WorkloadError("M3 auto lane command differs")
    environment = mpi_example.clean_environment(
        expected_lane,
        python=paths["python"],
        build_python=paths["build_python"],
        runtime_contract=snapshot["runtime_contract"],
        run_nonce=lane["run_nonce"],
    )
    environment.update(
        {
            "MEEP_GPU_BACKEND": "auto",
            "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND": "auto",
        }
    )
    environment.pop("GPMEEP_VALIDATION_STRICT_CUDA", None)
    if lane["environment_sha256"] != validation.environment_sha256(environment):
        raise WorkloadError("M3 auto environment identity differs")
    process = lane["process"]
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
    ):
        raise WorkloadError("M3 auto process record differs")
    rank_records, policy_result = validate_rank_bundle(
        root,
        case_command,
        snapshot["runtime_contract"],
        lane["run_nonce"],
        report["gpu_device"],
        policy,
    )
    if lane["rank_records"] != rank_records or lane["policy_result"] != policy_result:
        raise WorkloadError("M3 auto rank or policy replay differs")
    for name in ("stdout", "stderr"):
        if lane[name] != file_record(root / f"{name}.log", root):
            raise WorkloadError(f"M3 auto {name} record differs")
    if (
        process["stdout_size_bytes"] != lane["stdout"]["size_bytes"]
        or process["stderr_size_bytes"] != lane["stderr"]["size_bytes"]
    ):
        raise WorkloadError("M3 auto process output sizes differ")
    tree_path = output.joinpath(
        *pathlib.PurePosixPath(lane["tree_manifest"]["path"]).parts
    )
    if (
        lane["tree_manifest"] != file_record(tree_path, output)
        or _load_object(tree_path, "M3 auto tree manifest")
        != mpi_example.derive_tree(root)
    ):
        raise WorkloadError("M3 auto raw tree differs")
    validation.extract_json_metrics(
        (root / "stdout.log").read_text(encoding="utf-8"), case["comparison"]
    )
    if report["performance_evidence"] != {
        "valid_for_speed_gate": False,
        "reason": "one automatic-policy correctness sample",
    }:
        raise WorkloadError("M3 auto timing classification differs")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 60 <= args.timeout_seconds <= 48 * 3600:
        raise WorkloadError("M3 auto timeout must be in [60s,48h]")
    output = pathlib.Path(os.path.abspath(args.output))
    if output.exists() or output.is_symlink():
        raise WorkloadError("M3 auto output must be a fresh absent path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    try:
        paths = mpi_example._runtime_paths(args)
        if output == paths["repo"] or paths["repo"] in output.parents:
            raise WorkloadError("M3 auto output overlaps the source tree")
        receipt = provenance.verify_build_receipt(
            paths["build_receipt"], paths["repo"]
        )
        if receipt.get("build_kind") != validation.EXPECTED_BUILD_KIND:
            raise WorkloadError("M3 auto receipt build kind differs")
        case = mpi_example._case_from_manifest(
            paths["repo"], paths["manifest"], args.case
        )
        policy = policy_for(case["path"])
        snapshot = validation.validation_snapshot(
            paths["repo"], paths["build_python"], paths["python"]
        )
        if not snapshot.get("available"):
            raise WorkloadError(
                "M3 auto runtime snapshot failed: "
                + "; ".join(snapshot["problems"])
            )
        if snapshot["runtime_contract"]["receipt_id"] != receipt["receipt_id"]:
            raise WorkloadError("M3 auto receipt and runtime snapshot differ")
        gpu = mpi_example.normalized_gpu(args.gpu_device)
        expected_final = policy["active_backends"][-1]
        lane = mpi_example.Lane(LANE_NAME, expected_final, 1, (gpu,))
        root = output / "lane"
        for name in ("statistics", "identity", "work"):
            (root / name).mkdir(parents=True, mode=0o700, exist_ok=False)
        nonce = uuid.uuid4().hex
        environment = mpi_example.clean_environment(
            lane,
            python=paths["python"],
            build_python=paths["build_python"],
            runtime_contract=snapshot["runtime_contract"],
            run_nonce=nonce,
        )
        environment.update(
            {
                "MEEP_GPU_BACKEND": "auto",
                "GPMEEP_VALIDATION_EXPECTED_REQUESTED_BACKEND": "auto",
            }
        )
        environment.pop("GPMEEP_VALIDATION_STRICT_CUDA", None)
        case_command = validation.case_command(case, paths["python"], paths["repo"])
        command = mpi_example.lane_command(
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
            stdout_limit=mpi_example.MAX_STDOUT_BYTES,
            stderr_limit=mpi_example.MAX_STDERR_BYTES,
            combined_limit=(
                mpi_example.MAX_STDOUT_BYTES + mpi_example.MAX_STDERR_BYTES
            ),
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
            raise WorkloadError("M3 automatic lane process failed")
        validation.extract_json_metrics(
            (root / "stdout.log").read_text(encoding="utf-8"), case["comparison"]
        )
        rank_records, policy_result = validate_rank_bundle(
            root,
            case_command,
            snapshot["runtime_contract"],
            nonce,
            gpu,
            policy,
        )
        lane_record = {
            "name": LANE_NAME,
            "requested_backend": "auto",
            "expected_final_backend": expected_final,
            "ranks": 1,
            "devices": [gpu],
            "run_nonce": nonce,
            "command": command,
            "environment_sha256": validation.environment_sha256(environment),
            "process": process,
            "stdout": file_record(root / "stdout.log", root),
            "stderr": file_record(root / "stderr.log", root),
            "rank_records": rank_records,
            "tree_manifest": mpi_example._tree_record(root, output, lane),
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
                "M3 auto source/build changed during the case: "
                + "; ".join(problems)
            )
        report = {
            "schema": SCHEMA,
            "outcome": "PASS",
            "case_path": case["path"],
            "gpu_device": gpu,
            "receipt_id": receipt["receipt_id"],
            "paths": {name: str(path) for name, path in paths.items()},
            "runtime_snapshot": snapshot,
            "policy": policy,
            "lane": lane_record,
            "performance_evidence": {
                "valid_for_speed_gate": False,
                "reason": "one automatic-policy correctness sample",
            },
            "evidence_code": {
                "controller": absolute_file_record(
                    pathlib.Path(__file__).resolve(), "controller"
                ),
                "rank_launcher": absolute_file_record(
                    paths["launcher"], "rank launcher"
                ),
                "mpi_example_common": absolute_file_record(
                    SCRIPT_DIR / "run_m3_mpi_example_case.py", "MPI example common"
                ),
                "validation_runner": absolute_file_record(
                    SCRIPT_DIR / "run_validation.py", "validation runner"
                ),
                "example_oracle": absolute_file_record(
                    SCRIPT_DIR / "run_example_oracle.py", "example oracle"
                ),
                "manifest": absolute_file_record(paths["manifest"], "manifest"),
            },
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
        print(f"M3 auto example case error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
