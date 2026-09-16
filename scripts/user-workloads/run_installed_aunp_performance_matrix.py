#!/usr/bin/env python3
"""Replay the sealed two-sample AuNP performance matrix from installed gpmeep."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common  # noqa: E402
import run_aunp_performance_workload as performance  # noqa: E402


SCHEMA = "gpmeep-installed-aunp-performance-matrix-v2"
COMPLETE_SCHEMA = "gpmeep-installed-aunp-performance-matrix-complete-v2"
PURPOSE = "installed-package replay of the already sealed M3 speed anchor"
DISCLOSURE = (
    "thermal telemetry remains sealed in the official M3 anchor; this replay "
    "is a package-regression gate and does not mint a new speed claim"
)
REPEATS = 2
MAXIMUM_SAMPLE_SPREAD = 1.20
MAXIMUM_ANCHOR_SLOWDOWN = 1.25
SEALED_M3_ANCHOR = {
    "schema": "gpmeep-sealed-m3-anchor-v1",
    "qualification_eligible": True,
    "source_commit": "adc67c3466774bbdb1875d996e4e0e50aeb7a845",
    "report_sha256": "50d2d8539000c76fb554318e2aa641577342e0f069ea5078d6f5ce596c5c36dc",
    "complete_sha256": (
        "7d3d522c4dbcce60e9c264acfa925c5f0ac0de18ff0540e8e9368835e45a4d2f"
    ),
    "audit_sha256": "4b8a257b2bcf9349f2bb9ccdd0dec172bb2d7542d078082a637394a479d75639",
    "means_seconds": {
        "cpu8": {
            "fdtd": 163.82354227802716,
            "end_to_end": 178.19756312202662,
        },
        "gpu1": {
            "fdtd": 16.680280592525378,
            "end_to_end": 41.04173981299391,
        },
        "gpu2": {
            "fdtd": 10.951151260989718,
            "end_to_end": 30.025476827519014,
        },
    },
}
LANES = (
    {"role": "gpu2", "backend": "cuda", "ranks": 2},
    {"role": "gpu1", "backend": "cuda", "ranks": 1},
    {"role": "cpu8", "backend": "cpu", "ranks": 8},
)


class MatrixError(RuntimeError):
    pass


def _anchor_is_valid(value: Any) -> bool:
    return common.typed_json_equal(value, SEALED_M3_ANCHOR)


def _load_json(handle: Any, label: str) -> Any:
    try:
        return json.load(
            handle,
            object_pairs_hook=common.unique_json_object,
            parse_constant=common.reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, common.WorkloadError) as error:
        raise MatrixError(f"installed AuNP {label} is invalid JSON") from error


def _lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_positive(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise MatrixError(f"installed AuNP {label} is not finite and positive")
    return float(value)


def _mean(values: list[float]) -> float:
    if len(values) != REPEATS or any(
        not math.isfinite(value) or value <= 0 for value in values
    ):
        raise MatrixError("installed AuNP matrix requires two finite positive samples")
    return sum(values) / len(values)


def derive_performance(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    expected_order = [
        (repeat, lane["role"]) for repeat in range(REPEATS) for lane in LANES
    ]
    if (
        not all(isinstance(task, dict) for task in tasks)
        or [(task.get("repeat"), task.get("role")) for task in tasks] != expected_order
    ):
        raise MatrixError("installed AuNP sample order or inventory differs")
    means: dict[str, dict[str, float]] = {}
    dispersion: dict[str, dict[str, float]] = {}
    for lane in LANES:
        role = lane["role"]
        selected = [task for task in tasks if task["role"] == role]
        means[role] = {}
        dispersion[role] = {}
        for published, label in (
            ("fdtd_wall_seconds", "fdtd"),
            ("workload_end_to_end_seconds", "end_to_end"),
        ):
            values = [
                _finite_positive(task.get(published), f"{role} {label} timing")
                for task in selected
            ]
            means[role][label] = _mean(values)
            dispersion[role][label] = max(values) / min(values)
            if dispersion[role][label] > MAXIMUM_SAMPLE_SPREAD:
                raise MatrixError(
                    f"installed AuNP {role} {label} sample spread is too large"
                )
    comparisons = {}
    for reference, candidate, minimum in (
        ("cpu8", "gpu1", 1.5),
        ("cpu8", "gpu2", 2.0),
        ("gpu1", "gpu2", 1.1),
    ):
        name = f"{reference}-to-{candidate}"
        comparisons[name] = {}
        for label in ("fdtd", "end_to_end"):
            speedup = means[reference][label] / means[candidate][label]
            if speedup < minimum:
                raise MatrixError(
                    f"installed AuNP {name} {label} speedup {speedup:.6f} "
                    f"is below {minimum:.6f}"
                )
            comparisons[name][label] = {
                "reference_mean_seconds": means[reference][label],
                "candidate_mean_seconds": means[candidate][label],
                "mean_speedup": speedup,
                "minimum": minimum,
                "outcome": "PASS",
            }
    anchor_replay = {}
    anchor_means = SEALED_M3_ANCHOR["means_seconds"]
    for role, values in means.items():
        anchor_replay[role] = {}
        for label, value in values.items():
            slowdown = value / anchor_means[role][label]
            if slowdown > MAXIMUM_ANCHOR_SLOWDOWN:
                raise MatrixError(
                    f"installed AuNP {role} {label} slowdown {slowdown:.6f} "
                    f"exceeds {MAXIMUM_ANCHOR_SLOWDOWN:.6f}"
                )
            anchor_replay[role][label] = {
                "sealed_mean_seconds": anchor_means[role][label],
                "installed_mean_seconds": value,
                "slowdown_ratio": slowdown,
                "maximum_slowdown": MAXIMUM_ANCHOR_SLOWDOWN,
                "outcome": "PASS",
            }
    return {
        "means_seconds": means,
        "sample_max_to_min": dispersion,
        "comparisons": comparisons,
        "sealed_anchor_replay": anchor_replay,
    }


def _safe_fresh_output(path: pathlib.Path) -> pathlib.Path:
    if not path.is_absolute():
        raise MatrixError("installed AuNP matrix output must be absolute")
    if path.exists() or path.is_symlink():
        raise MatrixError("installed AuNP matrix output must be absent")
    parent = path.parent
    if parent.resolve(strict=True) != parent or not parent.is_dir():
        raise MatrixError("installed AuNP matrix output parent is not canonical")
    path.mkdir(mode=0o755)
    return path


def _verified_relative_record(
    root: pathlib.Path,
    path: pathlib.Path,
    record: Any,
    expected_relative: str,
    label: str,
) -> pathlib.Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise MatrixError(f"installed AuNP {label} record schema differs")
    if record.get("path") != expected_relative:
        raise MatrixError(f"installed AuNP {label} record path differs")
    expected_path = root.joinpath(*pathlib.PurePosixPath(expected_relative).parts)
    if path != expected_path:
        raise MatrixError(f"installed AuNP {label} resolved path differs")
    with common.StableFile(path, f"installed AuNP {label}", expected=record):
        pass
    return path


def _verified_absolute_record(record: Any, label: str) -> pathlib.Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise MatrixError(f"installed AuNP {label} record schema differs")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not pathlib.Path(raw_path).is_absolute():
        raise MatrixError(f"installed AuNP {label} path is not absolute")
    path = pathlib.Path(raw_path)
    if path.resolve(strict=True) != path:
        raise MatrixError(f"installed AuNP {label} path is not canonical")
    with common.StableFile(path, f"installed AuNP {label}", expected=record):
        pass
    return path


def _matrix_file_inventory(output: pathlib.Path) -> set[str]:
    files = set()
    for directory, directory_names, file_names in os.walk(output, followlinks=False):
        current = pathlib.Path(directory)
        if any((current / name).is_symlink() for name in directory_names):
            raise MatrixError("installed AuNP matrix contains a directory symlink")
        for name in file_names:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise MatrixError("installed AuNP matrix contains a special file")
            files.add(path.relative_to(output).as_posix())
    return files


def _clean_environment(
    prefix: pathlib.Path,
    lane: dict[str, Any],
    devices: list[str],
) -> dict[str, str]:
    root = pathlib.Path(str(prefix) + ".micromamba-root")
    runtime_home = root / "home"
    if runtime_home.resolve(strict=True) != runtime_home or not runtime_home.is_dir():
        raise MatrixError("installed AuNP private runtime HOME is absent")
    environment = {
        "HOME": str(runtime_home),
        "PATH": f"{prefix}/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "MEEP_GPU_BACKEND": lane["backend"],
    }
    if lane["role"] == "cpu8":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif lane["role"] == "gpu1":
        environment["CUDA_VISIBLE_DEVICES"] = devices[0]
        environment["MEEP_GPU_DEVICE"] = "0"
    else:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        environment["MEEP_GPU_ALLOW_OVERSUBSCRIBE"] = "0"
        environment["MEEP_GPU_MPI_TRANSPORT"] = "pinned"
        environment["MEEP_GPU_MPI_COMPLETION"] = "waitsome"
    return environment


def _run_sample(
    *,
    repo: pathlib.Path,
    prefix: pathlib.Path,
    archive: pathlib.Path,
    attestation: pathlib.Path,
    source_commit: str,
    package_sha256: str,
    output: pathlib.Path,
    repeat: int,
    lane: dict[str, Any],
    devices: list[str],
) -> dict[str, Any]:
    role = lane["role"]
    relative = pathlib.Path("lanes") / role / f"repeat-{repeat:02d}"
    lane_output = output / relative
    log_path = output / "logs" / role / f"repeat-{repeat:02d}.log"
    lane_output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if lane_output.exists() or lane_output.is_symlink() or log_path.exists():
        raise MatrixError("installed AuNP sample output is not fresh")
    command = [
        str(prefix / "bin/mpiexec"),
        "--bind-to",
        "core",
        "--map-by",
        "core:PE=1",
        "--report-bindings",
        "-n",
        str(lane["ranks"]),
        str(prefix / "bin/python3.11"),
        "-u",
        str(repo / "scripts/user-workloads/run_aunp_performance_workload.py"),
        "--archive",
        str(archive),
        "--output",
        str(lane_output),
        "--package-provenance",
        str(attestation),
        "--expected-source-commit",
        source_commit,
        "--expected-package-sha256",
        package_sha256,
        "--expected-backend",
        lane["backend"],
    ]
    started = time.perf_counter()
    with log_path.open("xb") as log:
        completed = subprocess.run(
            command,
            cwd=output,
            env=_clean_environment(prefix, lane, devices),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=20 * 60,
        )
        log.flush()
        os.fsync(log.fileno())
    process_wall = time.perf_counter() - started
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise MatrixError(
            f"installed AuNP {role} repeat {repeat} failed with "
            f"{completed.returncode}:\n{tail}"
        )
    summary = performance.validate_performance_output(
        lane_output,
        expected_backend=lane["backend"],
        expected_mpi_size=lane["ranks"],
    )
    rank_records = summary["rank_records"]
    identifiers = sorted(
        {
            record["records"][0]["device_identifier"]
            for record in rank_records
            if record["records"][0]["device_identifier"] is not None
        }
    )
    if role == "gpu2" and len(identifiers) != 2:
        raise MatrixError("installed AuNP GPU2 ranks did not select distinct devices")
    cpu_cores = {
        (item["hostname"], item["package_id"], item["core_id"])
        for record in rank_records
        for item in record["physical_cores"]
    }
    if role == "cpu8" and len(cpu_cores) != 8:
        raise MatrixError("installed AuNP CPU8 did not bind eight physical cores")
    return {
        "repeat": repeat,
        "role": role,
        "backend": lane["backend"],
        "ranks": lane["ranks"],
        "device_identifiers": identifiers,
        "physical_core_count": len(cpu_cores),
        "fdtd_wall_seconds": float(summary["fdtd_wall_seconds"]),
        "workload_end_to_end_seconds": float(summary["workload_end_to_end_seconds"]),
        "launcher_wall_seconds": process_wall,
        "phase_wall_seconds": summary["phase_wall_seconds"],
        "phase_timestep_deltas": [
            record["timestep_delta"] for record in rank_records[0]["records"]
        ],
        "runtime_provenance_kind": summary["runtime_provenance_kind"],
        "runtime_provenance_id": summary["build_receipt"]["receipt_id"],
        "output": relative.as_posix(),
        "summary": common.file_record(lane_output / "summary.json", output),
        "complete": common.file_record(lane_output / "COMPLETE", output),
        "log": common.file_record(log_path, output),
    }


def run(args: argparse.Namespace) -> int:
    repo = SCRIPT_DIR.parents[1].resolve(strict=True)
    prefix = args.prefix.resolve(strict=True)
    if pathlib.Path(sys.prefix).resolve(strict=True) != prefix:
        raise MatrixError(
            "installed AuNP matrix must run with the installed prefix Python"
        )
    if not _lower_hex(args.source_commit, 40):
        raise MatrixError("installed AuNP source commit is invalid")
    if not _lower_hex(args.package_sha256, 64):
        raise MatrixError("installed AuNP package SHA-256 is invalid")
    archive = common.verify_file(
        args.archive, common.AUNP_ARCHIVE_SHA256, "installed AuNP archive"
    )
    package_report = common.validate_package_provenance_attestation(
        args.package_provenance,
        args.source_commit,
        args.package_sha256,
    )
    if pathlib.Path(package_report["prefix"]) != prefix:
        raise MatrixError("installed AuNP package attestation prefix differs")
    devices = [value.strip() for value in args.gpu_devices.split(",") if value.strip()]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise MatrixError("installed AuNP matrix requires two distinct GPU selectors")
    output = _safe_fresh_output(args.output)
    tasks = []
    for repeat in range(REPEATS):
        for lane in LANES:
            tasks.append(
                _run_sample(
                    repo=repo,
                    prefix=prefix,
                    archive=archive,
                    attestation=args.package_provenance.resolve(strict=True),
                    source_commit=args.source_commit,
                    package_sha256=args.package_sha256,
                    output=output,
                    repeat=repeat,
                    lane=lane,
                    devices=devices,
                )
            )
    derived = derive_performance(tasks)
    report = {
        "schema": SCHEMA,
        "state": "PASS",
        "valid_for_new_speed_claim": False,
        "purpose": PURPOSE,
        "source_commit": args.source_commit,
        "package_sha256": args.package_sha256,
        "package_provenance": common.absolute_file_record(
            args.package_provenance, "installed package provenance"
        ),
        "archive": common.absolute_file_record(archive, "installed AuNP archive"),
        "qualification_profile": common.aunp_qualification_profile(),
        "sealed_m3_anchor": SEALED_M3_ANCHOR,
        "controller": common.file_record(pathlib.Path(__file__), repo),
        "adapter": common.file_record(
            SCRIPT_DIR / "run_aunp_performance_workload.py", repo
        ),
        "common": common.file_record(SCRIPT_DIR / "common.py", repo),
        "repeats_per_topology": REPEATS,
        "task_order": [task["role"] for task in tasks],
        "tasks": tasks,
        "performance": derived,
        "disclosure": DISCLOSURE,
    }
    common.atomic_write_json(output / "report.json", report)
    common.atomic_write_json(
        output / "COMPLETE",
        {
            "schema": COMPLETE_SCHEMA,
            "outcome": "PASS",
            "report": common.file_record(output / "report.json", output),
        },
    )
    validate(output)
    print(
        "installed AuNP performance matrix PASS: "
        + json.dumps(derived["means_seconds"], sort_keys=True)
    )
    return 0


def validate(output: pathlib.Path) -> dict[str, Any]:
    output = pathlib.Path(output)
    if not output.is_absolute() or output.is_symlink():
        raise MatrixError("installed AuNP matrix path is not absolute and canonical")
    resolved_output = output.resolve(strict=True)
    if resolved_output != output or not output.is_dir():
        raise MatrixError("installed AuNP matrix path is not absolute and canonical")
    with common.StableFile(output / "COMPLETE", "installed AuNP COMPLETE") as stable:
        with stable.file_object() as handle:
            complete = _load_json(handle, "COMPLETE")
    if (
        not isinstance(complete, dict)
        or set(complete) != {"schema", "outcome", "report"}
        or complete.get("schema") != COMPLETE_SCHEMA
        or complete.get("outcome") != "PASS"
    ):
        raise MatrixError("installed AuNP terminal marker differs")
    _verified_relative_record(
        output,
        output / "report.json",
        complete.get("report"),
        "report.json",
        "report",
    )
    with common.StableFile(
        output / "report.json", "installed AuNP report", expected=complete.get("report")
    ) as stable:
        with stable.file_object() as handle:
            report = _load_json(handle, "report")
    report_keys = {
        "schema",
        "state",
        "valid_for_new_speed_claim",
        "purpose",
        "source_commit",
        "package_sha256",
        "package_provenance",
        "archive",
        "qualification_profile",
        "sealed_m3_anchor",
        "controller",
        "adapter",
        "common",
        "repeats_per_topology",
        "task_order",
        "tasks",
        "performance",
        "disclosure",
    }
    if (
        not isinstance(report, dict)
        or set(report) != report_keys
        or report.get("schema") != SCHEMA
        or report.get("state") != "PASS"
        or report.get("valid_for_new_speed_claim") is not False
        or report.get("purpose") != PURPOSE
        or report.get("disclosure") != DISCLOSURE
        or report.get("repeats_per_topology") != REPEATS
        or report.get("qualification_profile") != common.aunp_qualification_profile()
        or not _anchor_is_valid(report.get("sealed_m3_anchor"))
    ):
        raise MatrixError("installed AuNP report contract differs")
    source_commit = report.get("source_commit")
    package_sha256 = report.get("package_sha256")
    if not _lower_hex(source_commit, 40) or not _lower_hex(package_sha256, 64):
        raise MatrixError("installed AuNP report identity differs")
    package_provenance = _verified_absolute_record(
        report.get("package_provenance"), "package provenance"
    )
    package_report = common.validate_package_provenance_attestation(
        package_provenance, source_commit, package_sha256
    )
    archive = _verified_absolute_record(report.get("archive"), "archive")
    common.verify_file(archive, common.AUNP_ARCHIVE_SHA256, "installed AuNP archive")
    repo = SCRIPT_DIR.parents[1].resolve(strict=True)
    for name, path, relative in (
        (
            "controller",
            pathlib.Path(__file__).resolve(strict=True),
            "scripts/user-workloads/run_installed_aunp_performance_matrix.py",
        ),
        (
            "adapter",
            SCRIPT_DIR / "run_aunp_performance_workload.py",
            "scripts/user-workloads/run_aunp_performance_workload.py",
        ),
        ("common", SCRIPT_DIR / "common.py", "scripts/user-workloads/common.py"),
    ):
        _verified_relative_record(repo, path, report.get(name), relative, name)
    tasks = report.get("tasks")
    if not isinstance(tasks, list):
        raise MatrixError("installed AuNP report has no tasks")
    expected_pairs = [(repeat, lane) for repeat in range(REPEATS) for lane in LANES]
    expected_order = [lane["role"] for _, lane in expected_pairs]
    if report.get("task_order") != expected_order:
        raise MatrixError("installed AuNP report task order differs")
    task_keys = {
        "repeat",
        "role",
        "backend",
        "ranks",
        "device_identifiers",
        "physical_core_count",
        "fdtd_wall_seconds",
        "workload_end_to_end_seconds",
        "launcher_wall_seconds",
        "phase_wall_seconds",
        "phase_timestep_deltas",
        "runtime_provenance_kind",
        "runtime_provenance_id",
        "output",
        "summary",
        "complete",
        "log",
    }
    expected_files = {"COMPLETE", "report.json"}
    if len(tasks) != len(expected_pairs):
        raise MatrixError("installed AuNP task inventory differs")
    for task, (repeat, expected_lane) in zip(tasks, expected_pairs):
        if (
            not isinstance(task, dict)
            or set(task) != task_keys
            or type(task.get("repeat")) is not int
            or task.get("repeat") != repeat
            or task.get("role") != expected_lane["role"]
            or task.get("backend") != expected_lane["backend"]
            or type(task.get("ranks")) is not int
            or task.get("ranks") != expected_lane["ranks"]
        ):
            raise MatrixError("installed AuNP task contract differs")
        relative = f"lanes/{expected_lane['role']}/repeat-{repeat:02d}"
        if task.get("output") != relative:
            raise MatrixError("installed AuNP task output path differs")
        lane_output = output.joinpath(*pathlib.PurePosixPath(relative).parts)
        summary = performance.validate_performance_output(
            lane_output,
            expected_backend=expected_lane["backend"],
            expected_mpi_size=expected_lane["ranks"],
        )
        log_relative = f"logs/{expected_lane['role']}/repeat-{repeat:02d}.log"
        for name, path, record_relative in (
            ("summary", lane_output / "summary.json", f"{relative}/summary.json"),
            ("complete", lane_output / "COMPLETE", f"{relative}/COMPLETE"),
            ("log", output / log_relative, log_relative),
        ):
            _verified_relative_record(
                output, path, task.get(name), record_relative, f"task {name}"
            )
            expected_files.add(record_relative)
        rank_records = summary["rank_records"]
        identifiers = sorted(
            {
                record["records"][0]["device_identifier"]
                for record in rank_records
                if record["records"][0]["device_identifier"] is not None
            }
        )
        cpu_cores = {
            (item["hostname"], item["package_id"], item["core_id"])
            for record in rank_records
            for item in record["physical_cores"]
        }
        expected_device_count = (
            expected_lane["ranks"] if expected_lane["backend"] == "cuda" else 0
        )
        phase_timestep_deltas = [
            record["timestep_delta"] for record in rank_records[0]["records"]
        ]
        if (
            len(identifiers) != expected_device_count
            or task.get("device_identifiers") != identifiers
            or len(cpu_cores) != expected_lane["ranks"]
            or task.get("physical_core_count") != len(cpu_cores)
            or task.get("phase_wall_seconds") != summary["phase_wall_seconds"]
            or task.get("phase_timestep_deltas") != phase_timestep_deltas
            or task.get("fdtd_wall_seconds") != summary["fdtd_wall_seconds"]
            or task.get("workload_end_to_end_seconds")
            != summary["workload_end_to_end_seconds"]
            or task.get("runtime_provenance_kind") != "installed-package-v1"
            or task.get("runtime_provenance_kind") != summary["runtime_provenance_kind"]
            or task.get("runtime_provenance_id")
            != summary["build_receipt"]["receipt_id"]
            or task.get("runtime_provenance_id") != package_report["report_id"]
        ):
            raise MatrixError("installed AuNP task summary was not re-derived")
        _finite_positive(task.get("launcher_wall_seconds"), "launcher wall time")
        for member in performance.AUNP_MEMBER_SHA256:
            expected_files.add(f"{relative}/input/{member}")
    derived = derive_performance(tasks)
    if report.get("performance") != derived:
        raise MatrixError("installed AuNP performance aggregate was not re-derived")
    actual_files = _matrix_file_inventory(output)
    if actual_files != expected_files:
        raise MatrixError(
            "installed AuNP matrix file inventory differs: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="action", required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--archive", required=True, type=pathlib.Path)
    execute.add_argument("--output", required=True, type=pathlib.Path)
    execute.add_argument("--prefix", required=True, type=pathlib.Path)
    execute.add_argument("--package-provenance", required=True, type=pathlib.Path)
    execute.add_argument("--source-commit", required=True)
    execute.add_argument("--package-sha256", required=True)
    execute.add_argument("--gpu-devices", default="0,1")
    execute.set_defaults(function=run)
    replay = subparsers.add_parser("validate")
    replay.add_argument("--output", required=True, type=pathlib.Path)
    replay.set_defaults(function=lambda args: (validate(args.output), 0)[1])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        MatrixError,
        common.WorkloadError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"installed AuNP performance matrix FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
