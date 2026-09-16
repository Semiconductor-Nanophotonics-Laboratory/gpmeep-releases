#!/usr/bin/env python3
"""Run one manifest-owned case on CPU8, CUDA1, and CUDA2 MPI topologies.

This is a correctness unit, not a performance benchmark.  It reuses the
production Python manifest command and full JSON-metric tolerances while adding
rank-isolated statistics, strict per-rank CUDA/fallback checks, physical GPU
identity checks, and complete raw-tree sealing.  A higher-level resumable M3
controller owns retries of these fresh-output units.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import pathlib
import pwd
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_DEFAULT = SCRIPT_DIR.parents[1]
USER_WORKLOAD_DIR = SCRIPT_DIR.parent / "user-workloads"
for import_root in (SCRIPT_DIR, SCRIPT_DIR.parent, USER_WORKLOAD_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import gpmeep_provenance as provenance  # noqa: E402
import m3_mpi_rank_launcher as rank_launcher  # noqa: E402
import run_user_workload_matrix as matrix  # noqa: E402
import run_validation as validation  # noqa: E402
from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    absolute_file_record,
    atomic_write_json,
    file_record,
)


SCHEMA = "gpmeep-m3-mpi-example-case-v1"
COMPLETE_SCHEMA = "gpmeep-m3-mpi-example-case-complete-v1"
TREE_SCHEMA = "gpmeep-m3-mpi-example-tree-v1"
LANE_ORDER = ("cpu-fp32-8r", "cuda-fp32-1g", "cuda-fp32-2g")
MAX_TREE_FILES = 200_000
MAX_TREE_BYTES = 256 * 1024**3
MAX_STDOUT_BYTES = 256 * 1024**2
MAX_STDERR_BYTES = 128 * 1024**2
MAX_JSON_BYTES = 512 * 1024**2
PATH_KEYS = {
    "repo",
    "manifest",
    "python",
    "build_python",
    "install_prefix",
    "build_receipt",
    "mpiexec",
    "launcher",
}
REPORT_KEYS = {
    "schema",
    "outcome",
    "case_path",
    "cpu_ranks",
    "gpu_devices",
    "receipt_id",
    "paths",
    "runtime_snapshot",
    "lanes",
    "comparisons",
    "performance_evidence",
    "evidence_code",
}
LANE_RECORD_KEYS = {
    "lane",
    "run_nonce",
    "command",
    "environment_sha256",
    "process",
    "stdout",
    "stderr",
    "rank_records",
    "tree_manifest",
}


@dataclass(frozen=True)
class Lane:
    name: str
    backend: str
    ranks: int
    devices: tuple[str, ...]


def normalized_gpu(value: str) -> str:
    return matrix._normalized_uuid(value)


def build_lanes(cpu_ranks: int, gpu_devices: tuple[str, ...]) -> list[Lane]:
    if (
        isinstance(cpu_ranks, bool)
        or not isinstance(cpu_ranks, int)
        or cpu_ranks != 8
    ):
        raise WorkloadError("M3 MPI example CPU baseline must use exactly 8 ranks")
    devices = tuple(normalized_gpu(value) for value in gpu_devices)
    normalized = [value.lower().removeprefix("gpu-").replace("-", "") for value in devices]
    if len(devices) != 2 or len(set(normalized)) != 2:
        raise WorkloadError("M3 MPI example requires two distinct physical GPUs")
    return [
        Lane("cpu-fp32-8r", "cpu", 8, ()),
        Lane("cuda-fp32-1g", "cuda", 1, devices[:1]),
        Lane("cuda-fp32-2g", "cuda", 2, devices),
    ]


def lane_root(output: pathlib.Path, lane: Lane) -> pathlib.Path:
    return output / "lanes" / lane.name


def lane_spec(lane: Lane) -> dict[str, Any]:
    return {
        "name": lane.name,
        "backend": lane.backend,
        "ranks": lane.ranks,
        "devices": list(lane.devices),
    }


def lane_command(
    lane: Lane,
    *,
    mpiexec: pathlib.Path,
    python: pathlib.Path,
    launcher: pathlib.Path,
    evidence_root: pathlib.Path,
    case_command: list[str],
) -> list[str]:
    return [
        str(mpiexec),
        "--bind-to",
        "core",
        "--map-by",
        "core:PE=1",
        "--report-bindings",
        "-n",
        str(lane.ranks),
        str(python),
        str(launcher),
        "--evidence-root",
        str(evidence_root),
        "--expected-ranks",
        str(lane.ranks),
        "--",
        *case_command,
    ]


def clean_environment(
    lane: Lane,
    *,
    python: pathlib.Path,
    build_python: pathlib.Path,
    runtime_contract: dict[str, Any],
    run_nonce: str,
) -> dict[str, str]:
    try:
        account_home = pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError) as exc:
        raise WorkloadError(f"M3 MPI launch account has no home directory: {exc}") from exc
    if not account_home.is_absolute() or not account_home.is_dir():
        raise WorkloadError("M3 MPI launch account home directory is unavailable")
    account_home = account_home.resolve(strict=True)
    environment = validation.isolated_process_environment()
    environment.update(
        {
            # Open MPI resolves its MCA configuration before the rank launcher
            # can replace HOME with the rank-private runtime directory.  Keep
            # the real account home for mpiexec startup; m3_mpi_rank_launcher
            # still gives every rank a fresh isolated HOME before Python exec.
            "HOME": str(account_home),
            "PATH": f"{python.parent}:/usr/bin:/bin",
            "PYTHONPATH": os.pathsep.join((str(SCRIPT_DIR), str(build_python))),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "FONTCONFIG_FILE": str(runtime_contract["fontconfig_file"]["path"]),
            "CUDA_CACHE_DISABLE": "1",
            "JAX_PLATFORMS": "cpu",
            "MEEP_GPU_BACKEND": lane.backend,
            "GPMEEP_VALIDATION_EXPECTED_BACKEND": lane.backend,
            "GPMEEP_VALIDATION_RUN_NONCE": run_nonce,
            "GPMEEP_VALIDATION_BUILD_RECEIPT_ID": runtime_contract["receipt_id"],
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(lane.devices),
            "MEEP_GPU_MPI_TRANSPORT": "pinned",
            "MEEP_GPU_MPI_COMPLETION": "waitsome",
            "UCX_MEMTYPE_CACHE": "n",
        }
    )
    if lane.backend == "cuda":
        environment["GPMEEP_VALIDATION_STRICT_CUDA"] = "1"
    trusted = []
    for candidate in (
        pathlib.Path(runtime_contract["libmeep"]["path"]).resolve().parent,
        pathlib.Path(runtime_contract["installed_environment"]).resolve() / "lib",
    ):
        if candidate.is_dir() and str(candidate) not in trusted:
            trusted.append(str(candidate))
    if trusted:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(trusted)
    return environment


def _identity(value: str) -> str:
    return value.lower().removeprefix("gpu-").replace("-", "")


def _load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    with StableFile(path, label) as stable:
        if stable.initial_stat.st_size > MAX_JSON_BYTES:
            raise WorkloadError(f"{label} exceeds the JSON evidence bound")
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


def _bounded_counter(
    stats: dict[str, Any], counter: str, lane: Lane, rank: int
) -> int:
    value = validation.phase_counter(stats, counter)
    if (
        type(value) is not int
        or value < 0
        or value > validation.UINT64_MAX
    ):
        raise WorkloadError(
            f"{lane.name} rank {rank} counter {counter!r} is not a "
            "bounded unsigned integer"
        )
    return value


def _bounded_counter_sum(values: list[int], label: str) -> int:
    total = 0
    for value in values:
        if total > validation.UINT64_MAX - value:
            raise WorkloadError(f"{label} aggregate counter overflow")
        total += value
    return total


def validate_mpi_counter_contract(
    lane: Lane,
    case: dict[str, Any],
    rank_statistics: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate_counters = case.get("mpi_aggregate_cuda_call_counters", [])
    per_rank_counters = case.get("mpi_required_per_rank_call_counters", [])
    required_counters = case.get("required_cuda_call_counters", [])
    if (
        not isinstance(aggregate_counters, list)
        or not isinstance(required_counters, list)
        or any(
            type(counter) is not str
            or counter not in required_counters
            or counter not in validation.REQUIRED_CUDA_ACTIVITY_COUNTERS
            for counter in aggregate_counters
        )
        or len(set(aggregate_counters)) != len(aggregate_counters)
    ):
        raise WorkloadError(
            f"{lane.name} aggregate MPI CUDA counter declaration is invalid"
        )
    if (
        not isinstance(per_rank_counters, list)
        or any(
            type(counter) is not str
            or counter
            not in validation.REQUIRED_CUDA_SHARED_ACTIVITY_COUNTERS
            for counter in per_rank_counters
        )
        or len(set(per_rank_counters)) != len(per_rank_counters)
    ):
        raise WorkloadError(
            f"{lane.name} per-rank MPI counter declaration is invalid"
        )
    if aggregate_counters and not per_rank_counters:
        raise WorkloadError(
            f"{lane.name} aggregate MPI CUDA activity has no per-rank "
            "participation counter"
        )
    if len(rank_statistics) != lane.ranks:
        raise WorkloadError(f"{lane.name} MPI counter rank inventory differs")

    evidence: dict[str, Any] = {"aggregate": {}, "per_rank": {}}
    for counter in aggregate_counters:
        values = [
            _bounded_counter(stats, counter, lane, rank)
            for rank, stats in enumerate(rank_statistics)
        ]
        total = _bounded_counter_sum(values, f"{lane.name} {counter!r}")
        item: dict[str, Any] = {"values": values, "sum": total}
        cpu_counter = validation.REQUIRED_CUDA_CPU_COUNTER_PAIRS.get(counter)
        if lane.backend == "cuda":
            if total <= 0:
                raise WorkloadError(
                    f"{lane.name} required aggregate CUDA activity counter "
                    f"{counter!r} did not execute"
                )
            if cpu_counter is not None:
                cpu_values = [
                    _bounded_counter(stats, cpu_counter, lane, rank)
                    for rank, stats in enumerate(rank_statistics)
                ]
                if any(cpu_values):
                    raise WorkloadError(
                        f"{lane.name} aggregate CUDA phase {counter!r} "
                        "recorded CPU fallback"
                    )
                item["cpu_counter"] = cpu_counter
                item["cpu_values"] = cpu_values
        elif lane.backend == "cpu":
            if counter in validation.REQUIRED_CUDA_SHARED_ACTIVITY_COUNTERS:
                if total <= 0:
                    raise WorkloadError(
                        f"{lane.name} CPU reference did not execute shared "
                        f"aggregate activity {counter!r}"
                    )
            elif total != 0:
                raise WorkloadError(
                    f"{lane.name} CPU reference executed aggregate CUDA "
                    f"activity {counter!r}"
                )
            if cpu_counter is not None:
                cpu_values = [
                    _bounded_counter(stats, cpu_counter, lane, rank)
                    for rank, stats in enumerate(rank_statistics)
                ]
                cpu_total = _bounded_counter_sum(
                    cpu_values, f"{lane.name} {cpu_counter!r}"
                )
                if cpu_total <= 0:
                    raise WorkloadError(
                        f"{lane.name} CPU reference did not execute paired "
                        f"aggregate activity {cpu_counter!r}"
                    )
                item["cpu_counter"] = cpu_counter
                item["cpu_values"] = cpu_values
                item["cpu_sum"] = cpu_total
        else:
            raise WorkloadError(f"{lane.name} has unknown backend {lane.backend!r}")
        evidence["aggregate"][counter] = item

    for counter in per_rank_counters:
        values = [
            _bounded_counter(stats, counter, lane, rank)
            for rank, stats in enumerate(rank_statistics)
        ]
        missing = [rank for rank, value in enumerate(values) if value <= 0]
        if missing:
            raise WorkloadError(
                f"{lane.name} required per-rank MPI activity counter "
                f"{counter!r} did not execute on ranks {missing}"
            )
        evidence["per_rank"][counter] = values
    return evidence


def validate_rank_bundle(
    lane: Lane,
    root: pathlib.Path,
    case: dict[str, Any],
    runtime_contract: dict[str, Any],
    run_nonce: str,
    expected_case_command: list[str],
) -> list[dict[str, Any]]:
    identities = sorted((root / "identity").glob("rank-*.json"))
    statistics = sorted((root / "statistics").glob("rank-*.json"))
    expected_names = [f"rank-{rank:05d}.json" for rank in range(lane.ranks)]
    if [path.name for path in identities] != expected_names or [
        path.name for path in statistics
    ] != expected_names:
        raise WorkloadError(f"{lane.name} rank evidence inventory differs")

    records = []
    rank_statistics = []
    selected_identifiers = []
    aggregate_counters = set(
        case.get("mpi_aggregate_cuda_call_counters", [])
    )
    per_rank_case = dict(case)
    per_rank_case["required_cuda_call_counters"] = [
        counter
        for counter in case.get("required_cuda_call_counters", [])
        if counter not in aggregate_counters
    ]
    for rank, (identity_path, statistics_path) in enumerate(
        zip(identities, statistics, strict=True)
    ):
        identity = _load_object(identity_path, f"{lane.name} rank identity")
        stats = _load_object(statistics_path, f"{lane.name} rank statistics")
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
        expected_statistics = root / "statistics" / expected_names[rank]
        expected_work = root / "work" / f"rank-{rank:05d}"
        if (
            set(identity) != expected_identity_keys
            or identity.get("schema") != rank_launcher.SCHEMA
            or identity.get("rank") != rank
            or identity.get("world_size") != lane.ranks
            or identity.get("local_rank") != rank
            or type(identity.get("pid")) is not int
            or identity["pid"] <= 0
            or identity.get("command") != expected_case_command
            or identity.get("executable") != expected_case_command[0]
            or identity.get("run_nonce") != run_nonce
            or identity.get("requested_backend") != lane.backend
            or identity.get("visible_devices") != ",".join(lane.devices)
            or identity.get("statistics") != str(expected_statistics)
            or identity.get("work") != str(expected_work)
            or stats.get("pid") != identity["pid"]
        ):
            raise WorkloadError(f"{lane.name} rank {rank} identity binding differs")
        passed, problems = validation.verify_backend_contract(
            per_rank_case, lane.backend, stats, runtime_contract, run_nonce
        )
        if not passed:
            raise WorkloadError(
                f"{lane.name} rank {rank} backend contract failed: "
                + "; ".join(problems)
            )
        if lane.backend == "cuda":
            selected = stats.get("selected_device_identifier")
            if not isinstance(selected, str):
                raise WorkloadError(f"{lane.name} rank {rank} has no GPU identity")
            selected_identifiers.append(selected)
        rank_statistics.append(stats)
        records.append(
            {
                "rank": rank,
                "identity": file_record(identity_path, root),
                "statistics": file_record(statistics_path, root),
                "pid": identity["pid"],
                "selected_device_identifier": stats.get(
                    "selected_device_identifier"
                ),
                "backend_contract": "PASS",
            }
        )
    if lane.backend == "cuda" and (
        len({_identity(value) for value in selected_identifiers}) != lane.ranks
        or {_identity(value) for value in selected_identifiers}
        != {_identity(value) for value in lane.devices}
    ):
        raise WorkloadError(f"{lane.name} did not bind one distinct GPU per rank")
    validate_mpi_counter_contract(lane, case, rank_statistics)
    if case.get("kind") == "unittest":
        validate_unittest_mpi_output(lane, root, case)
    return records


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_tree(root: pathlib.Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files = []
    directories = []
    total = 0
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise WorkloadError(f"M3 MPI raw tree contains symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(relative)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkloadError(f"M3 MPI raw tree contains special file: {relative}")
        total += metadata.st_size
        if len(files) >= MAX_TREE_FILES or total > MAX_TREE_BYTES:
            raise WorkloadError("M3 MPI raw tree exceeds its evidence bound")
        files.append(
            {
                "path": relative,
                "size_bytes": metadata.st_size,
                "sha256": _sha256_file(path),
                "mode": stat.S_IMODE(metadata.st_mode),
            }
        )
    return {
        "schema": TREE_SCHEMA,
        "root": root.name,
        "directories": directories,
        "files": files,
        "file_count": len(files),
        "total_bytes": total,
    }


def _tree_record(root: pathlib.Path, output: pathlib.Path, lane: Lane) -> dict[str, Any]:
    tree = derive_tree(root)
    path = output / "tree-manifests" / f"{lane.name}.json"
    atomic_write_json(path, tree)
    return file_record(path, output)


def _case_from_manifest(
    repo: pathlib.Path, manifest: pathlib.Path, case_path: str
) -> dict[str, Any]:
    cases = validation.materialize_cases(validation.load_json(manifest), repo)
    matches = [
        case
        for case in cases
        if case.get("id") == case_path or case.get("path") == case_path
    ]
    if len(matches) != 1:
        raise WorkloadError("M3 MPI example case selection is not exact")
    case = matches[0]
    comparison_mode = case.get("comparison", {}).get("mode")
    is_json_example = case.get("kind") == "example" and comparison_mode == "json_metrics"
    is_embedded_unittest = (
        case.get("kind") == "unittest"
        and comparison_mode == "embedded_oracle"
        and isinstance(case.get("expected_unittest"), dict)
        and case.get("allowed_unittest_skips") == 0
        and case.get("allowed_unittest_skip_details", []) == []
    )
    if (
        case.get("disposition") != validation.RUN_DISPOSITION
        or case.get("compute_scope") != "fdtd_cuda"
        or case.get("gpu_contract") != "cuda_dispatch"
        or not (is_json_example or is_embedded_unittest)
    ):
        raise WorkloadError(
            "M3 MPI case requires a runnable FDTD JSON example or pinned "
            "zero-skip embedded-oracle unittest"
        )
    return case


def _bounded_utf8(path: pathlib.Path, label: str, maximum: int) -> str:
    with StableFile(path, label) as stable:
        if stable.initial_stat.st_size > maximum:
            raise WorkloadError(f"{label} exceeds its text bound")
        with stable.file_object() as handle:
            payload = handle.read()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkloadError(f"{label} is not strict UTF-8") from exc


def validate_unittest_mpi_output(
    lane: Lane, root: pathlib.Path, case: dict[str, Any]
) -> dict[str, Any]:
    """Require every cooperative rank to report the exact pinned test suite."""
    stdout = _bounded_utf8(root / "stdout.log", f"{lane.name} stdout", MAX_STDOUT_BYTES)
    stderr = _bounded_utf8(root / "stderr.log", f"{lane.name} stderr", MAX_STDERR_BYTES)
    combined = stdout + "\n" + stderr
    expected = case["expected_unittest"]
    expected_count = expected["count"]
    expected_identities = expected["identities"]
    reported_counts = validation.parse_unittest_reported_counts(combined)
    identities = validation.parse_unittest_test_identities(combined)
    terminal_summaries = [
        line.strip()
        for line in combined.splitlines()
        if line.strip() == "OK"
        or line.strip().startswith("OK (")
        or line.strip().startswith("FAILED (")
    ]
    expected_identity_counts = Counter(expected_identities)
    for identity in expected_identity_counts:
        expected_identity_counts[identity] *= lane.ranks
    if (
        len(reported_counts) != lane.ranks
        or any(value != expected_count for value in reported_counts)
        or Counter(identities) != expected_identity_counts
        or len(terminal_summaries) != lane.ranks
        or any(value != "OK" for value in terminal_summaries)
        or validation.parse_unittest_skips(combined) != 0
        or validation.parse_unittest_skip_details(combined) != []
    ):
        raise WorkloadError(
            f"{lane.name} did not report one exact zero-skip unittest suite per rank"
        )
    return {
        "output_copies": lane.ranks,
        "test_count_per_copy": expected_count,
        "identities_per_copy": list(expected_identities),
        "skips": 0,
        "outcome": "PASS",
    }


def derive_comparisons(
    output: pathlib.Path, case: dict[str, Any]
) -> dict[str, Any]:
    stdout = {
        name: output / "lanes" / name / "stdout.log" for name in LANE_ORDER
    }
    pairs = {
        "cpu8-vs-gpu1": (LANE_ORDER[0], LANE_ORDER[1]),
        "cpu8-vs-gpu2": (LANE_ORDER[0], LANE_ORDER[2]),
        "gpu1-vs-gpu2": (LANE_ORDER[1], LANE_ORDER[2]),
    }
    comparisons = {}
    if case["comparison"]["mode"] == "embedded_oracle":
        for name, (reference, candidate) in pairs.items():
            comparisons[name] = {
                "mode": "embedded_oracle",
                "outcome": "PASS",
                "failure_count": 0,
                "reference": reference,
                "candidate": candidate,
                "reason": (
                    "both MPI topologies passed the exact pinned embedded "
                    "unittest oracle on every rank"
                ),
            }
        return comparisons
    if case["comparison"]["mode"] != "json_metrics":
        raise WorkloadError("M3 MPI comparison mode differs")
    for name, (reference, candidate) in pairs.items():
        result = validation.compare_json_metric_files(
            stdout[reference],
            stdout[candidate],
            case["comparison"],
            temporary_parent=output / "comparison-work" / name,
        )
        if result.get("outcome") != "PASS" or result.get("failure_count") != 0:
            raise WorkloadError(f"M3 MPI example numerical comparison failed: {name}")
        comparisons[name] = result
    return comparisons


def authoritative_runtime_paths(repo: pathlib.Path) -> dict[str, pathlib.Path]:
    repo = repo.resolve(strict=True)
    return {
        "repo": REPO_DEFAULT.resolve(strict=True),
        "manifest": (SCRIPT_DIR / "manifest.json").resolve(strict=True),
        "launcher": (SCRIPT_DIR / "m3_mpi_rank_launcher.py").resolve(strict=True),
        "python": (
            repo / ".envs" / "meep-gpu-cuda-mpi" / "bin" / "python3.11"
        ).resolve(strict=True),
        "mpiexec": (
            repo / ".envs" / "meep-gpu-cuda-mpi" / "bin" / "mpirun"
        ).resolve(strict=True),
        "build_python": (
            repo / "build" / "meep-cuda-mpi-python-fp32" / "python"
        ).resolve(strict=True),
        "install_prefix": (
            repo / "install" / "meep-cuda-mpi-python-fp32"
        ).resolve(strict=True),
        "build_receipt": (
            repo
            / "build"
            / "meep-cuda-mpi-python-fp32"
            / "build-provenance.json"
        ).resolve(strict=True),
    }


def _runtime_paths(args: argparse.Namespace) -> dict[str, pathlib.Path]:
    values = {
        "repo": args.repo,
        "manifest": args.manifest,
        "python": args.python,
        "build_python": args.build_python,
        "install_prefix": args.install_prefix,
        "build_receipt": args.build_receipt,
        "mpiexec": args.mpiexec,
        "launcher": SCRIPT_DIR / "m3_mpi_rank_launcher.py",
    }
    resolved = {}
    directory_keys = {"repo", "build_python", "install_prefix"}
    for name, raw in values.items():
        provided = pathlib.Path(os.path.abspath(raw))
        try:
            metadata = provided.lstat()
        except OSError as exc:
            raise WorkloadError(f"M3 MPI runtime path is unavailable: {name}: {exc}") from exc
        path = provided.resolve(strict=True)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or path != provided
            or (name in directory_keys) != stat.S_ISDIR(metadata.st_mode)
        ):
            raise WorkloadError(f"M3 MPI runtime path type differs: {name}")
        if name not in directory_keys and not stat.S_ISREG(metadata.st_mode):
            raise WorkloadError(f"M3 MPI runtime file differs: {name}")
        if name in {"python", "mpiexec"} and not os.access(path, os.X_OK):
            raise WorkloadError(f"M3 MPI runtime executable is not executable: {name}")
        resolved[name] = path
    expected = authoritative_runtime_paths(resolved["repo"])
    if resolved != expected:
        raise WorkloadError("M3 MPI runtime is not the authoritative repository build")
    return resolved


def _terminal(output: pathlib.Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": COMPLETE_SCHEMA,
        "outcome": "PASS",
        "report": file_record(output / "report.json", output),
        "case_path": report["case_path"],
        "receipt_id": report["receipt_id"],
    }


def verify_complete(output: pathlib.Path, terminal: str = "COMPLETE") -> dict[str, Any]:
    output = output.resolve(strict=True)
    if terminal not in {"COMPLETE", "PENDING_COMPLETE"}:
        raise WorkloadError("M3 MPI terminal filename differs")
    marker = _load_object(output / terminal, "M3 MPI COMPLETE")
    if set(marker) != {"schema", "outcome", "report", "case_path", "receipt_id"}:
        raise WorkloadError("M3 MPI COMPLETE schema differs")
    if marker.get("report") != file_record(output / "report.json", output):
        raise WorkloadError("M3 MPI COMPLETE report record differs")
    report = _load_object(
        output / "report.json",
        "M3 MPI report",
    )
    if (
        set(report) != REPORT_KEYS
        or marker.get("schema") != COMPLETE_SCHEMA
        or marker.get("outcome") != "PASS"
        or report.get("schema") != SCHEMA
        or report.get("outcome") != "PASS"
        or marker != _terminal(output, report)
    ):
        raise WorkloadError("M3 MPI terminal report binding differs")
    if (
        not isinstance(report.get("paths"), dict)
        or set(report["paths"]) != PATH_KEYS
    ):
        raise WorkloadError("M3 MPI retained path inventory differs")
    paths = {
        name: pathlib.Path(value).resolve(strict=True)
        for name, value in report["paths"].items()
    }
    if paths != authoritative_runtime_paths(paths["repo"]):
        raise WorkloadError("M3 MPI retained runtime paths are not authoritative")
    expected_code = {
        "controller": absolute_file_record(pathlib.Path(__file__).resolve(), "controller"),
        "rank_launcher": absolute_file_record(paths["launcher"], "rank launcher"),
        "validation_runner": absolute_file_record(
            SCRIPT_DIR / "run_validation.py", "runner"
        ),
        "manifest": absolute_file_record(paths["manifest"], "manifest"),
    }
    if report.get("evidence_code") != expected_code:
        raise WorkloadError("M3 MPI evidence-code identity differs")
    case = _case_from_manifest(paths["repo"], paths["manifest"], report["case_path"])
    snapshot = validation.validation_snapshot(
        paths["repo"], paths["build_python"], paths["python"]
    )
    unchanged, _snapshot_problems = validation.compare_validation_snapshots(
        report["runtime_snapshot"], snapshot
    )
    if not snapshot.get("available") or not unchanged:
        raise WorkloadError("M3 MPI live source/build snapshot differs")
    if snapshot["runtime_contract"].get("receipt_id") != report["receipt_id"]:
        raise WorkloadError("M3 MPI live receipt identity differs")
    lanes = build_lanes(report["cpu_ranks"], tuple(report["gpu_devices"]))
    if not isinstance(report.get("lanes"), dict) or set(report["lanes"]) != set(
        LANE_ORDER
    ):
        raise WorkloadError("M3 MPI retained lane inventory differs")
    expected_case_command = validation.case_command(case, paths["python"], paths["repo"])
    for lane in lanes:
        retained = report["lanes"].get(lane.name)
        root = lane_root(output, lane)
        if not isinstance(retained, dict) or set(retained) != LANE_RECORD_KEYS:
            raise WorkloadError(f"{lane.name} retained lane schema differs")
        nonce = retained.get("run_nonce")
        if (
            not isinstance(nonce, str)
            or len(nonce) != 32
            or any(character not in "0123456789abcdef" for character in nonce)
        ):
            raise WorkloadError(f"{lane.name} retained nonce differs")
        if retained.get("lane") != lane_spec(lane):
            raise WorkloadError(f"{lane.name} retained lane specification differs")
        if retained.get("command") != lane_command(
            lane,
            mpiexec=paths["mpiexec"],
            python=paths["python"],
            launcher=paths["launcher"],
            evidence_root=root,
            case_command=expected_case_command,
        ):
            raise WorkloadError(f"{lane.name} retained command differs")
        process = retained.get("process")
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
            raise WorkloadError(f"{lane.name} retained process is not a PASS")
        expected_environment = clean_environment(
            lane,
            python=paths["python"],
            build_python=paths["build_python"],
            runtime_contract=snapshot["runtime_contract"],
            run_nonce=retained["run_nonce"],
        )
        if retained.get("environment_sha256") != validation.environment_sha256(
            expected_environment
        ):
            raise WorkloadError(f"{lane.name} retained environment differs")
        if retained.get("rank_records") != validate_rank_bundle(
            lane,
            root,
            case,
            snapshot["runtime_contract"],
            retained["run_nonce"],
            expected_case_command,
        ):
            raise WorkloadError(f"{lane.name} rank replay differs")
        tree_path = output.joinpath(
            *pathlib.PurePosixPath(retained["tree_manifest"]["path"]).parts
        )
        if file_record(tree_path, output) != retained["tree_manifest"]:
            raise WorkloadError(f"{lane.name} tree manifest file changed")
        if _load_object(tree_path, f"{lane.name} tree manifest") != derive_tree(root):
            raise WorkloadError(f"{lane.name} raw tree changed")
        for name in ("stdout", "stderr"):
            path = root / f"{name}.log"
            if file_record(path, root) != retained[name]:
                raise WorkloadError(f"{lane.name} {name} changed")
        if (
            process["stdout_size_bytes"] != retained["stdout"]["size_bytes"]
            or process["stderr_size_bytes"] != retained["stderr"]["size_bytes"]
        ):
            raise WorkloadError(f"{lane.name} process output sizes differ")
    if report.get("comparisons") != derive_comparisons(output, case):
        raise WorkloadError("M3 MPI comparisons were not fully replayed")
    if report.get("performance_evidence") != {
        "valid_for_speed_gate": False,
        "reason": "one cold correctness sample per MPI topology",
    }:
        raise WorkloadError("M3 MPI correctness timing classification differs")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=REPO_DEFAULT)
    parser.add_argument("--manifest", type=pathlib.Path, default=SCRIPT_DIR / "manifest.json")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--python", type=pathlib.Path, required=True)
    parser.add_argument("--build-python", type=pathlib.Path, required=True)
    parser.add_argument("--install-prefix", type=pathlib.Path, required=True)
    parser.add_argument("--build-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--mpiexec", type=pathlib.Path, required=True)
    parser.add_argument("--cpu-ranks", type=int, default=8)
    parser.add_argument("--gpu-devices", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 60 <= args.timeout_seconds <= 48 * 3600:
        raise WorkloadError("M3 MPI lane timeout must be in [60s,48h]")
    output = pathlib.Path(os.path.abspath(args.output))
    if output.exists() or output.is_symlink():
        raise WorkloadError("M3 MPI output must be a fresh absent path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    try:
        paths = _runtime_paths(args)
        if output == paths["repo"] or paths["repo"] in output.parents:
            raise WorkloadError("M3 MPI output overlaps the source tree")
        if paths["build_python"].parent / "build-provenance.json" != paths[
            "build_receipt"
        ]:
            raise WorkloadError("M3 MPI build Python and receipt prefixes differ")
        receipt = provenance.verify_build_receipt(paths["build_receipt"], paths["repo"])
        if receipt.get("build_kind") != validation.EXPECTED_BUILD_KIND:
            raise WorkloadError("M3 MPI receipt build kind differs")
        case = _case_from_manifest(paths["repo"], paths["manifest"], args.case)
        snapshot = validation.validation_snapshot(
            paths["repo"], paths["build_python"], paths["python"]
        )
        if not snapshot.get("available"):
            raise WorkloadError(
                "M3 MPI runtime snapshot failed: " + "; ".join(snapshot["problems"])
            )
        if snapshot["runtime_contract"]["receipt_id"] != receipt["receipt_id"]:
            raise WorkloadError("M3 MPI receipt and runtime snapshot differ")
        lanes = build_lanes(
            args.cpu_ranks,
            tuple(value for value in args.gpu_devices.split(",") if value),
        )
        case_command = validation.case_command(case, paths["python"], paths["repo"])
        lane_records = {}
        for lane in lanes:
            root = lane_root(output, lane)
            for name in ("statistics", "identity", "work"):
                (root / name).mkdir(parents=True, mode=0o700, exist_ok=False)
            run_nonce = uuid.uuid4().hex
            environment = clean_environment(
                lane,
                python=paths["python"],
                build_python=paths["build_python"],
                runtime_contract=snapshot["runtime_contract"],
                run_nonce=run_nonce,
            )
            command = lane_command(
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
                stdout_limit=MAX_STDOUT_BYTES,
                stderr_limit=MAX_STDERR_BYTES,
                combined_limit=MAX_STDOUT_BYTES + MAX_STDERR_BYTES,
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
                raise WorkloadError(f"M3 MPI lane process failed: {lane.name}")
            if case["comparison"]["mode"] == "json_metrics":
                validation.extract_json_metrics(
                    (root / "stdout.log").read_text(encoding="utf-8"),
                    case["comparison"],
                )
            rank_records = validate_rank_bundle(
                lane,
                root,
                case,
                snapshot["runtime_contract"],
                run_nonce,
                case_command,
            )
            lane_records[lane.name] = {
                "lane": lane_spec(lane),
                "run_nonce": run_nonce,
                "command": command,
                "environment_sha256": validation.environment_sha256(environment),
                "process": process,
                "stdout": file_record(root / "stdout.log", root),
                "stderr": file_record(root / "stderr.log", root),
                "rank_records": rank_records,
                "tree_manifest": _tree_record(root, output, lane),
            }
        comparisons = derive_comparisons(output, case)
        ending = validation.validation_snapshot(
            paths["repo"], paths["build_python"], paths["python"]
        )
        unchanged, snapshot_problems = validation.compare_validation_snapshots(
            snapshot, ending
        )
        if not unchanged:
            raise WorkloadError(
                "M3 MPI source/build changed during the case: "
                + "; ".join(snapshot_problems)
            )
        report = {
            "schema": SCHEMA,
            "outcome": "PASS",
            "case_path": case["path"],
            "cpu_ranks": args.cpu_ranks,
            "gpu_devices": list(
                [lane.devices for lane in lanes if lane.name == LANE_ORDER[2]][0]
            ),
            "receipt_id": receipt["receipt_id"],
            "paths": {name: str(path) for name, path in paths.items()},
            "runtime_snapshot": snapshot,
            "lanes": lane_records,
            "comparisons": comparisons,
            "performance_evidence": {
                "valid_for_speed_gate": False,
                "reason": "one cold correctness sample per MPI topology",
            },
            "evidence_code": {
                "controller": absolute_file_record(pathlib.Path(__file__).resolve(), "controller"),
                "rank_launcher": absolute_file_record(paths["launcher"], "rank launcher"),
                "validation_runner": absolute_file_record(SCRIPT_DIR / "run_validation.py", "runner"),
                "manifest": absolute_file_record(paths["manifest"], "manifest"),
            },
        }
        atomic_write_json(output / "report.json", report)
        atomic_write_json(output / "PENDING_COMPLETE", _terminal(output, report))
        verify_complete(output, "PENDING_COMPLETE")
        os.replace(output / "PENDING_COMPLETE", output / "COMPLETE")
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
        print(f"M3 MPI example case error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
