#!/usr/bin/env python3
"""Qualify retained FP32 CUDA Near2Far against fair CPU/MPI baselines."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import pathlib
import re
import shutil
import stat
import statistics
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gpmeep_provenance as provenance

SCHEMA = "gpmeep-near2far-mpi-qualification-v2"
RECORD_SCHEMA = "gpmeep-near2far-mpi-evidence-v2"
BUNDLE_SCHEMA = "gpmeep-near2far-mpi-evidence-bundle-v1"
BUNDLE_MANIFEST_NAME = "evidence-manifest.json"
RELEASE_ATTESTATION_SCHEMA = "gpmeep-near2far-release-attestation-v1"
RELEASE_VERIFICATION_SCOPE = "host-local-receipt-bound-v1"
MAX_COMPLETE_BYTES = 64 * 1024
MAX_BUNDLE_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_REPORT_BYTES = 128 * 1024 * 1024
MAX_REPORT_MARKDOWN_BYTES = 4 * 1024 * 1024
MAX_RELEASE_ATTESTATION_BYTES = 1024 * 1024
MAX_LOCAL_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_RELEASE_RAW_TOTAL_BYTES = 256 * 1024 * 1024
MAX_EVIDENCE_TARGETS = 1_000_000
MAX_EVIDENCE_FREQUENCIES = 16_384
MAX_EVIDENCE_REPETITIONS = 10_000
MAX_EVIDENCE_FIELDS = 1_000_000
MAX_EVIDENCE_FILE_BYTES = 128 * 1024 * 1024
RELEASE_SMALL_MINIMUM_SPEEDUP = 1.0
RELEASE_BATCH_MINIMUM_SPEEDUP = 1.5
RELEASE_BATCH_MINIMUM_TWO_GPU_SCALING = 1.1
RELEASE_BUILD_KIND = "cuda-mpi-python-fp32"
RELEASE_QUALIFICATION_CONTRACT = "gpmeep-cuda-mpi-python-fp32-v2"
RELEASE_COLLECTIVE_PROBES = (
    (
        "selection-optout-batch",
        "PASS: Near2Far batch optout mismatch fails closed on every MPI rank",
    ),
    (
        "selection-backend-grid",
        "PASS: Near2Far grid backend mismatch fails closed on every MPI rank",
    ),
    (
        "chunkless-rank-batch",
        "PASS: chunkless rank enters the public 13-double Near2Far collective "
        "and receives the nonempty rank result",
    ),
    (
        "metadata-point-count",
        "PASS: Near2Far point-count metadata mismatch fails closed before "
        "result collectives",
    ),
    (
        "metadata-null-points",
        "PASS: Near2Far null-points metadata mismatch fails closed before "
        "result collectives",
    ),
    (
        "metadata-target-coordinates",
        "PASS: Near2Far target-coordinates metadata mismatch fails closed "
        "before result collectives",
    ),
    (
        "metadata-frequency-count",
        "PASS: Near2Far frequency-count metadata mismatch fails closed before "
        "result collectives",
    ),
    (
        "metadata-frequency-values",
        "PASS: Near2Far frequency-values metadata mismatch fails closed before "
        "result collectives",
    ),
    (
        "metadata-material-periodic",
        "PASS: Near2Far material-periodic metadata mismatch fails closed before "
        "result collectives",
    ),
    (
        "metadata-monitor-dimension",
        "PASS: Near2Far monitor-dimension metadata mismatch fails closed before "
        "result collectives",
    ),
    (
        "metadata-greencyl-tol",
        "PASS: Near2Far greencyl-tol metadata mismatch fails closed before "
        "result collectives",
    ),
    (
        "metadata-grid-shape",
        "PASS: Near2Far grid-shape metadata mismatch fails closed before "
        "result collectives",
    ),
    (
        "metadata-grid-monitor-dimension",
        "PASS: Near2Far grid-monitor-dimension metadata mismatch fails closed "
        "before result collectives",
    ),
    (
        "metadata-grid-frequency-count",
        "PASS: Near2Far grid-frequency-count metadata mismatch fails closed "
        "before result collectives",
    ),
    (
        "cross-rank-cancellation",
        "PASS: Near2Far collective gate detects cross-rank cancellation",
    ),
    (
        "distributed-lowlevel-mixed",
        "PASS: distributed Near2Far lowlevel preserves rank-local ownership "
        "and forces mixed CUDA before explicit sum",
    ),
)

# Each measured scenario writes stdout, stderr, and one record per MPI rank
# for five lanes with rank counts 1, 1, 2, 1, and 2.  Every collective probe
# and the rank-local abort probe write a stdout/stderr pair.  Derive the
# release inventory from that checked contract so adding a mandatory probe
# cannot leave a stale hard-coded file count behind.
RELEASE_MEASURED_SCENARIO_COUNT = 2
RELEASE_MEASURED_LANE_RANK_COUNTS = (1, 1, 2, 1, 2)
RELEASE_RAW_FILE_COUNT = (
    RELEASE_MEASURED_SCENARIO_COUNT
    * sum(2 + ranks for ranks in RELEASE_MEASURED_LANE_RANK_COUNTS)
    + 2 * len(RELEASE_COLLECTIVE_PROBES)
    + 2
)


class QualificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def exact_json_equal(left: Any, right: Any) -> bool:
    """JSON equality that never aliases bool/int or int/float values."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            set(left) == set(right)
            and all(exact_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            exact_json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def parse_args() -> argparse.Namespace:
    repo = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        type=pathlib.Path,
        default=(repo / "build/meep-cuda-mpi-python-fp32/tests/gpu-step-db"),
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--verify-evidence",
        type=pathlib.Path,
        help="verify an already-published evidence bundle and exit",
    )
    parser.add_argument(
        "--expected-bundle-id",
        help="externally pinned 64-hex bundle identity required for release verification",
    )
    parser.add_argument(
        "--expected-qualification-tier",
        choices=("development", "release"),
        help="caller-authoritative evidence tier required in verification mode",
    )
    parser.add_argument(
        "--release-attestation",
        type=pathlib.Path,
        help="external receipt+bundle attestation required for release verification",
    )
    parser.add_argument(
        "--write-release-attestation",
        type=pathlib.Path,
        help="fresh external attestation output required when generating a release bundle",
    )
    parser.add_argument("--batch-targets", type=int, default=512)
    parser.add_argument("--frequency-count", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--small-minimum-speedup", type=float, default=1.0)
    parser.add_argument("--batch-minimum-speedup", type=float, default=1.5)
    parser.add_argument("--batch-minimum-two-gpu-scaling", type=float, default=1.1)
    parser.add_argument("--lane-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--stdout-limit-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--stderr-limit-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument(
        "--qualification-tier",
        choices=("development", "release"),
        default="development",
    )
    parser.add_argument("--build-receipt", type=pathlib.Path)
    return parser.parse_args()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    data = _bounded_regular_file_bytes(
        path, f"artifact {path}", MAX_LOCAL_ARTIFACT_BYTES
    )
    return {
        "path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _is_elf(path: pathlib.Path) -> bool:
    try:
        with path.open("rb") as source:
            return source.read(4) == b"\x7fELF"
    except OSError:
        return False


def resolve_runtime_executable(executable: pathlib.Path) -> dict[str, Any]:
    """Require and bind the exact ELF that every qualification lane executes."""
    executable = executable.resolve(strict=True)
    require(
        _is_elf(executable),
        "--executable must name the direct ELF; libtool wrappers are forbidden",
    )
    return {
        "execution_contract": "direct-elf-v1",
        "invoked": file_record(executable),
        "actual_elf": file_record(executable),
        "actual_elf_path": str(executable),
    }


def loaded_dependency_records(
    actual_elf: pathlib.Path, repo: pathlib.Path
) -> dict[str, Any]:
    ldd = pathlib.Path("/usr/bin/ldd")
    require(ldd.is_file(), "/usr/bin/ldd is unavailable")
    result = provenance.bounded_command(
        [str(ldd), str(actual_elf)],
        cwd=repo,
        timeout_seconds=30.0,
        stdout_limit=4 * 1024 * 1024,
        stderr_limit=1024 * 1024,
    )
    require(
        result["error"] is None
        and not result["timeout"]
        and result["output_limit"] is None
        and result["exit_code"] == 0,
        "cannot resolve runtime dependencies for the measured ELF",
    )
    text = result["stdout"].decode("utf-8", errors="strict")
    # ldd appends the process-specific ASLR load address to each resolved
    # row. Preserve every other byte of dependency identity while removing
    # only that nondeterministic suffix before the provenance comparison.
    normalized_rows = sorted(
        row
        for line in text.splitlines()
        if (
            row := re.sub(
                r"\s+\(0x[0-9a-fA-F]+\)\s*$", "", line
            ).strip()
        )
    )
    normalized_ldd = ("\n".join(normalized_rows) + "\n").encode("utf-8")
    resolved: dict[str, pathlib.Path] = {}
    absolute_runtime_files: set[pathlib.Path] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if "=>" not in stripped:
            first_token = stripped.split(" ", 1)[0]
            if first_token.startswith("/"):
                absolute_runtime_files.add(
                    pathlib.Path(first_token).resolve(strict=True)
                )
            continue
        name, remainder = stripped.split("=>", 1)
        value = remainder.strip().split(" ", 1)[0]
        require(value != "not", f"runtime dependency is unresolved: {name.strip()}")
        if value.startswith("/"):
            resolved[name.strip()] = pathlib.Path(value).resolve(strict=True)
    for prefix in ("libmeep.so", "libcudart.so", "libmpi.so", "libhdf5.so"):
        require(
            any(name.startswith(prefix) for name in resolved),
            f"measured ELF lacks resolved {prefix} dependency",
        )
    require(
        len(absolute_runtime_files) == 1,
        "ldd did not identify exactly one absolute dynamic interpreter row",
    )
    cache = provenance.StatHashCache()
    return {
        "ldd_command": [str(ldd), str(actual_elf)],
        "ldd_normalized_rows": normalized_rows,
        "ldd_normalized_sha256": hashlib.sha256(normalized_ldd).hexdigest(),
        "libraries": {
            name: provenance.file_record(path, repo, hash_cache=cache)
            for name, path in sorted(resolved.items())
        },
        "dynamic_interpreter": provenance.file_record(
            next(iter(absolute_runtime_files)), repo, hash_cache=cache
        ),
    }


def _record_path(record: dict[str, Any], repo: pathlib.Path) -> pathlib.Path:
    require(isinstance(record, dict), "artifact record is not an object")
    value = record.get("path")
    require(isinstance(value, str), "artifact record lacks a path")
    path = pathlib.Path(value)
    return (path if path.is_absolute() else repo / path).resolve(strict=True)


def require_same_file_record(
    actual: dict[str, Any],
    expected: dict[str, Any],
    repo: pathlib.Path,
    context: str,
) -> None:
    _validate_file_record_shape(actual, context)
    _validate_file_record_shape(expected, f"expected {context}")
    require(
        _record_path(actual, repo) == _record_path(expected, repo)
        and actual["size_bytes"] == expected["size_bytes"]
        and actual["sha256"] == expected["sha256"],
        f"{context} differs from the build receipt",
    )


def require_release_runtime_binding(
    runtime: dict[str, Any],
    loaded_dependencies: dict[str, Any],
    receipt: dict[str, Any],
    repo: pathlib.Path,
) -> dict[str, Any]:
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, dict), "build receipt artifacts are absent")
    executable_record = artifacts.get("gpu_step_db_test")
    libmeep_record = artifacts.get("libmeep")
    require(
        isinstance(executable_record, dict),
        "build receipt lacks artifacts.gpu_step_db_test",
    )
    require(
        isinstance(libmeep_record, dict),
        "build receipt lacks artifacts.libmeep",
    )
    require_same_file_record(
        runtime["actual_elf"], executable_record, repo, "measured Near2Far ELF"
    )
    loaded_libraries = loaded_dependencies.get("libraries")
    require(isinstance(loaded_libraries, dict), "loaded library records are absent")
    loaded_libmeep = [
        record
        for name, record in loaded_libraries.items()
        if name.startswith("libmeep.so")
    ]
    require(
        len(loaded_libmeep) == 1,
        "measured Near2Far ELF must load exactly one libmeep",
    )
    require_same_file_record(
        loaded_libmeep[0], libmeep_record, repo, "loaded libmeep"
    )
    return {
        "gpu_step_db_test": executable_record,
        "libmeep": libmeep_record,
    }


def require_release_receipt_contract(receipt: dict[str, Any]) -> None:
    """Reject a valid-but-unqualified receipt in every release path."""
    require(
        receipt.get("build_kind") == RELEASE_BUILD_KIND,
        "release receipt has the wrong build kind",
    )
    configuration = receipt.get("configuration")
    require(
        isinstance(configuration, dict)
        and configuration.get("qualification_contract")
        == RELEASE_QUALIFICATION_CONTRACT,
        "release receipt has the wrong qualification contract",
    )


def cpu_inventory() -> dict[str, Any]:
    model = "unknown"
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else []
    )
    return {
        "model": model,
        "logical_cpu_count": os.cpu_count(),
        "available_affinity_cpu_count": len(affinity),
        "available_affinity_cpus": affinity,
    }


def capture_runtime_provenance(
    executable: pathlib.Path,
    repo: pathlib.Path,
    build_receipt: pathlib.Path | None,
    qualification_tier: str,
) -> dict[str, Any]:
    require(
        qualification_tier != "release" or build_receipt is not None,
        "release qualification requires --build-receipt",
    )
    runtime = resolve_runtime_executable(executable)
    actual = pathlib.Path(runtime["actual_elf_path"])
    verified: dict[str, Any] | None = None
    receipt_path: pathlib.Path | None = None
    receipt: dict[str, Any] | None = None
    if build_receipt is not None:
        try:
            receipt_path = build_receipt
            if not receipt_path.is_absolute():
                receipt_path = repo / receipt_path
            receipt_path = receipt_path.resolve(strict=True)
            verified = provenance.verify_build_receipt(
                receipt_path, repo, verify_source=True
            )
            if qualification_tier == "release":
                require_release_receipt_contract(verified)
        except (OSError, RuntimeError) as error:
            raise QualificationError(
                f"build receipt verification failed: {error}"
            ) from error
    loaded_dependencies = loaded_dependency_records(actual, repo)
    if verified is not None and receipt_path is not None:
        bound_artifacts = require_release_runtime_binding(
            runtime, loaded_dependencies, verified, repo
        )
        receipt = {
            "file": file_record(receipt_path),
            "receipt_id": verified["receipt_id"],
            "build_input_id": verified["build_input_id"],
            "artifact_set_id": verified["artifact_set_id"],
            "bound_artifacts": bound_artifacts,
        }
    return {
        "qualification_tier": qualification_tier,
        "runtime": runtime,
        "loaded_dependencies": loaded_dependencies,
        "git_head": provenance.git_output(repo, "rev-parse", "HEAD").strip(),
        "git_status_porcelain": provenance.git_output(
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).splitlines(),
        "source_snapshot": provenance.source_snapshot(repo),
        "build_receipt": receipt,
    }


def normalize_uuid(value: str) -> str:
    return value.strip().lower().removeprefix("gpu-").replace("-", "")


def validate_gpu_inventory_records(devices: Any) -> None:
    keys = {
        "index", "uuid", "name", "driver", "compute_capability",
        "memory_mib", "memory_used_mib", "utilization_percent",
    }
    require(
        isinstance(devices, list)
        and len(devices) >= 2
        and all(isinstance(item, dict) and set(item) == keys
                for item in devices),
        "GPU inventory is invalid",
    )
    indexes: set[int] = set()
    uuids: set[str] = set()
    for item in devices:
        require(
            all(isinstance(item[key], str) for key in keys),
            "GPU inventory values must be exact nvidia-smi strings",
        )
        uuid = normalize_uuid(item["uuid"])
        require(re.fullmatch(r"[0-9a-f]{32}", uuid) is not None,
                "GPU inventory contains a malformed physical UUID")
        require(
            all(
                re.fullmatch(r"0|[1-9][0-9]*", item[key]) is not None
                for key in (
                    "index", "memory_mib", "memory_used_mib",
                    "utilization_percent",
                )
            ),
            "GPU inventory numeric field is invalid",
        )
        try:
            index = int(item["index"])
            memory = int(item["memory_mib"])
            used = int(item["memory_used_mib"])
            utilization = int(item["utilization_percent"])
        except (TypeError, ValueError) as error:
            raise QualificationError(
                f"GPU inventory numeric field is invalid: {error}"
            ) from error
        require(
            index >= 0 and memory > 0 and 0 <= used <= memory
            and 0 <= utilization <= 100
            and re.fullmatch(r"[0-9]+\.[0-9]+", item["compute_capability"])
            is not None
            and bool(item["name"].strip())
            and bool(item["driver"].strip()),
            "GPU inventory contains an implausible record",
        )
        require(index not in indexes and uuid not in uuids,
                "GPU inventory contains duplicate index or UUID")
        indexes.add(index)
        uuids.add(uuid)


def validate_cpu_inventory_record(cpu: Any, required_threads: int) -> None:
    require(
        isinstance(cpu, dict)
        and set(cpu) == {
            "model", "logical_cpu_count", "available_affinity_cpu_count",
            "available_affinity_cpus",
        }
        and isinstance(cpu["model"], str)
        and bool(cpu["model"].strip())
        and type(cpu["logical_cpu_count"]) is int
        and cpu["logical_cpu_count"] > 0
        and type(cpu["available_affinity_cpu_count"]) is int
        and isinstance(cpu["available_affinity_cpus"], list)
        and all(type(index) is int and index >= 0
                for index in cpu["available_affinity_cpus"])
        and cpu["available_affinity_cpu_count"]
        == len(cpu["available_affinity_cpus"])
        and len(set(cpu["available_affinity_cpus"]))
        == len(cpu["available_affinity_cpus"])
        and 0 < cpu["available_affinity_cpu_count"]
        <= cpu["logical_cpu_count"]
        and max(cpu["available_affinity_cpus"], default=-1)
        < cpu["logical_cpu_count"]
        and cpu["available_affinity_cpu_count"] >= required_threads,
        "CPU inventory cannot support its CPU baseline",
    )


def inventory_gpus() -> list[dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    require(executable is not None, "nvidia-smi is unavailable")
    process = provenance.bounded_command(
        [
            executable,
            "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total,"
            "memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        timeout_seconds=30.0,
        stdout_limit=1024 * 1024,
        stderr_limit=1024 * 1024,
    )
    require(
        process["error"] is None
        and not process["timeout"]
        and process["output_limit"] is None
        and process["exit_code"] == 0,
        "NVIDIA driver/runtime inventory failed: "
        + process["stderr"].decode("utf-8", errors="replace").strip(),
    )
    devices = []
    for line in process["stdout"].decode("utf-8", errors="strict").splitlines():
        columns = [value.strip() for value in line.split(",")]
        require(len(columns) == 8, "unexpected nvidia-smi inventory row")
        devices.append(
            dict(
                zip(
                    (
                        "index", "uuid", "name", "driver",
                        "compute_capability", "memory_mib",
                        "memory_used_mib", "utilization_percent",
                    ),
                    columns,
                )
            )
        )
    validate_gpu_inventory_records(devices)
    return devices


def parse_record(path: pathlib.Path) -> dict[str, Any]:
    try:
        evidence_text = _bounded_regular_file_bytes(
            path, f"rank evidence {path}", MAX_EVIDENCE_FILE_BYTES
        ).decode("utf-8")
    except UnicodeDecodeError as error:
        raise QualificationError(
            f"rank evidence is not UTF-8 in {path}: {error}"
        ) from error
    values: dict[str, str] = {}
    fields: dict[int, dict[str, float]] = {}
    elapsed: dict[int, float] = {}
    for line in evidence_text.splitlines():
        require("=" in line, f"malformed evidence line in {path}")
        key, value = line.split("=", 1)
        require(key not in values, f"duplicate evidence key {key!r}")
        values[key] = value
        if key.startswith("elapsed_"):
            elapsed[int(key.removeprefix("elapsed_"))] = float.fromhex(value)
        elif key.startswith("field_"):
            stem, part = key.rsplit("_", 1)
            index = int(stem.removeprefix("field_"))
            fields.setdefault(index, {})[part] = float.fromhex(value)
    require(values.get("schema") == RECORD_SCHEMA, f"wrong evidence schema in {path}")
    integer_names = {
        "rank", "world_size", "selected_device", "local_chunk_count",
        "local_source_points", "target_count", "frequency_count", "repetitions",
        "live_buffers_before", "live_buffers_after", "cpu_transform_calls",
        "cuda_transform_calls", "cuda_submitted_chunks", "cuda_source_points",
        "cuda_output_points", "cuda_frequencies", "cuda_periodic_copies",
        "cuda_fast_precision_calls", "cuda_mixed_precision_calls",
        "cuda_cancellation_retries",
        "cuda_target_tiles", "cuda_frequency_tiles", "cuda_operation_tiles",
        "cuda_maximum_workspace_bytes",
        "cuda_descriptor_uploads", "cuda_kernel_launches", "mpi_allreduce_calls",
        "cuda_result_device_to_host_bytes",
        "cuda_condition_device_to_host_bytes",
        "dft_device_to_host_bytes_avoided",
        "mpi_allreduce_bytes", "cpu_curl_calls", "cuda_curl_calls",
    }
    try:
        repetitions = int(values["repetitions"])
        targets = int(values["target_count"])
        frequencies = int(values["frequency_count"])
    except (KeyError, ValueError) as error:
        raise QualificationError(
            f"evidence dimensions are invalid in {path}: {error}"
        ) from error
    require(
        1 <= repetitions <= MAX_EVIDENCE_REPETITIONS
        and 1 <= targets <= MAX_EVIDENCE_TARGETS
        and 1 <= frequencies <= MAX_EVIDENCE_FREQUENCIES
        and targets * frequencies * 6 <= MAX_EVIDENCE_FIELDS,
        f"evidence dimensions exceed verifier bounds in {path}",
    )
    require(
        len(elapsed) == repetitions
        and min(elapsed, default=-1) == 0
        and max(elapsed, default=-1) == repetitions - 1
        and len(set(elapsed)) == repetitions,
        f"timing index set differs in {path}",
    )
    expected_field_count = targets * frequencies * 6
    require(
        len(fields) == expected_field_count
        and min(fields, default=-1) == 0
        and max(fields, default=-1) == expected_field_count - 1
        and all(set(parts) == {"real", "imag"} for parts in fields.values()),
        f"field index/component set differs in {path}",
    )
    require(
        len(values) == 3 + len(integer_names) + repetitions + 2 * expected_field_count,
        f"evidence key inventory differs in {path}",
    )
    record: dict[str, Any] = {
        name: int(values[name]) for name in integer_names
    }
    record.update(
        backend=values["backend"],
        device_identifier=values["device_identifier"],
        elapsed=[elapsed[index] for index in sorted(elapsed)],
        fields=[
            [fields[index]["real"], fields[index]["imag"]]
            for index in sorted(fields)
        ],
        artifact=file_record(path),
    )
    require(
        all(math.isfinite(value) and value > 0 for value in record["elapsed"]),
        f"non-finite/nonpositive timing in {path}",
    )
    require(
        all(math.isfinite(part) for value in record["fields"] for part in value),
        f"non-finite field in {path}",
    )
    return record


def symmetric_error(reference: list[list[float]], candidate: list[list[float]]) -> dict[str, float]:
    require(len(reference) == len(candidate), "field-vector lengths differ")
    error2 = 0.0
    reference2 = 0.0
    maximum_error = 0.0
    maximum_reference = 0.0
    for ref, got in zip(reference, candidate):
        for ref_part, got_part in zip(ref, got):
            difference = got_part - ref_part
            error2 += difference * difference
            reference2 += ref_part * ref_part
            maximum_error = max(maximum_error, abs(difference))
            maximum_reference = max(maximum_reference, abs(ref_part))
    return {
        "nrmse": math.sqrt(error2 / max(reference2, 1e-300)),
        "normalized_max": maximum_error / max(maximum_reference, 1e-300),
    }


def validate_numeric_agreement(
    scenario_name: str,
    cpu_reference: list[list[float]],
    best_cpu: list[list[float]],
    one_gpu: list[list[float]],
    two_gpu: list[list[float]],
) -> dict[str, dict[str, float]]:
    errors = {
        "cpu_lane_error": symmetric_error(cpu_reference, best_cpu),
        "one_gpu_error": symmetric_error(cpu_reference, one_gpu),
        "two_gpu_cpu_error": symmetric_error(cpu_reference, two_gpu),
        "one_to_two_gpu_error": symmetric_error(one_gpu, two_gpu),
    }
    require(
        errors["cpu_lane_error"]["nrmse"] <= 2e-5
        and errors["cpu_lane_error"]["normalized_max"] <= 2e-4,
        f"{scenario_name} MPI/OMP CPU result differs",
    )
    for label, key in (
        ("1-GPU result differs from CPU", "one_gpu_error"),
        ("2-GPU result differs from CPU", "two_gpu_cpu_error"),
        ("2-GPU result differs from 1-GPU", "one_to_two_gpu_error"),
    ):
        require(
            errors[key]["nrmse"] <= 5e-4
            and errors[key]["normalized_max"] <= 3e-3,
            f"{scenario_name} {label}",
        )
    return errors


def validate_release_speed_gates(args: argparse.Namespace) -> None:
    if args.qualification_tier != "release":
        return
    floors = (
        (
            "--small-minimum-speedup",
            args.small_minimum_speedup,
            RELEASE_SMALL_MINIMUM_SPEEDUP,
        ),
        (
            "--batch-minimum-speedup",
            args.batch_minimum_speedup,
            RELEASE_BATCH_MINIMUM_SPEEDUP,
        ),
        (
            "--batch-minimum-two-gpu-scaling",
            args.batch_minimum_two_gpu_scaling,
            RELEASE_BATCH_MINIMUM_TWO_GPU_SCALING,
        ),
    )
    for name, value, floor in floors:
        require(value >= floor, f"release {name} cannot be lower than {floor:g}")


def validate_gpu_identity(
    scenario_name: str,
    one_gpu: dict[str, Any],
    two_gpu: dict[str, Any],
    devices: list[dict[str, str]],
) -> None:
    inventory_uuids = {normalize_uuid(item["uuid"]) for item in devices}
    inventory_by_index = {
        int(item["index"]): normalize_uuid(item["uuid"]) for item in devices
    }
    one_gpu_uuids = {
        normalize_uuid(record["device_identifier"])
        for record in one_gpu["records"]
    }
    two_gpu_uuids = {
        normalize_uuid(record["device_identifier"])
        for record in two_gpu["records"]
    }
    require(
        len(one_gpu_uuids) == 1 and one_gpu_uuids <= inventory_uuids,
        f"{scenario_name} 1-GPU lane lacks one inventory-bound UUID",
    )
    require(
        len(two_gpu_uuids) == 2 and two_gpu_uuids <= inventory_uuids,
        f"{scenario_name} 2-GPU lane lacks two inventory-bound UUIDs",
    )
    require(
        one_gpu_uuids <= two_gpu_uuids,
        f"{scenario_name} 1-GPU device is absent from the 2-GPU lane",
    )
    for label, lane in (("1-GPU", one_gpu), ("2-GPU", two_gpu)):
        require(
            all(
                inventory_by_index.get(record["selected_device"])
                == normalize_uuid(record["device_identifier"])
                for record in lane["records"]
            ),
            f"{scenario_name} {label} selected-device/UUID mapping differs",
        )


def validate_terminal_gpu_inventory(
    initial: list[dict[str, str]], terminal: list[dict[str, str]]
) -> None:
    require(
        exact_json_equal(terminal, initial),
        "GPU inventory changed during qualification",
    )


def validate_lane_records(
    name: str,
    backend: str,
    ranks: int,
    targets: int,
    frequencies: int,
    repetitions: int,
    records: list[dict[str, Any]],
) -> None:
    """Validate counters and collective results shared by live and sealed lanes."""
    require(
        [record.get("rank") for record in records] == list(range(ranks)),
        f"{name} rank set differs",
    )
    for record in records:
        require(record.get("world_size") == ranks, f"{name} world size differs")
        require(record.get("backend") == backend, f"{name} backend differs")
        require(record.get("target_count") == targets, f"{name} target count differs")
        require(
            record.get("frequency_count") == frequencies,
            f"{name} frequency count differs",
        )
        require(
            record.get("repetitions") == repetitions,
            f"{name} repetition count differs",
        )
        require(
            record.get("local_chunk_count", 0) > 0
            and record.get("local_source_points", 0) > 0,
            f"{name} rank {record.get('rank')} owns no Near2Far work",
        )
        require(
            record.get("mpi_allreduce_calls") == repetitions,
            f"{name} collective count differs",
        )
        collective_scalars_per_work = 13 if backend == "cuda" else 12
        require(
            record.get("mpi_allreduce_bytes")
            == repetitions
            * targets
            * frequencies
            * collective_scalars_per_work
            * 8,
            f"{name} collective byte count differs",
        )
        require(
            isinstance(record.get("elapsed"), list)
            and len(record["elapsed"]) == repetitions,
            f"{name} timing count differs",
        )
        require(
            isinstance(record.get("fields"), list)
            and len(record["fields"]) == targets * frequencies * 6,
            f"{name} field count differs",
        )
        require(
            record.get("cpu_curl_calls") == 0
            and record.get("cuda_curl_calls") == 0,
            f"{name} mixed unrelated FDTD curl work into the transform timer",
        )
        if backend == "cpu":
            require(
                record.get("selected_device") == -1
                and record.get("device_identifier") == "cpu",
                f"{name} CPU lane selected a GPU",
            )
            require(
                record.get("cpu_transform_calls") == repetitions * targets,
                f"{name} CPU transform count differs",
            )
            require(
                record.get("cuda_transform_calls") == 0,
                f"{name} CPU lane used CUDA",
            )
            for counter in (
                "cuda_submitted_chunks", "cuda_source_points",
                "cuda_output_points", "cuda_frequencies",
                "cuda_periodic_copies", "cuda_fast_precision_calls",
                "cuda_mixed_precision_calls", "cuda_cancellation_retries",
                "cuda_target_tiles", "cuda_frequency_tiles",
                "cuda_operation_tiles", "cuda_maximum_workspace_bytes",
                "cuda_descriptor_uploads", "cuda_kernel_launches",
                "cuda_result_device_to_host_bytes",
                "cuda_condition_device_to_host_bytes",
                "dft_device_to_host_bytes_avoided",
            ):
                require(record.get(counter) == 0,
                        f"{name} CPU lane has nonzero {counter}")
        else:
            require(record.get("selected_device", -1) >= 0,
                    f"{name} selected no GPU")
            require(record.get("cpu_transform_calls") == 0,
                    f"{name} used CPU Green fallback")
            require(record.get("cuda_transform_calls") == repetitions,
                    f"{name} CUDA transform count differs")
            require(
                record.get("cuda_submitted_chunks")
                == repetitions * record["local_chunk_count"],
                f"{name} submitted chunk count differs",
            )
            require(
                record.get("cuda_source_points")
                == repetitions * record["local_source_points"],
                f"{name} source-point count differs",
            )
            require(record.get("cuda_output_points") == repetitions * targets,
                    f"{name} output-point count differs")
            require(record.get("cuda_frequencies") == repetitions * frequencies,
                    f"{name} frequency execution count differs")
            require(record.get("cuda_periodic_copies") == repetitions,
                    f"{name} periodic-copy count differs")
            require(record.get("cuda_fast_precision_calls") == repetitions,
                    f"{name} did not retain the fast FP32 CUDA path")
            require(record.get("cuda_mixed_precision_calls") == 0,
                    f"{name} unexpectedly used mixed precision")
            require(record.get("cuda_cancellation_retries") == 0,
                    f"{name} unexpectedly retried for cancellation")
            require(record.get("cuda_target_tiles") == repetitions,
                    f"{name} target tile count differs")
            require(record.get("cuda_frequency_tiles") == repetitions,
                    f"{name} frequency tile count differs")
            require(record.get("cuda_operation_tiles") == repetitions,
                    f"{name} operation tile count differs")
            require(
                0 < record.get("cuda_maximum_workspace_bytes", 0)
                <= 64 * 1024 * 1024,
                f"{name} workspace escaped the 64 MiB ceiling",
            )
            require(record.get("cuda_descriptor_uploads") == 0,
                    f"{name} retained plan was re-uploaded in measured repetitions")
            require(
                record.get("cuda_kernel_launches")
                == 2 * record["cuda_operation_tiles"],
                f"{name} kernel/tile count differs",
            )
            require(
                record.get("cuda_result_device_to_host_bytes")
                == repetitions * targets * frequencies * 12 * 8,
                f"{name} result D2H byte count differs",
            )
            require(
                record.get("cuda_condition_device_to_host_bytes")
                == repetitions * targets * frequencies * 8,
                f"{name} condition-evidence D2H byte count differs",
            )
            require(record.get("dft_device_to_host_bytes_avoided", 0) > 0,
                    f"{name} lacks resident DFT readback avoidance")
            require(
                record.get("live_buffers_after")
                == record.get("live_buffers_before"),
                f"{name} retained buffer count changed during measurement",
            )
    for record in records[1:]:
        require(record["fields"] == records[0]["fields"],
                f"{name} ranks disagree on allreduce output")
        require(record["elapsed"] == records[0]["elapsed"],
                f"{name} ranks disagree on max timing")


def run_lane(
    executable: pathlib.Path,
    raw_root: pathlib.Path,
    name: str,
    backend: str,
    ranks: int,
    threads: int,
    targets: int,
    frequencies: int,
    repetitions: int,
    timeout_seconds: float = 300.0,
    stdout_limit: int = 8 * 1024 * 1024,
    stderr_limit: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    prefix = raw_root / name
    command = [
        "mpirun", "--bind-to", "core", "--map-by", f"slot:PE={threads}",
        "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
        "-np", str(ranks), str(executable),
    ]
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("MEEP_GPU_") or key in {
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
        }:
            environment.pop(key, None)
    environment.update(
        MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_WRITE=str(prefix),
        MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_BACKEND=backend,
        MEEP_GPU_TEST_NEAR2FAR_TARGET_COUNT=str(targets),
        MEEP_GPU_TEST_NEAR2FAR_FREQUENCY_COUNT=str(frequencies),
        MEEP_GPU_TEST_NEAR2FAR_REPETITIONS=str(repetitions),
        OMP_NUM_THREADS=str(threads),
        OMP_PROC_BIND="spread",
        OMP_PLACES="cores",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    started = time.time()
    process = provenance.bounded_command(
        command,
        cwd=raw_root,
        environment=environment,
        timeout_seconds=timeout_seconds,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )
    (raw_root / f"{name}.stdout").write_bytes(process["stdout"])
    (raw_root / f"{name}.stderr").write_bytes(process["stderr"])
    require(process["error"] is None, f"{name} command error: {process['error']}")
    require(not process["timeout"], f"{name} exceeded {timeout_seconds:g}s timeout")
    require(
        process["output_limit"] is None,
        f"{name} exceeded {process['output_limit']} output limit",
    )
    require(
        process["exit_code"] == 0,
        f"{name} failed with exit {process['exit_code']}: "
        + process["stderr"].decode("utf-8", errors="replace")[-2048:],
    )
    records = [parse_record(pathlib.Path(f"{prefix}.rank-{rank}.txt")) for rank in range(ranks)]
    validate_lane_records(
        name, backend, ranks, targets, frequencies, repetitions, records
    )
    samples = records[0]["elapsed"]
    return {
        "name": name,
        "backend": backend,
        "ranks": ranks,
        "threads_per_rank": threads,
        "targets": targets,
        "frequency_count": frequencies,
        "records": records,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "command": command,
        "wall_seconds": time.time() - started,
        "timeout_seconds": timeout_seconds,
        "stdout_limit_bytes": stdout_limit,
        "stderr_limit_bytes": stderr_limit,
        "stdout": file_record(raw_root / f"{name}.stdout"),
        "stderr": file_record(raw_root / f"{name}.stderr"),
    }


def run_failure_abort_probe(
    executable: pathlib.Path, raw_root: pathlib.Path,
    timeout_seconds: float = 30.0,
    stdout_limit: int = 1024 * 1024,
    stderr_limit: int = 1024 * 1024,
) -> dict[str, Any]:
    command = [
        "mpirun", "--bind-to", "core", "--map-by", "slot:PE=1",
        "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
        "-np", "2", str(executable),
    ]
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("MEEP_GPU_"):
            environment.pop(key, None)
    environment["MEEP_GPU_TEST_NEAR2FAR_ABORT"] = "1"
    process = provenance.bounded_command(
        command,
        cwd=raw_root,
        environment=environment,
        timeout_seconds=timeout_seconds,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )
    require(process["error"] is None,
            f"Near2Far failure probe command error: {process['error']}")
    require(not process["timeout"],
            "Near2Far rank-local failure probe hung instead of aborting")
    require(process["output_limit"] is None,
            "Near2Far rank-local failure probe exceeded its output limit")
    stdout_path = raw_root / "rank-local-failure.stdout"
    stderr_path = raw_root / "rank-local-failure.stderr"
    stdout_path.write_bytes(process["stdout"])
    stderr_path.write_bytes(process["stderr"])
    combined = (
        process["stdout"] + b"\n" + process["stderr"]
    ).decode("utf-8", errors="replace")
    require(process["exit_code"] != 0,
            "Near2Far rank-local failure probe returned success")
    for rank in (0, 1):
        require(
            f"gpmeep-near2far-abort-probe:rank={rank},stage=before-rank-local-cuda-failure"
            in combined,
            f"Near2Far failure probe lacks rank {rank} pre-failure marker",
        )
    require(
        "rank-local failure during distributed near-to-far processing" in combined,
        "Near2Far failure probe lacks the communicator-abort diagnostic",
    )
    require(
        "FAIL: Near2Far rank-local exception returned" not in combined,
        "Near2Far failure probe returned past the protected collective",
    )
    return {
        "command": command,
        "exit_code": process["exit_code"],
        "timeout_seconds": timeout_seconds,
        "stdout_limit_bytes": stdout_limit,
        "stderr_limit_bytes": stderr_limit,
        "stdout": file_record(stdout_path),
        "stderr": file_record(stderr_path),
    }


def run_collective_safety_probe(
    executable: pathlib.Path,
    raw_root: pathlib.Path,
    name: str,
    *,
    selection_mismatch: str | None = None,
    metadata_mismatch: str | None = None,
    api: str = "batch",
    chunkless_rank: bool = False,
    cross_rank_cancellation: bool = False,
    distributed_lowlevel: bool = False,
    timeout_seconds: float = 30.0,
    stdout_limit: int = 1024 * 1024,
    stderr_limit: int = 1024 * 1024,
) -> dict[str, Any]:
    require(
        sum((selection_mismatch is not None, metadata_mismatch is not None,
             chunkless_rank, cross_rank_cancellation,
             distributed_lowlevel)) == 1,
        "collective safety probe must select exactly one mode",
    )
    command = [
        "mpirun", "--bind-to", "core", "--map-by", "slot:PE=1",
        "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
        "-np", "2", str(executable),
    ]
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("MEEP_GPU_"):
            environment.pop(key, None)
    environment.update(
        MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_WRITE=str(raw_root / name),
        MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_BACKEND="cuda",
        MEEP_GPU_TEST_NEAR2FAR_TARGET_COUNT="4",
        MEEP_GPU_TEST_NEAR2FAR_FREQUENCY_COUNT="2",
        MEEP_GPU_TEST_NEAR2FAR_REPETITIONS="3",
        OMP_NUM_THREADS="1",
        OMP_PROC_BIND="spread",
        OMP_PLACES="cores",
    )
    if selection_mismatch is not None:
        require(selection_mismatch in {"optout", "backend"},
                "selection mismatch probe is invalid")
        require(api in {"batch", "grid"}, "selection mismatch API is invalid")
        environment["MEEP_GPU_TEST_NEAR2FAR_SELECTION_MISMATCH"] = selection_mismatch
        environment["MEEP_GPU_TEST_NEAR2FAR_SELECTION_API"] = api
        expected = (
            f"PASS: Near2Far {api} {selection_mismatch} mismatch fails closed "
            "on every MPI rank"
        )
    elif metadata_mismatch is not None:
        require(
            metadata_mismatch in {
                "point-count", "null-points", "target-coordinates",
                "frequency-count", "frequency-values", "material-periodic",
                "monitor-dimension", "grid-shape",
                "grid-monitor-dimension", "grid-frequency-count",
                "greencyl-tol"
            },
            "metadata mismatch probe is invalid",
        )
        environment["MEEP_GPU_TEST_NEAR2FAR_METADATA_MISMATCH"] = metadata_mismatch
        expected = (
            f"PASS: Near2Far {metadata_mismatch} metadata mismatch fails "
            "closed before result collectives"
        )
    elif chunkless_rank:
        environment["MEEP_GPU_TEST_NEAR2FAR_CHUNKLESS_RANK"] = "1"
        expected = (
            "PASS: chunkless rank enters the public 13-double Near2Far "
            "collective and receives the nonempty rank result"
        )
    elif cross_rank_cancellation:
        environment.pop("MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_WRITE", None)
        environment.pop("MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_BACKEND", None)
        environment["MEEP_GPU_TEST_NEAR2FAR_CROSS_RANK_CANCELLATION"] = "1"
        expected = (
            "PASS: Near2Far collective gate detects cross-rank cancellation"
        )
    else:
        environment["MEEP_GPU_TEST_NEAR2FAR_LOWLEVEL_DISTRIBUTED"] = "1"
        expected = (
            "PASS: distributed Near2Far lowlevel preserves rank-local "
            "ownership and forces mixed CUDA before explicit sum"
        )
    process = provenance.bounded_command(
        command,
        cwd=raw_root,
        environment=environment,
        timeout_seconds=timeout_seconds,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )
    require(process["error"] is None,
            f"{name} command error: {process['error']}")
    require(not process["timeout"], f"{name} hung in mismatched collectives")
    require(process["output_limit"] is None,
            f"{name} exceeded its output limit")
    stdout_path = raw_root / f"{name}.stdout"
    stderr_path = raw_root / f"{name}.stderr"
    stdout_path.write_bytes(process["stdout"])
    stderr_path.write_bytes(process["stderr"])
    combined = (process["stdout"] + b"\n" + process["stderr"]).decode(
        "utf-8", errors="replace"
    )
    require(process["exit_code"] == 0,
            f"{name} failed with exit {process['exit_code']}: {combined[-2048:]}")
    require(expected in combined, f"{name} lacks its completion marker")
    return {
        "name": name,
        "command": command,
        "exit_code": process["exit_code"],
        "timeout_seconds": timeout_seconds,
        "stdout_limit_bytes": stdout_limit,
        "stderr_limit_bytes": stderr_limit,
        "stdout": file_record(stdout_path),
        "stderr": file_record(stderr_path),
    }


def _atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# gpmeep Near2Far MPI qualification", "",
        f"- State: **{report['state'].upper()}**",
        f"- Tier: **{report['runtime_provenance']['qualification_tier']}**",
        f"- GPUs: {len(report['gpu_inventory'])}", "",
        "| Scenario | Best CPU | 1 GPU | 2 GPU | CPU/1GPU | CPU/2GPU | 1GPU/2GPU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in report["scenarios"]:
        lines.append(
            f"| {scenario['name']} | {scenario['best_cpu_seconds']:.6f} | "
            f"{scenario['one_gpu_seconds']:.6f} | {scenario['two_gpu_seconds']:.6f} | "
            f"{scenario['cpu_to_one_gpu_speedup']:.3f}x | "
            f"{scenario['cpu_to_two_gpu_speedup']:.3f}x | "
            f"{scenario['one_to_two_gpu_scaling']:.3f}x |"
        )
    return "\n".join(lines) + "\n"


def _report_raw_file_records(
    report: dict[str, Any], raw_root: pathlib.Path
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if {
                "path", "size_bytes", "sha256"
            }.issubset(value) and isinstance(value.get("path"), str):
                path = pathlib.Path(value["path"])
                if path.is_absolute():
                    try:
                        relative = path.resolve(strict=True).relative_to(raw_root)
                    except (OSError, ValueError):
                        pass
                    else:
                        normalized = {
                            "size_bytes": value["size_bytes"],
                            "sha256": value["sha256"],
                        }
                        key = relative.as_posix()
                        require(
                            key not in records or records[key] == normalized,
                            f"report contains conflicting raw record for {key}",
                        )
                        records[key] = normalized
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(report)
    return records


def _manifest_raw_file_records(
    raw_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    files = raw_manifest.get("files")
    require(isinstance(files, list), "raw evidence manifest lacks files")
    result: dict[str, dict[str, Any]] = {}
    for record in files:
        _validate_file_record_shape(
            record,
            "raw evidence manifest file",
            extra_keys={"kind", "mode_octal"},
        )
        result[record["path"]] = {
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
    require(len(result) == len(files), "raw evidence manifest has duplicate paths")
    return result


def _validate_file_record_shape(
    record: Any,
    context: str,
    *,
    extra_keys: set[str] | None = None,
) -> None:
    keys = {"path", "size_bytes", "sha256"} | (extra_keys or set())
    require(
        isinstance(record, dict)
        and set(record) == keys
        and isinstance(record.get("path"), str)
        and bool(record["path"])
        and type(record.get("size_bytes")) is int
        and record["size_bytes"] >= 0
        and isinstance(record.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
        f"{context} file record is invalid",
    )
    if extra_keys:
        require(
            record["kind"] == "file"
            and isinstance(record["mode_octal"], str)
            and re.fullmatch(r"[0-7]{4}", record["mode_octal"]) is not None,
            f"{context} file metadata is invalid",
        )


def _require_file_record_at(
    record: Any, expected: pathlib.Path, context: str
) -> pathlib.Path:
    _validate_file_record_shape(record, context)
    try:
        actual = pathlib.Path(record["path"]).resolve(strict=True)
        expected = expected.resolve(strict=True)
    except (OSError, TypeError) as error:
        raise QualificationError(f"cannot resolve {context}: {error}") from error
    require(actual == expected, f"{context} path differs")
    require(file_record(actual) == record, f"{context} file record differs")
    return actual


def _near2far_command(executable: pathlib.Path, ranks: int, threads: int) -> list[str]:
    return [
        "mpirun", "--bind-to", "core", "--map-by", f"slot:PE={threads}",
        "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
        "-np", str(ranks), str(executable),
    ]


def _finite_positive(value: Any, context: str) -> float:
    require(
        type(value) in {int, float}
        and math.isfinite(value)
        and value > 0,
        f"{context} must be finite and positive",
    )
    return float(value)


def _validate_sha256(value: Any, context: str) -> None:
    require(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{context} is not a canonical SHA-256",
    )


def _validate_release_runtime_provenance_shapes(
    runtime_provenance: dict[str, Any], executable: pathlib.Path
) -> None:
    require(
        isinstance(runtime_provenance.get("git_head"), str)
        and re.fullmatch(r"[0-9a-f]{40}", runtime_provenance["git_head"])
        is not None
        and isinstance(runtime_provenance.get("git_status_porcelain"), list)
        and all(isinstance(line, str)
                for line in runtime_provenance["git_status_porcelain"]),
        "release Git provenance is invalid",
    )
    receipt = runtime_provenance["build_receipt"]
    _validate_file_record_shape(receipt.get("file"), "release build receipt")
    for key in ("receipt_id", "build_input_id", "artifact_set_id"):
        _validate_sha256(receipt.get(key), f"release build receipt {key}")
    bound = receipt.get("bound_artifacts")
    require(
        isinstance(bound, dict)
        and set(bound) == {"gpu_step_db_test", "libmeep"},
        "release receipt-bound artifact inventory differs",
    )
    for name, record in bound.items():
        _validate_file_record_shape(record, f"release bound artifact {name}")

    loaded = runtime_provenance["loaded_dependencies"]
    command = loaded.get("ldd_command")
    rows = loaded.get("ldd_normalized_rows")
    require(
        isinstance(command, list)
        and len(command) == 2
        and all(isinstance(value, str) and value for value in command)
        and command == ["/usr/bin/ldd", str(executable)]
        and isinstance(rows, list)
        and bool(rows)
        and all(isinstance(row, str) and bool(row) for row in rows)
        and rows == sorted(rows),
        "release loaded-dependency command or rows are invalid",
    )
    _validate_sha256(
        loaded.get("ldd_normalized_sha256"),
        "release normalized ldd digest",
    )
    require(
        hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()
        == loaded["ldd_normalized_sha256"],
        "release normalized ldd digest differs from its rows",
    )
    libraries = loaded.get("libraries")
    require(
        isinstance(libraries, dict)
        and bool(libraries)
        and all(isinstance(name, str) and bool(name) for name in libraries),
        "release loaded-library inventory is invalid",
    )
    for name, record in libraries.items():
        _validate_file_record_shape(record, f"release loaded library {name}")
    _validate_file_record_shape(
        loaded.get("dynamic_interpreter"), "release dynamic interpreter"
    )

    source = runtime_provenance["source_snapshot"]
    require(
        isinstance(source.get("algorithm"), str)
        and bool(source["algorithm"])
        and type(source.get("file_count")) is int
        and source["file_count"] >= 1
        and isinstance(source.get("missing_paths"), list)
        and all(isinstance(path, str) for path in source["missing_paths"])
        and type(source.get("source_manifest_schema_version")) is int
        and source["source_manifest_schema_version"] == 2
        and isinstance(source.get("files"), list)
        and len(source["files"]) == source["file_count"],
        "release source snapshot metadata is invalid",
    )
    _validate_sha256(
        source.get("source_manifest_sha256"), "release source manifest digest"
    )
    _validate_sha256(source.get("sha256"), "release source snapshot digest")
    for record in source["files"]:
        _validate_file_record_shape(
            record,
            "release source file",
            extra_keys={"kind", "mode_octal"},
        )


def validate_report_semantics(
    report: dict[str, Any], raw_root: pathlib.Path
) -> None:
    """Recompute every release claim from its sealed rank-level evidence."""
    require(
        report.get("schema") == SCHEMA and report.get("state") == "complete",
        "Near2Far report is not complete",
    )
    runtime_provenance = report.get("runtime_provenance")
    require(isinstance(runtime_provenance, dict),
            "report runtime provenance is absent")
    tier = runtime_provenance.get("qualification_tier")
    require(tier in {"development", "release"},
            "report qualification tier is invalid")
    # Development bundles are diagnostic and intentionally allow custom lane
    # matrices. Release bundles have one canonical, fully recomputed contract.
    if tier != "release":
        return
    require(
        set(runtime_provenance) == {
            "qualification_tier", "runtime", "loaded_dependencies",
            "git_head", "git_status_porcelain", "source_snapshot",
            "build_receipt",
        },
        "release runtime-provenance inventory differs",
    )

    require(
        set(report) == {
            "schema", "state", "created_at_unix", "executable",
            "gpu_inventory", "gpu_inventory_after", "cpu_inventory",
            "runtime_provenance", "rank_local_failure_abort_probe",
            "collective_safety_probes", "configuration", "lanes",
            "scenarios",
        },
        "release report top-level inventory differs",
    )
    require(
        type(report["created_at_unix"]) is float,
        "report creation time must be a float",
    )
    _finite_positive(report["created_at_unix"], "report creation time")
    configuration = report.get("configuration")
    require(
        isinstance(configuration, dict)
        and set(configuration) == {
            "executable", "output", "verify_evidence", "batch_targets",
            "frequency_count", "repetitions", "cpu_threads",
            "small_minimum_speedup", "batch_minimum_speedup",
            "batch_minimum_two_gpu_scaling", "lane_timeout_seconds",
            "stdout_limit_bytes", "stderr_limit_bytes",
            "qualification_tier", "build_receipt",
        },
        "release configuration inventory differs",
    )
    require(configuration["qualification_tier"] == "release",
            "release configuration tier differs")
    require(configuration["verify_evidence"] is None,
            "generated release configuration contains a verification input")
    for name, minimum in (
        ("batch_targets", 2), ("frequency_count", 1),
        ("repetitions", 3), ("cpu_threads", 2),
        ("stdout_limit_bytes", 1), ("stderr_limit_bytes", 1),
    ):
        require(
            type(configuration[name]) is int and configuration[name] >= minimum,
            f"release configuration {name} is invalid",
        )
    require(
        configuration["batch_targets"] <= MAX_EVIDENCE_TARGETS
        and configuration["frequency_count"] <= MAX_EVIDENCE_FREQUENCIES
        and configuration["repetitions"] <= MAX_EVIDENCE_REPETITIONS
        and configuration["batch_targets"]
        * configuration["frequency_count"]
        * 6
        <= MAX_EVIDENCE_FIELDS,
        "release configuration exceeds evidence verifier bounds",
    )
    _finite_positive(configuration["lane_timeout_seconds"],
                     "release lane timeout")
    speed_namespace = argparse.Namespace(
        qualification_tier="release",
        small_minimum_speedup=_finite_positive(
            configuration["small_minimum_speedup"], "single-point speed gate"
        ),
        batch_minimum_speedup=_finite_positive(
            configuration["batch_minimum_speedup"], "batch speed gate"
        ),
        batch_minimum_two_gpu_scaling=_finite_positive(
            configuration["batch_minimum_two_gpu_scaling"],
            "batch two-GPU scaling gate",
        ),
    )
    validate_release_speed_gates(speed_namespace)

    raw_root = raw_root.resolve(strict=True)
    output = raw_root.parent
    require(
        pathlib.Path(configuration["output"]).resolve(strict=True) == output,
        "release configuration output path differs",
    )
    executable = pathlib.Path(configuration["executable"]).resolve(strict=True)
    _require_file_record_at(report["executable"], executable,
                            "release executable")
    runtime = runtime_provenance.get("runtime")
    require(
        isinstance(runtime, dict)
        and set(runtime) == {
            "execution_contract", "invoked", "actual_elf", "actual_elf_path",
        }
        and runtime.get("execution_contract") == "direct-elf-v1"
        and exact_json_equal(runtime.get("invoked"), report["executable"])
        and exact_json_equal(runtime.get("actual_elf"), report["executable"])
        and runtime.get("actual_elf_path") == str(executable),
        "release runtime executable identity differs",
    )
    receipt_summary = runtime_provenance.get("build_receipt")
    require(
        isinstance(receipt_summary, dict)
        and set(receipt_summary) == {
            "file", "receipt_id", "build_input_id", "artifact_set_id",
            "bound_artifacts",
        }
        and isinstance(configuration["build_receipt"], str)
        and receipt_summary.get("file", {}).get("path")
        == configuration["build_receipt"],
        "release configuration receipt identity differs",
    )
    loaded_dependencies = runtime_provenance.get("loaded_dependencies")
    require(
        isinstance(loaded_dependencies, dict)
        and set(loaded_dependencies) == {
            "ldd_command", "ldd_normalized_rows", "ldd_normalized_sha256",
            "libraries", "dynamic_interpreter",
        },
        "release loaded-dependency inventory differs",
    )
    source_snapshot = runtime_provenance.get("source_snapshot")
    require(
        isinstance(source_snapshot, dict)
        and set(source_snapshot) == {
            "algorithm", "file_count", "missing_paths",
            "source_manifest_schema_version", "source_manifest_sha256",
            "files", "sha256",
        },
        "release source-snapshot inventory differs",
    )
    _validate_release_runtime_provenance_shapes(
        runtime_provenance, executable
    )

    inventory = report.get("gpu_inventory")
    terminal_inventory = report.get("gpu_inventory_after")
    validate_gpu_inventory_records(inventory)
    validate_terminal_gpu_inventory(inventory, terminal_inventory)
    cpu = report.get("cpu_inventory")
    validate_cpu_inventory_record(cpu, configuration["cpu_threads"])

    lane_keys = {
        "name", "backend", "ranks", "threads_per_rank", "targets",
        "frequency_count", "records", "median_seconds", "minimum_seconds",
        "maximum_seconds", "command", "wall_seconds", "timeout_seconds",
        "stdout_limit_bytes", "stderr_limit_bytes", "stdout", "stderr",
    }
    lanes = report.get("lanes")
    require(isinstance(lanes, list) and len(lanes) == 10,
            "release lane inventory must contain exactly ten lanes")
    frequencies = configuration["frequency_count"]
    repetitions = configuration["repetitions"]
    cpu_threads = configuration["cpu_threads"]
    timeout = configuration["lane_timeout_seconds"]
    stdout_limit = configuration["stdout_limit_bytes"]
    stderr_limit = configuration["stderr_limit_bytes"]
    expected_lanes: list[tuple[str, str, int, int, int]] = []
    for scenario_name, targets in (
        ("single-point", 1), ("batch", configuration["batch_targets"])
    ):
        expected_lanes.extend(
            [
                (f"{scenario_name}-cpu-1x1", "cpu", 1, 1, targets),
                (
                    f"{scenario_name}-cpu-1x{cpu_threads}",
                    "cpu", 1, cpu_threads, targets,
                ),
                (
                    f"{scenario_name}-cpu-2x{max(1, cpu_threads // 2)}",
                    "cpu", 2, max(1, cpu_threads // 2), targets,
                ),
                (f"{scenario_name}-cuda-1", "cuda", 1, 1, targets),
                (f"{scenario_name}-cuda-2", "cuda", 2, 1, targets),
            ]
        )
    require(
        len({name for name, *_ in expected_lanes}) == len(expected_lanes),
        "release CPU-thread configuration creates duplicate lanes",
    )
    expected_raw_names: set[str] = set()
    validated_lanes: dict[str, dict[str, Any]] = {}
    for lane, (name, backend, ranks, threads, targets) in zip(
        lanes, expected_lanes
    ):
        require(isinstance(lane, dict) and set(lane) == lane_keys,
                f"{name} lane schema differs")
        require(
            lane["name"] == name
            and lane["backend"] == backend
            and type(lane["ranks"]) is int
            and lane["ranks"] == ranks
            and type(lane["threads_per_rank"]) is int
            and lane["threads_per_rank"] == threads
            and type(lane["targets"]) is int
            and lane["targets"] == targets
            and type(lane["frequency_count"]) is int
            and lane["frequency_count"] == frequencies,
            f"{name} lane topology differs",
        )
        require(lane["command"] == _near2far_command(executable, ranks, threads),
                f"{name} command differs")
        require(
            type(lane["timeout_seconds"]) is float
            and lane["timeout_seconds"] == timeout
            and type(lane["stdout_limit_bytes"]) is int
            and lane["stdout_limit_bytes"] == stdout_limit
            and type(lane["stderr_limit_bytes"]) is int
            and lane["stderr_limit_bytes"] == stderr_limit,
            f"{name} execution bounds differ",
        )
        require(
            type(lane["wall_seconds"]) is float,
            f"{name} wall time must be a float",
        )
        _finite_positive(lane["wall_seconds"], f"{name} wall time")
        stdout_name = f"{name}.stdout"
        stderr_name = f"{name}.stderr"
        stdout_path = _require_file_record_at(
            lane["stdout"], raw_root / stdout_name, f"{name} stdout"
        )
        _require_file_record_at(
            lane["stderr"], raw_root / stderr_name, f"{name} stderr"
        )
        expected_raw_names.update({stdout_name, stderr_name})
        completion = (
            f"PASS: wrote Near2Far MPI evidence for {ranks} rank(s), "
            f"backend={backend}, targets={targets}, repetitions={repetitions}"
        )
        require(
            completion
            in _bounded_regular_file_bytes(
                stdout_path, f"{name} stdout", MAX_EVIDENCE_FILE_BYTES
            ).decode("utf-8", errors="replace"),
            f"{name} lacks its completion marker",
        )
        require(
            isinstance(lane["records"], list)
            and len(lane["records"]) == ranks,
            f"{name} rank evidence count differs",
        )
        parsed_records = []
        for rank, recorded in enumerate(lane["records"]):
            record_name = f"{name}.rank-{rank}.txt"
            expected_raw_names.add(record_name)
            _require_file_record_at(
                recorded.get("artifact") if isinstance(recorded, dict) else None,
                raw_root / record_name,
                f"{name} rank {rank} evidence",
            )
            try:
                parsed = parse_record(raw_root / record_name)
            except (KeyError, OSError, TypeError, ValueError) as error:
                raise QualificationError(
                    f"cannot parse {name} rank {rank} evidence: {error}"
                ) from error
            require(exact_json_equal(recorded, parsed),
                    f"{name} rank {rank} report differs from raw evidence")
            parsed_records.append(parsed)
        validate_lane_records(
            name, backend, ranks, targets, frequencies, repetitions,
            parsed_records,
        )
        samples = parsed_records[0]["elapsed"]
        require(
            type(lane["median_seconds"]) is float
            and lane["median_seconds"] == statistics.median(samples)
            and type(lane["minimum_seconds"]) is float
            and lane["minimum_seconds"] == min(samples)
            and type(lane["maximum_seconds"]) is float
            and lane["maximum_seconds"] == max(samples)
            and lane["minimum_seconds"] <= lane["median_seconds"]
            <= lane["maximum_seconds"],
            f"{name} timing summary differs from raw evidence",
        )
        validated_lanes[name] = lane

    recomputed_scenarios = []
    for scenario_name, targets in (
        ("single-point", 1), ("batch", configuration["batch_targets"])
    ):
        cpu_names = [
            f"{scenario_name}-cpu-1x1",
            f"{scenario_name}-cpu-1x{cpu_threads}",
            f"{scenario_name}-cpu-2x{max(1, cpu_threads // 2)}",
        ]
        cpu_lanes = [validated_lanes[name] for name in cpu_names]
        one_gpu = validated_lanes[f"{scenario_name}-cuda-1"]
        two_gpu = validated_lanes[f"{scenario_name}-cuda-2"]
        scenario_lanes = [*cpu_lanes, one_gpu, two_gpu]
        require(
            len(
                {
                    sum(
                        record["local_source_points"]
                        for record in lane["records"]
                    )
                    for lane in scenario_lanes
                }
            )
            == 1,
            f"{scenario_name} lanes did not measure the same source workload",
        )
        best_cpu = min(cpu_lanes, key=lambda lane: lane["median_seconds"])
        cpu_one = best_cpu["median_seconds"] / one_gpu["median_seconds"]
        cpu_two = best_cpu["median_seconds"] / two_gpu["median_seconds"]
        gpu_scaling = one_gpu["median_seconds"] / two_gpu["median_seconds"]
        errors = validate_numeric_agreement(
            scenario_name,
            cpu_lanes[0]["records"][0]["fields"],
            best_cpu["records"][0]["fields"],
            one_gpu["records"][0]["fields"],
            two_gpu["records"][0]["fields"],
        )
        required_speedup = (
            configuration["small_minimum_speedup"]
            if scenario_name == "single-point"
            else configuration["batch_minimum_speedup"]
        )
        require(cpu_one >= required_speedup,
                f"{scenario_name} sealed 1-GPU speedup is below its gate")
        if scenario_name == "batch":
            require(
                gpu_scaling >= configuration["batch_minimum_two_gpu_scaling"],
                "batch sealed 2-GPU scaling is below its gate",
            )
        validate_gpu_identity(scenario_name, one_gpu, two_gpu, inventory)
        recomputed_scenarios.append(
            {
                "name": scenario_name,
                "targets": targets,
                "best_cpu_lane": best_cpu["name"],
                "best_cpu_seconds": best_cpu["median_seconds"],
                "one_gpu_seconds": one_gpu["median_seconds"],
                "two_gpu_seconds": two_gpu["median_seconds"],
                "cpu_to_one_gpu_speedup": cpu_one,
                "cpu_to_two_gpu_speedup": cpu_two,
                "one_to_two_gpu_scaling": gpu_scaling,
                **errors,
                "two_gpu_error": errors["one_to_two_gpu_error"],
                "required_one_gpu_speedup": required_speedup,
            }
        )
    require(exact_json_equal(report.get("scenarios"), recomputed_scenarios),
            "release scenario claims differ from sealed raw evidence")

    probe_timeout = min(30.0, timeout)
    probe_stdout_limit = min(stdout_limit, 1024 * 1024)
    probe_stderr_limit = min(stderr_limit, 1024 * 1024)
    probe_keys = {
        "name", "command", "exit_code", "timeout_seconds",
        "stdout_limit_bytes", "stderr_limit_bytes", "stdout", "stderr",
    }
    probes = report.get("collective_safety_probes")
    require(
        isinstance(probes, list)
        and len(probes) == len(RELEASE_COLLECTIVE_PROBES),
        "release collective probe inventory differs",
    )
    probe_command = _near2far_command(executable, 2, 1)
    for probe, (name, marker) in zip(probes, RELEASE_COLLECTIVE_PROBES):
        require(isinstance(probe, dict) and set(probe) == probe_keys,
                f"{name} probe schema differs")
        require(
            probe["name"] == name
            and probe["command"] == probe_command
            and type(probe["exit_code"]) is int
            and probe["exit_code"] == 0
            and type(probe["timeout_seconds"]) is float
            and probe["timeout_seconds"] == probe_timeout
            and type(probe["stdout_limit_bytes"]) is int
            and probe["stdout_limit_bytes"] == probe_stdout_limit
            and type(probe["stderr_limit_bytes"]) is int
            and probe["stderr_limit_bytes"] == probe_stderr_limit,
            f"{name} probe execution contract differs",
        )
        stdout_name = f"{name}.stdout"
        stderr_name = f"{name}.stderr"
        stdout_path = _require_file_record_at(
            probe["stdout"], raw_root / stdout_name, f"{name} stdout"
        )
        stderr_path = _require_file_record_at(
            probe["stderr"], raw_root / stderr_name, f"{name} stderr"
        )
        expected_raw_names.update({stdout_name, stderr_name})
        combined = (
            _bounded_regular_file_bytes(
                stdout_path, f"{name} stdout", MAX_EVIDENCE_FILE_BYTES
            )
            + b"\n"
            + _bounded_regular_file_bytes(
                stderr_path, f"{name} stderr", MAX_EVIDENCE_FILE_BYTES
            )
        ).decode("utf-8", errors="replace")
        require(marker in combined, f"{name} lacks its completion marker")

    abort = report.get("rank_local_failure_abort_probe")
    abort_keys = {
        "command", "exit_code", "timeout_seconds", "stdout_limit_bytes",
        "stderr_limit_bytes", "stdout", "stderr",
    }
    require(isinstance(abort, dict) and set(abort) == abort_keys,
            "rank-local abort probe schema differs")
    require(
        abort["command"] == probe_command
        and type(abort["exit_code"]) is int
        and abort["exit_code"] != 0
        and type(abort["timeout_seconds"]) is float
        and abort["timeout_seconds"] == probe_timeout
        and type(abort["stdout_limit_bytes"]) is int
        and abort["stdout_limit_bytes"] == probe_stdout_limit
        and type(abort["stderr_limit_bytes"]) is int
        and abort["stderr_limit_bytes"] == probe_stderr_limit,
        "rank-local abort probe execution contract differs",
    )
    abort_stdout = _require_file_record_at(
        abort["stdout"], raw_root / "rank-local-failure.stdout",
        "rank-local abort stdout",
    )
    abort_stderr = _require_file_record_at(
        abort["stderr"], raw_root / "rank-local-failure.stderr",
        "rank-local abort stderr",
    )
    expected_raw_names.update(
        {"rank-local-failure.stdout", "rank-local-failure.stderr"}
    )
    abort_text = (
        _bounded_regular_file_bytes(
            abort_stdout, "rank-local abort stdout", MAX_EVIDENCE_FILE_BYTES
        )
        + b"\n"
        + _bounded_regular_file_bytes(
            abort_stderr, "rank-local abort stderr", MAX_EVIDENCE_FILE_BYTES
        )
    ).decode("utf-8", errors="replace")
    for rank in (0, 1):
        require(
            f"gpmeep-near2far-abort-probe:rank={rank},"
            "stage=before-rank-local-cuda-failure" in abort_text,
            f"rank-local abort probe lacks rank {rank} marker",
        )
    require(
        "rank-local failure during distributed near-to-far processing"
        in abort_text
        and "FAIL: Near2Far rank-local exception returned" not in abort_text,
        "rank-local abort probe diagnostics differ",
    )
    terminal_raw_entries = list(
        itertools.islice(raw_root.iterdir(), RELEASE_RAW_FILE_COUNT + 1)
    )
    require(
        len(terminal_raw_entries) == RELEASE_RAW_FILE_COUNT
        and {path.name for path in terminal_raw_entries}
        == expected_raw_names
        and all(path.is_file() and not path.is_symlink()
                for path in terminal_raw_entries),
        "release raw evidence inventory differs",
    )


def write_report(
    output: pathlib.Path, report: dict[str, Any], repo: pathlib.Path
) -> dict[str, Any]:
    json_path = output / "report.json"
    markdown_path = output / "report.md"
    manifest_path = output / BUNDLE_MANIFEST_NAME
    validate_report_semantics(report, output / "raw")
    provenance.atomic_write_json(json_path, report)
    _atomic_write_bytes(markdown_path, _report_markdown(report).encode("utf-8"))
    raw_manifest = provenance.tree_manifest(output / "raw", output)
    require(
        raw_manifest["symlink_count"] == 0,
        "raw evidence directory must not contain symlinks",
    )
    require(
        _report_raw_file_records(report, output / "raw")
        == _manifest_raw_file_records(raw_manifest),
        "report/raw evidence file sets differ",
    )
    receipt = report["runtime_provenance"].get("build_receipt")
    receipt_record = receipt.get("file") if isinstance(receipt, dict) else None
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "state": "complete",
        "qualification_tier": report["runtime_provenance"]["qualification_tier"],
        "files": {
            "report_json": _bounded_provenance_file_record(
                json_path, output, "generated Near2Far report", MAX_REPORT_BYTES
            ),
            "report_markdown": _bounded_provenance_file_record(
                markdown_path,
                output,
                "generated Near2Far Markdown report",
                MAX_REPORT_MARKDOWN_BYTES,
            ),
        },
        "raw_manifest": raw_manifest,
        "build_receipt": receipt_record,
    }
    manifest["bundle_id"] = provenance.canonical_sha256(manifest)
    provenance.atomic_write_json(manifest_path, manifest)
    complete = {
        "schema": BUNDLE_SCHEMA,
        "state": "complete",
        "bundle_id": manifest["bundle_id"],
        "evidence_manifest": _bounded_provenance_file_record(
            manifest_path,
            output,
            "generated evidence manifest",
            MAX_BUNDLE_MANIFEST_BYTES,
        ),
    }
    provenance.atomic_write_json(output / "COMPLETE", complete)
    verify_evidence_bundle(output, repo, allow_unattested_release=True)
    return manifest


def _bounded_regular_file_bytes(
    path: pathlib.Path, context: str, maximum_bytes: int
) -> bytes:
    """Read one immutable-size regular file without following links or FIFOs."""
    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        before = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise QualificationError(f"cannot inspect {context}: {error}") from error
    require(
        stat.S_ISREG(before.st_mode) and resolved == lexical,
        f"{context} must be a direct regular file",
    )
    require(
        0 <= before.st_size <= maximum_bytes,
        f"{context} exceeds its {maximum_bytes}-byte size bound",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        descriptor = os.open(lexical, flags)
        opened = os.fstat(descriptor)
        fingerprint = (
            opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size,
            opened.st_mtime_ns, opened.st_ctime_ns,
        )
        require(
            stat.S_ISREG(opened.st_mode)
            and opened.st_dev == before.st_dev
            and opened.st_ino == before.st_ino
            and opened.st_size == before.st_size
            and opened.st_size <= maximum_bytes,
            f"{context} changed before it could be read",
        )
        chunks: list[bytes] = []
        total = 0
        while total <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        require(
            len(data) <= maximum_bytes
            and len(data) == after.st_size
            and fingerprint
            == (
                after.st_dev, after.st_ino, after.st_mode, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns,
            ),
            f"{context} changed while it was read",
        )
        return data
    except OSError as error:
        raise QualificationError(f"cannot read {context}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_json_object(
    path: pathlib.Path, context: str, maximum_bytes: int
) -> dict[str, Any]:
    try:
        value = json.loads(
            _bounded_regular_file_bytes(path, context, maximum_bytes).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot read {context}: {error}") from error
    require(isinstance(value, dict), f"{context} is not an object")
    return value


def _bounded_provenance_file_record(
    path: pathlib.Path,
    repo: pathlib.Path,
    context: str,
    maximum_bytes: int,
) -> dict[str, Any]:
    data = _bounded_regular_file_bytes(path, context, maximum_bytes)
    return {
        "path": provenance.display_path(path, repo),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _validate_release_raw_manifest_bounds(
    raw_root: pathlib.Path, raw_manifest: Any
) -> None:
    require(
        isinstance(raw_manifest, dict)
        and set(raw_manifest) == {
            "root", "file_count", "directory_count", "symlink_count",
            "tree_manifest_schema_version", "root_mode_octal", "sha256",
            "files", "directories", "symlinks",
        }
        and raw_manifest.get("root") == "raw"
        and type(raw_manifest.get("file_count")) is int
        and raw_manifest["file_count"] == RELEASE_RAW_FILE_COUNT
        and type(raw_manifest.get("directory_count")) is int
        and raw_manifest["directory_count"] == 0
        and type(raw_manifest.get("symlink_count")) is int
        and raw_manifest["symlink_count"] == 0
        and type(raw_manifest.get("tree_manifest_schema_version")) is int
        and raw_manifest["tree_manifest_schema_version"] == 2
        and isinstance(raw_manifest.get("root_mode_octal"), str)
        and re.fullmatch(r"[0-7]{4}", raw_manifest["root_mode_octal"])
        is not None
        and isinstance(raw_manifest.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", raw_manifest["sha256"])
        is not None
        and isinstance(raw_manifest.get("files"), list)
        and len(raw_manifest["files"]) == RELEASE_RAW_FILE_COUNT
        and raw_manifest.get("directories") == []
        and raw_manifest.get("symlinks") == [],
        "release raw manifest inventory differs",
    )
    for record in raw_manifest["files"]:
        _validate_file_record_shape(
            record,
            "release raw manifest",
            extra_keys={"kind", "mode_octal"},
        )
        require(
            pathlib.PurePath(record["path"]).name == record["path"],
            "release raw manifest path is not a flat basename",
        )
    manifest_sizes = {
        record.get("path"): record.get("size_bytes")
        for record in raw_manifest["files"]
        if isinstance(record, dict)
        and isinstance(record.get("path"), str)
    }
    sizes = list(manifest_sizes.values())
    require(
        len(manifest_sizes) == RELEASE_RAW_FILE_COUNT
        and all(type(size) is int and 0 <= size <= MAX_EVIDENCE_FILE_BYTES
                for size in sizes)
        and sum(sizes) <= MAX_RELEASE_RAW_TOTAL_BYTES,
        "release raw manifest exceeds verifier bounds",
    )
    try:
        root_info = raw_root.lstat()
        entries = list(
            itertools.islice(raw_root.iterdir(), RELEASE_RAW_FILE_COUNT + 1)
        )
    except OSError as error:
        raise QualificationError(f"cannot inspect release raw evidence: {error}") from error
    entry_records = []
    for entry in entries:
        try:
            info = entry.lstat()
            resolved = entry.resolve(strict=True)
        except OSError as error:
            raise QualificationError(
                f"cannot inspect release raw file {entry}: {error}"
            ) from error
        entry_records.append((entry, info, resolved))
    actual_sizes = {
        entry.name: info.st_size
        for entry, info, _ in entry_records
    }
    require(
        stat.S_ISDIR(root_info.st_mode)
        and raw_root.resolve(strict=True) == raw_root
        and len(entries) == RELEASE_RAW_FILE_COUNT
        and all(
            stat.S_ISREG(info.st_mode)
            and resolved == entry
            and info.st_size <= MAX_EVIDENCE_FILE_BYTES
            for entry, info, resolved in entry_records
        )
        and sum(actual_sizes.values()) <= MAX_RELEASE_RAW_TOTAL_BYTES
        and actual_sizes == manifest_sizes,
        "release raw evidence must be one bounded flat regular-file directory",
    )


def _bounded_release_raw_tree_manifest(
    raw_root: pathlib.Path,
    repo: pathlib.Path,
    expected_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the exact v2 flat-tree manifest without unbounded traversal."""
    _validate_release_raw_manifest_bounds(raw_root, expected_manifest)
    raw_root = pathlib.Path(os.path.abspath(os.fspath(raw_root)))
    try:
        root_before = raw_root.lstat()
        entries_before = sorted(
            itertools.islice(raw_root.iterdir(), RELEASE_RAW_FILE_COUNT + 1)
        )
    except OSError as error:
        raise QualificationError(f"cannot snapshot release raw evidence: {error}") from error
    require(
        len(entries_before) == RELEASE_RAW_FILE_COUNT,
        "release raw evidence count changed before hashing",
    )
    root_fingerprint = (
        root_before.st_dev, root_before.st_ino, root_before.st_mode,
        root_before.st_size, root_before.st_mtime_ns, root_before.st_ctime_ns,
    )
    files = []
    for path in entries_before:
        before = path.lstat()
        data = _bounded_regular_file_bytes(
            path, f"release raw evidence {path.name}", MAX_EVIDENCE_FILE_BYTES
        )
        after = path.lstat()
        require(
            (
                before.st_dev, before.st_ino, before.st_mode, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns,
            )
            == (
                after.st_dev, after.st_ino, after.st_mode, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns,
            ),
            f"release raw evidence changed while hashing: {path.name}",
        )
        files.append(
            {
                "path": path.name,
                "kind": "file",
                "mode_octal": f"{stat.S_IMODE(before.st_mode):04o}",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    try:
        root_after = raw_root.lstat()
        entries_after = sorted(
            itertools.islice(raw_root.iterdir(), RELEASE_RAW_FILE_COUNT + 1)
        )
    except OSError as error:
        raise QualificationError(f"cannot replay release raw evidence: {error}") from error
    require(
        entries_after == entries_before
        and root_fingerprint
        == (
            root_after.st_dev, root_after.st_ino, root_after.st_mode,
            root_after.st_size, root_after.st_mtime_ns, root_after.st_ctime_ns,
        ),
        "release raw evidence topology changed while hashing",
    )
    payload = {
        "directories": [],
        "files": files,
        "root_mode_octal": f"{stat.S_IMODE(root_before.st_mode):04o}",
        "symlinks": [],
    }
    manifest = {
        "root": provenance.display_path(raw_root, repo),
        "file_count": RELEASE_RAW_FILE_COUNT,
        "directory_count": 0,
        "symlink_count": 0,
        "tree_manifest_schema_version": 2,
        "root_mode_octal": payload["root_mode_octal"],
        "sha256": provenance.canonical_sha256(payload),
        "files": files,
        "directories": [],
        "symlinks": [],
    }
    require(
        exact_json_equal(manifest, expected_manifest),
        "sealed raw evidence changed",
    )
    return manifest


def write_release_attestation(
    path: pathlib.Path,
    output: pathlib.Path,
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    path = path.resolve()
    output = output.resolve(strict=True)
    require(not path.exists(), "release attestation output must be fresh")
    try:
        path.relative_to(output)
    except ValueError:
        pass
    else:
        raise QualificationError(
            "release attestation must be external to its evidence bundle"
        )
    receipt = report["runtime_provenance"]["build_receipt"]
    source = report["runtime_provenance"]["source_snapshot"]
    attestation = {
        "schema": RELEASE_ATTESTATION_SCHEMA,
        "state": "complete",
        "verification_scope": RELEASE_VERIFICATION_SCOPE,
        "created_at_unix": time.time(),
        "bundle_id": manifest["bundle_id"],
        "evidence_directory": str(output),
        "evidence_manifest": file_record(output / BUNDLE_MANIFEST_NAME),
        "report_json": file_record(output / "report.json"),
        "raw_manifest_sha256": manifest["raw_manifest"]["sha256"],
        "build_receipt": receipt["file"],
        "receipt_id": receipt["receipt_id"],
        "build_input_id": receipt["build_input_id"],
        "artifact_set_id": receipt["artifact_set_id"],
        "source_snapshot_sha256": source["sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
    }
    attestation["attestation_id"] = provenance.canonical_sha256(attestation)
    provenance.atomic_write_json(path, attestation)
    return attestation


def validate_release_attestation(
    path: pathlib.Path,
    output: pathlib.Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    attestation = _read_json_object(
        path, "release attestation", MAX_RELEASE_ATTESTATION_BYTES
    )
    unsigned = dict(attestation)
    attestation_id = unsigned.pop("attestation_id", None)
    require(
        set(unsigned) == {
            "schema", "state", "verification_scope", "created_at_unix", "bundle_id",
            "evidence_directory", "evidence_manifest", "report_json",
            "raw_manifest_sha256", "build_receipt", "receipt_id",
            "build_input_id", "artifact_set_id", "source_snapshot_sha256",
            "source_manifest_sha256",
        }
        and unsigned["schema"] == RELEASE_ATTESTATION_SCHEMA
        and unsigned["state"] == "complete"
        and unsigned["verification_scope"] == RELEASE_VERIFICATION_SCOPE
        and isinstance(attestation_id, str)
        and attestation_id == provenance.canonical_sha256(unsigned),
        "release attestation identity is invalid",
    )
    _finite_positive(unsigned["created_at_unix"],
                     "release attestation creation time")
    receipt = report["runtime_provenance"]["build_receipt"]
    source = report["runtime_provenance"]["source_snapshot"]
    require(
        unsigned["bundle_id"] == manifest["bundle_id"]
        and unsigned["evidence_directory"] == str(output)
        and exact_json_equal(
            unsigned["evidence_manifest"],
            file_record(output / BUNDLE_MANIFEST_NAME),
        )
        and exact_json_equal(
            unsigned["report_json"], file_record(output / "report.json")
        )
        and unsigned["raw_manifest_sha256"]
        == manifest["raw_manifest"]["sha256"]
        and exact_json_equal(unsigned["build_receipt"], receipt["file"])
        and unsigned["receipt_id"] == receipt["receipt_id"]
        and unsigned["build_input_id"] == receipt["build_input_id"]
        and unsigned["artifact_set_id"] == receipt["artifact_set_id"]
        and unsigned["source_snapshot_sha256"] == source["sha256"]
        and unsigned["source_manifest_sha256"]
        == source["source_manifest_sha256"],
        "release attestation differs from the receipt-bound evidence bundle",
    )
    return attestation


def verify_evidence_bundle(
    output: pathlib.Path,
    repo: pathlib.Path,
    *,
    expected_bundle_id: str | None = None,
    expected_qualification_tier: str | None = None,
    release_attestation: pathlib.Path | None = None,
    allow_unattested_release: bool = False,
) -> dict[str, Any]:
    try:
        output = output.resolve(strict=True)
        repo = repo.resolve(strict=True)
    except OSError as error:
        raise QualificationError(f"cannot resolve evidence bundle: {error}") from error
    require(output.is_dir(), "evidence bundle is not a directory")
    expected_names = {
        "COMPLETE", BUNDLE_MANIFEST_NAME, "raw", "report.json", "report.md"
    }
    top_level_entries = list(
        itertools.islice(output.iterdir(), len(expected_names) + 1)
    )
    require(
        len(top_level_entries) == len(expected_names)
        and {path.name for path in top_level_entries} == expected_names,
        "evidence bundle top-level file set differs",
    )
    require(
        expected_qualification_tier in {None, "development", "release"},
        "expected qualification tier is invalid",
    )
    external_authority_supplied = (
        expected_bundle_id is not None or release_attestation is not None
    )
    require(
        not external_authority_supplied
        or (expected_bundle_id is not None and release_attestation is not None),
        "release authority requires both a bundle ID and attestation",
    )
    require(
        expected_qualification_tier != "release" or external_authority_supplied,
        "release-tier verification requires a bundle ID and attestation",
    )
    complete_path = output / "COMPLETE"
    manifest_path = output / BUNDLE_MANIFEST_NAME
    manifest = _read_json_object(
        manifest_path, "evidence manifest", MAX_BUNDLE_MANIFEST_BYTES
    )
    unsigned_manifest = dict(manifest)
    recorded_bundle_id = unsigned_manifest.pop("bundle_id", None)
    require(
        set(manifest) == {
            "schema", "state", "qualification_tier", "files",
            "raw_manifest", "build_receipt", "bundle_id",
        }
        and manifest.get("schema") == BUNDLE_SCHEMA
        and manifest.get("state") == "complete"
        and recorded_bundle_id == provenance.canonical_sha256(unsigned_manifest),
        "evidence manifest identity is invalid",
    )
    if external_authority_supplied:
        require(
            isinstance(expected_bundle_id, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected_bundle_id) is not None
            and recorded_bundle_id == expected_bundle_id,
            "release verification requires the externally pinned bundle ID",
        )
    manifest_tier = manifest.get("qualification_tier")
    require(
        manifest_tier in {"development", "release"},
        "evidence manifest qualification tier is invalid",
    )
    if expected_qualification_tier is not None:
        require(
            manifest_tier == expected_qualification_tier,
            "evidence does not match the caller-required qualification tier",
        )
    if external_authority_supplied:
        require(
            manifest_tier == "release",
            "release authority inputs cannot verify a non-release bundle",
        )
    if manifest_tier == "release" and not allow_unattested_release:
        require(
            external_authority_supplied,
            "release verification requires external authority inputs",
        )
    if manifest_tier == "release":
        _validate_release_raw_manifest_bounds(
            output / "raw", manifest.get("raw_manifest")
        )
    complete = _read_json_object(
        complete_path, "COMPLETE seal", MAX_COMPLETE_BYTES
    )
    expected_complete = {
        "schema": BUNDLE_SCHEMA,
        "state": "complete",
        "bundle_id": recorded_bundle_id,
        "evidence_manifest": _bounded_provenance_file_record(
            manifest_path, output, "evidence manifest", MAX_BUNDLE_MANIFEST_BYTES
        ),
    }
    require(
        exact_json_equal(complete, expected_complete),
        "COMPLETE seal differs from evidence manifest",
    )
    report_path = output / "report.json"
    markdown_path = output / "report.md"
    expected_files = {
        "report_json": _bounded_provenance_file_record(
            report_path, output, "Near2Far report", MAX_REPORT_BYTES
        ),
        "report_markdown": _bounded_provenance_file_record(
            markdown_path,
            output,
            "Near2Far Markdown report",
            MAX_REPORT_MARKDOWN_BYTES,
        ),
    }
    require(
        exact_json_equal(manifest.get("files"), expected_files),
        "sealed report files changed",
    )
    report = _read_json_object(report_path, "Near2Far report", MAX_REPORT_BYTES)
    require(
        _bounded_regular_file_bytes(
            markdown_path, "Near2Far Markdown report", MAX_REPORT_MARKDOWN_BYTES
        ).decode("utf-8")
        == _report_markdown(report),
        "Near2Far Markdown report differs from the JSON claims",
    )
    require(
        report.get("schema") == SCHEMA and report.get("state") == "complete",
        "Near2Far report is not complete",
    )
    runtime_provenance = report.get("runtime_provenance")
    require(isinstance(runtime_provenance, dict), "report runtime provenance is absent")
    tier = runtime_provenance.get("qualification_tier")
    require(
        tier in {"development", "release"}
        and manifest.get("qualification_tier") == tier,
        "evidence qualification tier differs",
    )
    require(
        expected_qualification_tier is None
        or tier == expected_qualification_tier,
        "report does not match the caller-required qualification tier",
    )
    current_raw_manifest = (
        _bounded_release_raw_tree_manifest(
            output / "raw", output, manifest["raw_manifest"]
        )
        if tier == "release"
        else provenance.tree_manifest(output / "raw", output)
    )
    require(
        current_raw_manifest.get("symlink_count") == 0
        and exact_json_equal(manifest.get("raw_manifest"), current_raw_manifest),
        "sealed raw evidence changed",
    )
    require(
        exact_json_equal(
            _report_raw_file_records(report, output / "raw"),
            _manifest_raw_file_records(current_raw_manifest),
        ),
        "report/raw evidence file sets differ",
    )
    validate_report_semantics(report, output / "raw")
    receipt_summary = runtime_provenance.get("build_receipt")
    sealed_receipt = manifest.get("build_receipt")
    if tier == "release":
        require(
            isinstance(receipt_summary, dict) and isinstance(sealed_receipt, dict),
            "release evidence lacks a sealed build receipt",
        )
    if receipt_summary is not None or sealed_receipt is not None:
        require(
            isinstance(receipt_summary, dict) and isinstance(sealed_receipt, dict)
            and exact_json_equal(receipt_summary.get("file"), sealed_receipt),
            "report and outer seal bind different build receipts",
        )
        receipt_path = _record_path(sealed_receipt, repo)
        require(
            file_record(receipt_path) == sealed_receipt,
            "sealed build receipt file changed",
        )
        try:
            verified_receipt = provenance.verify_build_receipt(
                receipt_path, repo, verify_source=True
            )
        except (OSError, RuntimeError) as error:
            raise QualificationError(
                f"sealed build receipt verification failed: {error}"
            ) from error
        if tier == "release":
            require_release_receipt_contract(verified_receipt)
        for key in ("receipt_id", "build_input_id", "artifact_set_id"):
            require(
                receipt_summary.get(key) == verified_receipt.get(key),
                f"report build receipt {key} differs",
            )
        bound_artifacts = require_release_runtime_binding(
            runtime_provenance["runtime"],
            runtime_provenance["loaded_dependencies"],
            verified_receipt,
            repo,
        )
        require(
            exact_json_equal(
                receipt_summary.get("bound_artifacts"), bound_artifacts
            ),
            "report receipt-bound runtime artifacts differ",
        )
        require(
            runtime_provenance.get("git_head")
            == verified_receipt.get("git_head")
            and runtime_provenance.get("git_status_porcelain")
            == verified_receipt.get("git_status_porcelain")
            and exact_json_equal(
                runtime_provenance.get("source_snapshot"),
                verified_receipt.get("source_end"),
            ),
            "report source provenance differs from the sealed build receipt",
        )
    else:
        require(
            tier != "release",
            "release evidence lacks a sealed build receipt",
        )
    if (tier == "release" or external_authority_supplied) and not allow_unattested_release:
        validate_release_attestation(
            release_attestation, output, manifest, report
        )
    require(
        exact_json_equal(
            complete,
            _read_json_object(
            complete_path, "terminal COMPLETE seal", MAX_COMPLETE_BYTES
            ),
        )
        and exact_json_equal(
            manifest,
            _read_json_object(
            manifest_path,
            "terminal evidence manifest",
            MAX_BUNDLE_MANIFEST_BYTES,
            ),
        )
        and exact_json_equal(
            expected_files,
            {
            "report_json": _bounded_provenance_file_record(
                report_path, output, "terminal Near2Far report", MAX_REPORT_BYTES
            ),
            "report_markdown": _bounded_provenance_file_record(
                markdown_path,
                output,
                "terminal Near2Far Markdown report",
                MAX_REPORT_MARKDOWN_BYTES,
            ),
            },
        )
        and exact_json_equal(
            current_raw_manifest,
            (
            _bounded_release_raw_tree_manifest(
                output / "raw", output, manifest["raw_manifest"]
            )
            if tier == "release"
            else provenance.tree_manifest(output / "raw", output)
            ),
        ),
        "evidence bundle changed during verification",
    )
    return manifest


def main() -> int:
    args = parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    if args.verify_evidence is not None:
        require(args.output is None, "--verify-evidence cannot be combined with --output")
        require(
            args.qualification_tier == "development",
            "verification uses --expected-qualification-tier, not --qualification-tier",
        )
        require(
            args.write_release_attestation is None,
            "--verify-evidence cannot write a release attestation",
        )
        manifest = verify_evidence_bundle(
            args.verify_evidence,
            repo,
            expected_bundle_id=args.expected_bundle_id,
            expected_qualification_tier=args.expected_qualification_tier,
            release_attestation=args.release_attestation,
        )
        attestation = (
            _read_json_object(
                args.release_attestation.resolve(strict=True),
                "release attestation",
                MAX_RELEASE_ATTESTATION_BYTES,
            )
            if args.release_attestation is not None
            else None
        )
        suffix = (
            f" attestation {attestation['attestation_id']}"
            if attestation is not None
            else ""
        )
        print(
            f"verified Near2Far evidence bundle {manifest['bundle_id']}"
            f"{suffix}"
        )
        return 0
    require(args.output is not None, "--output is required for qualification")
    require(
        args.expected_qualification_tier is None,
        "evidence generation cannot accept an expected qualification tier",
    )
    require(
        1 < args.batch_targets <= MAX_EVIDENCE_TARGETS,
        "--batch-targets is outside evidence bounds",
    )
    require(
        0 < args.frequency_count <= MAX_EVIDENCE_FREQUENCIES
        and 3 <= args.repetitions <= MAX_EVIDENCE_REPETITIONS
        and args.batch_targets * args.frequency_count * 6
        <= MAX_EVIDENCE_FIELDS,
        "frequency/repetition counts are outside evidence bounds",
    )
    require(args.cpu_threads > 0, "--cpu-threads must be positive")
    require(math.isfinite(args.lane_timeout_seconds) and args.lane_timeout_seconds > 0,
            "--lane-timeout-seconds must be finite and positive")
    require(args.stdout_limit_bytes > 0 and args.stderr_limit_bytes > 0,
            "lane output limits must be positive")
    require(args.qualification_tier != "release" or args.build_receipt is not None,
            "release qualification requires --build-receipt")
    require(
        args.qualification_tier != "release"
        or args.write_release_attestation is not None,
        "release qualification requires --write-release-attestation",
    )
    require(
        args.expected_bundle_id is None and args.release_attestation is None,
        "evidence generation cannot accept verification authority inputs",
    )
    for value in (
        args.small_minimum_speedup,
        args.batch_minimum_speedup,
        args.batch_minimum_two_gpu_scaling,
    ):
        require(math.isfinite(value) and value > 0, "speed gates must be finite and positive")
    validate_release_speed_gates(args)
    executable = args.executable.resolve()
    require(executable.is_file() and os.access(executable, os.X_OK), "gpu-step-db executable is absent")
    runtime_provenance = capture_runtime_provenance(
        executable,
        repo,
        args.build_receipt,
        args.qualification_tier,
    )
    devices = inventory_gpus()
    output = args.output.resolve()
    require(not output.exists(), "output must be a fresh, absent path")
    output.mkdir(parents=True)
    raw_root = output / "raw"
    raw_root.mkdir()
    failure_abort_probe = run_failure_abort_probe(
        executable,
        raw_root,
        timeout_seconds=min(30.0, args.lane_timeout_seconds),
        stdout_limit=min(args.stdout_limit_bytes, 1024 * 1024),
        stderr_limit=min(args.stderr_limit_bytes, 1024 * 1024),
    )
    probe_options = {
        "timeout_seconds": min(30.0, args.lane_timeout_seconds),
        "stdout_limit": min(args.stdout_limit_bytes, 1024 * 1024),
        "stderr_limit": min(args.stderr_limit_bytes, 1024 * 1024),
    }
    collective_safety_probes = [
        run_collective_safety_probe(
            executable, raw_root, "selection-optout-batch",
            selection_mismatch="optout", api="batch", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "selection-backend-grid",
            selection_mismatch="backend", api="grid", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "chunkless-rank-batch",
            chunkless_rank=True, **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-point-count",
            metadata_mismatch="point-count", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-null-points",
            metadata_mismatch="null-points", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-target-coordinates",
            metadata_mismatch="target-coordinates", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-frequency-count",
            metadata_mismatch="frequency-count", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-frequency-values",
            metadata_mismatch="frequency-values", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-material-periodic",
            metadata_mismatch="material-periodic", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-monitor-dimension",
            metadata_mismatch="monitor-dimension", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-greencyl-tol",
            metadata_mismatch="greencyl-tol", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-grid-shape",
            metadata_mismatch="grid-shape", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-grid-monitor-dimension",
            metadata_mismatch="grid-monitor-dimension", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "metadata-grid-frequency-count",
            metadata_mismatch="grid-frequency-count", **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "cross-rank-cancellation",
            cross_rank_cancellation=True, **probe_options,
        ),
        run_collective_safety_probe(
            executable, raw_root, "distributed-lowlevel-mixed",
            distributed_lowlevel=True, **probe_options,
        ),
    ]
    def measured_lane(
        name: str, backend: str, ranks: int, threads: int, targets: int
    ) -> dict[str, Any]:
        return run_lane(
            executable,
            raw_root,
            name,
            backend,
            ranks,
            threads,
            targets,
            args.frequency_count,
            args.repetitions,
            timeout_seconds=args.lane_timeout_seconds,
            stdout_limit=args.stdout_limit_bytes,
            stderr_limit=args.stderr_limit_bytes,
        )

    lanes: list[dict[str, Any]] = []
    scenarios = []
    for scenario_name, targets in (("single-point", 1), ("batch", args.batch_targets)):
        cpu_lanes = [
            measured_lane(f"{scenario_name}-cpu-1x1", "cpu", 1, 1, targets),
            measured_lane(f"{scenario_name}-cpu-1x{args.cpu_threads}", "cpu", 1,
                          args.cpu_threads, targets),
            measured_lane(f"{scenario_name}-cpu-2x{max(1, args.cpu_threads // 2)}",
                          "cpu", 2, max(1, args.cpu_threads // 2), targets),
        ]
        one_gpu = measured_lane(
            f"{scenario_name}-cuda-1", "cuda", 1, 1, targets
        )
        two_gpu = measured_lane(
            f"{scenario_name}-cuda-2", "cuda", 2, 1, targets
        )
        lanes.extend([*cpu_lanes, one_gpu, two_gpu])
        best_cpu = min(cpu_lanes, key=lambda lane: lane["median_seconds"])
        cpu_one = best_cpu["median_seconds"] / one_gpu["median_seconds"]
        cpu_two = best_cpu["median_seconds"] / two_gpu["median_seconds"]
        gpu_scaling = one_gpu["median_seconds"] / two_gpu["median_seconds"]
        numeric_errors = validate_numeric_agreement(
            scenario_name,
            cpu_lanes[0]["records"][0]["fields"],
            best_cpu["records"][0]["fields"],
            one_gpu["records"][0]["fields"],
            two_gpu["records"][0]["fields"],
        )
        required_speedup = (
            args.small_minimum_speedup if scenario_name == "single-point"
            else args.batch_minimum_speedup
        )
        require(cpu_one >= required_speedup,
                f"{scenario_name} 1-GPU speedup {cpu_one:.3f}x is below {required_speedup:.3f}x")
        if scenario_name == "batch":
            require(gpu_scaling >= args.batch_minimum_two_gpu_scaling,
                    f"batch 2-GPU scaling {gpu_scaling:.3f}x is below gate")
        validate_gpu_identity(scenario_name, one_gpu, two_gpu, devices)
        scenarios.append({
            "name": scenario_name,
            "targets": targets,
            "best_cpu_lane": best_cpu["name"],
            "best_cpu_seconds": best_cpu["median_seconds"],
            "one_gpu_seconds": one_gpu["median_seconds"],
            "two_gpu_seconds": two_gpu["median_seconds"],
            "cpu_to_one_gpu_speedup": cpu_one,
            "cpu_to_two_gpu_speedup": cpu_two,
            "one_to_two_gpu_scaling": gpu_scaling,
            **numeric_errors,
            "two_gpu_error": numeric_errors["one_to_two_gpu_error"],
            "required_one_gpu_speedup": required_speedup,
        })
    terminal_provenance = capture_runtime_provenance(
        executable,
        repo,
        args.build_receipt,
        args.qualification_tier,
    )
    require(
        terminal_provenance == runtime_provenance,
        "source, measured ELF, loaded libraries, or build receipt changed "
        "during qualification",
    )
    terminal_devices = inventory_gpus()
    validate_terminal_gpu_inventory(devices, terminal_devices)
    configuration = vars(args).copy()
    for verification_option in (
        "expected_bundle_id", "expected_qualification_tier", "release_attestation",
        "write_release_attestation",
    ):
        configuration.pop(verification_option)
    configuration.update(
        executable=str(executable),
        output=str(output),
        build_receipt=(
            str(
                (
                    args.build_receipt
                    if args.build_receipt.is_absolute()
                    else repo / args.build_receipt
                ).resolve(strict=True)
            )
            if args.build_receipt is not None
            else None
        ),
    )
    report = {
        "schema": SCHEMA,
        "state": "complete",
        "created_at_unix": time.time(),
        "executable": file_record(executable),
        "gpu_inventory": devices,
        "gpu_inventory_after": terminal_devices,
        "cpu_inventory": cpu_inventory(),
        "runtime_provenance": runtime_provenance,
        "rank_local_failure_abort_probe": failure_abort_probe,
        "collective_safety_probes": collective_safety_probes,
        "configuration": configuration,
        "lanes": lanes,
        "scenarios": scenarios,
    }
    manifest = write_report(output, report, repo)
    if args.qualification_tier == "release":
        attestation = write_release_attestation(
            args.write_release_attestation, output, report, manifest
        )
        verify_evidence_bundle(
            output,
            repo,
            expected_bundle_id=manifest["bundle_id"],
            expected_qualification_tier="release",
            release_attestation=args.write_release_attestation,
        )
        print(
            f"Near2Far release bundle ID: {manifest['bundle_id']}\n"
            f"Near2Far release attestation ID: {attestation['attestation_id']}"
        )
    print(output / "report.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationError as error:
        print(f"BLOCKED/FAILED: {error}", file=sys.stderr)
        raise SystemExit(4)
