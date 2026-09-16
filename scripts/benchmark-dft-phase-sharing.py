#!/usr/bin/env python3
"""Paired, evidence-preserving A/B benchmark for CUDA DFT phase sharing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any


BENCHMARK_PREFIX = "gpmeep-multi-benchmark-v4:"
DEVICE_PREFIX = "gpmeep-multi-device-v1:"
PROCESS_PREFIX = "gpmeep-multi-process-v1:"
INITIAL_CONDITION_PREFIX = "gpmeep-initial-condition-v1:"
SOURCE_PROFILE_PREFIX = "gpmeep-source-profile-v1:"
INITIALIZATION_TIMING_PREFIX = "gpmeep-initialize-field-timing-v1:"
DFT_PREFIX = "gpmeep-dft-phase-sharing-v1:"
DFT_MULTI_PREFIX = "gpmeep-dft-multi-monitor-batch-v1:"
DFT_WARMUP_PLAN_PREFIX = "gpmeep-dft-warmup-plan-v1:"
PHASE_BATCH_POLICY_PREFIX = "gpmeep-phase-batch-policy-v1:"
CURL_PHASE_REPLAY_PREFIX = "gpmeep-curl-phase-replay-v1:"
EAGER_PREFIX = "gpmeep-eager-mpi-v1:"
PINGPONG_PREFIX = "gpmeep-receive-pingpong-v1:"
GRAPH_PREFIX = "gpmeep-boundary-phase-graph-v1:"
WORKER_GPMEEP_PREFIXES = (
    BENCHMARK_PREFIX,
    DEVICE_PREFIX,
    PROCESS_PREFIX,
    SOURCE_PROFILE_PREFIX,
    INITIAL_CONDITION_PREFIX,
    INITIALIZATION_TIMING_PREFIX,
    GRAPH_PREFIX,
    EAGER_PREFIX,
    PINGPONG_PREFIX,
    "gpmeep-boundary-eh-overlap-v1:",
    "gpmeep-halo-curl-overlap-v1:",
    "gpmeep-tile-coalescing-v1:",
    PHASE_BATCH_POLICY_PREFIX,
    CURL_PHASE_REPLAY_PREFIX,
    "gpmeep-boundary-descriptor-replay-v1:",
    DFT_PREFIX,
    DFT_MULTI_PREFIX,
    DFT_WARMUP_PLAN_PREFIX,
)
OBSERVABLES = (
    "cut_probe_values",
    "cut_probe_l2",
    "ez_l2",
    "ez_weighted_checksum",
    "energy",
    "dft_norm",
    "remote_cut_scalars",
    "remote_cut_l2",
)
SMALL_PAIRS = 8
LARGE_PAIRS = 6
SMALL_MINIMUM_MEDIAN_SPEEDUP_PERCENT = 5.0
LARGE_MAXIMUM_T90_TIME_RATIO = 1.01
T90_TWO_SIDED = {5: 2.0150483733330233, 7: 1.894578605061305}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: pathlib.Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_bytes(path: pathlib.Path, value: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path: pathlib.Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def file_record(path: pathlib.Path, root: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_manifest(result_dir: pathlib.Path) -> pathlib.Path:
    manifest_path = result_dir / "evidence-manifest.json"
    files = sorted(
        path
        for path in result_dir.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "home" not in path.relative_to(result_dir).parts
    )
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "files": [file_record(path, result_dir) for path in files],
        },
    )
    return manifest_path


def paired_log_interval(ratios: list[float]) -> dict[str, float]:
    degrees_of_freedom = len(ratios) - 1
    if degrees_of_freedom not in T90_TWO_SIDED:
        raise RuntimeError(
            f"no predeclared t90 critical value for df={degrees_of_freedom}"
        )
    logs = [math.log(value) for value in ratios]
    mean = statistics.mean(logs)
    margin = (
        T90_TWO_SIDED[degrees_of_freedom]
        * statistics.stdev(logs)
        / math.sqrt(len(logs))
    )
    return {
        "geometric_time_ratio_on_over_off": math.exp(mean),
        "t90_lower_time_ratio": math.exp(mean - margin),
        "t90_upper_time_ratio": math.exp(mean + margin),
    }


def command_output(
    argv: list[str], environment: dict[str, str] | None = None
) -> str:
    result = subprocess.run(
        argv,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {argv!r}\n{result.stdout}"
        )
    return result.stdout


def normalized_ldd_dependencies(output: str) -> list[str]:
    """Discard only ASLR load addresses from otherwise exact ldd rows."""
    normalized = []
    for line in output.splitlines():
        row = re.sub(r"\s+\(0x[0-9a-fA-F]+\)\s*$", "", line).strip()
        if row:
            normalized.append(row)
    return sorted(normalized)


def resolve_receipt_artifacts(
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    expected_receipt_id: str,
    requested_mpiexec: pathlib.Path,
) -> tuple[dict[str, Any], pathlib.Path, pathlib.Path, str]:
    sys.path.insert(0, str(repo / "scripts"))
    from gpmeep_provenance import verify_build_receipt

    receipt = verify_build_receipt(receipt_path, repo)
    if receipt.get("receipt_id") != expected_receipt_id:
        raise RuntimeError(
            "verified build receipt differs from the predeclared receipt ID"
        )
    if receipt.get("build_kind") != "cuda-mpi-python-fp32":
        raise RuntimeError("verified build receipt has the wrong build kind")
    toolchain = receipt.get("toolchain")
    mpiexec_record = (
        toolchain.get("mpiexec") if isinstance(toolchain, dict) else None
    )
    if not isinstance(mpiexec_record, dict):
        raise RuntimeError("verified build receipt has no mpiexec tool record")
    recorded_path_value = mpiexec_record.get("path")
    invoked_path_value = mpiexec_record.get("invoked_path")
    if not isinstance(recorded_path_value, str) or not isinstance(
        invoked_path_value, str
    ):
        raise RuntimeError("verified mpiexec tool record is malformed")
    recorded_mpiexec = pathlib.Path(recorded_path_value)
    if not recorded_mpiexec.is_absolute():
        recorded_mpiexec = repo / recorded_mpiexec
    recorded_mpiexec = recorded_mpiexec.resolve()
    invoked_mpiexec = pathlib.Path(invoked_path_value).resolve()
    requested_mpiexec = requested_mpiexec.resolve()
    if (
        requested_mpiexec != recorded_mpiexec
        or invoked_mpiexec != recorded_mpiexec
        or not recorded_mpiexec.is_file()
        or recorded_mpiexec.stat().st_size != mpiexec_record.get("size_bytes")
        or sha256_file(recorded_mpiexec) != mpiexec_record.get("sha256")
    ):
        raise RuntimeError(
            "requested mpiexec is not the exact receipt-recorded tool"
        )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("verified build receipt has no artifact map")

    def artifact(name: str) -> pathlib.Path:
        record = artifacts.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError(f"verified build receipt lacks {name!r}")
        path = pathlib.Path(record["path"])
        if not path.is_absolute():
            path = repo / path
        path = path.resolve()
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"receipt artifact changed: {path}")
        return path

    executable = artifact("gpu_mpi_performance")
    library = artifact("libmeep")
    dependency_prefix = recorded_mpiexec.parent.parent
    ldd_environment = {
        "HOME": "/tmp",
        "PATH": f"{dependency_prefix / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": (
            f"{library.parent}:{dependency_prefix / 'lib'}"
        ),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    ldd_output = command_output(
        ["/usr/bin/ldd", str(executable)], ldd_environment
    )
    matches = []
    for line in ldd_output.splitlines():
        match = re.match(r"\s*libmeep\.so[^ ]*\s+=>\s+(/[^ ]+)", line)
        if match:
            matches.append(pathlib.Path(match.group(1)).resolve())
    if matches != [library]:
        raise RuntimeError(
            f"direct benchmark ELF resolves unexpected libmeep: {matches}"
        )
    return receipt, executable, library, ldd_output


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate JSON key in benchmark record: {key}")
        value[key] = item
    return value


def reject_nonfinite_json_constant(value: str) -> None:
    raise RuntimeError(f"non-finite JSON constant in benchmark record: {value}")


def records(output: str, prefix: str) -> list[Any]:
    result = []
    for line in output.replace("\r", "\n").splitlines():
        position = line.find(prefix)
        if position >= 0:
            result.append(
                json.loads(
                    line[position + len(prefix) :],
                    object_pairs_hook=reject_duplicate_json_keys,
                    parse_constant=reject_nonfinite_json_constant,
                )
            )
    return result


def exactly_one(output: str, prefix: str, label: str) -> dict[str, Any]:
    found = records(output, prefix)
    if len(found) != 1 or not isinstance(found[0], dict):
        raise RuntimeError(
            f"expected exactly one {label} record, found {len(found)}"
        )
    return found[0]


def reject_unknown_gpmeep_markers(output: str) -> None:
    for line in output.replace("\r", "\n").splitlines():
        position = line.find("gpmeep-")
        if position >= 0 and not any(
            line.startswith(prefix, position)
            for prefix in WORKER_GPMEEP_PREFIXES
        ):
            raise RuntimeError(
                "worker output contains an unknown or mixed-version gpmeep marker"
            )


def validate_initialization_markers(
    output: str,
    benchmark: dict[str, Any],
    expected_profile: str = "trigonometric-v1",
    expected_applications_per_rank: int = 1,
    expected_source_profile: str = "single-ez-v1",
) -> dict[str, dict[str, Any]]:
    source_profile = exactly_one(
        output, SOURCE_PROFILE_PREFIX, "source-profile marker"
    )
    if set(source_profile) != {"profile"}:
        raise RuntimeError("source-profile marker has an invalid exact schema")
    if (
        type(source_profile["profile"]) is not str
        or source_profile["profile"] != expected_source_profile
    ):
        raise RuntimeError(
            "source-profile marker differs from the fixed profile"
        )
    initial_condition = exactly_one(
        output, INITIAL_CONDITION_PREFIX, "initial-condition marker"
    )
    if set(initial_condition) != {"profile"}:
        raise RuntimeError("initial-condition marker has an invalid exact schema")
    profile = initial_condition["profile"]
    if type(profile) is not str or profile != expected_profile:
        raise RuntimeError(
            "initial-condition marker differs from the fixed profile"
        )
    if benchmark.get("initial_condition") != profile:
        raise RuntimeError(
            "initial-condition marker and worker record differ"
        )

    timing = exactly_one(
        output,
        INITIALIZATION_TIMING_PREFIX,
        "initialize-field timing marker",
    )
    if set(timing) != {"applications_per_rank", "max_seconds"}:
        raise RuntimeError(
            "initialize-field timing marker has an invalid exact schema"
        )
    applications = timing["applications_per_rank"]
    seconds = timing["max_seconds"]
    if (
        type(expected_applications_per_rank) is not int
        or expected_applications_per_rank <= 0
    ):
        raise RuntimeError("expected initialization application count is invalid")
    if (
        type(applications) is not int
        or applications != expected_applications_per_rank
    ):
        if expected_applications_per_rank == 1:
            raise RuntimeError(
                "initialize-field timing marker must report one application per rank"
            )
        raise RuntimeError(
            "initialize-field timing marker application count differs from "
            f"expected {expected_applications_per_rank} per rank"
        )
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(float(seconds))
        or seconds <= 0
    ):
        raise RuntimeError(
            "initialize-field timing marker max_seconds must be finite and positive"
        )
    return {
        "source_profile": source_profile,
        "initial_condition": {"profile": profile},
        "initialization_timing": {
            "applications_per_rank": applications,
            "max_seconds": float(seconds),
        },
    }


def gpu_snapshot() -> dict[str, str]:
    return {
        "inventory": command_output(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,"
                "power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.total,"
                "utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "compute_apps": command_output(
            [
                "/usr/bin/nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "loadavg": pathlib.Path("/proc/loadavg").read_text(
            encoding="utf-8"
        ).strip(),
    }


def require_idle(snapshot: dict[str, str]) -> None:
    if snapshot["compute_apps"].strip():
        raise RuntimeError("unrelated GPU compute process is present")


def start_process_monitor(
    path: pathlib.Path,
) -> tuple[subprocess.Popen[bytes], Any]:
    stream = path.open("wb")
    process = subprocess.Popen(
        ["/usr/bin/nvidia-smi", "pmon", "-s", "um", "-d", "1"],
        stdout=stream,
        stderr=subprocess.STDOUT,
    )
    return process, stream


def stop_process_monitor(
    process: subprocess.Popen[bytes],
    stream: Any,
    path: pathlib.Path,
    expected_benchmark_pids: int,
) -> dict[str, Any]:
    alive_before_stop = process.poll() is None
    if alive_before_stop:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()
    observed_pids: set[int] = set()
    observed_gpus: set[int] = set()
    pid_to_gpu: dict[int, int] = {}
    unrelated = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 3 or columns[1] == "-":
            continue
        try:
            gpu_index = int(columns[0])
            pid = int(columns[1])
        except ValueError:
            unrelated.append(line)
            continue
        command = columns[-1]
        if "gpu-mpi-per" not in command:
            unrelated.append(line)
            continue
        previous_gpu = pid_to_gpu.setdefault(pid, gpu_index)
        if previous_gpu != gpu_index:
            unrelated.append(line)
            continue
        observed_gpus.add(gpu_index)
        observed_pids.add(pid)
    observed_pids_per_gpu = {
        str(gpu): sum(value == gpu for value in pid_to_gpu.values())
        for gpu in sorted(observed_gpus)
    }
    expected_pids_per_gpu = expected_benchmark_pids // 2
    record = {
        "schema_version": 1,
        "monitor_command": [
            "/usr/bin/nvidia-smi",
            "pmon",
            "-s",
            "um",
            "-d",
            "1",
        ],
        "monitor_returncode": process.returncode,
        "alive_before_runner_stop": alive_before_stop,
        "expected_benchmark_pids": expected_benchmark_pids,
        "expected_pids_per_gpu": expected_pids_per_gpu,
        "observed_benchmark_pids": sorted(observed_pids),
        "observed_gpu_indices": sorted(observed_gpus),
        "observed_pids_per_gpu": observed_pids_per_gpu,
        "unrelated_process_rows": unrelated,
        "raw": path.name,
        "raw_sha256": sha256_file(path),
    }
    record["pass"] = (
        not unrelated
        and expected_benchmark_pids > 0
        and expected_benchmark_pids % 2 == 0
        and alive_before_stop
        # Current nvidia-smi handles SIGTERM and exits 0; other releases may
        # expose the uncaught signal as -SIGTERM instead.
        and process.returncode in {0, -15}
        and len(observed_pids) == expected_benchmark_pids
        and observed_gpus == {0, 1}
        and observed_pids_per_gpu
        == {"0": expected_pids_per_gpu, "1": expected_pids_per_gpu}
    )
    return record


def validate_sample(
    condition: str,
    workload: dict[str, int | str],
    output: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    reject_unknown_gpmeep_markers(output)
    benchmark = exactly_one(output, BENCHMARK_PREFIX, "benchmark")
    validate_initialization_markers(output, benchmark)
    dft = exactly_one(output, DFT_PREFIX, "DFT phase-sharing")
    dft_multi = exactly_one(
        output, DFT_MULTI_PREFIX, "multi-monitor DFT batching"
    )
    eager = exactly_one(output, EAGER_PREFIX, "eager MPI")
    pingpong = exactly_one(output, PINGPONG_PREFIX, "receive ping-pong")
    graph = exactly_one(output, GRAPH_PREFIX, "boundary graph")
    devices = records(output, DEVICE_PREFIX)
    if (
        len(devices) != 2
        or any(not isinstance(item, dict) for item in devices)
        or {item.get("rank") for item in devices} != {0, 1}
        or {item.get("ordinal") for item in devices} != {0, 1}
        or len({item.get("uuid") for item in devices}) != 2
        or any(not item.get("uuid") for item in devices)
    ):
        raise RuntimeError("sample did not prove two distinct CUDA devices")
    if (
        benchmark.get("schema_version") != 4
        or benchmark.get("mpi_ranks") != 2
        or benchmark.get("pixels") != workload["pixels"]
        or benchmark.get("warmup_steps") != workload["warmup_steps"]
        or benchmark.get("steps") != workload["steps"]
        or benchmark.get("initial_condition") != "trigonometric-v1"
    ):
        raise RuntimeError("benchmark workload identity is inconsistent")
    if (
        benchmark.get("requested_transport") != "cuda-aware"
        or benchmark.get("selected_transport") != "cuda-aware"
        or benchmark.get("cpu_calls") != 0
        or not isinstance(benchmark.get("cuda_calls"), int)
        or benchmark["cuda_calls"] <= 0
        or benchmark.get("pinned_bytes") != 0
        or benchmark.get("cuda_aware_bytes", 0) <= 0
    ):
        raise RuntimeError("sample did not use exclusive CUDA-aware GPU work")
    if min(eager.values()) <= 0:
        raise RuntimeError("sample did not prove eager MPI overlap")
    if (
        pingpong.get("warmup_secondary_allocations", 0) <= 0
        or pingpong.get("secondary_selections", 0) <= 0
        or graph.get("launches", 0) <= 0
    ):
        raise RuntimeError("sample did not prove fixed communication optimizations")
    integer_fields = (
        "batch_calls",
        "submitted_updates",
        "phase_preparation_launches",
        "phase_reuses",
        "update_kernel_launches",
        "maximum_batch_size",
    )
    if any(type(dft.get(key)) is not int for key in integer_fields):
        raise RuntimeError("DFT record has a non-integer or missing counter")
    if (
        dft["batch_calls"] <= 0
        or dft["submitted_updates"] <= 0
        or dft["maximum_batch_size"] <= 0
        or dft["update_kernel_launches"] != dft["submitted_updates"]
        or dft["phase_preparation_launches"] + dft["phase_reuses"]
        != dft["submitted_updates"]
    ):
        raise RuntimeError("DFT batch counter accounting is inconsistent")
    multi_integer_fields = (
        "automatic_checks",
        "automatic_selected",
        "automatic_rejected",
        "forced_batches",
        "batched_updates",
        "unbatched_updates",
        "plan_uploads",
        "plan_reuses",
        "metadata_host_to_device_bytes",
    )
    if set(dft_multi) != set(multi_integer_fields) or any(
        type(dft_multi.get(key)) is not int or dft_multi[key] < 0
        for key in multi_integer_fields
    ):
        raise RuntimeError(
            "multi-monitor DFT record has an invalid exact schema"
        )
    if not (
        dft_multi["automatic_checks"] == 0
        and dft_multi["automatic_selected"] == 0
        and dft_multi["automatic_rejected"] == 0
        and dft_multi["forced_batches"] == 0
        and dft_multi["batched_updates"] == 0
        and dft_multi["unbatched_updates"] == dft["submitted_updates"]
        and dft_multi["plan_uploads"] == 0
        and dft_multi["plan_reuses"] == 0
        and dft_multi["metadata_host_to_device_bytes"] == 0
    ):
        raise RuntimeError(
            "phase-sharing isolation sample used multi-monitor DFT batching"
        )
    if condition == "on" and workload["name"] == "small":
        if not (
            dft["phase_reuses"] > 0
            and dft["phase_preparation_launches"] < dft["submitted_updates"]
        ):
            raise RuntimeError("enabled sample reused no DFT phases")
    elif workload["name"] == "large" and not (
        dft["phase_reuses"] == 0
        and dft["phase_preparation_launches"] == dft["submitted_updates"]
    ):
        raise RuntimeError(
            "large no-op workload unexpectedly found a reusable DFT phase"
        )
    elif condition == "off" and not (
        dft["phase_reuses"] == 0
        and dft["phase_preparation_launches"] == dft["submitted_updates"]
    ):
        raise RuntimeError("disabled sample unexpectedly reused a DFT phase")
    return benchmark, dft, devices


def run_sample(
    args: argparse.Namespace,
    workload: dict[str, int | str],
    pair: int,
    order: int,
    condition: str,
    base_environment: dict[str, str],
) -> dict[str, Any]:
    stem = f"{workload['name']}-pair{pair:02d}-{order}-{condition}"
    before = gpu_snapshot()
    require_idle(before)
    environment = dict(base_environment)
    environment.update(
        {
            "MEEP_GPU_MULTI_PIXELS": str(workload["pixels"]),
            "MEEP_GPU_MULTI_WARMUP_STEPS": str(workload["warmup_steps"]),
            "MEEP_GPU_MULTI_STEPS": str(workload["steps"]),
            "MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH": "1",
            "MEEP_GPU_EXPECT_NO_MULTI_MONITOR_DFT_BATCH": "1",
        }
    )
    if condition == "on" and workload["name"] == "small":
        environment["MEEP_GPU_EXPECT_DFT_PHASE_SHARING"] = "1"
    elif condition == "off":
        environment["MEEP_GPU_DISABLE_DFT_PHASE_SHARING"] = "1"
        environment["MEEP_GPU_EXPECT_NO_DFT_PHASE_SHARING"] = "1"
    argv = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=10s",
        f"{args.timeout}s",
        str(args.mpiexec),
        "--bind-to",
        "none",
        "-np",
        "2",
        str(args.executable),
    ]
    started = utc_now()
    wall_start = time.perf_counter()
    result = subprocess.run(
        argv,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_seconds = time.perf_counter() - wall_start
    raw_path = args.result_dir / f"{stem}.log"
    atomic_text(raw_path, result.stdout)
    after: dict[str, str] | None = None
    benchmark: dict[str, Any] | None = None
    dft: dict[str, Any] | None = None
    devices: list[dict[str, Any]] = []
    error: str | None = None
    try:
        after = gpu_snapshot()
        require_idle(after)
        if result.returncode:
            raise RuntimeError(
                f"sample exited with status {result.returncode}"
            )
        benchmark, dft, devices = validate_sample(
            condition, workload, result.stdout
        )
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    metadata = {
        "schema_version": 1,
        "status": "PASS" if error is None else "FAIL",
        "sample": {
            "workload": workload["name"],
            "pair": pair,
            "order": order,
            "condition": condition,
        },
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "command": argv,
        # This is the complete child environment, constructed from a fixed
        # allowlist rather than inherited from the invoking shell.
        "selected_environment": dict(sorted(environment.items())),
        "external_process_wall_seconds": wall_seconds,
        "returncode": result.returncode,
        "before": before,
        "after": after,
        "devices": devices,
        "dft": dft,
        "result": benchmark,
        "raw_log": raw_path.name,
        "raw_log_sha256": sha256_file(raw_path),
        "error": error,
    }
    atomic_json(args.result_dir / f"{stem}.json", metadata)
    if error is not None:
        raise RuntimeError(f"invalid sample {stem}: {error}; see {raw_path}")
    return metadata


def execute_samples(
    args: argparse.Namespace,
    workloads: list[dict[str, int | str]],
    base_environment: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    summaries = []
    for workload in workloads:
        pair_speedups: list[float] = []
        pair_time_ratios: list[float] = []
        on_seconds: list[float] = []
        off_seconds: list[float] = []
        order_strata = {"on_first": [], "off_first": []}
        reference_observables: dict[str, Any] | None = None
        for pair in range(1, int(workload["pairs"]) + 1):
            order = ("on", "off") if pair % 2 else ("off", "on")
            paired = {}
            for order_index, condition in enumerate(order, start=1):
                sample = run_sample(
                    args,
                    workload,
                    pair,
                    order_index,
                    condition,
                    base_environment,
                )
                samples.append(sample)
                paired[condition] = sample
                current_observables = {
                    key: sample["result"][key] for key in OBSERVABLES
                }
                if reference_observables is None:
                    reference_observables = current_observables
                elif current_observables != reference_observables:
                    raise RuntimeError(
                        f"{workload['name']} repeat changed a physical observable"
                    )
            on_result = paired["on"]["result"]
            off_result = paired["off"]["result"]
            if any(on_result[key] != off_result[key] for key in OBSERVABLES):
                raise RuntimeError(
                    f"{workload['name']} pair {pair} changed a physical observable"
                )
            if workload["name"] == "large" and paired["on"]["dft"] != paired[
                "off"
            ]["dft"]:
                raise RuntimeError(
                    "large no-op phase-sharing control changed DFT counters"
                )
            on_time = float(on_result["seconds"])
            off_time = float(off_result["seconds"])
            speedup = (off_time / on_time - 1.0) * 100.0
            ratio = on_time / off_time
            pair_speedups.append(speedup)
            pair_time_ratios.append(ratio)
            on_seconds.append(on_time)
            off_seconds.append(off_time)
            order_strata[
                "on_first" if order[0] == "on" else "off_first"
            ].append(speedup)
        median_speedup = statistics.median(pair_speedups)
        interval = paired_log_interval(pair_time_ratios)
        if workload["name"] == "small":
            passed = (
                median_speedup
                >= SMALL_MINIMUM_MEDIAN_SPEEDUP_PERCENT
            )
            gate = {
                "name": "minimum_median_paired_speedup_percent",
                "threshold": SMALL_MINIMUM_MEDIAN_SPEEDUP_PERCENT,
                "observed": median_speedup,
                "pass": passed,
            }
        else:
            passed = (
                interval["t90_upper_time_ratio"]
                <= LARGE_MAXIMUM_T90_TIME_RATIO
            )
            gate = {
                "name": "maximum_t90_upper_time_ratio_on_over_off",
                "threshold": LARGE_MAXIMUM_T90_TIME_RATIO,
                "observed": interval["t90_upper_time_ratio"],
                "pass": passed,
            }
        summaries.append(
            {
                "workload": workload["name"],
                "pair_speedup_percent": pair_speedups,
                "median_paired_speedup_percent": median_speedup,
                **interval,
                "ratio_of_median_times_on_over_off":
                    statistics.median(on_seconds)
                    / statistics.median(off_seconds),
                "on_seconds": on_seconds,
                "off_seconds": off_seconds,
                "on_population_cv": statistics.pstdev(on_seconds)
                    / statistics.mean(on_seconds),
                "off_population_cv": statistics.pstdev(off_seconds)
                    / statistics.mean(off_seconds),
                "order_strata_speedup_percent": order_strata,
                "gate": gate,
                "pass": passed,
            }
        )
    return samples, summaries


def parse_args() -> argparse.Namespace:
    repo = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--build-receipt",
        type=pathlib.Path,
        default=repo
        / "build/meep-cuda-mpi-python-fp32/build-provenance.json",
    )
    parser.add_argument("--expected-receipt-id", required=True)
    parser.add_argument(
        "--mpiexec",
        type=pathlib.Path,
        default=repo / ".envs/meep-gpu-cuda-mpi/bin/mpiexec",
    )
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir = args.result_dir.resolve()
    args.build_receipt = args.build_receipt.resolve()
    args.mpiexec = args.mpiexec.resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_receipt_id):
        raise RuntimeError("expected receipt ID must be a lowercase SHA-256")
    if args.timeout <= 0:
        raise RuntimeError("timeout is invalid")
    if args.result_dir.exists() and any(args.result_dir.iterdir()):
        raise RuntimeError(
            "authoritative result directory must be absent or empty"
        )
    args.result_dir.mkdir(parents=True, exist_ok=True)
    try:
        for path in (args.build_receipt, args.mpiexec):
            if not path.is_file():
                raise RuntimeError(
                    f"required benchmark artifact is absent: {path}"
                )
        return run_authoritative(args)
    except Exception as error:
        failed_path = args.result_dir / "FAILED.json"
        if not failed_path.exists():
            atomic_json(
                failed_path,
                {
                    "schema_version": 1,
                    "state": "FAILED_PREFLIGHT_OR_EXECUTION",
                    "finished_at_utc": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
        write_manifest(args.result_dir)
        raise


def run_authoritative(args: argparse.Namespace) -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    prefix = args.mpiexec.parent.parent
    receipt, args.executable, args.library, ldd_output = (
        resolve_receipt_artifacts(
            repo,
            args.build_receipt,
            args.expected_receipt_id,
            args.mpiexec,
        )
    )
    artifact_records = {
        "executable": file_record(args.executable, repo),
        "library": file_record(args.library, repo),
        "build_receipt": file_record(args.build_receipt, repo),
        "mpiexec": file_record(args.mpiexec, repo),
    }
    base_environment = {
        "HOME": str(args.result_dir / "home"),
        "PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": f"{args.library.parent}:{prefix / 'lib'}",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MEEP_GPU_BACKEND": "cuda",
        "MEEP_GPU_STRICT": "1",
        "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
        "MEEP_GPU_MPI_TRANSPORT": "cuda-aware",
        "MEEP_GPU_MULTI_INITIAL_CONDITION": "trigonometric-v1",
        "MEEP_GPU_MULTI_SOURCE_PROFILE": "single-ez-v1",
        "MEEP_GPU_EXPECT_BOUNDARY_PHASE_GRAPH": "1",
        "MEEP_GPU_EXPECT_EAGER_MPI": "1",
        "MEEP_GPU_EXPECT_RECEIVE_PINGPONG": "1",
        "OMPI_MCA_mca_base_param_files": str(
            repo / "environment/openmpi-qualification-mca-params.conf"
        ),
        "OMPI_MCA_mca_base_component_path": str(prefix / "lib/openmpi"),
        "PMIX_MCA_mca_base_param_files": str(
            repo / "environment/openmpi-qualification-mca-params.conf"
        ),
        "PMIX_MCA_mca_base_component_path": str(prefix / "lib/pmix"),
        "PRTE_MCA_mca_base_param_files": str(
            repo / "environment/openmpi-qualification-mca-params.conf"
        ),
    }
    pathlib.Path(base_environment["HOME"]).mkdir(exist_ok=True)
    workloads = [
        {
            "name": "small",
            "pixels": 16,
            "warmup_steps": 50,
            "steps": 5000,
            "pairs": SMALL_PAIRS,
        },
        {
            "name": "large",
            "pixels": 192,
            "warmup_steps": 12,
            "steps": 80,
            "pairs": LARGE_PAIRS,
        },
    ]
    expected_benchmark_pids = sum(
        int(workload["pairs"]) * 2 * 2 for workload in workloads
    )
    archive = args.result_dir / "archive"
    archive.mkdir()
    runner_copy = archive / "benchmark-dft-phase-sharing.py"
    receipt_copy = archive / "build-provenance.json"
    executable_copy = archive / args.executable.name
    library_copy = archive / args.library.name
    mpiexec_copy = archive / "receipt-mpiexec"
    atomic_bytes(runner_copy, pathlib.Path(__file__).read_bytes())
    atomic_bytes(receipt_copy, args.build_receipt.read_bytes())
    atomic_bytes(executable_copy, args.executable.read_bytes())
    atomic_bytes(library_copy, args.library.read_bytes())
    atomic_bytes(mpiexec_copy, args.mpiexec.read_bytes())
    protocol = {
        "schema_version": 1,
        "protocol_id": "m8.11-dft-phase-sharing-authoritative-v1",
        "created_at_utc": utc_now(),
        "receipt_id": receipt["receipt_id"],
        "artifact_set_id": receipt.get("artifact_set_id"),
        "workloads": workloads,
        "pair_order": "odd:on-off;even:off-on",
        "continuous_gpu_process_gate": {
            "expected_distinct_rank_pids": expected_benchmark_pids,
            "expected_distinct_rank_pids_per_gpu": expected_benchmark_pids // 2,
            "pid_must_map_to_one_gpu": True,
            "monitor_must_live_until_runner_stop": True,
            "accepted_monitor_returncodes_after_sigterm": [0, -15],
        },
        "small_gate": {
            "minimum_median_paired_speedup_percent":
                SMALL_MINIMUM_MEDIAN_SPEEDUP_PERCENT,
            "all_physical_observables_exact": True,
            "enabled_phase_reuse_required": True,
            "disabled_phase_reuse_forbidden": True,
        },
        "large_gate": {
            "maximum_t90_upper_time_ratio_on_over_off":
                LARGE_MAXIMUM_T90_TIME_RATIO,
            "all_physical_observables_exact": True,
            "enabled_and_disabled_reuse_must_be_zero": True,
        },
        "artifacts": artifact_records,
        "archived": {
            "runner": file_record(runner_copy, args.result_dir),
            "receipt": file_record(receipt_copy, args.result_dir),
            "executable": file_record(executable_copy, args.result_dir),
            "library": file_record(library_copy, args.result_dir),
            "mpiexec": file_record(mpiexec_copy, args.result_dir),
        },
        "ldd": ldd_output,
    }
    atomic_json(args.result_dir / "protocol.json", protocol)
    atomic_json(
        args.result_dir / "run-state-start.json",
        {
            "schema_version": 1,
            "state": "RUNNING",
            "started_at_utc": utc_now(),
            "receipt_id": receipt["receipt_id"],
        },
    )
    inventory = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "artifacts": artifact_records,
        "receipt_id": receipt["receipt_id"],
        "gpu": gpu_snapshot(),
        "lscpu": command_output(["/usr/bin/lscpu"]),
        "workloads": workloads,
    }
    require_idle(inventory["gpu"])
    atomic_json(args.result_dir / "inventory.json", inventory)

    monitor_path = args.result_dir / "continuous-gpu-process-monitor.log"
    monitor_process: subprocess.Popen[bytes] | None = None
    monitor_stream: Any = None
    monitor_stopped = False
    try:
        monitor_process, monitor_stream = start_process_monitor(monitor_path)
        samples, summaries = execute_samples(
            args, workloads, base_environment
        )
        monitor_record = stop_process_monitor(
            monitor_process,
            monitor_stream,
            monitor_path,
            expected_benchmark_pids,
        )
        monitor_stopped = True
        atomic_json(
            args.result_dir / "continuous-gpu-process-monitor.json",
            monitor_record,
        )
        if not monitor_record["pass"]:
            raise RuntimeError(
                "continuous GPU process monitor did not prove exclusive "
                "two-GPU benchmark use"
            )

        receipt_after, executable_after, library_after, ldd_after = (
            resolve_receipt_artifacts(
                repo,
                args.build_receipt,
                args.expected_receipt_id,
                args.mpiexec,
            )
        )
        if (
            receipt_after["receipt_id"] != receipt["receipt_id"]
            or executable_after != args.executable
            or library_after != args.library
            or normalized_ldd_dependencies(ldd_after)
            != normalized_ldd_dependencies(ldd_output)
            or sha256_file(runner_copy) != sha256_file(pathlib.Path(__file__))
            or sha256_file(executable_copy) != sha256_file(args.executable)
            or sha256_file(library_copy) != sha256_file(args.library)
            or sha256_file(mpiexec_copy) != sha256_file(args.mpiexec)
            or sha256_file(receipt_copy) != sha256_file(args.build_receipt)
        ):
            raise RuntimeError(
                "receipt, runner, or benchmark artifact changed during the experiment"
            )

        summary = {
            "schema_version": 1,
            "finished_at_utc": utc_now(),
            "receipt_id": receipt["receipt_id"],
            "artifact_set_id": receipt.get("artifact_set_id"),
            "artifacts": artifact_records,
            "sample_count": len(samples),
            "all_observables_exact": True,
            "continuous_contention_check_pass": True,
            "workloads": summaries,
            "pass": all(item["pass"] for item in summaries),
        }
        atomic_json(args.result_dir / "summary.json", summary)
        final_state = "COMPLETE" if summary["pass"] else "FAILED_GATES"
        atomic_json(
            args.result_dir / "run-state-final.json",
            {
                "schema_version": 1,
                "state": final_state,
                "finished_at_utc": utc_now(),
                "sample_count": len(samples),
                "receipt_id": receipt["receipt_id"],
            },
        )
        if not summary["pass"]:
            atomic_json(
                args.result_dir / "FAILED.json",
                {
                    "schema_version": 1,
                    "state": "FAILED_GATES",
                    "finished_at_utc": utc_now(),
                    "summary": "summary.json",
                },
            )
        if summary["pass"]:
            atomic_json(
                args.result_dir / "COMPLETE.json",
                {
                    "schema_version": 1,
                    "state": "COMPLETE",
                    "receipt_id": receipt["receipt_id"],
                    "manifest": "evidence-manifest.json",
                    "summary": file_record(
                        args.result_dir / "summary.json", args.result_dir
                    ),
                },
            )
        write_manifest(args.result_dir)
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["pass"] else 2
    except Exception as error:
        if not monitor_stopped:
            try:
                if monitor_process is None:
                    raise RuntimeError("continuous GPU process monitor did not start")
                monitor_record = stop_process_monitor(
                    monitor_process,
                    monitor_stream,
                    monitor_path,
                    expected_benchmark_pids,
                )
                atomic_json(
                    args.result_dir / "continuous-gpu-process-monitor.json",
                    monitor_record,
                )
            except Exception as monitor_error:
                atomic_json(
                    args.result_dir / "continuous-gpu-process-monitor-error.json",
                    {
                        "schema_version": 1,
                        "error": f"{type(monitor_error).__name__}: {monitor_error}",
                    },
                )
        atomic_json(
            args.result_dir / "FAILED.json",
            {
                "schema_version": 1,
                "state": "FAILED_EXECUTION",
                "finished_at_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        write_manifest(args.result_dir)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
