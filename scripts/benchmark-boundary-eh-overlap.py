#!/usr/bin/env python3
"""Receipt-bound paired A/B for CUDA-aware boundary/E-H overlap.

Valid negative performance evidence is published as COMPLETE. The report
separately decides whether the optimization is fast and stable enough to stay
enabled by default; correctness is always exact and fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
import time
import traceback
import types
from typing import Any


def load_source_module(name: str, path: pathlib.Path) -> types.ModuleType:
    resolved = path.resolve()
    module = types.ModuleType(name)
    module.__file__ = str(resolved)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(resolved.read_bytes(), str(resolved), "exec"), module.__dict__)
    return module


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
HELPER = load_source_module(
    "gpmeep_dft_evidence_helper_for_boundary_overlap",
    SCRIPT_DIR / "benchmark-dft-phase-sharing.py",
)

PROFILE_ID = "m8.11-boundary-eh-overlap-ab-v1"
PROFILE = {
    "profile_id": PROFILE_ID,
    "mpi_ranks": 2,
    "pixels": 192,
    "warmup_steps": 12,
    "measured_steps": 80,
    "warmup_processes_per_condition": 1,
    "measured_pairs": 6,
    "transport": "cuda-aware",
    "completion_policy": "waitsome",
    "initial_condition": "trigonometric-v1",
    "source_enabled": False,
    "overlap_material": True,
    "minimum_median_paired_speedup": 1.01,
    "minimum_faster_pairs": 4,
    "maximum_t90_upper_time_ratio": 1.0,
    "both_position_strata_must_be_positive": True,
}
PROFILE_SHA256 = "9eb662ba78a2faecda3751e642464378d3924edb3fd295658e0f046d6a3a9ed1"

BENCHMARK_PREFIX = "gpmeep-multi-benchmark-v4:"
DEVICE_PREFIX = "gpmeep-multi-device-v1:"
OVERLAP_PREFIX = "gpmeep-boundary-eh-overlap-v1:"
EAGER_PREFIX = "gpmeep-eager-mpi-v1:"
PINGPONG_PREFIX = "gpmeep-receive-pingpong-v1:"
GRAPH_PREFIX = "gpmeep-boundary-phase-graph-v1:"
DFT_PREFIX = "gpmeep-dft-phase-sharing-v1:"

# Wrapper runners import this module as their execution engine.  They must
# archive that otherwise-hidden source dependency alongside themselves so the
# published evidence remains executable and independently reviewable.
ADDITIONAL_RUNNER_SOURCES: dict[str, pathlib.Path] = {}

OBSERVABLE_KEYS = (
    "cut_probe_values",
    "cut_probe_l2",
    "ez_l2",
    "ez_weighted_checksum",
    "energy",
    "dft_norm",
    "remote_cut_scalars",
    "remote_cut_l2",
    "cpu_calls",
    "cuda_calls",
    "phase_calls",
    "h2d_bytes",
    "d2h_bytes",
    "mpi_messages",
    "mpi_scalars",
    "cuda_aware_bytes",
    "pinned_bytes",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_fixed_profile() -> None:
    if canonical_sha256(PROFILE) != PROFILE_SHA256:
        raise RuntimeError("fixed boundary overlap A/B profile changed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expected-receipt-id", required=True)
    parser.add_argument(
        "--mpiexec",
        type=pathlib.Path,
        default=REPO / ".envs/meep-gpu-cuda-mpi/bin/mpiexec",
    )
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args(argv)


def expected_schedule() -> list[tuple[str, int, int, str]]:
    schedule = [
        ("warmup", 0, 1, "enabled"),
        ("warmup", 0, 2, "disabled"),
    ]
    for pair in range(1, PROFILE["measured_pairs"] + 1):
        order = (
            ("enabled", "disabled")
            if pair % 2
            else ("disabled", "enabled")
        )
        schedule.extend(
            ("measured", pair, position, condition)
            for position, condition in enumerate(order, start=1)
        )
    return schedule


def validate_overlap(
    condition: str, record: dict[str, Any]
) -> None:
    expected_checks = (
        2 * PROFILE["measured_steps"] * PROFILE["mpi_ranks"]
    )
    fields = (
        "checks",
        "eligible_ranks",
        "eligible",
        "launched_h",
        "launched_e",
        "skipped_disabled",
        "skipped_unsupported_schedule",
        "skipped_no_remote",
        "skipped_cold_topology",
        "rejected",
    )
    if any(type(record.get(key)) is not int or record[key] < 0 for key in fields):
        raise RuntimeError("overlap counter record is incomplete or invalid")
    classified = sum(
        record[key]
        for key in (
            "eligible",
            "skipped_disabled",
            "skipped_unsupported_schedule",
            "skipped_no_remote",
            "skipped_cold_topology",
            "rejected",
        )
    )
    if (
        record["checks"] != expected_checks
        or classified != expected_checks
        or record["launched_h"] + record["launched_e"]
        != record["eligible"]
    ):
        raise RuntimeError("overlap counters are not an exact phase partition")
    if condition == "enabled":
        expected = {
            "eligible_ranks": 2,
            "eligible": expected_checks,
            "launched_h": expected_checks // 2,
            "launched_e": expected_checks // 2,
            "skipped_disabled": 0,
            "skipped_unsupported_schedule": 0,
            "skipped_no_remote": 0,
            "skipped_cold_topology": 0,
            "rejected": 0,
        }
    else:
        expected = {
            "eligible_ranks": 0,
            "eligible": 0,
            "launched_h": 0,
            "launched_e": 0,
            "skipped_disabled": expected_checks,
            "skipped_unsupported_schedule": 0,
            "skipped_no_remote": 0,
            "skipped_cold_topology": 0,
            "rejected": 0,
        }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{condition} overlap dispatch proof is incomplete")


def validate_sample(
    condition: str, output: str
) -> dict[str, Any]:
    HELPER.reject_unknown_gpmeep_markers(output)
    benchmark = HELPER.exactly_one(output, BENCHMARK_PREFIX, "benchmark")
    HELPER.validate_initialization_markers(
        output, benchmark, PROFILE["initial_condition"]
    )
    overlap = HELPER.exactly_one(output, OVERLAP_PREFIX, "overlap")
    eager = HELPER.exactly_one(output, EAGER_PREFIX, "eager MPI")
    pingpong = HELPER.exactly_one(output, PINGPONG_PREFIX, "receive ping-pong")
    graph = HELPER.exactly_one(output, GRAPH_PREFIX, "boundary graph")
    dft = HELPER.exactly_one(output, DFT_PREFIX, "DFT batch")
    devices = HELPER.records(output, DEVICE_PREFIX)
    if (
        len(devices) != 2
        or any(not isinstance(item, dict) for item in devices)
        or {item.get("rank") for item in devices} != {0, 1}
        or len({item.get("uuid") for item in devices}) != 2
        or any(not item.get("uuid") for item in devices)
    ):
        raise RuntimeError("sample did not prove two distinct physical GPUs")
    identity = {
        "schema_version": 4,
        "mpi_ranks": PROFILE["mpi_ranks"],
        "pixels": PROFILE["pixels"],
        "cells": PROFILE["pixels"] ** 3,
        "warmup_steps": PROFILE["warmup_steps"],
        "steps": PROFILE["measured_steps"],
        "source_enabled": PROFILE["source_enabled"],
        "overlap_material": PROFILE["overlap_material"],
        "requested_transport": PROFILE["transport"],
        "selected_transport": PROFILE["transport"],
        "completion_policy": PROFILE["completion_policy"],
        "initial_condition": PROFILE["initial_condition"],
    }
    if "loop_tile_base_db" in PROFILE:
        identity["loop_tile_base_db"] = PROFILE["loop_tile_base_db"]
    if any(benchmark.get(key) != value for key, value in identity.items()):
        raise RuntimeError("benchmark workload identity differs from the profile")
    if (
        benchmark.get("cpu_calls") != 0
        or type(benchmark.get("cuda_calls")) is not int
        or benchmark["cuda_calls"] <= 0
        or benchmark.get("pinned_bytes") != 0
        or benchmark.get("cuda_aware_bytes", 0) <= 0
    ):
        raise RuntimeError("sample did not use exclusive CUDA-aware GPU work")
    if any(type(eager.get(key)) is not int or eager[key] <= 0 for key in eager):
        raise RuntimeError("sample did not prove eager MPI request posting")
    if (
        pingpong.get("secondary_selections", 0) <= 0
        or graph.get("launches", 0) <= 0
        or dft.get("submitted_updates", 0) <= 0
    ):
        raise RuntimeError("sample omitted a fixed CUDA scheduling path")
    validate_overlap(condition, overlap)
    return {
        "benchmark": benchmark,
        "overlap": overlap,
        "eager": eager,
        "pingpong": pingpong,
        "graph": graph,
        "dft": dft,
        "devices": sorted(devices, key=lambda item: item["rank"]),
    }


def sample_stem(kind: str, pair: int, position: int, condition: str) -> str:
    return f"{kind}-pair{pair:02d}-position{position}-{condition}"


def condition_environment(
    base_environment: dict[str, str], condition: str
) -> dict[str, str]:
    environment = dict(base_environment)
    if condition == "enabled":
        environment["MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP"] = "1"
        environment["MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP"] = "1"
    elif condition == "disabled":
        environment["MEEP_GPU_DISABLE_BOUNDARY_EH_OVERLAP"] = "1"
        environment["MEEP_GPU_EXPECT_NO_BOUNDARY_EH_OVERLAP"] = "1"
    else:
        raise RuntimeError(f"unknown overlap A/B condition: {condition}")
    return environment


def run_sample(
    args: argparse.Namespace,
    kind: str,
    pair: int,
    position: int,
    condition: str,
    base_environment: dict[str, str],
) -> dict[str, Any]:
    stem = sample_stem(kind, pair, position, condition)
    before = HELPER.gpu_snapshot()
    HELPER.require_idle(before)
    environment = condition_environment(base_environment, condition)
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=10s",
        f"{args.timeout}s",
        str(args.mpiexec),
        "--bind-to",
        "none",
        "-np",
        str(PROFILE["mpi_ranks"]),
        str(args.executable),
    ]
    started_at = HELPER.utc_now()
    wall_start = time.perf_counter()
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    process_wall = time.perf_counter() - wall_start
    log_path = args.output / f"{stem}.log"
    HELPER.atomic_text(log_path, completed.stdout)
    after = HELPER.gpu_snapshot()
    HELPER.require_idle(after)
    if completed.returncode:
        raise RuntimeError(
            f"sample {stem} exited with status {completed.returncode}"
        )
    records = validate_sample(condition, completed.stdout)
    metadata = {
        "schema_version": 1,
        "kind": kind,
        "pair": pair,
        "position": position,
        "condition": condition,
        "started_at_utc": started_at,
        "finished_at_utc": HELPER.utc_now(),
        "command": command,
        "environment": dict(sorted(environment.items())),
        "external_process_wall_seconds": process_wall,
        "returncode": completed.returncode,
        "before": before,
        "after": after,
        "records": records,
        "log": log_path.name,
        "log_sha256": HELPER.sha256_file(log_path),
    }
    metadata_path = args.output / f"{stem}.json"
    HELPER.atomic_json(metadata_path, metadata)
    return metadata


def exact_observables(sample: dict[str, Any]) -> dict[str, Any]:
    benchmark = sample["records"]["benchmark"]
    return {key: benchmark[key] for key in OBSERVABLE_KEYS}


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    schedule = [
        (item["kind"], item["pair"], item["position"], item["condition"])
        for item in samples
    ]
    if schedule != expected_schedule():
        raise RuntimeError("sample schedule is incomplete or reordered")
    reference = exact_observables(samples[0])
    if any(exact_observables(sample) != reference for sample in samples[1:]):
        raise RuntimeError("enabled/disabled samples changed a physical observable")
    measured = [sample for sample in samples if sample["kind"] == "measured"]
    paired_speedups: list[float] = []
    time_ratios: list[float] = []
    faster_pairs = 0
    strata = {"enabled_first": [], "enabled_second": []}
    process_ratios = []
    for pair in range(1, PROFILE["measured_pairs"] + 1):
        current = {sample["condition"]: sample for sample in measured if sample["pair"] == pair}
        if set(current) != {"enabled", "disabled"}:
            raise RuntimeError(f"pair {pair} is incomplete")
        enabled = current["enabled"]
        disabled = current["disabled"]
        enabled_time = float(enabled["records"]["benchmark"]["seconds"])
        disabled_time = float(disabled["records"]["benchmark"]["seconds"])
        if not (math.isfinite(enabled_time) and math.isfinite(disabled_time)) or min(
            enabled_time, disabled_time
        ) <= 0:
            raise RuntimeError("sample contains an invalid workload time")
        speedup = disabled_time / enabled_time
        paired_speedups.append(speedup)
        time_ratios.append(enabled_time / disabled_time)
        process_ratios.append(
            enabled["external_process_wall_seconds"]
            / disabled["external_process_wall_seconds"]
        )
        if speedup > 1.0:
            faster_pairs += 1
        stratum = (
            "enabled_first" if enabled["position"] == 1 else "enabled_second"
        )
        strata[stratum].append(speedup)
    interval = HELPER.paired_log_interval(time_ratios)
    stratum_medians = {
        key: statistics.median(values) for key, values in strata.items()
    }
    promotion_checks = {
        "median_speedup": statistics.median(paired_speedups)
        >= PROFILE["minimum_median_paired_speedup"],
        "faster_pair_majority": faster_pairs >= PROFILE["minimum_faster_pairs"],
        "t90_upper_not_slower": interval["t90_upper_time_ratio"]
        <= PROFILE["maximum_t90_upper_time_ratio"],
        "both_position_strata_positive": all(
            value > 1.0 for value in stratum_medians.values()
        ),
    }
    promoted = all(promotion_checks.values())
    return {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "profile_sha256": PROFILE_SHA256,
        "sample_count": len(samples),
        "measured_pair_count": PROFILE["measured_pairs"],
        "all_observables_exact": True,
        "paired_speedups": paired_speedups,
        "paired_time_ratios_enabled_over_disabled": time_ratios,
        "median_paired_speedup": statistics.median(paired_speedups),
        "faster_pairs": faster_pairs,
        "position_stratum_median_speedups": stratum_medians,
        "process_wall_ratios_enabled_over_disabled": process_ratios,
        "balanced_process_wall_ratio": math.sqrt(
            statistics.median(
                process_ratios[index]
                for index in range(0, len(process_ratios), 2)
            )
            * statistics.median(
                process_ratios[index]
                for index in range(1, len(process_ratios), 2)
            )
        ),
        **interval,
        "promotion_checks": promotion_checks,
        "promote_default": promoted,
        "decision": (
            "retain-boundary-eh-overlap-default"
            if promoted
            else "disable-boundary-eh-overlap-default"
        ),
    }


def artifact_manifest(output: pathlib.Path) -> dict[str, Any]:
    manifest_path = output / "artifacts.sha256.json"
    excluded = {manifest_path, output / "COMPLETE", output / "FAILED.json"}
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path not in excluded
    )
    return {
        "schema_version": 1,
        "files": [HELPER.file_record(path, output) for path in files],
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Boundary E/H overlap paired A/B",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Median paired speedup: `{summary['median_paired_speedup']:.6f}x`",
        f"- Faster pairs: `{summary['faster_pairs']}/{summary['measured_pair_count']}`",
        f"- 90% upper enabled/disabled time ratio: `{summary['t90_upper_time_ratio']:.6f}`",
        f"- Exact observables: `{summary['all_observables_exact']}`",
        "",
        "Per-pair speedups: "
        + ", ".join(f"{value:.6f}x" for value in summary["paired_speedups"]),
        "",
    ]
    return "\n".join(lines)


def run_authoritative(args: argparse.Namespace) -> int:
    receipt, args.executable, args.library, ldd_output = HELPER.resolve_receipt_artifacts(
        REPO,
        args.build_receipt,
        args.expected_receipt_id,
        args.mpiexec,
    )
    prefix = args.mpiexec.parent.parent
    archive = args.output / "archive"
    archive.mkdir()
    archived = {
        "runner": archive / pathlib.Path(__file__).name,
        "helper": archive / "benchmark-dft-phase-sharing.py",
        "receipt": archive / "build-provenance.json",
        "executable": archive / args.executable.name,
        "library": archive / args.library.name,
        "mpiexec": archive / "receipt-mpiexec",
    }
    sources = {
        "runner": pathlib.Path(__file__),
        "helper": SCRIPT_DIR / "benchmark-dft-phase-sharing.py",
        "receipt": args.build_receipt,
        "executable": args.executable,
        "library": args.library,
        "mpiexec": args.mpiexec,
    }
    for archive_name, source in sorted(ADDITIONAL_RUNNER_SOURCES.items()):
        if (
            not archive_name
            or pathlib.PurePath(archive_name).name != archive_name
            or archive_name in {path.name for path in archived.values()}
        ):
            raise RuntimeError("invalid or duplicate additional runner source")
        archived[f"runner_dependency:{archive_name}"] = archive / archive_name
        sources[f"runner_dependency:{archive_name}"] = pathlib.Path(source).resolve()
    for key, destination in archived.items():
        HELPER.atomic_bytes(destination, sources[key].read_bytes())
    base_environment = {
        "HOME": str(args.output / "home"),
        "PATH": f"{prefix / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": f"{args.library.parent}:{prefix / 'lib'}",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MEEP_GPU_BACKEND": "cuda",
        "MEEP_GPU_STRICT": "1",
        "MEEP_GPU_MPI_TRANSPORT": PROFILE["transport"],
        "MEEP_GPU_MPI_COMPLETION": PROFILE["completion_policy"],
        "MEEP_GPU_MULTI_OVERLAP_MATERIAL": "1",
        "MEEP_GPU_MULTI_DISABLE_SOURCE": "1",
        "MEEP_GPU_MULTI_INITIAL_CONDITION": PROFILE["initial_condition"],
        "MEEP_GPU_MULTI_SOURCE_PROFILE": "single-ez-v1",
        "MEEP_GPU_MULTI_PIXELS": str(PROFILE["pixels"]),
        "MEEP_GPU_MULTI_WARMUP_STEPS": str(PROFILE["warmup_steps"]),
        "MEEP_GPU_MULTI_STEPS": str(PROFILE["measured_steps"]),
        "MEEP_GPU_EXPECT_BOUNDARY_PHASE_GRAPH": "1",
        "MEEP_GPU_EXPECT_EAGER_MPI": "1",
        "MEEP_GPU_EXPECT_RECEIVE_PINGPONG": "1",
        "OMPI_MCA_mca_base_param_files": str(
            REPO / "environment/openmpi-qualification-mca-params.conf"
        ),
        "OMPI_MCA_mca_base_component_path": str(prefix / "lib/openmpi"),
        "PMIX_MCA_mca_base_param_files": str(
            REPO / "environment/openmpi-qualification-mca-params.conf"
        ),
        "PMIX_MCA_mca_base_component_path": str(prefix / "lib/pmix"),
        "PRTE_MCA_mca_base_param_files": str(
            REPO / "environment/openmpi-qualification-mca-params.conf"
        ),
    }
    if "loop_tile_base_db" in PROFILE:
        base_environment["MEEP_GPU_MULTI_LOOP_TILE_BASE_DB"] = str(
            PROFILE["loop_tile_base_db"]
        )
    pathlib.Path(base_environment["HOME"]).mkdir()
    HELPER.atomic_json(
        args.output / "protocol.json",
        {
            "schema_version": 1,
            "profile": PROFILE,
            "profile_sha256": PROFILE_SHA256,
            "receipt_id": receipt["receipt_id"],
            "schedule": expected_schedule(),
            "archived": {
                key: HELPER.file_record(path, args.output)
                for key, path in archived.items()
            },
            "ldd": ldd_output,
        },
    )
    HELPER.atomic_json(
        args.output / "inventory.json",
        {
            "schema_version": 1,
            "receipt_id": receipt["receipt_id"],
            "gpu": HELPER.gpu_snapshot(),
            "lscpu": HELPER.command_output(["/usr/bin/lscpu"]),
        },
    )
    HELPER.require_idle(HELPER.gpu_snapshot())
    monitor_path = args.output / "continuous-gpu-process-monitor.log"
    monitor_process, monitor_stream = HELPER.start_process_monitor(monitor_path)
    monitor_stopped = False
    samples: list[dict[str, Any]] = []
    try:
        for kind, pair, position, condition in expected_schedule():
            samples.append(
                run_sample(
                    args,
                    kind,
                    pair,
                    position,
                    condition,
                    base_environment,
                )
            )
        expected_pids = len(samples) * PROFILE["mpi_ranks"]
        monitor = HELPER.stop_process_monitor(
            monitor_process, monitor_stream, monitor_path, expected_pids
        )
        monitor_stopped = True
        HELPER.atomic_json(args.output / "continuous-gpu-process-monitor.json", monitor)
        if not monitor["pass"]:
            raise RuntimeError("continuous GPU process monitor failed")
        summary = summarize(samples)
        receipt_after, executable_after, library_after, ldd_after = (
            HELPER.resolve_receipt_artifacts(
                REPO,
                args.build_receipt,
                args.expected_receipt_id,
                args.mpiexec,
            )
        )
        if (
            receipt_after["receipt_id"] != receipt["receipt_id"]
            or executable_after != args.executable
            or library_after != args.library
            or HELPER.normalized_ldd_dependencies(ldd_after)
            != HELPER.normalized_ldd_dependencies(ldd_output)
            or any(
                HELPER.sha256_file(archived[key])
                != HELPER.sha256_file(source)
                for key, source in sources.items()
            )
        ):
            raise RuntimeError("receipt or executable closure changed during A/B")
        summary.update(
            state="COMPLETE",
            finished_at_utc=HELPER.utc_now(),
            receipt_id=receipt["receipt_id"],
            continuous_gpu_process_monitor_pass=True,
        )
        report_path = args.output / "report.json"
        markdown_path = args.output / "report.md"
        HELPER.atomic_json(report_path, summary)
        HELPER.atomic_text(markdown_path, markdown_report(summary))
        manifest_path = args.output / "artifacts.sha256.json"
        HELPER.atomic_json(manifest_path, artifact_manifest(args.output))
        HELPER.atomic_json(
            args.output / "COMPLETE",
            {
                "schema_version": 1,
                "state": "COMPLETE",
                "receipt_id": receipt["receipt_id"],
                "decision": summary["decision"],
                "report": HELPER.file_record(report_path, args.output),
                "manifest": HELPER.file_record(manifest_path, args.output),
            },
        )
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except Exception:
        if not monitor_stopped:
            try:
                monitor = HELPER.stop_process_monitor(
                    monitor_process,
                    monitor_stream,
                    monitor_path,
                    len(samples) * PROFILE["mpi_ranks"],
                )
                HELPER.atomic_json(
                    args.output / "continuous-gpu-process-monitor.json", monitor
                )
            except Exception as monitor_error:
                HELPER.atomic_json(
                    args.output / "continuous-gpu-process-monitor-error.json",
                    {"error": f"{type(monitor_error).__name__}: {monitor_error}"},
                )
        raise


def main(argv: list[str] | None = None) -> int:
    assert_fixed_profile()
    args = parse_args(argv)
    args.output = args.output.resolve()
    args.build_receipt = args.build_receipt.resolve()
    args.mpiexec = args.mpiexec.resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_receipt_id):
        raise RuntimeError("expected receipt ID must be a lowercase SHA-256")
    if args.timeout <= 0:
        raise RuntimeError("timeout must be positive")
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise RuntimeError("output directory must be absent or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        return run_authoritative(args)
    except Exception as error:
        HELPER.atomic_json(
            args.output / "FAILED.json",
            {
                "schema_version": 1,
                "state": "FAILED",
                "finished_at_utc": HELPER.utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
