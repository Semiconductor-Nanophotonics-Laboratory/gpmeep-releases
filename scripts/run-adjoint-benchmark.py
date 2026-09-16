#!/usr/bin/env python3
"""Run and fail-closed gate repeated CPU/strict-CUDA adjoint benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any

from gpmeep_benchmark_evidence import (
    PHASE_COUNTERS,
    atomic_write_json,
    atomic_write_text,
    compare_gradient_vectors,
    compare_workloads,
    phase_counter_view,
    sha256_file,
    validate_backend_and_paths,
)
from gpmeep_provenance import (
    ProvenanceError,
    canonical_sha256,
    source_snapshot,
    verify_build_receipt,
)


PREFIX = "gpmeep-adjoint-benchmark:"
MATERIAL_GRADIENT_STATS_PREFIX = "material-gradient-stats:"
OBSERVABLE_TOLERANCES = {
    "objective": (1e-8, 5e-4),
    "gradient_l2": (1e-8, 2e-3),
    "gradient_projection": (1e-8, 2e-3),
    "gradient_sum": (1e-8, 2e-3),
}
RECORDED_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "MEEP_GPU_BACKEND",
    "MEEP_GPU_STRICT",
    "MEEP_GPU_AUTO_MIN_CELLS",
    "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS",
    "MEEP_GPU_DEVICE",
    "MEEP_MATERIAL_GRADIENT_PATH",
    "MEEP_MATERIAL_GRADIENT_STATS",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "PYTHONHASHSEED",
    "PYTHONPATH",
)
RELEVANT_SOURCE_PATHS = (
    "cuda/include/meep_cuda/runtime.hpp",
    "cuda/src/runtime.cu",
    "python/adjoint/__init__.py",
    "python/meep.i",
    "scripts/benchmark-adjoint.py",
    "scripts/gpmeep_benchmark_evidence.py",
    "scripts/gpmeep_provenance.py",
    "scripts/compare-material-gradient-benchmarks.py",
    "scripts/run-adjoint-benchmark.py",
    "src/dft.cpp",
    "src/gpu_backend.cpp",
    "src/gpu_backend_internal.hpp",
    "src/meep/mympi.hpp",
    "src/meepgeom.cpp",
    "src/mympi.cpp",
)
FINAL_ARTIFACTS = (
    "report.json",
    "report.md",
    "COMPLETE",
    "FAILED.json",
)


def git_output(repo: pathlib.Path, *arguments: str) -> str:
    proc = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {proc.stderr.strip()}"
        )
    return proc.stdout


def command_probe(command: list[str], cwd: pathlib.Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": False,
            "command": command,
            "error": str(error),
        }
    return {
        "available": proc.returncode == 0,
        "command": command,
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _one_real_file(candidates: list[pathlib.Path], description: str) -> pathlib.Path:
    resolved = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(resolved) != 1:
        raise RuntimeError(
            f"expected exactly one {description}, found {len(resolved)}: "
            + ", ".join(str(path) for path in resolved)
        )
    return resolved[0]


def expected_runtime_paths(
    build_python: pathlib.Path,
) -> dict[str, pathlib.Path]:
    package = build_python.resolve() / "meep"
    extension = _one_real_file(
        list(package.glob("_meep*.so")), "built Meep extension"
    )
    build_root = build_python.resolve().parent
    libmeep = _one_real_file(
        list((build_root / "src" / ".libs").glob("libmeep.so.*.*.*")),
        "built libmeep shared object",
    )
    meep_module = package / "__init__.py"
    if not meep_module.is_file():
        raise FileNotFoundError(f"built Meep package is absent: {meep_module}")
    return {
        "build_root": build_root,
        "meep_module": meep_module.resolve(),
        "extension": extension,
        "libmeep": libmeep,
    }


def _resolve_receipt_path(raw_path: str, repo: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(raw_path)
    return (repo / path).resolve() if not path.is_absolute() else path.resolve()


def verify_required_build_receipt(
    repo: pathlib.Path,
    runtime_paths: dict[str, pathlib.Path],
) -> dict[str, Any]:
    """Require a verified source-to-binary receipt for benchmark evidence."""

    receipt_path = runtime_paths["build_root"] / "build-provenance.json"
    if not receipt_path.is_file():
        raise RuntimeError(
            "required build provenance receipt is absent: "
            f"{receipt_path}; rebuild with scripts/build-meep-cuda-python.sh"
        )
    try:
        receipt = verify_build_receipt(receipt_path, repo)
    except ProvenanceError as error:
        raise RuntimeError(f"build provenance receipt is invalid: {error}") from error
    receipt_id = receipt["receipt_id"]

    expected_artifacts = {
        "python_extension": runtime_paths["extension"],
        "libmeep": runtime_paths["libmeep"],
    }
    receipt_artifacts = receipt.get("artifacts", {})
    for name, expected_path in expected_artifacts.items():
        record = receipt_artifacts.get(name, {})
        actual_path = (
            _resolve_receipt_path(str(record.get("path")), repo)
            if record.get("path")
            else None
        )
        if actual_path != expected_path.resolve():
            raise RuntimeError(f"build receipt {name} path is invalid")
        if record.get("sha256") != sha256_file(expected_path):
            raise RuntimeError(f"build receipt {name} SHA-256 is stale")

    for name, record in receipt.get("configuration_files", {}).items():
        if not isinstance(record, dict) or not record.get("path"):
            raise RuntimeError(f"build receipt configuration {name} is invalid")
        path = _resolve_receipt_path(str(record["path"]), repo)
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"build receipt configuration {name} is stale")
    return {
        "status": "verified",
        "path": str(receipt_path.resolve()),
        "sha256": sha256_file(receipt_path),
        "receipt_id": receipt_id,
        "build_input_id": receipt.get("build_input_id"),
        "artifact_set_id": receipt.get("artifact_set_id"),
    }


def collect_provenance(
    args: argparse.Namespace, repo: pathlib.Path
) -> dict[str, Any]:
    runtime_paths = expected_runtime_paths(args.build_python)
    source_hashes = {}
    for relative in RELEVANT_SOURCE_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"benchmark provenance source is absent: {path}"
            )
        source_hashes[relative] = sha256_file(path)
    return {
        "git_head": git_output(repo, "rev-parse", "HEAD").strip(),
        "git_status_porcelain": git_output(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines(),
        "source_snapshot": source_snapshot(repo),
        "relevant_source_sha256": source_hashes,
        "build_receipt": verify_required_build_receipt(repo, runtime_paths),
        "runtime_artifacts": {
            name: {
                "path": str(runtime_paths[name]),
                "sha256": sha256_file(runtime_paths[name]),
            }
            for name in ("extension", "libmeep")
        },
        "meep_module": {
            "path": str(runtime_paths["meep_module"]),
            "sha256": sha256_file(runtime_paths["meep_module"]),
        },
        "python": {
            "executable": str(args.python.resolve()),
            "sha256": sha256_file(args.python.resolve()),
            "probe": command_probe(
                [
                    str(args.python.resolve()),
                    "-c",
                    (
                        "import json,sys;"
                        "print(json.dumps({'executable':sys.executable,"
                        "'version':sys.version},sort_keys=True))"
                    ),
                ],
                repo,
            ),
        },
        "platform": platform.platform(),
        "cuda_toolchain": command_probe(
            [str(args.python.parent / "nvcc"), "--version"], repo
        ),
        "nvidia_driver": command_probe(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,"
                "compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            repo,
        ),
        "runner_command": [
            str(args.python.resolve()),
            str(pathlib.Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=pathlib.Path,
        default=repo / ".envs" / "meep-gpu-cuda" / "bin" / "python",
    )
    parser.add_argument(
        "--build-python",
        type=pathlib.Path,
        default=repo / "build" / "meep-cuda-python-fp32" / "python",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=repo / "benchmark-results" / "adjoint",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--run-time", type=float, default=40)
    parser.add_argument("--cell-size", type=float, default=8)
    parser.add_argument("--design-resolution", type=int, default=20)
    parser.add_argument("--minimum-speedup", type=float, default=1.25)
    parser.add_argument("--gradient-atol", type=float, default=1e-6)
    parser.add_argument("--gradient-rtol", type=float, default=2e-3)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument(
        "--material-gradient-path",
        choices=("legacy", "auto", "analytic-required"),
        default="auto",
    )
    return parser.parse_args(argv)


def extract_result_pointer(stdout: str) -> dict[str, Any]:
    matches = [
        line[len(PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {PREFIX!r} record, found {len(matches)}"
        )
    value = json.loads(matches[0])
    if not isinstance(value, dict):
        raise RuntimeError("adjoint benchmark pointer must be a JSON object")
    return value


def extract_material_gradient_stats(stdout: str) -> dict[str, int]:
    matches = [
        line[len(MATERIAL_GRADIENT_STATS_PREFIX) :].strip()
        for line in stdout.splitlines()
        if line.startswith(MATERIAL_GRADIENT_STATS_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one material-gradient-stats record, "
            f"found {len(matches)}"
        )
    result: dict[str, int] = {}
    for field in matches[0].split():
        key, separator, raw_value = field.partition("=")
        if not separator or not key or not raw_value:
            raise RuntimeError("material-gradient-stats record is malformed")
        try:
            result[key] = int(raw_value)
        except ValueError as error:
            raise RuntimeError(
                "material-gradient-stats contains a non-integer value"
            ) from error
    required = {
        "points",
        "forced_legacy_points",
        "analytic_points",
        "edge_legacy_points",
        "unsupported_points",
        "requests",
        "analytic_requests",
        "legacy_requests",
    }
    if set(result) != required:
        raise RuntimeError(
            "material-gradient-stats fields do not match the expected schema"
        )
    if result["requests"] != (
        result["analytic_requests"] + result["legacy_requests"]
    ):
        raise RuntimeError("material-gradient-stats request accounting is invalid")
    return result


def validate_material_gradient_stats(
    stats: dict[str, int], path: str
) -> list[str]:
    problems: list[str] = []
    if stats["points"] <= 0 or stats["requests"] <= 0:
        problems.append("MaterialGrid gradient performed no work")
    if path == "legacy":
        expected = {
            "forced_legacy_points": stats["points"],
            "analytic_points": 0,
            "edge_legacy_points": 0,
            "unsupported_points": 0,
            "analytic_requests": 0,
            "legacy_requests": stats["requests"],
        }
    elif path == "analytic-required":
        expected = {
            "forced_legacy_points": 0,
            "analytic_points": stats["points"],
            "edge_legacy_points": 0,
            "unsupported_points": 0,
            "analytic_requests": stats["requests"],
            "legacy_requests": 0,
        }
    else:
        expected = {}
    for field, expected_value in expected.items():
        if stats[field] != expected_value:
            problems.append(
                f"{path} expected {field}={expected_value}, got {stats[field]}"
            )
    return problems


def clean_child_environment(
    args: argparse.Namespace,
    repo: pathlib.Path,
    backend: str,
    stats_file: pathlib.Path,
    run_nonce: str,
    receipt_id: str,
) -> dict[str, str]:
    env = os.environ.copy()
    cuda_selection = {
        key: env[key]
        for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        if key in env
    }
    for key in list(env):
        if (
            key == "PYTHONPATH"
            or key.startswith("MEEP_GPU_")
            or key.startswith("MEEP_MATERIAL_")
            or key.startswith("GPMEEP_VALIDATION_")
            or key.startswith("OMP_")
            or key.startswith("CUDA_")
        ):
            env.pop(key, None)
    env.update(cuda_selection)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(repo / "scripts" / "python-validation"),
                    str(repo / "scripts"),
                    str(args.build_python.resolve()),
                )
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": str(args.omp_threads),
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(args.output / "matplotlib"),
            "MEEP_GPU_BACKEND": backend,
            "MEEP_MATERIAL_GRADIENT_PATH": args.material_gradient_path,
            "MEEP_MATERIAL_GRADIENT_STATS": "1",
            "GPMEEP_VALIDATION_EXPECTED_BACKEND": backend,
            "GPMEEP_VALIDATION_STATS_FILE": str(stats_file),
            "GPMEEP_VALIDATION_RUN_NONCE": run_nonce,
            "GPMEEP_VALIDATION_BUILD_RECEIPT_ID": receipt_id,
        }
    )
    if backend == "cuda":
        env.update(
            {
                "MEEP_GPU_STRICT": "1",
                "MEEP_GPU_AUTO_MIN_CELLS": "0",
                "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
                "GPMEEP_VALIDATION_STRICT_CUDA": "1",
            }
        )
    return env


def _site_statistics_path_gate(
    stats: dict[str, Any],
    result: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    provenance: dict[str, Any],
    run_nonce: str,
) -> list[str]:
    problems: list[str] = []
    if stats.get("capture_status") != "captured":
        problems.append("statistics were not captured")
    if stats.get("active_backend") != backend:
        problems.append("site statistics active backend mismatch")
    if stats.get("requested_backend") != backend:
        problems.append("site statistics requested backend mismatch")
    if stats.get("strict_cuda_marker") is not (backend == "cuda"):
        problems.append("site statistics strict marker mismatch")
    if stats.get("single_precision") is not True:
        problems.append("site statistics single-precision marker is false")
    if stats.get("gpu_compiled") is not True:
        problems.append("site statistics CUDA build marker is false")
    if stats.get("run_nonce") != run_nonce:
        problems.append("site statistics run nonce mismatch")
    if stats.get("build_receipt_id") != provenance["build_receipt"]["receipt_id"]:
        problems.append("site statistics build receipt ID mismatch")

    expected_records = {
        "python_executable": {
            "path": provenance["python"]["executable"],
            "sha256": provenance["python"]["sha256"],
        },
        "meep_module": provenance["meep_module"],
        "extension": provenance["runtime_artifacts"]["extension"],
        "libmeep": provenance["runtime_artifacts"]["libmeep"],
    }
    for name, expected in expected_records.items():
        actual = stats.get(name)
        if not isinstance(actual, dict):
            problems.append(f"site statistics {name} record is absent")
            continue
        if pathlib.Path(str(actual.get("path", ""))).resolve() != pathlib.Path(
            str(expected["path"])
        ).resolve():
            problems.append(f"site statistics {name} path mismatch")
        if actual.get("sha256") != expected["sha256"]:
            problems.append(f"site statistics {name} SHA-256 mismatch")

    site_counters = stats.get("statistics", {})
    child_counters = result.get("workload", {}).get("final_statistics", {})
    try:
        for phase_backend in ("cpu", "cuda"):
            if phase_counter_view(site_counters, phase_backend) != phase_counter_view(
                child_counters, phase_backend
            ):
                problems.append(
                    f"site/child final {phase_backend} counters disagree"
                )
    except ValueError as error:
        problems.append(str(error))
    return problems


def run_once(
    args: argparse.Namespace,
    repo: pathlib.Path,
    provenance: dict[str, Any],
    run_id: str,
    backend: str,
    iteration: int,
    warmup: bool,
) -> dict[str, Any]:
    label = f"{run_id}-{backend}-{'warmup' if warmup else 'run'}-{iteration}"
    stats_file = args.output / f"{label}-gpu-statistics.json"
    result_file = args.output / f"{label}-result.json"
    stdout_file = args.output / f"{label}.stdout.log"
    stderr_file = args.output / f"{label}.stderr.log"
    for path in (stats_file, result_file, stdout_file, stderr_file):
        path.unlink(missing_ok=True)
    run_nonce = uuid.uuid4().hex
    env = clean_child_environment(
        args,
        repo,
        backend,
        stats_file,
        run_nonce,
        provenance["build_receipt"]["receipt_id"],
    )
    command = [
        str(args.python.resolve()),
        str(repo / "scripts" / "benchmark-adjoint.py"),
        "--resolution",
        str(args.resolution),
        "--run-time",
        str(args.run_time),
        "--cell-size",
        str(args.cell_size),
        "--design-resolution",
        str(args.design_resolution),
        "--result-file",
        str(result_file),
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    process_seconds = time.perf_counter() - started
    atomic_write_text(stdout_file, proc.stdout)
    atomic_write_text(stderr_file, proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} exited {proc.returncode}; see {stderr_file}"
        )
    pointer = extract_result_pointer(proc.stdout)
    material_gradient_stats = extract_material_gradient_stats(proc.stdout)
    material_gradient_problems = validate_material_gradient_stats(
        material_gradient_stats, args.material_gradient_path
    )
    if material_gradient_problems:
        raise RuntimeError(
            f"{label} material-gradient path gate failed: "
            + "; ".join(material_gradient_problems)
        )
    if pathlib.Path(str(pointer.get("result_file", ""))).resolve() != result_file.resolve():
        raise RuntimeError(f"{label} reported an unexpected result path")
    if not result_file.is_file() or pointer.get("sha256") != sha256_file(result_file):
        raise RuntimeError(f"{label} result SHA-256 is absent or invalid")
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} evidence: {error}") from error
    if result.get("schema_version") not in (3, 4):
        raise RuntimeError(f"{label} result schema is unsupported")

    expected = {
        name: pathlib.Path(record["path"])
        for name, record in provenance["runtime_artifacts"].items()
    }
    path_gate = validate_backend_and_paths(
        result,
        expected_backend=backend,
        expected_python=args.python,
        expected_python_sha256=provenance["python"]["sha256"],
        expected_meep_module=pathlib.Path(provenance["meep_module"]["path"]),
        expected_meep_module_sha256=provenance["meep_module"]["sha256"],
        expected_extension=expected["extension"],
        expected_extension_sha256=provenance["runtime_artifacts"]["extension"][
            "sha256"
        ],
        expected_libmeep=expected["libmeep"],
        expected_libmeep_sha256=provenance["runtime_artifacts"]["libmeep"][
            "sha256"
        ],
    )
    path_gate["problems"].extend(
        _site_statistics_path_gate(
            stats, result, backend, args, provenance, run_nonce
        )
    )
    path_gate["pass"] = not path_gate["problems"]
    if not path_gate["pass"]:
        raise RuntimeError(
            f"{label} backend/path gate failed: " + "; ".join(path_gate["problems"])
        )
    return {
        "backend": backend,
        "iteration": iteration,
        "warmup": warmup,
        "process_seconds": process_seconds,
        "result": result,
        "result_file": str(result_file),
        "result_sha256": sha256_file(result_file),
        "statistics_file": str(stats_file),
        "statistics_sha256": sha256_file(stats_file),
        "statistics": stats,
        "backend_path_gate": path_gate,
        "material_gradient_stats": material_gradient_stats,
        "stdout_log": str(stdout_file),
        "stdout_sha256": sha256_file(stdout_file),
        "stderr_log": str(stderr_file),
        "stderr_sha256": sha256_file(stderr_file),
        "command": command,
        "environment": {
            key: env[key]
            for key in RECORDED_ENVIRONMENT_KEYS
            if key in env
        },
    }


def compare_observables(
    cpu_runs: list[dict[str, Any]], cuda_runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    comparisons = []
    for cpu, cuda in zip(cpu_runs, cuda_runs):
        cpu_payload = cpu["result"]["result"]
        cuda_payload = cuda["result"]["result"]
        for metric, (atol, rtol) in OBSERVABLE_TOLERANCES.items():
            cpu_value = float(cpu_payload[metric])
            cuda_value = float(cuda_payload[metric])
            error = abs(cpu_value - cuda_value)
            allowed = atol + rtol * max(abs(cpu_value), abs(cuda_value))
            comparisons.append(
                {
                    "iteration": cpu["iteration"],
                    "metric": metric,
                    "cpu": cpu_value,
                    "cuda": cuda_value,
                    "absolute_error": error,
                    "allowed_error": allowed,
                    "pass": error <= allowed,
                }
            )
    return comparisons


def prepare_output(output: pathlib.Path, run_id: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in FINAL_ARTIFACTS:
        (output / name).unlink(missing_ok=True)
    atomic_write_json(
        output / "state.json",
        {"schema_version": 1, "state": "RUNNING", "run_id": run_id},
    )


def mark_failed(output: pathlib.Path, run_id: str, error: BaseException) -> None:
    for name in ("report.json", "report.md", "COMPLETE"):
        (output / name).unlink(missing_ok=True)
    failure = {
        "schema_version": 1,
        "state": "FAILED",
        "run_id": run_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }
    atomic_write_json(output / "FAILED.json", failure)
    atomic_write_json(output / "state.json", failure)


def publish_complete(
    output: pathlib.Path,
    run_id: str,
    report: dict[str, Any],
    markdown: str,
) -> None:
    if report.get("state") != "COMPLETE" or report.get("gate", {}).get("pass") is not True:
        raise RuntimeError("refusing to publish an unqualified benchmark as COMPLETE")
    markdown_path = output / "report.md"
    json_path = output / "report.json"
    atomic_write_text(markdown_path, markdown)
    report["report_markdown"] = {
        "path": str(markdown_path),
        "sha256": sha256_file(markdown_path),
    }
    atomic_write_json(json_path, report)
    marker = {
        "schema_version": 1,
        "state": "COMPLETE",
        "run_id": run_id,
        "report": str(json_path),
        "report_sha256": sha256_file(json_path),
    }
    atomic_write_json(output / "COMPLETE", marker)
    atomic_write_json(output / "state.json", marker)


def execute(args: argparse.Namespace, repo: pathlib.Path, run_id: str) -> int:
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    if args.minimum_speedup <= 1:
        raise ValueError("minimum-speedup must be greater than one")
    if args.gradient_atol < 0 or args.gradient_rtol < 0:
        raise ValueError("gradient tolerances must be nonnegative")
    if args.omp_threads < 1:
        raise ValueError("omp-threads must be positive")
    if not args.python.is_file() or not args.build_python.is_dir():
        raise FileNotFoundError("isolated Python executable or build package is absent")
    provenance_before = collect_provenance(args, repo)

    all_runs: list[dict[str, Any]] = []
    for backend in ("cpu", "cuda"):
        for iteration in range(args.warmups):
            all_runs.append(
                run_once(
                    args,
                    repo,
                    provenance_before,
                    run_id,
                    backend,
                    iteration,
                    True,
                )
            )
    for iteration in range(args.repeats):
        order = ("cpu", "cuda") if iteration % 2 == 0 else ("cuda", "cpu")
        for backend in order:
            all_runs.append(
                run_once(
                    args,
                    repo,
                    provenance_before,
                    run_id,
                    backend,
                    iteration,
                    False,
                )
            )

    cpu_runs = sorted(
        (
            run
            for run in all_runs
            if run["backend"] == "cpu" and not run["warmup"]
        ),
        key=lambda run: run["iteration"],
    )
    cuda_runs = sorted(
        (
            run
            for run in all_runs
            if run["backend"] == "cuda" and not run["warmup"]
        ),
        key=lambda run: run["iteration"],
    )
    cpu_times = [float(run["result"]["timing"]["elapsed_seconds"]) for run in cpu_runs]
    cuda_times = [
        float(run["result"]["timing"]["elapsed_seconds"]) for run in cuda_runs
    ]
    cpu_median = statistics.median(cpu_times)
    cuda_median = statistics.median(cuda_times)
    speedup = cpu_median / cuda_median
    observable_comparisons = compare_observables(cpu_runs, cuda_runs)
    expected_design_variables = (
        int(round(2.0 * args.design_resolution)) + 1
    ) ** 2
    gradient_comparisons = []
    workload_comparisons = []
    for cpu, cuda in zip(cpu_runs, cuda_runs):
        gradient = compare_gradient_vectors(
            cpu["result"],
            cuda["result"],
            atol=args.gradient_atol,
            rtol=args.gradient_rtol,
            expected_count=expected_design_variables,
        )
        gradient["iteration"] = cpu["iteration"]
        gradient_comparisons.append(gradient)
        workload = compare_workloads(cpu["result"], cuda["result"])
        workload["iteration"] = cpu["iteration"]
        workload_comparisons.append(workload)

    cells = int(cpu_runs[0]["result"]["workload"]["yee_cells"])
    performance_pass = cells >= 65536 and speedup >= args.minimum_speedup
    scalar_pass = all(item["pass"] for item in observable_comparisons)
    gradient_pass = all(item["pass"] for item in gradient_comparisons)
    workload_pass = all(item["pass"] for item in workload_comparisons)
    path_pass = all(run["backend_path_gate"]["pass"] for run in all_runs)
    correctness_pass = scalar_pass and gradient_pass and workload_pass and path_pass
    passed = performance_pass and correctness_pass
    provenance_after = collect_provenance(args, repo)
    if provenance_before != provenance_after:
        raise RuntimeError(
            "source tree, build receipt, runtime artifact, or environment "
            "provenance changed during the benchmark"
        )

    report = {
        "schema_version": 3,
        "state": "COMPLETE",
        "run_id": run_id,
        "provenance": provenance_after,
        "workload": {
            "resolution": args.resolution,
            "run_time": args.run_time,
            "cell_size": args.cell_size,
            "design_resolution": args.design_resolution,
            "design_variables": expected_design_variables,
            "yee_cells": cells,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "omp_threads": args.omp_threads,
            "material_gradient_path": args.material_gradient_path,
        },
        "gate": {
            "minimum_speedup": args.minimum_speedup,
            "production_minimum_cells": 65536,
            "gradient_atol": args.gradient_atol,
            "gradient_rtol": args.gradient_rtol,
            "scalar_observables_pass": scalar_pass,
            "full_gradient_pass": gradient_pass,
            "workload_equality_pass": workload_pass,
            "backend_path_pass": path_pass,
            "correctness_pass": correctness_pass,
            "performance_pass": performance_pass,
            "pass": passed,
        },
        "timing": {
            "cpu_seconds": cpu_times,
            "cuda_seconds": cuda_times,
            "cpu_median_seconds": cpu_median,
            "cuda_median_seconds": cuda_median,
            "speedup_cpu_over_cuda": speedup,
        },
        "observable_comparisons": observable_comparisons,
        "gradient_comparisons": gradient_comparisons,
        "workload_comparisons": workload_comparisons,
        "runs": all_runs,
    }
    markdown = "\n".join(
        [
            "# Adjoint CPU/CUDA benchmark",
            "",
            f"- Run ID: `{run_id}`",
            f"- Workload: {cells:,} Yee cells, {expected_design_variables:,} "
            f"gradient values, resolution {args.resolution}, fixed post-source "
            f"run time {args.run_time:g}",
            f"- Material gradient path: `{args.material_gradient_path}`",
            f"- Git HEAD: `{provenance_after['git_head']}`",
            "- Source snapshot SHA-256: "
            f"`{provenance_after['source_snapshot']['sha256']}`",
            "- `_meep.so` SHA-256: "
            f"`{provenance_after['runtime_artifacts']['extension']['sha256']}`",
            "- `libmeep.so` SHA-256: "
            f"`{provenance_after['runtime_artifacts']['libmeep']['sha256']}`",
            f"- CPU median: {cpu_median:.6f} s",
            f"- CUDA median: {cuda_median:.6f} s",
            f"- Speedup: {speedup:.3f}× (required {args.minimum_speedup:.3f}×)",
            f"- Full gradient gate: {'PASS' if gradient_pass else 'FAIL'}",
            f"- Workload equality gate: {'PASS' if workload_pass else 'FAIL'}",
            f"- Backend/path gate: {'PASS' if path_pass else 'FAIL'}",
            f"- Performance gate: {'PASS' if performance_pass else 'FAIL'}",
            f"- Overall: {'PASS' if passed else 'FAIL'}",
            "",
        ]
    )
    if not passed:
        failed_gates = [
            name
            for name, value in report["gate"].items()
            if name.endswith("_pass") and value is False
        ]
        raise RuntimeError(
            "adjoint benchmark qualification gates failed: "
            + ", ".join(failed_gates)
        )
    publish_complete(args.output, run_id, report, markdown)
    print(f"Adjoint benchmark JSON: {args.output / 'report.json'}")
    print(f"Adjoint benchmark Markdown: {args.output / 'report.md'}")
    print(
        f"CPU median={cpu_median:.6f}s CUDA median={cuda_median:.6f}s "
        f"speedup={speedup:.3f}x gate={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = pathlib.Path(__file__).resolve().parent.parent
    run_id = uuid.uuid4().hex
    prepare_output(args.output, run_id)
    try:
        return execute(args, repo, run_id)
    except BaseException as error:
        mark_failed(args.output, run_id, error)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
