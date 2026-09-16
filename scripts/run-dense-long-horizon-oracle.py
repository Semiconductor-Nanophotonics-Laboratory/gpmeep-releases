#!/usr/bin/env python3
"""Generate and compare three independent dense long-horizon evidence lanes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import time


SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPARATOR_PATH = SOURCE_ROOT / "scripts" / "compare-dense-long-horizon.py"
PASS_PREFIX = "PASS: wrote deterministic dense long-horizon full-array evidence:"
MATERIALIZE_MARKER = "PASS: materialized dense-oracle runtime ELF"
SOURCE_INPUTS = (
    SOURCE_ROOT / "tests/gpu-step-db.cpp",
    SOURCE_ROOT / "src/step_db.cpp",
    SOURCE_ROOT / "scripts/compare-dense-long-horizon.py",
    pathlib.Path(__file__).resolve(),
    SOURCE_ROOT / "environment/openmpi-qualification-mca-params.conf",
)


def load_comparator():
    spec = importlib.util.spec_from_file_location(
        "gpmeep_dense_long_horizon_comparator", COMPARATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dense long-horizon comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARATOR = load_comparator()


def run_isolated_process(
    command: list[str],
    *,
    environment: dict[str, str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=SOURCE_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="strict",
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, _ = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(
            command, timeout_seconds, output=stdout
        ) from error
    return subprocess.CompletedProcess(command, process.returncode, stdout, None)


def bind_regular_file(path_value: os.PathLike[str] | str) -> dict[str, str]:
    path = pathlib.Path(path_value).resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"qualification input is not a regular file: {path}")
    return {"path": str(path), "sha256": COMPARATOR.sha256_file(path)}


def qualification_input_identity(
    lane_executables: dict[str, pathlib.Path],
) -> dict:
    build_roots = {
        lane: executable.parent.parent.resolve(strict=True)
        for lane, executable in lane_executables.items()
    }
    if len(set(build_roots.values())) != len(build_roots):
        raise RuntimeError("dense oracle lanes must use three distinct build roots")
    builds = {}
    for lane, build_root in build_roots.items():
        builds[lane] = {
            "build_root": str(build_root),
            "configuration": [
                bind_regular_file(build_root / relative)
                for relative in (
                    "config.status",
                    "config.h",
                    "Makefile",
                    "tests/Makefile",
                )
            ],
            "wrapper": bind_regular_file(lane_executables[lane]),
        }
    return {
        "sources": [bind_regular_file(path) for path in SOURCE_INPUTS],
        "builds": builds,
    }


def _clean_environment(
    runtime_home: pathlib.Path = pathlib.Path("/tmp/gpmeep-dense-oracle-empty-home"),
) -> dict[str, str]:
    # An allowlist is intentional: ambient MEEP, MPI, loader, profiler, and
    # allocator settings would otherwise make the three lanes incomparable.
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(runtime_home),
        "XDG_CACHE_HOME": str(runtime_home / "cache"),
        "LC_ALL": "C",
        "LANG": "C",
        "TMPDIR": "/tmp",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_CACHE_DISABLE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "MEEP_GPU_ALLOW_OVERSUBSCRIBE": "0",
    }


def expected_runtime_identity(executable: pathlib.Path) -> dict[str, dict[str, str]]:
    expected_elf = (
        executable.parent / ".libs" / ("lt-" + executable.name)
    ).resolve(strict=True)
    expected_libmeep = (
        executable.parent.parent / "src" / ".libs" / "libmeep.so"
    ).resolve(strict=True)
    return {
        "executable": {
            "path": str(expected_elf),
            "sha256": COMPARATOR.sha256_file(expected_elf),
        },
        "libmeep": {
            "path": str(expected_libmeep),
            "sha256": COMPARATOR.sha256_file(expected_libmeep),
        },
    }


def runtime_elf_exists(executable: pathlib.Path) -> bool:
    # Libtool materializes .libs/lt-<name> on the first wrapper execution when
    # the uninstalled shared library requires a relink.  Later lanes can bind
    # it both before and after execution; the first lane binds it immediately
    # after the wrapper has created it and the process reported /proc/self/exe.
    return (
        executable.parent / ".libs" / ("lt-" + executable.name)
    ).is_file()


def materialize_runtime_elf(
    executable: pathlib.Path,
    environment: dict[str, str],
    timeout_seconds: float,
) -> None:
    runtime_elf = executable.parent / ".libs" / ("lt-" + executable.name)
    backup = runtime_elf.with_name(
        runtime_elf.name + f".gpmeep-oracle-backup-{os.getpid()}"
    )
    if backup.exists() or backup.is_symlink():
        raise RuntimeError(f"stale runtime materialization backup exists: {backup}")
    had_runtime = runtime_elf_exists(executable)
    if had_runtime:
        if runtime_elf.is_symlink():
            raise RuntimeError("runtime ELF materialization target is a symlink")
        os.replace(runtime_elf, backup)
    materialize_environment = dict(environment)
    materialize_environment["MEEP_GPU_TEST_DENSE_MATERIALIZE_ONLY"] = "1"
    try:
        completed = run_isolated_process(
            [str(executable)],
            environment=materialize_environment,
            timeout_seconds=timeout_seconds,
        )
        markers = [
            line for line in completed.stdout.splitlines()
            if line == MATERIALIZE_MARKER
        ]
        if completed.returncode != 0 or len(markers) != 1:
            raise RuntimeError(
                "runtime ELF materialization failed with status "
                f"{completed.returncode}: {completed.stdout.strip()}"
            )
        if not runtime_elf_exists(executable) or runtime_elf.is_symlink():
            raise RuntimeError("runtime ELF materialization attestation failed")
    except Exception as error:
        if runtime_elf.exists() and not runtime_elf.is_symlink():
            runtime_elf.unlink()
        if had_runtime and backup.exists():
            os.replace(backup, runtime_elf)
        if isinstance(error, subprocess.TimeoutExpired):
            raise RuntimeError("runtime ELF materialization timed out") from error
        raise
    else:
        if had_runtime:
            backup.unlink()


def validate_runtime_identity(
    executable: pathlib.Path,
    parsed: dict,
    before_execution: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    expected = expected_runtime_identity(executable)
    if before_execution is not None and expected != before_execution:
        raise RuntimeError(
            "runtime executable or libmeep changed while the evidence lane ran"
        )
    for artifact, description in (
        ("executable", "runtime executable"),
        ("libmeep", "loaded libmeep"),
    ):
        recorded = parsed["runtime"][artifact]
        if recorded.get("path") != expected[artifact]["path"]:
            raise RuntimeError(
                f"{description} mismatch: expected {expected[artifact]['path']}, "
                f"got {recorded.get('path')}"
            )
        if recorded.get("sha256") != expected[artifact]["sha256"]:
            raise RuntimeError(
                f"{description} SHA-256 mismatch: expected "
                f"{expected[artifact]['sha256']}, got {recorded.get('sha256')}"
            )
    return expected


def run_lane(
    *,
    lane: str,
    executable_value: os.PathLike[str] | str,
    backend: str,
    evidence_path: pathlib.Path,
    log_path: pathlib.Path,
    timeout_seconds: float,
    cuda_visible_devices: str,
    runtime_home: pathlib.Path,
) -> dict:
    executable = pathlib.Path(executable_value).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"{lane}: executable is not an executable file: {executable}")
    environment = _clean_environment(runtime_home)
    if backend == "cuda":
        if not cuda_visible_devices or any(
            character not in "0123456789," for character in cuda_visible_devices
        ):
            raise RuntimeError("--cuda-visible-devices must be a nonempty ordinal list")
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        environment["MEEP_GPU_DEVICE"] = "0"
        cuda_prefix = SOURCE_ROOT / ".envs/meep-gpu-cuda-mpi"
        qualification_params = (
            SOURCE_ROOT / "environment/openmpi-qualification-mca-params.conf"
        )
        environment.update(
            {
                "OMPI_MCA_mca_base_param_files": str(qualification_params),
                "OMPI_MCA_mca_base_component_path": str(
                    cuda_prefix / "lib/openmpi"
                ),
                "PMIX_MCA_mca_base_param_files": str(qualification_params),
                "PMIX_MCA_mca_base_component_path": str(
                    cuda_prefix / "lib/pmix"
                ),
                "PRTE_MCA_mca_base_param_files": str(qualification_params),
            }
        )
    materialize_runtime_elf(executable, environment, timeout_seconds)
    runtime_before = expected_runtime_identity(executable)
    environment["MEEP_GPU_TEST_DENSE_EVIDENCE_WRITE"] = str(evidence_path)
    environment["MEEP_GPU_TEST_DENSE_EVIDENCE_BACKEND"] = backend

    started = time.monotonic()
    try:
        completed = run_isolated_process(
            [str(executable)],
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{lane}: timed out after {timeout_seconds:g} seconds"
        ) from error
    elapsed = time.monotonic() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode < 0:
        raise RuntimeError(f"{lane}: terminated by signal {-completed.returncode}")
    if completed.returncode != 0:
        raise RuntimeError(f"{lane}: exited with status {completed.returncode}")
    markers = [line for line in completed.stdout.splitlines() if line.startswith(PASS_PREFIX)]
    if len(markers) != 1:
        raise RuntimeError(f"{lane}: expected exactly one evidence PASS marker")
    if not evidence_path.is_file() or evidence_path.stat().st_size <= 0:
        raise RuntimeError(f"{lane}: did not create nonempty evidence")
    parsed = COMPARATOR.parse_evidence(evidence_path)
    if parsed["lane"] != lane:
        raise RuntimeError(f"{lane}: executable emitted {parsed['lane']} evidence")
    runtime_identity = validate_runtime_identity(
        executable, parsed, before_execution=runtime_before
    )
    return {
        "lane": lane,
        "backend": backend,
        "executable": str(executable),
        "executable_sha256": COMPARATOR.sha256_file(executable),
        "runtime_executable": parsed["runtime"]["executable"],
        "loaded_libmeep": parsed["runtime"]["libmeep"],
        "runtime_identity_before_after": runtime_identity,
        "elapsed_seconds": elapsed,
        "log": str(log_path),
        "log_sha256": COMPARATOR.sha256_file(log_path),
        "evidence": str(evidence_path),
        "evidence_sha256": parsed["sha256"],
        "pass_marker": markers[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--fp64-cpu-executable",
        type=pathlib.Path,
        default=SOURCE_ROOT / "build/cpu-fp64/tests/gpu-step-db",
    )
    parser.add_argument(
        "--fp32-cpu-executable",
        type=pathlib.Path,
        default=SOURCE_ROOT / "build/cpu-fp32/tests/gpu-step-db",
    )
    parser.add_argument(
        "--fp32-cuda-executable",
        type=pathlib.Path,
        default=SOURCE_ROOT / "build/meep-cuda-mpi-python-fp32/tests/gpu-step-db",
    )
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    arguments = parser.parse_args()
    if not math_is_finite_positive(arguments.timeout_seconds):
        parser.error("--timeout-seconds must be positive and finite")

    output = arguments.output_dir.resolve()
    try:
        output.mkdir(parents=False, mode=0o750)
    except FileExistsError as error:
        raise RuntimeError(
            f"refusing to reuse output directory with potentially stale evidence: {output}"
        ) from error
    runtime_home = output / ".runtime-home"
    runtime_home.mkdir(mode=0o700)
    (runtime_home / "cache").mkdir(mode=0o700)

    lane_specs = (
        ("cpu-fp64", arguments.fp64_cpu_executable, "cpu"),
        ("cpu-fp32", arguments.fp32_cpu_executable, "cpu"),
        ("cuda-fp32", arguments.fp32_cuda_executable, "cuda"),
    )
    lane_executables = {
        lane: pathlib.Path(executable).resolve(strict=True)
        for lane, executable, _ in lane_specs
    }
    qualification_inputs_before = qualification_input_identity(lane_executables)
    try:
        executions = []
        paths = {}
        for lane, executable, backend in lane_specs:
            evidence_path = output / f"{lane}.evidence"
            paths[lane] = evidence_path
            executions.append(
                run_lane(
                    lane=lane,
                    executable_value=executable,
                    backend=backend,
                    evidence_path=evidence_path,
                    log_path=output / f"{lane}.log",
                    timeout_seconds=arguments.timeout_seconds,
                    cuda_visible_devices=arguments.cuda_visible_devices,
                    runtime_home=runtime_home,
                )
            )
        report = COMPARATOR.compare_evidence(
            paths["cpu-fp64"], paths["cpu-fp32"], paths["cuda-fp32"]
        )
        qualification_inputs_after = qualification_input_identity(lane_executables)
        if qualification_inputs_after != qualification_inputs_before:
            raise RuntimeError(
                "source, build configuration, or wrapper changed while the oracle ran"
            )
        report["qualification_inputs"] = qualification_inputs_after
        report["executions"] = executions
        report_path = output / "report.json"
        COMPARATOR.atomic_write_json(report_path, report)
        marker = {
            "state": "COMPLETE",
            "report": str(report_path),
            "report_sha256": COMPARATOR.sha256_file(report_path),
        }
        COMPARATOR.atomic_write_json(output / "COMPLETE.json", marker)
    except Exception as error:
        failure = {"state": "FAILED", "error": f"{type(error).__name__}: {error}"}
        COMPARATOR.atomic_write_json(output / "FAILED.json", failure)
        raise

    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


def math_is_finite_positive(value: float) -> bool:
    # Kept local so the runner has no third-party dependency.
    return value > 0.0 and value != float("inf") and value == value


if __name__ == "__main__":
    raise SystemExit(main())
