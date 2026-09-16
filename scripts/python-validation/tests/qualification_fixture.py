"""Synthetic qualification-v2 evidence shared by validation self-tests."""

from __future__ import annotations

import json
import importlib.util
import os
import pathlib
import sys
from typing import Any

from gpmeep_provenance import file_record
from gpmeep_qualification_contract import (
    INSTALLED_CUDA_QUALIFICATION_NAME,
    INSTALLED_CUDA_QUALIFICATION_SPECS,
    INSTALLED_LAZY_API_LOG_NAME,
    INSTALLED_LAZY_API_QUALIFICATION_NAME,
    INSTALLED_LAZY_API_TEST_SPECS,
    INSTALLED_MPB_LOG_NAME,
    INSTALLED_MPB_QUALIFICATION_NAME,
    MATERIALIZATION_LOGS,
    marker_for,
)


_LAZY_PATH = pathlib.Path(__file__).resolve().parents[2] / (
    "run-installed-lazy-api-qualification.py"
)
_LAZY_SPEC = importlib.util.spec_from_file_location(
    "gpmeep_lazy_qualification_fixture", _LAZY_PATH
)
if _LAZY_SPEC is None or _LAZY_SPEC.loader is None:
    raise RuntimeError("cannot load lazy qualification fixture source")
_LAZY_MODULE = importlib.util.module_from_spec(_LAZY_SPEC)
sys.modules[_LAZY_SPEC.name] = _LAZY_MODULE
_LAZY_SPEC.loader.exec_module(_LAZY_MODULE)


def _artifact_value(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": record["path"],
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
    }


def write_synthetic_release_cuda_math_configuration(
    repo: pathlib.Path,
    build_root: pathlib.Path,
    install_prefix: pathlib.Path,
) -> dict[str, Any]:
    """Materialize one strict-math release configuration for test receipts.

    The qualification-v2 receipt verifier intentionally reconstructs CUDA
    compiler policy from several independently bound build products. Keep the
    two validation-harness receipt fixtures on this single producer so a new
    required binding cannot silently weaken or strand one fixture.
    """
    compiler_dir = repo / ".envs" / "meep-gpu-cuda-mpi" / "bin"
    compiler_dir.mkdir(parents=True, exist_ok=True)
    nvcc_path = compiler_dir / "nvcc"
    host_path = compiler_dir / "x86_64-conda-linux-gnu-c++"
    nvcc_path.write_text("synthetic nvcc\n", encoding="utf-8")
    host_path.write_text("synthetic host compiler\n", encoding="utf-8")

    canonical_environment = build_root / "canonical-build-environment.json"
    canonical_environment.parent.mkdir(parents=True, exist_ok=True)
    canonical_environment.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "environment": {
                    "CXX": str(host_path),
                    "MEEP_GPU_FAST_MATH": "OFF",
                    "NVCC_PREPEND_FLAGS": f"-ccbin={host_path}",
                },
                "normalized_shell_keys": ["SHLVL", "_"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    configure_argv = [
        "--enable-maintainer-mode",
        "--enable-shared",
        "--enable-single",
        "--enable-cuda",
        "--with-cuda-arch=80",
        "--with-openmp",
        "--with-mpi",
        "--with-python",
        "--without-scheme",
        f"--prefix={install_prefix.resolve()}",
        "--disable-cuda-fast-math",
    ]
    config_status = build_root / "config.status"
    config_status.write_text(
        "  set X /bin/bash '../../configure' "
        + " ".join(f"'{value}'" for value in configure_argv)
        + " 'CC=/synthetic'\n"
        + 'S["NVCCFLAGS"]="-O3"\n',
        encoding="utf-8",
    )
    runtime_flags = build_root / "src" / "cuda-runtime-flags.stamp"
    runtime_flags.parent.mkdir(parents=True, exist_ok=True)
    runtime_flags.write_text(
        f"NVCC={nvcc_path}\n"
        f"CUDAHOSTCXX={host_path}\n"
        "NVCCFLAGS=-O3\n"
        "CUDA_ARCH_FLAGS=-gencode=arch=compute_80,code=sm_80\n",
        encoding="utf-8",
    )
    runtime_cache = build_root / "cuda-runtime-qualification" / "CMakeCache.txt"
    runtime_cache.parent.mkdir(parents=True, exist_ok=True)
    runtime_cache.write_text(
        "MEEP_GPU_FAST_MATH:BOOL=OFF\n"
        "CMAKE_CUDA_FLAGS:STRING=\n"
        "CMAKE_CUDA_FLAGS_DEBUG:STRING=-g\n"
        "CMAKE_CUDA_FLAGS_MINSIZEREL:STRING=-O1 -DNDEBUG\n"
        "CMAKE_CUDA_FLAGS_RELEASE:STRING=-O3 -DNDEBUG\n"
        "CMAKE_CUDA_FLAGS_RELWITHDEBINFO:STRING=-O2 -g -DNDEBUG\n"
        f"CMAKE_CUDA_COMPILER:UNINITIALIZED={nvcc_path}\n"
        f"CMAKE_CUDA_HOST_COMPILER:UNINITIALIZED={host_path}\n",
        encoding="utf-8",
    )
    return {
        "configure_argv": configure_argv,
        "configuration_files": {
            "canonical_build_environment": canonical_environment,
            "config_status": config_status,
            "cuda_runtime_cmake_cache": runtime_cache,
            "cuda_runtime_flags_stamp": runtime_flags,
        },
        "nvcc": nvcc_path,
    }


def write_synthetic_auxiliary_logs(
    log_root: pathlib.Path,
    repo: pathlib.Path,
    artifact_paths: dict[str, pathlib.Path],
) -> None:
    """Write materialization and same-process CUDA-shaped fixture logs."""
    for name in MATERIALIZATION_LOGS:
        marker = f"gpmeep-libtool-materialization:{name}:PASS"
        (log_root / name).write_text(marker + "\n", encoding="utf-8")

    extension = file_record(
        artifact_paths["installed_python_extension"], repo
    )
    libmeep = file_record(artifact_paths["installed_libmeep"], repo)
    mpb_extension = file_record(
        artifact_paths["installed_mpb_extension"], repo
    )
    libpympb = file_record(artifact_paths["installed_libpympb"], repo)
    mpb_value = {
        "schema_version": 1,
        "qualification": INSTALLED_MPB_QUALIFICATION_NAME,
        "log_name": INSTALLED_MPB_LOG_NAME,
        "world_size": 1,
        "single_precision_meep": True,
        "mpb_extension": _artifact_value(mpb_extension),
        "libpympb": _artifact_value(libpympb),
        "python_extension": _artifact_value(extension),
        "libmeep": _artifact_value(libmeep),
        "frequencies": [0.25, 0.5],
        "repeat_frequencies": [0.25, 0.5],
        "maximum_repeat_error": 0.0,
    }
    (log_root / INSTALLED_MPB_LOG_NAME).write_text(
        json.dumps(mpb_value, sort_keys=True, separators=(",", ":"))
        + "\n"
        + marker_for(INSTALLED_MPB_LOG_NAME).decode("ascii")
        + "\n",
        encoding="utf-8",
    )
    lazy_tests = [
        {
            "source": file_record(repo / relative, repo),
            "selectors": list(selectors),
            "tests_run": tests_run,
            "pythonpath": (
                str((repo / "python" / "tests").resolve())
                + os.pathsep
                + str(artifact_paths["installed_python_extension"].resolve().parent.parent)
            ),
            "returncode": 0,
            "elapsed_seconds": 0.1,
            "stdout_sha256": "1" * 64,
            "stderr_sha256": "2" * 64,
        }
        for relative, selectors, tests_run in INSTALLED_LAZY_API_TEST_SPECS
    ]
    lazy_value = {
        "schema_version": 1,
        "qualification": INSTALLED_LAZY_API_QUALIFICATION_NAME,
        "log_name": INSTALLED_LAZY_API_LOG_NAME,
        "deferred_module_roots": ["jax", "matplotlib", "scipy"],
        "initial_modules": {"jax": False, "matplotlib": False, "scipy": False},
        "core_surface": {"meep.Simulation": True},
        "discoverable_surface": {"meep.visualization": True},
        "module_aliases": {"meep.visualization": "meep.visualization"},
        "wildcard_surface": {"meep.plot2D": True},
        "adjoint_wildcard_names": sorted(
            _LAZY_MODULE.HISTORICAL_ADJOINT_WILDCARD
        ),
        "plot2d_axes_annotation_is_matplotlib_axes": True,
        "tests": lazy_tests,
        "python_extension": _artifact_value(extension),
        "libmeep": _artifact_value(libmeep),
    }
    (log_root / INSTALLED_LAZY_API_LOG_NAME).write_text(
        json.dumps(lazy_value, sort_keys=True, separators=(",", ":"))
        + "\n"
        + marker_for(INSTALLED_LAZY_API_LOG_NAME).decode("ascii")
        + "\n",
        encoding="utf-8",
    )
    for log_name, spec in INSTALLED_CUDA_QUALIFICATION_SPECS.items():
        suite_file = file_record(repo / str(spec["test_file"]), repo)
        rank_records = []
        for rank in range(int(spec["world_size"])):
            rank_records.append(
                {
                    "world_rank": rank,
                    "world_size": spec["world_size"],
                    "suite": {
                        "tests_run": spec["tests_run"],
                        "failures": 0,
                        "errors": 0,
                        "skipped": 0,
                        "expected_failures": 0,
                        "unexpected_successes": 0,
                        "successful": True,
                    },
                    "python_extension": _artifact_value(extension),
                    "libmeep": _artifact_value(libmeep),
                    "probe": {
                        "requested_backend": "cuda",
                        "active_backend": "cuda",
                        "selected_device": rank,
                        "selected_device_identifier": f"GPU-fixture-{rank}",
                        "cuda_execution_selected": True,
                        "execution_diagnostic": "fixture selected CUDA",
                        "statistics": {
                            "dispatch": {
                                "cuda_curl_calls": 2,
                                "cpu_curl_calls": 0,
                            },
                            "field_updates": {
                                "cuda_update_eh_calls": 2,
                                "cpu_update_eh_calls": 0,
                            },
                            "sources": {
                                "cuda_source_calls": 1 if rank == 0 else 0,
                                "cpu_source_calls": 0,
                            },
                        },
                    },
                }
            )
        value = {
            "schema_version": 1,
            "qualification": INSTALLED_CUDA_QUALIFICATION_NAME,
            "log_name": log_name,
            "mode": spec["mode"],
            "expected_world_size": spec["world_size"],
            "environment": spec["environment"],
            "suite_file": _artifact_value(suite_file),
            "python_extension": _artifact_value(extension),
            "libmeep": _artifact_value(libmeep),
            "rank_records": rank_records,
        }
        prefix = ""
        if spec["mode"] == "adjoint":
            prefix = "".join(
                "gpmeep-fd-direction:"
                + json.dumps({"name": name, "pass": True})
                + "\n"
                for name in (
                    "cosine",
                    "quasiperiodic-sine",
                    "component-6",
                    "component-12",
                    "component-18",
                )
            )
        (log_root / log_name).write_text(
            prefix
            + json.dumps(value, sort_keys=True, separators=(",", ":"))
            + "\n"
            + marker_for(log_name).decode("ascii")
            + "\n",
            encoding="utf-8",
        )
