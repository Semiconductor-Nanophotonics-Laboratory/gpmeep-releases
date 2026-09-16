#!/usr/bin/env python3
"""Single source of truth for the authoritative qualification-v2 evidence.

The v2 name identifies the qualification envelope/family. Extending its exact
required gate set is a monotonic source- and identity-bound strengthening: an
older receipt is stale under the newer checked-in contract and fails closed;
it does not require a qualification-family rename.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import pathlib
import re
import select
import shlex
import signal
import stat
import struct
import sys
import tempfile
import types
from typing import Any, Iterable


CONTRACT_NAME = "gpmeep-cuda-mpi-python-fp32-v2"
CPU_REFERENCE_CONTRACT_NAME = "gpmeep-cpu-mpi-python-fp64-v1"
CPU_REFERENCE_BUILD_KIND = "cpu-mpi-python-fp64"
CPU_REFERENCE_MARKER = "gpmeep-cpu-mpi-fp64-qualification:"
CPU_REFERENCE_SCHEMA = "gpmeep-cpu-mpi-fp64-qualification-v1"
CPU_REFERENCE_LOG_SPECS = {
    "in-place-one-rank.log": ("in-place", 1, "python_extension", "libmeep"),
    "in-place-two-rank.log": ("in-place", 2, "python_extension", "libmeep"),
    "installed-one-rank.log": (
        "installed",
        1,
        "installed_python_extension",
        "installed_libmeep",
    ),
    "installed-two-rank.log": (
        "installed",
        2,
        "installed_python_extension",
        "installed_libmeep",
    ),
}
CPU_REFERENCE_CONFIGURATION_FILES = {
    "conda_prefix_content_audit",
    "config_h",
    "config_status",
    "environment_explicit",
    "openmpi_qualification_params",
}
CPU_REFERENCE_ARTIFACTS = {
    "python_extension",
    "libmeep",
    "installed_python_extension",
    "installed_libmeep",
}
CPU_REFERENCE_MANIFESTS = {
    "host_build_logs",
    "in_place_python",
    "installed_python",
    "installed_prefix",
    "installed_environment",
    "qualification_logs",
}
CPU_REFERENCE_TOOLS = {
    "autoreconf",
    "cc",
    "c++",
    "make",
    "python",
    "swig",
    "mpicxx",
    "mpiexec",
    "h5pcc",
}
SCHEMA_VERSION = 1
AUDIT_LOG_NAME = "required-qualification-logs.log"
SENTINEL_LOG_NAME = "qualification-mutation-sentinel.log"
STEP_FINITE_FAILURE_LOG_NAME = "step-finite-rank-failure-two-rank.log"
STEP_FINITE_FAILURE_DIAGNOSTIC = "simulation fields are NaN or Inf"
MAX_QUALIFICATION_LOG_BYTES = 256 * 1024 * 1024
IMMUTABLE_EMPTY_DIRECTORY_KIND = "immutable-empty-directory"
IMMUTABLE_EMPTY_DIRECTORIES_KEY = "immutable_empty_directories"
REQUIRED_IMMUTABLE_EMPTY_DIRECTORIES = {
    "qualification_pycache": "0700",
}
REQUIRED_IMMUTABLE_EMPTY_DIRECTORY_RELATIVE_PATHS = {
    "qualification_pycache": "qualification-pycache",
}

# This is deliberately an explicit checked-in set.  A builder edit which adds
# a new PASS-producing gate must update this contract; deriving the set from
# the mutable builder would make omission indistinguishable from success.
REQUIRED_QUALIFICATION_LOGS = (
    "ade-state-contract-cuda-required-hidden-one-rank.log",
    "ade-state-contract-one-rank.log",
    "automatic-backend-request-mismatch-two-rank.log",
    "automatic-explicit-duplicate-device-two-rank.log",
    "automatic-explicit-oversubscription-two-rank.log",
    "automatic-invalid-backend-hidden-one-rank.log",
    "automatic-invalid-device-two-rank.log",
    "automatic-overflow-threshold-hidden-one-rank.log",
    "automatic-threshold-mismatch-two-rank.log",
    "boundary-eh-overlap-default-off-two-rank.log",
    "boundary-eh-overlap-eager-disabled-two-rank.log",
    "boundary-eh-overlap-exact-comparison.log",
    "boundary-eh-overlap-mixed-disabled-two-rank.log",
    "boundary-eh-overlap-mixed-exact-comparison.log",
    "boundary-eh-overlap-opt-in-conflict-two-rank.log",
    "boundary-eh-overlap-opt-in-empty-two-rank.log",
    "boundary-eh-overlap-opt-in-invalid-two-rank.log",
    "boundary-eh-overlap-opt-in-zero-two-rank.log",
    "boundary-eh-overlap-pinned-two-rank.log",
    "boundary-eh-overlap-two-rank.log",
    "boundary-eh-overlap-waitall-default-off-two-rank.log",
    "boundary-eh-overlap-waitall-exact-comparison.log",
    "boundary-eh-overlap-waitall-two-rank.log",
    "boundary-graph-lifecycle-two-rank.log",
    "boundary-lifetime-double-failure-one-rank.log",
    "boundary-lifecycle-profile-mismatch-two-rank.log",
    "comms-manager-completion-policy-mismatch-two-rank.log",
    "comms-manager-destructor-unwind-two-rank.log",
    "comms-manager-eager-failure-two-rank.log",
    "comms-manager-invalid-completion-policy-two-rank.log",
    "comms-manager-pinned-failure-two-rank.log",
    "comms-manager-waitall-failure-two-rank.log",
    "cpp-mpi-test-suite.log",
    "cuda-architecture-metadata.log",
    "cuda-host-formula.log",
    "cuda-near2far-runtime-validation.log",
    "cuda-runtime-validation.log",
    "cuda-smoke.log",
    "dft-decimation-two-rank.log",
    "dft-norm-rank-failure-two-rank.log",
    "dft-phase-sharing-disabled-two-rank.log",
    "dft-phase-sharing-exact-two-rank.log",
    "eager-pingpong-two-rank.log",
    "eager-pingpong-waitall-two-rank.log",
    "gpu-backend-direct-elf-provenance.log",
    "gpu-mpi-performance-direct-elf-provenance.log",
    "gpu-step-db-auto-cpu-one-rank.log",
    "gpu-step-db-auto-cpu-two-rank.log",
    "gpu-step-db-auto-cuda-aware-two-rank.log",
    "gpu-step-db-auto-cuda-aware-waitall-two-rank.log",
    "gpu-step-db-auto-cuda-one-rank.log",
    "gpu-step-db-auto-cuda-two-rank.log",
    "gpu-step-db-cw-breakdown-one-rank.log",
    "gpu-step-db-cw-breakdown-two-rank.log",
    "gpu-step-db-cw-solver-one-rank.log",
    "gpu-step-db-cw-vector-one-rank.log",
    "gpu-step-db-direct-elf-provenance.log",
    "gpu-step-db-dynamic-claim-two-rank.log",
    "gpu-step-db-full-one-rank.log",
    "gpu-step-db-full-pinned-two-rank.log",
    "gpu-step-db-full-two-rank.log",
    "gpu-step-db-full-waitall-two-rank.log",
    "gpu-step-db-ldos-deterministic-one-rank.log",
    "gpu-step-db-ldos-deterministic-two-rank.log",
    "gpu-step-db-ldos-migration-one-rank.log",
    "gpu-step-db-near2far2d-two-gpu.log",
    "gpu-step-db-near2farcyl-two-gpu.log",
    "gpu-step-db-phase-source-one-rank.log",
    "gpu-step-db-phase-source-two-rank.log",
    "halo-curl-overlap-bfast-rejected-two-rank.log",
    "halo-curl-overlap-default-off-two-rank.log",
    "halo-curl-overlap-eager-disabled-two-rank.log",
    "halo-curl-overlap-exact-comparison.log",
    "halo-curl-overlap-opt-in-conflict-two-rank.log",
    "halo-curl-overlap-opt-in-empty-two-rank.log",
    "halo-curl-overlap-opt-in-invalid-two-rank.log",
    "halo-curl-overlap-opt-in-zero-two-rank.log",
    "halo-curl-overlap-pinned-two-rank.log",
    "halo-curl-overlap-small-rejected-two-rank.log",
    "halo-curl-overlap-tiled-rejected-two-rank.log",
    "halo-curl-overlap-two-rank.log",
    "in-place-singleton-import.log",
    "in-place-two-rank-import.log",
    "initial-condition-invalid-rank-two-rank.log",
    "initial-condition-preflight-mismatch-two-rank.log",
    "initial-condition-profile-mismatch-two-rank.log",
    "in-place-python-runtime-provenance.log",
    "installed-adjoint-default-material-grid-cuda.log",
    "installed-adjoint-default-material-grid.log",
    "installed-lazy-api-compatibility.log",
    "installed-mpb-runtime.log",
    "installed-python-gpu-backend-singleton.log",
    "installed-python-gpu-backend-two-rank.log",
    "installed-python-runtime-provenance.log",
    "installed-singleton-import.log",
    "installed-two-rank-import.log",
    "integrate2-rank-failure-two-rank.log",
    "mixed-dispersive-cuda-aware-two-rank.log",
    "mixed-dispersive-pinned-two-rank.log",
    "pingpong-disabled-two-rank.log",
    "pinned-boundary-two-rank.log",
    "python-gpu-backend-singleton.log",
    "python-gpu-backend-two-rank.log",
    SENTINEL_LOG_NAME,
    "runtime-dependency-closure.log",
    "source-enabled-mismatch-two-rank.log",
    "source-profile-invalid-rank-two-rank.log",
    "source-profile-mismatch-two-rank.log",
    "step-finite-rank-failure-two-rank.log",
    "tile-coalescing-disabled-two-rank.log",
    "tile-coalescing-exact-comparison.log",
    "tile-coalescing-two-rank.log",
)
ALL_REQUIRED_QUALIFICATION_LOGS = tuple(
    sorted((*REQUIRED_QUALIFICATION_LOGS, AUDIT_LOG_NAME))
)
MATERIALIZATION_LOGS = (
    "gpu-backend-libtool-materialization.log",
    "gpu-mpi-performance-libtool-materialization.log",
    "gpu-step-db-libtool-materialization.log",
)
ALL_QUALIFICATION_DIRECTORY_LOGS = tuple(
    sorted((*ALL_REQUIRED_QUALIFICATION_LOGS, *MATERIALIZATION_LOGS))
)
REQUIRED_ARTIFACT_BINDINGS = (
    "cuda_architecture_test",
    "cuda_formula_test",
    "cuda_near2far_runtime_validation",
    "cuda_runtime_validation",
    "cuda_smoke",
    "fd_allocation_shim",
    "gpu_backend_test",
    "gpu_mpi_performance",
    "gpu_step_db_test",
    "installed_libmeep",
    "installed_libpympb",
    "installed_mpb_extension",
    "installed_python_extension",
    "libmeep",
    "python_extension",
)
REQUIRED_CONFIGURATION_BINDINGS = (
    "canonical_build_environment",
    "config_status",
    "cuda_runtime_cmake_cache",
    "cuda_runtime_flags_stamp",
    "qualification_contract_v2",
    "qualification_identity_after",
    "qualification_identity_before",
)

def _contains_unsafe_cuda_math_token(tokens: Iterable[str]) -> bool:
    """Recognize unsafe compiler options without treating paths as options."""

    values = list(tokens)
    if any(
        token in {"--use_fast_math", "-use_fast_math", "-ffast-math", "-Ofast"}
        for token in values
    ):
        return True
    if any(
        token.startswith("--use_fast_math=")
        or token.startswith("-use_fast_math=")
        for token in values
    ):
        return True
    unsafe_values = {
        "--ftz": "true",
        "-ftz": "true",
        "--prec-div": "false",
        "-prec-div": "false",
        "--prec-sqrt": "false",
        "-prec-sqrt": "false",
    }
    for index, token in enumerate(values):
        for option, unsafe_value in unsafe_values.items():
            if token == option and index + 1 < len(values):
                if values[index + 1].lower() == unsafe_value:
                    return True
            elif token.startswith(option + "="):
                if token.split("=", 1)[1].lower() == unsafe_value:
                    return True
    return False


def _receipt_configuration_path(
    configuration_files: dict[str, Any], name: str, repo: pathlib.Path
) -> pathlib.Path:
    record = configuration_files.get(name)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise QualificationContractError(
            f"release CUDA math configuration binding is invalid: {name}"
        )
    return _resolve_recorded_path(record["path"], repo)


def _release_build_directory(receipt: dict[str, Any], repo: pathlib.Path) -> pathlib.Path:
    value = receipt.get("build_dir")
    if not isinstance(value, str) or not value:
        raise QualificationContractError("release receipt build directory is invalid")
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = repo / path
    try:
        resolved = path.resolve(strict=True)
        expected = (repo / "build" / "meep-cuda-mpi-python-fp32").resolve(
            strict=True
        )
    except OSError as exc:
        raise QualificationContractError("release receipt build directory is absent") from exc
    if resolved != expected:
        raise QualificationContractError(
            "release receipt build directory is not the fixed authoritative path"
        )
    return resolved


def _require_release_configuration_path(
    receipt: dict[str, Any], configuration_files: dict[str, Any],
    name: str, relative: str, repo: pathlib.Path
) -> pathlib.Path:
    recorded = _receipt_configuration_path(configuration_files, name, repo)
    expected = _release_build_directory(receipt, repo) / relative
    try:
        recorded_resolved = recorded.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise QualificationContractError(
            f"release CUDA math configuration path is absent: {name}"
        ) from exc
    if recorded_resolved != expected_resolved:
        raise QualificationContractError(
            f"release CUDA math configuration path is not build-bound: {name}"
        )
    return expected_resolved


def _parse_unique_key_value_file(path: pathlib.Path, label: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise QualificationContractError(f"cannot read {label}: {path}") from exc
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise QualificationContractError(f"{label} has duplicate or malformed entries")
        values[key] = value
    return values


def _parse_cmake_cache_assignments(path: pathlib.Path) -> dict[str, tuple[str, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise QualificationContractError(
            f"cannot read receipt-bound CUDA CMake cache: {path}"
        ) from exc
    values: dict[str, tuple[str, str]] = {}
    for line in lines:
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        match = re.fullmatch(r"([^:=]+):([^=]+)=(.*)", line)
        if match is None:
            continue
        key, value_type, value = match.groups()
        if key in values:
            raise QualificationContractError(
                "receipt-bound CUDA CMake cache has duplicate assignments"
            )
        values[key] = (value_type, value)
    return values


def validate_release_cuda_math_policy(
    receipt: dict[str, Any], repo: pathlib.Path
) -> None:
    """Require evidence that every authoritative CUDA build used strict math."""

    configuration = receipt.get("configuration")
    configuration_files = receipt.get("configuration_files")
    if not isinstance(configuration, dict) or not isinstance(configuration_files, dict):
        raise QualificationContractError("release CUDA math receipt groups are invalid")
    if receipt.get("build_kind") != "cuda-mpi-python-fp32":
        raise QualificationContractError("release receipt has the wrong build kind")
    builder = configuration.get("builder")
    expected_builder = (repo / "scripts" / "build-meep-cuda-mpi-python.sh").resolve()
    if (
        not isinstance(builder, dict)
        or not isinstance(builder.get("path"), str)
        or _resolve_recorded_path(builder["path"], repo).resolve()
        != expected_builder
    ):
        raise QualificationContractError(
            "release receipt was not produced by the authoritative builder"
        )
    configure_argv = configuration.get("configure_argv")
    if not isinstance(configure_argv, list) or not all(
        isinstance(value, str) for value in configure_argv
    ):
        raise QualificationContractError("release CUDA configure arguments are invalid")
    if (
        configure_argv.count("--disable-cuda-fast-math") != 1
        or "--enable-cuda-fast-math" in configure_argv
    ):
        raise QualificationContractError(
            "authoritative CUDA release requires explicit fast-math disable"
        )
    environment = configuration.get("environment")
    if not isinstance(environment, dict) or environment.get("MEEP_GPU_FAST_MATH") != "OFF":
        raise QualificationContractError(
            "build receipt does not record MEEP_GPU_FAST_MATH=OFF"
        )

    canonical_path = _require_release_configuration_path(
        receipt, configuration_files, "canonical_build_environment",
        "canonical-build-environment.json", repo
    )
    try:
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationContractError(
            "receipt-bound canonical build environment is invalid"
        ) from exc
    canonical_environment = canonical.get("environment") if isinstance(canonical, dict) else None
    if (
        not isinstance(canonical, dict)
        or canonical.get("schema_version") != 2
        or not isinstance(canonical_environment, dict)
        or canonical_environment.get("MEEP_GPU_FAST_MATH") != "OFF"
    ):
        raise QualificationContractError(
            "canonical build environment does not force MEEP_GPU_FAST_MATH=OFF"
        )
    forbidden_environment_keys = {
        "CUDAFLAGS",
        "CUDA_FLAGS",
        "CMAKE_CUDA_FLAGS",
        "NVCCFLAGS",
        "NVCC_APPEND_FLAGS",
    }
    if forbidden_environment_keys.intersection(canonical_environment):
        raise QualificationContractError(
            "canonical build environment contains unapproved variables for CUDA flags"
        )
    prepend = canonical_environment.get("NVCC_PREPEND_FLAGS")
    try:
        prepend_tokens = shlex.split(prepend) if isinstance(prepend, str) else []
    except ValueError as exc:
        raise QualificationContractError("NVCC_PREPEND_FLAGS is malformed") from exc
    if (
        len(prepend_tokens) != 1
        or not prepend_tokens[0].startswith("-ccbin=")
        or _contains_unsafe_cuda_math_token(prepend_tokens)
    ):
        raise QualificationContractError(
            "canonical build environment has unsafe NVCC implicit flags"
        )

    flags_path = _require_release_configuration_path(
        receipt, configuration_files, "cuda_runtime_flags_stamp",
        "src/cuda-runtime-flags.stamp", repo
    )
    stamp = _parse_unique_key_value_file(
        flags_path, "receipt-bound CUDA runtime flags stamp"
    )
    required_stamp = {"NVCC", "CUDAHOSTCXX", "NVCCFLAGS", "CUDA_ARCH_FLAGS"}
    if set(stamp) != required_stamp:
        raise QualificationContractError(
            "receipt-bound CUDA runtime flags stamp is incomplete"
        )
    try:
        nvcc_flags = shlex.split(stamp["NVCCFLAGS"])
        architecture_flags = shlex.split(stamp["CUDA_ARCH_FLAGS"])
    except ValueError as exc:
        raise QualificationContractError(
            "receipt-bound CUDA runtime flags stamp is malformed"
        ) from exc
    if (
        nvcc_flags != ["-O3"]
        or _contains_unsafe_cuda_math_token(nvcc_flags)
        or _contains_unsafe_cuda_math_token(architecture_flags)
        or not architecture_flags
        or any(
            re.fullmatch(
                r"-gencode=arch=compute_[0-9]+,code=(?:sm|compute)_[0-9]+",
                token,
            )
            is None
            for token in architecture_flags
        )
    ):
        raise QualificationContractError(
            "receipt-bound CUDA runtime flags are not strict -O3 math"
        )

    cache_path = _require_release_configuration_path(
        receipt, configuration_files, "cuda_runtime_cmake_cache",
        "cuda-runtime-qualification/CMakeCache.txt", repo
    )
    cache = _parse_cmake_cache_assignments(cache_path)
    if cache.get("MEEP_GPU_FAST_MATH") != ("BOOL", "OFF"):
        raise QualificationContractError(
            "receipt-bound CUDA CMake cache does not disable fast-math"
        )
    toolchain = receipt.get("toolchain")
    nvcc_record = toolchain.get("nvcc") if isinstance(toolchain, dict) else None
    if not isinstance(nvcc_record, dict) or not isinstance(nvcc_record.get("path"), str):
        raise QualificationContractError("release receipt NVCC identity is absent")
    try:
        stamp_nvcc = pathlib.Path(stamp["NVCC"]).resolve(strict=True)
        receipt_nvcc = _resolve_recorded_path(
            nvcc_record["path"], repo
        ).resolve(strict=True)
        stamp_host = pathlib.Path(stamp["CUDAHOSTCXX"]).resolve(strict=True)
        canonical_host = pathlib.Path(
            str(canonical_environment.get("CXX", ""))
        ).resolve(strict=True)
    except OSError as exc:
        raise QualificationContractError(
            "release CUDA compiler identity is absent"
        ) from exc
    if stamp_nvcc != receipt_nvcc or stamp_host != canonical_host:
        raise QualificationContractError(
            "release CUDA flags stamp compiler identity is unbound"
        )
    cmake_nvcc = cache.get("CMAKE_CUDA_COMPILER")
    cmake_host = cache.get("CMAKE_CUDA_HOST_COMPILER")
    if cmake_nvcc is None or cmake_host is None:
        raise QualificationContractError(
            "receipt-bound CUDA CMake compiler identities are absent"
        )
    try:
        cmake_nvcc_path = pathlib.Path(cmake_nvcc[1]).resolve(strict=True)
        cmake_host_path = pathlib.Path(cmake_host[1]).resolve(strict=True)
    except OSError as exc:
        raise QualificationContractError(
            "receipt-bound CUDA CMake compiler identity is absent"
        ) from exc
    if cmake_nvcc_path != receipt_nvcc or cmake_host_path != canonical_host:
        raise QualificationContractError(
            "receipt-bound CUDA CMake compiler identity is unbound"
        )
    expected_cmake_cuda_flags = {
        "CMAKE_CUDA_FLAGS": [],
        "CMAKE_CUDA_FLAGS_DEBUG": ["-g"],
        "CMAKE_CUDA_FLAGS_MINSIZEREL": ["-O1", "-DNDEBUG"],
        "CMAKE_CUDA_FLAGS_RELEASE": ["-O3", "-DNDEBUG"],
        "CMAKE_CUDA_FLAGS_RELWITHDEBINFO": ["-O2", "-g", "-DNDEBUG"],
    }
    for key, expected_tokens in expected_cmake_cuda_flags.items():
        assignment = cache.get(key)
        if assignment is None:
            raise QualificationContractError(
                f"receipt-bound CUDA CMake cache lacks {key}"
            )
        try:
            tokens = shlex.split(assignment[1])
        except ValueError as exc:
            raise QualificationContractError(
                f"receipt-bound CUDA CMake cache has malformed {key}"
            ) from exc
        if tokens != expected_tokens or _contains_unsafe_cuda_math_token(tokens):
            raise QualificationContractError(
                "receipt-bound CUDA CMake flags enable unsafe fast-math or differ from the release profile"
            )

    config_status_path = _require_release_configuration_path(
        receipt, configuration_files, "config_status", "config.status", repo
    )
    try:
        config_status = config_status_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QualificationContractError(
            "cannot read receipt-bound config.status"
        ) from exc
    if config_status.splitlines().count('S["NVCCFLAGS"]="-O3"') != 1:
        raise QualificationContractError(
            "receipt-bound config.status does not record strict NVCCFLAGS=-O3"
        )
    configure_line = next(
        (
            line.strip()
            for line in config_status.splitlines()
            if line.startswith("  set X /bin/bash")
        ),
        None,
    )
    if configure_line is None:
        raise QualificationContractError(
            "receipt-bound config.status lacks the configure invocation"
        )
    try:
        status_tokens = shlex.split(configure_line)
    except ValueError as exc:
        raise QualificationContractError(
            "receipt-bound config.status configure invocation is malformed"
        ) from exc
    if len(status_tokens) < 5 or status_tokens[:3] != ["set", "X", "/bin/bash"]:
        raise QualificationContractError(
            "receipt-bound config.status configure invocation is malformed"
        )
    status_configure: list[str] = []
    for token in status_tokens[4:]:
        if not token.startswith("--"):
            break
        status_configure.append(token)
    if status_configure != configure_argv:
        raise QualificationContractError(
            "receipt and config.status configure arguments disagree"
        )
PYTHON_RUNTIME_LOG_BINDINGS = {
    "in-place-python-runtime-provenance.log": ("python_extension", "libmeep"),
    "installed-python-runtime-provenance.log": (
        "installed_python_extension",
        "installed_libmeep",
    ),
    "installed-python-gpu-backend-singleton.log": (
        "installed_python_extension",
        "installed_libmeep",
    ),
    "installed-python-gpu-backend-two-rank.log": (
        "installed_python_extension",
        "installed_libmeep",
    ),
    "installed-adjoint-default-material-grid-cuda.log": (
        "installed_python_extension",
        "installed_libmeep",
    ),
    "installed-lazy-api-compatibility.log": (
        "installed_python_extension",
        "installed_libmeep",
    ),
}
INSTALLED_LAZY_API_LOG_NAME = "installed-lazy-api-compatibility.log"
INSTALLED_LAZY_API_QUALIFICATION_NAME = "gpmeep-installed-lazy-api-v1"
INSTALLED_MPB_LOG_NAME = "installed-mpb-runtime.log"
INSTALLED_MPB_QUALIFICATION_NAME = "gpmeep-installed-mpb-runtime-v1"
HISTORICAL_ADJOINT_WILDCARD_COUNT = 73
HISTORICAL_ADJOINT_WILDCARD_SHA256 = (
    "54cc6f290305ab995a81c1e083be4b4fe95c8f94d15202e80dacb1b351ffe5e1"
)
INSTALLED_LAZY_API_TEST_SPECS = (
    ("python/tests/test_ring.py", ("TestRing.test_pade",), 1),
    (
        "python/tests/test_visualization.py",
        ("TestVisualization.test_plot2D",),
        1,
    ),
    ("python/tests/test_adjoint_utils.py", (), 7),
    (
        "python/tests/test_adjoint_jax.py",
        (
            "UtilsTest.test_mode_monitor_helpers",
            "UtilsTest.test_dist_dft_pointers",
            (
                "WrapperTest.test_wrapper_gradients_0_"
                "1500_1550bw_01relative_gaussian_port1"
            ),
        ),
        3,
    ),
    (
        "python/tests/test_adjoint_solver.py",
        (
            "TestAdjointSolver.test_multifreq_monitor",
            "TestAdjointSolver.test_mode_source_bandwidth",
        ),
        2,
    ),
)
INSTALLED_CUDA_QUALIFICATION_NAME = (
    "gpmeep-installed-cuda-python-same-process-v1"
)
INSTALLED_CUDA_QUALIFICATION_SPECS = {
    "installed-python-gpu-backend-singleton.log": {
        "mode": "gpu-backend",
        "test_file": "python/tests/test_gpu_backend.py",
        "tests_run": 9,
        "world_size": 1,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "GPMEEP_REQUIRE_CUDA_TEST": "1",
            "MEEP_GPU_ALLOW_OVERSUBSCRIBE": None,
            "MEEP_GPU_BACKEND": "cuda",
            "MEEP_GPU_DEVICE": None,
            "MEEP_GPU_STRICT": "1",
            "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": None,
            "PYTHONNOUSERSITE": "1",
        },
    },
    "installed-python-gpu-backend-two-rank.log": {
        "mode": "gpu-backend",
        "test_file": "python/tests/test_gpu_backend.py",
        "tests_run": 9,
        "world_size": 2,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "GPMEEP_REQUIRE_CUDA_TEST": "1",
            "MEEP_GPU_ALLOW_OVERSUBSCRIBE": "0",
            "MEEP_GPU_BACKEND": "cuda",
            "MEEP_GPU_DEVICE": None,
            "MEEP_GPU_STRICT": "1",
            "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": None,
            "PYTHONNOUSERSITE": "1",
        },
    },
    "installed-adjoint-default-material-grid-cuda.log": {
        "mode": "adjoint",
        "test_file": "python/tests/test_adjoint_default_material_grid.py",
        "tests_run": 3,
        "world_size": 1,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "GPMEEP_REQUIRE_CUDA_TEST": "1",
            "MEEP_GPU_ALLOW_OVERSUBSCRIBE": None,
            "MEEP_GPU_BACKEND": "cuda",
            "MEEP_GPU_DEVICE": None,
            "MEEP_GPU_STRICT": "1",
            "MEEP_GPU_VALIDATION_ALLOW_SMALL_DOMAINS": "1",
            "PYTHONNOUSERSITE": "1",
        },
    },
}
DIRECT_ELF_LOG_BINDINGS = {
    "gpu-backend-direct-elf-provenance.log": "gpu_backend_test",
    "gpu-mpi-performance-direct-elf-provenance.log": "gpu_mpi_performance",
    "gpu-step-db-direct-elf-provenance.log": "gpu_step_db_test",
}


class QualificationContractError(RuntimeError):
    """Raised when qualification-v2 evidence is incomplete or unbound."""


def marker_for(name: str) -> bytes:
    return f"gpmeep-qualification:{name}:PASS".encode("ascii")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_path(path: pathlib.Path, repo: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _file_record(path: pathlib.Path, repo: pathlib.Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise QualificationContractError(
            f"qualification evidence is not a regular non-symlink file: {path}"
        )
    return {
        "path": _display_path(path, repo),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _resolve_recorded_path(value: str, repo: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else repo / path


def _literal_absolute_path(path: pathlib.Path) -> pathlib.Path:
    """Make a path absolute without following its final symlink."""
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def _display_literal_path(path: pathlib.Path, repo: pathlib.Path) -> str:
    absolute = _literal_absolute_path(path)
    repo_absolute = _literal_absolute_path(repo)
    try:
        return absolute.relative_to(repo_absolute).as_posix()
    except ValueError:
        return str(absolute)


def _immutable_empty_directory_record(
    path: pathlib.Path, repo: pathlib.Path
) -> dict[str, Any]:
    """Describe a named directory that must stay present and exactly empty."""
    absolute = _literal_absolute_path(path)
    try:
        info = os.lstat(absolute)
    except OSError as exc:
        raise QualificationContractError(
            f"immutable empty directory is absent: {absolute}: {exc}"
        ) from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise QualificationContractError(
            f"immutable empty directory is not a non-symlink directory: {absolute}"
        )
    try:
        with os.scandir(absolute) as entries:
            first_entry = next(entries, None)
    except OSError as exc:
        raise QualificationContractError(
            f"cannot inspect immutable empty directory: {absolute}: {exc}"
        ) from exc
    if first_entry is not None:
        raise QualificationContractError(
            "immutable empty directory is not empty: "
            f"{absolute} (found {first_entry.name!r})"
        )
    return {
        "kind": IMMUTABLE_EMPTY_DIRECTORY_KIND,
        "path": _display_literal_path(absolute, repo),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _identity_immutable_empty_directory_paths(
    identity: dict[str, Any],
    repo: pathlib.Path,
    *,
    require_contract_bindings: bool = False,
    expected_directory_parent: pathlib.Path | None = None,
) -> dict[str, pathlib.Path]:
    records = identity.get(IMMUTABLE_EMPTY_DIRECTORIES_KEY)
    if not isinstance(records, dict):
        raise QualificationContractError(
            "immutable empty directory identity bindings are invalid"
        )
    if require_contract_bindings and set(records) != set(
        REQUIRED_IMMUTABLE_EMPTY_DIRECTORIES
    ):
        raise QualificationContractError(
            "qualification identity has the wrong immutable empty directory "
            "binding set"
        )
    result: dict[str, pathlib.Path] = {}
    for name, record in sorted(records.items()):
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(record, dict)
            or set(record)
            != {"kind", "path", "mode", "device", "inode"}
            or record.get("kind") != IMMUTABLE_EMPTY_DIRECTORY_KIND
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("mode"), str)
            or not isinstance(record.get("device"), int)
            or isinstance(record.get("device"), bool)
            or not isinstance(record.get("inode"), int)
            or isinstance(record.get("inode"), bool)
        ):
            raise QualificationContractError(
                f"immutable empty directory identity binding is invalid: {name!r}"
            )
        path = _resolve_recorded_path(record["path"], repo)
        if require_contract_bindings and record["mode"] != (
            REQUIRED_IMMUTABLE_EMPTY_DIRECTORIES[name]
        ):
            raise QualificationContractError(
                f"immutable empty directory has the wrong required mode: {name}"
            )
        if expected_directory_parent is not None:
            expected_path = _literal_absolute_path(
                expected_directory_parent
                / REQUIRED_IMMUTABLE_EMPTY_DIRECTORY_RELATIVE_PATHS[name]
            )
            if _literal_absolute_path(path) != expected_path:
                raise QualificationContractError(
                    f"immutable empty directory has the wrong required path: {name}"
                )
        current = _immutable_empty_directory_record(path, repo)
        if current != record:
            raise QualificationContractError(
                f"immutable empty directory identity changed: {name}"
            )
        result[name] = _literal_absolute_path(path)
    return result


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationContractError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationContractError(f"qualification JSON is not an object: {path}")
    return value


def _atomic_write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_log(path: pathlib.Path, name: str) -> dict[str, Any]:
    if path.name != name or not path.is_file() or path.is_symlink():
        raise QualificationContractError(
            f"required qualification log is absent or unsafe: {name}"
        )
    size = path.stat().st_size
    if size > MAX_QUALIFICATION_LOG_BYTES:
        raise QualificationContractError(
            f"qualification log exceeds the evidence byte limit: {name}"
        )
    payload = path.read_bytes()
    if len(payload) != size:
        raise QualificationContractError(
            f"qualification log changed while being read: {name}"
        )
    marker = marker_for(name)
    # splitlines() accepts either Unix or CRLF input while still rejecting any
    # bytes after the marker.  Exact byte equality rejects whitespace/suffixes.
    lines = payload.splitlines()
    if not lines or lines[-1] != marker or lines.count(marker) != 1:
        raise QualificationContractError(
            f"qualification log lacks one unique exact terminal PASS marker: {name}"
        )
    if name == STEP_FINITE_FAILURE_LOG_NAME:
        validate_step_finite_failure_payload(payload)
    return {
        "path": name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "terminal_marker": marker.decode("ascii"),
    }


def validate_step_finite_failure_payload(payload: bytes) -> None:
    """Recheck the rank-complete expected-failure attestation at seal/replay."""
    lines = payload.splitlines()
    name = STEP_FINITE_FAILURE_LOG_NAME.encode("ascii")
    diagnostic = STEP_FINITE_FAILURE_DIAGNOSTIC.encode("ascii")
    for rank in (0, 1):
        marker = (
            f"gpmeep-finite-abort-probe:rank={rank},"
            "stage=before-failing-step"
        ).encode("ascii")
        if lines.count(marker) != 1:
            raise QualificationContractError(
                "finite expected-failure evidence does not contain exactly "
                f"one failing-step marker for rank {rank}"
            )
    if lines.count(b"meep: " + diagnostic) != 2:
        raise QualificationContractError(
            "finite expected-failure evidence does not contain exactly two "
            "rank-local NaN/Inf diagnostics"
        )
    if any(
        line.startswith(
            b"FAIL: fields::step returned after rank-local device NaN"
        )
        for line in lines
    ):
        raise QualificationContractError(
            "finite expected-failure evidence contains a successful step return"
        )

    prefix = b"gpmeep-expected-mpi-failure:" + name + b":status="
    attestations = [line for line in lines if line.startswith(prefix)]
    if len(attestations) != 1:
        raise QualificationContractError(
            "finite expected-failure evidence lacks one unique MPI-failure "
            "attestation"
        )
    raw_status, separator, attested_diagnostic = attestations[0][
        len(prefix) :
    ].partition(b":diagnostic=")
    try:
        status = int(raw_status.decode("ascii"))
    except (UnicodeError, ValueError) as exc:
        raise QualificationContractError(
            "finite expected-failure status is not an integer"
        ) from exc
    if (
        not separator
        or attested_diagnostic != diagnostic
        or not expected_failure_status_is_acceptable(status)
    ):
        raise QualificationContractError(
            "finite expected-failure attestation has an unacceptable status "
            "or diagnostic"
        )


def validate_logs(
    log_dir: pathlib.Path, names: Iterable[str]
) -> dict[str, dict[str, Any]]:
    if not log_dir.is_dir() or log_dir.is_symlink():
        raise QualificationContractError(
            f"qualification log directory is absent or unsafe: {log_dir}"
        )
    return {name: validate_log(log_dir / name, name) for name in sorted(names)}


def _validate_materialization_log(
    path: pathlib.Path, name: str
) -> dict[str, Any]:
    if path.name != name or not path.is_file() or path.is_symlink():
        raise QualificationContractError(
            f"required materialization log is absent or unsafe: {name}"
        )
    size = path.stat().st_size
    if size > MAX_QUALIFICATION_LOG_BYTES:
        raise QualificationContractError(
            f"materialization log exceeds the evidence byte limit: {name}"
        )
    payload = path.read_bytes()
    marker = f"gpmeep-libtool-materialization:{name}:PASS".encode("ascii")
    lines = payload.splitlines()
    if len(payload) != size or not lines or lines[-1] != marker:
        raise QualificationContractError(
            f"materialization log lacks its exact terminal marker: {name}"
        )
    if lines.count(marker) != 1:
        raise QualificationContractError(
            f"materialization log has a duplicate terminal marker: {name}"
        )
    return {
        "path": name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "terminal_marker": marker.decode("ascii"),
    }


def validate_no_uncontracted_pass_logs(
    log_dir: pathlib.Path, *, include_audit: bool = True
) -> dict[str, dict[str, Any]]:
    """Require the exact closed qualification-directory file set.

    The historical name is retained for callers, but markerless files are no
    longer ignored.  Before the seal creates its audit log, ``include_audit``
    is false; every replay and final receipt uses the complete 111-file set.
    """
    if not log_dir.is_dir() or log_dir.is_symlink():
        raise QualificationContractError(
            f"qualification log directory is absent or unsafe: {log_dir}"
        )
    required = set(REQUIRED_QUALIFICATION_LOGS) | set(MATERIALIZATION_LOGS)
    if include_audit:
        required.add(AUDIT_LOG_NAME)
    entries = list(log_dir.iterdir())
    actual = {path.name for path in entries}
    if len(actual) != len(entries) or actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        raise QualificationContractError(
            "qualification log directory is not the exact contracted set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for path in entries:
        if not path.is_file() or path.is_symlink():
            raise QualificationContractError(
                f"qualification log entry is not a safe regular file: {path.name}"
            )
    return {
        name: _validate_materialization_log(log_dir / name, name)
        for name in MATERIALIZATION_LOGS
    }


def _validate_attested_artifact(
    actual: Any,
    expected: Any,
    repo: pathlib.Path,
    *,
    context: str,
) -> None:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise QualificationContractError(
            f"Python runtime provenance binding is absent: {context}"
        )
    if not isinstance(actual.get("path"), str) or not isinstance(
        expected.get("path"), str
    ):
        raise QualificationContractError(
            f"Python runtime provenance path is invalid: {context}"
        )
    if (
        _resolve_recorded_path(actual["path"], repo).resolve()
        != _resolve_recorded_path(expected["path"], repo).resolve()
        or actual.get("size_bytes") != expected.get("size_bytes")
        or actual.get("sha256") != expected.get("sha256")
    ):
        raise QualificationContractError(
            f"Python runtime loaded artifact is not receipt-bound: {context}"
        )


def validate_python_runtime_logs(
    log_dir: pathlib.Path,
    artifacts: dict[str, Any],
    repo: pathlib.Path,
) -> None:
    for log_name, (extension_name, libmeep_name) in (
        PYTHON_RUNTIME_LOG_BINDINGS.items()
    ):
        try:
            lines = (log_dir / log_name).read_text(
                encoding="utf-8", errors="strict"
            ).splitlines()
        except (OSError, UnicodeError) as exc:
            raise QualificationContractError(
                f"cannot read Python runtime provenance: {log_name}: {exc}"
            ) from exc
        if len(lines) < 2 or lines[-1] != marker_for(log_name).decode("ascii"):
            raise QualificationContractError(
                f"Python runtime provenance log has unexpected framing: {log_name}"
            )
        try:
            attestation = json.loads(lines[-2])
        except json.JSONDecodeError as exc:
            raise QualificationContractError(
                f"Python runtime provenance is not JSON: {log_name}"
            ) from exc
        if not isinstance(attestation, dict):
            raise QualificationContractError(
                f"Python runtime provenance is not an object: {log_name}"
            )
        for attested_name, artifact_name in (
            ("python_extension", extension_name), ("libmeep", libmeep_name)
        ):
            actual = attestation.get(attested_name)
            expected = artifacts.get(artifact_name)
            _validate_attested_artifact(
                actual, expected, repo, context=f"{log_name}:{artifact_name}"
            )


def validate_installed_lazy_api_qualification_log(
    log_dir: pathlib.Path,
    artifacts: dict[str, Any],
    repo: pathlib.Path,
) -> None:
    path = log_dir / INSTALLED_LAZY_API_LOG_NAME
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
        value = json.loads(lines[-2]) if len(lines) >= 2 else None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationContractError(
            f"cannot read installed lazy API qualification: {exc}"
        ) from exc
    expected_top_level = {
        "schema_version",
        "qualification",
        "log_name",
        "deferred_module_roots",
        "initial_modules",
        "core_surface",
        "discoverable_surface",
        "module_aliases",
        "wildcard_surface",
        "adjoint_wildcard_names",
        "plot2d_axes_annotation_is_matplotlib_axes",
        "tests",
        "python_extension",
        "libmeep",
    }
    if (
        not lines
        or lines[-1] != marker_for(INSTALLED_LAZY_API_LOG_NAME).decode("ascii")
        or not isinstance(value, dict)
        or set(value) != expected_top_level
        or value.get("schema_version") != 1
        or value.get("qualification") != INSTALLED_LAZY_API_QUALIFICATION_NAME
        or value.get("log_name") != INSTALLED_LAZY_API_LOG_NAME
        or value.get("deferred_module_roots") != ["jax", "matplotlib", "scipy"]
        or value.get("initial_modules")
        != {"jax": False, "matplotlib": False, "scipy": False}
        or not isinstance(value.get("core_surface"), dict)
        or not value["core_surface"]
        or any(item is not True for item in value["core_surface"].values())
        or not isinstance(value.get("discoverable_surface"), dict)
        or not value["discoverable_surface"]
        or any(item is not True for item in value["discoverable_surface"].values())
        or not isinstance(value.get("module_aliases"), dict)
        or not value["module_aliases"]
        or any(
            not isinstance(item, str) or not item
            for item in value["module_aliases"].values()
        )
        or not isinstance(value.get("wildcard_surface"), dict)
        or not value["wildcard_surface"]
        or any(item is not True for item in value["wildcard_surface"].values())
        or not isinstance(value.get("adjoint_wildcard_names"), list)
        or value["adjoint_wildcard_names"]
        != sorted(set(value["adjoint_wildcard_names"]))
        or len(value["adjoint_wildcard_names"])
        != HISTORICAL_ADJOINT_WILDCARD_COUNT
        or _canonical_sha256(value["adjoint_wildcard_names"])
        != HISTORICAL_ADJOINT_WILDCARD_SHA256
        or value.get("plot2d_axes_annotation_is_matplotlib_axes") is not True
    ):
        raise QualificationContractError(
            "installed lazy API qualification schema is invalid"
        )
    tests = value.get("tests")
    if not isinstance(tests, list) or len(tests) != len(
        INSTALLED_LAZY_API_TEST_SPECS
    ):
        raise QualificationContractError(
            "installed lazy API qualification test inventory is incomplete"
        )
    for record, (relative, selectors, tests_run) in zip(
        tests, INSTALLED_LAZY_API_TEST_SPECS
    ):
        expected_source = _file_record((repo / relative).resolve(), repo)
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "source",
                "selectors",
                "tests_run",
                "pythonpath",
                "returncode",
                "elapsed_seconds",
                "stdout_sha256",
                "stderr_sha256",
            }
            or record.get("source") != expected_source
            or record.get("selectors") != list(selectors)
            or record.get("tests_run") != tests_run
            or record.get("pythonpath")
            != (
                str((repo / "python" / "tests").resolve())
                + os.pathsep
                + str(
                    _resolve_recorded_path(
                        str(
                            artifacts["installed_python_extension"]["path"]
                        ),
                        repo,
                    ).parent.parent
                )
            )
            or record.get("returncode") != 0
            or isinstance(record.get("elapsed_seconds"), bool)
            or not isinstance(record.get("elapsed_seconds"), (int, float))
            or record["elapsed_seconds"] <= 0
            or any(
                not isinstance(record.get(name), str)
                or re.fullmatch(r"[0-9a-f]{64}", record[name]) is None
                for name in ("stdout_sha256", "stderr_sha256")
            )
        ):
            raise QualificationContractError(
                f"installed lazy API test record is invalid: {relative}"
            )
    for attested_name, artifact_name in (
        ("python_extension", "installed_python_extension"),
        ("libmeep", "installed_libmeep"),
    ):
        _validate_attested_artifact(
            value.get(attested_name),
            artifacts.get(artifact_name),
            repo,
            context=f"{INSTALLED_LAZY_API_LOG_NAME}:{artifact_name}",
        )


def validate_installed_mpb_qualification_log(
    log_dir: pathlib.Path,
    artifacts: dict[str, Any],
    repo: pathlib.Path,
) -> None:
    try:
        lines = (log_dir / INSTALLED_MPB_LOG_NAME).read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
        value = json.loads(lines[-2]) if len(lines) >= 2 else None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationContractError(
            f"cannot read installed MPB qualification: {exc}"
        ) from exc
    expected_keys = {
        "schema_version",
        "qualification",
        "log_name",
        "world_size",
        "single_precision_meep",
        "mpb_extension",
        "libpympb",
        "python_extension",
        "libmeep",
        "frequencies",
        "repeat_frequencies",
        "maximum_repeat_error",
    }
    frequencies = value.get("frequencies") if isinstance(value, dict) else None
    repeated = value.get("repeat_frequencies") if isinstance(value, dict) else None
    repeat_error = (
        max(abs(left - right) for left, right in zip(frequencies, repeated))
        if isinstance(frequencies, list)
        and isinstance(repeated, list)
        and len(frequencies) == len(repeated) == 2
        and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in (*frequencies, *repeated)
        )
        else math.inf
    )
    if (
        not lines
        or lines[-1] != marker_for(INSTALLED_MPB_LOG_NAME).decode("ascii")
        or not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("qualification") != INSTALLED_MPB_QUALIFICATION_NAME
        or value.get("log_name") != INSTALLED_MPB_LOG_NAME
        or value.get("world_size") != 1
        or value.get("single_precision_meep") is not True
        or not isinstance(frequencies, list)
        or len(frequencies) != 2
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            and item > 0
            for item in frequencies
        )
        or frequencies != sorted(frequencies)
        or not isinstance(repeated, list)
        or len(repeated) != 2
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            and item > 0
            for item in repeated
        )
        or not isinstance(value.get("maximum_repeat_error"), (int, float))
        or isinstance(value.get("maximum_repeat_error"), bool)
        or not math.isfinite(value["maximum_repeat_error"])
        or value["maximum_repeat_error"] < 0
        or abs(value["maximum_repeat_error"] - repeat_error) > 1e-15
        or value["maximum_repeat_error"] > 1e-7
    ):
        raise QualificationContractError("installed MPB qualification is invalid")
    for attested_name, artifact_name in (
        ("mpb_extension", "installed_mpb_extension"),
        ("libpympb", "installed_libpympb"),
        ("python_extension", "installed_python_extension"),
        ("libmeep", "installed_libmeep"),
    ):
        _validate_attested_artifact(
            value.get(attested_name),
            artifacts.get(artifact_name),
            repo,
            context=f"{INSTALLED_MPB_LOG_NAME}:{artifact_name}",
        )


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def validate_installed_cuda_qualification_logs(
    log_dir: pathlib.Path,
    artifacts: dict[str, Any],
    repo: pathlib.Path,
) -> None:
    expected_extension = artifacts.get("installed_python_extension")
    expected_libmeep = artifacts.get("installed_libmeep")
    for log_name, spec in INSTALLED_CUDA_QUALIFICATION_SPECS.items():
        try:
            lines = (log_dir / log_name).read_text(
                encoding="utf-8", errors="strict"
            ).splitlines()
            value = json.loads(lines[-2]) if len(lines) >= 2 else None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise QualificationContractError(
                f"cannot read installed CUDA qualification: {log_name}: {exc}"
            ) from exc
        if (
            not lines
            or lines[-1] != marker_for(log_name).decode("ascii")
            or not isinstance(value, dict)
        ):
            raise QualificationContractError(
                f"installed CUDA qualification has invalid framing: {log_name}"
            )
        required_top_level = {
            "schema_version",
            "qualification",
            "log_name",
            "mode",
            "expected_world_size",
            "environment",
            "suite_file",
            "python_extension",
            "libmeep",
            "rank_records",
        }
        if set(value) != required_top_level or (
            value.get("schema_version") != 1
            or value.get("qualification") != INSTALLED_CUDA_QUALIFICATION_NAME
            or value.get("log_name") != log_name
            or value.get("mode") != spec["mode"]
            or value.get("expected_world_size") != spec["world_size"]
            or value.get("environment") != spec["environment"]
        ):
            raise QualificationContractError(
                f"installed CUDA qualification envelope is invalid: {log_name}"
            )
        _validate_attested_artifact(
            value.get("python_extension"),
            expected_extension,
            repo,
            context=f"{log_name}:top-level-extension",
        )
        _validate_attested_artifact(
            value.get("libmeep"),
            expected_libmeep,
            repo,
            context=f"{log_name}:top-level-libmeep",
        )
        suite_file = value.get("suite_file")
        expected_suite_path = (repo / str(spec["test_file"])).resolve()
        if (
            not isinstance(suite_file, dict)
            or set(suite_file) != {"path", "size_bytes", "sha256"}
            or not isinstance(suite_file.get("path"), str)
            or _resolve_recorded_path(suite_file["path"], repo).resolve()
            != expected_suite_path
            or suite_file.get("size_bytes") != expected_suite_path.stat().st_size
            or suite_file.get("sha256") != _sha256_file(expected_suite_path)
        ):
            raise QualificationContractError(
                f"installed CUDA test suite is not source-bound: {log_name}"
            )
        rank_records = value.get("rank_records")
        world_size = int(spec["world_size"])
        if not isinstance(rank_records, list) or len(rank_records) != world_size:
            raise QualificationContractError(
                f"installed CUDA qualification has incomplete ranks: {log_name}"
            )
        by_rank: dict[int, dict[str, Any]] = {}
        identifiers: set[str] = set()
        cuda_source_calls = 0
        for record in rank_records:
            if not isinstance(record, dict) or set(record) != {
                "world_rank",
                "world_size",
                "suite",
                "python_extension",
                "libmeep",
                "probe",
            }:
                raise QualificationContractError(
                    f"installed CUDA rank record is invalid: {log_name}"
                )
            rank = record.get("world_rank")
            if (
                not _strict_int(rank)
                or rank >= world_size
                or rank in by_rank
                or record.get("world_size") != world_size
            ):
                raise QualificationContractError(
                    f"installed CUDA rank identity is invalid: {log_name}"
                )
            by_rank[rank] = record
            suite = record.get("suite")
            if not isinstance(suite, dict) or set(suite) != {
                "tests_run",
                "failures",
                "errors",
                "skipped",
                "expected_failures",
                "unexpected_successes",
                "successful",
            }:
                raise QualificationContractError(
                    f"installed CUDA unittest result is invalid: {log_name}"
                )
            if (
                suite.get("tests_run") != spec["tests_run"]
                or any(
                    suite.get(name) != 0
                    for name in (
                        "failures",
                        "errors",
                        "skipped",
                        "expected_failures",
                        "unexpected_successes",
                    )
                )
                or suite.get("successful") is not True
            ):
                raise QualificationContractError(
                    f"installed CUDA unittest suite did not pass exactly: {log_name}"
                )
            _validate_attested_artifact(
                record.get("python_extension"),
                expected_extension,
                repo,
                context=f"{log_name}:rank-{rank}-extension",
            )
            _validate_attested_artifact(
                record.get("libmeep"),
                expected_libmeep,
                repo,
                context=f"{log_name}:rank-{rank}-libmeep",
            )
            probe = record.get("probe")
            if not isinstance(probe, dict) or set(probe) != {
                "requested_backend",
                "active_backend",
                "selected_device",
                "selected_device_identifier",
                "cuda_execution_selected",
                "execution_diagnostic",
                "statistics",
            }:
                raise QualificationContractError(
                    f"installed CUDA execution probe is invalid: {log_name}"
                )
            identifier = probe.get("selected_device_identifier")
            statistics = probe.get("statistics")
            if (
                probe.get("requested_backend") != "cuda"
                or probe.get("active_backend") != "cuda"
                or probe.get("cuda_execution_selected") is not True
                or not _strict_int(probe.get("selected_device"))
                or not isinstance(identifier, str)
                or not identifier
                or identifier in identifiers
                or not isinstance(probe.get("execution_diagnostic"), str)
                or "CUDA" not in probe["execution_diagnostic"]
                or not isinstance(statistics, dict)
            ):
                raise QualificationContractError(
                    f"installed CUDA execution was not rank-local CUDA: {log_name}"
                )
            identifiers.add(identifier)
            dispatch = statistics.get("dispatch")
            updates = statistics.get("field_updates")
            sources = statistics.get("sources")
            if not all(
                isinstance(group, dict)
                for group in (dispatch, updates, sources)
            ):
                raise QualificationContractError(
                    f"installed CUDA statistics are incomplete: {log_name}"
                )
            assert isinstance(dispatch, dict)
            assert isinstance(updates, dict)
            assert isinstance(sources, dict)
            if (
                not _strict_int(dispatch.get("cuda_curl_calls"), minimum=1)
                or dispatch.get("cpu_curl_calls") != 0
                or not _strict_int(updates.get("cuda_update_eh_calls"), minimum=1)
                or updates.get("cpu_update_eh_calls") != 0
                or sources.get("cpu_source_calls") != 0
                or not _strict_int(sources.get("cuda_source_calls"))
            ):
                raise QualificationContractError(
                    f"installed CUDA execution counters show CPU fallback: {log_name}"
                )
            cuda_source_calls += sources["cuda_source_calls"]
        if set(by_rank) != set(range(world_size)) or len(identifiers) != world_size:
            raise QualificationContractError(
                f"installed CUDA ranks do not map to distinct devices: {log_name}"
            )
        if cuda_source_calls <= 0:
            raise QualificationContractError(
                f"installed CUDA execution did not dispatch a source: {log_name}"
            )
        if spec["mode"] == "adjoint":
            prefix = "gpmeep-fd-direction:"
            fd_records = []
            for line in lines[:-2]:
                if line.startswith(prefix):
                    try:
                        fd_records.append(json.loads(line[len(prefix) :]))
                    except json.JSONDecodeError as exc:
                        raise QualificationContractError(
                            f"installed CUDA FD record is invalid: {log_name}"
                        ) from exc
            expected_names = {
                "cosine",
                "quasiperiodic-sine",
                "component-6",
                "component-12",
                "component-18",
            }
            if (
                len(fd_records) != 5
                or {record.get("name") for record in fd_records} != expected_names
                or any(record.get("pass") is not True for record in fd_records)
            ):
                raise QualificationContractError(
                    f"installed CUDA adjoint FD evidence is incomplete: {log_name}"
                )


def validate_mutation_sentinel_log(
    log_dir: pathlib.Path, identity: dict[str, Any]
) -> None:
    name = SENTINEL_LOG_NAME
    try:
        lines = (log_dir / name).read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
        value = json.loads(lines[-2]) if len(lines) >= 2 else None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationContractError(
            f"cannot read mutation sentinel attestation: {exc}"
        ) from exc
    if (
        not lines
        or lines[-1] != marker_for(name).decode("ascii")
        or not isinstance(value, dict)
        or value.get("identity_sha256") != _canonical_sha256(identity)
        or not isinstance(value.get("watch_count"), int)
        or value["watch_count"] <= 0
    ):
        raise QualificationContractError(
            "mutation sentinel attestation is not identity-bound"
        )


def validate_direct_elf_logs(
    log_dir: pathlib.Path,
    artifacts: dict[str, Any],
    repo: pathlib.Path,
) -> None:
    libmeep = artifacts.get("libmeep")
    if not isinstance(libmeep, dict) or not isinstance(libmeep.get("path"), str):
        raise QualificationContractError("direct ELF libmeep binding is absent")
    for log_name, artifact_name in DIRECT_ELF_LOG_BINDINGS.items():
        artifact = artifacts.get(artifact_name)
        if not isinstance(artifact, dict) or not isinstance(
            artifact.get("path"), str
        ):
            raise QualificationContractError(
                f"direct ELF artifact binding is absent: {artifact_name}"
            )
        lines = (log_dir / log_name).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if (
            len(lines) < 5
            or not lines[0].startswith("executable=")
            or not lines[1].startswith("loaded_libmeep=")
        ):
            raise QualificationContractError(
                f"direct ELF provenance framing is invalid: {log_name}"
            )
        executable = pathlib.Path(lines[0].split("=", 1)[1]).resolve()
        loaded_libmeep = pathlib.Path(lines[1].split("=", 1)[1]).resolve()
        expected_executable = _resolve_recorded_path(artifact["path"], repo).resolve()
        expected_libmeep = _resolve_recorded_path(libmeep["path"], repo).resolve()
        if executable != expected_executable or loaded_libmeep != expected_libmeep:
            raise QualificationContractError(
                f"direct ELF runtime paths are not receipt-bound: {log_name}"
            )
        sha256_records: dict[pathlib.Path, str] = {}
        for line in lines[2:]:
            digest, separator, raw_path = line.partition("  ")
            if separator and len(digest) == 64 and all(
                character in "0123456789abcdef" for character in digest
            ):
                sha256_records[pathlib.Path(raw_path).resolve()] = digest
        if (
            sha256_records.get(executable) != artifact.get("sha256")
            or sha256_records.get(loaded_libmeep) != libmeep.get("sha256")
        ):
            raise QualificationContractError(
                f"direct ELF runtime hashes are not receipt-bound: {log_name}"
            )


def expected_failure_status_is_acceptable(status: int) -> bool:
    """Accept a program diagnostic exit, never timeout/exec/signal status."""
    return 1 <= status <= 123


_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_ONLYDIR = 0x01000000
_IN_DONT_FOLLOW = 0x02000000
_IN_EXCL_UNLINK = 0x04000000
_MUTATION_MASK = (
    _IN_MODIFY
    | _IN_ATTRIB
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
)
_EVENT_STRUCT = struct.Struct("iIII")


def _identity_artifact_paths(
    identity: dict[str, Any], repo: pathlib.Path
) -> dict[str, pathlib.Path]:
    artifacts = identity.get("artifacts")
    if not isinstance(artifacts, dict):
        raise QualificationContractError("sentinel identity artifacts are invalid")
    result: dict[str, pathlib.Path] = {}
    for name in REQUIRED_ARTIFACT_BINDINGS:
        record = artifacts.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise QualificationContractError(
                f"sentinel identity artifact is absent: {name}"
            )
        result[name] = _resolve_recorded_path(record["path"], repo).resolve()
    return result


def _identity_installed_prefix_path(
    identity: dict[str, Any], repo: pathlib.Path
) -> pathlib.Path:
    manifest = identity.get("installed_prefix")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("root"), str):
        raise QualificationContractError(
            "sentinel installed-prefix identity is invalid"
        )
    path = _resolve_recorded_path(manifest["root"], repo).resolve(strict=True)
    if not path.is_dir():
        raise QualificationContractError(
            "sentinel installed-prefix directory is absent"
        )
    return path


def _sentinel_identity_matches(
    identity: dict[str, Any],
    repo: pathlib.Path,
    prefix: pathlib.Path,
    artifact_paths: dict[str, pathlib.Path],
    installed_prefix: pathlib.Path,
    immutable_empty_directories: dict[str, pathlib.Path],
) -> bool:
    try:
        return create_identity_snapshot(
            repo,
            prefix,
            artifact_paths,
            installed_prefix=installed_prefix,
            immutable_empty_directories=immutable_empty_directories,
            require_contract_bindings=True,
        ) == identity
    except (OSError, RuntimeError, QualificationContractError):
        return False


def run_mutation_sentinel(
    identity_path: pathlib.Path,
    repo: pathlib.Path,
    prefix: pathlib.Path,
    ready_path: pathlib.Path,
) -> None:
    """Detect even transient writes/renames during the qualification epoch."""
    identity = _load_json(identity_path)
    artifact_paths = _identity_artifact_paths(identity, repo)
    installed_prefix = _identity_installed_prefix_path(identity, repo)
    immutable_empty_directories = _identity_immutable_empty_directory_paths(
        identity,
        repo,
        require_contract_bindings=True,
        expected_directory_parent=identity_path.parent,
    )
    libc = ctypes.CDLL(None, use_errno=True)
    init = libc.inotify_init1
    init.argtypes = [ctypes.c_int]
    init.restype = ctypes.c_int
    add_watch = libc.inotify_add_watch
    add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    add_watch.restype = ctypes.c_int
    descriptor = init(os.O_CLOEXEC | os.O_NONBLOCK)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise QualificationContractError(
            f"cannot initialize inotify sentinel: {os.strerror(error)}"
        )
    rules: dict[int, list[tuple[str, pathlib.Path, frozenset[str]]]] = {}

    def watch(
        path: pathlib.Path,
        kind: str,
        names: Iterable[str] = (),
        *,
        directory: bool = False,
    ) -> None:
        mask = _MUTATION_MASK | _IN_EXCL_UNLINK
        if directory:
            mask |= _IN_ONLYDIR
        else:
            mask |= _IN_DONT_FOLLOW
        wd = add_watch(descriptor, os.fsencode(path), mask)
        if wd < 0:
            error = ctypes.get_errno()
            raise QualificationContractError(
                f"cannot watch qualification identity path {path}: "
                f"{os.strerror(error)}"
            )
        rule = (kind, path, frozenset(names))
        if rule not in rules.setdefault(wd, []):
            rules[wd].append(rule)

    source = identity.get("source")
    source_files = source.get("files") if isinstance(source, dict) else None
    if not isinstance(source_files, list):
        os.close(descriptor)
        raise QualificationContractError("sentinel source identity is invalid")
    source_directories: set[pathlib.Path] = set()
    try:
        for record in source_files:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise QualificationContractError("sentinel source record is invalid")
            path = (repo / record["path"]).resolve()
            watch(path, "file")
            parent = path.parent
            while parent == repo or repo in parent.parents:
                source_directories.add(parent)
                if parent == repo:
                    break
                parent = parent.parent
        for directory in source_directories:
            watch(directory, "source-dir", directory=True)
        artifact_parents: dict[pathlib.Path, set[str]] = {}
        for path in set(artifact_paths.values()):
            watch(path, "file")
            artifact_parents.setdefault(path.parent, set()).add(path.name)
        for parent, names in artifact_parents.items():
            watch(parent, "named-parent", names, directory=True)
        immutable_empty_parents: dict[pathlib.Path, set[str]] = {}
        for path in set(immutable_empty_directories.values()):
            watch(path, "immutable-empty-directory", directory=True)
            immutable_empty_parents.setdefault(path.parent, set()).add(path.name)
        for parent, names in immutable_empty_parents.items():
            watch(parent, "immutable-empty-parent", names, directory=True)
        for directory in sorted(
            (
                path
                for path in prefix.rglob("*")
                if path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: str(path),
        ):
            if "__pycache__" not in directory.parts:
                watch(directory, "prefix", directory=True)
        watch(prefix, "prefix", directory=True)
        if installed_prefix != prefix:
            for directory in sorted(
                (
                    path
                    for path in installed_prefix.rglob("*")
                    if path.is_dir() and not path.is_symlink()
                ),
                key=lambda path: str(path),
            ):
                watch(directory, "installed-prefix", directory=True)
            watch(installed_prefix, "installed-prefix", directory=True)

        stopped = False

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stopped
            stopped = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        def mutation_from_events() -> str | None:
            while True:
                try:
                    payload = os.read(descriptor, 1024 * 1024)
                except BlockingIOError:
                    return None
                if not payload:
                    return None
                offset = 0
                while offset < len(payload):
                    wd, mask, _cookie, length = _EVENT_STRUCT.unpack_from(
                        payload, offset
                    )
                    offset += _EVENT_STRUCT.size
                    raw_name = payload[offset : offset + length].split(b"\0", 1)[0]
                    offset += length
                    if mask & _IN_Q_OVERFLOW:
                        return "inotify event queue overflowed"
                    for kind, base, names in rules.get(wd, ()):
                        name = os.fsdecode(raw_name)
                        if kind in ("prefix", "installed-prefix", "source-dir"):
                            target = base / name if name else base
                            if kind == "prefix" and (
                                target.suffix == ".pyc"
                                or "__pycache__" in target.parts
                            ):
                                continue
                            return f"{kind} mutation: {target} mask=0x{mask:x}"
                        if (
                            kind in ("named-parent", "immutable-empty-parent")
                            and name
                            and name not in names
                        ):
                            continue
                        return f"identity mutation: {base / name} mask=0x{mask:x}"

        if not _sentinel_identity_matches(
            identity,
            repo,
            prefix,
            artifact_paths,
            installed_prefix,
            immutable_empty_directories,
        ):
            raise QualificationContractError(
                "identity changed while mutation sentinel was starting"
            )
        startup_mutation = mutation_from_events()
        if startup_mutation is not None:
            raise QualificationContractError(startup_mutation)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor_ready, temporary_name = tempfile.mkstemp(
            prefix=f".{ready_path.name}.", suffix=".tmp", dir=ready_path.parent
        )
        with os.fdopen(descriptor_ready, "w", encoding="utf-8") as target:
            target.write("READY\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, ready_path)

        while not stopped:
            readable, _, _ = select.select([descriptor], [], [], 1.0)
            if readable:
                mutation = mutation_from_events()
                if mutation is not None:
                    raise QualificationContractError(mutation)
        if not _sentinel_identity_matches(
            identity,
            repo,
            prefix,
            artifact_paths,
            installed_prefix,
            immutable_empty_directories,
        ):
            raise QualificationContractError(
                "identity changed while mutation sentinel was stopping"
            )
        final_mutation = mutation_from_events()
        if final_mutation is not None:
            raise QualificationContractError(final_mutation)
        attestation = {
            "identity_sha256": _canonical_sha256(identity),
            "watch_count": len(rules),
        }
        print(json.dumps(attestation, sort_keys=True, separators=(",", ":")))
        print(marker_for(SENTINEL_LOG_NAME).decode("ascii"))
    finally:
        os.close(descriptor)


def python_runtime_attestation(
    expected_extension: pathlib.Path,
    expected_libmeep: pathlib.Path,
) -> dict[str, Any]:
    import importlib

    extension = pathlib.Path(
        str(importlib.import_module("meep._meep").__file__)
    ).resolve()
    expected_extension = expected_extension.resolve()
    expected_libmeep = expected_libmeep.resolve()
    if extension != expected_extension:
        raise QualificationContractError(
            f"Python loaded unexpected _meep: {extension}"
        )
    try:
        mappings = pathlib.Path("/proc/self/maps").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as exc:
        raise QualificationContractError(
            f"cannot inspect Python runtime mappings: {exc}"
        ) from exc
    loaded_libmeep = {
        pathlib.Path(line.split()[-1]).resolve()
        for line in mappings
        if len(line.split()) >= 6
        and pathlib.Path(line.split()[-1]).name.startswith("libmeep.so")
    }
    if loaded_libmeep != {expected_libmeep}:
        raise QualificationContractError(
            "Python loaded unexpected libmeep mapping(s): "
            + ", ".join(str(path) for path in sorted(loaded_libmeep))
        )
    return {
        "python_extension": {
            "path": str(extension),
            "size_bytes": extension.stat().st_size,
            "sha256": _sha256_file(extension),
        },
        "libmeep": {
            "path": str(expected_libmeep),
            "size_bytes": expected_libmeep.stat().st_size,
            "sha256": _sha256_file(expected_libmeep),
        },
    }


def attest_python_runtime(
    expected_extension: pathlib.Path,
    expected_libmeep: pathlib.Path,
    log_name: str,
) -> None:
    if log_name not in REQUIRED_QUALIFICATION_LOGS:
        raise QualificationContractError(
            f"Python runtime provenance log is not contract-listed: {log_name}"
        )
    value = python_runtime_attestation(expected_extension, expected_libmeep)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    print(marker_for(log_name).decode("ascii"))
    sys.stdout.flush()
    sys.stderr.flush()


def _load_provenance_from_source() -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().with_name("gpmeep_provenance.py")
    module = types.ModuleType("gpmeep_provenance_for_qualification")
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def create_identity_snapshot(
    repo: pathlib.Path,
    prefix: pathlib.Path,
    artifacts: dict[str, pathlib.Path],
    installed_prefix: pathlib.Path | None = None,
    immutable_empty_directories: dict[str, pathlib.Path] | None = None,
    *,
    require_contract_bindings: bool = False,
) -> dict[str, Any]:
    if tuple(sorted(artifacts)) != REQUIRED_ARTIFACT_BINDINGS:
        raise QualificationContractError(
            "qualification identity snapshot has the wrong artifact binding set"
        )
    provenance = _load_provenance_from_source()
    if installed_prefix is None:
        installed_prefix = prefix
    if immutable_empty_directories is None:
        immutable_empty_directories = {}
    if not isinstance(immutable_empty_directories, dict) or any(
        not isinstance(name, str) or not name
        for name in immutable_empty_directories
    ):
        raise QualificationContractError(
            "qualification immutable empty directory names are invalid"
        )
    value = {
        "schema_version": SCHEMA_VERSION,
        "qualification_contract": CONTRACT_NAME,
        "source": provenance.source_snapshot(repo),
        "installed_environment": provenance.tree_manifest(
            prefix, repo, excluded_suffixes=(".pyc",)
        ),
        "installed_prefix": provenance.tree_manifest(installed_prefix, repo),
        "artifacts": {
            name: provenance.file_record(path, repo)
            for name, path in sorted(artifacts.items())
        },
        IMMUTABLE_EMPTY_DIRECTORIES_KEY: {
            name: _immutable_empty_directory_record(path, repo)
            for name, path in sorted(immutable_empty_directories.items())
        },
    }
    if require_contract_bindings:
        _identity_immutable_empty_directory_paths(
            value, repo, require_contract_bindings=True
        )
    return value


def seal_contract(
    repo: pathlib.Path,
    log_dir: pathlib.Path,
    before_path: pathlib.Path,
    after_path: pathlib.Path,
    output: pathlib.Path,
) -> None:
    before = _load_json(before_path)
    after = _load_json(after_path)
    if before != after:
        raise QualificationContractError(
            "qualification source/artifact/prefix identities changed during testing"
        )
    if (
        before.get("schema_version") != SCHEMA_VERSION
        or before.get("qualification_contract") != CONTRACT_NAME
        or tuple(sorted(before.get("artifacts", {}))) != REQUIRED_ARTIFACT_BINDINGS
    ):
        raise QualificationContractError("qualification identity snapshot is invalid")
    _identity_immutable_empty_directory_paths(
        before,
        repo,
        require_contract_bindings=True,
        expected_directory_parent=before_path.parent,
    )
    _identity_immutable_empty_directory_paths(
        after,
        repo,
        require_contract_bindings=True,
        expected_directory_parent=after_path.parent,
    )
    validate_logs(log_dir, REQUIRED_QUALIFICATION_LOGS)
    validate_no_uncontracted_pass_logs(log_dir, include_audit=False)
    validate_mutation_sentinel_log(log_dir, before)
    validate_python_runtime_logs(log_dir, before["artifacts"], repo)
    validate_installed_lazy_api_qualification_log(
        log_dir, before["artifacts"], repo
    )
    validate_installed_mpb_qualification_log(
        log_dir, before["artifacts"], repo
    )
    validate_installed_cuda_qualification_logs(
        log_dir, before["artifacts"], repo
    )
    validate_direct_elf_logs(log_dir, before["artifacts"], repo)
    audit_path = log_dir / AUDIT_LOG_NAME
    audit_payload = "".join(
        f"verified={name}\n" for name in REQUIRED_QUALIFICATION_LOGS
    ) + marker_for(AUDIT_LOG_NAME).decode("ascii") + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{AUDIT_LOG_NAME}.", suffix=".tmp", dir=log_dir
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(audit_payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, audit_path)
    finally:
        temporary.unlink(missing_ok=True)
    logs = validate_logs(log_dir, ALL_REQUIRED_QUALIFICATION_LOGS)
    materialization_logs = validate_no_uncontracted_pass_logs(
        log_dir, include_audit=True
    )
    seal = {
        "schema_version": SCHEMA_VERSION,
        "qualification_contract": CONTRACT_NAME,
        "required_logs": list(ALL_REQUIRED_QUALIFICATION_LOGS),
        "logs": logs,
        "materialization_logs": materialization_logs,
        "identity_sha256": _canonical_sha256(before),
        "identity_before": _file_record(before_path, repo),
        "identity_after": _file_record(after_path, repo),
        "artifacts": before["artifacts"],
        "source": before["source"],
        "installed_environment": before["installed_environment"],
        "installed_prefix": before["installed_prefix"],
        IMMUTABLE_EMPTY_DIRECTORIES_KEY: before[
            IMMUTABLE_EMPTY_DIRECTORIES_KEY
        ],
    }
    _atomic_write_json(output, seal)


def _cpu_reference_exact_keys(
    value: Any, expected: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise QualificationContractError(
            f"CPU reference {label} keys are not exact: {observed}"
        )
    return value


def _cpu_reference_json(payload: str, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QualificationContractError(
                    f"CPU reference {label} has duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise QualificationContractError(
            f"CPU reference {label} has non-finite JSON token {value}"
        )

    try:
        value = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationContractError(
            f"CPU reference {label} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise QualificationContractError(f"CPU reference {label} is not an object")
    return value


def _cpu_reference_record_path(
    records: dict[str, Any], name: str, repo: pathlib.Path, label: str
) -> pathlib.Path:
    record = records.get(name)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise QualificationContractError(
            f"CPU reference {label}.{name} path record is invalid"
        )
    return _resolve_recorded_path(record["path"], repo).resolve(strict=True)


def _cpu_reference_runtime_identity(
    records: dict[str, Any], name: str, repo: pathlib.Path, label: str
) -> dict[str, Any]:
    record = records.get(name)
    path = _cpu_reference_record_path(records, name, repo, label)
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("size_bytes"), int)
        or isinstance(record.get("size_bytes"), bool)
        or record["size_bytes"] < 0
        or not isinstance(record.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
    ):
        raise QualificationContractError(
            f"CPU reference {label}.{name} identity is invalid"
        )
    return {
        "path": str(path),
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
    }


def _validate_cpu_reference_log(
    path: pathlib.Path,
    *,
    runtime_kind: str,
    mpi_size: int,
    expected_python: dict[str, Any],
    expected_extension: dict[str, Any],
    expected_libmeep: dict[str, Any],
) -> tuple[float, float, float]:
    if not path.is_file() or path.is_symlink():
        raise QualificationContractError(
            f"CPU reference qualification log is absent or unsafe: {path.name}"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise QualificationContractError(
            f"cannot read CPU reference qualification log {path.name}: {exc}"
        ) from exc
    if len(payload) > 16 * 1024 * 1024:
        raise QualificationContractError(
            f"CPU reference qualification log is oversized: {path.name}"
        )
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise QualificationContractError(
            f"CPU reference qualification log is not UTF-8: {path.name}"
        ) from exc
    markers = [line for line in lines if line.startswith(CPU_REFERENCE_MARKER)]
    if len(markers) != 1:
        raise QualificationContractError(
            f"CPU reference qualification log lacks one unique marker: {path.name}"
        )
    record = _cpu_reference_json(
        markers[0][len(CPU_REFERENCE_MARKER) :], path.name
    )
    _cpu_reference_exact_keys(
        record,
        {
            "schema",
            "mpi_size",
            "runtime_kind",
            "single_precision",
            "cuda_compiled",
            "ranks",
        },
        path.name,
    )
    if (
        not isinstance(record["mpi_size"], int)
        or isinstance(record["mpi_size"], bool)
        or not isinstance(record["runtime_kind"], str)
        or record["schema"] != CPU_REFERENCE_SCHEMA
        or record["mpi_size"] != mpi_size
        or record["runtime_kind"] != runtime_kind
        or record["single_precision"] is not False
        or record["cuda_compiled"] is not False
    ):
        raise QualificationContractError(
            f"CPU reference qualification identity is invalid: {path.name}"
        )
    ranks = record["ranks"]
    if not isinstance(ranks, list) or len(ranks) != mpi_size:
        raise QualificationContractError(
            f"CPU reference rank inventory is invalid: {path.name}"
        )
    expected_runtime = {
        "python": expected_python,
        "python_extension": expected_extension,
        "libmeep": expected_libmeep,
    }
    reference_observables: tuple[float, float, float] | None = None
    expected_statistics = {
        "curl_calls",
        "curl_points",
        "update_eh_calls",
        "update_eh_points",
        "source_calls",
        "source_points",
        "boundary_calls",
        "boundary_points",
        "dft_calls",
        "dft_points",
    }
    for expected_rank, rank_record in enumerate(ranks):
        rank_record = _cpu_reference_exact_keys(
            rank_record,
            {
                "rank",
                "energy",
                "dft_norm",
                "meep_time",
                "statistics",
                "runtime_artifacts",
            },
            f"{path.name} rank {expected_rank}",
        )
        if rank_record["rank"] != expected_rank or isinstance(
            rank_record["rank"], bool
        ):
            raise QualificationContractError(
                f"CPU reference rank ordering is invalid: {path.name}"
            )
        energy = rank_record["energy"]
        dft_norm = rank_record["dft_norm"]
        meep_time = rank_record["meep_time"]
        if (
            isinstance(energy, bool)
            or not isinstance(energy, (int, float))
            or not math.isfinite(energy)
            or energy <= 0
            or isinstance(dft_norm, bool)
            or not isinstance(dft_norm, (int, float))
            or not math.isfinite(dft_norm)
            or dft_norm <= 0
            or isinstance(meep_time, bool)
            or not isinstance(meep_time, (int, float))
            or not math.isfinite(meep_time)
            or not math.isclose(
                float(meep_time), 8.0, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise QualificationContractError(
                f"CPU reference physical observables are invalid: {path.name}"
            )
        observables = (float(energy), float(dft_norm), float(meep_time))
        if reference_observables is None:
            reference_observables = observables
        elif (
            not math.isclose(
                observables[0], reference_observables[0], rel_tol=1e-12, abs_tol=1e-14
            )
            or not math.isclose(
                observables[1], reference_observables[1], rel_tol=1e-12, abs_tol=1e-14
            )
            or not math.isclose(
                observables[2], reference_observables[2],
                rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise QualificationContractError(
                f"CPU reference ranks disagree on observables: {path.name}"
            )
        runtime = _cpu_reference_exact_keys(
            rank_record["runtime_artifacts"],
            set(expected_runtime),
            f"{path.name} rank {expected_rank} runtime",
        )
        if runtime != expected_runtime:
            raise QualificationContractError(
                f"CPU reference loaded runtime is not receipt-bound: {path.name}"
            )
        statistics = _cpu_reference_exact_keys(
            rank_record["statistics"],
            expected_statistics,
            f"{path.name} rank {expected_rank} statistics",
        )
        for name, value in statistics.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise QualificationContractError(
                    f"CPU reference statistic is invalid: {path.name} {name}"
                )
            if not name.startswith("boundary_") and value <= 0:
                raise QualificationContractError(
                    f"CPU reference has no positive work: {path.name} {name}"
                )
        if mpi_size > 1 and (
            statistics["boundary_calls"] <= 0
            or statistics["boundary_points"] <= 0
        ):
            raise QualificationContractError(
                f"CPU reference has no distributed boundary work: {path.name}"
            )
    if reference_observables is None:
        raise QualificationContractError(
            f"CPU reference qualification has no observables: {path.name}"
        )
    return reference_observables


def validate_cpu_reference_receipt(
    receipt: dict[str, Any], repo: pathlib.Path
) -> None:
    repo = repo.resolve(strict=True)
    configuration = receipt.get("configuration")
    if receipt.get("build_kind") != CPU_REFERENCE_BUILD_KIND or not isinstance(
        configuration, dict
    ) or configuration.get("qualification_contract") != CPU_REFERENCE_CONTRACT_NAME:
        raise QualificationContractError(
            "CPU FP64 MPI+Python receipt has the wrong qualification contract"
        )
    if receipt.get("git_status_porcelain") != []:
        raise QualificationContractError(
            "CPU reference receipt was not built from a clean source tree"
        )
    _cpu_reference_exact_keys(
        configuration,
        {
            "builder",
            "qualification_contract",
            "configure_argv",
            "environment",
            "lockfiles",
        },
        "configuration",
    )
    expected_build = (repo / "build" / "meep-cpu-mpi-python-fp64").resolve(
        strict=True
    )
    expected_install = (repo / "install" / "meep-cpu-mpi-python-fp64").resolve(
        strict=True
    )
    expected_environment = (repo / ".envs" / "gpmeep-cpu-mpi-fp64").resolve(
        strict=True
    )
    if pathlib.Path(str(receipt.get("build_dir", ""))).resolve(
        strict=True
    ) != expected_build:
        raise QualificationContractError("CPU reference build directory is not fixed")
    expected_builder = (repo / "scripts" / "build-meep-cpu-mpi-python-fp64.sh").resolve(
        strict=True
    )
    if _cpu_reference_record_path(
        configuration, "builder", repo, "configuration"
    ) != expected_builder:
        raise QualificationContractError("CPU reference builder path is not authoritative")
    environment = _cpu_reference_exact_keys(
        configuration.get("environment"),
        {
            "MEEP_GPU_MAKE_JOBS",
            "MEEP_GPU_BACKEND",
            "CC",
            "CXX",
            "FC",
            "F77",
            "MPICXX",
            "HOME",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "MPLCONFIGDIR",
        },
        "environment",
    )
    jobs = environment["MEEP_GPU_MAKE_JOBS"]
    if (
        not isinstance(jobs, str)
        or not jobs.isdigit()
        or int(jobs) <= 0
        or environment["MEEP_GPU_BACKEND"] != "cpu"
    ):
        raise QualificationContractError("CPU reference build environment is invalid")
    for name in ("CC", "CXX", "FC", "F77", "MPICXX"):
        value = environment[name]
        if not isinstance(value, str):
            raise QualificationContractError(
                f"CPU reference compiler environment is invalid: {name}"
            )
        try:
            compiler = pathlib.Path(value).resolve(strict=True)
            compiler.relative_to(expected_environment / "bin")
        except (OSError, ValueError) as exc:
            raise QualificationContractError(
                f"CPU reference compiler escapes the environment: {name}"
            ) from exc
    expected_build_home = expected_build / "build-home"
    if {
        "HOME": environment["HOME"],
        "XDG_CACHE_HOME": environment["XDG_CACHE_HOME"],
        "XDG_CONFIG_HOME": environment["XDG_CONFIG_HOME"],
        "MPLCONFIGDIR": environment["MPLCONFIGDIR"],
    } != {
        "HOME": str(expected_build_home),
        "XDG_CACHE_HOME": str(expected_build_home / ".cache"),
        "XDG_CONFIG_HOME": str(expected_build_home / ".config"),
        "MPLCONFIGDIR": str(expected_build_home / ".matplotlib"),
    }:
        raise QualificationContractError(
            "CPU reference build home/cache paths are not isolated"
        )
    lockfiles = _cpu_reference_exact_keys(
        configuration.get("lockfiles"), {"environment_lock"}, "lockfiles"
    )
    lock_identity = _cpu_reference_runtime_identity(
        lockfiles, "environment_lock", repo, "lockfiles"
    )
    expected_lock = (
        repo / "environment" / "locks" / "cuda-mpi-linux-64.lock"
    ).resolve(strict=True)
    if pathlib.Path(lock_identity["path"]) != expected_lock:
        raise QualificationContractError(
            "CPU reference environment lock path is not authoritative"
        )
    arguments = configuration.get("configure_argv")
    if not isinstance(arguments, list) or not all(
        isinstance(item, str) for item in arguments
    ) or len(arguments) != len(set(arguments)):
        raise QualificationContractError(
            "CPU reference configure argument vector is invalid"
        )
    prefix_arguments = [item for item in arguments if item.startswith("--prefix=")]
    fixed_arguments = set(arguments) - set(prefix_arguments)
    if len(prefix_arguments) != 1 or fixed_arguments != {
        "--enable-maintainer-mode",
        "--enable-shared",
        "--disable-single",
        "--disable-cuda",
        "--disable-cuda-fast-math",
        "--with-openmp",
        "--with-mpi",
        "--with-python",
        "--without-scheme",
    }:
        raise QualificationContractError(
            "CPU reference configure argument contract is not exact"
        )
    configuration_files = _cpu_reference_exact_keys(
        receipt.get("configuration_files"),
        CPU_REFERENCE_CONFIGURATION_FILES,
        "configuration files",
    )
    artifacts = _cpu_reference_exact_keys(
        receipt.get("artifacts"), CPU_REFERENCE_ARTIFACTS, "artifacts"
    )
    manifests = _cpu_reference_exact_keys(
        receipt.get("manifests"), CPU_REFERENCE_MANIFESTS, "manifests"
    )
    toolchain = _cpu_reference_exact_keys(
        receipt.get("toolchain"), CPU_REFERENCE_TOOLS, "toolchain"
    )
    configuration_paths = {
        name: _cpu_reference_record_path(
            configuration_files, name, repo, "configuration files"
        )
        for name in CPU_REFERENCE_CONFIGURATION_FILES
    }
    expected_configuration_paths = {
        "config_h": expected_build / "config.h",
        "config_status": expected_build / "config.status",
        "environment_explicit": expected_build / "environment-explicit.lock",
        "conda_prefix_content_audit": expected_build
        / "conda-prefix-content-audit.json",
        "openmpi_qualification_params": repo
        / "environment"
        / "openmpi-qualification-mca-params.conf",
    }
    if configuration_paths != expected_configuration_paths:
        raise QualificationContractError(
            "CPU reference configuration-file paths are not fixed"
        )
    audit_path = configuration_paths["conda_prefix_content_audit"]
    try:
        audit = _cpu_reference_json(
            audit_path.read_text(encoding="utf-8"), "conda prefix content audit"
        )
        lock_lines = pathlib.Path(lock_identity["path"]).read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError as exc:
        raise QualificationContractError(
            f"CPU reference environment audit input is unreadable: {exc}"
        ) from exc
    _cpu_reference_exact_keys(
        audit,
        {
            "schema_version",
            "environment_prefix",
            "package_cache",
            "archive_snapshot",
            "archive_decoder",
            "lock",
            "metadata_package_count",
            "owned_path_count",
            "hashed_installed_path_count",
            "archive_bound_hardlink_count",
            "relocated_hardlink_count",
            "unhashed_bytecode_path_count",
            "generated_entry_points",
            "softlink_count",
            "softlinks",
            "unowned_allowed_path_count",
            "package_records",
            "pass",
        },
        "conda prefix content audit",
    )
    locked_packages = []
    for line in lock_lines:
        value = line.strip()
        if "://" not in value:
            continue
        url, separator, digest = value.rpartition("#")
        if (
            not separator
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise QualificationContractError(
                "CPU reference exact lock has an invalid package pin"
            )
        locked_packages.append({"url": url, "sha256": digest})
    locked_packages.sort(key=lambda item: item["url"])
    audit_lock = _cpu_reference_exact_keys(
        audit["lock"], {"path", "package_count", "sha256"}, "audit lock"
    )
    audit_package_cache = _cpu_reference_exact_keys(
        audit["package_cache"], {"path", "archive_count"}, "audit package cache"
    )
    audit_snapshot = _cpu_reference_exact_keys(
        audit["archive_snapshot"],
        {
            "open_flags",
            "hash_and_parse_same_private_snapshot",
            "descriptor_retained_until_terminal_verification",
            "pathname_fingerprint_rechecked",
        },
        "audit archive snapshot",
    )
    audit_decoder = _cpu_reference_exact_keys(
        audit["archive_decoder"],
        {"path", "sha256", "version", "timeout_seconds"},
        "audit archive decoder",
    )
    count_fields = (
        "metadata_package_count",
        "owned_path_count",
        "hashed_installed_path_count",
        "archive_bound_hardlink_count",
        "relocated_hardlink_count",
        "unhashed_bytecode_path_count",
        "softlink_count",
        "unowned_allowed_path_count",
    )
    try:
        package_cache_path = pathlib.Path(audit_package_cache["path"])
        package_cache_canonical = (
            package_cache_path.is_absolute()
            and package_cache_path.is_dir()
            and package_cache_path.resolve() == package_cache_path
        )
        decoder_path = pathlib.Path(audit_decoder["path"])
        decoder_canonical = (
            decoder_path == pathlib.Path("/usr/bin/zstd")
            and decoder_path.is_file()
            and not decoder_path.is_symlink()
            and decoder_path.resolve() == decoder_path
        )
    except (OSError, TypeError):
        package_cache_canonical = False
        decoder_canonical = False
    if (
        audit["schema_version"] != 3
        or not locked_packages
        or len(locked_packages)
        != len({item["url"] for item in locked_packages})
        or audit["environment_prefix"] != str(expected_environment)
        or audit["pass"] is not True
        or audit_lock["path"] != lock_identity["path"]
        or audit_lock["sha256"] != lock_identity["sha256"]
        or audit_lock["package_count"] != len(locked_packages)
        or audit["metadata_package_count"] != len(locked_packages)
        or audit_package_cache["archive_count"] != len(locked_packages)
        or not package_cache_canonical
        or audit_snapshot["open_flags"]
        != ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"]
        or audit_snapshot["hash_and_parse_same_private_snapshot"] is not True
        or audit_snapshot[
            "descriptor_retained_until_terminal_verification"
        ] is not True
        or audit_snapshot["pathname_fingerprint_rechecked"] is not True
        or not decoder_canonical
        or re.fullmatch(r"[0-9a-f]{64}", str(audit_decoder["sha256"])) is None
        or audit_decoder["sha256"] != _sha256_file(decoder_path)
        or not isinstance(audit_decoder["version"], str)
        or not audit_decoder["version"]
        or audit_decoder["timeout_seconds"] != 120
        or audit["package_records"] != locked_packages
        or not isinstance(audit["generated_entry_points"], list)
        or not isinstance(audit["softlinks"], list)
        or audit["softlink_count"] != len(audit["softlinks"])
        or audit["archive_bound_hardlink_count"] <= 0
        or audit["archive_bound_hardlink_count"]
        + audit["softlink_count"]
        + audit["unhashed_bytecode_path_count"]
        + len(audit["generated_entry_points"])
        != audit["owned_path_count"]
        or any(
            not isinstance(audit[name], int)
            or isinstance(audit[name], bool)
            or audit[name] < 0
            for name in count_fields
        )
    ):
        raise QualificationContractError(
            "CPU reference conda-prefix content audit is not exact"
        )
    for name in CPU_REFERENCE_TOOLS:
        tool_path = _cpu_reference_record_path(toolchain, name, repo, "toolchain")
        try:
            tool_path.relative_to(expected_environment / "bin")
        except ValueError as exc:
            raise QualificationContractError(
                f"CPU reference tool escapes the environment: {name}"
            ) from exc
    artifact_paths = {
        name: _cpu_reference_record_path(artifacts, name, repo, "artifacts")
        for name in CPU_REFERENCE_ARTIFACTS
    }
    installed_extension = artifact_paths["installed_python_extension"]
    try:
        installed_extension_parts = installed_extension.relative_to(
            expected_install
        ).parts
    except ValueError:
        installed_extension_parts = ()
    installed_extension_layout_is_exact = (
        len(installed_extension_parts) == 5
        and installed_extension_parts[0] == "lib"
        and re.fullmatch(r"python[0-9]+\.[0-9]+", installed_extension_parts[1])
        is not None
        and installed_extension_parts[2:4] == ("site-packages", "meep")
        and installed_extension_parts[4].startswith("_meep.so")
    )
    if (
        artifact_paths["python_extension"].parent
        != expected_build / "python" / "meep"
        or not artifact_paths["python_extension"].name.startswith("_meep.so")
        or artifact_paths["libmeep"].parent != expected_build / "src" / ".libs"
        or not artifact_paths["libmeep"].name.startswith("libmeep.so")
        or not installed_extension_layout_is_exact
        or artifact_paths["installed_libmeep"].parent != expected_install / "lib"
        or not artifact_paths["installed_libmeep"].name.startswith("libmeep.so")
    ):
        raise QualificationContractError("CPU reference artifact paths are not fixed")
    expected_python = _cpu_reference_runtime_identity(
        toolchain, "python", repo, "toolchain"
    )
    artifact_identities = {
        name: _cpu_reference_runtime_identity(
            artifacts, name, repo, "artifacts"
        )
        for name in CPU_REFERENCE_ARTIFACTS
    }
    manifest_roots: dict[str, pathlib.Path] = {}
    for name, manifest in manifests.items():
        if not isinstance(manifest, dict) or not isinstance(manifest.get("root"), str):
            raise QualificationContractError(
                f"CPU reference manifest root is invalid: {name}"
            )
        manifest_roots[name] = _resolve_recorded_path(
            manifest["root"], repo
        ).resolve(strict=True)
    installed_prefix = manifest_roots["installed_prefix"]
    configured_prefix = pathlib.Path(prefix_arguments[0].split("=", 1)[1]).resolve(
        strict=True
    )
    if configured_prefix != installed_prefix or installed_prefix != expected_install:
        raise QualificationContractError(
            "CPU reference configured/install prefix identity differs"
        )
    if manifest_roots != {
        "in_place_python": expected_build / "python" / "meep",
        "installed_python": artifact_paths["installed_python_extension"].parent,
        "installed_prefix": expected_install,
        "installed_environment": expected_environment,
        "qualification_logs": expected_build / "cpu-mpi-fp64-qualification",
        "host_build_logs": expected_build / "host-build-logs",
    }:
        raise QualificationContractError("CPU reference manifest roots are not fixed")
    if any("excluded_suffixes" in manifest for manifest in manifests.values()):
        raise QualificationContractError(
            "CPU reference runtime manifests must not exclude bytecode"
        )
    for name in ("installed_python_extension", "installed_libmeep"):
        if installed_prefix not in artifact_paths[name].parents:
            raise QualificationContractError(
                f"CPU reference installed artifact escapes prefix: {name}"
            )
    log_directory = manifest_roots["qualification_logs"]
    actual_logs = {path.name for path in log_directory.iterdir()}
    if actual_logs != set(CPU_REFERENCE_LOG_SPECS):
        raise QualificationContractError(
            f"CPU reference qualification log inventory is not exact: {sorted(actual_logs)}"
        )
    host_logs = manifest_roots["host_build_logs"]
    expected_host_logs = {
        f"{label}.log" for label in ("autoreconf", "configure", "build", "check", "install")
    }
    if {path.name for path in host_logs.iterdir()} != expected_host_logs:
        raise QualificationContractError(
            "CPU reference host-build log inventory is not exact"
        )
    for name in expected_host_logs:
        label = name.removesuffix(".log")
        payload = (host_logs / name).read_bytes()
        lines = payload.splitlines()
        if (
            len(payload) > 256 * 1024 * 1024
            or not lines
            or lines.count(
                f"gpmeep-host-build-step-pass:{label}".encode("ascii")
            )
            != 1
            or lines[-1]
            != f"gpmeep-host-build-step-pass:{label}".encode("ascii")
        ):
            raise QualificationContractError(
                f"CPU reference host-build log lacks its pass marker: {name}"
            )
    lane_observables: dict[str, tuple[float, float, float]] = {}
    for name, (runtime_kind, mpi_size, extension_name, libmeep_name) in (
        CPU_REFERENCE_LOG_SPECS.items()
    ):
        lane_observables[name] = _validate_cpu_reference_log(
            log_directory / name,
            runtime_kind=runtime_kind,
            mpi_size=mpi_size,
            expected_python=expected_python,
            expected_extension=artifact_identities[extension_name],
            expected_libmeep=artifact_identities[libmeep_name],
        )
    reference = lane_observables["in-place-one-rank.log"]
    for name, observed in lane_observables.items():
        if (
            not math.isclose(observed[0], reference[0], rel_tol=1e-10, abs_tol=1e-13)
            or not math.isclose(
                observed[1], reference[1], rel_tol=1e-10, abs_tol=1e-13
            )
            or not math.isclose(
                observed[2], reference[2], rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise QualificationContractError(
                f"CPU reference qualification lanes disagree: {name}"
            )


def validate_v2_receipt(receipt: dict[str, Any], repo: pathlib.Path) -> None:
    configuration = receipt.get("configuration")
    contract = (
        configuration.get("qualification_contract")
        if isinstance(configuration, dict)
        else None
    )
    if (
        receipt.get("build_kind") == CPU_REFERENCE_BUILD_KIND
        or contract == CPU_REFERENCE_CONTRACT_NAME
    ):
        validate_cpu_reference_receipt(receipt, repo)
        return
    if receipt.get("build_kind") == "cuda-mpi-python-fp32" and (
        not isinstance(configuration, dict)
        or configuration.get("qualification_contract") != CONTRACT_NAME
    ):
        raise QualificationContractError(
            "CUDA/MPI/Python FP32 receipt has the wrong qualification contract"
        )
    if not isinstance(configuration, dict) or (
        configuration.get("qualification_contract") != CONTRACT_NAME
    ):
        return
    configuration_files = receipt.get("configuration_files")
    artifacts = receipt.get("artifacts")
    manifests = receipt.get("manifests")
    if not all(
        isinstance(item, dict)
        for item in (configuration_files, artifacts, manifests)
    ):
        raise QualificationContractError("qualification-v2 receipt groups are invalid")
    assert isinstance(configuration_files, dict)
    assert isinstance(artifacts, dict)
    assert isinstance(manifests, dict)
    missing_configuration = sorted(
        set(REQUIRED_CONFIGURATION_BINDINGS) - set(configuration_files)
    )
    missing_artifacts = sorted(set(REQUIRED_ARTIFACT_BINDINGS) - set(artifacts))
    if missing_configuration or missing_artifacts:
        raise QualificationContractError(
            "qualification-v2 receipt lacks required bindings: "
            + ", ".join((*missing_configuration, *missing_artifacts))
        )
    validate_release_cuda_math_policy(receipt, repo)
    log_manifest = manifests.get("qualification_logs")
    environment_manifest = manifests.get("installed_environment")
    installed_prefix_manifest = manifests.get("installed_prefix")
    if not all(
        isinstance(item, dict)
        for item in (log_manifest, environment_manifest, installed_prefix_manifest)
    ):
        raise QualificationContractError(
            "qualification-v2 receipt lacks log/environment manifests"
        )
    seal_record = configuration_files["qualification_contract_v2"]
    before_record = configuration_files["qualification_identity_before"]
    after_record = configuration_files["qualification_identity_after"]
    if not all(
        isinstance(record, dict) and isinstance(record.get("path"), str)
        for record in (seal_record, before_record, after_record)
    ):
        raise QualificationContractError("qualification-v2 configuration binding is invalid")
    seal = _load_json(_resolve_recorded_path(seal_record["path"], repo))
    before = _load_json(_resolve_recorded_path(before_record["path"], repo))
    after = _load_json(_resolve_recorded_path(after_record["path"], repo))
    if before != after or seal.get("identity_sha256") != _canonical_sha256(before):
        raise QualificationContractError("qualification identity epochs do not match")
    _identity_immutable_empty_directory_paths(
        before,
        repo,
        require_contract_bindings=True,
        expected_directory_parent=_resolve_recorded_path(
            before_record["path"], repo
        ).parent,
    )
    _identity_immutable_empty_directory_paths(
        after,
        repo,
        require_contract_bindings=True,
        expected_directory_parent=_resolve_recorded_path(
            after_record["path"], repo
        ).parent,
    )
    if (
        seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("qualification_contract") != CONTRACT_NAME
        or seal.get("required_logs") != list(ALL_REQUIRED_QUALIFICATION_LOGS)
        or tuple(sorted(seal.get("materialization_logs", {})))
        != MATERIALIZATION_LOGS
        or seal.get("artifacts") != before.get("artifacts")
        or seal.get("source") != before.get("source")
        or seal.get("installed_environment") != before.get("installed_environment")
        or seal.get("installed_prefix") != before.get("installed_prefix")
        or seal.get(IMMUTABLE_EMPTY_DIRECTORIES_KEY)
        != before.get(IMMUTABLE_EMPTY_DIRECTORIES_KEY)
    ):
        raise QualificationContractError("qualification-v2 seal payload is invalid")
    if (
        seal.get("identity_before") != before_record
        or seal.get("identity_after") != after_record
    ):
        raise QualificationContractError(
            "qualification identity files are not seal/receipt-bound"
        )
    if before.get("source") not in (receipt.get("source_start"), receipt.get("source_end")):
        raise QualificationContractError("qualification source is not receipt-bound")
    if before.get("installed_environment") != environment_manifest:
        raise QualificationContractError("qualification prefix is not receipt-bound")
    if before.get("installed_prefix") != installed_prefix_manifest:
        raise QualificationContractError(
            "qualification install prefix is not receipt-bound"
        )
    installed_libmeep = artifacts.get("installed_libmeep")
    if not isinstance(installed_libmeep, dict) or not isinstance(
        installed_libmeep.get("path"), str
    ) or not isinstance(installed_prefix_manifest.get("root"), str):
        raise QualificationContractError(
            "qualification install-prefix root binding is invalid"
        )
    expected_installed_prefix = _resolve_recorded_path(
        installed_libmeep["path"], repo
    ).resolve().parent.parent
    actual_installed_prefix = _resolve_recorded_path(
        installed_prefix_manifest["root"], repo
    ).resolve()
    if actual_installed_prefix != expected_installed_prefix:
        raise QualificationContractError(
            "qualification install-prefix root is not artifact-bound"
        )
    for name in REQUIRED_ARTIFACT_BINDINGS:
        if before.get("artifacts", {}).get(name) != artifacts.get(name):
            raise QualificationContractError(
                f"tested artifact identity is not final-receipt-bound: {name}"
            )
    manifest_files = log_manifest.get("files")
    if not isinstance(manifest_files, list):
        raise QualificationContractError("qualification log manifest is invalid")
    manifest_by_path = {
        record.get("path"): record
        for record in manifest_files
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    if (
        len(manifest_files) != len(ALL_QUALIFICATION_DIRECTORY_LOGS)
        or set(manifest_by_path) != set(ALL_QUALIFICATION_DIRECTORY_LOGS)
    ):
        raise QualificationContractError(
            "qualification log manifest is not the exact contracted file set"
        )
    seal_logs = seal.get("logs")
    if not isinstance(seal_logs, dict) or tuple(sorted(seal_logs)) != ALL_REQUIRED_QUALIFICATION_LOGS:
        raise QualificationContractError("qualification seal log set is incomplete")
    log_root = _resolve_recorded_path(str(log_manifest.get("root", "")), repo)
    current_logs = validate_logs(log_root, ALL_REQUIRED_QUALIFICATION_LOGS)
    current_materialization_logs = validate_no_uncontracted_pass_logs(log_root)
    validate_mutation_sentinel_log(log_root, before)
    validate_python_runtime_logs(log_root, artifacts, repo)
    validate_installed_lazy_api_qualification_log(log_root, artifacts, repo)
    validate_installed_mpb_qualification_log(log_root, artifacts, repo)
    validate_installed_cuda_qualification_logs(log_root, artifacts, repo)
    validate_direct_elf_logs(log_root, artifacts, repo)
    for name in ALL_REQUIRED_QUALIFICATION_LOGS:
        sealed = seal_logs.get(name)
        manifested = manifest_by_path.get(name)
        if sealed != current_logs[name]:
            raise QualificationContractError(f"qualification log seal mismatch: {name}")
        if not isinstance(manifested, dict) or any(
            manifested.get(key) != sealed.get(key)
            for key in ("path", "size_bytes", "sha256")
        ):
            raise QualificationContractError(f"qualification log manifest mismatch: {name}")
    sealed_materialization_logs = seal.get("materialization_logs")
    if not isinstance(sealed_materialization_logs, dict):
        raise QualificationContractError(
            "qualification seal materialization log set is invalid"
        )
    for name in MATERIALIZATION_LOGS:
        current = current_materialization_logs[name]
        sealed = sealed_materialization_logs.get(name)
        manifested = manifest_by_path.get(name)
        if sealed != current:
            raise QualificationContractError(
                f"materialization log seal mismatch: {name}"
            )
        if not isinstance(manifested, dict) or any(
            manifested.get(key) != current.get(key)
            for key in ("path", "size_bytes", "sha256")
        ):
            raise QualificationContractError(
                f"materialization log manifest mismatch: {name}"
            )


def _named_paths(
    values: list[str], *, resolve: bool = True
) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise QualificationContractError("expected unique NAME=PATH")
        path = pathlib.Path(raw_path)
        result[name] = path.resolve() if resolve else _literal_absolute_path(path)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--repo", required=True, type=pathlib.Path)
    snapshot.add_argument("--prefix", required=True, type=pathlib.Path)
    snapshot.add_argument("--installed-prefix", required=True, type=pathlib.Path)
    snapshot.add_argument("--output", required=True, type=pathlib.Path)
    snapshot.add_argument("--artifact", action="append", default=[])
    snapshot.add_argument(
        "--immutable-empty-directory", action="append", default=[]
    )
    seal = commands.add_parser("seal")
    seal.add_argument("--repo", required=True, type=pathlib.Path)
    seal.add_argument("--log-dir", required=True, type=pathlib.Path)
    seal.add_argument("--before", required=True, type=pathlib.Path)
    seal.add_argument("--after", required=True, type=pathlib.Path)
    seal.add_argument("--output", required=True, type=pathlib.Path)
    status = commands.add_parser("check-expected-failure-status")
    status.add_argument("status", type=int)
    python_runtime = commands.add_parser("attest-python-runtime")
    python_runtime.add_argument(
        "--expected-extension", required=True, type=pathlib.Path
    )
    python_runtime.add_argument(
        "--expected-libmeep", required=True, type=pathlib.Path
    )
    python_runtime.add_argument("--log-name", required=True)
    sentinel = commands.add_parser("sentinel")
    sentinel.add_argument("--identity", required=True, type=pathlib.Path)
    sentinel.add_argument("--repo", required=True, type=pathlib.Path)
    sentinel.add_argument("--prefix", required=True, type=pathlib.Path)
    sentinel.add_argument("--ready", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.action == "snapshot":
        output = args.output.resolve()
        value = create_identity_snapshot(
            args.repo.resolve(),
            args.prefix.resolve(),
            _named_paths(args.artifact),
            installed_prefix=args.installed_prefix.resolve(),
            immutable_empty_directories=_named_paths(
                args.immutable_empty_directory, resolve=False
            ),
            require_contract_bindings=True,
        )
        _identity_immutable_empty_directory_paths(
            value,
            args.repo.resolve(),
            require_contract_bindings=True,
            expected_directory_parent=output.parent,
        )
        _atomic_write_json(output, value)
        return 0
    if args.action == "seal":
        seal_contract(
            args.repo.resolve(), args.log_dir.resolve(), args.before.resolve(),
            args.after.resolve(), args.output.resolve()
        )
        return 0
    if args.action == "attest-python-runtime":
        attest_python_runtime(
            args.expected_extension, args.expected_libmeep, args.log_name
        )
        # Meep/MPI finalizers may print after Python atexit handlers, which
        # would destroy the exact-terminal-marker contract.  This process has
        # completed its sole read-only attestation; exit without running those
        # later emitters after explicitly flushing the evidence above.
        os._exit(0)
    if args.action == "sentinel":
        run_mutation_sentinel(
            args.identity.resolve(),
            args.repo.resolve(),
            args.prefix.resolve(),
            args.ready.resolve(),
        )
        return 0
    return 0 if expected_failure_status_is_acceptable(args.status) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except QualificationContractError as exc:
        print(f"qualification contract error: {exc}", file=sys.stderr)
        raise SystemExit(1)
