#!/usr/bin/env python3
"""Run gpmeep Python cases against CPU and strict CUDA backends.

This runner intentionally uses only the Python standard library.  It treats
classification, dependency blockers, unittest skips, backend provenance, and
GPU dispatch evidence as first-class results instead of silently skipping
them.
"""

from __future__ import annotations

import argparse
import codecs
import copy
import ctypes
import datetime as dt
import errno
import fnmatch
import glob
import hashlib
import heapq
import importlib.util
import json
import math
import os
import pathlib
import platform
import re
import selectors
import signal
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, BinaryIO, Callable, Iterable, Iterator

SCRIPT_SELF_DIR = pathlib.Path(__file__).resolve().parent
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
for import_root in (SCRIPTS_DIR, SCRIPT_SELF_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpmeep_provenance import (  # noqa: E402
    ProvenanceError,
    bounded_command,
    file_record as provenance_file_record,
    sha256_file as provenance_sha256_file,
    source_snapshot as provenance_source_snapshot,
    trusted_git,
    verify_build_receipt,
)
from gpmeep_qualification_contract import (  # noqa: E402
    CONTRACT_NAME as CENTRAL_QUALIFICATION_CONTRACT,
    QualificationContractError,
    validate_v2_receipt,
)


SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 4
METRIC_EVIDENCE_SCHEMA = "gpmeep-json-metric-evidence-v2"
DIRECT_METRIC_EVIDENCE_SCHEMA = "gpmeep-json-metric-evidence-v3-direct"
EXPECTED_BUILD_KIND = "cuda-mpi-python-fp32"
EXPECTED_QUALIFICATION_CONTRACT = "gpmeep-cuda-mpi-python-fp32-v2"
if EXPECTED_QUALIFICATION_CONTRACT != CENTRAL_QUALIFICATION_CONTRACT:
    raise RuntimeError("qualification contract constants disagree")
EXPECTED_BUILDER_RELATIVE_PATH = pathlib.Path(
    "scripts/build-meep-cuda-mpi-python.sh"
)
EXIT_OK = 0
EXIT_EXECUTION_FAILED = 1
EXIT_MANIFEST_INVALID = 2
EXIT_COMPARISON_FAILED = 3
EXIT_BLOCKED = 4
EXIT_BACKEND_CONTRACT = 5
EXIT_EVIDENCE_INTEGRITY = 6
MAX_INLINE_JSON_METRICS = 4096
MAX_INLINE_JSON_BYTES = 1024 * 1024
# Small payloads retain the canonical external-sort path so historical report
# evidence remains byte-for-byte stable.  Large same-order payloads can avoid
# writing several gigabytes of derived sort records while still parsing both
# complete documents twice and binding every emitted scalar to parser origin.
DIRECT_JSON_METRIC_MIN_PAYLOAD_BYTES = MAX_INLINE_JSON_BYTES
MAX_REPORTED_JSON_FAILURES = 4096
# Batch 5 retains nine complete 400 x 400 Cherenkov fields plus the independent
# resolution-doubled 800 x 800 field.  The generic oracle represents complex
# arrays as separate real/imaginary leaves, yielding about 4.2 million metrics
# for this intentionally full-field case.  Keep a narrow finite ceiling above
# that audited workload; the independent payload/path/report byte ceilings and
# one-million-value per-result limit remain unchanged.
MAX_JSON_METRICS = 5_000_000
MAX_JSON_PAYLOAD_BYTES = 512 * 1024 * 1024
MAX_JSON_METRIC_PATH_BYTES = 4096
MAX_JSON_METRIC_PATH_BYTES_TOTAL = 512 * 1024 * 1024
MAX_JSON_TOLERANCE_RULES = 1024
MAX_JSON_TOLERANCE_RULE_BYTES_TOTAL = 256 * 1024
MAX_NORMALIZED_STDOUT_PREVIEW_BYTES = 4096
MAX_REPORT_JSON_BYTES = 16 * 1024 * 1024
MAX_JSON_NESTING_DEPTH = 128
# External-sort chunks deliberately stay small: the Python objects used while
# sorting cost several times more than their encoded records.  This keeps the
# comparator's working set bounded even for the multi-million-value field
# contracts while still using large sequential I/O.
JSON_METRIC_SORT_CHUNK_BYTES = 8 * 1024 * 1024
JSON_METRIC_STREAM_READ_BYTES = 1024 * 1024
MAX_JSON_NUMBER_TOKEN_BYTES = 16 * 1024
JSON_SORT_RECORD_DOMAIN = b"gpmeep-json-sort-record-v1\0"
JSON_SORT_RECORD_DIGEST_BYTES = hashlib.sha256().digest_size
# Release evidence must never hide an integrity fault behind a retry.  A
# checksum failure is terminal; developers can rerun the complete comparison
# and retain the failed attempt as separate evidence.
JSON_SORT_RECORD_READ_ATTEMPTS = 1
# Child output is streamed to evidence files rather than accumulated in a
# subprocess PIPE.  The stdout ceiling intentionally accommodates the audited
# 512 MiB JSON metric payload contract plus a narrow amount of non-metric
# output.  stderr is diagnostic-only and therefore substantially smaller.
MAX_BACKEND_STDOUT_BYTES = MAX_JSON_PAYLOAD_BYTES + 16 * 1024 * 1024
MAX_BACKEND_STDERR_BYTES = 64 * 1024 * 1024
MAX_BACKEND_OUTPUT_BYTES = (
    MAX_BACKEND_STDOUT_BYTES + MAX_BACKEND_STDERR_BYTES
)
MAX_CAPTURE_READ_BYTES = 64 * 1024
UINT64_MAX = (1 << 64) - 1
STRICT_CUDA_PHASE_CALL_COUNTERS = {
    "boundaries.cpu_boundary_calls",
    "boundaries.cuda_boundary_calls",
    "dfts.cpu_dft_calls",
    "dfts.cuda_dft_calls",
    "dft_reductions.cpu_dft_reduction_calls",
    "dft_reductions.cuda_dft_reduction_calls",
    "dft_materializations.cpu_dft_array_materialization_calls",
    "dft_materializations.cuda_dft_array_materialization_calls",
    "dft_materializations.cpu_dft_output_calls",
    "dft_materializations.cuda_dft_output_calls",
    "eigenmode_overlaps.cpu_eigenmode_overlap_calls",
    "eigenmode_overlaps.cuda_eigenmode_overlap_calls",
    "ldos.cpu_ldos_reduction_calls",
    "ldos.cuda_ldos_reduction_calls",
    "dispatch.cpu_curl_calls",
    "dispatch.cuda_curl_calls",
    "field_updates.cpu_update_eh_calls",
    "field_updates.cuda_update_eh_calls",
    "polarizations.cpu_polarization_calls",
    "polarizations.cuda_polarization_calls",
    "sources.cpu_source_calls",
    "sources.cuda_source_calls",
    "near2far.cpu_near2far_adjoint_calls",
    "near2far.cuda_near2far_adjoint_calls",
}
PHASE_CALL_COUNTERS = STRICT_CUDA_PHASE_CALL_COUNTERS | {
    "dft_batches.dft_batch_calls",
    "dft_checkpoints.cpu_dft_checkpoint_load_calls",
    "dft_checkpoints.cpu_dft_checkpoint_save_calls",
    "dft_checkpoints.cuda_dft_checkpoint_load_calls",
    "dft_checkpoints.cuda_dft_checkpoint_save_calls",
    "dft_materializations.host_synthetic_material_array_calls",
    "dft_materializations.cuda_dft_output_staging_calls",
    "dft_materializations.cuda_dft_materialization_kernel_launches",
    "dft_materializations.cuda_dft_materialization_result_device_to_host_bytes",
    "dft_materializations.dft_array_mpi_allreduce_calls",
    "dft_reductions.dft_reduction_mpi_allreduce_calls",
    "dft_scales.cpu_dft_scale_calls",
    "dft_scales.cuda_dft_scale_calls",
    "eigenmode_overlaps.cuda_eigenmode_mode_flux_calls",
    "eigenmode_overlaps.cuda_eigenmode_mode_mode_calls",
    "eigenmode_overlaps.host_mode_profile_sampling_calls",
    "eigenmode_overlaps.eigenmode_mpi_allreduce_calls",
    "near2far.cpu_near2far_transform_calls",
    "near2far.cuda_near2far_fast_precision_calls",
    "near2far.cuda_near2far_mixed_precision_calls",
    "near2far.cuda_near2far_cancellation_retries",
    "near2far.cuda_near2far_transform_calls",
    "near2far.cuda_near2far_adjoint_fast_precision_calls",
    "near2far.cuda_near2far_adjoint_mixed_precision_calls",
    "near2far.cuda_near2far_adjoint_cancellation_retries",
    "near2far.near2far_mpi_allreduce_calls",
}

# ``required_cuda_call_counters`` predates the resident CUDA consumers and is
# retained as a manifest compatibility name.  Its values now describe three
# deliberately distinct kinds of required activity:
#
# * paired phase counters prove that CUDA ran and the corresponding CPU phase
#   did not run during a strict-CUDA execution;
# * CUDA-only activity (kernel launches, staging and compact transfers) has no
#   meaningful positive CPU counterpart; and
# * shared activity proves that a backend-independent consumer, such as the
#   MPI reduction following a CUDA materialization, also ran.
#
# Keep the latter category explicit.  Merely accepting every registered
# non-CUDA counter here would let a manifest accidentally use unrelated host
# work as evidence for a CUDA path.
REQUIRED_CUDA_SHARED_ACTIVITY_COUNTERS = {
    "dft_materializations.dft_array_mpi_allreduce_calls",
    "near2far.near2far_mpi_allreduce_calls",
}
REQUIRED_CUDA_CPU_COUNTER_PAIRS = {
    counter: counter.replace(".cuda_", ".cpu_", 1)
    for counter in PHASE_CALL_COUNTERS
    if ".cuda_" in counter
    and counter.replace(".cuda_", ".cpu_", 1) in PHASE_CALL_COUNTERS
}
REQUIRED_CUDA_ONLY_ACTIVITY_COUNTERS = {
    counter
    for counter in PHASE_CALL_COUNTERS
    if ".cuda_" in counter
    and counter not in REQUIRED_CUDA_CPU_COUNTER_PAIRS
}
REQUIRED_CUDA_ACTIVITY_COUNTERS = (
    set(REQUIRED_CUDA_CPU_COUNTER_PAIRS)
    | REQUIRED_CUDA_ONLY_ACTIVITY_COUNTERS
    | REQUIRED_CUDA_SHARED_ACTIVITY_COUNTERS
)

SAFE_AMBIENT_ENVIRONMENT = {
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "TERM",
    "TMP",
    "TMPDIR",
    "TEMP",
    "TZ",
}

RUN_DISPOSITION = "run"
KNOWN_DISPOSITIONS = {
    RUN_DISPOSITION,
    "compatibility_only",
    "covered_by_test",
    "deferred_milestone",
    "expected_feature_gap",
    "external_dependency",
    "mpi_only",
    "notebook_duplicate",
    "oracle_pending",
}
NONBLOCKING_DISPOSITIONS = KNOWN_DISPOSITIONS - {RUN_DISPOSITION}
BLOCKING_SELECTED_DISPOSITIONS = {
    "expected_feature_gap",
    "external_dependency",
    "oracle_pending",
}
KNOWN_KINDS = {"unittest", "example", "notebook"}
KNOWN_COMPARISONS = {
    "embedded_oracle",
    "json_metrics",
    "normalized_stdout",
}
KNOWN_COMPUTE_SCOPES = {
    "coverage_only",
    "fdtd_cuda",
    "host_only",
    "mpi_fdtd_cuda",
}
DISPOSITION_COMPUTE_SCOPES = {
    RUN_DISPOSITION: {"fdtd_cuda", "host_only"},
    "compatibility_only": {"host_only"},
    "covered_by_test": {"coverage_only"},
    "deferred_milestone": {"fdtd_cuda", "host_only"},
    "expected_feature_gap": {"fdtd_cuda", "host_only"},
    "external_dependency": {"fdtd_cuda", "host_only"},
    "mpi_only": {"mpi_fdtd_cuda"},
    "notebook_duplicate": {"coverage_only"},
    "oracle_pending": {"fdtd_cuda", "host_only"},
}
KNOWN_COVERAGE_RELATIONS = {
    "feature_union",
    "intended_semantics",
    "same_workflow",
    "strict_superset",
}
COVERED_DISPOSITION = "covered_by_test"
COVERED_PASS_OUTCOME = "PASS_COVERED"
COVERAGE_FAILED_OUTCOME = "COVERAGE_FAILED"
COVERAGE_UNRESOLVED_OUTCOME = "COVERAGE_UNRESOLVED"
UNittest_SKIP_RE = re.compile(
    r"(?:OK\s*\(\s*skipped\s*=\s*|skipped\s*=\s*)(\d+)", re.IGNORECASE
)
UNittest_SKIP_DETAIL_RE = re.compile(
    r"^(?P<test>.+?)\s+\.\.\.\s+skipped\s+"
    r"['\"](?P<reason>.*?)['\"]\s*$"
)
UNittest_VERBOSE_TEST_HEADER_RE = re.compile(
    r"^\S.*\([^()]*\.test[^()]*\)\s*$"
)
UNittest_VERBOSE_TEST_ID_RE = re.compile(
    r"^(?!(?:FAIL|ERROR|UNEXPECTED SUCCESS):\s)"
    r"\S.*\((?P<identity>[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*\.test[A-Za-z0-9_]*)\)"
    r"(?:\s+\.\.\.\s+.*)?$"
)
# MPI launchers may merge the verbose prefixes from several ranks onto one
# physical line before any rank writes its trailing result.  Keep the strict
# whole-line parser above for ordinary output and multiline docstring records,
# but recognize every canonical ``(module.Class.test_method) ...`` fragment
# when such interleaving occurs.  Requiring the ellipsis excludes traceback
# headers such as ``FAIL: test_method (...)`` from being counted as executions.
UNittest_VERBOSE_INLINE_TEST_ID_RE = re.compile(
    r"\((?P<identity>[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*\.test[A-Za-z0-9_]*)\)"
    r"\s+\.\.\."
)
UNittest_RAN_RE = re.compile(
    r"^Ran\s+(?P<count>\d+)\s+tests?\s+in\s+.+$", re.MULTILINE
)


class ManifestError(RuntimeError):
    """Raised for an invalid or incomplete validation manifest."""


def isolated_process_environment() -> dict[str, str]:
    """Return the small ambient environment allowlist used by child probes."""
    return {
        key: value
        for key, value in os.environ.items()
        if key in SAFE_AMBIENT_ENVIRONMENT
    }


def environment_sha256(environment: dict[str, str]) -> str:
    payload = json.dumps(
        environment,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def unique_json_object(
    pairs: list[tuple[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise ManifestError(f"JSON contains duplicate key {key!r}")
        result[key] = item
    return result


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_json_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value


def command_output(command: list[str], cwd: pathlib.Path) -> dict[str, Any]:
    result = bounded_command(
        command,
        cwd=cwd,
        environment=isolated_process_environment(),
        timeout_seconds=20,
        stdout_limit=8 * 1024 * 1024,
        stderr_limit=8 * 1024 * 1024,
    )
    if (
        result["error"] is not None
        or result["timeout"]
        or result["output_limit"] is not None
    ):
        return {
            "command": command,
            "available": False,
            "error": result["error"] or "command exceeded time/output limits",
        }
    return {
        "command": command,
        "available": True,
        "exit_code": result["exit_code"],
        "stdout": result["stdout"].decode("utf-8", errors="replace").strip(),
        "stderr": result["stderr"].decode("utf-8", errors="replace").strip(),
    }


def bounded_text_record(text: str, preview_bytes: int) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    preview = encoded[:preview_bytes].decode("utf-8", errors="ignore")
    return {
        "size_bytes": len(encoded),
        "line_count": len(text.splitlines()),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "preview": preview,
        "preview_bytes": len(preview.encode("utf-8")),
        "truncated": len(encoded) > preview_bytes,
    }


def strict_utf8(value: str, description: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ManifestError(f"{description} is not valid UTF-8") from exc


def python_package_probe(
    python: pathlib.Path,
    repo: pathlib.Path,
    python_path: pathlib.Path | None,
    library_path: pathlib.Path | None = None,
    runtime_parent: pathlib.Path | None = None,
    fontconfig_file: pathlib.Path | None = None,
) -> dict[str, Any]:
    if python_path is None or not python_path.is_dir():
        return {
            "available": False,
            "error": f"Python package path is absent: {python_path}",
        }
    remove_runtime_parent = runtime_parent is None
    if runtime_parent is None:
        runtime_parent = pathlib.Path(
            tempfile.mkdtemp(prefix="gpmeep-package-probe-")
        )
    else:
        runtime_parent.mkdir(parents=True, exist_ok=True)
    runtime_root = runtime_parent / uuid.uuid4().hex
    runtime_root.mkdir(mode=0o700, exist_ok=False)
    runtime_home = runtime_root / "home"
    runtime_cache = runtime_root / "cache"
    runtime_config = runtime_root / "config"
    runtime_matplotlib = runtime_root / "matplotlib"
    for path in (
        runtime_home,
        runtime_cache,
        runtime_config,
        runtime_matplotlib,
    ):
        path.mkdir(mode=0o700, exist_ok=False)

    env = isolated_process_environment()
    env.update(
        {
            "PYTHONPATH": str(python_path),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MEEP_GPU_BACKEND": "cpu",
            "MPLBACKEND": "Agg",
            "HOME": str(runtime_home),
            "XDG_CACHE_HOME": str(runtime_cache),
            "XDG_CONFIG_HOME": str(runtime_config),
            "MPLCONFIGDIR": str(runtime_matplotlib),
            "CUDA_CACHE_DISABLE": "1",
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
        }
    )
    if fontconfig_file is not None:
        env["FONTCONFIG_FILE"] = str(fontconfig_file)
    if library_path and library_path.is_dir():
        env["LD_LIBRARY_PATH"] = str(library_path)
    probe = (
        "import json,meep;"
        "print(json.dumps({"
        "'module':meep.__file__,"
        "'version':getattr(meep,'__version__',None),"
        "'single_precision':bool(meep.is_single_precision()),"
        "'with_mpi':bool(meep.with_mpi()),"
        "'gpu_compiled':bool(meep.gpu.compiled),"
        "'gpu_architectures':meep.gpu.compiled_architectures"
        "},sort_keys=True))"
    )
    probe_stdout = runtime_root / "probe.stdout"
    probe_stderr = runtime_root / "probe.stderr"
    try:
        capture = run_bounded_process(
            [str(python), "-c", probe],
            cwd=repo,
            env=env,
            timeout_seconds=30,
            stdout_file=probe_stdout,
            stderr_file=probe_stderr,
            stdout_limit=1024 * 1024,
            stderr_limit=1024 * 1024,
            combined_limit=2 * 1024 * 1024,
        )
        stdout = probe_stdout.read_text(encoding="utf-8")
        stderr = probe_stderr.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        shutil.rmtree(runtime_root, ignore_errors=True)
        if remove_runtime_parent:
            shutil.rmtree(runtime_parent, ignore_errors=True)
        return {"available": False, "error": str(exc)}
    result: dict[str, Any] = {
        "available": (
            capture["exit_code"] == 0
            and capture["timeout"] is False
            and capture["output_limit"] is None
            and capture["spawn_error"] is None
        ),
        "exit_code": capture["exit_code"],
        "stdout": bounded_text_record(stdout.strip(), 8192),
        "stderr": bounded_text_record(stderr.strip(), 8192),
        "python_path": str(python_path),
        "library_path": str(library_path) if library_path else None,
        "isolated_runtime": True,
        "runtime_environment": {
            "HOME": str(runtime_home),
            "XDG_CACHE_HOME": str(runtime_cache),
            "XDG_CONFIG_HOME": str(runtime_config),
            "MPLCONFIGDIR": str(runtime_matplotlib),
            "FONTCONFIG_FILE": (
                str(fontconfig_file) if fontconfig_file is not None else None
            ),
        },
        "environment_keys": sorted(env),
        "environment_sha256": environment_sha256(env),
        "environment": dict(sorted(env.items())),
        "output_limit": capture["output_limit"],
        "timeout": capture["timeout"],
    }
    if capture["spawn_error"] is not None:
        result["error"] = capture["spawn_error"]
    elif capture["timeout"]:
        result["error"] = "package probe timed out"
    elif capture["output_limit"] is not None:
        result["error"] = "package probe exceeded its output limit"
    if result["available"]:
        parsed = None
        for line in stdout.splitlines():
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "module" in candidate:
                parsed = candidate
                break
        if parsed is None:
            result["available"] = False
            result["error"] = "package probe emitted no package JSON object"
        else:
            result["package"] = parsed
    shutil.rmtree(runtime_root)
    if remove_runtime_parent:
        shutil.rmtree(runtime_parent)
    return result


def collect_provenance(
    repo: pathlib.Path,
    python: pathlib.Path,
    build_python: pathlib.Path,
    install_prefix: pathlib.Path | None,
    manifest_path: pathlib.Path,
    validation_start: dict[str, Any] | None = None,
    probe_runtime_parent: pathlib.Path | None = None,
    probe_fontconfig_file: pathlib.Path | None = None,
) -> dict[str, Any]:
    git_head = command_output([trusted_git(), "rev-parse", "HEAD"], repo)
    git_status = command_output([trusted_git(), "status", "--short"], repo)
    source_candidates = [
        repo / "python" / "meep.i",
        repo / "src" / "gpu_backend.cpp",
        repo / "cuda" / "src" / "runtime.cu",
    ]
    if validation_start is None:
        captured_source_snapshot = source_snapshot(repo)
        discovered_build_extensions = build_extension_paths(build_python)
        captured_build_extensions = [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in discovered_build_extensions
        ]
    else:
        captured_source_snapshot = validation_start["source_snapshot"]
        extension = validation_start["build_extension"]
        captured_build_extensions = (
            [
                {
                    "path": extension["path"],
                    "size_bytes": extension["size_bytes"],
                    "sha256": extension["sha256"],
                }
            ]
            if extension.get("available")
            else []
        )
    install_extension_paths: list[pathlib.Path] = []
    install_python_paths: list[pathlib.Path] = []
    if install_prefix:
        install_python_paths = sorted(
            path.resolve()
            for path in install_prefix.glob("lib/python*/site-packages")
            if path.is_dir()
        )
        install_extension_paths = sorted(
            path.resolve()
            for pattern in (
                "lib/python*/site-packages/meep/_meep*.so",
                "lib/python*/site-packages/meep/_meep*.dylib",
            )
            for path in install_prefix.glob(pattern)
        )
    return {
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": str(python.resolve()),
        "python_version": command_output(
            [str(python), "-c", "import sys; print(sys.version)"], repo
        ),
        "repository": str(repo),
        "git_head": git_head,
        "git_status": git_status,
        "source_snapshot": captured_source_snapshot,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "build_python": str(build_python.resolve()),
        "build_python_exists": build_python.is_dir(),
        "build_extensions": captured_build_extensions,
        "build_package_probe": python_package_probe(
            python,
            repo,
            build_python,
            None,
            probe_runtime_parent,
            probe_fontconfig_file,
        ),
        "install_prefix": str(install_prefix.resolve()) if install_prefix else None,
        "install_extensions": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in install_extension_paths
        ],
        "install_package_probe": python_package_probe(
            python,
            repo,
            install_python_paths[0] if install_python_paths else None,
            install_prefix / "lib" if install_prefix else None,
            probe_runtime_parent,
            probe_fontconfig_file,
        ),
        "source_files": [
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": file_sha256(path) if path.is_file() else None,
            }
            for path in source_candidates
        ],
        "environment_selection": {
            name: os.environ.get(name)
            for name in (
                "CONDA_PREFIX",
                "CUDA_DEVICE_ORDER",
                "CUDA_VISIBLE_DEVICES",
                "MEEP_GPU_DEVICE",
            )
        },
    }


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_member_path(path: pathlib.Path, evidence_root: pathlib.Path) -> str:
    resolved_root = evidence_root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ProvenanceError(
            f"evidence file escaped the evidence root: {resolved_path}"
        ) from exc
    if path.is_symlink() or not resolved_path.is_file():
        raise ProvenanceError(
            f"evidence member is not a regular non-symlink file: {resolved_path}"
        )
    return relative.as_posix()


def evidence_file_record(
    path: pathlib.Path, evidence_root: pathlib.Path
) -> dict[str, Any]:
    member = evidence_member_path(path, evidence_root)
    stat = path.stat()
    return {
        "path": member,
        "size_bytes": stat.st_size,
        "sha256": file_sha256(path),
    }


def resolve_evidence_member(
    evidence_root: pathlib.Path, member: str
) -> pathlib.Path:
    if not isinstance(member, str) or not member:
        raise ProvenanceError("evidence member path must be a nonempty string")
    candidate = pathlib.PurePosixPath(member)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ProvenanceError(f"unsafe evidence member path: {member!r}")
    root = evidence_root.resolve()
    resolved = (root / pathlib.Path(*candidate.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(f"evidence member escaped root: {member!r}") from exc
    return resolved


def build_extension_paths(build_python: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        [
            path.resolve()
            for pattern in (
                "meep/.libs/_meep*.so",
                "meep/_meep*.so",
                "meep/.libs/_meep*.dylib",
                "meep/_meep*.dylib",
            )
            for path in build_python.glob(pattern)
            if path.is_file()
        ]
    )


def build_extension_snapshot(build_python: pathlib.Path) -> dict[str, Any]:
    try:
        paths = build_extension_paths(build_python)
    except OSError as exc:
        return {"available": False, "error": str(exc), "candidates": []}
    if len(paths) != 1:
        return {
            "available": False,
            "error": (
                "expected exactly one built Meep extension, found "
                f"{len(paths)} under {build_python / 'meep'}"
            ),
            "candidates": [str(path) for path in paths],
        }
    path = paths[0]
    try:
        return {
            "available": True,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    except OSError as exc:
        return {
            "available": False,
            "error": f"cannot snapshot built extension {path}: {exc}",
            "candidates": [str(path)],
        }


def _recorded_path(value: str, repo: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def validate_authoritative_build_contract(
    receipt: dict[str, Any], repo: pathlib.Path, build_python: pathlib.Path
) -> None:
    try:
        validate_v2_receipt(receipt, repo)
    except QualificationContractError as exc:
        raise ProvenanceError(f"qualification-v2 contract failed: {exc}") from exc
    configuration = receipt.get("configuration")
    if not isinstance(configuration, dict):
        raise ProvenanceError("build receipt configuration is invalid")
    if (
        configuration.get("qualification_contract")
        != EXPECTED_QUALIFICATION_CONTRACT
    ):
        raise ProvenanceError("build receipt has the wrong qualification contract")

    builder = configuration.get("builder")
    expected_builder = (repo / EXPECTED_BUILDER_RELATIVE_PATH).resolve()
    if (
        not isinstance(builder, dict)
        or not isinstance(builder.get("path"), str)
        or _recorded_path(builder["path"], repo) != expected_builder
    ):
        raise ProvenanceError(
            "build receipt was not produced by the authoritative MPI builder"
        )

    configure_argv = configuration.get("configure_argv")
    if not isinstance(configure_argv, list) or not all(
        isinstance(value, str) for value in configure_argv
    ):
        raise ProvenanceError("build receipt configure argument list is invalid")
    cuda_arch_args = [
        value
        for value in configure_argv
        if value.startswith("--with-cuda-arch=")
    ]
    prefix_args = [
        value for value in configure_argv if value.startswith("--prefix=")
    ]
    if len(cuda_arch_args) != 1 or len(prefix_args) != 1:
        raise ProvenanceError(
            "build receipt CUDA architecture/install prefix is ambiguous"
        )
    cuda_arch = cuda_arch_args[0]
    if re.fullmatch(r"--with-cuda-arch=[A-Za-z0-9_.+,;-]+", cuda_arch) is None:
        raise ProvenanceError("build receipt CUDA architecture is malformed")
    expected_install = (
        repo / "install" / "meep-cuda-mpi-python-fp32"
    ).resolve()
    if pathlib.Path(prefix_args[0].split("=", 1)[1]).resolve() != expected_install:
        raise ProvenanceError(
            "build receipt install prefix is not the authoritative MPI prefix"
        )
    expected_configure = [
        "--enable-maintainer-mode",
        "--enable-shared",
        "--enable-single",
        "--enable-cuda",
        cuda_arch,
        "--with-openmp",
        "--with-mpi",
        "--with-python",
        "--without-scheme",
        f"--prefix={expected_install}",
        "--disable-cuda-fast-math",
    ]
    if configure_argv != expected_configure:
        raise ProvenanceError(
            "build receipt configure arguments are not the exact "
            "CUDA/MPI/Python FP32 contract"
        )

    configuration_files = receipt.get("configuration_files")
    manifests = receipt.get("manifests")
    if not isinstance(configuration_files, dict) or not isinstance(
        manifests, dict
    ):
        raise ProvenanceError(
            "build receipt cache/prefix qualification closure is incomplete"
        )
    fontconfig_record = configuration_files.get("qualification_fontconfig")
    installed_environment = manifests.get("installed_environment")
    build_home = manifests.get("build_home")
    qualification_home = manifests.get("qualification_home")
    if (
        not isinstance(fontconfig_record, dict)
        or not isinstance(fontconfig_record.get("path"), str)
        or not isinstance(installed_environment, dict)
        or not isinstance(installed_environment.get("root"), str)
        or not isinstance(build_home, dict)
        or not isinstance(build_home.get("root"), str)
        or not isinstance(qualification_home, dict)
        or not isinstance(qualification_home.get("root"), str)
    ):
        raise ProvenanceError(
            "build receipt cache/prefix qualification closure is incomplete"
        )

    expected_fontconfig = (
        build_python.parent / "qualification-fontconfig.conf"
    ).resolve()
    if _recorded_path(fontconfig_record["path"], repo) != expected_fontconfig:
        raise ProvenanceError(
            "build receipt qualification Fontconfig path is not authoritative"
        )
    expected_environment = (repo / ".envs" / "meep-gpu-cuda-mpi").resolve()
    if (
        _recorded_path(installed_environment["root"], repo)
        != expected_environment
    ):
        raise ProvenanceError(
            "build receipt installed-environment manifest path is not authoritative"
        )
    build_home_path = _recorded_path(build_home["root"], repo)
    expected_build_home_parent = (repo / ".micromamba" / "cache").resolve()
    if (
        build_home_path.parent != expected_build_home_parent
        or not build_home_path.name.startswith("build-home-")
    ):
        raise ProvenanceError(
            "build receipt build-home manifest path is not authoritative"
        )
    expected_qualification_home = (
        build_python.parent / "qualification-home"
    ).resolve()
    if (
        _recorded_path(qualification_home["root"], repo)
        != expected_qualification_home
    ):
        raise ProvenanceError(
            "build receipt qualification-home manifest path is not authoritative"
        )

    try:
        fontconfig_root = ET.parse(expected_fontconfig).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ProvenanceError(
            f"qualification Fontconfig is invalid: {exc}"
        ) from exc
    if fontconfig_root.tag != "fontconfig" or fontconfig_root.attrib:
        raise ProvenanceError("qualification Fontconfig root is invalid")
    children = list(fontconfig_root)
    if [item.tag for item in children] != ["dir", "include", "cachedir"]:
        raise ProvenanceError(
            "qualification Fontconfig structure is not authoritative"
        )
    expected_attributes = (
        {},
        {"ignore_missing": "yes"},
        {"prefix": "xdg"},
    )
    if any(
        item.attrib != attributes or list(item)
        for item, attributes in zip(children, expected_attributes)
    ):
        raise ProvenanceError(
            "qualification Fontconfig attributes are not authoritative"
        )
    font_dirs = [
        (item.text or "").strip() for item in fontconfig_root.findall("dir")
    ]
    includes = [
        (item.text or "").strip()
        for item in fontconfig_root.findall("include")
    ]
    cache_dirs = [
        (item.text or "").strip()
        for item in fontconfig_root.findall("cachedir")
    ]
    if font_dirs != [str(expected_environment / "fonts")]:
        raise ProvenanceError(
            "qualification Fontconfig font directory is not authoritative"
        )
    if includes != [str(expected_environment / "etc" / "fonts" / "conf.d")]:
        raise ProvenanceError(
            "qualification Fontconfig include directory is not authoritative"
        )
    if cache_dirs != ["fontconfig"]:
        raise ProvenanceError(
            "qualification Fontconfig cache directory is not XDG-routed"
        )


def validation_snapshot(
    repo: pathlib.Path, build_python: pathlib.Path, python: pathlib.Path
) -> dict[str, Any]:
    problems: list[str] = []
    try:
        source = source_snapshot(repo)
    except (OSError, RuntimeError, ValueError) as exc:
        source = {"available": False, "error": str(exc)}
    if not source.get("available"):
        problems.append(
            "source snapshot unavailable: "
            + str(source.get("error", "unknown error"))
        )
    extension = build_extension_snapshot(build_python)
    if not extension.get("available"):
        problems.append(
            "build extension snapshot unavailable: "
            + str(extension.get("error", "unknown error"))
        )
    receipt_path = build_python.parent / "build-provenance.json"
    receipt_evidence: dict[str, Any]
    runtime_contract: dict[str, Any] = {}
    try:
        receipt = verify_build_receipt(receipt_path, repo)
        if receipt.get("build_kind") != EXPECTED_BUILD_KIND:
            raise ProvenanceError("build receipt has the wrong build_kind")
        validate_authoritative_build_contract(receipt, repo, build_python)
        artifacts = receipt.get("artifacts", {})
        configuration_files = receipt.get("configuration_files", {})
        manifests = receipt.get("manifests", {})
        extension_record = artifacts.get("python_extension")
        libmeep_record = artifacts.get("libmeep")
        fontconfig_record = configuration_files.get("qualification_fontconfig")
        build_home_record = manifests.get("build_home")
        qualification_home_record = manifests.get("qualification_home")
        installed_environment_record = manifests.get("installed_environment")
        if not isinstance(extension_record, dict) or not isinstance(
            libmeep_record, dict
        ):
            raise ProvenanceError(
                "build receipt lacks python_extension/libmeep artifacts"
            )
        if not all(
            isinstance(record, dict)
            for record in (
                fontconfig_record,
                build_home_record,
                qualification_home_record,
                installed_environment_record,
            )
        ):
            raise ProvenanceError(
                "build receipt lacks authoritative runtime isolation paths"
            )
        receipt_extension_path = _recorded_path(
            str(extension_record.get("path", "")), repo
        )
        if (
            not extension.get("available")
            or receipt_extension_path != pathlib.Path(extension["path"]).resolve()
            or extension_record.get("sha256") != extension.get("sha256")
        ):
            raise ProvenanceError(
                "build extension snapshot differs from the build receipt"
            )
        meep_module = build_python / "meep" / "__init__.py"
        python_record = provenance_file_record(python.resolve(), repo)
        python_record["path"] = str(python.resolve())
        meep_module_record = provenance_file_record(meep_module.resolve(), repo)
        meep_module_record["path"] = str(meep_module.resolve())
        runtime_contract = {
            "python_executable": python_record,
            "meep_module": meep_module_record,
            "extension": {
                **extension_record,
                "path": str(receipt_extension_path),
            },
            "libmeep": {
                **libmeep_record,
                "path": str(
                    _recorded_path(str(libmeep_record.get("path", "")), repo)
                ),
            },
            "fontconfig_file": {
                **fontconfig_record,
                "path": str(
                    _recorded_path(str(fontconfig_record.get("path", "")), repo)
                ),
            },
            "build_home": str(
                _recorded_path(str(build_home_record.get("root", "")), repo)
            ),
            "qualification_home": str(
                _recorded_path(
                    str(qualification_home_record.get("root", "")), repo
                )
            ),
            "installed_environment": str(
                _recorded_path(
                    str(installed_environment_record.get("root", "")), repo
                )
            ),
            "receipt_id": receipt["receipt_id"],
        }
        receipt_evidence = {
            "available": True,
            "path": str(receipt_path.resolve()),
            "sha256": provenance_sha256_file(receipt_path),
            "receipt_id": receipt["receipt_id"],
            "build_input_id": receipt["build_input_id"],
            "artifact_set_id": receipt["artifact_set_id"],
        }
    except (OSError, ProvenanceError, KeyError, TypeError, ValueError) as exc:
        receipt_evidence = {
            "available": False,
            "path": str(receipt_path.resolve()),
            "error": str(exc),
        }
        problems.append(f"build receipt unavailable or invalid: {exc}")
    return {
        "captured_at_utc": utc_now(),
        "available": not problems,
        "source_snapshot": source,
        "build_extension": extension,
        "build_receipt": receipt_evidence,
        "runtime_contract": runtime_contract,
        "problems": problems,
    }


def compare_validation_snapshots(
    start: dict[str, Any], end: dict[str, Any]
) -> tuple[bool, list[str]]:
    problems = [f"start: {item}" for item in start.get("problems", [])]
    problems.extend(f"end: {item}" for item in end.get("problems", []))
    start_source = start.get("source_snapshot", {})
    end_source = end.get("source_snapshot", {})
    if (
        start_source.get("available")
        and end_source.get("available")
        and start_source != end_source
    ):
        problems.append("source tree changed during validation")
    start_extension = start.get("build_extension", {})
    end_extension = end.get("build_extension", {})
    if (
        start_extension.get("available")
        and end_extension.get("available")
        and start_extension != end_extension
    ):
        problems.append("built extension changed during validation")
    start_receipt = start.get("build_receipt", {})
    end_receipt = end.get("build_receipt", {})
    if (
        start_receipt.get("available")
        and end_receipt.get("available")
        and start_receipt != end_receipt
    ):
        problems.append("build receipt changed during validation")
    start_runtime = start.get("runtime_contract", {})
    end_runtime = end.get("runtime_contract", {})
    if start_runtime and end_runtime and start_runtime != end_runtime:
        problems.append("runtime artifact contract changed during validation")
    return not problems, problems


def prepare_output(output: pathlib.Path) -> None:
    if output.is_symlink():
        raise ManifestError("evidence output must not be a symbolic link")
    if output.exists():
        if not output.is_dir():
            raise ManifestError("evidence output exists and is not a directory")
        if any(output.iterdir()):
            raise ManifestError(
                "evidence output must be a new or empty directory; refusing "
                "to reuse files from an earlier validation"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)


def atomic_write_text(path: pathlib.Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        encoded = value.encode("utf-8")
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        try:
            linkat = ctypes.CDLL(None, use_errno=True).linkat
        except AttributeError as exc:
            raise OSError(errno.ENOSYS, "linkat is unavailable") from exc
        linkat.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        )
        linkat.restype = ctypes.c_int
        # Link the exact open inode, not its raceable temporary pathname, and
        # never replace a concurrently-created final evidence file.
        if linkat(descriptor, b"", -100, os.fsencode(path), 0x1000) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def source_snapshot(repo: pathlib.Path) -> dict[str, Any]:
    """Return the build receipt's canonical source identity in compact form."""
    try:
        snapshot = provenance_source_snapshot(repo)
    except (OSError, RuntimeError, ProvenanceError) as error:
        return {"available": False, "error": str(error)}
    return {
        "available": True,
        "algorithm": snapshot["algorithm"],
        "file_count": snapshot["file_count"],
        "sha256": snapshot["sha256"],
    }


def merge_case(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key != "glob":
            result[key] = copy.deepcopy(value)
    return result


def case_directory_id(case_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._")
    if not slug:
        slug = "case"
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return f"{slug[:80]}-{digest}"


def materialize_cases(manifest: dict[str, Any], repo: pathlib.Path) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"manifest schema_version must be {SCHEMA_VERSION}, "
            f"got {manifest.get('schema_version')!r}"
        )
    rules = manifest.get("inventory")
    overrides = manifest.get("overrides", [])
    if not isinstance(rules, list) or not rules:
        raise ManifestError("manifest inventory must be a nonempty array")
    if not isinstance(overrides, list):
        raise ManifestError("manifest overrides must be an array")

    by_path: dict[str, dict[str, Any]] = {}
    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, dict) or not isinstance(rule.get("glob"), str):
            raise ManifestError(f"inventory rule {rule_index} needs a string glob")
        matches = sorted(glob.glob(str(repo / rule["glob"]), recursive=True))
        if not matches:
            raise ManifestError(f"inventory glob matched no files: {rule['glob']}")
        for raw_path in matches:
            path = pathlib.Path(raw_path)
            if not path.is_file():
                continue
            relative = path.resolve().relative_to(repo.resolve()).as_posix()
            if relative in by_path:
                raise ManifestError(f"inventory rules overlap for {relative}")
            case = merge_case(rule, {"path": relative})
            case.pop("glob", None)
            case["id"] = case.get("id", relative)
            case["_inventory_rule"] = rule_index
            by_path[relative] = case

    matched_overrides: Counter[int] = Counter()
    for override_index, override in enumerate(overrides):
        if not isinstance(override, dict) or not isinstance(override.get("glob"), str):
            raise ManifestError(f"override {override_index} needs a string glob")
        pattern = override["glob"]
        for relative, case in list(by_path.items()):
            if pathlib.PurePath(relative).match(pattern):
                by_path[relative] = merge_case(case, override)
                matched_overrides[override_index] += 1
        if matched_overrides[override_index] == 0:
            raise ManifestError(f"override glob matched no inventory case: {pattern}")

    cases = sorted(by_path.values(), key=lambda item: item["path"])
    seen_ids: set[str] = set()
    seen_directories: dict[str, str] = {}
    for case in cases:
        validate_case(case, repo)
        if case["id"] in seen_ids:
            raise ManifestError(f"duplicate case id: {case['id']}")
        seen_ids.add(case["id"])
        directory = case_directory_id(case["id"])
        previous = seen_directories.get(directory)
        if previous is not None and previous != case["id"]:
            raise ManifestError(
                f"case directory collision between {previous!r} and "
                f"{case['id']!r}"
            )
        seen_directories[directory] = case["id"]
    validate_coverage_graph(cases)
    return cases


def validate_coverage_graph(cases: list[dict[str, Any]]) -> None:
    """Validate coverage references against the complete materialized catalog."""
    by_id = {case["id"]: case for case in cases}

    # Detect cycles independently of the terminal-target rule so a cycle is
    # diagnosed as such rather than depending on materialization order.
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(case_id: str, trail: list[str]) -> None:
        if case_id in visiting:
            cycle_start = trail.index(case_id)
            cycle = trail[cycle_start:] + [case_id]
            raise ManifestError(
                "coverage cycle: " + " -> ".join(cycle)
            )
        if case_id in visited:
            return
        visiting.add(case_id)
        trail.append(case_id)
        case = by_id[case_id]
        for target_id in case.get("covered_by", []):
            if target_id in by_id:
                visit(target_id, trail)
        trail.pop()
        visiting.remove(case_id)
        visited.add(case_id)

    for case_id in by_id:
        visit(case_id, [])

    for case in cases:
        if case["disposition"] != COVERED_DISPOSITION:
            continue
        for target_id in case["covered_by"]:
            target = by_id.get(target_id)
            if target is None:
                raise ManifestError(
                    f"{case['path']}: covered_by target does not exist: "
                    f"{target_id!r}"
                )
            if target["disposition"] != RUN_DISPOSITION:
                raise ManifestError(
                    f"{case['path']}: covered_by target must be terminal "
                    f"runnable case: {target_id!r}"
                )


def validate_case(
    case: dict[str, Any],
    repo: pathlib.Path,
    *,
    require_source_file: bool = True,
) -> None:
    path = case.get("path")
    kind = case.get("kind")
    disposition = case.get("disposition")
    if not isinstance(path, str) or (
        require_source_file and not (repo / path).is_file()
    ):
        raise ManifestError(f"case path does not exist: {path!r}")
    if kind not in KNOWN_KINDS:
        raise ManifestError(f"{path}: unknown kind {kind!r}")
    if disposition not in KNOWN_DISPOSITIONS:
        raise ManifestError(f"{path}: unknown disposition {disposition!r}")
    compute_scope = case.get("compute_scope")
    if compute_scope not in KNOWN_COMPUTE_SCOPES:
        raise ManifestError(
            f"{path}: compute_scope must be one of "
            f"{', '.join(sorted(KNOWN_COMPUTE_SCOPES))}"
        )
    allowed_compute_scopes = DISPOSITION_COMPUTE_SCOPES[disposition]
    if compute_scope not in allowed_compute_scopes:
        raise ManifestError(
            f"{path}: disposition {disposition!r} requires compute_scope in "
            f"{', '.join(sorted(allowed_compute_scopes))}"
        )
    if not isinstance(case.get("reason"), str) or not case["reason"].strip():
        raise ManifestError(f"{path}: every case needs a nonempty reason")
    tier = case.get("tier")
    if not isinstance(tier, list) or not tier or not all(isinstance(x, str) for x in tier):
        raise ManifestError(f"{path}: tier must be a nonempty string array")
    covered_by = case.get("covered_by")
    coverage_relation = case.get("coverage_relation")
    if disposition == COVERED_DISPOSITION:
        if compute_scope != "coverage_only":
            raise ManifestError(
                f"{path}: covered_by_test requires compute_scope coverage_only"
            )
        if (
            not isinstance(covered_by, list)
            or not covered_by
            or not all(isinstance(value, str) and value for value in covered_by)
        ):
            raise ManifestError(
                f"{path}: covered_by_test requires a nonempty covered_by ID array"
            )
        if len(set(covered_by)) != len(covered_by):
            raise ManifestError(f"{path}: covered_by target IDs must be unique")
        if case.get("id") in covered_by:
            raise ManifestError(f"{path}: covered_by cannot reference itself")
        if coverage_relation not in KNOWN_COVERAGE_RELATIONS:
            raise ManifestError(
                f"{path}: covered_by_test requires a known coverage_relation"
            )
    elif covered_by is not None or coverage_relation is not None:
        raise ManifestError(
            f"{path}: covered_by and coverage_relation are only valid for "
            "covered_by_test"
        )
    if disposition == RUN_DISPOSITION:
        comparison = case.get("comparison", {})
        if comparison.get("mode") not in KNOWN_COMPARISONS:
            raise ManifestError(f"{path}: runnable case needs a known comparison mode")
        timeout = case.get("timeout_seconds")
        if not isinstance(timeout, int) or timeout <= 0:
            raise ManifestError(f"{path}: runnable case needs positive timeout_seconds")
        gpu_contract = case.get("gpu_contract")
        if gpu_contract not in ("none", "cuda_dispatch"):
            raise ManifestError(f"{path}: gpu_contract must be none or cuda_dispatch")
        if compute_scope == "host_only" and gpu_contract != "none":
            raise ManifestError(
                f"{path}: host_only runnable case requires gpu_contract none"
            )
        required_cuda_call_counters = case.get(
            "required_cuda_call_counters", []
        )
        if (
            not isinstance(required_cuda_call_counters, list)
            or not all(
                isinstance(counter, str)
                and counter in REQUIRED_CUDA_ACTIVITY_COUNTERS
                for counter in required_cuda_call_counters
            )
            or len(set(required_cuda_call_counters))
            != len(required_cuda_call_counters)
        ):
            raise ManifestError(
                f"{path}: required_cuda_call_counters must be a unique "
                "array of registered CUDA activity counters"
            )
        if required_cuda_call_counters and gpu_contract != "cuda_dispatch":
            raise ManifestError(
                f"{path}: required CUDA call counters need cuda_dispatch"
            )
        mpi_aggregate_cuda_call_counters = case.get(
            "mpi_aggregate_cuda_call_counters", []
        )
        if (
            not isinstance(mpi_aggregate_cuda_call_counters, list)
            or not all(
                isinstance(counter, str)
                and counter in required_cuda_call_counters
                for counter in mpi_aggregate_cuda_call_counters
            )
            or len(set(mpi_aggregate_cuda_call_counters))
            != len(mpi_aggregate_cuda_call_counters)
        ):
            raise ManifestError(
                f"{path}: mpi_aggregate_cuda_call_counters must be a unique "
                "subset of required_cuda_call_counters"
            )
        mpi_required_per_rank_call_counters = case.get(
            "mpi_required_per_rank_call_counters", []
        )
        if (
            not isinstance(mpi_required_per_rank_call_counters, list)
            or not all(
                isinstance(counter, str)
                and counter in REQUIRED_CUDA_SHARED_ACTIVITY_COUNTERS
                for counter in mpi_required_per_rank_call_counters
            )
            or len(set(mpi_required_per_rank_call_counters))
            != len(mpi_required_per_rank_call_counters)
        ):
            raise ManifestError(
                f"{path}: mpi_required_per_rank_call_counters must be a "
                "unique array of registered shared activity counters"
            )
        if (
            mpi_aggregate_cuda_call_counters
            and not mpi_required_per_rank_call_counters
        ):
            raise ManifestError(
                f"{path}: aggregate MPI CUDA activity requires explicit "
                "per-rank shared activity counters"
            )
        if (
            mpi_aggregate_cuda_call_counters
            or mpi_required_per_rank_call_counters
        ) and gpu_contract != "cuda_dispatch":
            raise ManifestError(
                f"{path}: MPI activity counters need cuda_dispatch"
            )
        require_active_backend_at_exit = case.get(
            "require_active_backend_at_exit", True
        )
        if not isinstance(require_active_backend_at_exit, bool):
            raise ManifestError(
                f"{path}: require_active_backend_at_exit must be a boolean"
            )
        allowed_skips = case.get("allowed_unittest_skips", 0)
        if (
            not isinstance(allowed_skips, int)
            or isinstance(allowed_skips, bool)
            or allowed_skips < 0
        ):
            raise ManifestError(
                f"{path}: allowed_unittest_skips must be a nonnegative integer"
            )
        allowed_skip_details = case.get(
            "allowed_unittest_skip_details", []
        )
        if not isinstance(allowed_skip_details, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("test"), str)
            and item["test"]
            and isinstance(item.get("reason"), str)
            and item["reason"]
            for item in allowed_skip_details
        ):
            raise ManifestError(
                f"{path}: allowed_unittest_skip_details must contain "
                "nonempty test/reason patterns"
            )
        if allowed_skips != len(allowed_skip_details):
            raise ManifestError(
                f"{path}: every allowed unittest skip needs an identity "
                "and reason pattern"
            )
        expected_unittest = case.get("expected_unittest")
        if expected_unittest is not None:
            if kind != "unittest" or not isinstance(expected_unittest, dict):
                raise ManifestError(
                    f"{path}: expected_unittest is only valid for unittest "
                    "cases and must be an object"
                )
            if set(expected_unittest) != {"count", "identities"}:
                raise ManifestError(
                    f"{path}: expected_unittest needs exactly count and identities"
                )
            expected_count = expected_unittest["count"]
            expected_identities = expected_unittest["identities"]
            if (
                not isinstance(expected_count, int)
                or isinstance(expected_count, bool)
                or expected_count <= 0
            ):
                raise ManifestError(
                    f"{path}: expected_unittest count must be a positive integer"
                )
            if not isinstance(expected_identities, list) or not all(
                isinstance(identity, str) and identity
                for identity in expected_identities
            ):
                raise ManifestError(
                    f"{path}: expected_unittest identities must be nonempty strings"
                )
            if len(set(expected_identities)) != len(expected_identities):
                raise ManifestError(
                    f"{path}: expected_unittest identities must be unique"
                )
            if expected_count != len(expected_identities):
                raise ManifestError(
                    f"{path}: expected_unittest count must equal the number of "
                    "identities"
                )
        if (
            not require_active_backend_at_exit
            and gpu_contract == "none"
            and (kind != "unittest" or expected_unittest is None)
        ):
            raise ManifestError(
                f"{path}: releasing the backend at exit with no dispatch "
                "contract requires pinned expected_unittest identities"
            )
        if kind == "unittest" and expected_unittest is None:
            raise ManifestError(
                f"{path}: every runnable unittest requires pinned "
                "expected_unittest identities"
            )
        if kind == "example" and comparison.get("mode") == "embedded_oracle":
            raise ManifestError(
                f"{path}: an example cannot claim an embedded oracle; "
                "provide normalized_stdout metrics or classify oracle_pending"
            )
        if comparison.get("mode") == "json_metrics":
            prefix = comparison.get("prefix")
            if not isinstance(prefix, str) or not prefix:
                raise ManifestError(
                    f"{path}: json_metrics needs a nonempty prefix"
                )
            tolerances = comparison.get("tolerances")
            if not isinstance(tolerances, dict) or not tolerances:
                raise ManifestError(
                    f"{path}: json_metrics needs a nonempty tolerances object"
                )
            if len(tolerances) > MAX_JSON_TOLERANCE_RULES:
                raise ManifestError(
                    f"{path}: json_metrics has {len(tolerances)} tolerance "
                    f"rules; limit is {MAX_JSON_TOLERANCE_RULES}"
                )
            tolerance_rule_bytes = sum(
                len(metric.encode("utf-8"))
                for metric in tolerances
                if isinstance(metric, str)
            )
            if tolerance_rule_bytes > MAX_JSON_TOLERANCE_RULE_BYTES_TOTAL:
                raise ManifestError(
                    f"{path}: json_metrics tolerance rule names exceed the "
                    f"{MAX_JSON_TOLERANCE_RULE_BYTES_TOTAL}-byte budget"
                )
            for metric, tolerance in tolerances.items():
                if not isinstance(metric, str) or not metric or not isinstance(
                    tolerance, dict
                ):
                    raise ManifestError(
                        f"{path}: invalid json_metrics tolerance"
                    )
                if len(metric.encode("utf-8")) > MAX_JSON_METRIC_PATH_BYTES:
                    raise ManifestError(
                        f"{path}: json_metrics tolerance rule exceeds the "
                        f"{MAX_JSON_METRIC_PATH_BYTES}-byte path limit"
                    )
                unknown_fields = set(tolerance) - {
                    "atol",
                    "rtol",
                    "diagnostic_only",
                }
                if unknown_fields:
                    raise ManifestError(
                        f"{path}: {metric} has unknown tolerance fields: "
                        + ", ".join(sorted(unknown_fields))
                    )
                for name in ("atol", "rtol"):
                    value = tolerance.get(name, 0.0)
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        raise ManifestError(
                            f"{path}: {metric} {name} must be finite and "
                            "nonnegative"
                        )
                if not isinstance(
                    tolerance.get("diagnostic_only", False), bool
                ):
                    raise ManifestError(
                        f"{path}: {metric} diagnostic_only must be boolean"
                    )
            if all(
                tolerance.get("diagnostic_only", False)
                for tolerance in tolerances.values()
            ):
                raise ManifestError(
                    f"{path}: json_metrics needs at least one "
                    "verdict-bearing tolerance"
                )
    elif not case.get("milestone") and disposition in {
        "deferred_milestone",
        "expected_feature_gap",
        "oracle_pending",
    }:
        raise ManifestError(f"{path}: {disposition} needs a milestone")


def dependency_available(name: str, python: pathlib.Path, repo: pathlib.Path) -> bool:
    probe = (
        "import importlib.util,sys;"
        f"sys.exit(0 if importlib.util.find_spec({name!r}) else 1)"
    )
    result = bounded_command(
        [str(python), "-c", probe],
        cwd=repo,
        environment=isolated_process_environment(),
        timeout_seconds=20,
        stdout_limit=64 * 1024,
        stderr_limit=64 * 1024,
    )
    return (
        result["exit_code"] == 0
        and result["error"] is None
        and not result["timeout"]
        and result["output_limit"] is None
    )


def classify_inventory_case(case: dict[str, Any], selected: bool) -> dict[str, Any]:
    if not selected:
        outcome = "NOT_SELECTED"
    else:
        outcome = case["disposition"].upper()
    return {
        "id": case["id"],
        "path": case["path"],
        "kind": case["kind"],
        "tier": case["tier"],
        "disposition": case["disposition"],
        "compute_scope": case["compute_scope"],
        "reason": case["reason"],
        "milestone": case.get("milestone"),
        "selected": selected,
        "outcome": outcome,
        "runs": {},
        "comparison": None,
    }


def resolve_coverage_outcomes(
    cases: Iterable[dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    """Derive selected covered-case outcomes from all terminal target results."""
    case_by_id = {case["id"]: case for case in cases}
    result_by_id = {result["id"]: result for result in results}
    for result in results:
        if not result["selected"]:
            continue
        case = case_by_id[result["id"]]
        if case["disposition"] != COVERED_DISPOSITION:
            continue
        targets = [result_by_id[target_id] for target_id in case["covered_by"]]
        if any(not target["selected"] for target in targets):
            result["outcome"] = COVERAGE_UNRESOLVED_OUTCOME
        elif all(target["outcome"] == "PASS" for target in targets):
            result["outcome"] = COVERED_PASS_OUTCOME
        else:
            result["outcome"] = COVERAGE_FAILED_OUTCOME


def case_command(case: dict[str, Any], python: pathlib.Path, repo: pathlib.Path) -> list[str]:
    if "command" in case:
        command = case["command"]
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise ManifestError(f"{case['path']}: command must be a string array")
        case_path = str((repo / case["path"]).resolve())
        return [
            value.replace("{python}", str(python))
            .replace("{repo}", str(repo))
            .replace("{path}", case_path)
            for value in command
        ]
    if case["kind"] == "unittest":
        test_path = (repo / case["path"]).resolve()
        return [
            str(python),
            "-m",
            "unittest",
            "discover",
            "-v",
            "-s",
            str(test_path.parent),
            "-p",
            test_path.name,
        ]
    if case["kind"] == "example":
        return [str(python), str((repo / case["path"]).resolve())]
    if case["kind"] == "notebook":
        return [
            str(python),
            "-m",
            "jupyter",
            "nbconvert",
            "--execute",
            "--to",
            "notebook",
            "--stdout",
            str((repo / case["path"]).resolve()),
        ]
    raise ManifestError(f"{case['path']}: cannot form command")


def read_stats(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        return {"capture_status": "missing"}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"capture_status": "invalid", "error": str(exc)}
    if not isinstance(result, dict):
        return {"capture_status": "invalid", "error": "stats root is not an object"}
    return result


def parse_unittest_skips(text: str) -> int:
    matches = [int(value) for value in UNittest_SKIP_RE.findall(text)]
    return max(matches, default=0)


def parse_unittest_skip_details(text: str) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = UNittest_SKIP_DETAIL_RE.match(line)
        if match is None:
            continue
        test = match.group("test").strip()
        # unittest's verbose renderer prints a test method's docstring on the
        # result line and the stable method identity on the preceding line:
        #
        # test_method (module.Case.test_method)
        # Human-readable docstring ... skipped 'reason'
        #
        # Prefer that preceding identity when present so exact skip policies
        # cannot be accidentally authorized by mutable prose.
        if (
            index > 0
            and UNittest_VERBOSE_TEST_HEADER_RE.match(lines[index - 1])
        ):
            test = lines[index - 1].strip()
        details.append({"test": test, "reason": match.group("reason")})
    return details


def parse_unittest_test_identities(text: str) -> list[str]:
    identities: list[str] = []
    for line in text.splitlines():
        inline_matches = list(UNittest_VERBOSE_INLINE_TEST_ID_RE.finditer(line))
        if inline_matches:
            identities.extend(match.group("identity") for match in inline_matches)
            continue
        match = UNittest_VERBOSE_TEST_ID_RE.match(line)
        if match is not None:
            identities.append(match.group("identity"))
    return identities


def parse_unittest_reported_counts(text: str) -> list[int]:
    return [int(match.group("count")) for match in UNittest_RAN_RE.finditer(text)]


def verify_unittest_test_contract(
    case: dict[str, Any], text: str
) -> tuple[bool, dict[str, Any], list[str]]:
    reported_counts = parse_unittest_reported_counts(text)
    identities = parse_unittest_test_identities(text)
    expected = case.get("expected_unittest")
    expected_count = expected["count"] if expected is not None else None
    expected_identities = expected["identities"] if expected is not None else None
    problems: list[str] = []
    terminal_summaries = [
        line.strip()
        for line in text.splitlines()
        if re.fullmatch(
            r"(?:OK(?:\s*\([^\n]*\))?|FAILED(?:\s*\([^\n]*\))?)",
            line.strip(),
        )
    ]
    if len(terminal_summaries) != 1:
        problems.append(
            "expected exactly one unittest terminal OK/FAILED summary, got "
            f"{len(terminal_summaries)}"
        )
    elif not terminal_summaries[0].startswith("OK"):
        problems.append(
            "unittest terminal summary reports failure: "
            + terminal_summaries[0]
        )
    if len(reported_counts) != 1:
        problems.append(
            "expected exactly one unittest 'Ran N tests' summary, got "
            f"{len(reported_counts)}"
        )
        reported_count = None
    else:
        reported_count = reported_counts[0]
        if reported_count <= 0:
            problems.append("unittest discovered no tests")
        if reported_count != len(identities):
            problems.append(
                f"unittest reported {reported_count} test(s) but emitted "
                f"{len(identities)} identifiable verbose test record(s)"
            )
    duplicate_identities = sorted(
        identity
        for identity, count in Counter(identities).items()
        if count > 1
    )
    if duplicate_identities:
        problems.append(
            "unittest emitted duplicate test identities: "
            + ", ".join(duplicate_identities)
        )
    if expected is not None:
        if reported_count != expected_count:
            problems.append(
                f"unittest reported {reported_count!r} test(s), expected "
                f"{expected_count}"
            )
        if Counter(identities) != Counter(expected_identities):
            missing = sorted((Counter(expected_identities) - Counter(identities)).elements())
            unexpected = sorted((Counter(identities) - Counter(expected_identities)).elements())
            if missing:
                problems.append(
                    "expected unittest identities were not run: " + ", ".join(missing)
                )
            if unexpected:
                problems.append(
                    "unexpected unittest identities were run: "
                    + ", ".join(unexpected)
                )
    evidence = {
        "reported_counts": reported_counts,
        "reported_count": reported_count,
        "identities": identities,
        "expected_count": expected_count,
        "expected_identities": expected_identities,
        "terminal_summaries": terminal_summaries,
    }
    return not problems, evidence, problems


def verify_unittest_skip_policy(
    case: dict[str, Any], skip_count: int,
    details: list[dict[str, str]],
) -> tuple[bool, list[str]]:
    allowed = case.get("allowed_unittest_skip_details", [])
    problems: list[str] = []
    if skip_count != len(details):
        problems.append(
            f"unittest reported {skip_count} skip(s) but emitted "
            f"{len(details)} identifiable verbose skip record(s)"
        )
        return False, problems
    used: set[int] = set()
    for detail in details:
        matches = [
            index
            for index, policy in enumerate(allowed)
            if index not in used
            and fnmatch.fnmatchcase(detail["test"], policy["test"])
            and fnmatch.fnmatchcase(detail["reason"], policy["reason"])
        ]
        if len(matches) != 1:
            problems.append(
                "unapproved or ambiguous unittest skip: "
                f"{detail['test']!r}, reason={detail['reason']!r}"
            )
            continue
        used.add(matches[0])
    return not problems, problems


def flatten_phase_counters(
    stats: dict[str, Any], prefix: str, *, strict_only: bool = False
) -> int:
    total = 0
    groups = stats.get("statistics")
    if not isinstance(groups, dict):
        return 0
    for group_name, group in groups.items():
        if not isinstance(group, dict):
            continue
        for name, value in group.items():
            qualified = f"{group_name}.{name}"
            if (
                name.startswith(prefix)
                and name.endswith("_calls")
                and isinstance(value, int)
                and (
                    not strict_only
                    or qualified in STRICT_CUDA_PHASE_CALL_COUNTERS
                )
            ):
                total += value
    return total


def phase_counter(stats: dict[str, Any], qualified: str) -> Any:
    group_name, counter_name = qualified.split(".", 1)
    groups = stats.get("statistics")
    group = groups.get(group_name) if isinstance(groups, dict) else None
    return group.get(counter_name) if isinstance(group, dict) else None


def validate_phase_counters(stats: dict[str, Any]) -> list[str]:
    """Validate every dispatch call counter before any aggregate is trusted."""
    problems: list[str] = []
    groups = stats.get("statistics")
    if not isinstance(groups, dict):
        return problems
    for group_name, group in groups.items():
        if not isinstance(group_name, str) or not isinstance(group, dict):
            continue
        for name, value in group.items():
            if not isinstance(name, str) or not name.endswith("_calls"):
                continue
            qualified = f"{group_name}.{name}"
            if qualified not in PHASE_CALL_COUNTERS:
                problems.append(
                    f"unknown GPU phase call counter {qualified!r}"
                )
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value > UINT64_MAX
            ):
                problems.append(
                    f"GPU phase call counter {qualified!r} must be a "
                    "nonnegative uint64 integer"
                )
    return problems


def verify_backend_contract(
    case: dict[str, Any], backend: str, stats: dict[str, Any],
    expected_runtime: dict[str, Any], run_nonce: str,
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if stats.get("capture_status") != "captured":
        return False, [f"GPU evidence was not captured ({stats.get('capture_status')})"]
    if not stats.get("gpu_api_available"):
        return False, ["mp.gpu API was unavailable"]
    if stats.get("run_nonce") != run_nonce:
        problems.append("GPU evidence nonce does not match this child run")
    if stats.get("build_receipt_id") != expected_runtime.get("receipt_id"):
        problems.append("GPU evidence build receipt ID mismatch")
    if stats.get("single_precision") is not True:
        problems.append("Meep runtime is not single precision")
    if stats.get("with_mpi") is not True:
        problems.append("Meep runtime is not MPI-enabled")
    if stats.get("gpu_compiled") is not True:
        problems.append("Meep runtime is not CUDA-enabled")
    expected_strict = backend == "cuda"
    if stats.get("strict_cuda_marker") is not expected_strict:
        problems.append("strict CUDA marker does not match the backend")
    for name in ("python_executable", "meep_module", "extension", "libmeep"):
        expected = expected_runtime.get(name)
        actual = stats.get(name)
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            problems.append(f"{name} runtime evidence is absent")
            continue
        if pathlib.Path(str(actual.get("path", ""))).resolve() != pathlib.Path(
            str(expected.get("path", ""))
        ).resolve():
            problems.append(f"loaded {name} path differs from the build receipt")
        if actual.get("sha256") != expected.get("sha256"):
            problems.append(f"loaded {name} SHA-256 differs from the build receipt")
    if not isinstance(stats.get("statistics"), dict):
        problems.append("mp.gpu statistics were unavailable")
    else:
        problems.extend(validate_phase_counters(stats))
    requested = stats.get("requested_backend")
    active = stats.get("active_backend")
    if backend == "cpu":
        if case.get("require_active_backend_at_exit", True):
            if requested != "cpu":
                problems.append(f"requested backend is {requested!r}, not 'cpu'")
            if active != "cpu":
                problems.append(f"active backend is {active!r}, not 'cpu'")
        if (
            case["gpu_contract"] == "cuda_dispatch"
            and flatten_phase_counters(stats, "cuda_") != 0
        ):
            problems.append("CPU reference run recorded CUDA phase calls")
    elif backend == "cuda":
        if stats.get("runtime_available") is not True:
            problems.append("CUDA runtime is unavailable")
        if case.get("require_active_backend_at_exit", True):
            selected_device = stats.get("selected_device")
            if (
                not isinstance(selected_device, int)
                or isinstance(selected_device, bool)
                or selected_device < 0
            ):
                problems.append("strict CUDA run selected no CUDA device")
            if requested != "cuda":
                problems.append(f"requested backend is {requested!r}, not 'cuda'")
            if active != "cuda":
                problems.append(f"active backend is {active!r}, not 'cuda'")
        if case["gpu_contract"] == "cuda_dispatch":
            cuda_calls = flatten_phase_counters(
                stats, "cuda_", strict_only=True
            )
            cpu_calls = flatten_phase_counters(
                stats, "cpu_", strict_only=True
            )
            if cuda_calls <= 0:
                problems.append("strict CUDA run recorded no CUDA phase calls")
            if cpu_calls != 0:
                problems.append(f"strict CUDA run recorded {cpu_calls} CPU fallback calls")
    for counter in case.get("required_cuda_call_counters", []):
        activity_value = phase_counter(stats, counter)
        cpu_counter = REQUIRED_CUDA_CPU_COUNTER_PAIRS.get(counter)
        cpu_value = (
            phase_counter(stats, cpu_counter)
            if cpu_counter is not None
            else None
        )
        activity_positive = (
            isinstance(activity_value, int)
            and not isinstance(activity_value, bool)
            and 0 < activity_value <= UINT64_MAX
        )
        if backend == "cuda":
            if not activity_positive:
                problems.append(
                    f"required CUDA activity counter {counter!r} did not execute"
                )
            if cpu_counter is not None and cpu_value != 0:
                problems.append(
                    f"required phase {counter!r} recorded CPU fallback"
                )
        elif backend == "cpu":
            if counter in REQUIRED_CUDA_SHARED_ACTIVITY_COUNTERS:
                if not activity_positive:
                    problems.append(
                        "CPU reference did not execute shared required activity "
                        f"{counter!r}"
                    )
            elif activity_value != 0:
                problems.append(
                    "CPU reference executed required CUDA-only activity "
                    f"{counter!r}"
                )
            if cpu_counter is not None:
                cpu_positive = (
                    isinstance(cpu_value, int)
                    and not isinstance(cpu_value, bool)
                    and 0 < cpu_value <= UINT64_MAX
                )
                if not cpu_positive:
                    problems.append(
                        "CPU reference did not execute paired counter "
                        f"{cpu_counter!r}"
                    )
    return not problems, problems


def normalize_stdout(text: str, comparison: dict[str, Any]) -> str:
    ignored = comparison.get("ignore_regex", [])
    if not isinstance(ignored, list) or not all(isinstance(x, str) for x in ignored):
        raise ManifestError("normalized_stdout ignore_regex must be a string array")
    lines = []
    for line in text.splitlines():
        normalized = line
        for pattern in ignored:
            normalized = re.sub(pattern, "", normalized)
        normalized = normalized.strip()
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)


def extract_json_metrics(
    text: str, comparison: dict[str, Any]
) -> dict[str, float]:
    prefix = comparison["prefix"]
    payloads = [
        line[len(prefix) :].strip()
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(payloads) != 1:
        raise ManifestError(
            f"expected exactly one {prefix!r} metrics line, got "
            f"{len(payloads)}"
        )
    payload_bytes = len(strict_utf8(payloads[0], "json_metrics payload"))
    if payload_bytes > MAX_JSON_PAYLOAD_BYTES:
        raise ManifestError(
            f"json_metrics payload is {payload_bytes} bytes; limit is "
            f"{MAX_JSON_PAYLOAD_BYTES}"
        )
    depth = 0
    in_string = False
    escaped = False
    for character in payloads[0]:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING_DEPTH:
                raise ManifestError(
                    "json_metrics nesting depth exceeds the "
                    f"{MAX_JSON_NESTING_DEPTH}-level limit"
                )
        elif character in "]}":
            depth -= 1
            if depth < 0:
                # Let the JSON decoder provide the precise syntax error while
                # avoiding a misleading nesting-limit diagnosis.
                break
    try:
        value = json.loads(
            payloads[0], object_pairs_hook=unique_json_object
        )
    except (json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise ManifestError(f"invalid json_metrics payload: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ManifestError("json_metrics payload must be a nonempty object")

    flattened: dict[str, float] = {}
    flattened_path_bytes = 0

    pending: list[tuple[Any, str]] = [(value, "")]
    while pending:
        item, path = pending.pop()
        if isinstance(item, bool):
            raise ManifestError(
                f"json_metrics value {path!r} must be numeric"
            )
        if isinstance(item, (int, float)):
            try:
                number = float(item)
            except OverflowError as exc:
                raise ManifestError(
                    f"json_metrics value {path!r} must be finite"
                ) from exc
            if not math.isfinite(number):
                raise ManifestError(
                    f"json_metrics value {path!r} must be finite"
                )
            if path in flattened:
                raise ManifestError(
                    f"json_metrics flattened path {path!r} is ambiguous"
                )
            encoded_path = strict_utf8(path, "json_metrics metric path")
            if len(encoded_path) > MAX_JSON_METRIC_PATH_BYTES:
                raise ManifestError(
                    f"json_metrics metric path exceeds the "
                    f"{MAX_JSON_METRIC_PATH_BYTES}-byte limit"
                )
            if len(flattened) >= MAX_JSON_METRICS:
                raise ManifestError(
                    f"json_metrics exceeds the {MAX_JSON_METRICS}-metric limit"
                )
            flattened_path_bytes += len(encoded_path)
            if flattened_path_bytes > MAX_JSON_METRIC_PATH_BYTES_TOTAL:
                raise ManifestError(
                    "json_metrics flattened paths exceed the "
                    f"{MAX_JSON_METRIC_PATH_BYTES_TOTAL}-byte budget"
                )
            flattened[path] = number
            continue
        if isinstance(item, list):
            if not item:
                raise ManifestError(
                    f"json_metrics array {path!r} must be nonempty"
                )
            for index in range(len(item) - 1, -1, -1):
                pending.append((item[index], f"{path}[{index}]"))
            continue
        if isinstance(item, dict):
            if not item:
                raise ManifestError(
                    f"json_metrics object {path!r} must be nonempty"
                )
            for key in reversed(sorted(item)):
                if not isinstance(key, str) or not key:
                    raise ManifestError(
                        "json_metrics object keys must be nonempty strings"
                    )
                pending.append(
                    (item[key], f"{path}.{key}" if path else key)
                )
            continue
        raise ManifestError(
            f"json_metrics value {path!r} has unsupported type"
        )
    return flattened


_JSON_NUMBER_RE = re.compile(
    rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z"
)


def _metric_payload_extent(
    descriptor: int, name: str, prefix: str
) -> tuple[int, int]:
    """Locate one prefixed JSON line without materializing that line.

    The complete stdout file is also checked as strict UTF-8, preserving the
    previous read_text() gate.  Only a prefix-sized fragment of any line is
    retained, so a hundreds-of-megabytes metric line has constant memory cost.
    """
    prefix_bytes = strict_utf8(prefix, "json_metrics prefix")
    if not prefix_bytes or b"\n" in prefix_bytes or b"\r" in prefix_bytes:
        raise ManifestError("json_metrics prefix must be a nonempty single line")
    match_count = 0
    match_extent: tuple[int, int] | None = None
    line_start = 0
    line_size = 0
    line_prefix = bytearray()
    absolute = 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    source = _PreadReader(descriptor, name)

    def finish_line(line_end: int) -> None:
        nonlocal line_start, line_size, line_prefix, match_count, match_extent
        if line_size >= len(prefix_bytes) and bytes(line_prefix) == prefix_bytes:
            payload_start = line_start + len(prefix_bytes)
            payload_size = line_end - payload_start
            if payload_size > MAX_JSON_PAYLOAD_BYTES:
                raise ManifestError(
                    f"json_metrics payload is {payload_size} bytes; limit is "
                    f"{MAX_JSON_PAYLOAD_BYTES}"
                )
            match_count += 1
            if match_extent is None:
                match_extent = (payload_start, line_end)
        line_start = line_end + 1
        line_size = 0
        line_prefix = bytearray()

    try:
        while True:
            chunk = source.read(JSON_METRIC_STREAM_READ_BYTES)
            if not chunk:
                break
            decoder.decode(chunk, final=False)
            segment_start = 0
            while True:
                newline = chunk.find(b"\n", segment_start)
                segment_end = len(chunk) if newline < 0 else newline
                segment = chunk[segment_start:segment_end]
                needed = len(prefix_bytes) - len(line_prefix)
                if needed > 0:
                    line_prefix.extend(segment[:needed])
                line_size += len(segment)
                if newline < 0:
                    break
                finish_line(absolute + newline)
                segment_start = newline + 1
            absolute += len(chunk)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise ManifestError(f"json_metrics stdout is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise ManifestError(f"cannot read json_metrics stdout {name}: {exc}") from exc
    if line_start < absolute or line_size:
        finish_line(absolute)
    if match_count != 1:
        raise ManifestError(
            f"expected exactly one {prefix!r} metrics line, got {match_count}"
        )
    assert match_extent is not None
    return match_extent


class _StableMetricSnapshot:
    def __init__(
        self,
        path: pathlib.Path,
        descriptor: int,
        identity: tuple[int, int, int, int, int],
        record: dict[str, Any],
    ):
        self.path = path
        self.descriptor = descriptor
        self.identity = identity
        self.record = record

    def verify(self, label: str) -> None:
        try:
            descriptor_stat = os.fstat(self.descriptor)
            path_stat = self.path.stat(follow_symlinks=False)
        except OSError as exc:
            raise ManifestError(
                f"private {label} json_metrics snapshot is unavailable: {exc}"
            ) from exc
        if (
            _stat_identity(descriptor_stat) != self.identity
            or _stat_identity(path_stat) != self.identity
            or descriptor_stat.st_size != self.record["size_bytes"]
            or _fd_sha256(self.descriptor) != self.record["sha256"]
        ):
            raise ManifestError(
                f"private {label} json_metrics snapshot changed during comparison"
            )

    def close(self) -> None:
        os.close(self.descriptor)


def _copy_verified_metric_evidence(
    source_path: pathlib.Path,
    snapshot_path: pathlib.Path,
    expected_record: dict[str, Any] | None,
) -> _StableMetricSnapshot:
    """Copy one evidence file through one stable FD and bind its digest.

    Parsing the private read-only copy eliminates the former extent-scan/path-
    reopen race.  When run evidence is available, the copied bytes must match
    the already-recorded size and SHA-256 before any metric is interpreted.
    """
    if expected_record is not None:
        if expected_record.get("available") is not True:
            raise ManifestError("json_metrics stdout evidence is unavailable")
        expected_size = expected_record.get("size_bytes")
        expected_sha256 = expected_record.get("sha256")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        ):
            raise ManifestError("json_metrics stdout evidence record is invalid")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source_path, flags)
    except OSError as exc:
        raise ManifestError(
            f"cannot open json_metrics stdout {source_path}: {exc}"
        ) from exc
    digest = hashlib.sha256()
    copied_size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ManifestError(
                f"json_metrics stdout is not a regular file: {source_path}"
            )
        if before.st_size > MAX_BACKEND_STDOUT_BYTES:
            raise ManifestError(
                "json_metrics stdout exceeds the bounded backend stdout limit"
            )
        with os.fdopen(os.dup(descriptor), "rb") as source, snapshot_path.open(
            "xb"
        ) as destination:
            while True:
                block = source.read(JSON_METRIC_STREAM_READ_BYTES)
                if not block:
                    break
                destination.write(block)
                digest.update(block)
                copied_size += len(block)
                if copied_size > MAX_BACKEND_STDOUT_BYTES:
                    raise ManifestError(
                        "json_metrics stdout grew beyond the bounded backend "
                        "stdout limit while being snapshotted"
                    )
            destination.flush()
            os.fsync(destination.fileno())
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or copied_size != before.st_size:
        raise ManifestError(
            "json_metrics stdout changed while it was being snapshotted"
        )
    actual_sha256 = digest.hexdigest()
    if expected_record is not None and (
        copied_size != expected_record["size_bytes"]
        or actual_sha256 != expected_record["sha256"]
    ):
        raise ManifestError(
            "json_metrics stdout differs from its recorded evidence digest"
        )
    snapshot_path.chmod(0o400)
    snapshot_descriptor = _open_stable_readonly(snapshot_path)
    record = {"size_bytes": copied_size, "sha256": actual_sha256}
    try:
        snapshot_stat = os.fstat(snapshot_descriptor)
        path_stat = snapshot_path.stat(follow_symlinks=False)
        identity = _stat_identity(snapshot_stat)
        if (
            identity != _stat_identity(path_stat)
            or snapshot_stat.st_size != copied_size
            or _fd_sha256(snapshot_descriptor) != actual_sha256
        ):
            raise ManifestError(
                "private json_metrics snapshot differs immediately after creation"
            )
    except Exception:
        os.close(snapshot_descriptor)
        raise
    return _StableMetricSnapshot(
        snapshot_path, snapshot_descriptor, identity, record
    )


class _PayloadCursor:
    """Small buffered byte cursor constrained to a metric payload extent."""

    def __init__(self, descriptor: int, name: str, start: int, end: int):
        self._source = _PreadReader(descriptor, name, start)
        self._remaining = end - start
        self._buffer = b""
        self._index = 0
        self.offset = 0

    def close(self) -> None:
        return None

    def _fill(self) -> bool:
        if self._index < len(self._buffer):
            return True
        if self._remaining <= 0:
            self._buffer = b""
            self._index = 0
            return False
        block = self._source.read(
            min(JSON_METRIC_STREAM_READ_BYTES, self._remaining)
        )
        if not block:
            raise ManifestError("json_metrics payload ended unexpectedly")
        self._remaining -= len(block)
        self._buffer = block
        self._index = 0
        return True

    def peek(self) -> int | None:
        return self._buffer[self._index] if self._fill() else None

    def take(self) -> int:
        value = self.peek()
        if value is None:
            raise ManifestError("json_metrics payload ended unexpectedly")
        self._index += 1
        self.offset += 1
        return value

    def skip_whitespace(self) -> None:
        while self.peek() in (0x20, 0x09, 0x0A, 0x0D):
            self.take()


def _sort_record_digest(header: bytes, key: bytes, value_bytes: bytes) -> bytes:
    digest = hashlib.sha256()
    digest.update(JSON_SORT_RECORD_DOMAIN)
    digest.update(header)
    digest.update(key)
    digest.update(value_bytes)
    return digest.digest()


def _verify_sort_record_digest(
    key: bytes, value: float, expected_digest: bytes
) -> None:
    header = struct.pack(">I", len(key))
    value_bytes = struct.pack(">d", value)
    if _sort_record_digest(header, key, value_bytes) != expected_digest:
        raise ManifestError(
            "json_metrics temporary sort record changed in memory: "
            f"key_size={len(key)}, stored_sha256={expected_digest.hex()}, "
            f"computed_sha256={_sort_record_digest(header, key, value_bytes).hex()}, "
            f"key_sha256={hashlib.sha256(key).hexdigest()}"
        )


def _write_sort_record(
    stream: BinaryIO,
    key: bytes,
    value: float,
    expected_digest: bytes | None = None,
) -> bytes:
    if len(key) > 0xFFFFFFFF:
        raise ManifestError("json_metrics sort key is too large")
    header = struct.pack(">I", len(key))
    value_bytes = struct.pack(">d", value)
    digest = _sort_record_digest(header, key, value_bytes)
    if expected_digest is not None and digest != expected_digest:
        raise ManifestError(
            "json_metrics temporary sort record changed in memory"
        )
    # One immutable byte string prevents buffered-I/O component boundaries
    # from separating a key/value from the checksum that authenticates it.
    record = header + key + value_bytes + digest
    remaining = memoryview(record)
    while remaining:
        written = stream.write(remaining)
        if not isinstance(written, int) or written <= 0:
            raise ManifestError(
                "json_metrics temporary sort record write was partial: "
                f"remaining={len(remaining)}, written={written!r}"
            )
        remaining = remaining[written:]
    return digest


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ManifestError("json_metrics temporary sort record is truncated")
    return value


def _read_sort_record(
    stream: BinaryIO,
) -> tuple[bytes, float, bytes] | None:
    record_start = stream.tell()
    last_error: ManifestError | None = None
    for _ in range(JSON_SORT_RECORD_READ_ATTEMPTS):
        stream.seek(record_start)
        header = stream.read(4)
        if not header:
            return None
        try:
            if len(header) != 4:
                raise ManifestError(
                    "json_metrics temporary sort record is truncated"
                )
            key_size = struct.unpack(">I", header)[0]
            if key_size > MAX_JSON_METRIC_PATH_BYTES + 8:
                raise ManifestError(
                    "json_metrics temporary sort key exceeds its limit"
                )
            key = _read_exact(stream, key_size)
            value_bytes = _read_exact(stream, 8)
            digest = _read_exact(stream, JSON_SORT_RECORD_DIGEST_BYTES)
            computed_digest = _sort_record_digest(header, key, value_bytes)
            if digest != computed_digest:
                stream_name = getattr(stream, "name", "<unnamed-stream>")
                raise ManifestError(
                    "json_metrics temporary sort record failed its checksum: "
                    f"stream={stream_name!r}, offset={record_start}, "
                    f"key_size={key_size}, stored_sha256={digest.hex()}, "
                    f"computed_sha256={computed_digest.hex()}, "
                    f"key_sha256={hashlib.sha256(key).hexdigest()}"
                )
            value = struct.unpack(">d", value_bytes)[0]
            return key, value, digest
        except ManifestError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class _PreadReader:
    """Independent seek/read cursor over one already-open stable file."""

    def __init__(self, descriptor: int, name: str, offset: int = 0):
        self.descriptor = descriptor
        self.name = name
        self.offset = offset

    def tell(self) -> int:
        return self.offset

    def seek(self, offset: int) -> int:
        if offset < 0:
            raise OSError("negative stable-file offset")
        self.offset = offset
        return offset

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = os.fstat(self.descriptor).st_size - self.offset
        value = os.pread(self.descriptor, size, self.offset)
        self.offset += len(value)
        return value

    def close(self) -> None:
        return None


def _fd_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(descriptor, JSON_METRIC_STREAM_READ_BYTES, offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


class _SortChunk:
    def __init__(
        self,
        path: pathlib.Path,
        identity: tuple[int, int, int, int, int],
        sha256: str,
        record_count: int,
    ):
        self.path = path
        self.identity = identity
        self.sha256 = sha256
        self.record_count = record_count


def _open_stable_readonly(path: pathlib.Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"cannot open stable evidence file {path}: {exc}") from exc


def _validate_sort_chunk(
    path: pathlib.Path, expected_count: int
) -> _SortChunk:
    path.chmod(0o400)
    descriptor = _open_stable_readonly(path)
    count = 0
    try:
        before = os.fstat(descriptor)
        source = _PreadReader(descriptor, str(path))
        while _read_sort_record(source) is not None:
            count += 1
        digest = _fd_sha256(descriptor)
        after = os.fstat(descriptor)
        path_stat = path.stat(follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = _stat_identity(before)
    if identity != _stat_identity(after) or identity != _stat_identity(path_stat):
        raise ManifestError(
            f"json_metrics temporary sort chunk changed during sealing: {path}"
        )
    if count != expected_count:
        raise ManifestError(
            "json_metrics temporary sort chunk record count differs: "
            f"path={path}, expected={expected_count}, actual={count}"
        )
    return _SortChunk(path, identity, digest, count)


def _external_sort_records(
    source_path: pathlib.Path,
    output_directory: pathlib.Path,
    stem: str,
) -> list[_SortChunk]:
    chunks: list[_SortChunk] = []
    with source_path.open("rb") as source:
        while True:
            records: list[tuple[bytes, float, bytes]] = []
            retained_bytes = 0
            while retained_bytes < JSON_METRIC_SORT_CHUNK_BYTES:
                record = _read_sort_record(source)
                if record is None:
                    break
                records.append(record)
                retained_bytes += len(record[0]) + 96
            if not records:
                break
            records.sort(key=lambda item: item[0])
            chunk_path = output_directory / f"{stem}-{len(chunks):05d}.sort"
            with chunk_path.open("xb") as output:
                for key, value, digest in records:
                    _write_sort_record(output, key, value, digest)
                output.flush()
                os.fsync(output.fileno())
            chunks.append(_validate_sort_chunk(chunk_path, len(records)))
    return chunks


def _iter_sorted_chunks(
    chunks: list[_SortChunk], description: str
) -> Iterator[tuple[bytes, float]]:
    streams: list[_PreadReader] = []
    descriptors: list[int] = []
    heap: list[tuple[bytes, int, float, bytes]] = []
    previous: bytes | None = None
    try:
        for index, chunk in enumerate(chunks):
            descriptor = _open_stable_readonly(chunk.path)
            descriptors.append(descriptor)
            before = os.fstat(descriptor)
            path_stat = chunk.path.stat(follow_symlinks=False)
            if (
                _stat_identity(before) != chunk.identity
                or _stat_identity(path_stat) != chunk.identity
                or _fd_sha256(descriptor) != chunk.sha256
            ):
                raise ManifestError(
                    "json_metrics temporary sort chunk differs before merge: "
                    f"{chunk.path}"
                )
            stream = _PreadReader(descriptor, str(chunk.path))
            streams.append(stream)
            record = _read_sort_record(stream)
            if record is not None:
                heapq.heappush(
                    heap, (record[0], index, record[1], record[2])
                )
        while heap:
            key, index, value, digest = heapq.heappop(heap)
            _verify_sort_record_digest(key, value, digest)
            if previous == key:
                try:
                    shown = key.decode("utf-8")
                except UnicodeDecodeError:
                    shown = repr(key)
                raise ManifestError(
                    f"json_metrics flattened path {shown!r} is ambiguous"
                    if description == "metric"
                    else f"JSON contains duplicate key {shown[8:]!r}"
                )
            previous = key
            yield key, value
            record = _read_sort_record(streams[index])
            if record is not None:
                heapq.heappush(
                    heap, (record[0], index, record[1], record[2])
                )
    finally:
        closing_early = sys.exc_info()[0] is GeneratorExit
        integrity_problems: list[str] = []
        for chunk, descriptor in zip(chunks, descriptors):
            try:
                after = os.fstat(descriptor)
                path_stat = chunk.path.stat(follow_symlinks=False)
                if (
                    _stat_identity(after) != chunk.identity
                    or _stat_identity(path_stat) != chunk.identity
                    or _fd_sha256(descriptor) != chunk.sha256
                ):
                    integrity_problems.append(str(chunk.path))
            except OSError:
                integrity_problems.append(str(chunk.path))
            finally:
                os.close(descriptor)
        if integrity_problems and not closing_early:
            raise ManifestError(
                "json_metrics temporary sort chunk changed during merge: "
                + ", ".join(integrity_problems[:8])
            )


_METRIC_MULTISET_MASK = (1 << 256) - 1


class _MetricMultisetAccumulator:
    """Order-independent cryptographic binding for a complete metric set."""

    def __init__(self) -> None:
        self.count = 0
        self.primary_sum = 0
        self.secondary_sum = 0
        self.digest_xor = 0

    def add_digest(self, record_digest: bytes) -> None:
        if len(record_digest) != hashlib.sha256().digest_size:
            raise ManifestError("json_metrics record digest size is invalid")
        primary = int.from_bytes(record_digest, "big")
        secondary = int.from_bytes(
            hashlib.sha256(
                b"gpmeep-json-metric-multiset-secondary-v1\0"
                + record_digest
            ).digest(),
            "big",
        )
        self.count += 1
        self.primary_sum = (
            self.primary_sum + primary
        ) & _METRIC_MULTISET_MASK
        self.secondary_sum = (
            self.secondary_sum + secondary
        ) & _METRIC_MULTISET_MASK
        self.digest_xor ^= primary

    def add_metric(self, key: bytes, value: float) -> None:
        header = struct.pack(">I", len(key))
        value_bytes = struct.pack(">d", value)
        self.add_digest(_sort_record_digest(header, key, value_bytes))

    def binding(self) -> tuple[int, int, int, int]:
        return (
            self.count,
            self.primary_sum,
            self.secondary_sum,
            self.digest_xor,
        )


class _StreamingMetricParser:
    def __init__(
        self,
        cursor: _PayloadCursor,
        metric_stream: BinaryIO,
        object_key_stream: BinaryIO,
    ):
        self.cursor = cursor
        self.metric_stream = metric_stream
        self.object_key_stream = object_key_stream
        self.metric_count = 0
        self.path_bytes = 0
        self.object_serial = 0
        self.metric_multiset = _MetricMultisetAccumulator()
        self.object_key_multiset = _MetricMultisetAccumulator()

    def _error(self, message: str) -> ManifestError:
        return ManifestError(
            f"invalid json_metrics payload near byte {self.cursor.offset}: {message}"
        )

    def _expect(self, expected: int) -> None:
        actual = self.cursor.take()
        if actual != expected:
            raise self._error(
                f"expected {chr(expected)!r}, got {chr(actual)!r}"
            )

    def _string(self) -> str:
        self._expect(ord('"'))
        encoded = bytearray(b'"')
        escaped = False
        while True:
            byte = self.cursor.take()
            encoded.append(byte)
            if len(encoded) > 12 * MAX_JSON_METRIC_PATH_BYTES + 2:
                raise self._error("object key exceeds the metric path limit")
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                break
            elif byte < 0x20:
                raise self._error("unescaped control byte in object key")
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise self._error(f"invalid object key: {exc}") from exc
        if not isinstance(value, str):
            raise self._error("object key is not a string")
        return value

    def _metric(self, path: str, value: float) -> None:
        if not math.isfinite(value):
            raise ManifestError(
                f"json_metrics value {path!r} must be finite"
            )
        encoded_path = strict_utf8(path, "json_metrics metric path")
        if len(encoded_path) > MAX_JSON_METRIC_PATH_BYTES:
            raise ManifestError(
                "json_metrics metric path exceeds the "
                f"{MAX_JSON_METRIC_PATH_BYTES}-byte limit"
            )
        if self.metric_count >= MAX_JSON_METRICS:
            raise ManifestError(
                f"json_metrics exceeds the {MAX_JSON_METRICS}-metric limit"
            )
        self.metric_count += 1
        self.path_bytes += len(encoded_path)
        if self.path_bytes > MAX_JSON_METRIC_PATH_BYTES_TOTAL:
            raise ManifestError(
                "json_metrics flattened paths exceed the "
                f"{MAX_JSON_METRIC_PATH_BYTES_TOTAL}-byte budget"
            )
        record_digest = _write_sort_record(
            self.metric_stream, encoded_path, value
        )
        self.metric_multiset.add_digest(record_digest)

    def _number(self, path: str) -> None:
        token = bytearray()
        while True:
            byte = self.cursor.peek()
            if byte is None or byte in (0x20, 0x09, 0x0A, 0x0D, ord(","), ord("]"), ord("}")):
                break
            token.append(self.cursor.take())
            if len(token) > MAX_JSON_NUMBER_TOKEN_BYTES:
                raise ManifestError(
                    f"json_metrics value {path!r} must be finite"
                )
        raw = bytes(token)
        if not _JSON_NUMBER_RE.fullmatch(raw):
            raise self._error("invalid numeric value")
        try:
            number_text = raw.decode("ascii")
            # Match json.loads followed by flatten_json_metrics' float()
            # conversion exactly.  In particular JSON's integer spelling
            # ``-0`` is decoded as the integer 0 and therefore canonicalizes
            # to +0.0; directly calling float("-0") would incorrectly retain
            # a negative-zero sign bit and change the evidence digests.
            if not any(marker in raw for marker in (b".", b"e", b"E")):
                value = float(int(number_text, 10))
            else:
                value = float(number_text)
        except (UnicodeDecodeError, ValueError, OverflowError) as exc:
            raise ManifestError(
                f"json_metrics value {path!r} must be finite"
            ) from exc
        self._metric(path, value)

    def _value(self, path: str, depth: int) -> None:
        self.cursor.skip_whitespace()
        byte = self.cursor.peek()
        if byte == ord("{"):
            self._object(path, depth + 1)
        elif byte == ord("["):
            self._array(path, depth + 1)
        elif byte is not None and (byte == ord("-") or ord("0") <= byte <= ord("9")):
            self._number(path)
        elif byte is None:
            raise self._error("expected a value")
        else:
            raise ManifestError(
                f"json_metrics value {path!r} must be numeric"
            )

    def _check_depth(self, depth: int) -> None:
        if depth > MAX_JSON_NESTING_DEPTH:
            raise ManifestError(
                "json_metrics nesting depth exceeds the "
                f"{MAX_JSON_NESTING_DEPTH}-level limit"
            )

    def _object(self, path: str, depth: int) -> None:
        self._check_depth(depth)
        self._expect(ord("{"))
        self.object_serial += 1
        if self.object_serial > UINT64_MAX:
            raise ManifestError("json_metrics has too many objects")
        object_prefix = self.object_serial.to_bytes(8, "big")
        self.cursor.skip_whitespace()
        if self.cursor.peek() == ord("}"):
            raise ManifestError(
                f"json_metrics object {path!r} must be nonempty"
            )
        while True:
            self.cursor.skip_whitespace()
            if self.cursor.peek() != ord('"'):
                raise self._error("object key must be a string")
            key = self._string()
            if not key:
                raise ManifestError(
                    "json_metrics object keys must be nonempty strings"
                )
            encoded_key = strict_utf8(key, "json_metrics object key")
            child_path = f"{path}.{key}" if path else key
            child_path_size = len(
                strict_utf8(child_path, "json_metrics metric path")
            )
            if child_path_size > MAX_JSON_METRIC_PATH_BYTES:
                raise ManifestError(
                    "json_metrics metric path exceeds the "
                    f"{MAX_JSON_METRIC_PATH_BYTES}-byte limit"
                )
            object_digest = _write_sort_record(
                self.object_key_stream, object_prefix + encoded_key, 0.0
            )
            self.object_key_multiset.add_digest(object_digest)
            self.cursor.skip_whitespace()
            self._expect(ord(":"))
            self._value(child_path, depth)
            self.cursor.skip_whitespace()
            separator = self.cursor.take()
            if separator == ord("}"):
                return
            if separator != ord(","):
                raise self._error("expected ',' or '}'")

    def _array(self, path: str, depth: int) -> None:
        self._check_depth(depth)
        self._expect(ord("["))
        self.cursor.skip_whitespace()
        if self.cursor.peek() == ord("]"):
            raise ManifestError(
                f"json_metrics array {path!r} must be nonempty"
            )
        index = 0
        while True:
            self._value(f"{path}[{index}]", depth)
            index += 1
            self.cursor.skip_whitespace()
            separator = self.cursor.take()
            if separator == ord("]"):
                return
            if separator != ord(","):
                raise self._error("expected ',' or ']'")

    def parse(self) -> None:
        self.cursor.skip_whitespace()
        if self.cursor.peek() != ord("{"):
            raise ManifestError("json_metrics payload must be a nonempty object")
        self._object("", 1)
        self.cursor.skip_whitespace()
        if self.cursor.peek() is not None:
            raise self._error("trailing data after the JSON object")


class _FastPathUnavailable(Exception):
    """The exact direct parser must replay through unordered external sort."""


DIRECT_MAX_OBJECT_KEYS = 4096


class _DirectMetricParser:
    """Yield every metric directly for same-order CPU/CUDA full comparison."""

    def __init__(self, cursor: _PayloadCursor, *, bind_origin: bool):
        self.cursor = cursor
        self.bind_origin = bind_origin
        self.metric_count = 0
        self.path_bytes = 0
        self.metric_multiset = _MetricMultisetAccumulator()

    def _error(self, message: str) -> ManifestError:
        return ManifestError(
            f"invalid json_metrics payload near byte {self.cursor.offset}: {message}"
        )

    def _expect(self, expected: int) -> None:
        actual = self.cursor.take()
        if actual != expected:
            raise self._error(
                f"expected {chr(expected)!r}, got {chr(actual)!r}"
            )

    def _string(self) -> str:
        self._expect(ord('"'))
        encoded = bytearray(b'"')
        escaped = False
        while True:
            byte = self.cursor.take()
            encoded.append(byte)
            if len(encoded) > 12 * MAX_JSON_METRIC_PATH_BYTES + 2:
                raise self._error("object key exceeds the metric path limit")
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                break
            elif byte < 0x20:
                raise self._error("unescaped control byte in object key")
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise self._error(f"invalid object key: {exc}") from exc
        if not isinstance(value, str):
            raise self._error("object key is not a string")
        return value

    def _metric(self, path: str, value: float) -> tuple[bytes, float]:
        if not math.isfinite(value):
            raise ManifestError(f"json_metrics value {path!r} must be finite")
        encoded_path = strict_utf8(path, "json_metrics metric path")
        if len(encoded_path) > MAX_JSON_METRIC_PATH_BYTES:
            raise ManifestError(
                "json_metrics metric path exceeds the "
                f"{MAX_JSON_METRIC_PATH_BYTES}-byte limit"
            )
        if self.metric_count >= MAX_JSON_METRICS:
            raise ManifestError(
                f"json_metrics exceeds the {MAX_JSON_METRICS}-metric limit"
            )
        self.metric_count += 1
        self.path_bytes += len(encoded_path)
        if self.path_bytes > MAX_JSON_METRIC_PATH_BYTES_TOTAL:
            raise ManifestError(
                "json_metrics flattened paths exceed the "
                f"{MAX_JSON_METRIC_PATH_BYTES_TOTAL}-byte budget"
            )
        if self.bind_origin:
            self.metric_multiset.add_metric(encoded_path, value)
        return encoded_path, value

    def _number(
        self, path: str
    ) -> Iterator[tuple[bytes, float]]:
        token = bytearray()
        while True:
            byte = self.cursor.peek()
            if byte is None or byte in (
                0x20,
                0x09,
                0x0A,
                0x0D,
                ord(","),
                ord("]"),
                ord("}"),
            ):
                break
            token.append(self.cursor.take())
            if len(token) > MAX_JSON_NUMBER_TOKEN_BYTES:
                raise ManifestError(
                    f"json_metrics value {path!r} must be finite"
                )
        raw = bytes(token)
        if not _JSON_NUMBER_RE.fullmatch(raw):
            raise self._error("invalid numeric value")
        try:
            number_text = raw.decode("ascii")
            if not any(marker in raw for marker in (b".", b"e", b"E")):
                value = float(int(number_text, 10))
            else:
                value = float(number_text)
        except (UnicodeDecodeError, ValueError, OverflowError) as exc:
            raise ManifestError(
                f"json_metrics value {path!r} must be finite"
            ) from exc
        yield self._metric(path, value)

    def _value(
        self, path: str, depth: int
    ) -> Iterator[tuple[bytes, float]]:
        self.cursor.skip_whitespace()
        byte = self.cursor.peek()
        if byte == ord("{"):
            yield from self._object(path, depth + 1)
        elif byte == ord("["):
            yield from self._array(path, depth + 1)
        elif byte is not None and (
            byte == ord("-") or ord("0") <= byte <= ord("9")
        ):
            yield from self._number(path)
        elif byte is None:
            raise self._error("expected a value")
        else:
            raise ManifestError(
                f"json_metrics value {path!r} must be numeric"
            )

    def _check_depth(self, depth: int) -> None:
        if depth > MAX_JSON_NESTING_DEPTH:
            raise ManifestError(
                "json_metrics nesting depth exceeds the "
                f"{MAX_JSON_NESTING_DEPTH}-level limit"
            )

    def _object(
        self, path: str, depth: int
    ) -> Iterator[tuple[bytes, float]]:
        self._check_depth(depth)
        self._expect(ord("{"))
        self.cursor.skip_whitespace()
        if self.cursor.peek() == ord("}"):
            raise ManifestError(
                f"json_metrics object {path!r} must be nonempty"
            )
        keys: set[str] = set()
        while True:
            self.cursor.skip_whitespace()
            if self.cursor.peek() != ord('"'):
                raise self._error("object key must be a string")
            key = self._string()
            if not key:
                raise ManifestError(
                    "json_metrics object keys must be nonempty strings"
                )
            if key in keys:
                raise ManifestError(f"JSON contains duplicate key {key!r}")
            keys.add(key)
            if len(keys) > DIRECT_MAX_OBJECT_KEYS or any(
                marker in key for marker in (".", "[", "]")
            ):
                raise _FastPathUnavailable(
                    "direct json_metrics path requires unordered fallback"
                )
            child_path = f"{path}.{key}" if path else key
            if len(strict_utf8(child_path, "json_metrics metric path")) > (
                MAX_JSON_METRIC_PATH_BYTES
            ):
                raise ManifestError(
                    "json_metrics metric path exceeds the "
                    f"{MAX_JSON_METRIC_PATH_BYTES}-byte limit"
                )
            self.cursor.skip_whitespace()
            self._expect(ord(":"))
            yield from self._value(child_path, depth)
            self.cursor.skip_whitespace()
            separator = self.cursor.take()
            if separator == ord("}"):
                return
            if separator != ord(","):
                raise self._error("expected ',' or '}'")

    def _array(
        self, path: str, depth: int
    ) -> Iterator[tuple[bytes, float]]:
        self._check_depth(depth)
        self._expect(ord("["))
        self.cursor.skip_whitespace()
        if self.cursor.peek() == ord("]"):
            raise ManifestError(
                f"json_metrics array {path!r} must be nonempty"
            )
        index = 0
        while True:
            yield from self._value(f"{path}[{index}]", depth)
            index += 1
            self.cursor.skip_whitespace()
            separator = self.cursor.take()
            if separator == ord("]"):
                return
            if separator != ord(","):
                raise self._error("expected ',' or ']'")

    def iter_metrics(self) -> Iterator[tuple[bytes, float]]:
        self.cursor.skip_whitespace()
        if self.cursor.peek() != ord("{"):
            raise ManifestError(
                "json_metrics payload must be a nonempty object"
            )
        yield from self._object("", 1)
        self.cursor.skip_whitespace()
        if self.cursor.peek() is not None:
            raise self._error("trailing data after the JSON object")


class _MetricSource:
    def __init__(
        self,
        iterator_factory: Callable[[], Iterator[tuple[bytes, float]]],
        origin_binding: tuple[int, int, int, int] | None = None,
    ):
        self.iterator_factory = iterator_factory
        self.origin_binding = origin_binding

    def iter_metrics(self) -> Iterator[tuple[bytes, float]]:
        return self.iterator_factory()


class _DirectMetricSource(_MetricSource):
    def __init__(
        self,
        snapshot: _StableMetricSnapshot,
        comparison: dict[str, Any],
    ):
        self.snapshot = snapshot
        self.comparison = comparison
        self.start, self.end = _metric_payload_extent(
            snapshot.descriptor,
            str(snapshot.path),
            comparison["prefix"],
        )
        self._iteration_count = 0
        super().__init__(self._new_iterator)

    def _new_iterator(self) -> Iterator[tuple[bytes, float]]:
        self._iteration_count += 1
        bind_origin = self._iteration_count == 1
        cursor = _PayloadCursor(
            self.snapshot.descriptor,
            str(self.snapshot.path),
            self.start,
            self.end,
        )
        parser = _DirectMetricParser(cursor, bind_origin=bind_origin)
        try:
            yield from parser.iter_metrics()
            if bind_origin:
                self.origin_binding = parser.metric_multiset.binding()
        finally:
            cursor.close()


def _streaming_metric_source(
    snapshot: _StableMetricSnapshot,
    comparison: dict[str, Any],
    temporary_root: pathlib.Path,
    stem: str,
) -> _MetricSource:
    start, end = _metric_payload_extent(
        snapshot.descriptor, str(snapshot.path), comparison["prefix"]
    )
    metric_spool = temporary_root / f"{stem}-metrics.spool"
    object_spool = temporary_root / f"{stem}-object-keys.spool"
    cursor = _PayloadCursor(
        snapshot.descriptor, str(snapshot.path), start, end
    )
    try:
        with metric_spool.open("xb") as metrics, object_spool.open("xb") as keys:
            parser = _StreamingMetricParser(cursor, metrics, keys)
            parser.parse()
    finally:
        cursor.close()
    metric_chunks = _external_sort_records(
        metric_spool, temporary_root, f"{stem}-metric"
    )
    object_chunks = _external_sort_records(
        object_spool, temporary_root, f"{stem}-object"
    )
    # Exhaustion performs exact adjacent duplicate detection.  Object serials
    # are document-local and need not be stable across CPU/GPU runs.
    observed_object_keys = _MetricMultisetAccumulator()
    for key, value in _iter_sorted_chunks(object_chunks, "object key"):
        observed_object_keys.add_metric(key, value)
    if (
        observed_object_keys.binding()
        != parser.object_key_multiset.binding()
    ):
        raise ManifestError(
            "json_metrics sorted object keys differ from parser origin"
        )
    return _MetricSource(
        lambda: _iter_sorted_chunks(metric_chunks, "metric"),
        parser.metric_multiset.binding(),
    )


def flattened_metrics_sha256(metrics: dict[str, float]) -> str:
    """Hash sorted metric paths and IEEE-754 values canonically."""
    digest = hashlib.sha256()
    digest.update(b"gpmeep-json-metrics-v1\0")
    for key in sorted(metrics):
        encoded_key = strict_utf8(key, "json_metrics metric path")
        digest.update(len(encoded_key).to_bytes(8, "big"))
        digest.update(encoded_key)
        digest.update(struct.pack(">d", metrics[key]))
    return digest.hexdigest()


def flattened_metric_keys_sha256(metrics: dict[str, float]) -> str:
    digest = hashlib.sha256()
    digest.update(b"gpmeep-json-metric-keys-v1\0")
    for key in sorted(metrics):
        encoded_key = strict_utf8(key, "json_metrics metric path")
        digest.update(len(encoded_key).to_bytes(8, "big"))
        digest.update(encoded_key)
    return digest.hexdigest()


def flattened_metric_storage_bytes(metrics: dict[str, float]) -> int:
    return sum(
        len(strict_utf8(key, "json_metrics metric path")) + 8
        for key in metrics
    )


def validate_json_tolerance_rules(comparison: dict[str, Any]) -> dict[str, Any]:
    tolerance_rules = comparison.get("tolerances")
    if not isinstance(tolerance_rules, dict) or not tolerance_rules:
        raise ManifestError("json_metrics needs a nonempty tolerances object")
    if len(tolerance_rules) > MAX_JSON_TOLERANCE_RULES:
        raise ManifestError(
            f"json_metrics has {len(tolerance_rules)} tolerance rules; limit "
            f"is {MAX_JSON_TOLERANCE_RULES}"
        )
    total_rule_bytes = 0
    for rule, tolerance in tolerance_rules.items():
        if not isinstance(rule, str) or not rule or not isinstance(tolerance, dict):
            raise ManifestError("invalid json_metrics tolerance")
        encoded_rule = strict_utf8(rule, "json_metrics tolerance rule")
        if len(encoded_rule) > MAX_JSON_METRIC_PATH_BYTES:
            raise ManifestError(
                "json_metrics tolerance rule exceeds the "
                f"{MAX_JSON_METRIC_PATH_BYTES}-byte path limit"
            )
        total_rule_bytes += len(encoded_rule)
        unknown_fields = set(tolerance) - {
            "atol",
            "rtol",
            "diagnostic_only",
        }
        if unknown_fields:
            raise ManifestError(
                f"json_metrics {rule} has unknown tolerance fields: "
                + ", ".join(sorted(unknown_fields))
            )
        for name in ("atol", "rtol"):
            value = tolerance.get(name, 0.0)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ManifestError(
                    f"json_metrics {rule} {name} must be finite and nonnegative"
                )
        if not isinstance(tolerance.get("diagnostic_only", False), bool):
            raise ManifestError(
                f"json_metrics {rule} diagnostic_only must be boolean"
            )
    if all(
        tolerance.get("diagnostic_only", False)
        for tolerance in tolerance_rules.values()
    ):
        raise ManifestError(
            "json_metrics needs at least one verdict-bearing tolerance"
        )
    if total_rule_bytes > MAX_JSON_TOLERANCE_RULE_BYTES_TOTAL:
        raise ManifestError(
            "json_metrics tolerance rule names exceed the "
            f"{MAX_JSON_TOLERANCE_RULE_BYTES_TOTAL}-byte budget"
        )
    return tolerance_rules


def finite_nonnegative_sum(first: float, second: float) -> float | None:
    if first > sys.float_info.max - second:
        return None
    return first + second


def finite_nonnegative_product(first: float, second: float) -> float | None:
    if first == 0.0 or second == 0.0:
        return 0.0
    if first > sys.float_info.max / second:
        return None
    return first * second


def finite_absolute_difference(first: float, second: float) -> float | None:
    if (first >= 0.0) == (second >= 0.0):
        return abs(first - second)
    return finite_nonnegative_sum(abs(first), abs(second))


def update_optional_float_digest(
    digest: Any, value: float | None
) -> None:
    if value is None:
        digest.update(b"\x00")
    else:
        if not math.isfinite(value):
            raise ManifestError("comparison derived a non-finite number")
        digest.update(b"\x01")
        digest.update(struct.pack(">d", value))


def compare_json_metrics(
    cpu_text: str,
    cuda_text: str,
    comparison: dict[str, Any],
    failure_sidecar: pathlib.Path | None = None,
) -> dict[str, Any]:
    cpu = extract_json_metrics(cpu_text, comparison)
    cuda = extract_json_metrics(cuda_text, comparison)
    if set(cpu) != set(cuda):
        cpu_only = sorted(set(cpu) - set(cuda))
        cuda_only = sorted(set(cuda) - set(cpu))
        return {
            "outcome": "MISMATCH",
            "reason": "CPU/CUDA metric keys differ",
            "metric_evidence": {
                "cpu_count": len(cpu),
                "cuda_count": len(cuda),
                "cpu_sha256": flattened_metrics_sha256(cpu),
                "cuda_sha256": flattened_metrics_sha256(cuda),
                "cpu_keys_sha256": flattened_metric_keys_sha256(cpu),
                "cuda_keys_sha256": flattened_metric_keys_sha256(cuda),
                "cpu_only_count": len(cpu_only),
                "cuda_only_count": len(cuda_only),
                "cpu_only_sample": cpu_only[:32],
                "cuda_only_sample": cuda_only[:32],
            },
        }

    tolerance_rules = validate_json_tolerance_rules(comparison)
    used_tolerances: set[str] = set()
    failures: list[dict[str, Any]] = []
    failure_count = 0
    diagnostic_only_count = 0
    diagnostic_only_out_of_tolerance_count = 0
    flattened_storage_bytes = flattened_metric_storage_bytes(cpu)
    inline_metrics = (
        len(cpu) <= MAX_INLINE_JSON_METRICS
        and 2 * flattened_storage_bytes <= MAX_INLINE_JSON_BYTES
    )
    differences: dict[str, dict[str, Any]] = {}
    difference_summary: dict[str, dict[str, Any]] = {}
    comparison_digest = hashlib.sha256()
    comparison_digest.update(b"gpmeep-json-comparison-records-v1\0")
    failure_digest = hashlib.sha256()
    failure_digest.update(b"gpmeep-json-failure-records-v1\0")
    failure_stream = None
    if failure_sidecar is not None:
        failure_sidecar.parent.mkdir(parents=True, exist_ok=True)
        failure_stream = failure_sidecar.open("wb")
    for metric in sorted(cpu):
        tolerance_rule = metric if metric in tolerance_rules else None
        if tolerance_rule is None:
            matching_rules = [
                rule
                for rule in tolerance_rules
                if rule != "*" and fnmatch.fnmatchcase(metric, rule)
            ]
            if len(matching_rules) > 1:
                raise ManifestError(
                    f"json_metrics metric {metric!r} matches ambiguous "
                    "tolerance rules: "
                    + ", ".join(sorted(matching_rules))
                )
            if matching_rules:
                tolerance_rule = matching_rules[0]
            elif "*" in tolerance_rules:
                tolerance_rule = "*"
        tolerance = (
            tolerance_rules[tolerance_rule]
            if tolerance_rule is not None
            else None
        )
        if tolerance is None:
            raise ManifestError(
                f"json_metrics has no tolerance for {metric!r}"
            )
        used_tolerances.add(tolerance_rule)
        atol = float(tolerance.get("atol", 0.0))
        rtol = float(tolerance.get("rtol", 0.0))
        diagnostic_only = bool(tolerance.get("diagnostic_only", False))
        if diagnostic_only:
            diagnostic_only_count += 1
        absolute = finite_absolute_difference(cpu[metric], cuda[metric])
        relative_allowance = finite_nonnegative_product(
            rtol, max(abs(cpu[metric]), abs(cuda[metric]))
        )
        allowed = (
            finite_nonnegative_sum(atol, relative_allowance)
            if relative_allowance is not None
            else None
        )
        arithmetic_overflow = absolute is None or allowed is None
        within_tolerance = (
            not arithmetic_overflow
            and absolute is not None
            and allowed is not None
            and absolute <= allowed
        )
        if diagnostic_only and not arithmetic_overflow and not within_tolerance:
            diagnostic_only_out_of_tolerance_count += 1
        passed = not arithmetic_overflow and (
            within_tolerance or diagnostic_only
        )
        for text_value in (metric, tolerance_rule):
            encoded_value = strict_utf8(text_value, "json_metrics comparison key")
            comparison_digest.update(len(encoded_value).to_bytes(8, "big"))
            comparison_digest.update(encoded_value)
        for numeric_value in (cpu[metric], cuda[metric], atol, rtol):
            comparison_digest.update(struct.pack(">d", numeric_value))
        update_optional_float_digest(comparison_digest, absolute)
        update_optional_float_digest(comparison_digest, allowed)
        if diagnostic_only:
            comparison_digest.update(b"\x02diagnostic-only-v1\0")
            comparison_digest.update(
                b"\x01" if within_tolerance else b"\x00"
            )
        comparison_digest.update(b"\x01" if passed else b"\x00")
        difference = {
            "absolute_error": absolute,
            "allowed_error": allowed,
            "tolerance_rule": tolerance_rule,
            "arithmetic_overflow": arithmetic_overflow,
        }
        if diagnostic_only:
            difference.update(
                {
                    "diagnostic_only": True,
                    "within_tolerance": within_tolerance,
                }
            )
        if inline_metrics:
            differences[metric] = difference
        summary = difference_summary.setdefault(
            tolerance_rule,
            {
                "metric_count": 0,
                "failure_count": 0,
                "zero_tolerance_failure_count": 0,
                "max_absolute_error": -1.0,
                "max_absolute_error_metric": None,
                "max_absolute_error_overflow": False,
                "max_normalized_error": -1.0,
                "max_normalized_error_metric": None,
                "max_normalized_error_overflow": False,
                "max_absolute_error_record": None,
                "max_normalized_error_record": None,
                "arithmetic_overflow_count": 0,
            },
        )
        summary["metric_count"] += 1
        if diagnostic_only:
            summary["diagnostic_only"] = True
            summary["diagnostic_only_out_of_tolerance_count"] = (
                summary.get(
                    "diagnostic_only_out_of_tolerance_count", 0
                )
                + int(not arithmetic_overflow and not within_tolerance)
            )
        if arithmetic_overflow:
            summary["arithmetic_overflow_count"] += 1
        if absolute is None:
            if not summary["max_absolute_error_overflow"]:
                summary["max_absolute_error"] = None
                summary["max_absolute_error_metric"] = metric
                summary["max_absolute_error_overflow"] = True
                summary["max_absolute_error_record"] = {
                    "metric": metric,
                    "cpu": cpu[metric],
                    "cuda": cuda[metric],
                    **difference,
                }
        elif (
            not summary["max_absolute_error_overflow"]
            and absolute > summary["max_absolute_error"]
        ):
            summary["max_absolute_error"] = absolute
            summary["max_absolute_error_metric"] = metric
            summary["max_absolute_error_record"] = {
                "metric": metric,
                "cpu": cpu[metric],
                "cuda": cuda[metric],
                **difference,
            }
        normalized_error: float | None = None
        normalized_overflow = False
        if absolute is not None and allowed is not None and allowed > 0:
            normalized_error = absolute / allowed
            if not math.isfinite(normalized_error):
                normalized_error = None
                normalized_overflow = True
            if normalized_overflow:
                if not summary["max_normalized_error_overflow"]:
                    summary["max_normalized_error"] = None
                    summary["max_normalized_error_metric"] = metric
                    summary["max_normalized_error_overflow"] = True
                    summary["max_normalized_error_record"] = {
                        "metric": metric,
                        "cpu": cpu[metric],
                        "cuda": cuda[metric],
                        "normalized_error": None,
                        "normalized_error_overflow": True,
                        **difference,
                    }
            elif (
                not summary["max_normalized_error_overflow"]
                and normalized_error is not None
                and normalized_error > summary["max_normalized_error"]
            ):
                summary["max_normalized_error"] = normalized_error
                summary["max_normalized_error_metric"] = metric
                summary["max_normalized_error_record"] = {
                    "metric": metric,
                    "cpu": cpu[metric],
                    "cuda": cuda[metric],
                    "normalized_error": normalized_error,
                    "normalized_error_overflow": False,
                    **difference,
                }
        elif (
            absolute == 0
            and allowed == 0
            and not summary["max_normalized_error_overflow"]
            and summary["max_normalized_error"] < 0
        ):
            summary["max_normalized_error"] = 0.0
            summary["max_normalized_error_metric"] = metric
            summary["max_normalized_error_record"] = {
                "metric": metric,
                "cpu": cpu[metric],
                "cuda": cuda[metric],
                "normalized_error": 0.0,
                "normalized_error_overflow": False,
                **difference,
            }
        if not passed:
            failure_count += 1
            summary["failure_count"] += 1
            if allowed == 0:
                summary["zero_tolerance_failure_count"] += 1
            failure_record = {
                "metric": metric,
                "cpu": cpu[metric],
                "cuda": cuda[metric],
                "absolute_error": absolute,
                "allowed_error": allowed,
                "arithmetic_overflow": arithmetic_overflow,
                "normalized_error": normalized_error,
                "normalized_error_overflow": normalized_overflow,
            }
            failure_bytes = json.dumps(
                failure_record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            failure_digest.update(len(failure_bytes).to_bytes(8, "big"))
            failure_digest.update(failure_bytes)
            if failure_stream is not None:
                failure_stream.write(failure_bytes + b"\n")
            if len(failures) < MAX_REPORTED_JSON_FAILURES:
                failures.append(failure_record)
    unused_tolerances = set(tolerance_rules) - {"*"} - used_tolerances
    if unused_tolerances:
        if failure_stream is not None:
            failure_stream.close()
            failure_sidecar.unlink(missing_ok=True)
        raise ManifestError(
            "json_metrics tolerances reference missing metrics: "
            + ", ".join(sorted(unused_tolerances))
        )
    if failure_stream is not None:
        failure_stream.flush()
        os.fsync(failure_stream.fileno())
        failure_stream.close()
        if failure_count == 0:
            failure_sidecar.unlink(missing_ok=True)
    result = {
        "outcome": "PASS" if failure_count == 0 else "MISMATCH",
        "reason": (
            (
                "all verdict-bearing JSON metrics are within tolerance; "
                f"{diagnostic_only_out_of_tolerance_count} diagnostic-only "
                "metric(s) exceed their reporting tolerance"
            )
            if failure_count == 0
            and diagnostic_only_out_of_tolerance_count
            else "all JSON metrics are within tolerance"
            if failure_count == 0
            else f"{failure_count} JSON metric(s) exceed tolerance"
        ),
        "metric_evidence": {
            "schema": METRIC_EVIDENCE_SCHEMA,
            "count": len(cpu),
            "passed_count": len(cpu) - failure_count,
            "failure_count": failure_count,
            "metric_keys_sha256": flattened_metric_keys_sha256(cpu),
            "cpu_sha256": flattened_metrics_sha256(cpu),
            "cuda_sha256": flattened_metrics_sha256(cuda),
            "comparison_records_sha256": comparison_digest.hexdigest(),
            "inline_limit": MAX_INLINE_JSON_METRICS,
            "inline_byte_limit": MAX_INLINE_JSON_BYTES,
            "flattened_storage_bytes": flattened_storage_bytes,
            "inlined": inline_metrics,
            **(
                {
                    "diagnostic_only_count": diagnostic_only_count,
                    "diagnostic_only_out_of_tolerance_count": (
                        diagnostic_only_out_of_tolerance_count
                    ),
                }
                if diagnostic_only_count
                else {}
            ),
        },
        "difference_summary": difference_summary,
        "failure_count": failure_count,
        "failures_reported": len(failures),
        "failures_truncated": failure_count > len(failures),
        "failure_records_sha256": failure_digest.hexdigest(),
        "failures": failures,
    }
    if inline_metrics:
        result.update(
            {
                "cpu_metrics": cpu,
                "cuda_metrics": cuda,
                "differences": differences,
            }
        )
    return result


def _update_metric_digests(
    metrics_digest: Any,
    keys_digest: Any,
    key: bytes,
    value: float,
) -> None:
    for digest in (metrics_digest, keys_digest):
        digest.update(len(key).to_bytes(8, "big"))
        digest.update(key)
    metrics_digest.update(struct.pack(">d", value))


def _compare_metric_sources(
    cpu_source: _MetricSource,
    cuda_source: _MetricSource,
    comparison: dict[str, Any],
    failure_sidecar: pathlib.Path | None,
    *,
    aligned_document_order: bool = False,
) -> dict[str, Any]:
    """Compare two replayable full metric streams.

    The normal sources are path-sorted.  Direct sources preserve JSON document
    order and are accepted only when CPU and CUDA yield the exact same path at
    every position; any ordering difference requests the semantic-preserving
    external-sort fallback before comparison results are emitted.
    """
    cpu_metric_digest = hashlib.sha256(b"gpmeep-json-metrics-v1\0")
    cuda_metric_digest = hashlib.sha256(b"gpmeep-json-metrics-v1\0")
    cpu_key_digest = hashlib.sha256(b"gpmeep-json-metric-keys-v1\0")
    cuda_key_digest = hashlib.sha256(b"gpmeep-json-metric-keys-v1\0")
    cpu_count = 0
    cuda_count = 0
    cpu_storage_bytes = 0
    cuda_storage_bytes = 0
    cpu_only_count = 0
    cuda_only_count = 0
    cpu_only_sample: list[str] = []
    cuda_only_sample: list[str] = []
    cpu_observed_multiset = _MetricMultisetAccumulator()
    cuda_observed_multiset = _MetricMultisetAccumulator()
    cpu_iter = iter(cpu_source.iter_metrics())
    cuda_iter = iter(cuda_source.iter_metrics())
    if aligned_document_order:
        while True:
            cpu_item = next(cpu_iter, None)
            cuda_item = next(cuda_iter, None)
            if cpu_item is None or cuda_item is None:
                if cpu_item is not None or cuda_item is not None:
                    raise _FastPathUnavailable(
                        "CPU/CUDA direct metric counts differ"
                    )
                break
            cpu_key, cpu_value = cpu_item
            cuda_key, cuda_value = cuda_item
            if cpu_key != cuda_key:
                raise _FastPathUnavailable(
                    "CPU/CUDA direct metric path order differs"
                )
            _update_metric_digests(
                cpu_metric_digest, cpu_key_digest, cpu_key, cpu_value
            )
            _update_metric_digests(
                cuda_metric_digest, cuda_key_digest, cuda_key, cuda_value
            )
            cpu_observed_multiset.add_metric(cpu_key, cpu_value)
            cuda_observed_multiset.add_metric(cuda_key, cuda_value)
            cpu_count += 1
            cuda_count += 1
            cpu_storage_bytes += len(cpu_key) + 8
            cuda_storage_bytes += len(cuda_key) + 8
    else:
        cpu_item = next(cpu_iter, None)
        cuda_item = next(cuda_iter, None)
        while cpu_item is not None or cuda_item is not None:
            if cuda_item is None or (
                cpu_item is not None and cpu_item[0] < cuda_item[0]
            ):
                assert cpu_item is not None
                key, value = cpu_item
                _update_metric_digests(
                    cpu_metric_digest, cpu_key_digest, key, value
                )
                cpu_observed_multiset.add_metric(key, value)
                cpu_count += 1
                cpu_storage_bytes += len(key) + 8
                cpu_only_count += 1
                if len(cpu_only_sample) < 32:
                    cpu_only_sample.append(
                        key.decode("utf-8", errors="strict")
                    )
                cpu_item = next(cpu_iter, None)
            elif cpu_item is None or cuda_item[0] < cpu_item[0]:
                key, value = cuda_item
                _update_metric_digests(
                    cuda_metric_digest, cuda_key_digest, key, value
                )
                cuda_observed_multiset.add_metric(key, value)
                cuda_count += 1
                cuda_storage_bytes += len(key) + 8
                cuda_only_count += 1
                if len(cuda_only_sample) < 32:
                    cuda_only_sample.append(
                        key.decode("utf-8", errors="strict")
                    )
                cuda_item = next(cuda_iter, None)
            else:
                assert cpu_item is not None and cuda_item is not None
                cpu_key, cpu_value = cpu_item
                cuda_key, cuda_value = cuda_item
                _update_metric_digests(
                    cpu_metric_digest, cpu_key_digest, cpu_key, cpu_value
                )
                _update_metric_digests(
                    cuda_metric_digest, cuda_key_digest, cuda_key, cuda_value
                )
                cpu_observed_multiset.add_metric(cpu_key, cpu_value)
                cuda_observed_multiset.add_metric(cuda_key, cuda_value)
                cpu_count += 1
                cuda_count += 1
                cpu_storage_bytes += len(cpu_key) + 8
                cuda_storage_bytes += len(cuda_key) + 8
                cpu_item = next(cpu_iter, None)
                cuda_item = next(cuda_iter, None)

    for label, source, observed in (
        ("CPU", cpu_source, cpu_observed_multiset),
        ("CUDA", cuda_source, cuda_observed_multiset),
    ):
        if (
            source.origin_binding is not None
            and observed.binding() != source.origin_binding
        ):
            raise ManifestError(
                f"{label} json_metrics comparison stream differs from parser origin"
            )

    if cpu_only_count or cuda_only_count:
        return {
            "outcome": "MISMATCH",
            "reason": "CPU/CUDA metric keys differ",
            "metric_evidence": {
                "cpu_count": cpu_count,
                "cuda_count": cuda_count,
                "cpu_sha256": cpu_metric_digest.hexdigest(),
                "cuda_sha256": cuda_metric_digest.hexdigest(),
                "cpu_keys_sha256": cpu_key_digest.hexdigest(),
                "cuda_keys_sha256": cuda_key_digest.hexdigest(),
                "cpu_only_count": cpu_only_count,
                "cuda_only_count": cuda_only_count,
                "cpu_only_sample": cpu_only_sample,
                "cuda_only_sample": cuda_only_sample,
            },
        }
    if cpu_count == 0:
        raise ManifestError("json_metrics payload must be a nonempty object")
    if cpu_count != cuda_count or cpu_storage_bytes != cuda_storage_bytes:
        raise ManifestError("json_metrics sorted source accounting disagrees")

    tolerance_rules = validate_json_tolerance_rules(comparison)
    used_tolerances: set[str] = set()
    failures: list[dict[str, Any]] = []
    failure_count = 0
    diagnostic_only_count = 0
    diagnostic_only_out_of_tolerance_count = 0
    inline_metrics = (
        cpu_count <= MAX_INLINE_JSON_METRICS
        and 2 * cpu_storage_bytes <= MAX_INLINE_JSON_BYTES
    )
    cpu_metrics: dict[str, float] = {}
    cuda_metrics: dict[str, float] = {}
    differences: dict[str, dict[str, Any]] = {}
    difference_summary: dict[str, dict[str, Any]] = {}
    comparison_digest = hashlib.sha256(
        b"gpmeep-json-comparison-records-v1\0"
    )
    failure_digest = hashlib.sha256(
        b"gpmeep-json-failure-records-v1\0"
    )
    failure_stream: BinaryIO | None = None
    if failure_sidecar is not None:
        failure_sidecar.parent.mkdir(parents=True, exist_ok=True)
        failure_stream = failure_sidecar.open("wb")
    try:
        cpu_second = cpu_source.iter_metrics()
        cuda_second = cuda_source.iter_metrics()
        cpu_replay_digest = hashlib.sha256(b"gpmeep-json-metrics-v1\0")
        cuda_replay_digest = hashlib.sha256(b"gpmeep-json-metrics-v1\0")
        cpu_replay_keys_digest = hashlib.sha256(
            b"gpmeep-json-metric-keys-v1\0"
        )
        cuda_replay_keys_digest = hashlib.sha256(
            b"gpmeep-json-metric-keys-v1\0"
        )
        for replay_index in range(cpu_count):
            cpu_item = next(cpu_second, None)
            cuda_item = next(cuda_second, None)
            if cpu_item is None or cuda_item is None:
                raise ManifestError(
                    "json_metrics replay changed between comparison passes"
                )
            metric_bytes, cpu_value = cpu_item
            cuda_metric_bytes, cuda_value = cuda_item
            if metric_bytes != cuda_metric_bytes:
                raise ManifestError(
                    "json_metrics replay changed between comparison passes "
                    f"at record {replay_index}: CPU key SHA-256 "
                    f"{hashlib.sha256(metric_bytes).hexdigest()}, CUDA key "
                    f"SHA-256 {hashlib.sha256(cuda_metric_bytes).hexdigest()}"
                )
            _update_metric_digests(
                cpu_replay_digest,
                cpu_replay_keys_digest,
                metric_bytes,
                cpu_value,
            )
            _update_metric_digests(
                cuda_replay_digest,
                cuda_replay_keys_digest,
                cuda_metric_bytes,
                cuda_value,
            )
            metric = metric_bytes.decode("utf-8", errors="strict")
            tolerance_rule = metric if metric in tolerance_rules else None
            if tolerance_rule is None:
                matching_rules = [
                    rule
                    for rule in tolerance_rules
                    if rule != "*" and fnmatch.fnmatchcase(metric, rule)
                ]
                if len(matching_rules) > 1:
                    raise ManifestError(
                        f"json_metrics metric {metric!r} matches ambiguous "
                        "tolerance rules: "
                        + ", ".join(sorted(matching_rules))
                    )
                if matching_rules:
                    tolerance_rule = matching_rules[0]
                elif "*" in tolerance_rules:
                    tolerance_rule = "*"
            tolerance = (
                tolerance_rules[tolerance_rule]
                if tolerance_rule is not None
                else None
            )
            if tolerance is None:
                raise ManifestError(
                    f"json_metrics has no tolerance for {metric!r}"
                )
            used_tolerances.add(tolerance_rule)
            atol = float(tolerance.get("atol", 0.0))
            rtol = float(tolerance.get("rtol", 0.0))
            diagnostic_only = bool(tolerance.get("diagnostic_only", False))
            if diagnostic_only:
                diagnostic_only_count += 1
            absolute = finite_absolute_difference(cpu_value, cuda_value)
            relative_allowance = finite_nonnegative_product(
                rtol, max(abs(cpu_value), abs(cuda_value))
            )
            allowed = (
                finite_nonnegative_sum(atol, relative_allowance)
                if relative_allowance is not None
                else None
            )
            arithmetic_overflow = absolute is None or allowed is None
            within_tolerance = (
                not arithmetic_overflow
                and absolute is not None
                and allowed is not None
                and absolute <= allowed
            )
            if (
                diagnostic_only
                and not arithmetic_overflow
                and not within_tolerance
            ):
                diagnostic_only_out_of_tolerance_count += 1
            passed = not arithmetic_overflow and (
                within_tolerance or diagnostic_only
            )
            for text_value in (metric, tolerance_rule):
                encoded_value = strict_utf8(
                    text_value, "json_metrics comparison key"
                )
                comparison_digest.update(
                    len(encoded_value).to_bytes(8, "big")
                )
                comparison_digest.update(encoded_value)
            for numeric_value in (cpu_value, cuda_value, atol, rtol):
                comparison_digest.update(struct.pack(">d", numeric_value))
            update_optional_float_digest(comparison_digest, absolute)
            update_optional_float_digest(comparison_digest, allowed)
            if diagnostic_only:
                comparison_digest.update(b"\x02diagnostic-only-v1\0")
                comparison_digest.update(
                    b"\x01" if within_tolerance else b"\x00"
                )
            comparison_digest.update(b"\x01" if passed else b"\x00")
            difference = {
                "absolute_error": absolute,
                "allowed_error": allowed,
                "tolerance_rule": tolerance_rule,
                "arithmetic_overflow": arithmetic_overflow,
            }
            if diagnostic_only:
                difference.update(
                    {
                        "diagnostic_only": True,
                        "within_tolerance": within_tolerance,
                    }
                )
            if inline_metrics:
                cpu_metrics[metric] = cpu_value
                cuda_metrics[metric] = cuda_value
                differences[metric] = difference
            summary = difference_summary.setdefault(
                tolerance_rule,
                {
                    "metric_count": 0,
                    "failure_count": 0,
                    "zero_tolerance_failure_count": 0,
                    "max_absolute_error": -1.0,
                    "max_absolute_error_metric": None,
                    "max_absolute_error_overflow": False,
                    "max_normalized_error": -1.0,
                    "max_normalized_error_metric": None,
                    "max_normalized_error_overflow": False,
                    "max_absolute_error_record": None,
                    "max_normalized_error_record": None,
                    "arithmetic_overflow_count": 0,
                },
            )
            summary["metric_count"] += 1
            if diagnostic_only:
                summary["diagnostic_only"] = True
                summary["diagnostic_only_out_of_tolerance_count"] = (
                    summary.get(
                        "diagnostic_only_out_of_tolerance_count", 0
                    )
                    + int(not arithmetic_overflow and not within_tolerance)
                )
            if arithmetic_overflow:
                summary["arithmetic_overflow_count"] += 1
            if absolute is None:
                if not summary["max_absolute_error_overflow"]:
                    summary["max_absolute_error"] = None
                    summary["max_absolute_error_metric"] = metric
                    summary["max_absolute_error_overflow"] = True
                    summary["max_absolute_error_record"] = {
                        "metric": metric,
                        "cpu": cpu_value,
                        "cuda": cuda_value,
                        **difference,
                    }
            elif (
                not summary["max_absolute_error_overflow"]
                and absolute > summary["max_absolute_error"]
            ):
                summary["max_absolute_error"] = absolute
                summary["max_absolute_error_metric"] = metric
                summary["max_absolute_error_record"] = {
                    "metric": metric,
                    "cpu": cpu_value,
                    "cuda": cuda_value,
                    **difference,
                }
            normalized_error: float | None = None
            normalized_overflow = False
            if absolute is not None and allowed is not None and allowed > 0:
                normalized_error = absolute / allowed
                if not math.isfinite(normalized_error):
                    normalized_error = None
                    normalized_overflow = True
                if normalized_overflow:
                    if not summary["max_normalized_error_overflow"]:
                        summary["max_normalized_error"] = None
                        summary["max_normalized_error_metric"] = metric
                        summary["max_normalized_error_overflow"] = True
                        summary["max_normalized_error_record"] = {
                            "metric": metric,
                            "cpu": cpu_value,
                            "cuda": cuda_value,
                            "normalized_error": None,
                            "normalized_error_overflow": True,
                            **difference,
                        }
                elif (
                    not summary["max_normalized_error_overflow"]
                    and normalized_error is not None
                    and normalized_error > summary["max_normalized_error"]
                ):
                    summary["max_normalized_error"] = normalized_error
                    summary["max_normalized_error_metric"] = metric
                    summary["max_normalized_error_record"] = {
                        "metric": metric,
                        "cpu": cpu_value,
                        "cuda": cuda_value,
                        "normalized_error": normalized_error,
                        "normalized_error_overflow": False,
                        **difference,
                    }
            elif (
                absolute == 0
                and allowed == 0
                and not summary["max_normalized_error_overflow"]
                and summary["max_normalized_error"] < 0
            ):
                summary["max_normalized_error"] = 0.0
                summary["max_normalized_error_metric"] = metric
                summary["max_normalized_error_record"] = {
                    "metric": metric,
                    "cpu": cpu_value,
                    "cuda": cuda_value,
                    "normalized_error": 0.0,
                    "normalized_error_overflow": False,
                    **difference,
                }
            if not passed:
                failure_count += 1
                summary["failure_count"] += 1
                if allowed == 0:
                    summary["zero_tolerance_failure_count"] += 1
                failure_record = {
                    "metric": metric,
                    "cpu": cpu_value,
                    "cuda": cuda_value,
                    "absolute_error": absolute,
                    "allowed_error": allowed,
                    "arithmetic_overflow": arithmetic_overflow,
                    "normalized_error": normalized_error,
                    "normalized_error_overflow": normalized_overflow,
                }
                failure_bytes = json.dumps(
                    failure_record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                failure_digest.update(
                    len(failure_bytes).to_bytes(8, "big")
                )
                failure_digest.update(failure_bytes)
                if failure_stream is not None:
                    failure_stream.write(failure_bytes + b"\n")
                if len(failures) < MAX_REPORTED_JSON_FAILURES:
                    failures.append(failure_record)
        cpu_replay_extra = next(cpu_second, None)
        cuda_replay_extra = next(cuda_second, None)
        if cpu_replay_extra is not None or cuda_replay_extra is not None:
            raise ManifestError(
                "json_metrics replay changed between comparison passes"
            )
        for label, source, observed in (
            ("CPU", cpu_source, cpu_observed_multiset),
            ("CUDA", cuda_source, cuda_observed_multiset),
        ):
            if (
                source.origin_binding is not None
                and source.origin_binding != observed.binding()
            ):
                raise ManifestError(
                    f"{label} json_metrics replay differs from parser origin"
                )
        if (
            cpu_replay_digest.hexdigest() != cpu_metric_digest.hexdigest()
            or cuda_replay_digest.hexdigest()
            != cuda_metric_digest.hexdigest()
            or cpu_replay_keys_digest.hexdigest()
            != cpu_key_digest.hexdigest()
            or cuda_replay_keys_digest.hexdigest()
            != cuda_key_digest.hexdigest()
        ):
            raise ManifestError(
                "json_metrics replay changed between comparison passes"
            )
    finally:
        if failure_stream is not None:
            failure_stream.flush()
            os.fsync(failure_stream.fileno())
            failure_stream.close()

    unused_tolerances = set(tolerance_rules) - {"*"} - used_tolerances
    if unused_tolerances:
        if failure_sidecar is not None:
            failure_sidecar.unlink(missing_ok=True)
        raise ManifestError(
            "json_metrics tolerances reference missing metrics: "
            + ", ".join(sorted(unused_tolerances))
        )
    if failure_count == 0 and failure_sidecar is not None:
        failure_sidecar.unlink(missing_ok=True)
    result = {
        "outcome": "PASS" if failure_count == 0 else "MISMATCH",
        "reason": (
            (
                "all verdict-bearing JSON metrics are within tolerance; "
                f"{diagnostic_only_out_of_tolerance_count} diagnostic-only "
                "metric(s) exceed their reporting tolerance"
            )
            if failure_count == 0
            and diagnostic_only_out_of_tolerance_count
            else "all JSON metrics are within tolerance"
            if failure_count == 0
            else f"{failure_count} JSON metric(s) exceed tolerance"
        ),
        "metric_evidence": {
            "schema": (
                DIRECT_METRIC_EVIDENCE_SCHEMA
                if aligned_document_order
                else METRIC_EVIDENCE_SCHEMA
            ),
            **(
                {
                    "digest_order": "document-order",
                    "comparison_strategy": "aligned-direct",
                }
                if aligned_document_order
                else {}
            ),
            "count": cpu_count,
            "passed_count": cpu_count - failure_count,
            "failure_count": failure_count,
            "metric_keys_sha256": cpu_key_digest.hexdigest(),
            "cpu_sha256": cpu_metric_digest.hexdigest(),
            "cuda_sha256": cuda_metric_digest.hexdigest(),
            "comparison_records_sha256": comparison_digest.hexdigest(),
            "inline_limit": MAX_INLINE_JSON_METRICS,
            "inline_byte_limit": MAX_INLINE_JSON_BYTES,
            "flattened_storage_bytes": cpu_storage_bytes,
            "inlined": inline_metrics,
            **(
                {
                    "diagnostic_only_count": diagnostic_only_count,
                    "diagnostic_only_out_of_tolerance_count": (
                        diagnostic_only_out_of_tolerance_count
                    ),
                }
                if diagnostic_only_count
                else {}
            ),
        },
        "difference_summary": difference_summary,
        "failure_count": failure_count,
        "failures_reported": len(failures),
        "failures_truncated": failure_count > len(failures),
        "failure_records_sha256": failure_digest.hexdigest(),
        "failures": failures,
    }
    if inline_metrics:
        result.update(
            {
                "cpu_metrics": cpu_metrics,
                "cuda_metrics": cuda_metrics,
                "differences": differences,
            }
        )
    return result


def compare_json_metric_files(
    cpu_stdout: pathlib.Path,
    cuda_stdout: pathlib.Path,
    comparison: dict[str, Any],
    failure_sidecar: pathlib.Path | None = None,
    temporary_parent: pathlib.Path | None = None,
    expected_cpu_stdout: dict[str, Any] | None = None,
    expected_cuda_stdout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict full-vector comparison with bounded resident memory.

    Both JSON documents are parsed completely and independently.  Large files
    with identical flattened-path order are compared directly with no derived
    metric files.  Any path-order difference or direct-parser ambiguity is
    replayed through external sorting, so JSON object order still has no
    semantic effect.  Both strategies visit and bind every CPU/GPU scalar.
    """
    validate_json_tolerance_rules(comparison)
    if temporary_parent is not None:
        temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="gpmeep-json-compare-",
        dir=str(temporary_parent) if temporary_parent is not None else None,
    ) as temporary:
        temporary_root = pathlib.Path(temporary)
        cpu_snapshot = temporary_root / "cpu-stdout.snapshot"
        cuda_snapshot = temporary_root / "cuda-stdout.snapshot"
        snapshots: list[_StableMetricSnapshot] = []
        try:
            cpu_stable = _copy_verified_metric_evidence(
                cpu_stdout, cpu_snapshot, expected_cpu_stdout
            )
            snapshots.append(cpu_stable)
            cuda_stable = _copy_verified_metric_evidence(
                cuda_stdout, cuda_snapshot, expected_cuda_stdout
            )
            snapshots.append(cuda_stable)
            use_direct = max(
                cpu_stable.record["size_bytes"],
                cuda_stable.record["size_bytes"],
            ) > DIRECT_JSON_METRIC_MIN_PAYLOAD_BYTES
            if use_direct:
                try:
                    cpu_source = _DirectMetricSource(cpu_stable, comparison)
                    cuda_source = _DirectMetricSource(cuda_stable, comparison)
                    result = _compare_metric_sources(
                        cpu_source,
                        cuda_source,
                        comparison,
                        failure_sidecar,
                        aligned_document_order=True,
                    )
                except _FastPathUnavailable:
                    cpu_source = _streaming_metric_source(
                        cpu_stable, comparison, temporary_root, "cpu"
                    )
                    cuda_source = _streaming_metric_source(
                        cuda_stable, comparison, temporary_root, "cuda"
                    )
                    result = _compare_metric_sources(
                        cpu_source,
                        cuda_source,
                        comparison,
                        failure_sidecar,
                    )
            else:
                cpu_source = _streaming_metric_source(
                    cpu_stable, comparison, temporary_root, "cpu"
                )
                cuda_source = _streaming_metric_source(
                    cuda_stable, comparison, temporary_root, "cuda"
                )
                result = _compare_metric_sources(
                    cpu_source, cuda_source, comparison, failure_sidecar
                )
            cpu_stable.verify("CPU")
            cuda_stable.verify("CUDA")
            return result
        finally:
            for snapshot in reversed(snapshots):
                snapshot.close()


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        if proc.poll() is None:
            proc.terminate()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        if proc.poll() is None:
            proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        if proc.poll() is None:
            proc.kill()


def run_bounded_process(
    command: list[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str],
    timeout_seconds: int,
    stdout_file: pathlib.Path,
    stderr_file: pathlib.Path,
    stdout_limit: int | None = None,
    stderr_limit: int | None = None,
    combined_limit: int | None = None,
) -> dict[str, Any]:
    """Stream a child to durable evidence files with fail-closed byte caps."""
    limits = {
        "stdout": (
            MAX_BACKEND_STDOUT_BYTES if stdout_limit is None else stdout_limit
        ),
        "stderr": (
            MAX_BACKEND_STDERR_BYTES if stderr_limit is None else stderr_limit
        ),
    }
    effective_combined_limit = (
        MAX_BACKEND_OUTPUT_BYTES if combined_limit is None else combined_limit
    )
    sizes = {"stdout": 0, "stderr": 0}
    output_limit: str | None = None
    timed_out = False
    spawn_error: str | None = None
    exit_code: int | None = None
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    with stdout_file.open("xb", buffering=0) as stdout_destination, \
        stderr_file.open("xb", buffering=0) as stderr_destination:
        destinations = {
            "stdout": stdout_destination,
            "stderr": stderr_destination,
        }
        try:
            proc = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            spawn_error = str(exc)
            encoded = spawn_error.encode("utf-8", errors="replace")
            stderr_destination.write(encoded[: limits["stderr"]])
            sizes["stderr"] = min(len(encoded), limits["stderr"])
        else:
            assert proc.stdout is not None and proc.stderr is not None
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
            selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
            deadline = time.monotonic() + timeout_seconds
            terminated = False
            forced_drain_deadline: float | None = None
            try:
                while selector.get_map() or proc.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 and not terminated:
                        timed_out = True
                        terminated = True
                        _terminate_process_group(proc)
                        forced_drain_deadline = time.monotonic() + 0.25
                    if (
                        terminated
                        and forced_drain_deadline is not None
                        and time.monotonic() >= forced_drain_deadline
                    ):
                        # A descendant can escape the child's process group and
                        # keep inherited pipe writers open forever.  After a
                        # bounded drain window, close our read ends so a failed
                        # run can never hang waiting for hostile descendants.
                        for key in list(selector.get_map().values()):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    if selector.get_map():
                        events = selector.select(
                            timeout=max(0.0, min(0.1, remaining))
                            if not terminated
                            else 0.1
                        )
                    else:
                        # A child can close both pipes and continue running.
                        # Keep the deadline loop alive instead of entering an
                        # unbounded wait with no descriptors registered.
                        time.sleep(
                            0.01
                            if terminated
                            else max(0.0, min(0.01, remaining))
                        )
                        events = []
                    for key, _ in events:
                        stream_name = key.data
                        try:
                            chunk = os.read(
                                key.fileobj.fileno(), MAX_CAPTURE_READ_BYTES
                            )
                        except OSError:
                            chunk = b""
                        if not chunk:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        individual_remaining = max(
                            0, limits[stream_name] - sizes[stream_name]
                        )
                        total_remaining = max(
                            0,
                            effective_combined_limit - sum(sizes.values()),
                        )
                        accepted = min(
                            len(chunk), individual_remaining, total_remaining
                        )
                        if accepted:
                            destinations[stream_name].write(chunk[:accepted])
                            sizes[stream_name] += accepted
                        if accepted != len(chunk) and output_limit is None:
                            output_limit = stream_name
                            if not terminated:
                                terminated = True
                                _terminate_process_group(proc)
                                forced_drain_deadline = time.monotonic() + 0.25
                exit_code = proc.poll()
            finally:
                selector.close()
                for stream in (proc.stdout, proc.stderr):
                    if not stream.closed:
                        stream.close()
                if proc.poll() is None:
                    _terminate_process_group(proc)
                try:
                    exit_code = proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired as exc:
                    proc.kill()
                    try:
                        exit_code = proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired as final_exc:
                        raise RuntimeError(
                            "child process could not be reaped after termination"
                        ) from final_exc
        for destination in destinations.values():
            destination.flush()
            os.fsync(destination.fileno())
    return {
        "exit_code": exit_code,
        "timeout": timed_out,
        "output_limit": output_limit,
        "spawn_error": spawn_error,
        "stdout_size_bytes": sizes["stdout"],
        "stderr_size_bytes": sizes["stderr"],
    }


def is_strict_utf8_file(path: pathlib.Path) -> bool:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        with path.open("rb") as source:
            for block in iter(
                lambda: source.read(MAX_CAPTURE_READ_BYTES), b""
            ):
                decoder.decode(block, final=False)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def fsync_evidence_directories(
    root: pathlib.Path, results: Iterable[dict[str, Any]]
) -> None:
    directories = {root.resolve()}
    root_resolved = root.resolve()
    for result in results:
        for run in result.get("runs", {}).values():
            records = run.get("evidence_files", {})
            for record in records.values():
                member = record.get("path") if isinstance(record, dict) else None
                if not isinstance(member, str):
                    continue
                current = resolve_evidence_member(root, member).parent.resolve()
                while current != root_resolved:
                    directories.add(current)
                    current = current.parent
        failure = (result.get("comparison") or {}).get("failure_evidence")
        if isinstance(failure, dict) and isinstance(failure.get("path"), str):
            directories.add(
                resolve_evidence_member(root, failure["path"]).parent.resolve()
            )
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        descriptor = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def run_backend(
    case: dict[str, Any],
    backend: str,
    repo: pathlib.Path,
    python: pathlib.Path,
    build_python: pathlib.Path,
    hook_dir: pathlib.Path,
    result_dir: pathlib.Path,
    expected_runtime: dict[str, Any],
) -> dict[str, Any]:
    safe_id = case_directory_id(case["id"])
    case_dir = result_dir / "runs" / safe_id / backend
    work_dir = case_dir / "work"
    case_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    stats_file = case_dir / "gpu-statistics.json"
    stdout_file = case_dir / "stdout.log"
    stderr_file = case_dir / "stderr.log"
    for stale_path in (stats_file, stdout_file, stderr_file):
        stale_path.unlink(missing_ok=True)
    run_nonce = uuid.uuid4().hex
    runtime_parent = case_dir / "runtime"
    if runtime_parent.exists():
        shutil.rmtree(runtime_parent)
    runtime_root = (runtime_parent / run_nonce).resolve()
    result_root = result_dir.resolve()
    try:
        runtime_root.relative_to(result_root)
    except ValueError as exc:
        raise ProvenanceError(
            "validation runtime directory escaped the evidence root"
        ) from exc
    sealed_roots = [
        pathlib.Path(expected_runtime[key]).resolve()
        for key in (
            "build_home",
            "qualification_home",
            "installed_environment",
        )
    ]
    if any(
        runtime_root == sealed_root or sealed_root in runtime_root.parents
        for sealed_root in sealed_roots
    ):
        raise ProvenanceError(
            "validation runtime directory overlaps receipt-sealed state"
        )
    runtime_home = runtime_root / "home"
    runtime_cache = runtime_root / "cache"
    runtime_config = runtime_root / "config"
    runtime_matplotlib = runtime_root / "matplotlib"
    runtime_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    for path in (
        runtime_home,
        runtime_cache,
        runtime_config,
        runtime_matplotlib,
    ):
        path.mkdir(mode=0o700, exist_ok=False)

    ambient = os.environ
    preserved_cuda = {
        key: ambient[key]
        for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        if key in ambient
    }
    env = isolated_process_environment()
    env.update(preserved_cuda)
    python_paths = [str(hook_dir), str(build_python)]
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(runtime_home),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(runtime_matplotlib),
            "XDG_CACHE_HOME": str(runtime_cache),
            "XDG_CONFIG_HOME": str(runtime_config),
            "FONTCONFIG_FILE": str(expected_runtime["fontconfig_file"]["path"]),
            "CUDA_CACHE_DISABLE": "1",
            "JAX_PLATFORMS": "cpu",
            "MEEP_GPU_BACKEND": backend,
            "GPMEEP_VALIDATION_STATS_FILE": str(stats_file),
            "GPMEEP_VALIDATION_EXPECTED_BACKEND": backend,
            "GPMEEP_VALIDATION_RUN_NONCE": run_nonce,
            "GPMEEP_VALIDATION_BUILD_RECEIPT_ID": expected_runtime["receipt_id"],
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
        }
    )
    trusted_library_directories = []
    for candidate in (
        pathlib.Path(expected_runtime["libmeep"]["path"]).resolve().parent,
        pathlib.Path(expected_runtime["installed_environment"]).resolve()
        / "lib",
    ):
        if candidate.is_dir() and str(candidate) not in trusted_library_directories:
            trusted_library_directories.append(str(candidate))
    if trusted_library_directories:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(trusted_library_directories)
    if backend == "cuda":
        # In gpmeep, selecting "cuda" is the strict mode: unsupported work must
        # throw rather than fall back.  This marker is recorded by the hook too.
        env["GPMEEP_VALIDATION_STRICT_CUDA"] = "1"

    command = case_command(case, python, repo)
    started_at_utc = utc_now()
    started = time.monotonic()
    capture = run_bounded_process(
        command,
        cwd=work_dir,
        env=env,
        timeout_seconds=case["timeout_seconds"],
        stdout_file=stdout_file,
        stderr_file=stderr_file,
    )
    timeout = capture["timeout"]
    output_limit = capture["output_limit"]
    exit_code = capture["exit_code"]
    invalid_utf8 = not (
        is_strict_utf8_file(stdout_file)
        and is_strict_utf8_file(stderr_file)
    )
    stdout = ""
    stderr = ""
    if output_limit is None and not invalid_utf8:
        try:
            # unittest identity parsing requires the complete streams.  JSON
            # metric examples are parsed later directly from the bounded files.
            if case["kind"] == "unittest":
                stdout = stdout_file.read_text(encoding="utf-8")
                stderr = stderr_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            invalid_utf8 = True
    if capture["spawn_error"] is not None:
        stderr = capture["spawn_error"]
    duration = time.monotonic() - started
    if stats_file.is_file():
        descriptor = os.open(stats_file, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    stats = read_stats(stats_file)
    evidence_files: dict[str, dict[str, Any]] = {}
    for name, path in (
        ("stdout", stdout_file),
        ("stderr", stderr_file),
        ("statistics", stats_file),
    ):
        if path.is_file():
            evidence_files[name] = {
                "available": True,
                **evidence_file_record(path, result_dir),
            }
        else:
            evidence_files[name] = {
                "available": False,
                "path": (
                    path.resolve().relative_to(result_dir.resolve()).as_posix()
                ),
            }
    combined_output = stdout + "\n" + stderr
    skips = (
        parse_unittest_skips(combined_output)
        if case["kind"] == "unittest"
        else 0
    )
    skip_details = (
        parse_unittest_skip_details(combined_output)
        if case["kind"] == "unittest"
        else []
    )
    if case["kind"] == "unittest":
        (
            unittest_contract_ok,
            unittest_evidence,
            unittest_contract_problems,
        ) = verify_unittest_test_contract(case, stderr)
    else:
        unittest_contract_ok = True
        unittest_evidence = {
            "reported_counts": [],
            "reported_count": None,
            "identities": [],
            "expected_count": None,
            "expected_identities": None,
            "terminal_summaries": [],
        }
        unittest_contract_problems = []
    allowed_skips = case.get("allowed_unittest_skips", 0)
    process_ok = (
        exit_code == 0
        and not timeout
        and output_limit is None
        and not invalid_utf8
    )
    skip_identity_ok, skip_policy_problems = verify_unittest_skip_policy(
        case, skips, skip_details
    )
    skips_ok = skips <= allowed_skips and skip_identity_ok
    contract_ok, contract_problems = verify_backend_contract(
        case, backend, stats, expected_runtime, run_nonce
    )
    outcome = "PASS"
    if timeout:
        outcome = "TIMEOUT"
    elif output_limit is not None:
        outcome = "OUTPUT_LIMIT"
    elif invalid_utf8:
        outcome = "OUTPUT_INVALID_UTF8"
    elif not process_ok:
        outcome = "PROCESS_FAILED"
    elif not skips_ok:
        outcome = "UNDECLARED_SKIP"
    elif not unittest_contract_ok:
        outcome = "UNITTEST_CONTRACT_FAILED"
    elif not contract_ok:
        outcome = "BACKEND_CONTRACT_FAILED"
    elif skips:
        outcome = "PASS_WITH_DECLARED_SKIPS"
    return {
        "backend": backend,
        "command": command,
        "cwd": work_dir.resolve().relative_to(result_dir.resolve()).as_posix(),
        "evidence_case_directory": safe_id,
        "environment_contract": {
            "MEEP_GPU_BACKEND": backend,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_CACHE_DISABLE": "1",
            "JAX_PLATFORMS": "cpu",
            "HOME": str(runtime_home),
            "MPLCONFIGDIR": str(runtime_matplotlib),
            "XDG_CACHE_HOME": str(runtime_cache),
            "XDG_CONFIG_HOME": str(runtime_config),
            "FONTCONFIG_FILE": str(expected_runtime["fontconfig_file"]["path"]),
            "strict_cuda": backend == "cuda",
            "keys": sorted(env),
            "sha256": environment_sha256(env),
            "environment": dict(sorted(env.items())),
        },
        "started_at_utc": started_at_utc,
        "run_nonce": run_nonce,
        "duration_seconds": duration,
        "exit_code": exit_code,
        "timeout": timeout,
        "output_limit": output_limit,
        "output_sizes": {
            "stdout": capture["stdout_size_bytes"],
            "stderr": capture["stderr_size_bytes"],
            "combined": (
                capture["stdout_size_bytes"]
                + capture["stderr_size_bytes"]
            ),
        },
        "output_limits": {
            "stdout": MAX_BACKEND_STDOUT_BYTES,
            "stderr": MAX_BACKEND_STDERR_BYTES,
            "combined": MAX_BACKEND_OUTPUT_BYTES,
        },
        "unittest_skips": skips,
        "unittest_skip_details": skip_details,
        "allowed_unittest_skips": allowed_skips,
        "unittest_skip_policy_problems": skip_policy_problems,
        "unittest_reported_test_counts": unittest_evidence["reported_counts"],
        "unittest_test_count": unittest_evidence["reported_count"],
        "unittest_test_identities": unittest_evidence["identities"],
        "expected_unittest_test_count": unittest_evidence["expected_count"],
        "expected_unittest_test_identities": unittest_evidence[
            "expected_identities"
        ],
        "unittest_test_contract_ok": unittest_contract_ok,
        "unittest_test_contract_problems": unittest_contract_problems,
        "unittest_terminal_summaries": unittest_evidence[
            "terminal_summaries"
        ],
        "stdout_log": evidence_member_path(stdout_file, result_dir),
        "stderr_log": evidence_member_path(stderr_file, result_dir),
        "statistics_file": (
            evidence_member_path(stats_file, result_dir)
            if stats_file.is_file()
            else stats_file.resolve().relative_to(result_dir.resolve()).as_posix()
        ),
        "evidence_files": evidence_files,
        "statistics": stats,
        "backend_contract_ok": contract_ok,
        "backend_contract_problems": contract_problems,
        "outcome": outcome,
    }


def compare_runs(
    case: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    evidence_root: pathlib.Path,
) -> dict[str, Any]:
    failed = [backend for backend, run in runs.items() if not run["outcome"].startswith("PASS")]
    if failed:
        return {
            "mode": case["comparison"]["mode"],
            "outcome": "NOT_COMPARABLE",
            "reason": f"backend runs did not pass: {', '.join(failed)}",
        }
    if set(runs) != {"cpu", "cuda"}:
        return {
            "mode": case["comparison"]["mode"],
            "outcome": "SINGLE_BACKEND_ONLY",
            "reason": "both CPU and CUDA are required for a cross-backend comparison",
        }
    mode = case["comparison"]["mode"]
    cpu_duration = runs["cpu"]["duration_seconds"]
    cuda_duration = runs["cuda"]["duration_seconds"]
    result: dict[str, Any] = {
        "mode": mode,
        "cpu_seconds": cpu_duration,
        "cuda_seconds": cuda_duration,
        "speedup_cpu_over_cuda": (
            cpu_duration / cuda_duration if cuda_duration > 0 else None
        ),
    }
    if mode == "embedded_oracle":
        result.update(
            {
                "outcome": "PASS",
                "reason": "both backends passed the test's embedded numerical oracle",
            }
        )
    elif mode == "normalized_stdout":
        cpu_stdout = resolve_evidence_member(
            evidence_root, runs["cpu"]["stdout_log"]
        ).read_text(encoding="utf-8")
        cuda_stdout = resolve_evidence_member(
            evidence_root, runs["cuda"]["stdout_log"]
        ).read_text(encoding="utf-8")
        cpu_value = normalize_stdout(cpu_stdout, case["comparison"])
        cuda_value = normalize_stdout(cuda_stdout, case["comparison"])
        equal = cpu_value == cuda_value
        result.update(
            {
                "outcome": "PASS" if equal else "MISMATCH",
                "reason": "normalized stdout is equal" if equal else "normalized stdout differs",
                "normalized_stdout_evidence": {
                    "preview_byte_limit": MAX_NORMALIZED_STDOUT_PREVIEW_BYTES,
                    "cpu": bounded_text_record(
                        cpu_value, MAX_NORMALIZED_STDOUT_PREVIEW_BYTES
                    ),
                    "cuda": bounded_text_record(
                        cuda_value, MAX_NORMALIZED_STDOUT_PREVIEW_BYTES
                    ),
                },
            }
        )
    elif mode == "json_metrics":
        cpu_stdout = resolve_evidence_member(
            evidence_root, runs["cpu"]["stdout_log"]
        )
        cuda_stdout = resolve_evidence_member(
            evidence_root, runs["cuda"]["stdout_log"]
        )
        comparison_directory = (
            evidence_root
            / "runs"
            / runs["cpu"]["evidence_case_directory"]
            / "comparison"
        )
        failure_sidecar = comparison_directory / "json-metric-failures.ndjson"
        raw_metric_evidence: dict[str, Any] | None = None
        try:
            cpu_stdout_record = runs["cpu"].get("evidence_files", {}).get(
                "stdout"
            )
            cuda_stdout_record = runs["cuda"].get("evidence_files", {}).get(
                "stdout"
            )
            if not isinstance(cpu_stdout_record, dict) or not isinstance(
                cuda_stdout_record, dict
            ):
                raise ManifestError(
                    "json_metrics stdout evidence records are absent"
                )
            if cpu_stdout_record.get("path") != runs["cpu"]["stdout_log"]:
                raise ManifestError(
                    "CPU json_metrics stdout path differs from its evidence record"
                )
            if cuda_stdout_record.get("path") != runs["cuda"]["stdout_log"]:
                raise ManifestError(
                    "CUDA json_metrics stdout path differs from its evidence record"
                )
            raw_metric_evidence = {
                "cpu_stdout": cpu_stdout_record,
                "cuda_stdout": cuda_stdout_record,
            }
            result.update(
                compare_json_metric_files(
                    cpu_stdout,
                    cuda_stdout,
                    case["comparison"],
                    failure_sidecar=failure_sidecar,
                    temporary_parent=comparison_directory,
                    expected_cpu_stdout=cpu_stdout_record,
                    expected_cuda_stdout=cuda_stdout_record,
                )
            )
            if failure_sidecar.is_file():
                result["failure_evidence"] = {
                    "available": True,
                    **evidence_file_record(failure_sidecar, evidence_root),
                }
            elif comparison_directory.is_dir():
                comparison_directory.rmdir()
        except ManifestError as exc:
            failure_sidecar.unlink(missing_ok=True)
            if comparison_directory.is_dir() and not any(
                comparison_directory.iterdir()
            ):
                comparison_directory.rmdir()
            result.update(
                {
                    "outcome": "MISMATCH",
                    "reason": str(exc),
                }
            )
        if raw_metric_evidence is not None:
            result["raw_metric_evidence"] = raw_metric_evidence
    else:
        raise ManifestError(f"unknown comparison mode: {mode}")
    return result


def verify_failure_sidecar(
    path: pathlib.Path, comparison: dict[str, Any]
) -> list[str]:
    problems: list[str] = []
    digest = hashlib.sha256()
    digest.update(b"gpmeep-json-failure-records-v1\0")
    count = 0
    with path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            payload = line[:-1] if line.endswith(b"\n") else line
            try:
                value = json.loads(
                    payload.decode("utf-8"),
                    object_pairs_hook=unique_json_object,
                )
                canonical = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            except (UnicodeDecodeError, json.JSONDecodeError, ManifestError, ValueError) as exc:
                problems.append(
                    f"failure sidecar line {line_number} is invalid: {exc}"
                )
                continue
            if canonical != payload:
                problems.append(
                    f"failure sidecar line {line_number} is not canonical JSON"
                )
            digest.update(len(canonical).to_bytes(8, "big"))
            digest.update(canonical)
            count += 1
    if count != comparison.get("failure_count"):
        problems.append(
            f"failure sidecar has {count} records, expected "
            f"{comparison.get('failure_count')!r}"
        )
    if digest.hexdigest() != comparison.get("failure_records_sha256"):
        problems.append("failure sidecar digest disagrees with comparison")
    return problems


def verify_run_evidence_closure(
    results: list[dict[str, Any]], evidence_root: pathlib.Path
) -> list[str]:
    problems: list[str] = []
    bound_members: dict[str, str] = {}
    runtime_prefixes: set[str] = set()
    root = evidence_root.resolve()
    for result in results:
        expected_directory = case_directory_id(result["id"])
        for backend, run in result.get("runs", {}).items():
            owner = f"{result['id']}:{backend}"
            if run.get("evidence_case_directory") != expected_directory:
                problems.append(f"{owner}: case-directory identity mismatch")
            expected_prefix = f"runs/{expected_directory}/{backend}/"
            nonce = run.get("run_nonce")
            if isinstance(nonce, str) and nonce:
                runtime_prefixes.add(expected_prefix + f"runtime/{nonce}/")
            evidence_files = run.get("evidence_files")
            if not isinstance(evidence_files, dict):
                problems.append(f"{owner}: evidence_files is absent")
                continue
            for label, record in evidence_files.items():
                if not isinstance(record, dict):
                    problems.append(f"{owner}:{label}: invalid evidence record")
                    continue
                member = record.get("path")
                if not isinstance(member, str) or not member.startswith(
                    expected_prefix
                ):
                    problems.append(
                        f"{owner}:{label}: evidence path is outside its case directory"
                    )
                    continue
                previous = bound_members.get(member)
                if previous is not None and previous != f"{owner}:{label}":
                    problems.append(
                        f"evidence member {member!r} is claimed by both "
                        f"{previous} and {owner}:{label}"
                    )
                bound_members[member] = f"{owner}:{label}"
                if not record.get("available"):
                    continue
                try:
                    path = resolve_evidence_member(root, member)
                except ProvenanceError as exc:
                    problems.append(f"{owner}:{label}: {exc}")
                    continue
                if path.is_symlink() or not path.is_file():
                    problems.append(
                        f"{owner}:{label}: evidence is not a regular non-symlink file"
                    )
                    continue
                size = path.stat().st_size
                if size != record.get("size_bytes"):
                    problems.append(f"{owner}:{label}: evidence size changed")
                if file_sha256(path) != record.get("sha256"):
                    problems.append(f"{owner}:{label}: evidence SHA-256 changed")
            for label, record_name in (
                ("stdout_log", "stdout"),
                ("stderr_log", "stderr"),
                ("statistics_file", "statistics"),
            ):
                record = evidence_files.get(record_name, {})
                if run.get(label) != record.get("path"):
                    problems.append(
                        f"{owner}: {label} does not match its evidence record"
                    )
            raw = (result.get("comparison") or {}).get("raw_metric_evidence")
            if isinstance(raw, dict):
                expected_raw = raw.get(f"{backend}_stdout")
                if expected_raw != evidence_files.get("stdout"):
                    problems.append(
                        f"{owner}: raw metric evidence is not bound to stdout"
                    )
        failure_record = (result.get("comparison") or {}).get(
            "failure_evidence"
        )
        if isinstance(failure_record, dict):
            owner = f"{result['id']}:comparison-failures"
            member = failure_record.get("path")
            expected_prefix = f"runs/{expected_directory}/comparison/"
            if not isinstance(member, str) or not member.startswith(
                expected_prefix
            ):
                problems.append(f"{owner}: unsafe failure sidecar path")
            elif member in bound_members:
                problems.append(f"{owner}: duplicate evidence member {member!r}")
            else:
                bound_members[member] = owner
                try:
                    path = resolve_evidence_member(root, member)
                except ProvenanceError as exc:
                    problems.append(f"{owner}: {exc}")
                else:
                    if path.is_symlink() or not path.is_file():
                        problems.append(f"{owner}: sidecar is not a regular file")
                    elif path.stat().st_size != failure_record.get("size_bytes"):
                        problems.append(f"{owner}: sidecar size changed")
                    elif file_sha256(path) != failure_record.get("sha256"):
                        problems.append(f"{owner}: sidecar SHA-256 changed")
                    else:
                        problems.extend(
                            f"{owner}: {problem}"
                            for problem in verify_failure_sidecar(
                                path, result["comparison"]
                            )
                        )

    runs_root = root / "runs"
    if runs_root.exists():
        for path in sorted(runs_root.rglob("*")):
            if path.is_symlink():
                problems.append(f"run evidence contains symlink: {path}")
                continue
            if not path.is_file():
                continue
            member = path.resolve().relative_to(root).as_posix()
            if member in bound_members:
                continue
            if any(member.startswith(prefix) for prefix in runtime_prefixes):
                continue
            problems.append(f"unbound run evidence file: {member}")
    return problems


def outcome_exit_code(results: Iterable[dict[str, Any]]) -> int:
    exit_code = EXIT_OK
    for result in results:
        for run in result.get("runs", {}).values():
            if run["outcome"] == "BACKEND_CONTRACT_FAILED":
                exit_code = max(exit_code, EXIT_BACKEND_CONTRACT)
            elif run["outcome"] in {
                "PROCESS_FAILED",
                "TIMEOUT",
                "OUTPUT_LIMIT",
                "OUTPUT_INVALID_UTF8",
                "UNDECLARED_SKIP",
                "UNITTEST_CONTRACT_FAILED",
            }:
                exit_code = max(exit_code, EXIT_EXECUTION_FAILED)
        comparison = result.get("comparison")
        if comparison and comparison.get("outcome") == "MISMATCH":
            exit_code = max(exit_code, EXIT_COMPARISON_FAILED)
        if (
            result.get("outcome") in {
                "BLOCKED_DEPENDENCY",
                COVERAGE_FAILED_OUTCOME,
                COVERAGE_UNRESOLVED_OUTCOME,
                "UNSUPPORTED_SELECTION",
            }
            or (
                result.get("selected")
                and result.get("disposition") in BLOCKING_SELECTED_DISPOSITIONS
            )
        ):
            exit_code = max(exit_code, EXIT_BLOCKED)
    return exit_code


def markdown_report(report: dict[str, Any]) -> str:
    performance = report["configuration"]["performance_evidence"]
    lines = [
        "# gpmeep Python validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Git HEAD: `{report['provenance']['git_head'].get('stdout', '<unavailable>')}`",
        f"- Manifest: `{report['provenance']['manifest']}`",
        f"- Python: `{report['provenance']['python_executable']}`",
        f"- Build package: `{report['provenance']['build_python']}`",
        f"- Requested backends: `{', '.join(report['configuration']['backends'])}`",
        f"- Timing valid for speed gate: `{performance['valid_for_speed_gate']}`",
        f"- Concurrent GPU work detected: `{performance['concurrency_detected']}`",
        "- Source/build unchanged during validation: "
        f"`{report['provenance']['validation_window']['unchanged']}`",
        f"- Overall exit code: `{report['exit_code']}`",
        "",
        "## Summary",
        "",
        "| Outcome | Count |",
        "| --- | ---: |",
    ]
    for outcome, count in sorted(report["summary"].items()):
        lines.append(f"| {outcome} | {count} |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Scope | Outcome | CPU | CUDA | Comparison | Observed CPU/CUDA ratio (non-gating) |",
            "| --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    follow_up_bottlenecks: list[tuple[str, float]] = []
    for result in report["results"]:
        cpu = result["runs"].get("cpu", {}).get("outcome", "—")
        cuda = result["runs"].get("cuda", {}).get("outcome", "—")
        comparison = (result.get("comparison") or {}).get("outcome", "—")
        speedup = (result.get("comparison") or {}).get("speedup_cpu_over_cuda")
        speedup_text = f"{speedup:.3f}×" if isinstance(speedup, (int, float)) else "—"
        if (
            result.get("selected") is True
            and result.get("compute_scope") == "fdtd_cuda"
            and comparison == "PASS"
            and isinstance(speedup, (int, float))
            and speedup < 1.0
        ):
            follow_up_bottlenecks.append((result["path"], speedup))
        lines.append(
            f"| `{result['path']}` | {result['compute_scope']} | "
            f"{result['outcome']} | {cpu} | "
            f"{cuda} | {comparison} | {speedup_text} |"
        )
    lines.extend(
        [
            "",
            "## Follow-up performance candidates",
            "",
            "These are selected correctness runs whose observed single-run CPU/CUDA ratio was below 1×. "
            "They are explicit optimization candidates, not speed-gate conclusions.",
            "",
        ]
    )
    if follow_up_bottlenecks:
        for path, ratio in follow_up_bottlenecks:
            lines.append(f"- `{path}`: observed non-gating ratio `{ratio:.3f}×`")
    else:
        lines.append("- None observed in the selected correctness runs.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `cuda` means gpmeep strict CUDA selection; a CPU phase counter makes the backend contract fail.",
            "- `host_only` cases are compatibility/oracle checks and are excluded from GPU coverage and performance claims.",
            "- Non-run cases remain visible with a reason and milestone; they are never silently skipped.",
            "- The observed CPU/CUDA ratios include Python startup, have no warm-up or repeated samples, and are not a release performance gate.",
            "",
        ]
    )
    return "\n".join(lines)


def select_case(case: dict[str, Any], tiers: set[str], ids: set[str]) -> bool:
    if ids:
        return case["id"] in ids or case["path"] in ids
    return bool(tiers.intersection(case["tier"]))


def parse_args(argv: list[str]) -> argparse.Namespace:
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_default = script_dir.parent.parent
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=pathlib.Path, default=repo_default)
    parser.add_argument("--manifest", type=pathlib.Path, default=script_dir / "manifest.json")
    parser.add_argument(
        "--build-python",
        type=pathlib.Path,
        default=(
            repo_default / "build" / "meep-cuda-mpi-python-fp32" / "python"
        ),
    )
    parser.add_argument(
        "--install-prefix",
        type=pathlib.Path,
        default=repo_default / "install" / "meep-cuda-mpi-python-fp32",
    )
    parser.add_argument("--python", type=pathlib.Path, default=pathlib.Path(sys.executable))
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="evidence directory (default: artifacts/python-validation/<UTC timestamp>)",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda", "both"),
        default="both",
        help="backend runs to execute",
    )
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        help="tier to select; repeatable (default: smoke)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="exact case id or repository-relative path; repeatable",
    )
    parser.add_argument(
        "--concurrent-gpu-work",
        action="store_true",
        help=(
            "record that another GPU workload overlapped this run; timings are "
            "explicitly invalid for a speed gate"
        ),
    )
    parser.add_argument("--list", action="store_true", help="list materialized inventory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    repo = args.repo.resolve()
    manifest_path = args.manifest.resolve()
    build_python = args.build_python.resolve()
    install_prefix = args.install_prefix.resolve() if args.install_prefix else None
    python = args.python.resolve()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.resolve()
        if args.output
        else repo / "artifacts" / "python-validation" / timestamp
    )
    try:
        manifest = load_json(manifest_path)
        cases = materialize_cases(manifest, repo)
        if not args.list:
            prepare_output(output)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return EXIT_MANIFEST_INVALID

    if args.list:
        for case in cases:
            print(
                f"{case['path']}\t{case['disposition']}\t"
                f"{','.join(case['tier'])}\t{case['reason']}"
            )
        return EXIT_OK

    tiers = set(args.tier or ["smoke"])
    ids = set(args.case)
    unknown_ids = ids - {case["id"] for case in cases} - {case["path"] for case in cases}
    if unknown_ids:
        print(f"unknown case selection: {', '.join(sorted(unknown_ids))}", file=sys.stderr)
        return EXIT_MANIFEST_INVALID
    known_tiers = {
        tier for case in cases for tier in case.get("tier", [])
    }
    unknown_tiers = tiers - known_tiers
    if unknown_tiers:
        print(
            f"unknown tier selection: {', '.join(sorted(unknown_tiers))}",
            file=sys.stderr,
        )
        return EXIT_MANIFEST_INVALID
    if not any(select_case(case, tiers, ids) for case in cases):
        print("case selection matched no inventory cases", file=sys.stderr)
        return EXIT_MANIFEST_INVALID
    hook_dir = pathlib.Path(__file__).resolve().parent
    backends = ["cpu", "cuda"] if args.backend == "both" else [args.backend]

    start_snapshot = validation_snapshot(repo, build_python, python)
    provenance = collect_provenance(
        repo,
        python,
        build_python,
        install_prefix,
        manifest_path,
        validation_start=start_snapshot,
        probe_runtime_parent=output / "provenance-runtime",
        probe_fontconfig_file=(
            pathlib.Path(
                start_snapshot["runtime_contract"]["fontconfig_file"]["path"]
            )
            if start_snapshot.get("runtime_contract", {})
            .get("fontconfig_file", {})
            .get("path")
            else None
        ),
    )
    invalid_timing_reasons = [
        "whole-process single-sample timing includes Python startup",
        "no benchmark warm-up or repeated-sample statistics",
    ]
    if args.concurrent_gpu_work:
        invalid_timing_reasons.append("concurrency_detected")
    performance_evidence = {
        "valid_for_speed_gate": False,
        "concurrency_detected": bool(args.concurrent_gpu_work),
        "invalid_reasons": invalid_timing_reasons,
    }
    results: list[dict[str, Any]] = []
    for case in cases:
        selected = select_case(case, tiers, ids)
        result = classify_inventory_case(case, selected)
        if not selected or case["disposition"] != RUN_DISPOSITION:
            results.append(result)
            continue
        if not start_snapshot["available"]:
            result["outcome"] = "EVIDENCE_INTEGRITY_FAILED"
            result["reason"] = (
                "validation did not start because its source/build snapshot "
                "was unavailable: " + "; ".join(start_snapshot["problems"])
            )
            results.append(result)
            continue
        missing = [
            dependency
            for dependency in case.get("dependencies", [])
            if not dependency_available(dependency, python, repo)
        ]
        if missing:
            result["outcome"] = "BLOCKED_DEPENDENCY"
            result["reason"] = (
                f"required Python modules are missing: {', '.join(missing)}; "
                f"declared reason: {case['reason']}"
            )
            result["missing_dependencies"] = missing
            results.append(result)
            continue
        for backend in backends:
            result["runs"][backend] = run_backend(
                case, backend, repo, python, build_python, hook_dir, output,
                start_snapshot["runtime_contract"],
            )
        result["comparison"] = compare_runs(case, result["runs"], output)
        failed_runs = [
            run["outcome"]
            for run in result["runs"].values()
            if not run["outcome"].startswith("PASS")
        ]
        if failed_runs:
            result["outcome"] = failed_runs[0]
        else:
            result["outcome"] = result["comparison"]["outcome"]
        results.append(result)

    resolve_coverage_outcomes(cases, results)

    # A COMPLETE marker must never get ahead of the raw process evidence on
    # durable storage.  Flush the evidence directory topology bottom-up before
    # hashing it into the report.
    fsync_evidence_directories(output, results)
    end_snapshot = validation_snapshot(repo, build_python, python)
    integrity_ok, integrity_problems = compare_validation_snapshots(
        start_snapshot, end_snapshot
    )
    evidence_closure_problems = verify_run_evidence_closure(results, output)
    if evidence_closure_problems:
        integrity_ok = False
        integrity_problems.extend(evidence_closure_problems)
    provenance["validation_window"] = {
        "start": start_snapshot,
        "end": end_snapshot,
        "unchanged": integrity_ok,
        "problems": integrity_problems,
    }
    selected_results = [result for result in results if result["selected"]]
    exit_code = outcome_exit_code(selected_results)
    if not integrity_ok:
        exit_code = EXIT_EVIDENCE_INTEGRITY
    summary = Counter(result["outcome"] for result in results)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "configuration": {
            "tiers": sorted(tiers),
            "case_ids": sorted(ids),
            "backends": backends,
            "performance_evidence": performance_evidence,
        },
        "provenance": provenance,
        "inventory": {
            "total": len(cases),
            "selected": len(selected_results),
            "scheme_scope": "excluded by project requirement",
        },
        "summary": dict(sorted(summary.items())),
        "results": results,
        "exit_code": exit_code,
        "exit_code_meaning": {
            str(EXIT_OK): "all selected runnable cases passed",
            str(EXIT_EXECUTION_FAILED): (
                "a process, timeout, skip policy, or unittest contract failed"
            ),
            str(EXIT_MANIFEST_INVALID): "manifest or invocation is invalid",
            str(EXIT_COMPARISON_FAILED): "CPU/CUDA observable comparison failed",
            str(EXIT_BLOCKED): "a selected runnable case has an undeclared blocker",
            str(EXIT_BACKEND_CONTRACT): "backend provenance or strict CUDA dispatch failed",
            str(EXIT_EVIDENCE_INTEGRITY): (
                "source-tree or built-extension evidence was unavailable or "
                "changed during validation"
            ),
        },
    }
    json_path = output / "report.json"
    markdown_path = output / "report.md"
    complete_path = output / "COMPLETE"
    try:
        report_json = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
    except (TypeError, ValueError) as exc:
        print(f"evidence serialization error: {exc}", file=sys.stderr)
        return EXIT_EVIDENCE_INTEGRITY
    report_size = len(report_json.encode("utf-8"))
    if report_size > MAX_REPORT_JSON_BYTES:
        print(
            f"evidence report is {report_size} bytes; limit is "
            f"{MAX_REPORT_JSON_BYTES}",
            file=sys.stderr,
        )
        return EXIT_EVIDENCE_INTEGRITY
    atomic_write_text(json_path, report_json)
    markdown_text = markdown_report(report)
    atomic_write_text(markdown_path, markdown_text)
    final_closure_problems = verify_run_evidence_closure(results, output)
    if final_closure_problems:
        print(
            "evidence changed before completion: "
            + "; ".join(final_closure_problems),
            file=sys.stderr,
        )
        return EXIT_EVIDENCE_INTEGRITY
    complete = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "completed_at_utc": utc_now(),
        "exit_code": exit_code,
        "report_sha256": file_sha256(json_path),
        "report_size_bytes": json_path.stat().st_size,
        "markdown_sha256": file_sha256(markdown_path),
        "markdown_size_bytes": markdown_path.stat().st_size,
    }
    atomic_write_text(
        complete_path,
        json.dumps(
            complete, indent=2, sort_keys=True, allow_nan=False
        ) + "\n",
    )
    print(f"JSON evidence: {json_path}")
    print(f"Markdown evidence: {markdown_path}")
    print(f"selected={len(selected_results)} exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
