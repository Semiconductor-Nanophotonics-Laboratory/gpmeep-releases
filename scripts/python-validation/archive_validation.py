#!/usr/bin/env python3
"""Seal and verify a relocatable gpmeep Python-validation evidence archive.

The runner output is treated as immutable input.  This tool verifies its
COMPLETE/report chain and every referenced stdout, stderr, and statistics
record, then copies only the audited closure into a new archive directory.
Derived per-run HOME/XDG/Matplotlib cache files are enumerated in the archive
manifest but intentionally not copied.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


# The verifier is copied inside each sealed archive and must be observational:
# importing its sibling contract modules must never create ``__pycache__`` in
# the archive being verified.  Set this before the first local import instead
# of relying on callers to provide PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True


SCRIPT_SELF_DIR = pathlib.Path(__file__).resolve().parent
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
for import_root in (SCRIPTS_DIR, SCRIPT_SELF_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpmeep_provenance import (  # noqa: E402
    ProvenanceError,
    StatHashCache,
    atomic_write_json,
    bounded_command,
    canonical_sha256,
    verify_build_receipt,
)
import run_validation as runner_contract  # noqa: E402


SCHEMA_VERSION = 1
RUNNER_REPORT_SCHEMA_VERSION = 4
ARCHIVE_KIND = "gpmeep-python-validation-archive-v1"
MANIFEST_NAME = "ARCHIVE_MANIFEST.json"
COMPLETE_NAME = "ARCHIVE_COMPLETE"
RUNNER_ROOT = pathlib.PurePosixPath("runner-output")
INPUT_ROOT = pathlib.PurePosixPath("inputs")
SOURCE_ROOT = pathlib.PurePosixPath("sources")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RUN_FILES = {
    "stdout": ("stdout_log", "stdout.log"),
    "stderr": ("stderr_log", "stderr.log"),
    "statistics": ("statistics_file", "gpu-statistics.json"),
}
ARCHIVE_FILE_MODE = 0o644
ARCHIVE_DIRECTORY_MODE = 0o755
MINIMUM_TAR_OUTPUT_LIMIT_BYTES = 64 * 1024 * 1024
TAR_MEMBER_OVERHEAD_LIMIT_BYTES = 64 * 1024
TAR_TRAILER_OVERHEAD_LIMIT_BYTES = 10 * 1024 * 1024
TRUSTED_TAR_CANDIDATES = (
    pathlib.Path("/usr/bin/tar"),
    pathlib.Path("/bin/tar"),
    pathlib.Path(sys.executable).resolve().parent / "tar",
)


def trusted_tar() -> pathlib.Path:
    for candidate in dict.fromkeys(TRUSTED_TAR_CANDIDATES):
        try:
            info = os.stat(candidate, follow_symlinks=True)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode) and os.access(candidate, os.X_OK):
            return candidate
    raise ArchiveError(
        "trusted tar is unavailable at fixed system/Python-prefix paths"
    )
RUN_RECORD_KEYS = {
    "backend",
    "command",
    "cwd",
    "evidence_case_directory",
    "environment_contract",
    "started_at_utc",
    "run_nonce",
    "duration_seconds",
    "exit_code",
    "timeout",
    "output_limit",
    "output_sizes",
    "output_limits",
    "unittest_skips",
    "unittest_skip_details",
    "allowed_unittest_skips",
    "unittest_skip_policy_problems",
    "unittest_reported_test_counts",
    "unittest_test_count",
    "unittest_test_identities",
    "expected_unittest_test_count",
    "expected_unittest_test_identities",
    "unittest_test_contract_ok",
    "unittest_test_contract_problems",
    "unittest_terminal_summaries",
    "stdout_log",
    "stderr_log",
    "statistics_file",
    "evidence_files",
    "statistics",
    "backend_contract_ok",
    "backend_contract_problems",
    "outcome",
}
RUN_ENVIRONMENT_CONTRACT_KEYS = {
    "MEEP_GPU_BACKEND",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "CUDA_CACHE_DISABLE",
    "JAX_PLATFORMS",
    "HOME",
    "MPLCONFIGDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "FONTCONFIG_FILE",
    "strict_cuda",
    "keys",
    "sha256",
    "environment",
}
RUNNER_CONFIGURATION_KEYS = {
    "tiers",
    "case_ids",
    "backends",
    "performance_evidence",
}
PERFORMANCE_EVIDENCE_KEYS = {
    "valid_for_speed_gate",
    "concurrency_detected",
    "invalid_reasons",
}
PROVENANCE_KEYS = {
    "captured_at_utc",
    "hostname",
    "platform",
    "machine",
    "python_executable",
    "python_version",
    "repository",
    "git_head",
    "git_status",
    "source_snapshot",
    "manifest",
    "manifest_sha256",
    "build_python",
    "build_python_exists",
    "build_extensions",
    "build_package_probe",
    "install_prefix",
    "install_extensions",
    "install_package_probe",
    "source_files",
    "environment_selection",
    "validation_window",
}
VALIDATION_WINDOW_KEYS = {"start", "end", "unchanged", "problems"}
VALIDATION_ENDPOINT_KEYS = {
    "captured_at_utc",
    "available",
    "source_snapshot",
    "build_extension",
    "build_receipt",
    "runtime_contract",
    "problems",
}
RUNTIME_CONTRACT_KEYS = {
    "python_executable",
    "meep_module",
    "extension",
    "libmeep",
    "fontconfig_file",
    "build_home",
    "qualification_home",
    "installed_environment",
    "receipt_id",
}
RESULT_RECORD_KEYS = {
    "id",
    "path",
    "kind",
    "tier",
    "disposition",
    "compute_scope",
    "reason",
    "milestone",
    "selected",
    "outcome",
    "runs",
    "comparison",
}
RUN_CONTROLLED_ENVIRONMENT = {
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "HOME",
    "MPLBACKEND",
    "MPLCONFIGDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "FONTCONFIG_FILE",
    "CUDA_CACHE_DISABLE",
    "JAX_PLATFORMS",
    "MEEP_GPU_BACKEND",
    "GPMEEP_VALIDATION_STATS_FILE",
    "GPMEEP_VALIDATION_EXPECTED_BACKEND",
    "GPMEEP_VALIDATION_RUN_NONCE",
    "GPMEEP_VALIDATION_BUILD_RECEIPT_ID",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "LD_LIBRARY_PATH",
}
FIXED_REPO_INPUTS = {
    "manifest.json": pathlib.PurePosixPath(
        "scripts/python-validation/manifest.json"
    ),
    "run_validation.py": pathlib.PurePosixPath(
        "scripts/python-validation/run_validation.py"
    ),
    "run_example_oracle.py": pathlib.PurePosixPath(
        "scripts/python-validation/run_example_oracle.py"
    ),
    "archive_validation.py": pathlib.PurePosixPath(
        "scripts/python-validation/archive_validation.py"
    ),
    "gpmeep_provenance.py": pathlib.PurePosixPath(
        "scripts/gpmeep_provenance.py"
    ),
    "gpmeep_qualification_contract.py": pathlib.PurePosixPath(
        "scripts/gpmeep_qualification_contract.py"
    ),
    "gpmeep-control-python.py": pathlib.PurePosixPath(
        "scripts/gpmeep-control-python.py"
    ),
    "sitecustomize.py": pathlib.PurePosixPath(
        "scripts/python-validation/sitecustomize.py"
    ),
    "absorber_branch_matrix.py": pathlib.PurePosixPath(
        "scripts/python-validation/absorber_branch_matrix.py"
    ),
    "perturbation_branch_matrix.py": pathlib.PurePosixPath(
        "scripts/python-validation/perturbation_branch_matrix.py"
    ),
    "run_point_dipole_cyl_validation.py": pathlib.PurePosixPath(
        "scripts/python-validation/run_point_dipole_cyl_validation.py"
    ),
    "stochastic_branch_matrix.py": pathlib.PurePosixPath(
        "scripts/python-validation/stochastic_branch_matrix.py"
    ),
    "stochastic_line_basis_matrix.py": pathlib.PurePosixPath(
        "scripts/python-validation/stochastic_line_basis_matrix.py"
    ),
    "stochastic_reciprocity_matrix.py": pathlib.PurePosixPath(
        "scripts/python-validation/stochastic_reciprocity_matrix.py"
    ),
    "build-meep-cuda-mpi-python.sh": pathlib.PurePosixPath(
        "scripts/build-meep-cuda-mpi-python.sh"
    ),
}


class ArchiveError(RuntimeError):
    """Raised when an archive cannot be verified or safely published."""


@dataclass(frozen=True)
class FileSnapshot:
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CopySpec:
    member: str
    source: pathlib.Path
    expected: FileSnapshot
    category: str


@dataclass(frozen=True)
class EvidenceReference:
    source_relative_path: str
    archive_path: str
    recorded_path_sha256: str
    size_bytes: int
    sha256: str


def _reject_json_constant(value: str) -> None:
    raise ArchiveError(f"JSON contains non-finite constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveError(f"{label} must contain a JSON object")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _absolute(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(os.fspath(path)))


def _assert_path_chain_without_symlinks(
    path: pathlib.Path, *, final_kind: str | None
) -> pathlib.Path:
    absolute = _absolute(path)
    current = pathlib.Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if index == len(parts) - 1 and final_kind == "missing":
                return absolute
            raise ArchiveError(f"required path is absent: {current}")
        if stat.S_ISLNK(info.st_mode):
            raise ArchiveError(f"symbolic links are forbidden: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ArchiveError(f"path ancestor is not a directory: {current}")
    if final_kind is None:
        return absolute
    info = os.lstat(absolute)
    if final_kind == "file" and not stat.S_ISREG(info.st_mode):
        raise ArchiveError(f"required path is not a regular file: {absolute}")
    if final_kind == "directory" and not stat.S_ISDIR(info.st_mode):
        raise ArchiveError(f"required path is not a directory: {absolute}")
    if final_kind == "missing":
        raise ArchiveError(f"destination already exists: {absolute}")
    return absolute


def _file_fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_stable_file(path: pathlib.Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArchiveError(f"source is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            _file_fingerprint(before) != _file_fingerprint(after)
            or _file_fingerprint(before) != _file_fingerprint(current)
        ):
            raise ArchiveError(f"file changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _snapshot(path: pathlib.Path) -> FileSnapshot:
    data = _read_stable_file(path)
    return FileSnapshot(len(data), _sha256_bytes(data))


def _load_json_file(path: pathlib.Path, label: str) -> tuple[dict[str, Any], bytes]:
    data = _read_stable_file(path)
    return _load_json_bytes(data, label), data


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ArchiveError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArchiveError(f"{label} must be an object")
    return value


RUNNER_REPORT_KEYS = {
    "schema_version",
    "generated_at_utc",
    "configuration",
    "provenance",
    "inventory",
    "summary",
    "results",
    "exit_code",
    "exit_code_meaning",
}
RUNNER_COMPLETE_KEYS = {
    "schema_version",
    "completed_at_utc",
    "exit_code",
    "report_sha256",
    "report_size_bytes",
    "markdown_sha256",
    "markdown_size_bytes",
}
RUNNER_EXIT_CODES = {
    runner_contract.EXIT_OK,
    runner_contract.EXIT_EXECUTION_FAILED,
    runner_contract.EXIT_MANIFEST_INVALID,
    runner_contract.EXIT_COMPARISON_FAILED,
    runner_contract.EXIT_BLOCKED,
    runner_contract.EXIT_BACKEND_CONTRACT,
    runner_contract.EXIT_EVIDENCE_INTEGRITY,
}


def _validate_runner_control(
    report: dict[str, Any], runner_complete: dict[str, Any]
) -> None:
    if report.get("schema_version") != RUNNER_REPORT_SCHEMA_VERSION:
        raise ArchiveError(
            "runner report schema must be exactly "
            f"{RUNNER_REPORT_SCHEMA_VERSION}"
        )
    if runner_complete.get("schema_version") != RUNNER_REPORT_SCHEMA_VERSION:
        raise ArchiveError(
            "runner COMPLETE schema must be exactly "
            f"{RUNNER_REPORT_SCHEMA_VERSION}"
        )
    if set(report) != RUNNER_REPORT_KEYS:
        raise ArchiveError("runner report has unexpected or missing fields")
    if set(runner_complete) != RUNNER_COMPLETE_KEYS:
        raise ArchiveError("runner COMPLETE has unexpected or missing fields")
    exit_code = report.get("exit_code")
    if (
        not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or exit_code not in RUNNER_EXIT_CODES
        or runner_complete.get("exit_code") != exit_code
    ):
        raise ArchiveError("runner exit-code contract is invalid")
    for key in ("generated_at_utc",):
        if not isinstance(report.get(key), str) or not report[key]:
            raise ArchiveError(f"runner report {key} is invalid")
    if (
        not isinstance(runner_complete.get("completed_at_utc"), str)
        or not runner_complete["completed_at_utc"]
    ):
        raise ArchiveError("runner COMPLETE timestamp is invalid")
    for key in ("report_sha256", "markdown_sha256"):
        _require_sha256(runner_complete.get(key), f"runner COMPLETE {key}")
    for key in ("report_size_bytes", "markdown_size_bytes"):
        value = runner_complete.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ArchiveError(f"runner COMPLETE {key} is invalid")
    for key in ("configuration", "provenance", "inventory", "summary"):
        _require_dict(report.get(key), f"runner report {key}")
    if not isinstance(report.get("results"), list):
        raise ArchiveError("runner report results must be an array")
    configuration = _require_dict(
        report.get("configuration"), "runner configuration"
    )
    if set(configuration) != RUNNER_CONFIGURATION_KEYS:
        raise ArchiveError("runner configuration fields differ")
    performance = _require_dict(
        configuration.get("performance_evidence"), "performance evidence"
    )
    if set(performance) != PERFORMANCE_EVIDENCE_KEYS:
        raise ArchiveError("runner performance-evidence fields differ")
    provenance = _require_dict(report.get("provenance"), "report provenance")
    if set(provenance) != PROVENANCE_KEYS:
        raise ArchiveError("runner provenance fields differ")
    window = _require_dict(
        provenance.get("validation_window"), "validation window"
    )
    if set(window) != VALIDATION_WINDOW_KEYS:
        raise ArchiveError("validation-window fields differ")
    for endpoint_name in ("start", "end"):
        endpoint = _require_dict(
            window.get(endpoint_name), f"validation {endpoint_name}"
        )
        if set(endpoint) != VALIDATION_ENDPOINT_KEYS:
            raise ArchiveError("validation endpoint fields differ")
        runtime = _require_dict(
            endpoint.get("runtime_contract"), "validation runtime contract"
        )
        if set(runtime) != RUNTIME_CONTRACT_KEYS:
            raise ArchiveError("validation runtime-contract fields differ")
    for index, result_value in enumerate(report["results"]):
        result = _require_dict(result_value, f"runner result {index}")
        allowed = set(RESULT_RECORD_KEYS)
        if "missing_dependencies" in result:
            allowed.add("missing_dependencies")
            missing = result["missing_dependencies"]
            if (
                result.get("outcome") != "BLOCKED_DEPENDENCY"
                or not isinstance(missing, list)
                or not missing
                or not all(isinstance(value, str) and value for value in missing)
            ):
                raise ArchiveError("result dependency-blocker fields differ")
        if set(result) != allowed:
            raise ArchiveError("runner result fields differ")
    summary = _require_dict(report.get("summary"), "runner summary")
    if not all(
        isinstance(key, str)
        and key
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for key, value in summary.items()
    ):
        raise ArchiveError("runner summary shape is invalid")
    meanings = _require_dict(
        report.get("exit_code_meaning"), "runner exit-code meanings"
    )
    if set(meanings) != {str(value) for value in RUNNER_EXIT_CODES} or not all(
        isinstance(value, str) and value for value in meanings.values()
    ):
        raise ArchiveError("runner exit-code meanings are invalid")


def _member_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ArchiveError(f"invalid archive member path {value!r}")
    path = pathlib.PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ArchiveError(f"archive member must be a normalized relative path: {value!r}")
    return path.as_posix()


def _repo_relative(value: Any, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArchiveError(f"{label} must be a repository-relative POSIX path")
    path = pathlib.PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ArchiveError(f"{label} escapes the repository: {value!r}")
    return path


def _evidence_record(value: Any, label: str) -> FileSnapshot:
    record = _require_dict(value, label)
    if record.get("available") is not True or not isinstance(record.get("path"), str):
        raise ArchiveError(f"{label} is not an available file record")
    size = record.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ArchiveError(f"{label}.size_bytes is invalid")
    return FileSnapshot(size, _require_sha256(record.get("sha256"), f"{label}.sha256"))


def _reported_run_relative(
    recorded: str, backend: str, filename: str
) -> tuple[pathlib.PurePosixPath, str | None]:
    if not recorded or "\\" in recorded:
        raise ArchiveError(f"invalid reported evidence path {recorded!r}")
    path = pathlib.PurePosixPath(recorded)
    parts = path.parts
    if (
        len(parts) < 4
        or parts[-4] != "runs"
        or parts[-2] != backend
        or parts[-1] != filename
        or parts[-3] in ("", ".", "..")
    ):
        raise ArchiveError(
            f"reported evidence path has the wrong run layout: {recorded!r}"
        )
    relative = pathlib.PurePosixPath(*parts[-4:])
    if not path.is_absolute():
        if path != relative:
            raise ArchiveError(
                f"relative evidence path has an unexpected prefix: {recorded!r}"
            )
        return relative, None
    original_root = pathlib.PurePosixPath(*parts[:-4]).as_posix() or "/"
    if not original_root.startswith("/"):
        original_root = "/" + original_root
    return relative, original_root


def _collect_report_evidence(
    report: dict[str, Any], input_root: pathlib.Path | None
) -> tuple[
    list[tuple[pathlib.PurePosixPath, FileSnapshot, str]],
    set[str],
    set[str],
    set[str],
]:
    results = report.get("results")
    if not isinstance(results, list):
        raise ArchiveError("report.results must be an array")
    references: list[tuple[pathlib.PurePosixPath, FileSnapshot, str]] = []
    seen_relative: set[str] = set()
    seen_result_ids: set[str] = set()
    safe_owners: dict[str, str] = {}
    original_roots: set[str] = set()
    path_modes: set[str] = set()
    executed_sources: set[str] = set()
    runtime_roots: set[str] = set()
    for result_index, result_value in enumerate(results):
        result = _require_dict(result_value, f"report.results[{result_index}]")
        result_id = result.get("id")
        if not isinstance(result_id, str) or not result_id:
            raise ArchiveError(f"report.results[{result_index}].id is invalid")
        if result_id in seen_result_ids:
            raise ArchiveError(f"duplicate result id in report: {result_id!r}")
        seen_result_ids.add(result_id)
        expected_case_directory = runner_contract.case_directory_id(result_id)
        runs = result.get("runs")
        if not isinstance(runs, dict):
            raise ArchiveError(f"report result {result_id!r} has invalid runs")
        if result.get("selected") is True and runs:
            source = _repo_relative(result.get("path"), f"result {result_id!r} path")
            executed_sources.add(source.as_posix())
        result_safe_components: set[str] = set()
        for backend, run_value in sorted(runs.items()):
            if not isinstance(backend, str) or not backend or "/" in backend:
                raise ArchiveError(f"result {result_id!r} has invalid backend key")
            run = _require_dict(run_value, f"result {result_id!r} run {backend!r}")
            if run.get("backend") != backend:
                raise ArchiveError(f"result {result_id!r} backend identity differs")
            evidence_files = _require_dict(
                run.get("evidence_files"),
                f"result {result_id!r} run {backend!r} evidence_files",
            )
            if set(evidence_files) != set(RUN_FILES):
                raise ArchiveError(
                    f"result {result_id!r} run {backend!r} must reference exactly "
                    "stdout, stderr, and statistics"
                )
            run_records: dict[str, dict[str, Any]] = {}
            run_safe_components: set[str] = set()
            for evidence_name, (direct_key, filename) in RUN_FILES.items():
                label = f"result {result_id!r} {backend}.{evidence_name}"
                record = _require_dict(evidence_files[evidence_name], label)
                expected = _evidence_record(record, label)
                direct_path = run.get(direct_key)
                if direct_path != record.get("path"):
                    raise ArchiveError(f"{label} direct path and file record differ")
                relative, original_root = _reported_run_relative(
                    record["path"], backend, filename
                )
                if original_root is not None:
                    original_roots.add(original_root)
                    path_modes.add("absolute")
                else:
                    path_modes.add("relative")
                relative_text = relative.as_posix()
                if relative_text in seen_relative:
                    raise ArchiveError(
                        f"duplicate/colliding raw evidence path: {relative_text}"
                    )
                seen_relative.add(relative_text)
                safe_component = relative.parts[1]
                run_safe_components.add(safe_component)
                owner = safe_owners.setdefault(safe_component, result_id)
                if owner != result_id:
                    raise ArchiveError(
                        f"case directory collision between {owner!r} and {result_id!r}"
                    )
                if input_root is not None:
                    source_path = input_root.joinpath(*relative.parts)
                    _assert_path_chain_without_symlinks(source_path, final_kind="file")
                    if _snapshot(source_path) != expected:
                        raise ArchiveError(f"raw evidence differs from report: {relative_text}")
                references.append((relative, expected, record["path"]))
                run_records[evidence_name] = record
            if len(run_safe_components) != 1:
                raise ArchiveError(
                    f"result {result_id!r} run {backend!r} spans case directories"
                )
            safe_component = next(iter(run_safe_components))
            if safe_component != expected_case_directory:
                raise ArchiveError(
                    f"result {result_id!r} uses a noncanonical case directory"
                )
            result_safe_components.add(safe_component)
            recorded_case_directory = run.get("evidence_case_directory")
            if recorded_case_directory != safe_component:
                raise ArchiveError(
                    f"result {result_id!r} run {backend!r} case directory differs"
                )
            nonce = run.get("run_nonce")
            if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
                raise ArchiveError(
                    f"result {result_id!r} run {backend!r} has an invalid nonce"
                )
            runtime_root = pathlib.PurePosixPath(
                "runs", safe_component, backend, "runtime", nonce
            ).as_posix()
            if runtime_root in runtime_roots:
                raise ArchiveError(f"duplicate runtime root: {runtime_root}")
            runtime_roots.add(runtime_root)
        if len(result_safe_components) > 1:
            raise ArchiveError(
                f"result {result_id!r} uses multiple case directories"
            )
        comparison = result.get("comparison")
        needs_raw_metrics = (
            isinstance(comparison, dict)
            and comparison.get("mode") == "json_metrics"
            and set(runs) == {"cpu", "cuda"}
            and all(
                isinstance(run, dict)
                and str(run.get("outcome", "")).startswith("PASS")
                for run in runs.values()
            )
        )
        if needs_raw_metrics and "raw_metric_evidence" not in comparison:
            raise ArchiveError(
                f"result {result_id!r} lacks mandatory raw metric evidence"
            )
        if (
            isinstance(comparison, dict)
            and comparison.get("mode") == "json_metrics"
            and "raw_metric_evidence" in comparison
        ):
            raw = _require_dict(
                comparison.get("raw_metric_evidence"),
                f"result {result_id!r} raw_metric_evidence",
            )
            expected_raw = {
                "cpu_stdout": (
                    _require_dict(runs.get("cpu"), "CPU run")
                    .get("evidence_files", {})
                    .get("stdout")
                ),
                "cuda_stdout": (
                    _require_dict(runs.get("cuda"), "CUDA run")
                    .get("evidence_files", {})
                    .get("stdout")
                ),
            }
            if raw != expected_raw:
                raise ArchiveError(
                    f"result {result_id!r} raw metric evidence is not bound "
                    "to its run stdout records"
                )
            failure_record = comparison.get("failure_evidence")
            if comparison.get("failure_count", 0) > 0 and failure_record is None:
                raise ArchiveError(
                    f"result {result_id!r} lacks its metric failure sidecar"
                )
            if failure_record is not None:
                if comparison.get("failure_count", 0) <= 0:
                    raise ArchiveError(
                        f"result {result_id!r} has a failure sidecar without failures"
                    )
                expected = _evidence_record(
                    failure_record,
                    f"result {result_id!r} failure_evidence",
                )
                relative, original_root = _reported_run_relative(
                    failure_record["path"],
                    "comparison",
                    "json-metric-failures.ndjson",
                )
                if not result_safe_components or relative.parts[1] != next(
                    iter(result_safe_components)
                ):
                    raise ArchiveError(
                        f"result {result_id!r} failure sidecar uses the wrong directory"
                    )
                if original_root is not None:
                    original_roots.add(original_root)
                    path_modes.add("absolute")
                else:
                    path_modes.add("relative")
                relative_text = relative.as_posix()
                if relative_text in seen_relative:
                    raise ArchiveError(
                        f"duplicate/colliding raw evidence path: {relative_text}"
                    )
                seen_relative.add(relative_text)
                if input_root is not None:
                    source_path = input_root.joinpath(*relative.parts)
                    _assert_path_chain_without_symlinks(
                        source_path, final_kind="file"
                    )
                    if _snapshot(source_path) != expected:
                        raise ArchiveError(
                            f"failure sidecar differs from report: {relative_text}"
                        )
                references.append((relative, expected, failure_record["path"]))
    if len(original_roots) > 1:
        raise ArchiveError("report evidence paths name multiple original output roots")
    if len(path_modes) > 1:
        raise ArchiveError("report mixes absolute and relative evidence paths")
    return references, executed_sources, original_roots, runtime_roots


def _inventory_glob_matches(path: str, pattern: str) -> bool:
    """Match the runner's recursive inventory globs without a source tree."""
    candidates = {pattern}
    pending = [pattern]
    while pending:
        candidate = pending.pop()
        marker = "**/"
        index = candidate.find(marker)
        if index < 0:
            continue
        shortened = candidate[:index] + candidate[index + len(marker) :]
        if shortened not in candidates:
            candidates.add(shortened)
            pending.append(shortened)
    return any(pathlib.PurePath(path).match(value) for value in candidates)


def _merge_case(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key != "glob":
            result[key] = copy.deepcopy(value)
    return result


def _materialize_report_catalog(
    manifest: dict[str, Any],
    report: dict[str, Any],
    source_paths: Iterable[str],
) -> dict[str, dict[str, Any]]:
    if manifest.get("schema_version") != runner_contract.SCHEMA_VERSION:
        raise ArchiveError("archived validation manifest schema is invalid")
    inventory_rules = manifest.get("inventory")
    overrides = manifest.get("overrides")
    if not isinstance(inventory_rules, list) or not inventory_rules:
        raise ArchiveError("archived validation inventory is invalid")
    if not isinstance(overrides, list):
        raise ArchiveError("archived validation overrides are invalid")
    results = report.get("results")
    if not isinstance(results, list):
        raise ArchiveError("runner report results are invalid")

    # The receipt source manifest, not the report, is the independent universe
    # from which inventory globs are expanded.  Otherwise a producer can omit a
    # source file and make the report and its totals self-consistently smaller.
    canonical_source_paths: list[str] = []
    for index, value in enumerate(source_paths):
        canonical_source_paths.append(
            _repo_relative(value, f"receipt source path {index}").as_posix()
        )
    if len(set(canonical_source_paths)) != len(canonical_source_paths):
        raise ArchiveError("receipt source paths contain duplicates")

    cases: dict[str, dict[str, Any]] = {}
    matched_inventory = Counter()
    for path_text in sorted(canonical_source_paths):
        matching_rules = []
        for rule_index, rule_value in enumerate(inventory_rules):
            rule = _require_dict(rule_value, f"inventory rule {rule_index}")
            pattern = rule.get("glob")
            if not isinstance(pattern, str):
                raise ArchiveError("inventory rule glob is invalid")
            if _inventory_glob_matches(path_text, pattern):
                matching_rules.append((rule_index, rule))
        if not matching_rules:
            continue
        if len(matching_rules) != 1:
            raise ArchiveError(
                f"receipt source path {path_text!r} matches "
                f"{len(matching_rules)} inventory rules"
            )
        rule_index, rule = matching_rules[0]
        matched_inventory[rule_index] += 1
        case = _merge_case(rule, {"path": path_text})
        case.pop("glob", None)
        case["id"] = case.get("id", path_text)
        cases[path_text] = case
    if set(matched_inventory) != set(range(len(inventory_rules))):
        raise ArchiveError("receipt sources do not cover every inventory rule")

    report_paths: list[str] = []
    for index, result_value in enumerate(results):
        result = _require_dict(result_value, f"report result {index}")
        report_paths.append(
            _repo_relative(
                result.get("path"), f"report result {index} path"
            ).as_posix()
        )
    if len(set(report_paths)) != len(report_paths):
        raise ArchiveError("duplicate report case path")
    if set(report_paths) != set(cases):
        missing = sorted(set(cases) - set(report_paths))
        extra = sorted(set(report_paths) - set(cases))
        raise ArchiveError(
            "report case inventory differs from receipt source expansion"
            f" (missing={missing}, extra={extra})"
        )
    if report_paths != sorted(cases):
        raise ArchiveError("report cases are not in canonical source order")

    matched_overrides = Counter()
    for override_index, override_value in enumerate(overrides):
        override = _require_dict(
            override_value, f"manifest override {override_index}"
        )
        pattern = override.get("glob")
        if not isinstance(pattern, str):
            raise ArchiveError("manifest override glob is invalid")
        for path_text, case in list(cases.items()):
            if pathlib.PurePath(path_text).match(pattern):
                cases[path_text] = _merge_case(case, override)
                matched_overrides[override_index] += 1
        if matched_overrides[override_index] == 0:
            raise ArchiveError(
                f"manifest override matched no reported case: {pattern}"
            )

    for path_text, case in cases.items():
        try:
            runner_contract.validate_case(
                case, pathlib.Path("/"), require_source_file=False
            )
        except runner_contract.ManifestError as exc:
            raise ArchiveError(
                f"archived validation case {path_text!r} is invalid: {exc}"
            ) from exc
    case_ids = [case.get("id") for case in cases.values()]
    if len(set(case_ids)) != len(case_ids):
        raise ArchiveError("archived validation case IDs are not unique")
    try:
        runner_contract.validate_coverage_graph(list(cases.values()))
    except runner_contract.ManifestError as exc:
        raise ArchiveError(
            f"archived validation coverage graph is invalid: {exc}"
        ) from exc

    configuration = _require_dict(
        report.get("configuration"), "runner configuration"
    )
    tiers = configuration.get("tiers")
    case_ids = configuration.get("case_ids")
    backends = configuration.get("backends")
    if (
        not isinstance(tiers, list)
        or tiers != sorted(tiers)
        or not all(isinstance(value, str) for value in tiers)
        or not isinstance(case_ids, list)
        or case_ids != sorted(case_ids)
        or not all(isinstance(value, str) for value in case_ids)
        or backends not in (["cpu"], ["cuda"], ["cpu", "cuda"])
    ):
        raise ArchiveError("runner selection configuration is invalid")
    tier_set = set(tiers)
    id_set = set(case_ids)
    known_tiers = {
        tier
        for case in cases.values()
        for tier in case.get("tier", [])
        if isinstance(tier, str)
    }
    unknown_tiers = tier_set - known_tiers
    if unknown_tiers:
        raise ArchiveError(
            "runner configuration contains unknown tier selections: "
            + ", ".join(sorted(unknown_tiers))
        )
    known_selections = set(cases)
    known_selections.update(case.get("id") for case in cases.values())
    unknown_ids = id_set - known_selections
    if unknown_ids:
        raise ArchiveError(
            "runner configuration contains unknown case selections: "
            + ", ".join(sorted(unknown_ids))
        )
    results_by_path = {
        _require_dict(value, "report result")["path"]: value
        for value in results
    }
    selected_count = 0
    for path_text, case in cases.items():
        result = _require_dict(results_by_path[path_text], "report result")
        selected = (
            case.get("id") in id_set or path_text in id_set
            if id_set
            else bool(tier_set.intersection(case.get("tier", [])))
        )
        expected_fields = {
            "id": case.get("id"),
            "path": path_text,
            "kind": case.get("kind"),
            "tier": case.get("tier"),
            "disposition": case.get("disposition"),
            "compute_scope": case.get("compute_scope"),
            "reason": case.get("reason"),
            "milestone": case.get("milestone"),
            "selected": selected,
        }
        for key, expected in expected_fields.items():
            if result.get(key) != expected:
                raise ArchiveError(
                    f"report case {path_text!r} {key} differs from manifest"
                )
        selected_count += int(selected)
        if not selected:
            if (
                result.get("outcome") != "NOT_SELECTED"
                or result.get("runs") != {}
                or result.get("comparison") is not None
            ):
                raise ArchiveError(
                    f"unselected case {path_text!r} has execution evidence"
                )
        elif case.get("disposition") == runner_contract.COVERED_DISPOSITION:
            if result.get("runs") != {} or result.get("comparison") is not None:
                raise ArchiveError(
                    f"covered case {path_text!r} has execution evidence"
                )
        elif case.get("disposition") != runner_contract.RUN_DISPOSITION:
            if (
                result.get("outcome") != str(case.get("disposition")).upper()
                or result.get("runs") != {}
                or result.get("comparison") is not None
            ):
                raise ArchiveError(
                    f"classified case {path_text!r} differs from manifest"
                )
    inventory = _require_dict(report.get("inventory"), "report inventory")
    if inventory != {
        "total": len(cases),
        "selected": selected_count,
        "scheme_scope": "excluded by project requirement",
    }:
        raise ArchiveError("report inventory totals differ from manifest")
    if selected_count <= 0:
        raise ArchiveError("runner selection matched no inventory cases")
    return cases


def _receipt_repository(receipt: dict[str, Any]) -> pathlib.Path:
    configuration = _require_dict(
        receipt.get("configuration"), "receipt configuration"
    )
    argv = configuration.get("configure_argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise ArchiveError("receipt configure argument list is invalid")
    prefixes = [value.split("=", 1)[1] for value in argv if value.startswith("--prefix=")]
    if len(prefixes) != 1:
        raise ArchiveError("receipt install prefix is ambiguous")
    install = pathlib.Path(prefixes[0])
    if (
        not install.is_absolute()
        or install.name != "meep-cuda-mpi-python-fp32"
        or install.parent.name != "install"
    ):
        raise ArchiveError("receipt install prefix is not authoritative")
    return install.parent.parent.resolve()


def _receipt_absolute_path(
    value: Any, repository: pathlib.Path, label: str
) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise ArchiveError(f"{label} path is invalid")
    path = pathlib.PurePosixPath(value)
    if ".." in path.parts:
        raise ArchiveError(f"{label} path escapes the repository")
    candidate = pathlib.Path(value)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repository / candidate).resolve()
    )


def _absolute_receipt_record(
    value: Any, repository: pathlib.Path, label: str
) -> dict[str, Any]:
    record = _require_dict(value, label)
    if not {"path", "size_bytes", "sha256"}.issubset(record):
        raise ArchiveError(f"{label} file record fields differ")
    size = record.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ArchiveError(f"{label} size is invalid")
    return {
        "path": str(_receipt_absolute_path(record.get("path"), repository, label)),
        "size_bytes": size,
        "sha256": _require_sha256(record.get("sha256"), f"{label} SHA"),
    }


def _receipt_runtime_contract(
    receipt: dict[str, Any], repository: pathlib.Path
) -> dict[str, Any]:
    toolchain = _require_dict(receipt.get("toolchain"), "receipt toolchain")
    artifacts = _require_dict(receipt.get("artifacts"), "receipt artifacts")
    configuration_files = _require_dict(
        receipt.get("configuration_files"), "receipt configuration files"
    )
    manifests = _require_dict(receipt.get("manifests"), "receipt manifests")
    in_place = _require_dict(
        manifests.get("in_place_python"), "receipt in-place Python manifest"
    )
    in_place_root = _receipt_absolute_path(
        in_place.get("root"), repository, "in-place Python manifest root"
    )
    in_place_files = in_place.get("files")
    if not isinstance(in_place_files, list):
        raise ArchiveError("in-place Python manifest files are invalid")
    module_relative = (
        "__init__.py" if in_place_root.name == "meep" else "meep/__init__.py"
    )
    module_records = [
        value
        for value in in_place_files
        if isinstance(value, dict) and value.get("path") == module_relative
    ]
    if len(module_records) != 1:
        raise ArchiveError("receipt does not uniquely bind meep/__init__.py")
    module = dict(module_records[0])
    module["path"] = str(in_place_root / module_relative)
    module_record = _absolute_receipt_record(
        module, pathlib.Path("/"), "receipt Meep module"
    )

    def manifest_root(name: str) -> str:
        manifest = _require_dict(
            manifests.get(name), f"receipt {name} manifest"
        )
        return str(
            _receipt_absolute_path(
                manifest.get("root"), repository, f"receipt {name} root"
            )
        )

    return {
        "python_executable": _absolute_receipt_record(
            toolchain.get("python"), repository, "receipt Python executable"
        ),
        "meep_module": module_record,
        "extension": _absolute_receipt_record(
            artifacts.get("python_extension"),
            repository,
            "receipt Python extension",
        ),
        "libmeep": _absolute_receipt_record(
            artifacts.get("libmeep"), repository, "receipt libmeep"
        ),
        "fontconfig_file": _absolute_receipt_record(
            configuration_files.get("qualification_fontconfig"),
            repository,
            "receipt qualification Fontconfig",
        ),
        "build_home": manifest_root("build_home"),
        "qualification_home": manifest_root("qualification_home"),
        "installed_environment": manifest_root("installed_environment"),
        "receipt_id": _require_sha256(
            receipt.get("receipt_id"), "receipt ID"
        ),
    }


def _validation_runtime_contract(
    report: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    window = _require_dict(
        _require_dict(report.get("provenance"), "report provenance").get(
            "validation_window"
        ),
        "validation window",
    )
    start = _require_dict(window.get("start"), "validation start")
    end = _require_dict(window.get("end"), "validation end")
    start_runtime = _require_dict(
        start.get("runtime_contract"), "validation start runtime contract"
    )
    end_runtime = _require_dict(
        end.get("runtime_contract"), "validation end runtime contract"
    )
    if not start_runtime or start_runtime != end_runtime:
        raise ArchiveError("validation runtime contract is absent or changed")
    expected_runtime = _receipt_runtime_contract(
        receipt, _receipt_repository(receipt)
    )
    if start_runtime != expected_runtime:
        raise ArchiveError("validation runtime contract differs from build receipt")
    return start_runtime


def _canonical_case_command(
    case: dict[str, Any],
    report: dict[str, Any],
    expected_repository: pathlib.Path,
    expected_python: pathlib.Path,
) -> list[str]:
    provenance = _require_dict(report.get("provenance"), "report provenance")
    repo_text = provenance.get("repository")
    python_text = provenance.get("python_executable")
    if not isinstance(repo_text, str) or not pathlib.Path(repo_text).is_absolute():
        raise ArchiveError("report repository path is invalid")
    if not isinstance(python_text, str) or not pathlib.Path(python_text).is_absolute():
        raise ArchiveError("report Python executable path is invalid")
    repo = pathlib.Path(repo_text).resolve()
    python = pathlib.Path(python_text).resolve()
    if repo != expected_repository.resolve():
        raise ArchiveError("report repository differs from the build receipt")
    if python != expected_python.resolve():
        raise ArchiveError("report Python differs from the runtime contract")
    try:
        return runner_contract.case_command(case, python, repo)
    except runner_contract.ManifestError as exc:
        raise ArchiveError(f"cannot reconstruct case command: {exc}") from exc


def _raw_run_member(
    evidence_root: pathlib.Path,
    run: dict[str, Any],
    backend: str,
    evidence_name: str,
) -> pathlib.Path:
    filename = RUN_FILES[evidence_name][1]
    record = _require_dict(
        _require_dict(run.get("evidence_files"), "run evidence files").get(
            evidence_name
        ),
        f"run {evidence_name}",
    )
    relative, _ = _reported_run_relative(record.get("path"), backend, filename)
    path = evidence_root.joinpath(*relative.parts)
    _assert_path_chain_without_symlinks(path, final_kind="file")
    return path


def _verify_run_environment_contract(
    *,
    run: dict[str, Any],
    backend: str,
    case: dict[str, Any],
    report: dict[str, Any],
    runtime: dict[str, Any],
    expected_repository: pathlib.Path,
    evidence_root: pathlib.Path,
    bind_output_root: bool,
) -> None:
    contract = _require_dict(
        run.get("environment_contract"), "run environment contract"
    )
    if set(contract) != RUN_ENVIRONMENT_CONTRACT_KEYS:
        raise ArchiveError("run environment contract fields differ")
    environment = _require_dict(
        contract.get("environment"), "run exact environment"
    )
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ArchiveError("run exact environment is invalid")
    if (
        contract.get("keys") != sorted(environment)
        or contract.get("sha256")
        != runner_contract.environment_sha256(environment)
    ):
        raise ArchiveError("run exact environment digest differs")

    allowed = (
        set(runner_contract.SAFE_AMBIENT_ENVIRONMENT)
        | {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"}
        | RUN_CONTROLLED_ENVIRONMENT
        | ({"GPMEEP_VALIDATION_STRICT_CUDA"} if backend == "cuda" else set())
    )
    required = set(RUN_CONTROLLED_ENVIRONMENT)
    if backend == "cuda":
        required.add("GPMEEP_VALIDATION_STRICT_CUDA")
    if not required.issubset(environment) or not set(environment).issubset(allowed):
        raise ArchiveError("run exact environment key allowlist differs")

    nonce = run.get("run_nonce")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise ArchiveError("run nonce is invalid")
    safe_id = runner_contract.case_directory_id(str(case.get("id")))
    expected_cwd = (
        pathlib.PurePosixPath("runs") / safe_id / backend / "work"
    ).as_posix()
    if (
        run.get("cwd") != expected_cwd
        or run.get("evidence_case_directory") != safe_id
    ):
        raise ArchiveError("run working/evidence directory differs")

    home = pathlib.Path(environment["HOME"])
    runtime_root = home.parent
    expected_tail = ("runs", safe_id, backend, "runtime", nonce)
    if (
        not home.is_absolute()
        or home.name != "home"
        or tuple(runtime_root.parts[-len(expected_tail) :]) != expected_tail
    ):
        raise ArchiveError("run isolated runtime root differs")
    case_root = runtime_root.parents[1]
    claimed_output_root = runtime_root.parents[4]
    if bind_output_root and claimed_output_root.resolve() != evidence_root.resolve():
        raise ArchiveError("run isolated environment root differs from evidence root")
    derived_paths = {
        "HOME": runtime_root / "home",
        "MPLCONFIGDIR": runtime_root / "matplotlib",
        "XDG_CACHE_HOME": runtime_root / "cache",
        "XDG_CONFIG_HOME": runtime_root / "config",
        "GPMEEP_VALIDATION_STATS_FILE": case_root / "gpu-statistics.json",
    }
    if any(environment[key] != str(value) for key, value in derived_paths.items()):
        raise ArchiveError("run isolated environment path differs")

    provenance = _require_dict(report.get("provenance"), "report provenance")
    build_python_text = provenance.get("build_python")
    if (
        not isinstance(build_python_text, str)
        or not pathlib.Path(build_python_text).is_absolute()
    ):
        raise ArchiveError("report build-Python path is invalid")
    meep_module = _require_dict(runtime.get("meep_module"), "runtime Meep module")
    extension = _require_dict(runtime.get("extension"), "runtime extension")
    if not isinstance(meep_module.get("path"), str) or not isinstance(
        extension.get("path"), str
    ):
        raise ArchiveError("runtime Python package paths are invalid")
    module_path = pathlib.Path(meep_module["path"]).resolve()
    extension_path = pathlib.Path(extension["path"]).resolve()
    if module_path.parent != extension_path.parent:
        raise ArchiveError("runtime Python module/extension package differs")
    trusted_build_python = module_path.parent.parent
    if pathlib.Path(build_python_text).resolve() != trusted_build_python:
        raise ArchiveError("report build-Python path differs from build receipt")
    expected_pythonpath = os.pathsep.join(
        (
            str(expected_repository / "scripts/python-validation"),
            str(trusted_build_python),
        )
    )
    libmeep = _require_dict(runtime.get("libmeep"), "runtime libmeep")
    installed = runtime.get("installed_environment")
    fontconfig = _require_dict(
        runtime.get("fontconfig_file"), "runtime Fontconfig file"
    )
    if (
        not isinstance(libmeep.get("path"), str)
        or not pathlib.Path(libmeep["path"]).is_absolute()
        or not isinstance(installed, str)
        or not pathlib.Path(installed).is_absolute()
        or not isinstance(fontconfig.get("path"), str)
        or not pathlib.Path(fontconfig["path"]).is_absolute()
    ):
        raise ArchiveError("runtime library/isolation paths are invalid")
    library_path = os.pathsep.join(
        dict.fromkeys(
            (
                str(pathlib.Path(libmeep["path"]).resolve().parent),
                str(pathlib.Path(installed).resolve() / "lib"),
            )
        )
    )
    fixed_values = {
        "PYTHONPATH": expected_pythonpath,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
        "FONTCONFIG_FILE": str(pathlib.Path(fontconfig["path"]).resolve()),
        "CUDA_CACHE_DISABLE": "1",
        "JAX_PLATFORMS": "cpu",
        "MEEP_GPU_BACKEND": backend,
        "GPMEEP_VALIDATION_EXPECTED_BACKEND": backend,
        "GPMEEP_VALIDATION_RUN_NONCE": nonce,
        "GPMEEP_VALIDATION_BUILD_RECEIPT_ID": str(runtime.get("receipt_id")),
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "FALSE",
        "LD_LIBRARY_PATH": library_path,
    }
    if backend == "cuda":
        fixed_values["GPMEEP_VALIDATION_STRICT_CUDA"] = "1"
    if any(environment.get(key) != value for key, value in fixed_values.items()):
        raise ArchiveError("run controlled environment value differs")

    direct = {
        "MEEP_GPU_BACKEND": backend,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_CACHE_DISABLE": "1",
        "JAX_PLATFORMS": "cpu",
        "HOME": environment["HOME"],
        "MPLCONFIGDIR": environment["MPLCONFIGDIR"],
        "XDG_CACHE_HOME": environment["XDG_CACHE_HOME"],
        "XDG_CONFIG_HOME": environment["XDG_CONFIG_HOME"],
        "FONTCONFIG_FILE": environment["FONTCONFIG_FILE"],
        "strict_cuda": backend == "cuda",
    }
    if any(contract.get(key) != value for key, value in direct.items()):
        raise ArchiveError("run summarized environment contract differs")


def _replay_run_outcome(
    *,
    case: dict[str, Any],
    backend: str,
    run: dict[str, Any],
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
    contract_ok: bool,
    contract_problems: list[str],
) -> str:
    stdout_valid = runner_contract.is_strict_utf8_file(stdout_path)
    stderr_valid = runner_contract.is_strict_utf8_file(stderr_path)
    invalid_utf8 = not (stdout_valid and stderr_valid)
    output_limit = run.get("output_limit")
    if output_limit not in (None, "stdout", "stderr"):
        raise ArchiveError("run output-limit marker is invalid")
    timeout = run.get("timeout")
    if not isinstance(timeout, bool):
        raise ArchiveError("run timeout marker is invalid")
    exit_code = run.get("exit_code")
    if not (
        exit_code is None
        or (isinstance(exit_code, int) and not isinstance(exit_code, bool))
    ):
        raise ArchiveError("run exit code is invalid")

    stdout = ""
    stderr = ""
    if output_limit is None and not invalid_utf8 and case.get("kind") == "unittest":
        stdout = stdout_path.read_text(encoding="utf-8")
        stderr = stderr_path.read_text(encoding="utf-8")
    combined = stdout + "\n" + stderr
    if case.get("kind") == "unittest":
        skips = runner_contract.parse_unittest_skips(combined)
        details = runner_contract.parse_unittest_skip_details(combined)
        unittest_ok, evidence, unittest_problems = (
            runner_contract.verify_unittest_test_contract(case, stderr)
        )
    else:
        skips = 0
        details = []
        unittest_ok = True
        evidence = {
            "reported_counts": [],
            "reported_count": None,
            "identities": [],
            "expected_count": None,
            "expected_identities": None,
            "terminal_summaries": [],
        }
        unittest_problems = []
    skip_ok, skip_problems = runner_contract.verify_unittest_skip_policy(
        case, skips, details
    )
    allowed_skips = case.get("allowed_unittest_skips", 0)
    skips_ok = skips <= allowed_skips and skip_ok
    expected_fields = {
        "unittest_skips": skips,
        "unittest_skip_details": details,
        "allowed_unittest_skips": allowed_skips,
        "unittest_skip_policy_problems": skip_problems,
        "unittest_reported_test_counts": evidence["reported_counts"],
        "unittest_test_count": evidence["reported_count"],
        "unittest_test_identities": evidence["identities"],
        "expected_unittest_test_count": evidence["expected_count"],
        "expected_unittest_test_identities": evidence["expected_identities"],
        "unittest_test_contract_ok": unittest_ok,
        "unittest_test_contract_problems": unittest_problems,
        "unittest_terminal_summaries": evidence["terminal_summaries"],
        "backend_contract_ok": contract_ok,
        "backend_contract_problems": contract_problems,
    }
    if any(run.get(key) != value for key, value in expected_fields.items()):
        raise ArchiveError("run raw unittest/backend replay differs")

    process_ok = (
        exit_code == 0
        and not timeout
        and output_limit is None
        and not invalid_utf8
    )
    if timeout:
        return "TIMEOUT"
    if output_limit is not None:
        return "OUTPUT_LIMIT"
    if invalid_utf8:
        return "OUTPUT_INVALID_UTF8"
    if not process_ok:
        return "PROCESS_FAILED"
    if not skips_ok:
        return "UNDECLARED_SKIP"
    if not unittest_ok:
        return "UNITTEST_CONTRACT_FAILED"
    if not contract_ok:
        return "BACKEND_CONTRACT_FAILED"
    if skips:
        return "PASS_WITH_DECLARED_SKIPS"
    return "PASS"


def _replay_report_semantics(
    report: dict[str, Any],
    manifest: dict[str, Any],
    evidence_root: pathlib.Path,
    receipt: dict[str, Any],
    source_paths: Iterable[str],
    *,
    bind_environment_to_evidence_root: bool,
) -> None:
    cases = _materialize_report_catalog(manifest, report, source_paths)
    configuration = _require_dict(
        report.get("configuration"), "runner configuration"
    )
    performance = _require_dict(
        configuration.get("performance_evidence"), "performance evidence"
    )
    required_timing_reasons = {
        "whole-process single-sample timing includes Python startup",
        "no benchmark warm-up or repeated-sample statistics",
    }
    if (
        performance.get("valid_for_speed_gate") is not False
        or not isinstance(performance.get("invalid_reasons"), list)
        or not required_timing_reasons.issubset(
            set(performance["invalid_reasons"])
        )
    ):
        raise ArchiveError("runner timing-validity contract is invalid")

    results = report["results"]
    start_runtime = _validation_runtime_contract(report, receipt)
    expected_repository = _receipt_repository(receipt)
    python_record = _require_dict(
        start_runtime.get("python_executable"), "runtime Python executable"
    )
    python_path = python_record.get("path")
    if not isinstance(python_path, str) or not pathlib.Path(python_path).is_absolute():
        raise ArchiveError("runtime Python executable path is invalid")
    expected_python = pathlib.Path(python_path).resolve()
    for result_value in results:
        result = _require_dict(result_value, "report result")
        if not result.get("selected"):
            continue
        case = cases[result["path"]]
        if case.get("disposition") != runner_contract.RUN_DISPOSITION:
            continue
        runs = _require_dict(result.get("runs"), "report runs")
        if not runs:
            # Dependency blockers are intentionally represented without raw
            # runs and remain visible through the recomputed summary/exit code.
            if result.get("outcome") != "BLOCKED_DEPENDENCY":
                raise ArchiveError("selected runnable case lacks run evidence")
            continue
        if list(sorted(runs)) != list(sorted(configuration["backends"])):
            raise ArchiveError("case run backends differ from configuration")
        expected_command = _canonical_case_command(
            case, report, expected_repository, expected_python
        )
        for backend, run_value in runs.items():
            run = _require_dict(run_value, f"run {backend}")
            if set(run) != RUN_RECORD_KEYS:
                raise ArchiveError("run record has unexpected or missing fields")
            if run.get("backend") != backend or run.get("command") != expected_command:
                raise ArchiveError(
                    f"case {result['path']!r} command differs from manifest"
                )
            duration = run.get("duration_seconds")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not float("-inf") < float(duration) < float("inf")
                or duration < 0
            ):
                raise ArchiveError("run duration contract is invalid")
            if not isinstance(run.get("started_at_utc"), str) or not run[
                "started_at_utc"
            ]:
                raise ArchiveError("run start timestamp is invalid")
            _verify_run_environment_contract(
                run=run,
                backend=backend,
                case=case,
                report=report,
                runtime=start_runtime,
                expected_repository=expected_repository,
                evidence_root=evidence_root,
                bind_output_root=bind_environment_to_evidence_root,
            )
            stats_path = _raw_run_member(
                evidence_root, run, backend, "statistics"
            )
            stdout_path = _raw_run_member(
                evidence_root, run, backend, "stdout"
            )
            stderr_path = _raw_run_member(
                evidence_root, run, backend, "stderr"
            )
            raw_stats, _ = _load_json_file(
                stats_path, f"raw statistics for {result['path']} {backend}"
            )
            if raw_stats != run.get("statistics"):
                raise ArchiveError("embedded statistics differ from raw evidence")
            phase_problems = runner_contract.validate_phase_counters(raw_stats)
            if phase_problems:
                raise ArchiveError(
                    "raw GPU phase-counter contract failed: "
                    + "; ".join(phase_problems)
                )
            if raw_stats.get("run_nonce") not in (None, run.get("run_nonce")):
                raise ArchiveError("raw statistics nonce differs from run")
            if (
                case.get("require_active_backend_at_exit", True)
                and raw_stats.get("active_backend") not in (None, backend)
            ):
                raise ArchiveError("raw statistics active backend differs")
            if raw_stats.get("strict_cuda_marker") != (backend == "cuda"):
                raise ArchiveError("raw statistics strict CUDA marker differs")
            contract_ok, contract_problems = (
                runner_contract.verify_backend_contract(
                    case,
                    backend,
                    raw_stats,
                    start_runtime,
                    run.get("run_nonce"),
                )
            )
            stdout_size = stdout_path.stat().st_size
            stderr_size = stderr_path.stat().st_size
            if run.get("output_sizes") != {
                "stdout": stdout_size,
                "stderr": stderr_size,
                "combined": stdout_size + stderr_size,
            }:
                raise ArchiveError("run output-size contract differs")
            if run.get("output_limits") != {
                "stdout": runner_contract.MAX_BACKEND_STDOUT_BYTES,
                "stderr": runner_contract.MAX_BACKEND_STDERR_BYTES,
                "combined": runner_contract.MAX_BACKEND_OUTPUT_BYTES,
            }:
                raise ArchiveError("run output-limit contract differs")
            replayed_outcome = _replay_run_outcome(
                case=case,
                backend=backend,
                run=run,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                contract_ok=contract_ok,
                contract_problems=contract_problems,
            )
            if run.get("outcome") != replayed_outcome:
                raise ArchiveError("run outcome differs from raw replay")

        failed = [
            backend
            for backend, run in runs.items()
            if not str(run.get("outcome", "")).startswith("PASS")
        ]
        if failed:
            expected_comparison = {
                "mode": case["comparison"]["mode"],
                "outcome": "NOT_COMPARABLE",
                "reason": f"backend runs did not pass: {', '.join(failed)}",
            }
        elif set(runs) != {"cpu", "cuda"}:
            expected_comparison = {
                "mode": case["comparison"]["mode"],
                "outcome": "SINGLE_BACKEND_ONLY",
                "reason": (
                    "both CPU and CUDA are required for a cross-backend comparison"
                ),
            }
        else:
            cpu_seconds = runs["cpu"].get("duration_seconds")
            cuda_seconds = runs["cuda"].get("duration_seconds")
            if (
                not isinstance(cpu_seconds, (int, float))
                or isinstance(cpu_seconds, bool)
                or not isinstance(cuda_seconds, (int, float))
                or isinstance(cuda_seconds, bool)
                or cpu_seconds < 0
                or cuda_seconds < 0
            ):
                raise ArchiveError("run duration contract is invalid")
            expected_comparison = {
                "mode": case["comparison"]["mode"],
                "cpu_seconds": cpu_seconds,
                "cuda_seconds": cuda_seconds,
                "speedup_cpu_over_cuda": (
                    cpu_seconds / cuda_seconds if cuda_seconds > 0 else None
                ),
            }
            cpu_stdout_path = _raw_run_member(
                evidence_root, runs["cpu"], "cpu", "stdout"
            )
            cuda_stdout_path = _raw_run_member(
                evidence_root, runs["cuda"], "cuda", "stdout"
            )
            mode = case["comparison"]["mode"]
            if mode == "embedded_oracle":
                expected_comparison.update(
                    {
                        "outcome": "PASS",
                        "reason": (
                            "both backends passed the test's embedded numerical oracle"
                        ),
                    }
                )
            elif mode == "normalized_stdout":
                try:
                    cpu_stdout = cpu_stdout_path.read_text(encoding="utf-8")
                    cuda_stdout = cuda_stdout_path.read_text(encoding="utf-8")
                except UnicodeDecodeError as exc:
                    raise ArchiveError(
                        "raw stdout is not valid UTF-8"
                    ) from exc
                cpu_value = runner_contract.normalize_stdout(
                    cpu_stdout, case["comparison"]
                )
                cuda_value = runner_contract.normalize_stdout(
                    cuda_stdout, case["comparison"]
                )
                equal = cpu_value == cuda_value
                expected_comparison.update(
                    {
                        "outcome": "PASS" if equal else "MISMATCH",
                        "reason": (
                            "normalized stdout is equal"
                            if equal
                            else "normalized stdout differs"
                        ),
                        "normalized_stdout_evidence": {
                            "preview_byte_limit": (
                                runner_contract.MAX_NORMALIZED_STDOUT_PREVIEW_BYTES
                            ),
                            "cpu": runner_contract.bounded_text_record(
                                cpu_value,
                                runner_contract.MAX_NORMALIZED_STDOUT_PREVIEW_BYTES,
                            ),
                            "cuda": runner_contract.bounded_text_record(
                                cuda_value,
                                runner_contract.MAX_NORMALIZED_STDOUT_PREVIEW_BYTES,
                            ),
                        },
                    }
                )
            elif mode == "json_metrics":
                try:
                    expected_comparison.update(
                        runner_contract.compare_json_metric_files(
                            cpu_stdout_path,
                            cuda_stdout_path,
                            case["comparison"],
                            expected_cpu_stdout=runs["cpu"][
                                "evidence_files"
                            ]["stdout"],
                            expected_cuda_stdout=runs["cuda"][
                                "evidence_files"
                            ]["stdout"],
                        )
                    )
                except runner_contract.ManifestError as exc:
                    expected_comparison.update(
                        {"outcome": "MISMATCH", "reason": str(exc)}
                    )
                expected_comparison["raw_metric_evidence"] = {
                    "cpu_stdout": runs["cpu"]["evidence_files"]["stdout"],
                    "cuda_stdout": runs["cuda"]["evidence_files"]["stdout"],
                }
                actual_comparison = _require_dict(
                    result.get("comparison"), "result comparison"
                )
                failure_evidence = actual_comparison.get("failure_evidence")
                if failure_evidence is not None:
                    expected_comparison["failure_evidence"] = failure_evidence
                    failure_relative, _ = _reported_run_relative(
                        _require_dict(
                            failure_evidence, "failure evidence"
                        ).get("path"),
                        "comparison",
                        "json-metric-failures.ndjson",
                    )
                    sidecar = evidence_root.joinpath(*failure_relative.parts)
                    sidecar_problems = runner_contract.verify_failure_sidecar(
                        sidecar, actual_comparison
                    )
                    if sidecar_problems:
                        raise ArchiveError(
                            "metric failure sidecar replay failed: "
                            + "; ".join(sidecar_problems)
                        )
            else:
                raise ArchiveError(f"unknown comparison mode: {mode}")
        if result.get("comparison") != expected_comparison:
            raise ArchiveError(
                f"case {result['path']!r} comparison replay differs"
            )
        expected_outcome = (
            next(
                run["outcome"]
                for run in runs.values()
                if not str(run.get("outcome", "")).startswith("PASS")
            )
            if failed
            else expected_comparison["outcome"]
        )
        if result.get("outcome") != expected_outcome:
            raise ArchiveError(f"case {result['path']!r} outcome differs")

    # Covered cases carry no direct process evidence.  Recompute their outcome
    # from the already replayed terminal target cases so a producer cannot
    # turn an unresolved or failed dependency into PASS_COVERED by rewriting
    # only the report summary and exit code.
    replayed_results = copy.deepcopy(results)
    runner_contract.resolve_coverage_outcomes(
        list(cases.values()), replayed_results
    )
    for actual, replayed in zip(results, replayed_results):
        case = cases[actual["path"]]
        if (
            actual.get("selected")
            and case.get("disposition")
            == runner_contract.COVERED_DISPOSITION
            and actual.get("outcome") != replayed.get("outcome")
        ):
            raise ArchiveError(
                f"covered case {actual['path']!r} outcome differs from targets"
            )

    summary = dict(
        sorted(Counter(result["outcome"] for result in replayed_results).items())
    )
    if report.get("summary") != summary:
        raise ArchiveError("runner summary differs from replayed outcomes")
    selected_results = [
        result for result in replayed_results if result.get("selected")
    ]
    expected_exit = runner_contract.outcome_exit_code(selected_results)
    window = _require_dict(
        _require_dict(report.get("provenance"), "report provenance").get(
            "validation_window"
        ),
        "validation window",
    )
    if window.get("unchanged") is not True:
        expected_exit = runner_contract.EXIT_EVIDENCE_INTEGRITY
    if report.get("exit_code") != expected_exit:
        raise ArchiveError("runner exit code differs from replayed outcomes")


def _compact_snapshot_matches(report_value: Any, receipt_value: Any) -> bool:
    if not isinstance(report_value, dict) or not isinstance(receipt_value, dict):
        return False
    return (
        report_value.get("available") is True
        and report_value.get("sha256") == receipt_value.get("sha256")
        and report_value.get("file_count") == receipt_value.get("file_count")
        and report_value.get("algorithm") == receipt_value.get("algorithm")
    )


def _verify_receipt_ids(receipt: dict[str, Any]) -> None:
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id", None)
    if receipt_id != canonical_sha256(unsigned):
        raise ArchiveError("build receipt ID does not match its payload")
    expected_input = canonical_sha256(
        {
            "schema_version": receipt.get("schema_version"),
            "build_kind": receipt.get("build_kind"),
            "source_start": receipt.get("source_start"),
            "configuration": receipt.get("configuration"),
        }
    )
    if receipt.get("build_input_id") != expected_input:
        raise ArchiveError("build input ID does not match its payload")
    expected_artifacts = canonical_sha256(
        {
            "configuration_files": receipt.get("configuration_files"),
            "toolchain": receipt.get("toolchain"),
            "artifacts": receipt.get("artifacts"),
            "manifests": receipt.get("manifests"),
            **(
                {
                    "immutable_directories": receipt.get(
                        "immutable_directories"
                    )
                }
                if "immutable_directories" in receipt
                else {}
            ),
        }
    )
    if receipt.get("artifact_set_id") != expected_artifacts:
        raise ArchiveError("artifact-set ID does not match its payload")


def _verify_report_receipt_contract(
    report: dict[str, Any], receipt: dict[str, Any], receipt_snapshot: FileSnapshot
) -> None:
    _verify_receipt_ids(receipt)
    if receipt.get("schema_version") != 1 or receipt.get("state") != "complete":
        raise ArchiveError("build receipt is not complete schema version 1 evidence")
    if receipt.get("build_kind") != "cuda-mpi-python-fp32":
        raise ArchiveError("build receipt has the wrong build kind")
    configuration = _require_dict(
        receipt.get("configuration"), "receipt configuration"
    )
    builder = _require_dict(configuration.get("builder"), "receipt builder")
    expected_builder = FIXED_REPO_INPUTS[
        "build-meep-cuda-mpi-python.sh"
    ].as_posix()
    if builder.get("path") != expected_builder:
        raise ArchiveError("build receipt names the wrong authoritative builder")
    provenance = _require_dict(report.get("provenance"), "report.provenance")
    for probe_name in ("build_package_probe", "install_package_probe"):
        probe = provenance.get(probe_name)
        if probe is None:
            continue
        probe = _require_dict(probe, f"report {probe_name}")
        environment = probe.get("environment")
        if (
            not isinstance(environment, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in environment.items()
            )
            or probe.get("environment_keys") != sorted(environment)
            or probe.get("environment_sha256")
            != runner_contract.environment_sha256(environment)
        ):
            raise ArchiveError(f"report {probe_name} environment differs")
    window = _require_dict(
        provenance.get("validation_window"), "report validation window"
    )
    if window.get("unchanged") is not True or window.get("problems") not in ([], None):
        raise ArchiveError("report validation window is not source/build stable")
    for endpoint in ("start", "end"):
        snapshot = _require_dict(window.get(endpoint), f"validation {endpoint}")
        if not _compact_snapshot_matches(
            snapshot.get("source_snapshot"), receipt.get(f"source_{endpoint}")
        ):
            raise ArchiveError(
                f"validation {endpoint} source snapshot differs from receipt"
            )
        record = _require_dict(
            snapshot.get("build_receipt"), f"validation {endpoint} receipt"
        )
        expected = {
            "sha256": receipt_snapshot.sha256,
            "receipt_id": receipt.get("receipt_id"),
            "build_input_id": receipt.get("build_input_id"),
            "artifact_set_id": receipt.get("artifact_set_id"),
        }
        if record.get("available") is not True or any(
            record.get(key) != value for key, value in expected.items()
        ):
            raise ArchiveError(f"validation {endpoint} receipt record differs")
    if not _compact_snapshot_matches(
        provenance.get("source_snapshot"), receipt.get("source_start")
    ):
        raise ArchiveError("report source snapshot differs from build receipt")


def _receipt_source_records(receipt: dict[str, Any]) -> dict[str, FileSnapshot]:
    source = _require_dict(receipt.get("source_end"), "receipt source_end")
    files = source.get("files")
    if not isinstance(files, list):
        raise ArchiveError("receipt source_end lacks a file manifest")
    result: dict[str, FileSnapshot] = {}
    for index, value in enumerate(files):
        record = _require_dict(value, f"receipt source file {index}")
        path = record.get("path")
        if not isinstance(path, str) or path in result or record.get("missing"):
            raise ArchiveError("receipt source file manifest is invalid")
        size = record.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArchiveError("receipt source file size is invalid")
        result[path] = FileSnapshot(
            size, _require_sha256(record.get("sha256"), "receipt source file SHA")
        )
    return result


def _runtime_cache_member(
    relative: pathlib.PurePosixPath, runtime_roots: set[str]
) -> bool:
    if len(relative.parts) < 7:
        return False
    root = pathlib.PurePosixPath(*relative.parts[:5]).as_posix()
    return (
        root in runtime_roots
        and relative.parts[5] in {"home", "cache", "config", "matplotlib"}
    )


def _scan_runner_output(
    root: pathlib.Path,
    allowed_files: set[str],
    runtime_roots: set[str],
) -> list[dict[str, Any]]:
    exclusions: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        relative = pathlib.PurePosixPath(path.relative_to(root).as_posix())
        if stat.S_ISLNK(info.st_mode):
            raise ArchiveError(f"symbolic link in runner output: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ArchiveError(f"non-regular runner output member: {relative}")
        relative_text = relative.as_posix()
        if relative_text in allowed_files:
            continue
        if _runtime_cache_member(relative, runtime_roots):
            current = _snapshot(path)
            exclusions.append(
                {
                    "path": relative_text,
                    "size_bytes": current.size_bytes,
                    "sha256": current.sha256,
                    "reason": "derived per-run HOME/XDG/Matplotlib cache",
                }
            )
            continue
        raise ArchiveError(
            f"runner output contains an unreferenced non-cache file: {relative_text}"
        )
    return exclusions


def _add_copy_spec(
    specs: dict[str, CopySpec],
    member: str,
    source: pathlib.Path,
    expected: FileSnapshot,
    category: str,
) -> None:
    member = _member_name(member)
    if member in specs:
        raise ArchiveError(f"duplicate archive member: {member}")
    specs[member] = CopySpec(member, source, expected, category)


def _copy_verified(spec: CopySpec, root: pathlib.Path) -> None:
    destination = root.joinpath(*pathlib.PurePosixPath(spec.member).parts)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_fd = os.open(spec.source, flags)
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ArchiveError(f"copy source is not regular: {spec.source}")
        digest = hashlib.sha256()
        size = 0
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            ARCHIVE_FILE_MODE,
        )
        try:
            os.fchmod(destination_fd, ARCHIVE_FILE_MODE)
            while True:
                block = os.read(source_fd, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
                view = memoryview(block)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        after = os.fstat(source_fd)
        current = os.stat(spec.source, follow_symlinks=False)
        if (
            _file_fingerprint(before) != _file_fingerprint(after)
            or _file_fingerprint(before) != _file_fingerprint(current)
        ):
            raise ArchiveError(f"source changed while copying: {spec.source}")
        actual = FileSnapshot(size, digest.hexdigest())
        if actual != spec.expected:
            raise ArchiveError(f"copied source differs from expected record: {spec.source}")
    finally:
        os.close(source_fd)


def _payload_records(specs: dict[str, CopySpec]) -> list[dict[str, Any]]:
    return [
        {
            "path": member,
            "size_bytes": spec.expected.size_bytes,
            "sha256": spec.expected.sha256,
            "category": spec.category,
            "mode": ARCHIVE_FILE_MODE,
        }
        for member, spec in sorted(specs.items())
    ]


def _verify_payload_files(root: pathlib.Path, records: list[dict[str, Any]]) -> None:
    expected_paths: set[str] = set()
    for index, record_value in enumerate(records):
        record = _require_dict(record_value, f"archive file record {index}")
        if set(record) != {"path", "size_bytes", "sha256", "category", "mode"}:
            raise ArchiveError("archive file record has unexpected fields")
        if not isinstance(record.get("category"), str) or not record["category"]:
            raise ArchiveError("archive file record category is invalid")
        member = _member_name(record.get("path"))
        if member in expected_paths:
            raise ArchiveError(f"duplicate archive manifest member: {member}")
        expected_paths.add(member)
        size = record.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArchiveError(f"invalid size for archive member {member}")
        expected = FileSnapshot(
            size, _require_sha256(record.get("sha256"), f"{member} SHA")
        )
        path = root.joinpath(*pathlib.PurePosixPath(member).parts)
        _assert_path_chain_without_symlinks(path, final_kind="file")
        if record.get("mode") != ARCHIVE_FILE_MODE or stat.S_IMODE(
            os.lstat(path).st_mode
        ) != ARCHIVE_FILE_MODE:
            raise ArchiveError(f"archive payload member mode differs: {member}")
        if _snapshot(path) != expected:
            raise ArchiveError(f"archive payload member changed: {member}")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise ArchiveError(f"archive contains a symbolic link: {relative}")
        if stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1:
                raise ArchiveError(
                    f"archive contains a hard-linked file: {relative}"
                )
            if stat.S_IMODE(info.st_mode) != ARCHIVE_FILE_MODE:
                raise ArchiveError(f"archive file mode differs: {relative}")
            actual_files.add(relative)
        elif stat.S_ISDIR(info.st_mode):
            if stat.S_IMODE(info.st_mode) != ARCHIVE_DIRECTORY_MODE:
                raise ArchiveError(f"archive directory mode differs: {relative}")
            actual_directories.add(relative)
        else:
            raise ArchiveError(f"archive contains a non-regular member: {relative}")
    allowed_control = {MANIFEST_NAME, COMPLETE_NAME}
    unexpected_files = actual_files - expected_paths - allowed_control
    if unexpected_files:
        raise ArchiveError(
            "archive contains unbound files: " + ", ".join(sorted(unexpected_files))
        )
    expected_directories: set[str] = set()
    for member in expected_paths | (actual_files & allowed_control):
        parent = pathlib.PurePosixPath(member).parent
        while parent != pathlib.PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    extra_directories = actual_directories - expected_directories
    if extra_directories:
        raise ArchiveError(
            "archive contains unbound directories: "
            + ", ".join(sorted(extra_directories))
        )


def _fsync_directories(root: pathlib.Path) -> None:
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _normalize_archive_modes(root: pathlib.Path, *, include_root: bool) -> None:
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        if stat.S_ISDIR(info.st_mode):
            os.chmod(path, ARCHIVE_DIRECTORY_MODE)
        elif stat.S_ISREG(info.st_mode):
            os.chmod(path, ARCHIVE_FILE_MODE)
        else:
            raise ArchiveError(f"archive contains a non-regular member: {path}")
    if include_root:
        os.chmod(root, ARCHIVE_DIRECTORY_MODE)


def _is_within(candidate: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _fsync_directory(path: pathlib.Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_noreplace(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Atomically publish without replacing a concurrently-created target."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise ArchiveError(
            "atomic no-replace publication is unavailable on this platform"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(_absolute(source)),
        -100,
        os.fsencode(_absolute(destination)),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise ArchiveError(
            f"publication destination already exists: {destination}"
        )
    if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise ArchiveError(
            "filesystem does not support atomic no-replace publication"
        )
    raise ArchiveError(
        f"atomic no-replace publication failed: {os.strerror(error)}"
    )


def _verify_copy_sources(specs: dict[str, CopySpec]) -> None:
    for spec in sorted(specs.values(), key=lambda item: item.member):
        _assert_path_chain_without_symlinks(spec.source, final_kind="file")
        if _snapshot(spec.source) != spec.expected:
            raise ArchiveError(
                f"archive source changed before publication: {spec.source}"
            )


def _expected_source_contract(
    *,
    report: dict[str, Any],
    report_snapshot: FileSnapshot,
    report_markdown_snapshot: FileSnapshot,
    runner_complete_snapshot: FileSnapshot,
    receipt: dict[str, Any],
    receipt_snapshot: FileSnapshot,
    manifest_snapshot: FileSnapshot,
    references: list[tuple[pathlib.PurePosixPath, FileSnapshot, str]],
    executed_sources: set[str],
    original_roots: set[str],
) -> dict[str, Any]:
    return {
        "runner_report_schema_version": report.get("schema_version"),
        "runner_exit_code": report.get("exit_code"),
        "report_sha256": report_snapshot.sha256,
        "report_markdown_sha256": report_markdown_snapshot.sha256,
        "runner_complete_sha256": runner_complete_snapshot.sha256,
        "build_receipt_sha256": receipt_snapshot.sha256,
        "build_receipt_id": receipt.get("receipt_id"),
        "build_input_id": receipt.get("build_input_id"),
        "artifact_set_id": receipt.get("artifact_set_id"),
        "source_snapshot_sha256": _require_dict(
            receipt.get("source_end"), "receipt source_end"
        ).get("sha256"),
        "manifest_sha256": manifest_snapshot.sha256,
        "raw_evidence_file_count": len(references),
        "executed_case_source_count": len(executed_sources),
        "executed_case_sources_sha256": canonical_sha256(
            sorted(executed_sources)
        ),
        "fixed_input_paths_sha256": canonical_sha256(
            sorted(path.as_posix() for path in FIXED_REPO_INPUTS.values())
        ),
        "recorded_output_root_count": len(original_roots),
        "recorded_output_root_sha256": (
            _sha256_bytes(next(iter(original_roots)).encode("utf-8"))
            if original_roots
            else None
        ),
    }


def _expected_payload_categories(
    references: list[tuple[pathlib.PurePosixPath, FileSnapshot, str]],
    executed_sources: set[str],
) -> dict[str, str]:
    expected = {
        (RUNNER_ROOT / "report.json").as_posix(): "runner-control",
        (RUNNER_ROOT / "COMPLETE").as_posix(): "runner-control",
        (RUNNER_ROOT / "report.md").as_posix(): "runner-report",
        (INPUT_ROOT / "build-provenance.json").as_posix(): "build-receipt",
    }
    for relative, _, _ in references:
        expected[(RUNNER_ROOT / relative).as_posix()] = "runner-raw-evidence"
    for archive_name in FIXED_REPO_INPUTS:
        expected[(INPUT_ROOT / archive_name).as_posix()] = "validation-input"
    for source_text in executed_sources:
        expected[
            (SOURCE_ROOT / pathlib.PurePosixPath(source_text)).as_posix()
        ] = "executed-case-source"
    return expected


def deterministic_tar_metadata() -> dict[str, Any]:
    argv = [
        "{trusted_tar}",
        "--sort=name",
        "--format=posix",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "--mode=u+rwX,go+rX,go-w",
        "--pax-option=delete=atime,delete=ctime",
        "--no-xattrs",
        "--no-acls",
        "--no-selinux",
        "--hard-dereference",
        "-C",
        "{archive_dir}",
        "-cf",
        "-",
        ".",
    ]
    return {
        "schema_version": 1,
        "environment": {
            "LC_ALL": "C",
            "TZ": "UTC",
            "TAR_OPTIONS": "unset",
        },
        "command_argv_template": argv,
        "command_template": (
            "env -u TAR_OPTIONS LC_ALL=C TZ=UTC " + shlex.join(argv)
        ),
        "checksum_argv_template": ["sha256sum", "{tar_path}"],
        "archive_created_by_sealer": "optional-via---tar",
        "trusted_tar_policy": "fixed system/Python-prefix absolute path plus full replay",
        "output_publication": (
            "O_TMPFILE stdout fd, fsync, full tar replay, "
            "linkat(AT_EMPTY_PATH) no-replace"
        ),
    }


def _verify_tar_payload(
    tar_path: pathlib.Path, archive_root: pathlib.Path
) -> None:
    expected: dict[str, tuple[str, int, FileSnapshot | None]] = {
        "": ("directory", ARCHIVE_DIRECTORY_MODE, None)
    }
    for path in sorted(archive_root.rglob("*")):
        relative = path.relative_to(archive_root).as_posix()
        info = os.lstat(path)
        if stat.S_ISDIR(info.st_mode):
            expected[relative] = (
                "directory",
                ARCHIVE_DIRECTORY_MODE,
                None,
            )
        elif stat.S_ISREG(info.st_mode):
            expected[relative] = (
                "file",
                ARCHIVE_FILE_MODE,
                _snapshot(path),
            )
        else:
            raise ArchiveError(f"archive contains a non-regular member: {relative}")

    actual: dict[str, tuple[str, int, FileSnapshot | None]] = {}
    try:
        with tarfile.open(tar_path, mode="r:") as archive:
            for member in archive:
                raw_name = member.name
                while raw_name.startswith("./"):
                    raw_name = raw_name[2:]
                name = "" if raw_name in ("", ".") else _member_name(raw_name)
                if name in actual:
                    raise ArchiveError(f"deterministic tar repeats member {name!r}")
                if member.uid != 0 or member.gid != 0 or member.mtime != 0:
                    raise ArchiveError("deterministic tar ownership/time differs")
                mode = stat.S_IMODE(member.mode)
                if member.isdir():
                    actual[name] = ("directory", mode, None)
                elif member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ArchiveError(f"cannot read tar member {name!r}")
                    digest = hashlib.sha256()
                    size = 0
                    for block in iter(lambda: extracted.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                    actual[name] = (
                        "file",
                        mode,
                        FileSnapshot(size, digest.hexdigest()),
                    )
                else:
                    raise ArchiveError(
                        f"deterministic tar has unsafe member type: {name!r}"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveError(f"cannot parse deterministic tar: {exc}") from exc
    if actual != expected:
        raise ArchiveError("deterministic tar payload differs from archive tree")


def _archive_tree_fingerprint(root: pathlib.Path) -> str:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise ArchiveError(f"archive contains a symbolic link: {relative}")
        if stat.S_ISDIR(info.st_mode):
            records.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "mode": stat.S_IMODE(info.st_mode),
                }
            )
        elif stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1:
                raise ArchiveError(
                    f"archive contains a hard-linked file: {relative}"
                )
            snapshot = _snapshot(path)
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size_bytes": snapshot.size_bytes,
                    "sha256": snapshot.sha256,
                    "mode": stat.S_IMODE(info.st_mode),
                }
            )
        else:
            raise ArchiveError(f"archive contains a non-regular member: {relative}")
    return canonical_sha256(records)


def _deterministic_tar_output_limit(root: pathlib.Path) -> int:
    """Bound tar output from the already-validated source tree size.

    POSIX tar pads file payloads and may emit per-member extended metadata.
    A 64 KiB allowance per member is deliberately much larger than those
    headers (including filesystem-length path records), while the trailer
    allowance covers GNU tar's final record padding.  The bound scales with
    legitimate evidence instead of silently imposing the collector's 64 MiB
    default on production validation archives.
    """
    payload_bytes = 0
    member_count = 1  # The archive root directory is itself a tar member.
    for path in root.rglob("*"):
        info = os.lstat(path)
        member_count += 1
        if stat.S_ISREG(info.st_mode):
            payload_bytes += info.st_size
        elif not stat.S_ISDIR(info.st_mode):
            raise ArchiveError(
                "archive contains a non-file member while sizing tar output"
            )
    calculated = (
        payload_bytes
        + member_count * TAR_MEMBER_OVERHEAD_LIMIT_BYTES
        + TAR_TRAILER_OVERHEAD_LIMIT_BYTES
    )
    return max(MINIMUM_TAR_OUTPUT_LIMIT_BYTES, calculated)


def create_deterministic_tar(
    archive_dir: pathlib.Path, tar_path: pathlib.Path
) -> dict[str, Any]:
    archive_root = _assert_path_chain_without_symlinks(
        archive_dir, final_kind="directory"
    )
    destination = _assert_path_chain_without_symlinks(
        tar_path, final_kind="missing"
    )
    _assert_path_chain_without_symlinks(
        destination.parent, final_kind="directory"
    )
    if _is_within(destination, archive_root):
        raise ArchiveError("tar destination must not be inside the archive")
    before = _archive_tree_fingerprint(archive_root)
    output_limit_bytes = _deterministic_tar_output_limit(archive_root)
    metadata = deterministic_tar_metadata()
    tar_executable = trusted_tar()
    argv = [
        value.replace("{archive_dir}", str(archive_root)).replace(
            "{tar_path}", str(destination)
        ).replace("{trusted_tar}", str(tar_executable))
        for value in metadata["command_argv_template"]
    ]
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
    }
    flags = (
        os.O_RDWR
        | os.O_TMPFILE
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        output_fd = os.open(destination.parent, flags, 0o644)
    except OSError as exc:
        raise ArchiveError(
            "destination filesystem does not support safe unnamed tar staging"
        ) from exc
    try:
        os.fchmod(output_fd, 0o644)
        result = bounded_command(
            argv,
            cwd=archive_root,
            environment=environment,
            timeout_seconds=3600,
            stdout_limit=output_limit_bytes,
            stderr_limit=8 * 1024 * 1024,
            stdout_fd=output_fd,
        )
        if (
            result["exit_code"] != 0
            or result["error"] is not None
            or result["timeout"]
            or result["output_limit"] is not None
        ):
            raise ArchiveError(
                "deterministic tar failed: "
                + result["stderr"].decode("utf-8", errors="replace").strip()
            )
        os.fsync(output_fd)
        staged_fd_path = pathlib.Path(f"/proc/self/fd/{output_fd}")
        _verify_tar_payload(staged_fd_path, archive_root)
        after = _archive_tree_fingerprint(archive_root)
        if after != before:
            raise ArchiveError("archive changed while deterministic tar was created")
        try:
            linkat = ctypes.CDLL(None, use_errno=True).linkat
        except AttributeError as exc:
            raise ArchiveError("safe tar publication is unavailable") from exc
        linkat.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        )
        linkat.restype = ctypes.c_int
        if linkat(output_fd, b"", -100, os.fsencode(destination), 0x1000) != 0:
            error = ctypes.get_errno()
            if error in (errno.EEXIST, errno.ENOTEMPTY):
                raise ArchiveError(
                    f"publication destination already exists: {destination}"
                )
            raise ArchiveError(
                f"safe tar publication failed: {os.strerror(error)}"
            )
        _fsync_directory(destination.parent)
        snapshot = _snapshot(destination)
        return {
            "path": str(destination),
            "size_bytes": snapshot.size_bytes,
            "sha256": snapshot.sha256,
            "source_tree_sha256": before,
            "output_limit_bytes": output_limit_bytes,
        }
    finally:
        os.close(output_fd)


def seal_archive(
    input_dir: pathlib.Path,
    archive_dir: pathlib.Path,
    repo: pathlib.Path,
    build_receipt: pathlib.Path,
) -> dict[str, Any]:
    input_root = _assert_path_chain_without_symlinks(
        input_dir, final_kind="directory"
    )
    repo_root = _assert_path_chain_without_symlinks(repo, final_kind="directory")
    receipt_path = _assert_path_chain_without_symlinks(
        build_receipt, final_kind="file"
    )
    archive_root = _assert_path_chain_without_symlinks(
        archive_dir, final_kind="missing"
    )
    _assert_path_chain_without_symlinks(archive_root.parent, final_kind="directory")
    if _is_within(archive_root, input_root):
        raise ArchiveError("archive destination must not be inside runner output")
    if _is_within(archive_root, repo_root):
        raise ArchiveError("archive destination must not be inside the repository")

    report_path = _assert_path_chain_without_symlinks(
        input_root / "report.json", final_kind="file"
    )
    runner_complete_path = _assert_path_chain_without_symlinks(
        input_root / "COMPLETE", final_kind="file"
    )
    report_md_path = _assert_path_chain_without_symlinks(
        input_root / "report.md", final_kind="file"
    )
    report, report_bytes = _load_json_file(report_path, "runner report")
    runner_complete, runner_complete_bytes = _load_json_file(
        runner_complete_path, "runner COMPLETE"
    )
    _validate_runner_control(report, runner_complete)
    report_digest = _sha256_bytes(report_bytes)
    report_snapshot = FileSnapshot(len(report_bytes), report_digest)
    report_markdown_snapshot = _snapshot(report_md_path)
    runner_complete_snapshot = FileSnapshot(
        len(runner_complete_bytes), _sha256_bytes(runner_complete_bytes)
    )
    if runner_complete.get("report_sha256") != report_digest:
        raise ArchiveError("runner COMPLETE does not bind report.json")
    if runner_complete.get("exit_code") != report.get("exit_code"):
        raise ArchiveError("runner COMPLETE/report exit codes differ")
    if (
        runner_complete.get("report_size_bytes") != report_snapshot.size_bytes
        or runner_complete.get("markdown_sha256")
        != report_markdown_snapshot.sha256
        or runner_complete.get("markdown_size_bytes")
        != report_markdown_snapshot.size_bytes
    ):
        raise ArchiveError("runner COMPLETE does not bind report sizes/Markdown")

    receipt_snapshot = _snapshot(receipt_path)
    receipt, _ = _load_json_file(receipt_path, "build receipt")
    hash_cache = StatHashCache()
    try:
        verified_receipt = verify_build_receipt(
            receipt_path, repo_root, hash_cache=hash_cache
        )
    except (OSError, RuntimeError, ProvenanceError) as exc:
        raise ArchiveError(f"build receipt verification failed: {exc}") from exc
    if verified_receipt != receipt:
        raise ArchiveError("verified build receipt payload changed")
    _verify_report_receipt_contract(report, receipt, receipt_snapshot)

    (
        references,
        executed_sources,
        original_roots,
        runtime_roots,
    ) = _collect_report_evidence(report, input_root)
    allowed_output_files = {"report.json", "report.md", "COMPLETE"}
    allowed_output_files.update(relative.as_posix() for relative, _, _ in references)
    exclusions = _scan_runner_output(
        input_root, allowed_output_files, runtime_roots
    )

    provenance = _require_dict(report.get("provenance"), "report.provenance")
    manifest_path = repo_root.joinpath(*FIXED_REPO_INPUTS["manifest.json"].parts)
    _assert_path_chain_without_symlinks(manifest_path, final_kind="file")
    manifest_snapshot = _snapshot(manifest_path)
    validation_manifest, _ = _load_json_file(
        manifest_path, "validation manifest"
    )
    if provenance.get("manifest_sha256") != manifest_snapshot.sha256:
        raise ArchiveError("current validation manifest differs from the report")
    source_records = _receipt_source_records(receipt)
    _replay_report_semantics(
        report,
        validation_manifest,
        input_root,
        receipt,
        source_records,
        bind_environment_to_evidence_root=True,
    )
    builder_record = _require_dict(
        _require_dict(receipt.get("configuration"), "receipt configuration").get(
            "builder"
        ),
        "receipt builder",
    )
    expected_builder_path = FIXED_REPO_INPUTS[
        "build-meep-cuda-mpi-python.sh"
    ].as_posix()
    if builder_record.get("path") != expected_builder_path:
        raise ArchiveError("receipt was not produced by the authoritative builder")

    specs: dict[str, CopySpec] = {}
    _add_copy_spec(
        specs,
        (RUNNER_ROOT / "report.json").as_posix(),
        report_path,
        report_snapshot,
        "runner-control",
    )
    _add_copy_spec(
        specs,
        (RUNNER_ROOT / "COMPLETE").as_posix(),
        runner_complete_path,
        runner_complete_snapshot,
        "runner-control",
    )
    _add_copy_spec(
        specs,
        (RUNNER_ROOT / "report.md").as_posix(),
        report_md_path,
        report_markdown_snapshot,
        "runner-report",
    )
    evidence_relocation: list[EvidenceReference] = []
    for relative, expected, recorded_path in references:
        member = (RUNNER_ROOT / relative).as_posix()
        source = input_root.joinpath(*relative.parts)
        _add_copy_spec(specs, member, source, expected, "runner-raw-evidence")
        evidence_relocation.append(
            EvidenceReference(
                source_relative_path=relative.as_posix(),
                archive_path=member,
                recorded_path_sha256=_sha256_bytes(recorded_path.encode("utf-8")),
                size_bytes=expected.size_bytes,
                sha256=expected.sha256,
            )
        )
    for archive_name, relative in FIXED_REPO_INPUTS.items():
        source = repo_root.joinpath(*relative.parts)
        _assert_path_chain_without_symlinks(source, final_kind="file")
        current = _snapshot(source)
        receipt_record = source_records.get(relative.as_posix())
        if receipt_record != current:
            raise ArchiveError(
                f"required input is absent from the receipt source closure: {relative}"
            )
        _add_copy_spec(
            specs,
            (INPUT_ROOT / archive_name).as_posix(),
            source,
            current,
            "validation-input",
        )
    _add_copy_spec(
        specs,
        (INPUT_ROOT / "build-provenance.json").as_posix(),
        receipt_path,
        receipt_snapshot,
        "build-receipt",
    )
    for source_text in sorted(executed_sources):
        relative = _repo_relative(source_text, "executed source")
        source = repo_root.joinpath(*relative.parts)
        _assert_path_chain_without_symlinks(source, final_kind="file")
        current = _snapshot(source)
        if source_records.get(relative.as_posix()) != current:
            raise ArchiveError(
                f"executed source differs from receipt closure: {relative}"
            )
        _add_copy_spec(
            specs,
            (SOURCE_ROOT / relative).as_posix(),
            source,
            current,
            "executed-case-source",
        )

    payload_records = _payload_records(specs)
    source_contract = _expected_source_contract(
        report=report,
        report_snapshot=report_snapshot,
        report_markdown_snapshot=report_markdown_snapshot,
        runner_complete_snapshot=runner_complete_snapshot,
        receipt=receipt,
        receipt_snapshot=receipt_snapshot,
        manifest_snapshot=manifest_snapshot,
        references=references,
        executed_sources=executed_sources,
        original_roots=original_roots,
    )
    archive_manifest = {
        "schema_version": SCHEMA_VERSION,
        "archive_kind": ARCHIVE_KIND,
        "source_contract": source_contract,
        "relocation": {
            "policy": "validated-runs-suffix-rebound-under-archive-root-v1",
            "evidence": [item.__dict__ for item in evidence_relocation],
        },
        "runtime_cache_exclusion": {
            "policy": "exclude-owned-run-runtime-tree-v2",
            "runtime_root_count": len(runtime_roots),
            "runtime_roots_sha256": canonical_sha256(sorted(runtime_roots)),
            "file_count": len(exclusions),
            "records_sha256": canonical_sha256(exclusions),
            "files": exclusions,
        },
        "publication_safety": {
            "policy": "private-sibling-stage-verify-fsync-atomic-noreplace-v2",
            "source_rescan_before_publication": True,
            "repository_destination_forbidden": True,
            "concurrent_ancestor_mutation_forbidden": True,
        },
        "files": payload_records,
        "file_count": len(payload_records),
        "payload_sha256": canonical_sha256(payload_records),
        "deterministic_tar": deterministic_tar_metadata(),
    }
    staging_root = archive_root.parent / (
        f".{archive_root.name}.staging-{uuid.uuid4().hex}"
    )
    _assert_path_chain_without_symlinks(staging_root, final_kind="missing")
    try:
        staging_root.mkdir(mode=0o700)
        for spec in sorted(specs.values(), key=lambda item: item.member):
            _copy_verified(spec, staging_root)
        _normalize_archive_modes(staging_root, include_root=False)
        _verify_payload_files(staging_root, payload_records)
        atomic_write_json(staging_root / MANIFEST_NAME, archive_manifest)
        os.chmod(staging_root / MANIFEST_NAME, ARCHIVE_FILE_MODE)
        manifest_bytes = _read_stable_file(staging_root / MANIFEST_NAME)
        _verify_payload_files(staging_root, payload_records)
        archive_complete = {
            "schema_version": SCHEMA_VERSION,
            "state": "COMPLETE",
            "archive_kind": ARCHIVE_KIND,
            "archive_manifest_path": MANIFEST_NAME,
            "archive_manifest_sha256": _sha256_bytes(manifest_bytes),
            "report_path": (RUNNER_ROOT / "report.json").as_posix(),
            "report_sha256": report_digest,
            "runner_complete_path": (RUNNER_ROOT / "COMPLETE").as_posix(),
            "runner_complete_sha256": runner_complete_snapshot.sha256,
            "build_receipt_id": receipt.get("receipt_id"),
            "payload_file_count": len(payload_records),
            "payload_sha256": canonical_sha256(payload_records),
        }
        atomic_write_json(staging_root / COMPLETE_NAME, archive_complete)
        os.chmod(staging_root / COMPLETE_NAME, ARCHIVE_FILE_MODE)
        os.chmod(staging_root, ARCHIVE_DIRECTORY_MODE)
        _fsync_directories(staging_root)
        verify_archive(staging_root)

        # Recheck every external source, the complete repository receipt, and
        # the exact owned-runtime exclusion list immediately before publication.
        _verify_copy_sources(specs)
        try:
            current_receipt = verify_build_receipt(
                receipt_path, repo_root, hash_cache=StatHashCache()
            )
        except (OSError, RuntimeError, ProvenanceError) as exc:
            raise ArchiveError(
                f"build receipt changed before archive publication: {exc}"
            ) from exc
        if current_receipt != receipt:
            raise ArchiveError("build receipt changed before archive publication")
        current_exclusions = _scan_runner_output(
            input_root, allowed_output_files, runtime_roots
        )
        if current_exclusions != exclusions:
            raise ArchiveError("runner runtime cache changed before publication")
        _verify_report_receipt_contract(report, receipt, receipt_snapshot)

        _fsync_directories(staging_root)
        _publish_noreplace(staging_root, archive_root)
        _fsync_directory(archive_root.parent)
        verify_archive(archive_root)
        return archive_complete
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def verify_archive(
    archive_dir: pathlib.Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_receipt_id: str | None = None,
) -> dict[str, Any]:
    if expected_receipt_id is not None and expected_manifest_sha256 is None:
        raise ArchiveError(
            "expected receipt ID alone authenticates only the build; "
            "an expected archive manifest SHA is also required for results"
        )
    root = _assert_path_chain_without_symlinks(
        archive_dir, final_kind="directory"
    )
    if stat.S_IMODE(os.lstat(root).st_mode) != ARCHIVE_DIRECTORY_MODE:
        raise ArchiveError("archive root directory mode differs")
    complete_path = _assert_path_chain_without_symlinks(
        root / COMPLETE_NAME, final_kind="file"
    )
    manifest_path = _assert_path_chain_without_symlinks(
        root / MANIFEST_NAME, final_kind="file"
    )
    complete, _ = _load_json_file(complete_path, "ARCHIVE_COMPLETE")
    manifest, manifest_bytes = _load_json_file(
        manifest_path, "ARCHIVE_MANIFEST.json"
    )
    if (
        complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("state") != "COMPLETE"
        or complete.get("archive_kind") != ARCHIVE_KIND
    ):
        raise ArchiveError("ARCHIVE_COMPLETE contract is invalid")
    if set(complete) != {
        "schema_version",
        "state",
        "archive_kind",
        "archive_manifest_path",
        "archive_manifest_sha256",
        "report_path",
        "report_sha256",
        "runner_complete_path",
        "runner_complete_sha256",
        "build_receipt_id",
        "payload_file_count",
        "payload_sha256",
    }:
        raise ArchiveError("ARCHIVE_COMPLETE has unexpected fields")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("archive_kind") != ARCHIVE_KIND
    ):
        raise ArchiveError("archive manifest contract is invalid")
    if set(manifest) != {
        "schema_version",
        "archive_kind",
        "source_contract",
        "relocation",
        "runtime_cache_exclusion",
        "publication_safety",
        "files",
        "file_count",
        "payload_sha256",
        "deterministic_tar",
    }:
        raise ArchiveError("archive manifest has unexpected fields")
    if complete.get("archive_manifest_path") != MANIFEST_NAME:
        raise ArchiveError("ARCHIVE_COMPLETE names the wrong manifest")
    if complete.get("archive_manifest_sha256") != _sha256_bytes(manifest_bytes):
        raise ArchiveError("ARCHIVE_COMPLETE does not bind the archive manifest")
    if (
        expected_manifest_sha256 is not None
        and _require_sha256(
            expected_manifest_sha256, "expected archive manifest SHA"
        )
        != _sha256_bytes(manifest_bytes)
    ):
        raise ArchiveError("archive manifest differs from trusted expected SHA")
    records = manifest.get("files")
    if not isinstance(records, list) or records != sorted(
        records, key=lambda item: item.get("path", "") if isinstance(item, dict) else ""
    ):
        raise ArchiveError("archive file records must be a sorted array")
    if manifest.get("file_count") != len(records):
        raise ArchiveError("archive manifest file count differs")
    payload_digest = canonical_sha256(records)
    if manifest.get("payload_sha256") != payload_digest:
        raise ArchiveError("archive manifest payload digest differs")
    if (
        complete.get("payload_file_count") != len(records)
        or complete.get("payload_sha256") != payload_digest
    ):
        raise ArchiveError("ARCHIVE_COMPLETE payload contract differs")
    _verify_payload_files(root, records)

    source_contract = _require_dict(manifest.get("source_contract"), "source_contract")
    report_member = _member_name(complete.get("report_path"))
    if report_member != (RUNNER_ROOT / "report.json").as_posix():
        raise ArchiveError("ARCHIVE_COMPLETE report path is not fixed")
    report_path = root.joinpath(*pathlib.PurePosixPath(report_member).parts)
    report, report_bytes = _load_json_file(report_path, "archived runner report")
    if (
        complete.get("report_sha256") != _sha256_bytes(report_bytes)
        or source_contract.get("report_sha256") != _sha256_bytes(report_bytes)
    ):
        raise ArchiveError("archived runner report digest differs")
    runner_complete_member = _member_name(complete.get("runner_complete_path"))
    if runner_complete_member != (RUNNER_ROOT / "COMPLETE").as_posix():
        raise ArchiveError("ARCHIVE_COMPLETE runner COMPLETE path is not fixed")
    runner_complete_path = root.joinpath(
        *pathlib.PurePosixPath(runner_complete_member).parts
    )
    runner_complete, runner_complete_bytes = _load_json_file(
        runner_complete_path, "archived runner COMPLETE"
    )
    _validate_runner_control(report, runner_complete)
    report_markdown_path = root / RUNNER_ROOT / "report.md"
    _assert_path_chain_without_symlinks(
        report_markdown_path, final_kind="file"
    )
    report_markdown_snapshot = _snapshot(report_markdown_path)
    if (
        complete.get("runner_complete_sha256")
        != _sha256_bytes(runner_complete_bytes)
        or source_contract.get("runner_complete_sha256")
        != _sha256_bytes(runner_complete_bytes)
        or runner_complete.get("report_sha256") != _sha256_bytes(report_bytes)
        or runner_complete.get("exit_code") != report.get("exit_code")
        or runner_complete.get("schema_version") != report.get("schema_version")
    ):
        raise ArchiveError("archived runner COMPLETE chain differs")
    if (
        runner_complete.get("report_size_bytes") != len(report_bytes)
        or runner_complete.get("markdown_sha256")
        != report_markdown_snapshot.sha256
        or runner_complete.get("markdown_size_bytes")
        != report_markdown_snapshot.size_bytes
    ):
        raise ArchiveError("archived runner Markdown/size chain differs")

    (
        references,
        executed_sources,
        original_roots,
        runtime_roots,
    ) = _collect_report_evidence(report, None)
    expected_original_root_sha = (
        _sha256_bytes(next(iter(original_roots)).encode("utf-8"))
        if original_roots
        else None
    )
    if (
        source_contract.get("runner_report_schema_version")
        != report.get("schema_version")
        or source_contract.get("runner_exit_code") != report.get("exit_code")
        or source_contract.get("raw_evidence_file_count") != len(references)
        or source_contract.get("executed_case_source_count")
        != len(executed_sources)
        or source_contract.get("recorded_output_root_count")
        != len(original_roots)
        or source_contract.get("recorded_output_root_sha256")
        != expected_original_root_sha
    ):
        raise ArchiveError("archive source/run count contract differs")
    relocation = _require_dict(manifest.get("relocation"), "relocation")
    if (
        relocation.get("policy")
        != "validated-runs-suffix-rebound-under-archive-root-v1"
    ):
        raise ArchiveError("relocation policy differs")
    relocation_records = relocation.get("evidence")
    if not isinstance(relocation_records, list):
        raise ArchiveError("relocation evidence must be an array")
    expected_relocation = [
        EvidenceReference(
            source_relative_path=relative.as_posix(),
            archive_path=(RUNNER_ROOT / relative).as_posix(),
            recorded_path_sha256=_sha256_bytes(recorded_path.encode("utf-8")),
            size_bytes=expected.size_bytes,
            sha256=expected.sha256,
        ).__dict__
        for relative, expected, recorded_path in references
    ]
    if relocation_records != expected_relocation:
        raise ArchiveError("relocation map differs from report evidence records")
    record_by_path = {
        record["path"]: record for record in records if isinstance(record, dict)
    }
    expected_categories = _expected_payload_categories(
        references, executed_sources
    )
    actual_categories = {
        path: record.get("category") for path, record in record_by_path.items()
    }
    if actual_categories != expected_categories:
        raise ArchiveError("archive payload categories or members differ")
    for item in expected_relocation:
        record = record_by_path.get(item["archive_path"])
        if (
            not isinstance(record, dict)
            or record.get("size_bytes") != item["size_bytes"]
            or record.get("sha256") != item["sha256"]
        ):
            raise ArchiveError("relocated raw evidence differs from its report record")

    receipt_member = (INPUT_ROOT / "build-provenance.json").as_posix()
    receipt_path = root.joinpath(*pathlib.PurePosixPath(receipt_member).parts)
    receipt, receipt_bytes = _load_json_file(receipt_path, "archived build receipt")
    _verify_receipt_ids(receipt)
    receipt_snapshot = FileSnapshot(len(receipt_bytes), _sha256_bytes(receipt_bytes))
    _verify_report_receipt_contract(report, receipt, receipt_snapshot)
    if (
        expected_receipt_id is not None
        and _require_sha256(expected_receipt_id, "expected receipt ID")
        != receipt.get("receipt_id")
    ):
        raise ArchiveError("archive receipt differs from trusted expected ID")
    if (
        source_contract.get("build_receipt_sha256") != receipt_snapshot.sha256
        or source_contract.get("build_receipt_id") != receipt.get("receipt_id")
        or source_contract.get("build_input_id") != receipt.get("build_input_id")
        or source_contract.get("artifact_set_id") != receipt.get("artifact_set_id")
        or source_contract.get("source_snapshot_sha256")
        != _require_dict(receipt.get("source_end"), "receipt source_end").get(
            "sha256"
        )
        or complete.get("build_receipt_id") != receipt.get("receipt_id")
    ):
        raise ArchiveError("archived build receipt binding differs")

    receipt_sources = _receipt_source_records(receipt)
    for archive_name, relative in FIXED_REPO_INPUTS.items():
        member = (INPUT_ROOT / archive_name).as_posix()
        record = record_by_path.get(member)
        expected = receipt_sources.get(relative.as_posix())
        if (
            not isinstance(record, dict)
            or expected is None
            or record.get("size_bytes") != expected.size_bytes
            or record.get("sha256") != expected.sha256
        ):
            raise ArchiveError(f"archived validation input differs: {relative}")
    archived_manifest = record_by_path.get((INPUT_ROOT / "manifest.json").as_posix())
    report_provenance = _require_dict(report.get("provenance"), "report provenance")
    if (
        not isinstance(archived_manifest, dict)
        or source_contract.get("manifest_sha256")
        != archived_manifest.get("sha256")
        or report_provenance.get("manifest_sha256")
        != archived_manifest.get("sha256")
    ):
        raise ArchiveError("archived validation manifest binding differs")
    archived_manifest_snapshot = FileSnapshot(
        archived_manifest["size_bytes"], archived_manifest["sha256"]
    )
    archived_manifest_path = root / INPUT_ROOT / "manifest.json"
    validation_manifest, _ = _load_json_file(
        archived_manifest_path, "archived validation manifest"
    )
    _replay_report_semantics(
        report,
        validation_manifest,
        root / RUNNER_ROOT,
        receipt,
        receipt_sources,
        bind_environment_to_evidence_root=False,
    )
    expected_source_contract = _expected_source_contract(
        report=report,
        report_snapshot=FileSnapshot(
            len(report_bytes), _sha256_bytes(report_bytes)
        ),
        report_markdown_snapshot=report_markdown_snapshot,
        runner_complete_snapshot=FileSnapshot(
            len(runner_complete_bytes), _sha256_bytes(runner_complete_bytes)
        ),
        receipt=receipt,
        receipt_snapshot=receipt_snapshot,
        manifest_snapshot=archived_manifest_snapshot,
        references=references,
        executed_sources=executed_sources,
        original_roots=original_roots,
    )
    if source_contract != expected_source_contract:
        raise ArchiveError("archive source contract differs from recomputed contract")
    for source_text in sorted(executed_sources):
        member = (SOURCE_ROOT / pathlib.PurePosixPath(source_text)).as_posix()
        record = record_by_path.get(member)
        expected = receipt_sources.get(source_text)
        if (
            not isinstance(record, dict)
            or expected is None
            or record.get("size_bytes") != expected.size_bytes
            or record.get("sha256") != expected.sha256
        ):
            raise ArchiveError(f"archived executed source differs: {source_text}")

    runtime_policy = _require_dict(
        manifest.get("runtime_cache_exclusion"), "runtime cache exclusion"
    )
    if set(runtime_policy) != {
        "policy",
        "runtime_root_count",
        "runtime_roots_sha256",
        "file_count",
        "records_sha256",
        "files",
    }:
        raise ArchiveError("runtime cache exclusion has unexpected fields")
    if runtime_policy.get("policy") != "exclude-owned-run-runtime-tree-v2":
        raise ArchiveError("runtime cache exclusion policy differs")
    if (
        runtime_policy.get("runtime_root_count") != len(runtime_roots)
        or runtime_policy.get("runtime_roots_sha256")
        != canonical_sha256(sorted(runtime_roots))
    ):
        raise ArchiveError("runtime cache ownership roots differ")
    excluded_files = runtime_policy.get("files")
    if not isinstance(excluded_files, list) or runtime_policy.get(
        "file_count"
    ) != len(excluded_files):
        raise ArchiveError("runtime cache exclusion file count differs")
    excluded_paths: set[str] = set()
    for index, value in enumerate(excluded_files):
        item = _require_dict(value, f"runtime exclusion {index}")
        path = _member_name(item.get("path"))
        if path in excluded_paths or not _runtime_cache_member(
            pathlib.PurePosixPath(path), runtime_roots
        ):
            raise ArchiveError("runtime cache exclusion path is invalid")
        excluded_paths.add(path)
        size = item.get("size_bytes")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or SHA256_RE.fullmatch(str(item.get("sha256", ""))) is None
            or item.get("reason")
            != "derived per-run HOME/XDG/Matplotlib cache"
        ):
            raise ArchiveError("runtime cache exclusion record is invalid")
    if excluded_files != sorted(excluded_files, key=lambda item: item["path"]):
        raise ArchiveError("runtime cache exclusion records are not sorted")
    if runtime_policy.get("records_sha256") != canonical_sha256(excluded_files):
        raise ArchiveError("runtime cache exclusion digest differs")
    if any(
        (
            len(pathlib.PurePosixPath(record["path"]).parts) >= 5
            and pathlib.PurePosixPath(record["path"]).parts[0]
            == "runner-output"
            and pathlib.PurePosixPath(record["path"]).parts[1] == "runs"
            and pathlib.PurePosixPath(record["path"]).parts[4] == "runtime"
        )
        for record in records
    ):
        raise ArchiveError("runtime cache was included in the archive payload")
    if manifest.get("publication_safety") != {
        "policy": "private-sibling-stage-verify-fsync-atomic-noreplace-v2",
        "source_rescan_before_publication": True,
        "repository_destination_forbidden": True,
        "concurrent_ancestor_mutation_forbidden": True,
    }:
        raise ArchiveError("archive publication safety policy differs")
    if manifest.get("deterministic_tar") != deterministic_tar_metadata():
        raise ArchiveError("deterministic tar metadata differs")
    return complete


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    seal = subparsers.add_parser("seal", help="create a new sealed archive directory")
    seal.add_argument("--input", type=pathlib.Path, required=True)
    seal.add_argument("--archive", type=pathlib.Path, required=True)
    seal.add_argument("--repo", type=pathlib.Path, required=True)
    seal.add_argument("--build-receipt", type=pathlib.Path, required=True)
    seal.add_argument(
        "--tar",
        type=pathlib.Path,
        help="optionally create a deterministic tar archive at this new path",
    )
    verify = subparsers.add_parser("verify", help="verify an existing archive")
    verify.add_argument("--archive", type=pathlib.Path, required=True)
    verify.add_argument("--expected-manifest-sha256")
    verify.add_argument("--expected-receipt-id")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.action == "seal":
            complete = seal_archive(
                args.input, args.archive, args.repo, args.build_receipt
            )
            print(f"sealed archive: {_absolute(args.archive)}")
            print(f"archive manifest SHA-256: {complete['archive_manifest_sha256']}")
            if args.tar is not None:
                tar_record = create_deterministic_tar(args.archive, args.tar)
                print(f"deterministic tar: {tar_record['path']}")
                print(f"deterministic tar SHA-256: {tar_record['sha256']}")
            metadata = deterministic_tar_metadata()
            print(f"deterministic tar template: {metadata['command_template']}")
            print("tar checksum template: sha256sum {tar_path}")
        else:
            complete = verify_archive(
                args.archive,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_receipt_id=args.expected_receipt_id,
            )
            print(f"verified archive: {_absolute(args.archive)}")
            print(f"report SHA-256: {complete['report_sha256']}")
        return 0
    except (ArchiveError, OSError, ValueError) as exc:
        print(f"archive validation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
