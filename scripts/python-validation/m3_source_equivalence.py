#!/usr/bin/env python3
"""Derive a validation-script-independent identity from a build receipt."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import stat
from typing import Any


SCHEMA = "gpmeep-non-validation-source-identity-v2"
# Exact M1-to-final validation-harness delta. In particular, build scripts are
# not excluded. Any later path addition or edit must be consciously reviewed
# and added here or the retained source-manifest identity will differ.
EXCLUDED_PATHS = frozenset(
    {
        "scripts/benchmark-adjoint.py",
        "scripts/compare-mpi-adjoint-benchmarks.py",
        "scripts/python-validation/m3_execution_plan.py",
        "scripts/python-validation/m3_feature_plan.py",
        "scripts/python-validation/m3_mpi_rank_launcher.py",
        "scripts/python-validation/m3_performance_plan.py",
        "scripts/python-validation/m3_repeated_performance_replay.py",
        "scripts/python-validation/m3_source_equivalence.py",
        "scripts/python-validation/m3_specialized_plan.py",
        "scripts/python-validation/m3_specialized_replay.py",
        "scripts/python-validation/replay_external_m1.py",
        "scripts/python-validation/seal_m3_plan_replay.py",
        "scripts/python-validation/seal_m3_post_audit_release.py",
        "scripts/python-validation/seal_milestone_adversarial_audit.py",
        "scripts/python-validation/run_example_oracle.py",
        "scripts/python-validation/run_m3_auto_example_case.py",
        "scripts/python-validation/run_m3_host_matrix.py",
        "scripts/python-validation/run_m3_host_oracle.py",
        "scripts/python-validation/run_m3_mpi_example_case.py",
        "scripts/python-validation/run_m3_paired_matrix.py",
        "scripts/python-validation/run_m3_performance_sample.py",
        "scripts/python-validation/run_m3_release_aggregation.py",
        "scripts/python-validation/run_m3_repeated_performance_matrix.py",
        "scripts/python-validation/run_m3_specialized_matrix.py",
        "scripts/python-validation/tests/test_example_oracle.py",
        "scripts/run-mpi-adjoint-benchmark.py",
        "scripts/run-mpi-completion-ab.py",
        "scripts/tests/test_aunp_exact_resume.py",
        "scripts/tests/test_aunp_performance_workload.py",
        "scripts/tests/test_capture_gpu_telemetry_v2.py",
        "scripts/tests/test_capture_host_telemetry_v2.py",
        "scripts/tests/test_external_m1_replay.py",
        "scripts/tests/test_hybrid_aunp_matrix.py",
        "scripts/tests/test_m3_auto_example_case.py",
        "scripts/tests/test_m3_execution_plan.py",
        "scripts/tests/test_m3_feature_plan.py",
        "scripts/tests/test_m3_host_matrix.py",
        "scripts/tests/test_m3_host_oracle.py",
        "scripts/tests/test_m3_mpi_example_case.py",
        "scripts/tests/test_m3_mpi_rank_launcher.py",
        "scripts/tests/test_m3_paired_matrix.py",
        "scripts/tests/test_m3_performance_plan.py",
        "scripts/tests/test_m3_performance_sample.py",
        "scripts/tests/test_m3_plan_replay_seal.py",
        "scripts/tests/test_m3_post_audit_release.py",
        "scripts/tests/test_milestone_adversarial_audit.py",
        "scripts/tests/test_m3_release_aggregation.py",
        "scripts/tests/test_m3_repeated_performance_matrix.py",
        "scripts/tests/test_m3_repeated_performance_replay.py",
        "scripts/tests/test_m3_source_equivalence.py",
        "scripts/tests/test_m3_specialized_matrix.py",
        "scripts/tests/test_m3_specialized_plan.py",
        "scripts/tests/test_m3_specialized_replay.py",
        "scripts/tests/test_mpi_adjoint_runner.py",
        "scripts/tests/test_run_observed_process.py",
        "scripts/tests/test_user_workloads.py",
        "scripts/tests/test_verify_aunp_performance_workload.py",
        "scripts/tests/test_verify_telemetry_v2.py",
        "scripts/user-workloads/aunp_exact_resume.py",
        "scripts/user-workloads/capture_gpu_telemetry_v2.py",
        "scripts/user-workloads/capture_host_telemetry_v2.py",
        "scripts/user-workloads/compare_user_workloads.py",
        "scripts/user-workloads/run_aunp_performance_workload.py",
        "scripts/user-workloads/run_aunp_workload.py",
        "scripts/user-workloads/run_hybrid_aunp_matrix.py",
        "scripts/user-workloads/run_observed_process.py",
        "scripts/user-workloads/run_user_workload_matrix.py",
        "scripts/user-workloads/verify_aunp_performance_workload.py",
        "scripts/user-workloads/verify_telemetry_v2.py",
    }
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SourceIdentityError(RuntimeError):
    """Raised when a receipt cannot prove non-validation source equivalence."""


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(os.path.abspath(path))
    try:
        before = path.lstat()
    except OSError as exc:
        raise SourceIdentityError(f"build receipt is unavailable: {exc}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size > 64 * 1024**2
    ):
        raise SourceIdentityError("build receipt is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise SourceIdentityError("build receipt changed before reading")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            try:
                value = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise SourceIdentityError(f"build receipt is invalid JSON: {exc}") from exc
        after = path.lstat()
        if (
            stat.S_ISLNK(after.st_mode)
            or (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
        ):
            raise SourceIdentityError("build receipt changed while reading")
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise SourceIdentityError("build receipt is not a JSON object")
    return value


def _source_rows(receipt: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = receipt.get("source_start")
    if (
        not isinstance(source, dict)
        or source != receipt.get("source_end")
        or type(source.get("file_count")) is not int
        or source["file_count"] <= 0
        or not isinstance(source.get("files"), list)
        or len(source["files"]) != source["file_count"]
        or SHA256.fullmatch(str(source.get("sha256", ""))) is None
        or SHA256.fullmatch(str(source.get("source_manifest_sha256", ""))) is None
    ):
        raise SourceIdentityError("build receipt source snapshot differs")
    rows = source["files"]
    paths = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "kind",
            "mode_octal",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise SourceIdentityError("source manifest row schema differs")
        path = row["path"]
        pure = pathlib.PurePosixPath(path) if isinstance(path, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or pure.as_posix() != path
            or row["kind"] != "file"
            or re.fullmatch(r"0[0-7]{3}", str(row["mode_octal"])) is None
            or SHA256.fullmatch(str(row["sha256"])) is None
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] < 0
        ):
            raise SourceIdentityError("source manifest row identity differs")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise SourceIdentityError("source manifest path inventory differs")
    return source, rows


def _derive(
    receipt: dict[str, Any], source: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    retained = [
        row for row in rows if row["path"] not in EXCLUDED_PATHS
    ]
    excluded = len(rows) - len(retained)
    if not retained or excluded <= 0:
        raise SourceIdentityError("validation-source partition is empty")
    payload = json.dumps(
        retained,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or SHA256.fullmatch(receipt_id) is None:
        raise SourceIdentityError("build receipt identity differs")
    return {
        "schema": SCHEMA,
        "policy": {
            "excluded_paths": sorted(EXCLUDED_PATHS),
            "meaning": (
                "all receipt-bound source files outside the exact reviewed "
                "M1-to-final validation-harness delta"
            ),
        },
        "receipt_id": receipt_id,
        "full_source_sha256": source["sha256"],
        "retained_file_count": len(retained),
        "excluded_file_count": excluded,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def derive(receipt_path: pathlib.Path) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    source, rows = _source_rows(receipt)
    return _derive(receipt, source, rows)


def compare(first_path: pathlib.Path, second_path: pathlib.Path) -> dict[str, Any]:
    first_receipt = _read_json(first_path)
    second_receipt = _read_json(second_path)
    first_source, first_rows = _source_rows(first_receipt)
    second_source, second_rows = _source_rows(second_receipt)
    first = _derive(first_receipt, first_source, first_rows)
    second = _derive(second_receipt, second_source, second_rows)
    first_by_path = {row["path"]: row for row in first_rows}
    second_by_path = {row["path"]: row for row in second_rows}
    first_paths = set(first_by_path)
    second_paths = set(second_by_path)
    added = sorted(second_paths - first_paths)
    removed = sorted(first_paths - second_paths)
    modified = sorted(
        path
        for path in first_paths & second_paths
        if first_by_path[path] != second_by_path[path]
    )
    changed = set((*added, *removed, *modified))
    unexpected = sorted(changed - EXCLUDED_PATHS)
    if (
        first["policy"] != second["policy"]
        or first["retained_file_count"] != second["retained_file_count"]
        or first["sha256"] != second["sha256"]
        or unexpected
    ):
        raise SourceIdentityError(
            "receipt source delta escapes the exact validation allowlist"
            + (f": {', '.join(unexpected)}" if unexpected else "")
        )
    return {
        "schema": "gpmeep-m1-final-source-bridge-v1",
        "outcome": "PASS",
        "first": first,
        "second": second,
        "delta": {
            "added_paths": added,
            "removed_paths": removed,
            "modified_paths": modified,
            "changed_path_count": len(changed),
            "allowlisted_path_count": len(EXCLUDED_PATHS),
            "unexpected_paths": [],
        },
        "retained_source": {
            "file_count": first["retained_file_count"],
            "sha256": first["sha256"],
        },
    }
