#!/usr/bin/env python3
"""Independently replay a terminal fixed-work AuNP performance sample."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    StableFile,
    WorkloadError,
    file_record,
    validate_runtime_against_receipt,
)
import run_aunp_performance_workload as performance  # noqa: E402


SCHEMA = "gpmeep-aunp-r4000-performance-replay-v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--expected-backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--expected-mpi-size", required=True, type=int)
    parser.add_argument("--result", type=pathlib.Path)
    return parser.parse_args(argv)


def _external_record_path(record: Any, label: str) -> pathlib.Path:
    if not isinstance(record, dict):
        raise WorkloadError(f"AuNP performance {label} record is absent")
    raw = record.get("path")
    if not isinstance(raw, str) or not pathlib.Path(raw).is_absolute():
        raise WorkloadError(f"AuNP performance {label} path is not absolute")
    path = pathlib.Path(raw)
    with StableFile(path, f"AuNP performance {label}", expected=record):
        pass
    return path.resolve()


def _write_exclusive(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise WorkloadError("AuNP performance replay write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def replay(
    repo: pathlib.Path,
    output: pathlib.Path,
    expected_backend: str,
    expected_mpi_size: int,
) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    output = output.resolve(strict=True)
    if expected_mpi_size <= 0:
        raise WorkloadError("AuNP performance expected MPI size is invalid")
    summary = performance.validate_performance_output(
        output,
        expected_backend=expected_backend,
        expected_mpi_size=expected_mpi_size,
    )
    receipt_record = summary.get("build_receipt")
    receipt_path = _external_record_path(receipt_record, "build receipt")
    scripts_directory = repo / "scripts"
    if str(scripts_directory) not in sys.path:
        sys.path.insert(0, str(scripts_directory))
    try:
        from gpmeep_provenance import ProvenanceError, verify_build_receipt

        receipt = verify_build_receipt(receipt_path, repo)
    except (ImportError, ProvenanceError) as exc:
        raise WorkloadError(
            f"AuNP performance build receipt could not be replayed: {exc}"
        ) from exc
    for key in ("receipt_id", "build_input_id", "artifact_set_id", "source_start"):
        if receipt_record.get(key) != receipt.get(key):
            raise WorkloadError(f"AuNP performance build receipt {key} differs")
    for rank_record in summary["rank_records"]:
        rank = int(rank_record["rank"])
        if rank_record.get("build_receipt_id") != receipt["receipt_id"]:
            raise WorkloadError(f"AuNP performance rank {rank} receipt ID differs")
        runtime = rank_record.get("runtime_artifacts")
        if not isinstance(runtime, dict) or not runtime:
            raise WorkloadError(
                f"AuNP performance rank {rank} runtime provenance is absent"
            )
        for name, record in runtime.items():
            _external_record_path(record, f"rank {rank} runtime {name}")
        validate_runtime_against_receipt(runtime, receipt, repo)
    phase_timesteps = [
        int(summary["rank_records"][0]["records"][index]["timestep_delta"])
        for index in range(len(performance.PHASE_SPECS))
    ]
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "expected_backend": expected_backend,
        "expected_mpi_size": expected_mpi_size,
        "fdtd_wall_seconds": summary["fdtd_wall_seconds"],
        "phase_wall_seconds": summary["phase_wall_seconds"],
        "phase_timestep_deltas": phase_timesteps,
        "workload_end_to_end_seconds": summary["workload_end_to_end_seconds"],
        "build_receipt": {
            key: receipt_record[key]
            for key in (
                "path",
                "size_bytes",
                "sha256",
                "receipt_id",
                "build_input_id",
                "artifact_set_id",
            )
        },
        "receipt_id": receipt["receipt_id"],
        "source_start": {
            key: receipt["source_start"][key]
            for key in (
                "file_count",
                "sha256",
                "source_manifest_schema_version",
                "source_manifest_sha256",
            )
        },
        "artifacts": {
            "summary": file_record(output / "summary.json", output),
            "complete": file_record(output / "COMPLETE", output),
            "adapter": file_record(
                SCRIPT_DIR / "run_aunp_performance_workload.py", repo
            ),
            "common": file_record(SCRIPT_DIR / "common.py", repo),
            "verifier": file_record(pathlib.Path(__file__), repo),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = replay(
        args.repo,
        args.output,
        args.expected_backend,
        args.expected_mpi_size,
    )
    payload = (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if args.result is None:
        sys.stdout.buffer.write(payload)
    else:
        result = args.result.resolve()
        try:
            result.relative_to(args.output.resolve())
        except ValueError:
            pass
        else:
            raise WorkloadError(
                "AuNP performance replay result must be outside the sealed sample"
            )
        _write_exclusive(result, payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkloadError as error:
        print(f"AuNP performance replay error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
