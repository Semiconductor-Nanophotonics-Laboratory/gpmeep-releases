#!/usr/bin/env python3
"""Replay an M1 terminal from its immutable source worktree."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import types
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m3_source_equivalence as source_equivalence  # noqa: E402


SCHEMA = "gpmeep-external-m1-terminal-replay-v2"
PREFIX = "GPMEEP_EXTERNAL_M1_REPLAY="


class ReplayError(RuntimeError):
    """Raised when the immutable M1 verifier cannot be replayed."""


def _module(path: pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("gpmeep_external_m1_verifier", path)
    if spec is None or spec.loader is None:
        raise ReplayError("could not load the external M1 verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"could not read verified M1 checkpoint: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError("verified M1 checkpoint is not an object")
    return value


def replay(repo: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    repo = pathlib.Path(os.path.abspath(repo))
    output = pathlib.Path(os.path.abspath(output))
    if (
        repo.is_symlink()
        or not repo.is_dir()
        or repo.resolve(strict=True) != repo
        or output.is_symlink()
        or not output.is_dir()
        or output.resolve(strict=True) != output
    ):
        raise ReplayError("external M1 root is not canonical")
    workload_dir = repo / "scripts/user-workloads"
    if str(workload_dir) in sys.path:
        sys.path.remove(str(workload_dir))
    sys.path.insert(0, str(workload_dir))
    verifier_path = workload_dir / "run_hybrid_ters_matrix.py"
    module = _module(verifier_path.resolve(strict=True))
    try:
        comparator = module.matrix.COMPARATOR
        original_compare = comparator.compare_ters
    except AttributeError as exc:
        raise ReplayError("external M1 exact comparator is absent") from exc
    captured_comparisons: list[dict[str, Any]] = []

    def capture_compare(arguments: Any) -> dict[str, Any]:
        value = original_compare(arguments)
        if not isinstance(value, dict):
            raise ReplayError("external M1 exact comparator returned a non-object")
        captured_comparisons.append(value)
        return value

    comparator.compare_ters = capture_compare
    try:
        report = module.verify_hybrid_complete(output)
    finally:
        comparator.compare_ters = original_compare
    if (
        len(captured_comparisons) != 1
        or captured_comparisons[0].get("outcome") != "PASS"
    ):
        raise ReplayError("external M1 verifier did not replay one exact comparison")
    checkpoint = _load_json(output / "CHECKPOINT.json")
    try:
        receipt_path = pathlib.Path(
            checkpoint["contract"]["builds"]["shared_fp32"]["receipt_path"]
        ).resolve(strict=True)
    except (KeyError, TypeError, OSError) as exc:
        raise ReplayError("external M1 FP32 receipt path differs") from exc
    tasks = report.get("tasks")
    if not isinstance(tasks, list):
        raise ReplayError("external M1 task inventory differs")
    verifier_record = module.matrix.stable_file_record(
        verifier_path, repo, "external M1 verifier"
    )
    verifier_record["path"] = str(verifier_path.resolve(strict=True))
    receipt_record = module.matrix.stable_file_record(
        receipt_path, repo, "external M1 shared FP32 receipt"
    )
    receipt_record["path"] = str(receipt_path)
    return {
        "schema": SCHEMA,
        "outcome": "PASS",
        "source_snapshot_sha256": report["source_snapshot_sha256"],
        "gpu_devices": report["gpu_devices"],
        "tasks": [{"task": record["task"]} for record in tasks],
        "task_count": len(tasks),
        "performance": report["performance"],
        "timing_stability": report["timing_stability"],
        "exact_comparison": captured_comparisons[0],
        "shared_fp32_source_identity": source_equivalence.derive(receipt_path),
        "shared_fp32_receipt": receipt_record,
        "verifier": verifier_record,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    value = replay(args.repo, args.output)
    print(PREFIX + json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"external M1 replay error: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
