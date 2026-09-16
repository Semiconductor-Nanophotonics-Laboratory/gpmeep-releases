#!/usr/bin/env python3
"""Load the frozen M3 release-performance execution ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import re
import stat
import sys
from dataclasses import asdict, dataclass
from typing import Any


SCHEMA = "gpmeep-m3-performance-plan-v1"
PLAN_NAME = "M3_PERFORMANCE_EXECUTION_UNITS_DRAFT.tsv"
PLAN_SHA256 = "b847ccb9de1897d9cc0ca11446e9b31c299e9c7bb9660256dbdc5f27fa2a6381"
FIELDS = (
    "launch_order",
    "unit_id",
    "driver",
    "target",
    "topologies",
    "warmup_cycles",
    "measured_cycles",
    "timing_contract",
    "acceptance",
    "timeout_seconds",
)
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
DRIVER_TOPOLOGIES = {
    "external-m1-hybrid": "cpu8,cuda1,cuda2",
    "external-m2-hybrid": "cpu8,cuda1,cuda2",
    "repeated-mpi-example": "cpu8,cuda1,cuda2",
    "repeated-mpi-example-auto": None,
    "external-specialized-unit": "cpu8,cuda1,cuda2",
}
EXPECTED_ROWS = (
    (
        "performance-m1-ters",
        "external-m1-hybrid",
        "milestone:M1-TERS",
        "cpu8,cuda1,cuda2",
    ),
    (
        "performance-m2-aunp",
        "external-m2-hybrid",
        "milestone:M2-AuNP",
        "cpu8,cuda1,cuda2",
    ),
    (
        "performance-edge-emitter-3d",
        "repeated-mpi-example",
        "python/examples/edge_emitter_3D.py",
        "cpu8,cuda1,cuda2",
    ),
    (
        "performance-metasurface-crossover",
        "repeated-mpi-example-auto",
        "python/examples/metasurface_lens.py",
        "cpu8,cpu1,cuda1,cuda2,auto1",
    ),
    (
        "performance-near2far-transform",
        "external-specialized-unit",
        "specialized-near2far-mpi-qualification",
        "cpu8,cuda1,cuda2",
    ),
    (
        "performance-materialgrid-adjoint",
        "external-specialized-unit",
        "specialized-mpi-adjoint-qualification",
        "cpu8,cuda1,cuda2",
    ),
    (
        "performance-long-horizon-policy",
        "repeated-mpi-example-auto",
        "python/examples/stochastic_emitter_line.py",
        "cpu8,cpu1,cuda1,auto1",
    ),
)
TIMING_CONTRACTS = {
    "controller-rederived-fixed-window-fdtd-wall",
    "controller-rederived-fixed-work-phase-wall",
    "sum-all-simulation-run-wall-seconds",
    "run-index-18-small-and-19-large-wall-seconds",
    "rederived-native-batch-lane-median",
    "rederived-forward-adjoint-trajectory-wall",
}


class PlanError(RuntimeError):
    """Raised when the performance plan is not the frozen release plan."""


@dataclass(frozen=True)
class PerformanceUnit:
    launch_order: int
    unit_id: str
    driver: str
    target: str
    topologies: tuple[str, ...]
    warmup_cycles: int
    measured_cycles: int
    timing_contract: str
    acceptance: str
    timeout_seconds: int


def _stable_bytes(path: pathlib.Path) -> bytes:
    path = pathlib.Path(os.path.abspath(path))
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PlanError("M3 performance plan is not a regular non-symlink file")
    if before.st_size > 4 * 1024 * 1024:
        raise PlanError("M3 performance plan exceeds its input bound")
    with path.open("rb") as handle:
        value = handle.read()
        descriptor = os.fstat(handle.fileno())
    after = path.lstat()

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if identity(before) != identity(descriptor) or identity(before) != identity(after):
        raise PlanError("M3 performance plan changed while it was read")
    return value


def _canonical_positive(value: str, label: str, *, allow_zero: bool = False) -> int:
    if not value.isascii() or not value.isdigit() or (
        len(value) > 1 and value.startswith("0")
    ):
        raise PlanError(f"{label} is not a canonical integer")
    number = int(value)
    if number < (0 if allow_zero else 1):
        raise PlanError(f"{label} is outside its positive range")
    return number


def _relative_target(value: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
        or path.as_posix() != value
    ):
        raise PlanError("M3 performance target is not a safe relative path")
    return path


def parse_plan(value: bytes) -> list[PerformanceUnit]:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PlanError("M3 performance plan is not strict UTF-8") from exc
    if (
        "\r" in text
        or "\x00" in text
        or '"' in text
        or not text.endswith("\n")
        or any(line.count("\t") != len(FIELDS) - 1 for line in text.splitlines())
    ):
        raise PlanError("M3 performance plan has noncanonical text encoding")
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t", strict=True)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise PlanError("M3 performance plan header differs")
        rows = list(reader)
    except csv.Error as exc:
        raise PlanError(f"M3 performance plan is malformed TSV: {exc}") from exc
    if len(rows) != len(EXPECTED_ROWS):
        raise PlanError("M3 performance plan must contain exactly seven units")
    units = []
    for order, (row, expected) in enumerate(zip(rows, EXPECTED_ROWS, strict=True), 1):
        if set(row) != set(FIELDS) or any(not item for item in row.values()):
            raise PlanError("M3 performance plan row schema differs")
        if _canonical_positive(row["launch_order"], "launch order") != order:
            raise PlanError("M3 performance launch order is not contiguous")
        unit_id, driver, target, topologies = expected
        if (
            row["unit_id"] != unit_id
            or SAFE_ID.fullmatch(row["unit_id"]) is None
            or row["driver"] != driver
            or row["target"] != target
            or row["topologies"] != topologies
            or DRIVER_TOPOLOGIES.get(driver, topologies) not in {None, topologies}
        ):
            raise PlanError("M3 performance unit identity differs")
        if target.startswith("python/"):
            _relative_target(target)
        elif driver.startswith("external-") and not (
            target.startswith("milestone:") or target.startswith("specialized-")
        ):
            raise PlanError("M3 performance external target differs")
        warmups = _canonical_positive(
            row["warmup_cycles"], "warmup cycles", allow_zero=True
        )
        measured = _canonical_positive(row["measured_cycles"], "measured cycles")
        expected_measured = 7 if unit_id == "performance-near2far-transform" else 5
        if measured != expected_measured or (
            driver.startswith("repeated-") and warmups != 1
        ):
            raise PlanError("M3 performance repetition contract differs")
        if driver.startswith("external-m") and warmups != 0:
            raise PlanError("M3 external milestone warmup contract differs")
        if row["timing_contract"] not in TIMING_CONTRACTS:
            raise PlanError("M3 performance timing contract differs")
        if len(row["acceptance"].encode("utf-8")) > 2048:
            raise PlanError("M3 performance acceptance text is too large")
        timeout = _canonical_positive(row["timeout_seconds"], "timeout")
        if not 600 <= timeout <= 48 * 3600:
            raise PlanError("M3 performance timeout is outside [10m,48h]")
        units.append(
            PerformanceUnit(
                order,
                unit_id,
                driver,
                target,
                tuple(topologies.split(",")),
                warmups,
                measured,
                row["timing_contract"],
                row["acceptance"],
                timeout,
            )
        )
    if len({unit.timing_contract for unit in units}) != 6:
        raise PlanError("M3 performance timing-family inventory differs")
    return units


def load_release_plan(plan_root: pathlib.Path, repo: pathlib.Path) -> dict[str, Any]:
    plan_root = pathlib.Path(os.path.abspath(plan_root))
    repo = pathlib.Path(os.path.abspath(repo))
    for root, label in ((plan_root, "plan root"), (repo, "repository")):
        if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
            raise PlanError(f"M3 performance {label} is not canonical")
    path = plan_root / PLAN_NAME
    value = _stable_bytes(path)
    digest = hashlib.sha256(value).hexdigest()
    if digest != PLAN_SHA256:
        raise PlanError("M3 performance plan SHA-256 differs")
    units = parse_plan(value)
    for unit in units:
        if not unit.target.startswith("python/"):
            continue
        target = repo.joinpath(*pathlib.PurePosixPath(unit.target).parts)
        metadata = target.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or target.resolve(strict=True) != target
        ):
            raise PlanError(
                f"M3 performance target is not a repository file: {unit.target}"
            )
    return {
        "schema": SCHEMA,
        "input": {
            "path": str(path.resolve(strict=True)),
            "size_bytes": len(value),
            "sha256": digest,
        },
        "units": [
            {**asdict(unit), "topologies": list(unit.topologies)} for unit in units
        ],
        "counts": {
            "physical_units": 7,
            "reused_terminal_units": 4,
            "new_repeated_example_units": 3,
            "release_speed_gate_units": 6,
            "policy_gap_units": 1,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-root", required=True, type=pathlib.Path)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            load_release_plan(args.plan_root, args.repo),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, PlanError, ValueError) as error:
        print(f"M3 performance plan error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
