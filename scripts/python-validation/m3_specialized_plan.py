#!/usr/bin/env python3
"""Load the frozen M3 distributed-feature execution ledger."""

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
from dataclasses import asdict, dataclass
from typing import Any


SCHEMA = "gpmeep-m3-specialized-plan-v1"
PLAN_NAME = "M3_SPECIALIZED_EXECUTION_UNITS_DRAFT.tsv"
PLAN_SHA256 = "e63c4c493f41609f9cc22d604dbfa9e2781409659faa63d401f20460c8ab080d"
FIELDS = (
    "launch_order",
    "unit_id",
    "driver",
    "target_path",
    "coverage_family",
    "topologies",
    "oracle_contract",
    "timeout_seconds",
)
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
DRIVER_TOPOLOGIES = {
    "python-mpi-example-case": "cpu8,cuda1,cuda2",
    "run-mpi-python-validation": "cpu2,cuda2-pinned,cuda2-aware",
    "run-near2far-mpi-qualification": "cpu8,cuda1,cuda2",
    "run-mpi-adjoint-benchmark": "cpu8,cuda1,cuda2",
}
EXPECTED_ROWS = (
    (
        "specialized-mpi-python-core",
        "run-mpi-python-validation",
        "scripts/mpi-python-validation-probe.py",
    ),
    (
        "specialized-differential-scattering",
        "python-mpi-example-case",
        "python/examples/differential_cross_section.py",
    ),
    (
        "specialized-zone-plate",
        "python-mpi-example-case",
        "python/examples/zone_plate.py",
    ),
    (
        "specialized-edge-emitter-3d",
        "python-mpi-example-case",
        "python/examples/edge_emitter_3D.py",
    ),
    (
        "specialized-near2far-3d-adjoint",
        "python-mpi-example-case",
        "python/examples/adjoint_optimization/near2far_3d_validation.py",
    ),
)


class PlanError(RuntimeError):
    """Raised when the distributed-feature plan is not the frozen plan."""


@dataclass(frozen=True)
class SpecializedUnit:
    launch_order: int
    unit_id: str
    driver: str
    target_path: str
    coverage_family: str
    topologies: tuple[str, ...]
    oracle_contract: str
    timeout_seconds: int


def _stable_bytes(path: pathlib.Path) -> bytes:
    path = pathlib.Path(os.path.abspath(path))
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PlanError("M3 specialized plan is not a regular non-symlink file")
    if before.st_size > 4 * 1024 * 1024:
        raise PlanError("M3 specialized plan exceeds its input bound")
    with path.open("rb") as handle:
        value = handle.read()
        descriptor = os.fstat(handle.fileno())
    after = path.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(descriptor) or identity(before) != identity(after):
        raise PlanError("M3 specialized plan changed while it was read")
    return value


def _relative_path(value: str) -> str:
    path = pathlib.PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
        or path.as_posix() != value
    ):
        raise PlanError("M3 specialized target is not a safe relative path")
    return value


def _positive(value: str, label: str) -> int:
    if (
        not value.isascii()
        or not value.isdigit()
        or value.startswith("0")
        or int(value) <= 0
    ):
        raise PlanError(f"{label} is not a canonical positive integer")
    return int(value)


def parse_plan(value: bytes) -> list[SpecializedUnit]:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PlanError("M3 specialized plan is not strict UTF-8") from exc
    if (
        "\r" in text
        or "\x00" in text
        or '"' in text
        or not text.endswith("\n")
        or any(line.count("\t") != len(FIELDS) - 1 for line in text.splitlines())
    ):
        raise PlanError("M3 specialized plan has noncanonical text encoding")
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t", strict=True)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise PlanError("M3 specialized plan header differs")
        rows = list(reader)
    except csv.Error as exc:
        raise PlanError(f"M3 specialized plan is malformed TSV: {exc}") from exc
    if len(rows) != len(EXPECTED_ROWS):
        raise PlanError("M3 specialized plan must contain exactly five units")
    units = []
    for index, (row, expected) in enumerate(zip(rows, EXPECTED_ROWS, strict=True), 1):
        if set(row) != set(FIELDS) or any(not item for item in row.values()):
            raise PlanError("M3 specialized plan row schema differs")
        order = _positive(row["launch_order"], "specialized launch order")
        if order != index:
            raise PlanError("M3 specialized launch order is not contiguous")
        unit_id, driver, target = expected
        if (
            row["unit_id"] != unit_id
            or SAFE_ID.fullmatch(row["unit_id"]) is None
            or row["driver"] != driver
            or row["target_path"] != target
        ):
            raise PlanError("M3 specialized unit identity differs")
        if row["topologies"] != DRIVER_TOPOLOGIES.get(driver):
            raise PlanError("M3 specialized topology contract differs")
        if SAFE_ID.fullmatch(row["coverage_family"]) is None:
            raise PlanError("M3 specialized coverage family is not canonical")
        _relative_path(row["target_path"])
        timeout = _positive(row["timeout_seconds"], "specialized timeout")
        if not 600 <= timeout <= 48 * 3600:
            raise PlanError("M3 specialized timeout is outside [10m,48h]")
        if len(row["oracle_contract"].encode("utf-8")) > 4096:
            raise PlanError("M3 specialized oracle contract is too large")
        units.append(
            SpecializedUnit(
                order,
                unit_id,
                driver,
                target,
                row["coverage_family"],
                tuple(row["topologies"].split(",")),
                row["oracle_contract"],
                timeout,
            )
        )
    if len({unit.coverage_family for unit in units}) != len(units):
        raise PlanError("M3 specialized coverage families are not unique")
    return units


def load_release_plan(plan_root: pathlib.Path, repo: pathlib.Path) -> dict[str, Any]:
    plan_root = pathlib.Path(os.path.abspath(plan_root))
    repo = pathlib.Path(os.path.abspath(repo))
    for path, label in ((plan_root, "plan root"), (repo, "repository")):
        if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
            raise PlanError(f"M3 specialized {label} is not canonical")
    path = plan_root / PLAN_NAME
    value = _stable_bytes(path)
    digest = hashlib.sha256(value).hexdigest()
    if digest != PLAN_SHA256:
        raise PlanError("M3 specialized plan SHA-256 differs")
    units = parse_plan(value)
    for unit in units:
        target = repo.joinpath(*pathlib.PurePosixPath(unit.target_path).parts)
        metadata = target.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or target.resolve(strict=True) != target
        ):
            raise PlanError(f"M3 specialized target is not a repository file: {unit.target_path}")
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
            "physical_units": 5,
            "python_mpi_example_units": 4,
            "specialized_qualification_units": 1,
            "cpu8_gpu1_gpu2_units": 4,
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
        print(f"M3 specialized plan error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
