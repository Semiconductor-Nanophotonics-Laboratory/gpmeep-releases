#!/usr/bin/env python3
"""Load and validate the frozen Milestone 3 execution-unit ledgers."""

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


SCHEMA = "gpmeep-m3-execution-plan-v1"
SELECTED_NAME = "M3_EXECUTION_UNITS_DRAFT.tsv"
STRONGER_NAME = "M3_STRONGER_EXECUTION_UNITS_DRAFT.tsv"
HOST_NAME = "M3_HOST_EXECUTION_UNITS_DRAFT.tsv"
EXPECTED_SHA256 = {
    SELECTED_NAME: "c01769d2fce96ef9a103f9a88a7f1f5f8f13b13a7e0b72b0f8f781637b0b7bb8",
    STRONGER_NAME: "ca9da419fd37dba881deebe182aae255efe0e076c803ef3e66689ba37972bdd4",
    HOST_NAME: "b95112a3d61d9d12d529f1a34badbaf5da423c4789616e79274cf572d651bfc8",
}
SELECTED_FIELDS = (
    "launch_order",
    "unit_id",
    "cluster_id",
    "case_path",
    "backends",
    "purpose",
    "release_role",
)
STRONGER_FIELDS = (
    "launch_order",
    "unit_id",
    "covered_example",
    "test_case_path",
    "backends",
    "driver",
    "oracle_contract",
    "timeout_seconds",
)
HOST_FIELDS = (
    "launch_order",
    "unit_id",
    "driver",
    "target_path",
    "backend",
    "covers_examples",
    "oracle_contract",
    "timeout_seconds",
)
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
HOST_DRIVERS = {
    "python-validation-exact-case",
    "python-unittest-file",
    "runpy-finite-array-adapter",
    "render-from-sealed-producer-metrics",
}


class PlanError(RuntimeError):
    """Raised when an M3 execution plan is not the frozen release plan."""


@dataclass(frozen=True)
class PairedUnit:
    launch_order: int
    unit_id: str
    section: str
    case_path: str
    purpose: str
    timeout_seconds: int | None
    release_role: str
    covered_example: str | None


@dataclass(frozen=True)
class HostUnit:
    launch_order: int
    unit_id: str
    driver: str
    target_path: str
    covers_examples: tuple[str, ...]
    oracle_contract: str
    timeout_seconds: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_regular_bytes(path: pathlib.Path, label: str) -> bytes:
    path = pathlib.Path(os.path.abspath(path))
    try:
        before = path.lstat()
    except OSError as exc:
        raise PlanError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PlanError(f"{label} is not a regular non-symlink file")
    if before.st_size > 4 * 1024 * 1024:
        raise PlanError(f"{label} exceeds the plan input size bound")
    try:
        with path.open("rb") as handle:
            value = handle.read()
            descriptor = os.fstat(handle.fileno())
    except OSError as exc:
        raise PlanError(f"{label} could not be read: {exc}") from exc
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
        raise PlanError(f"{label} changed while it was read")
    return value


def _rows(value: bytes, fields: tuple[str, ...], label: str) -> list[dict[str, str]]:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PlanError(f"{label} is not strict UTF-8") from exc
    if (
        "\r" in text
        or not text.endswith("\n")
        or "\x00" in text
        or '"' in text
        or any(line.count("\t") != len(fields) - 1 for line in text.splitlines())
    ):
        raise PlanError(f"{label} has noncanonical text encoding")
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t", strict=True)
        if tuple(reader.fieldnames or ()) != fields:
            raise PlanError(f"{label} header differs")
        result = list(reader)
    except csv.Error as exc:
        raise PlanError(f"{label} is malformed TSV: {exc}") from exc
    if not result or any(set(row) != set(fields) for row in result):
        raise PlanError(f"{label} row schema differs")
    if any(any(value == "" for value in row.values()) for row in result):
        raise PlanError(f"{label} contains an empty field")
    return result


def _positive_integer(value: str, label: str) -> int:
    if not value.isascii() or not value.isdigit() or value.startswith("0"):
        raise PlanError(f"{label} is not a canonical positive integer")
    number = int(value)
    if number <= 0:
        raise PlanError(f"{label} is not positive")
    return number


def _relative_path(value: str, label: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise PlanError(f"{label} is not a safe repository-relative path")
    if path.as_posix() != value:
        raise PlanError(f"{label} is not canonical")
    return path


def _unit_id(value: str, label: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise PlanError(f"{label} is not a safe unit ID")
    return value


def parse_selected(value: bytes) -> list[PairedUnit]:
    rows = _rows(value, SELECTED_FIELDS, SELECTED_NAME)
    units = []
    for expected_order, row in enumerate(rows, start=1):
        order = _positive_integer(row["launch_order"], "selected launch_order")
        if order != expected_order:
            raise PlanError("selected launch order is not contiguous")
        if row["backends"] != "cpu,cuda":
            raise PlanError("selected unit backend topology differs")
        if row["release_role"] not in {"fail_fast_canary", "full_matrix"}:
            raise PlanError("selected release role differs")
        expected_role = "fail_fast_canary" if order <= 10 else "full_matrix"
        if row["release_role"] != expected_role:
            raise PlanError("selected canary partition differs")
        _relative_path(row["case_path"], "selected case path")
        units.append(
            PairedUnit(
                order,
                _unit_id(row["unit_id"], "selected unit ID"),
                _unit_id(row["cluster_id"], "selected cluster ID"),
                row["case_path"],
                row["purpose"],
                None,
                row["release_role"],
                None,
            )
        )
    return units


def parse_stronger(value: bytes) -> list[PairedUnit]:
    rows = _rows(value, STRONGER_FIELDS, STRONGER_NAME)
    units = []
    for expected_order, row in enumerate(rows, start=1):
        order = _positive_integer(row["launch_order"], "stronger launch_order")
        if order != expected_order:
            raise PlanError("stronger launch order is not contiguous")
        if row["backends"] != "cpu,cuda":
            raise PlanError("stronger unit backend topology differs")
        if row["driver"] != "python-validation-exact-case":
            raise PlanError("stronger unit driver differs")
        _relative_path(row["covered_example"], "stronger covered example")
        _relative_path(row["test_case_path"], "stronger test path")
        units.append(
            PairedUnit(
                order,
                _unit_id(row["unit_id"], "stronger unit ID"),
                "stronger-tests",
                row["test_case_path"],
                row["oracle_contract"],
                _positive_integer(row["timeout_seconds"], "stronger timeout"),
                "stronger_test",
                row["covered_example"],
            )
        )
    return units


def parse_host(value: bytes) -> list[HostUnit]:
    rows = _rows(value, HOST_FIELDS, HOST_NAME)
    units = []
    for expected_order, row in enumerate(rows, start=1):
        order = _positive_integer(row["launch_order"], "host launch_order")
        if order != expected_order:
            raise PlanError("host launch order is not contiguous")
        if row["backend"] != "cpu" or row["driver"] not in HOST_DRIVERS:
            raise PlanError("host unit execution policy differs")
        _relative_path(row["target_path"], "host target path")
        covered = tuple(row["covers_examples"].split(";"))
        if not covered or any(not item for item in covered):
            raise PlanError("host coverage list is invalid")
        for item in covered:
            _relative_path(item, "host covered example")
        units.append(
            HostUnit(
                order,
                _unit_id(row["unit_id"], "host unit ID"),
                row["driver"],
                row["target_path"],
                covered,
                row["oracle_contract"],
                _positive_integer(row["timeout_seconds"], "host timeout"),
            )
        )
    return units


def _existing_repo_file(repo: pathlib.Path, relative: str, label: str) -> None:
    path = repo.joinpath(*pathlib.PurePosixPath(relative).parts)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PlanError(f"{label} is unavailable: {relative}: {exc}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or path.resolve(strict=True) != path
    ):
        raise PlanError(f"{label} is not a regular non-symlink file: {relative}")


def load_release_plan(plan_root: pathlib.Path, repo: pathlib.Path) -> dict[str, Any]:
    plan_root = pathlib.Path(os.path.abspath(plan_root))
    repo = pathlib.Path(os.path.abspath(repo))
    if (
        plan_root.is_symlink()
        or not plan_root.is_dir()
        or plan_root.resolve(strict=True) != plan_root
    ):
        raise PlanError("M3 plan root is not a non-symlink directory")
    if repo.is_symlink() or not repo.is_dir() or repo.resolve(strict=True) != repo:
        raise PlanError("M3 repository root is not a non-symlink directory")
    payloads = {}
    records = {}
    for name, digest in EXPECTED_SHA256.items():
        path = plan_root / name
        value = _stable_regular_bytes(path, name)
        actual = sha256_bytes(value)
        if actual != digest:
            raise PlanError(f"{name} SHA-256 differs")
        payloads[name] = value
        records[name] = {
            "path": str(path.resolve(strict=True)),
            "size_bytes": len(value),
            "sha256": actual,
        }
    selected = parse_selected(payloads[SELECTED_NAME])
    stronger = parse_stronger(payloads[STRONGER_NAME])
    host = parse_host(payloads[HOST_NAME])
    if (len(selected), len(stronger), len(host)) != (56, 14, 4):
        raise PlanError("M3 physical-unit inventory differs from 56+14+4")
    unit_ids = [unit.unit_id for unit in (*selected, *stronger, *host)]
    if len(unit_ids) != len(set(unit_ids)):
        raise PlanError("M3 physical unit IDs are not globally unique")
    paired_paths = [unit.case_path for unit in (*selected, *stronger)]
    if len(paired_paths) != len(set(paired_paths)):
        raise PlanError("M3 paired case paths are not unique")
    covered = [item for unit in host for item in unit.covers_examples]
    if len(covered) != 15 or len(covered) != len(set(covered)):
        raise PlanError("M3 host-only coverage is not an exact 15-path partition")
    for unit in (*selected, *stronger):
        _existing_repo_file(repo, unit.case_path, "M3 paired case")
        if unit.covered_example is not None:
            _existing_repo_file(repo, unit.covered_example, "M3 covered example")
    for unit in host:
        _existing_repo_file(repo, unit.target_path, "M3 host target")
        for path in unit.covers_examples:
            _existing_repo_file(repo, path, "M3 host covered example")
    return {
        "schema": SCHEMA,
        "inputs": records,
        "selected": [asdict(unit) for unit in selected],
        "stronger": [asdict(unit) for unit in stronger],
        "host": [asdict(unit) for unit in host],
        "counts": {
            "selected_units": 56,
            "stronger_units": 14,
            "host_units": 4,
            "paired_units": 70,
            "physical_units": 74,
            "backend_executions": 144,
            "host_example_paths": 15,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-root", required=True, type=pathlib.Path)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = load_release_plan(args.plan_root, args.repo)
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, PlanError, ValueError) as error:
        print(f"M3 execution plan error: {error}", file=os.sys.stderr, flush=True)
        raise SystemExit(2)
