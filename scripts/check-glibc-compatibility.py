#!/usr/bin/env python3
"""Fail a build when a gpmeep-owned ELF exceeds its declared GLIBC floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
from typing import Any


SCHEMA = "gpmeep-glibc-compatibility-v1"
VERSION_PATTERN = re.compile(r"\bName: (GLIBC_([0-9]+(?:\.[0-9]+)+))\b")


class CompatibilityError(RuntimeError):
    pass


def version_key(value: str) -> tuple[int, ...]:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value) is None:
        raise CompatibilityError(f"invalid GLIBC version {value!r}")
    return tuple(int(part) for part in value.split("."))


def parse_required_versions(output: str) -> list[str]:
    values = {match.group(2) for match in VERSION_PATTERN.finditer(output)}
    return sorted(values, key=version_key)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_elf(path: pathlib.Path, root: pathlib.Path) -> tuple[pathlib.Path, str]:
    absolute = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise CompatibilityError(f"ELF input is unavailable: {absolute}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise CompatibilityError(f"ELF input is not a regular file: {absolute}")
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise CompatibilityError(f"ELF input escapes audit root: {absolute}") from error
    if not resolved.is_file():
        raise CompatibilityError(f"ELF input is not a regular file: {absolute}")
    with resolved.open("rb") as stream:
        magic = stream.read(4)
    if magic != b"\x7fELF":
        raise CompatibilityError(f"input is not an ELF file: {absolute}")
    return resolved, relative


def audit(
    *, root: pathlib.Path, paths: list[pathlib.Path], maximum: str, readelf: pathlib.Path
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise CompatibilityError(f"audit root is not a directory: {root}")
    maximum_key = version_key(maximum)
    if not readelf.is_absolute() or not os.access(readelf, os.X_OK):
        raise CompatibilityError(f"readelf is not an executable absolute path: {readelf}")
    if not paths:
        raise CompatibilityError("no gpmeep ELF inputs were supplied")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    overall: list[str] = []
    for raw in paths:
        path, relative = _regular_elf(raw, root)
        if relative in seen:
            raise CompatibilityError(f"duplicate ELF input: {relative}")
        seen.add(relative)
        completed = subprocess.run(
            [str(readelf), "--version-info", "--wide", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            raise CompatibilityError(
                f"readelf failed for {relative}: rc={completed.returncode}"
            )
        versions = parse_required_versions(completed.stdout)
        if not versions:
            raise CompatibilityError(f"ELF has no GLIBC version requirements: {relative}")
        maximum_required = versions[-1]
        if version_key(maximum_required) > maximum_key:
            raise CompatibilityError(
                f"{relative} requires GLIBC_{maximum_required}, above {maximum}"
            )
        overall.extend(versions)
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "required_versions": versions,
                "maximum_required": maximum_required,
            }
        )
    overall_versions = sorted(set(overall), key=version_key)
    return {
        "schema": SCHEMA,
        "state": "PASS",
        "maximum_allowed": maximum,
        "overall_maximum_required": overall_versions[-1],
        "elf_count": len(records),
        "elf_files": sorted(records, key=lambda item: item["path"]),
    }


def _write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    path = pathlib.Path(os.path.abspath(os.fspath(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
    except FileExistsError as error:
        raise CompatibilityError(f"refusing to overwrite GLIBC report: {path}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--maximum", required=True)
    parser.add_argument("--readelf", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("paths", type=pathlib.Path, nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readelf = args.readelf
    if readelf is None:
        candidate = shutil.which("readelf")
        if candidate is None:
            raise CompatibilityError("readelf is unavailable")
        readelf = pathlib.Path(candidate).resolve(strict=True)
    report = audit(
        root=args.root,
        paths=args.paths,
        maximum=args.maximum,
        readelf=readelf,
    )
    _write_new(args.report, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CompatibilityError, OSError, subprocess.SubprocessError) as error:
        print(f"gpmeep GLIBC compatibility FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
