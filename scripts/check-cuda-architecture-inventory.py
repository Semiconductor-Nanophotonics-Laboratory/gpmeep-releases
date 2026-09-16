#!/usr/bin/env python3
"""Prove that a gpmeep CUDA ELF contains the requested native and PTX targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
from typing import Any


SCHEMA = "gpmeep-cuda-architecture-inventory-v1"
_LISTING_RECORDS = {
    "elf": re.compile(r"^ELF file\s+[0-9]+:\s*(\S.*)$"),
    "ptx": re.compile(r"^PTX file\s+[0-9]+:\s*(\S.*)$"),
}
_ARCHITECTURE = re.compile(r"(?:^|[._-])sm_([0-9]{2,3})(?=$|[._-])")


class InventoryError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_expected(value: str) -> list[int]:
    if value == "":
        return []
    if re.fullmatch(r"[0-9]{2,3}(?:,[0-9]{2,3})*", value) is None:
        raise InventoryError(f"invalid expected architecture list {value!r}")
    result = [int(item) for item in value.split(",")]
    if any(item < 60 for item in result) or len(result) != len(set(result)):
        raise InventoryError("expected architectures must be unique and >= 60")
    return sorted(result)


def parse_listing(output: str, kind: str) -> tuple[list[int], int]:
    try:
        record_pattern = _LISTING_RECORDS[kind]
    except KeyError as error:
        raise InventoryError(f"unsupported CUDA listing kind {kind!r}") from error
    architectures: set[int] = set()
    count = 0
    for line in output.splitlines():
        match = record_pattern.fullmatch(line.strip())
        if match is None:
            continue
        count += 1
        architecture = _ARCHITECTURE.search(match.group(1))
        if architecture is None:
            raise InventoryError(
                f"{kind.upper()} listing record has no sm_NN architecture: "
                f"{match.group(1)!r}"
            )
        architectures.add(int(architecture.group(1)))
    return sorted(architectures), count


def _run(cuobjdump: pathlib.Path, option: str, binary: pathlib.Path) -> str:
    completed = subprocess.run(
        [str(cuobjdump), option, str(binary)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise InventoryError(f"cuobjdump {option} failed with rc={completed.returncode}")
    return completed.stdout


def _version(cuobjdump: pathlib.Path) -> str:
    completed = subprocess.run(
        [str(cuobjdump), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise InventoryError(
            f"cuobjdump --version failed with rc={completed.returncode}"
        )
    version_lines = [
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    ]
    if not version_lines:
        raise InventoryError("cuobjdump emitted no version identity")
    return " ".join(version_lines)


def _regular_binary(path: pathlib.Path, root: pathlib.Path) -> tuple[pathlib.Path, str]:
    absolute = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise InventoryError(f"CUDA binary is unavailable: {absolute}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise InventoryError(f"CUDA binary is not regular: {absolute}")
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise InventoryError(f"CUDA binary escapes audit root: {absolute}") from error
    if not resolved.is_file():
        raise InventoryError(f"CUDA binary is not regular: {absolute}")
    with resolved.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise InventoryError(f"CUDA binary is not ELF: {absolute}")
    return resolved, relative


def audit(
    *,
    root: pathlib.Path,
    binary: pathlib.Path,
    cuobjdump: pathlib.Path,
    expected_real: list[int],
    expected_virtual: list[int],
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise InventoryError(f"audit root is not a directory: {root}")
    if not cuobjdump.is_absolute() or not os.access(cuobjdump, os.X_OK):
        raise InventoryError(f"cuobjdump is not an executable absolute path: {cuobjdump}")
    binary, relative = _regular_binary(binary, root)
    elf_output = _run(cuobjdump, "--list-elf", binary)
    ptx_output = _run(cuobjdump, "--list-ptx", binary)
    actual_real, elf_records = parse_listing(elf_output, "elf")
    actual_virtual, ptx_records = parse_listing(ptx_output, "ptx")
    if actual_real != expected_real:
        raise InventoryError(
            f"native CUDA inventory {actual_real} differs from expected {expected_real}"
        )
    if actual_virtual != expected_virtual:
        raise InventoryError(
            f"PTX CUDA inventory {actual_virtual} differs from expected {expected_virtual}"
        )
    if not actual_real or elf_records == 0:
        raise InventoryError("CUDA binary contains no native SASS inventory")
    return {
        "schema": SCHEMA,
        "state": "PASS",
        "cuda_binary": {
            "path": relative,
            "size_bytes": binary.stat().st_size,
            "sha256": sha256_file(binary),
        },
        "cuobjdump_version": _version(cuobjdump),
        "real_architectures": actual_real,
        "virtual_architectures": actual_virtual,
        "elf_record_count": elf_records,
        "ptx_record_count": ptx_records,
        "elf_listing_sha256": hashlib.sha256(elf_output.encode("utf-8")).hexdigest(),
        "ptx_listing_sha256": hashlib.sha256(ptx_output.encode("utf-8")).hexdigest(),
    }


def _write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    path = pathlib.Path(os.path.abspath(os.fspath(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
    except FileExistsError as error:
        raise InventoryError(f"refusing to overwrite CUDA inventory: {path}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--cuobjdump", type=pathlib.Path, required=True)
    parser.add_argument("--expected-real", required=True)
    parser.add_argument("--expected-virtual", required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(
        root=args.root,
        binary=args.binary,
        cuobjdump=args.cuobjdump,
        expected_real=parse_expected(args.expected_real),
        expected_virtual=parse_expected(args.expected_virtual),
    )
    _write_new(args.report, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InventoryError, OSError, subprocess.SubprocessError) as error:
        print(f"gpmeep CUDA architecture inventory FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
