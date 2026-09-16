#!/usr/bin/env python3
"""Prove ordinary imports do not rewrite normalized relocated-source pycs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import stat
import sys
from typing import Any


def _canonical_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        info = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is absent: {lexical}") from error
    if not stat.S_ISDIR(info.st_mode) or resolved != lexical:
        raise RuntimeError(f"{label} is not a canonical directory: {lexical}")
    return lexical


def _canonical_regular(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        info = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is absent: {lexical}") from error
    if not stat.S_ISREG(info.st_mode) or resolved != lexical:
        raise RuntimeError(f"{label} is not a canonical regular file: {lexical}")
    return lexical


def _safe_relative(value: Any) -> pathlib.PurePosixPath:
    if not isinstance(value, str):
        raise RuntimeError("normalization path is missing")
    relative = pathlib.PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in value
    ):
        raise RuntimeError(f"unsafe normalization path: {value!r}")
    return relative


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    parent = _canonical_directory(path.parent, "probe report parent")
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"probe report already exists: {path}")
    temporary = parent / f".{path.name}.tmp.{os.getpid()}"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def probe(prefix: pathlib.Path, normalization_path: pathlib.Path) -> dict[str, Any]:
    prefix = _canonical_directory(prefix, "conda prefix")
    normalization_path = _canonical_regular(
        normalization_path, "relocated-bytecode normalization"
    )
    try:
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("relocated-bytecode normalization is invalid") from error
    records = normalization.get("records") if isinstance(normalization, dict) else None
    if (
        normalization.get("schema")
        != "gpmeep-conda-relocated-bytecode-normalization-v1"
        or normalization.get("environment_prefix") != str(prefix)
        or normalization.get("pass") is not True
        or not isinstance(records, list)
        or len(records) != normalization.get("normalized_count")
        or not records
    ):
        raise RuntimeError("relocated-bytecode normalization is invalid")
    before: dict[str, tuple[int, str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("relocated-bytecode normalization record is invalid")
        relative = _safe_relative(record.get("path"))
        path = _canonical_regular(prefix.joinpath(*relative.parts), "normalized bytecode")
        digest = _sha256(path)
        if (
            digest != record.get("after_sha256")
            or path.stat().st_size != record.get("after_size_bytes")
        ):
            raise RuntimeError(f"normalized bytecode changed before import: {relative}")
        before[relative.as_posix()] = (path.stat().st_size, digest)

    imported = [
        "contourpy.util._build_config",
        "numpy.__config__",
        "scipy.__config__",
        "_sysconfigdata__linux_x86_64-linux-gnu",
        "_sysconfigdata_x86_64_conda_linux_gnu",
    ]
    for name in imported:
        importlib.import_module(name)

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    python_config_candidates = sorted(
        prefix.glob(f"lib/python{python_version}/config-{python_version}-*/python-config.py")
    )
    if len(python_config_candidates) != 1:
        raise RuntimeError("python-config module path is ambiguous")
    python_config = _canonical_regular(
        python_config_candidates[0], "python-config module"
    )
    specification = importlib.util.spec_from_file_location(
        "_gpmeep_python_config_probe", python_config
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load python-config module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    imported.append("python-config.py")

    after: list[dict[str, Any]] = []
    for relative, (size, digest) in sorted(before.items()):
        path = _canonical_regular(
            prefix.joinpath(*pathlib.PurePosixPath(relative).parts),
            "normalized bytecode",
        )
        if path.stat().st_size != size or _sha256(path) != digest:
            raise RuntimeError(f"ordinary import rewrote normalized bytecode: {relative}")
        after.append({"path": relative, "sha256": digest, "size_bytes": size})
    return {
        "schema": "gpmeep-conda-relocated-bytecode-import-stability-v1",
        "bytecode": after,
        "cache_tag": sys.implementation.cache_tag,
        "imports": imported,
        "ordinary_import": True,
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "python_pycache_prefix": sys.pycache_prefix,
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prefix", required=True, type=pathlib.Path)
    parser.add_argument("--normalization", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        result = probe(args.prefix, args.normalization)
        if result["python_dont_write_bytecode"] or result["python_pycache_prefix"] is not None:
            raise RuntimeError("probe was not launched with ordinary bytecode behavior")
        _write_json(args.output, result)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"relocated bytecode import probe error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
