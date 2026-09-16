#!/usr/bin/env python3
"""Compile deterministic CHECKED_HASH pycs for relocated conda sources.

This helper is intentionally executed by the Python interpreter inside the
audited prefix.  It writes only to a caller-owned private output directory;
installation and audit policy remain the responsibility of the protected
system-Python controllers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import py_compile
import re
import stat
import sys
from typing import Any


PYC_PATTERN = re.compile(
    r"(.+)/__pycache__/([^/]+)\.cpython-([0-9]+)(?:\.opt-([0-9]+))?\.pyc\Z"
)


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
        raise RuntimeError("relocated-bytecode path is missing")
    relative = pathlib.PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in value
    ):
        raise RuntimeError(f"unsafe relocated-bytecode path: {value!r}")
    return relative


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    parent = _canonical_directory(path.parent, "compiler report parent")
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"compiler report already exists: {path}")
    temporary = parent / f".{path.name}.tmp.{os.getpid()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    descriptor = os.open(temporary, flags, 0o600)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def compile_request(
    prefix: pathlib.Path, request_path: pathlib.Path, output: pathlib.Path
) -> dict[str, Any]:
    prefix = _canonical_directory(prefix, "conda prefix")
    request_path = _canonical_regular(request_path, "compiler request")
    output = _canonical_directory(output, "compiler output")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid relocated-bytecode compiler request") from error
    records = request.get("records") if isinstance(request, dict) else None
    if not isinstance(records, list) or not records:
        raise RuntimeError("relocated-bytecode compiler request is empty")
    expected_tag = sys.implementation.cache_tag
    if not isinstance(expected_tag, str) or not expected_tag.startswith("cpython-"):
        raise RuntimeError("unsupported Python cache tag")
    expected_digits = expected_tag.removeprefix("cpython-")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError("relocated-bytecode compiler record is invalid")
        target_relative = _safe_relative(record.get("path"))
        source_relative = _safe_relative(record.get("source_path"))
        target_text = target_relative.as_posix()
        if target_text in seen:
            raise RuntimeError("duplicate relocated-bytecode compiler path")
        seen.add(target_text)
        match = PYC_PATTERN.fullmatch(target_text)
        if (
            match is None
            or match.group(3) != expected_digits
            or source_relative.as_posix() != f"{match.group(1)}/{match.group(2)}.py"
        ):
            raise RuntimeError(f"bytecode/source mapping is invalid: {target_text}")
        source = _canonical_regular(prefix.joinpath(*source_relative.parts), "relocated source")
        source_bytes = source.read_bytes()
        source_before = source.stat()
        compiled = output / f"{index:08d}.pyc"
        optimization = int(match.group(4)) if match.group(4) is not None else -1
        py_compile.compile(
            os.fspath(source),
            cfile=os.fspath(compiled),
            dfile=os.fspath(source),
            doraise=True,
            optimize=optimization,
            invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
        )
        compiled = _canonical_regular(compiled, "compiled relocated bytecode")
        payload = compiled.read_bytes()
        if len(payload) < 16:
            raise RuntimeError(f"compiled bytecode is truncated: {target_text}")
        flags = int.from_bytes(payload[4:8], "little")
        source_hash = importlib.util.source_hash(source_bytes)
        if (
            payload[:4] != importlib.util.MAGIC_NUMBER
            or flags != 3
            or payload[8:16] != source_hash
        ):
            raise RuntimeError(f"compiled bytecode header is invalid: {target_text}")
        source_after = source.stat()
        if (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_mode,
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
        ) != (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_mode,
            source_after.st_size,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        ) or hashlib.sha256(source.read_bytes()).hexdigest() != hashlib.sha256(source_bytes).hexdigest():
            raise RuntimeError(f"relocated source changed during compilation: {target_text}")
        result.append(
            {
                "cache_tag": expected_tag,
                "compiled_file": compiled.name,
                "magic_hex": importlib.util.MAGIC_NUMBER.hex(),
                "optimization": optimization,
                "path": target_text,
                "pyc_sha256": _sha256(compiled),
                "pyc_size_bytes": compiled.stat().st_size,
                "source_hash_hex": source_hash.hex(),
                "source_path": source_relative.as_posix(),
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )
    return {
        "schema": "gpmeep-conda-relocated-bytecode-compilation-v1",
        "cache_tag": expected_tag,
        "magic_hex": importlib.util.MAGIC_NUMBER.hex(),
        "records": result,
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prefix", required=True, type=pathlib.Path)
    parser.add_argument("--request", required=True, type=pathlib.Path)
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument("--report", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        report = compile_request(args.prefix, args.request, args.output_directory)
        _write_json(args.report, report)
    except (OSError, RuntimeError, ValueError, py_compile.PyCompileError) as error:
        print(f"relocated bytecode compilation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
