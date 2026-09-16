#!/usr/bin/env python3
"""Replace invalid archive pycs whose conda-owned sources were relocated."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
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


def _fingerprint(path: pathlib.Path) -> tuple[int, ...]:
    info = path.lstat()
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    destination = pathlib.Path(os.path.abspath(os.fspath(path)))
    parent = _canonical_directory(destination.parent, "normalization report parent")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"normalization report already exists: {destination}")
    temporary = parent / f".{destination.name}.tmp.{os.getpid()}"
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
    os.replace(temporary, destination)


def normalize(
    prefix: pathlib.Path,
    audit_path: pathlib.Path,
    compiler: pathlib.Path,
) -> dict[str, Any]:
    prefix = _canonical_directory(prefix, "conda prefix")
    audit_path = _canonical_regular(audit_path, "pre-normalization prefix audit")
    compiler = _canonical_regular(compiler, "relocated-bytecode compiler")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("pre-normalization prefix audit is invalid") from error
    relocated_records = (
        audit.get("relocated_source_bytecode") if isinstance(audit, dict) else None
    )
    generated_records = (
        audit.get("generated_source_bytecode") if isinstance(audit, dict) else None
    )
    if (
        audit.get("schema_version") != 4
        or audit.get("pass") is not True
        or audit.get("environment_prefix") != str(prefix)
        or not isinstance(relocated_records, list)
        or not relocated_records
        or not isinstance(generated_records, list)
        or not generated_records
    ):
        raise RuntimeError("pre-normalization prefix audit has no bytecode policy")
    records = [
        *((record, "relocated") for record in relocated_records),
        *((record, "generated") for record in generated_records),
    ]
    records.sort(key=lambda item: item[0].get("path", ""))
    request_records: list[dict[str, str]] = []
    before: dict[str, dict[str, Any]] = {}
    for record, ownership in records:
        required_state = "archive-exact" if ownership == "relocated" else "absent"
        if not isinstance(record, dict) or record.get("current_state") != required_state:
            raise RuntimeError("pre-normalization bytecode is not archive-exact")
        relative = _safe_relative(record.get("path"))
        source_relative = _safe_relative(record.get("source_path"))
        source = _canonical_regular(
            prefix.joinpath(*source_relative.parts), "relocated bytecode source"
        )
        path = prefix.joinpath(*relative.parts)
        if ownership == "relocated":
            path = _canonical_regular(path, "archive bytecode")
            expected_before_digest = record.get("archive_sha256")
            if _sha256(path) != expected_before_digest:
                raise RuntimeError(f"pre-normalization bytes changed: {relative.as_posix()}")
            path_fingerprint: tuple[int, ...] | None = _fingerprint(path)
            path_mode = stat.S_IMODE(path.stat().st_mode)
            path_size = path.stat().st_size
        else:
            if path.exists() or path.is_symlink():
                raise RuntimeError(f"generated bytecode appeared before normalization: {relative}")
            if path.parent.exists() or path.parent.is_symlink():
                _canonical_directory(path.parent, "generated bytecode parent")
            else:
                _canonical_directory(
                    path.parent.parent, "generated bytecode source directory"
                )
            expected_before_digest = None
            path_fingerprint = None
            path_mode = 0o644
            path_size = 0
        if _sha256(source) != record.get("installed_source_sha256"):
            raise RuntimeError(f"pre-normalization bytes changed: {relative.as_posix()}")
        request_records.append(
            {"path": relative.as_posix(), "source_path": source_relative.as_posix()}
        )
        before[relative.as_posix()] = {
            "fingerprint": path_fingerprint,
            "mode": path_mode,
            "ownership": ownership,
            "sha256": expected_before_digest,
            "size_bytes": path_size,
            "source_fingerprint": _fingerprint(source),
            "source_sha256": record["installed_source_sha256"],
        }

    python_version = audit.get("python_version")
    if not isinstance(python_version, str):
        raise RuntimeError("pre-normalization prefix audit omits Python version")
    python = _canonical_regular(prefix / f"bin/python{python_version}", "prefix Python")
    with tempfile.TemporaryDirectory(
        prefix=".gpmeep-relocated-pyc-", dir=prefix.parent
    ) as temporary_text:
        temporary = pathlib.Path(temporary_text)
        compiled = temporary / "compiled"
        compiled.mkdir(mode=0o700)
        request = temporary / "request.json"
        request.write_text(
            json.dumps({"records": request_records}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        compiler_report = temporary / "compiler-report.json"
        completed = subprocess.run(
            [
                os.fspath(python),
                "-I",
                "-S",
                "-B",
                os.fspath(compiler),
                "--prefix",
                os.fspath(prefix),
                "--request",
                os.fspath(request),
                "--output-directory",
                os.fspath(compiled),
                "--report",
                os.fspath(compiler_report),
            ],
            cwd=prefix.parent,
            env={
                "HOME": "/tmp",
                "PATH": f"{prefix}/bin:/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": "/dev/null",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                "relocated-bytecode compiler failed"
                + (f": {diagnostic}" if diagnostic else "")
            )
        try:
            compilation = json.loads(compiler_report.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("relocated-bytecode compiler report is invalid") from error
        compiled_records = compilation.get("records") if isinstance(compilation, dict) else None
        if (
            compilation.get("schema")
            != "gpmeep-conda-relocated-bytecode-compilation-v1"
            or compilation.get("pass") is not True
            or not isinstance(compiled_records, list)
            or len(compiled_records) != len(request_records)
        ):
            raise RuntimeError("relocated-bytecode compiler report is invalid")

        normalized: list[dict[str, Any]] = []
        created_cache_directories: set[str] = set()
        for expected_pair, compiled_record in zip(records, compiled_records):
            expected, ownership = expected_pair
            if (
                not isinstance(compiled_record, dict)
                or compiled_record.get("path") != expected.get("path")
                or compiled_record.get("source_path") != expected.get("source_path")
                or compiled_record.get("source_sha256")
                != expected.get("installed_source_sha256")
                or compiled_record.get("pyc_sha256")
                != expected.get("derived_sha256")
                or compiled_record.get("source_hash_hex")
                != expected.get("installed_source_hash_hex")
            ):
                raise RuntimeError("compiled bytecode differs from the audited derivation")
            generated = _canonical_regular(
                compiled / str(compiled_record.get("compiled_file")),
                "compiled relocated bytecode",
            )
            if (
                _sha256(generated) != compiled_record.get("pyc_sha256")
                or generated.stat().st_size != compiled_record.get("pyc_size_bytes")
            ):
                raise RuntimeError("compiled relocated bytecode changed")
            target_relative = _safe_relative(compiled_record["path"])
            target = prefix.joinpath(*target_relative.parts)
            old = before[target_relative.as_posix()]
            source = _canonical_regular(
                prefix.joinpath(*_safe_relative(compiled_record["source_path"]).parts),
                "relocated source",
            )
            if old["ownership"] == "relocated":
                target = _canonical_regular(target, "archive bytecode")
                target_unchanged = (
                    _fingerprint(target) == old["fingerprint"]
                    and _sha256(target) == old["sha256"]
                )
            else:
                target_unchanged = not target.exists() and not target.is_symlink()
            if not target_unchanged or (
                _fingerprint(source) != old["source_fingerprint"]
                or _sha256(source) != old["source_sha256"]
            ):
                raise RuntimeError(f"prefix changed before normalization: {target_relative}")
            if old["ownership"] == "generated" and not target.parent.exists():
                source_directory = _canonical_directory(
                    target.parent.parent, "generated bytecode source directory"
                )
                if target.parent != source_directory / "__pycache__":
                    raise RuntimeError(
                        f"generated bytecode cache path is invalid: {target_relative}"
                    )
                target.parent.mkdir(mode=0o755)
                _canonical_directory(target.parent, "generated bytecode parent")
                created_cache_directories.add(
                    target.parent.relative_to(prefix).as_posix()
                )
            else:
                _canonical_directory(target.parent, "bytecode parent")
            os.chmod(generated, old["mode"])
            os.replace(generated, target)
            target = _canonical_regular(target, "normalized relocated bytecode")
            if (
                _sha256(target) != compiled_record["pyc_sha256"]
                or target.stat().st_size != compiled_record["pyc_size_bytes"]
            ):
                raise RuntimeError(f"relocated bytecode replacement failed: {target_relative}")
            normalized.append(
                {
                    "after_sha256": compiled_record["pyc_sha256"],
                    "after_size_bytes": compiled_record["pyc_size_bytes"],
                    "before_sha256": old["sha256"],
                    "before_size_bytes": old["size_bytes"],
                    "cache_tag": compiled_record["cache_tag"],
                    "installed_source_hash_hex": compiled_record["source_hash_hex"],
                    "installed_source_sha256": compiled_record["source_sha256"],
                    "ownership": ownership,
                    "path": target_relative.as_posix(),
                    "source_path": compiled_record["source_path"],
                }
            )
    return {
        "schema": "gpmeep-conda-relocated-bytecode-normalization-v1",
        "environment_prefix": str(prefix),
        "created_cache_directories": sorted(created_cache_directories),
        "created_cache_directory_count": len(created_cache_directories),
        "normalized_count": len(normalized),
        "records": normalized,
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prefix", required=True, type=pathlib.Path)
    parser.add_argument("--pre-audit", required=True, type=pathlib.Path)
    parser.add_argument("--compiler", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        report = normalize(args.prefix, args.pre_audit, args.compiler)
        _write_json(args.output, report)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"relocated bytecode normalization error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
