#!/usr/bin/env python3
"""Audit installed conda-prefix bytes after an exact-lock force reinstall."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
import zipfile
from typing import Any, BinaryIO, Iterator


def _load_provenance_from_source() -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().with_name("gpmeep_provenance.py")
    module = types.ModuleType("gpmeep_provenance")
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


_PROVENANCE = _load_provenance_from_source()
atomic_write_json = _PROVENANCE.atomic_write_json
StatHashCache = _PROVENANCE.StatHashCache

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ZSTD = pathlib.Path("/usr/bin/zstd")
ZSTD_TIMEOUT_SECONDS = 120
PYC_HARDLINK_PATTERN = re.compile(
    r"(.+)/__pycache__/([^/]+)\.cpython-([0-9]+)(?:\.opt-[0-9]+)?\.pyc\Z"
)


class _OpenedExactArchive:
    """One pathname-bound archive descriptor retained for the whole audit.

    Parsing never reopens the package-cache pathname.  Instead, one private
    snapshot is populated and hashed from this descriptor, and both metadata
    and payload are decoded from those exact snapshot bytes.  This closes the
    hash-then-reopen race even if an attacker can atomically rename cache
    entries while the audit is running.
    """

    def __init__(
        self,
        path: pathlib.Path,
        descriptor: int,
        fingerprint: tuple[int, ...],
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.fingerprint = fingerprint

    def _validate(self, context: str) -> None:
        try:
            opened = os.fstat(self.descriptor)
        except OSError as error:
            raise RuntimeError(
                f"exact-lock package archive descriptor changed {context}: "
                f"{self.path}"
            ) from error
        if (
            StatHashCache._fingerprint(opened) != self.fingerprint
            or not StatHashCache._path_matches_fingerprint(
                self.path, self.fingerprint
            )
        ):
            raise RuntimeError(
                f"exact-lock package archive changed {context}: {self.path.name}"
            )

    @contextlib.contextmanager
    def trusted_snapshot(self, expected_digest: str) -> Iterator[BinaryIO]:
        self._validate("before snapshotting")
        digest = hashlib.sha256()
        offset = 0
        with tempfile.TemporaryFile(mode="w+b") as snapshot:
            while True:
                try:
                    block = os.pread(self.descriptor, 1024 * 1024, offset)
                except OSError as error:
                    raise RuntimeError(
                        "cannot snapshot exact-lock package archive: "
                        + self.path.name
                    ) from error
                if not block:
                    break
                snapshot.write(block)
                digest.update(block)
                offset += len(block)
            self._validate("while snapshotting")
            if digest.hexdigest() != expected_digest:
                raise RuntimeError(
                    "exact-lock package archive changed: " + self.path.name
                )
            snapshot.flush()
            snapshot.seek(0)
            yield snapshot

    def verify_terminal(self, expected_digest: str) -> None:
        self._validate("before terminal verification")
        digest = hashlib.sha256()
        offset = 0
        while True:
            try:
                block = os.pread(self.descriptor, 1024 * 1024, offset)
            except OSError as error:
                raise RuntimeError(
                    "cannot recheck exact-lock package archive: " + self.path.name
                ) from error
            if not block:
                break
            digest.update(block)
            offset += len(block)
        self._validate("during terminal verification")
        if digest.hexdigest() != expected_digest:
            raise RuntimeError(
                "exact-lock package archive changed during the audit: "
                + self.path.name
            )


@contextlib.contextmanager
def _open_exact_archive(path: pathlib.Path) -> Iterator[_OpenedExactArchive]:
    lexical = _canonical_regular(path, "exact-lock package archive")
    initial = lexical.lstat()
    fingerprint = StatHashCache._fingerprint(initial)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_CLOEXEC"):
        raise RuntimeError(
            "exact-lock package archive audit requires O_NOFOLLOW and O_CLOEXEC"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, flags)
    except OSError as error:
        raise RuntimeError(
            f"exact-lock package archive changed before opening: {lexical}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or StatHashCache._fingerprint(opened) != fingerprint
            or not StatHashCache._path_matches_fingerprint(lexical, fingerprint)
        ):
            raise RuntimeError(
                f"exact-lock package archive changed before opening: {lexical}"
            )
        yield _OpenedExactArchive(lexical, descriptor, fingerprint)
    finally:
        os.close(descriptor)


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


def _package_archive_path(package_cache: pathlib.Path, url: str) -> pathlib.Path:
    scheme, separator, remainder = url.partition("://")
    parts = remainder.split("/")
    if (
        not separator
        or re.fullmatch(r"[a-z][a-z0-9+.-]*", scheme) is None
        or len(parts) < 2
        or any(not part or part in {".", ".."} for part in parts)
        or any(character in remainder for character in "?#\\")
    ):
        raise RuntimeError("exact lock contains an unmappable package URL")
    return _canonical_regular(
        package_cache.joinpath(scheme, *parts), "exact-lock package archive"
    )


def _archive_relative(value: str, label: str) -> str:
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError(f"exact-lock package archive contains an unsafe {label}")
    text = relative.as_posix()
    if text.startswith("./"):
        text = text[2:]
    if not text or text == ".":
        raise RuntimeError(f"exact-lock package archive contains an unsafe {label}")
    return text


@contextlib.contextmanager
def _conda_tar_segment(
    archive: BinaryIO, archive_name: str, segment_prefix: str
) -> Iterator[tarfile.TarFile]:
    compressed_name: str | None = None
    tar_name: str | None = None
    try:
        archive.seek(0)
        with zipfile.ZipFile(archive) as container:
            names = container.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError("exact-lock .conda archive has duplicate members")
            segments = [
                name
                for name in names
                if pathlib.PurePosixPath(name).name.startswith(segment_prefix + "-")
                and name.endswith(".tar.zst")
            ]
            if len(segments) != 1:
                raise RuntimeError(
                    f"exact-lock .conda archive has no unique {segment_prefix} segment"
                )
            compressed_fd, compressed_name = tempfile.mkstemp(
                prefix=f"gpmeep-conda-{segment_prefix}-", suffix=".tar.zst"
            )
            with os.fdopen(compressed_fd, "wb") as destination:
                with container.open(segments[0]) as source:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
        tar_fd, tar_name = tempfile.mkstemp(
            prefix=f"gpmeep-conda-{segment_prefix}-", suffix=".tar"
        )
        os.close(tar_fd)
        try:
            completed = subprocess.run(
                [
                    os.fspath(ZSTD),
                    "-q",
                    "-d",
                    "-f",
                    "-o",
                    tar_name,
                    "--",
                    compressed_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=ZSTD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"exact-lock .conda {segment_prefix} decompression exceeded "
                f"{ZSTD_TIMEOUT_SECONDS} seconds: {archive_name}"
            ) from error
        if completed.returncode != 0:
            diagnostic = completed.stderr[:4096].decode(
                "utf-8", errors="replace"
            ).strip()
            raise RuntimeError(
                f"cannot decompress exact-lock .conda {segment_prefix} "
                f"segment {archive_name} (status {completed.returncode}): "
                + diagnostic
            )
        with tarfile.open(tar_name, mode="r:") as payload:
            yield payload
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise RuntimeError(
            f"cannot inspect exact-lock package archive: {archive_name}"
        ) from error
    finally:
        for temporary_name in (compressed_name, tar_name):
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass


def _tar_json(payload: tarfile.TarFile, name: str, required: bool) -> Any:
    try:
        member = payload.getmember(name)
    except KeyError:
        if required:
            raise RuntimeError(f"exact-lock package archive omits {name}")
        return None
    if not member.isfile():
        raise RuntimeError(f"exact-lock package archive has invalid {name}")
    source = payload.extractfile(member)
    if source is None:
        raise RuntimeError(f"exact-lock package archive has unreadable {name}")
    try:
        return json.loads(source.read().decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"exact-lock package archive has invalid {name}") from error


def _parse_archive_info(payload: tarfile.TarFile) -> dict[str, Any]:
    paths_data = _tar_json(payload, "info/paths.json", True)
    index = _tar_json(payload, "info/index.json", True)
    link = _tar_json(payload, "info/link.json", False)
    paths = paths_data.get("paths") if isinstance(paths_data, dict) else None
    if not isinstance(paths, list) or not isinstance(index, dict):
        raise RuntimeError("exact-lock package archive metadata is invalid")
    records: dict[str, dict[str, Any]] = {}
    for record in paths:
        if not isinstance(record, dict) or not isinstance(record.get("_path"), str):
            raise RuntimeError("exact-lock package path record is invalid")
        relative = _archive_relative(record["_path"], "package path")
        if relative in records:
            raise RuntimeError("exact-lock package archive has duplicate path metadata")
        path_type = record.get("path_type")
        if path_type not in {"hardlink", "softlink"}:
            raise RuntimeError("exact-lock package archive has unsupported path metadata")
        if path_type == "hardlink":
            digest = record.get("sha256")
            size = record.get("size_in_bytes")
            if (
                not isinstance(digest, str)
                or SHA256_PATTERN.fullmatch(digest) is None
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
            ):
                raise RuntimeError("exact-lock package hardlink metadata is invalid")
            placeholder = record.get("prefix_placeholder")
            mode = record.get("file_mode")
            if placeholder is not None and (
                not isinstance(placeholder, str) or mode not in {"text", "binary"}
            ):
                raise RuntimeError("exact-lock package prefix metadata is invalid")
            if placeholder is None and mode is not None:
                raise RuntimeError("exact-lock package file mode lacks a placeholder")
        records[relative] = dict(record)
    if link is not None and not isinstance(link, dict):
        raise RuntimeError("exact-lock package link metadata is invalid")
    for key in ("name", "version", "build"):
        if not isinstance(index.get(key), str) or not index[key]:
            raise RuntimeError("exact-lock package index metadata is invalid")
    return {"paths": records, "index": index, "link": link or {}}


def _package_archive_info(
    archive: BinaryIO, archive_name: str
) -> dict[str, Any]:
    if archive_name.endswith(".conda"):
        with _conda_tar_segment(archive, archive_name, "info") as payload:
            return _parse_archive_info(payload)
    if archive_name.endswith(".tar.bz2"):
        try:
            archive.seek(0)
            with tarfile.open(fileobj=archive, mode="r:bz2") as payload:
                return _parse_archive_info(payload)
        except (OSError, tarfile.TarError) as error:
            raise RuntimeError(
                f"cannot inspect exact-lock package archive: {archive_name}"
            ) from error
    raise RuntimeError(f"unsupported exact-lock package archive: {archive_name}")


def _hash_relocated_payload(
    source: BinaryIO, record: dict[str, Any], prefix: pathlib.Path
) -> tuple[str, str, int]:
    raw = hashlib.sha256()
    installed = hashlib.sha256()
    placeholder = record.get("prefix_placeholder")
    old = placeholder.encode("utf-8") if isinstance(placeholder, str) else None
    replacement: bytes | None = None
    if old is not None:
        prefix_bytes = os.fsencode(prefix)
        if len(prefix_bytes) > len(old):
            raise RuntimeError("conda prefix exceeds an exact package placeholder")
        replacement = prefix_bytes
    pending = b""
    replacement_count = 0
    size = 0
    while True:
        block = source.read(1024 * 1024)
        if not block:
            break
        size += len(block)
        raw.update(block)
        if old is None:
            installed.update(block)
            continue
        pending += block
        if record.get("file_mode") == "binary":
            segments = pending.split(b"\0")
            pending = segments.pop()
            transformed = bytearray()
            for segment in segments:
                count = segment.count(old)
                transformed.extend(segment.replace(old, replacement))
                transformed.extend(
                    b"\0" * (1 + count * (len(old) - len(replacement)))
                )
                replacement_count += count
            installed.update(transformed)
            continue
        while True:
            location = pending.find(old)
            if location >= 0:
                installed.update(pending[:location])
                assert replacement is not None
                installed.update(replacement)
                pending = pending[location + len(old) :]
                replacement_count += 1
            else:
                retained = max(0, len(old) - 1)
                flush = len(pending) - retained
                if flush > 0:
                    installed.update(pending[:flush])
                    pending = pending[flush:]
                break
    if old is not None:
        if record.get("file_mode") == "binary":
            count = pending.count(old)
            installed.update(pending.replace(old, replacement))
            installed.update(b"\0" * (count * (len(old) - len(replacement))))
            replacement_count += count
        else:
            installed.update(pending)
        if replacement_count == 0:
            raise RuntimeError("exact package placeholder is absent from its payload")
    return raw.hexdigest(), installed.hexdigest(), size


def _selected_package_payload(
    archive: BinaryIO,
    archive_name: str,
    records: dict[str, dict[str, Any]],
    prefix: pathlib.Path,
) -> dict[str, dict[str, Any]]:
    selected = {
        path: record
        for path, record in records.items()
        if record.get("path_type") == "softlink"
        or record.get("prefix_placeholder") is not None
    }
    if not selected:
        return {}

    @contextlib.contextmanager
    def open_payload() -> Iterator[tarfile.TarFile]:
        if archive_name.endswith(".conda"):
            with _conda_tar_segment(archive, archive_name, "pkg") as payload:
                yield payload
        else:
            try:
                archive.seek(0)
                with tarfile.open(fileobj=archive, mode="r:bz2") as payload:
                    yield payload
            except (OSError, tarfile.TarError) as error:
                raise RuntimeError(
                    f"cannot inspect exact-lock package archive: {archive_name}"
                ) from error

    result: dict[str, dict[str, Any]] = {}
    with open_payload() as payload:
        members: dict[str, tarfile.TarInfo] = {}
        for member in payload.getmembers():
            relative = _archive_relative(member.name, "payload path")
            if relative in members:
                raise RuntimeError("exact-lock package payload has a duplicate path")
            members[relative] = member
        for relative, record in selected.items():
            member = members.get(relative)
            if member is None:
                raise RuntimeError(
                    f"exact-lock package payload omits {relative}"
                )
            if record.get("path_type") == "softlink":
                if not member.issym() or os.path.isabs(member.linkname):
                    raise RuntimeError(
                        f"exact-lock package payload has invalid symlink {relative}"
                    )
                result[relative] = {"link_target": member.linkname}
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"exact-lock package payload has invalid hardlink {relative}"
                )
            source = payload.extractfile(member)
            if source is None:
                raise RuntimeError(
                    f"exact-lock package payload has unreadable hardlink {relative}"
                )
            raw_digest, installed_digest, size = _hash_relocated_payload(
                source, record, prefix
            )
            if raw_digest != record["sha256"] or size != record["size_in_bytes"]:
                raise RuntimeError(
                    f"exact-lock package payload differs from path metadata: {relative}"
                )
            result[relative] = {
                "raw_sha256": raw_digest,
                "installed_sha256": installed_digest,
            }
    return result


def _installed_archive_paths(
    archive_info: dict[str, Any], python_version: str
) -> dict[str, tuple[str, dict[str, Any]]]:
    noarch = archive_info["link"].get("noarch")
    python_noarch = isinstance(noarch, dict) and noarch.get("type") == "python"
    result: dict[str, tuple[str, dict[str, Any]]] = {}
    for archive_path, record in archive_info["paths"].items():
        installed_path = archive_path
        if python_noarch and archive_path.startswith("site-packages/"):
            installed_path = (
                f"lib/python{python_version}/site-packages/"
                + archive_path[len("site-packages/") :]
            )
        elif python_noarch and archive_path.startswith("python-scripts/"):
            installed_path = "bin/" + archive_path[len("python-scripts/") :]
        if installed_path in result:
            raise RuntimeError("exact-lock package paths collide after relocation")
        result[installed_path] = (archive_path, record)
    return result


def _entry_point_specifications(archive_info: dict[str, Any]) -> dict[str, tuple[str, str]]:
    noarch = archive_info["link"].get("noarch")
    if not isinstance(noarch, dict) or noarch.get("type") != "python":
        return {}
    values = noarch.get("entry_points", [])
    if not isinstance(values, list):
        raise RuntimeError("exact-lock package entry-point metadata is invalid")
    result: dict[str, tuple[str, str]] = {}
    pattern = re.compile(
        r"\s*([A-Za-z0-9_.-]+)\s*=\s*"
        r"([A-Za-z_][A-Za-z0-9_.]*)\s*:\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)\s*\Z"
    )
    for value in values:
        match = pattern.fullmatch(value) if isinstance(value, str) else None
        if match is None or match.group(1) in result:
            raise RuntimeError("exact-lock package entry-point specification is invalid")
        result[match.group(1)] = (match.group(2), match.group(3))
    return result


def _entry_point_bytes(
    prefix: pathlib.Path, python_version: str, module: str, function: str
) -> bytes:
    return (
        f"#!{prefix}/bin/python{python_version}\n"
        "# -*- coding: utf-8 -*-\n"
        "import re\n"
        "import sys\n\n"
        f"from {module} import {function}\n\n"
        "if __name__ == '__main__':\n"
        "    sys.argv[0] = re.sub(r'(-script\\.pyw?|\\.exe)?$', '', sys.argv[0])\n"
        f"    sys.exit({function}())\n"
    ).encode("utf-8")


def _compile_relocated_source_bytecode(
    prefix: pathlib.Path,
    python_version: str,
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not candidates:
        return {}
    compiler = _canonical_regular(
        pathlib.Path(__file__).resolve().with_name(
            "compile-conda-relocated-bytecode.py"
        ),
        "relocated-bytecode compiler",
    )
    python = _canonical_regular(
        prefix / f"bin/python{python_version}", "audited prefix Python"
    )
    with tempfile.TemporaryDirectory(prefix="gpmeep-prefix-pyc-audit-") as temporary_text:
        temporary = pathlib.Path(temporary_text)
        output = temporary / "compiled"
        output.mkdir(mode=0o700)
        request = temporary / "request.json"
        request.write_text(
            json.dumps(
                {
                    "records": [
                        {"path": item["path"], "source_path": item["source_path"]}
                        for item in candidates
                    ]
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report_path = temporary / "report.json"
        try:
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
                    os.fspath(output),
                    "--report",
                    os.fspath(report_path),
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
                check=False,
                timeout=120,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "relocated-source bytecode compilation exceeded 120 seconds"
            ) from error
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                "cannot derive relocated-source bytecode"
                + (f": {diagnostic}" if diagnostic else "")
            )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("relocated-source bytecode report is invalid") from error
    records = report.get("records") if isinstance(report, dict) else None
    if (
        report.get("schema")
        != "gpmeep-conda-relocated-bytecode-compilation-v1"
        or report.get("pass") is not True
        or not isinstance(records, list)
        or len(records) != len(candidates)
    ):
        raise RuntimeError("relocated-source bytecode report is invalid")
    result: dict[str, dict[str, Any]] = {}
    for expected, record in zip(candidates, records):
        if (
            not isinstance(record, dict)
            or record.get("path") != expected["path"]
            or record.get("source_path") != expected["source_path"]
            or record.get("source_sha256") != expected["installed_source_sha256"]
            or record.get("cache_tag") != f"cpython-{python_version.replace('.', '')}"
        ):
            raise RuntimeError("relocated-source bytecode derivation is inconsistent")
        result[record["path"]] = record
    return result


def _lock_packages(
    lock: pathlib.Path, hash_cache: Any
) -> tuple[dict[str, str], str]:
    result: dict[str, str] = {}
    payload = hash_cache.read_bytes(lock)
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise RuntimeError("exact lock is not valid UTF-8") from error
    for line in text.splitlines():
        value = line.strip()
        if "://" not in value:
            continue
        url, separator, digest = value.rpartition("#")
        if not separator or SHA256_PATTERN.fullmatch(digest) is None or url in result:
            raise RuntimeError("exact lock contains an invalid or duplicate package pin")
        result[url] = digest
    if not result:
        raise RuntimeError("exact lock contains no SHA-256 package pins")
    return result, hashlib.sha256(payload).hexdigest()


def _allowed_unowned(relative: str, metadata_paths: set[str]) -> bool:
    return (
        relative == ".messages.txt"
        or relative == "conda-meta/history"
        or relative in metadata_paths
        or relative.startswith("var/cache/fontconfig/")
    )


def _audit(
    prefix: pathlib.Path,
    lock: pathlib.Path,
    package_cache: pathlib.Path,
    archive_stack: contextlib.ExitStack,
) -> dict[str, Any]:
    prefix = pathlib.Path(os.path.abspath(os.fspath(prefix)))
    try:
        prefix_info = prefix.lstat()
        resolved_prefix = prefix.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("conda prefix is absent") from error
    if not stat.S_ISDIR(prefix_info.st_mode) or resolved_prefix != prefix:
        raise RuntimeError("conda prefix is not a canonical directory")
    hash_cache = StatHashCache()
    lock = _canonical_regular(lock, "exact lock")
    package_cache = _canonical_directory(package_cache, "conda package cache")
    zstd = _canonical_regular(ZSTD, "zstd decoder")
    zstd_digest = hash_cache.digest(zstd)
    try:
        zstd_version_process = subprocess.run(
            [os.fspath(zstd), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("zstd version probe exceeded 10 seconds") from error
    if zstd_version_process.returncode != 0:
        raise RuntimeError("zstd version probe failed")
    try:
        zstd_version = zstd_version_process.stdout.decode(
            "utf-8", errors="strict"
        ).strip()
    except UnicodeError as error:
        raise RuntimeError("zstd version output is not valid UTF-8") from error
    if not zstd_version:
        raise RuntimeError("zstd version output is empty")
    conda_meta = prefix / "conda-meta"
    try:
        conda_meta_info = conda_meta.lstat()
    except OSError as error:
        raise RuntimeError("conda prefix metadata is absent") from error
    if (
        not stat.S_ISDIR(conda_meta_info.st_mode)
        or conda_meta.resolve(strict=True) != conda_meta
    ):
        raise RuntimeError("conda prefix metadata is absent")
    locked, lock_digest = _lock_packages(lock, hash_cache)
    metadata_files = sorted(conda_meta.glob("*.json"))
    if len(metadata_files) != len(locked):
        raise RuntimeError("conda metadata package count differs from the exact lock")
    owned: dict[str, dict[str, Any]] = {}
    package_records: list[dict[str, str]] = []
    unhashed_bytecode = 0
    generated_entry_points: list[dict[str, Any]] = []
    softlinks: list[dict[str, str]] = []
    metadata_digests: dict[pathlib.Path, str] = {}
    hardlink_digests: dict[pathlib.Path, str] = {}
    entry_point_digests: dict[pathlib.Path, str] = {}
    archive_digests: dict[pathlib.Path, str] = {}
    archive_handles: dict[pathlib.Path, _OpenedExactArchive] = {}
    package_inputs: list[dict[str, Any]] = []
    for metadata_path in metadata_files:
        metadata_path = _canonical_regular(metadata_path, "conda metadata")
        try:
            metadata_payload = hash_cache.read_bytes(metadata_path)
            metadata = json.loads(metadata_payload.decode("utf-8", errors="strict"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid conda metadata: {metadata_path.name}") from error
        metadata_digests[metadata_path] = hashlib.sha256(metadata_payload).hexdigest()
        url = metadata.get("url")
        digest = metadata.get("sha256")
        if not isinstance(url, str) or locked.get(url) != digest:
            raise RuntimeError(
                f"conda metadata is not bound to the exact lock: {metadata_path.name}"
            )
        paths = metadata.get("paths_data", {}).get("paths")
        if not isinstance(paths, list):
            raise RuntimeError(f"conda paths metadata is absent: {metadata_path.name}")
        package_archive = _package_archive_path(package_cache, url)
        if package_archive in archive_handles:
            raise RuntimeError(
                "exact-lock package URLs collide at one canonical archive path"
            )
        opened_archive = archive_stack.enter_context(
            _open_exact_archive(package_archive)
        )
        with opened_archive.trusted_snapshot(digest) as trusted_archive:
            archive_info = _package_archive_info(
                trusted_archive, package_archive.name
            )
            selected_payload = _selected_package_payload(
                trusted_archive,
                package_archive.name,
                archive_info["paths"],
                prefix,
            )
        archive_digests[package_archive] = digest
        archive_handles[package_archive] = opened_archive
        index = archive_info["index"]
        for field in ("name", "version", "build"):
            if metadata.get(field) != index[field]:
                raise RuntimeError(
                    f"conda metadata identity differs from its exact archive: "
                    f"{metadata_path.name}"
                )
        package_records.append({"url": url, "sha256": digest})
        package_inputs.append(
            {
                "metadata_path": metadata_path,
                "metadata": metadata,
                "paths": paths,
                "archive": package_archive,
                "archive_info": archive_info,
                "selected_payload": selected_payload,
                "url": url,
                "digest": digest,
            }
        )

    python_versions: set[str] = set()
    for package in package_inputs:
        index = package["archive_info"]["index"]
        if index["name"] == "python":
            match = re.fullmatch(r"([0-9]+)\.([0-9]+)(?:\..*)?", index["version"])
            if match is None:
                raise RuntimeError("exact-lock Python package has an invalid version")
            python_versions.add(f"{match.group(1)}.{match.group(2)}")
    if len(python_versions) != 1:
        raise RuntimeError("exact lock does not identify one Python major/minor version")
    python_version = next(iter(python_versions))
    python_cache_tag = python_version.replace(".", "")
    archive_bound_hardlinks = 0
    relocated_hardlinks = 0
    relocated_source_bytecode_candidates: list[dict[str, Any]] = []
    generated_source_bytecode_candidates: list[dict[str, Any]] = []

    for package in package_inputs:
        metadata_path = package["metadata_path"]
        paths = package["paths"]
        package_archive = package["archive"]
        archive_info = package["archive_info"]
        digest = package["digest"]
        installed_archive_paths = _installed_archive_paths(
            archive_info, python_version
        )
        selected_payload = package["selected_payload"]
        metadata_by_path: dict[str, dict[str, Any]] = {}
        for record in paths:
            if not isinstance(record, dict) or not isinstance(record.get("_path"), str):
                raise RuntimeError("conda package path record is invalid")
            relative = pathlib.PurePosixPath(record["_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("conda package path escapes the environment prefix")
            relative_text = relative.as_posix()
            if relative_text in metadata_by_path or relative_text in owned:
                raise RuntimeError(f"conda package path has duplicate owners: {relative_text}")
            metadata_by_path[relative_text] = record
            owned[relative_text] = record

        archive_installed_set = set(installed_archive_paths)
        metadata_archive_set = {
            relative
            for relative, record in metadata_by_path.items()
            if record.get("path_type") in {"hardlink", "softlink"}
        }
        if archive_installed_set != metadata_archive_set:
            missing = sorted(archive_installed_set - metadata_archive_set)
            extra = sorted(metadata_archive_set - archive_installed_set)
            detail = (missing or extra)[:1]
            raise RuntimeError(
                "conda installed path topology differs from its exact archive"
                + (f": {detail[0]}" if detail else "")
            )

        entry_specs = _entry_point_specifications(archive_info)
        expected_entry_paths = {"bin/" + name for name in entry_specs}
        observed_entry_paths = {
            relative
            for relative, record in metadata_by_path.items()
            if record.get("path_type") == "unix_python_entry_point"
        }
        if expected_entry_paths != observed_entry_paths:
            raise RuntimeError(
                "generated Python entry points differ from the exact package archive"
            )

        relocated_bytecode_by_path: dict[str, dict[str, Any]] = {}
        for relative_text, record in metadata_by_path.items():
            match = PYC_HARDLINK_PATTERN.fullmatch(relative_text)
            if (
                match is None
                or match.group(3) != python_cache_tag
                or record.get("path_type") != "hardlink"
            ):
                continue
            source_path = f"{match.group(1)}/{match.group(2)}.py"
            source_record = metadata_by_path.get(source_path)
            source_archive = installed_archive_paths.get(source_path)
            if (
                source_record is None
                or source_record.get("path_type") != "hardlink"
                or source_archive is None
                or source_archive[1].get("path_type") != "hardlink"
                or source_archive[1].get("prefix_placeholder") is None
            ):
                continue
            source_archive_path, source_archive_record = source_archive
            installed_source_digest = selected_payload.get(
                source_archive_path, {}
            ).get("installed_sha256")
            if (
                SHA256_PATTERN.fullmatch(str(installed_source_digest)) is None
                or source_record.get("sha256_in_prefix") != installed_source_digest
                or source_record.get("sha256") != source_archive_record.get("sha256")
                or source_record.get("sha256") == installed_source_digest
            ):
                raise RuntimeError(
                    f"relocated bytecode source metadata is invalid: {source_path}"
                )
            relocated_bytecode_by_path[relative_text] = {
                "installed_source_sha256": installed_source_digest,
                "source_path": source_path,
            }

        for relative_text, record in metadata_by_path.items():
            path = prefix / relative_text
            path_type = record.get("path_type")
            expected_digest = record.get("sha256_in_prefix")
            if path_type == "softlink":
                archive_path, archive_record = installed_archive_paths[relative_text]
                if archive_record.get("path_type") != "softlink":
                    raise RuntimeError(
                        f"installed path type differs from exact archive: {relative_text}"
                    )
                try:
                    path_info = path.lstat()
                    target_text = os.readlink(path)
                except OSError as error:
                    raise RuntimeError(
                        f"package-owned symlink is invalid: {relative_text}"
                    ) from error
                if not stat.S_ISLNK(path_info.st_mode) or os.path.isabs(target_text):
                    raise RuntimeError(f"package-owned symlink is invalid: {relative_text}")
                try:
                    target = path.resolve(strict=True)
                    target.relative_to(prefix)
                    target_info = target.stat()
                except (OSError, ValueError) as error:
                    raise RuntimeError(
                        f"package-owned symlink escapes the prefix: {relative_text}"
                    ) from error
                if SHA256_PATTERN.fullmatch(str(expected_digest)) is None:
                    raise RuntimeError(f"invalid installed SHA-256: {relative_text}")
                archive_target = selected_payload.get(archive_path, {}).get(
                    "link_target"
                )
                if archive_target != target_text:
                    raise RuntimeError(
                        f"package-owned symlink differs from its exact archive: {relative_text}"
                    )
                if stat.S_ISREG(target_info.st_mode):
                    target_kind = "file"
                    actual_digest = hash_cache.digest(target)
                elif stat.S_ISDIR(target_info.st_mode):
                    target_kind = "directory"
                    actual_digest = EMPTY_SHA256
                else:
                    raise RuntimeError(
                        f"package-owned symlink target is invalid: {relative_text}"
                    )
                softlinks.append(
                    {
                        "path": relative_text,
                        "link_target": target_text,
                        "target_kind": target_kind,
                        "sha256_in_prefix": expected_digest,
                        "target_sha256": actual_digest,
                        "package_archive_sha256": digest,
                    }
                )
                try:
                    final_info = path.lstat()
                    final_text = os.readlink(path)
                    final_target = path.resolve(strict=True)
                except OSError as error:
                    raise RuntimeError(
                        f"package-owned symlink changed: {relative_text}"
                    ) from error
                if (
                    StatHashCache._fingerprint(final_info)
                    != StatHashCache._fingerprint(path_info)
                    or final_text != target_text
                    or final_target != target
                ):
                    raise RuntimeError(
                        f"package-owned symlink changed: {relative_text}"
                    )
            elif path_type == "hardlink":
                archive_path, archive_record = installed_archive_paths[relative_text]
                if archive_record.get("path_type") != "hardlink":
                    raise RuntimeError(
                        f"installed path type differs from exact archive: {relative_text}"
                    )
                if (
                    not isinstance(expected_digest, str)
                    or SHA256_PATTERN.fullmatch(expected_digest) is None
                ):
                    raise RuntimeError(f"invalid installed SHA-256: {relative_text}")
                if (
                    record.get("sha256") != archive_record.get("sha256")
                    or (
                        record.get("size_in_bytes") is not None
                        and record.get("size_in_bytes")
                        != archive_record.get("size_in_bytes")
                    )
                ):
                    raise RuntimeError(
                        f"installed hardlink metadata differs from exact archive: {relative_text}"
                    )
                if archive_record.get("prefix_placeholder") is None:
                    archive_expected_digest = archive_record["sha256"]
                else:
                    archive_expected_digest = selected_payload.get(
                        archive_path, {}
                    ).get("installed_sha256")
                    relocated_hardlinks += 1
                if archive_expected_digest != expected_digest:
                    raise RuntimeError(
                        f"installed SHA-256 differs from exact archive: {relative_text}"
                    )
                canonical = _canonical_regular(path, "package-owned prefix file")
                current_digest = hash_cache.digest(canonical)
                relocated_bytecode = relocated_bytecode_by_path.get(relative_text)
                if relocated_bytecode is None:
                    if current_digest != archive_expected_digest:
                        raise RuntimeError(f"package-owned prefix file changed: {relative_text}")
                    hardlink_digests[canonical] = archive_expected_digest
                    archive_bound_hardlinks += 1
                else:
                    relocated_source_bytecode_candidates.append(
                        {
                            "archive_sha256": archive_expected_digest,
                            "current_sha256": current_digest,
                            "installed_source_sha256": relocated_bytecode[
                                "installed_source_sha256"
                            ],
                            "path": relative_text,
                            "source_path": relocated_bytecode["source_path"],
                        }
                    )
                    hardlink_digests[canonical] = current_digest
            elif path_type == "pyc_file":
                match = PYC_HARDLINK_PATTERN.fullmatch(relative_text)
                if match is None or match.group(3) != python_cache_tag:
                    raise RuntimeError(
                        f"generated bytecode path is not archive-derived: {relative_text}"
                    )
                source_path = f"{match.group(1)}/{match.group(2)}.py"
                source = installed_archive_paths.get(source_path)
                if source is None or source[1].get("path_type") != "hardlink":
                    raise RuntimeError(
                        f"generated bytecode source is absent from exact archive: {relative_text}"
                    )
                source_archive_path, source_archive_record = source
                source_record = metadata_by_path.get(source_path)
                if source_record is None or source_record.get("path_type") != "hardlink":
                    raise RuntimeError(
                        f"generated bytecode source metadata is absent: {relative_text}"
                    )
                if source_archive_record.get("prefix_placeholder") is None:
                    installed_source_digest = source_archive_record.get("sha256")
                else:
                    installed_source_digest = selected_payload.get(
                        source_archive_path, {}
                    ).get("installed_sha256")
                if (
                    SHA256_PATTERN.fullmatch(str(installed_source_digest)) is None
                    or source_record.get("sha256_in_prefix")
                    != installed_source_digest
                ):
                    raise RuntimeError(
                        f"generated bytecode source digest is invalid: {relative_text}"
                    )
                if path.is_symlink():
                    raise RuntimeError(
                        f"generated bytecode is a symlink: {relative_text}"
                    )
                if path.exists():
                    canonical = _canonical_regular(path, "generated source bytecode")
                    current_digest: str | None = hash_cache.digest(canonical)
                    hardlink_digests[canonical] = current_digest
                else:
                    current_digest = None
                generated_source_bytecode_candidates.append(
                    {
                        "current_sha256": current_digest,
                        "installed_source_sha256": installed_source_digest,
                        "path": relative_text,
                        "source_path": source_path,
                    }
                )
                unhashed_bytecode += 1
            elif path_type == "unix_python_entry_point":
                canonical = _canonical_regular(path, "generated Python entry point")
                entry_info = canonical.lstat()
                entry_digest = hash_cache.digest(canonical)
                name = pathlib.PurePosixPath(relative_text).name
                module, function = entry_specs[name]
                expected_entry_digest = hashlib.sha256(
                    _entry_point_bytes(prefix, python_version, module, function)
                ).hexdigest()
                if entry_digest != expected_entry_digest:
                    raise RuntimeError(
                        f"generated Python entry point differs from exact archive: {relative_text}"
                    )
                if not hash_cache._path_matches_fingerprint(
                    canonical, StatHashCache._fingerprint(entry_info)
                ):
                    raise RuntimeError(
                        f"generated Python entry point changed: {relative_text}"
                    )
                generated_entry_points.append(
                    {
                        "path": relative_text,
                        "size_bytes": entry_info.st_size,
                        "sha256": entry_digest,
                    }
                )
                entry_point_digests[canonical] = entry_digest
            else:
                raise RuntimeError(f"unverifiable conda path record: {relative_text}")

    relocated_source_bytecode_candidates.sort(key=lambda item: item["path"])
    generated_source_bytecode_candidates.sort(key=lambda item: item["path"])
    all_source_bytecode_candidates = [
        *relocated_source_bytecode_candidates,
        *generated_source_bytecode_candidates,
    ]
    all_source_bytecode_candidates.sort(key=lambda item: item["path"])
    derived_bytecode = _compile_relocated_source_bytecode(
        prefix, python_version, all_source_bytecode_candidates
    )
    relocated_source_bytecode: list[dict[str, Any]] = []
    for candidate in relocated_source_bytecode_candidates:
        compiled = derived_bytecode.get(candidate["path"])
        if compiled is None:
            raise RuntimeError("relocated-source bytecode derivation is incomplete")
        if compiled.get("source_sha256") != candidate["installed_source_sha256"]:
            raise RuntimeError(
                f"relocated-source bytecode source differs: {candidate['path']}"
            )
        if candidate["current_sha256"] == candidate["archive_sha256"]:
            current_state = "archive-exact"
        elif candidate["current_sha256"] == compiled.get("pyc_sha256"):
            current_state = "relocated-source-compiled"
        else:
            raise RuntimeError(
                f"package-owned relocated-source bytecode changed: {candidate['path']}"
            )
        relocated_source_bytecode.append(
            {
                "archive_sha256": candidate["archive_sha256"],
                "current_sha256": candidate["current_sha256"],
                "current_state": current_state,
                "derived_sha256": compiled["pyc_sha256"],
                "derived_size_bytes": compiled["pyc_size_bytes"],
                "installed_source_hash_hex": compiled["source_hash_hex"],
                "installed_source_sha256": candidate["installed_source_sha256"],
                "path": candidate["path"],
                "source_path": candidate["source_path"],
            }
        )
    generated_source_bytecode: list[dict[str, Any]] = []
    for candidate in generated_source_bytecode_candidates:
        compiled = derived_bytecode.get(candidate["path"])
        if compiled is None:
            raise RuntimeError("generated source bytecode derivation is incomplete")
        if compiled.get("source_sha256") != candidate["installed_source_sha256"]:
            raise RuntimeError(
                f"generated bytecode source differs: {candidate['path']}"
            )
        if candidate["current_sha256"] is None:
            current_state = "absent"
        elif candidate["current_sha256"] == compiled.get("pyc_sha256"):
            current_state = "source-compiled"
        else:
            raise RuntimeError(
                f"generated source bytecode is unverifiable: {candidate['path']}"
            )
        generated_source_bytecode.append(
            {
                "current_sha256": candidate["current_sha256"],
                "current_state": current_state,
                "derived_sha256": compiled["pyc_sha256"],
                "derived_size_bytes": compiled["pyc_size_bytes"],
                "installed_source_hash_hex": compiled["source_hash_hex"],
                "installed_source_sha256": candidate["installed_source_sha256"],
                "path": candidate["path"],
                "source_path": candidate["source_path"],
            }
        )

    actual_paths = {
        path.relative_to(prefix).as_posix()
        for path in prefix.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    metadata_paths = {
        path.relative_to(prefix).as_posix() for path in metadata_files
    }
    unowned = sorted(
        relative
        for relative in actual_paths - set(owned)
        if not _allowed_unowned(relative, metadata_paths)
    )
    if unowned:
        raise RuntimeError(
            "unowned files appeared in the exact environment: " + ", ".join(unowned[:20])
        )
    if not hash_cache.verify(lock, lock_digest):
        raise RuntimeError("exact lock changed during the prefix audit")
    for metadata_path, expected_digest in metadata_digests.items():
        if not hash_cache.verify(metadata_path, expected_digest):
            raise RuntimeError(
                f"conda metadata changed during the audit: {metadata_path.name}"
            )
    for hardlink, expected_digest in hardlink_digests.items():
        if not hash_cache.verify(hardlink, expected_digest):
            raise RuntimeError(
                "package-owned prefix file changed during the audit: "
                + str(hardlink.relative_to(prefix))
            )
    for entry_point, expected_digest in entry_point_digests.items():
        if not hash_cache.verify(entry_point, expected_digest):
            raise RuntimeError(
                "generated Python entry point changed during the audit: "
                + str(entry_point.relative_to(prefix))
            )
    for archive, expected_digest in archive_digests.items():
        archive_handles[archive].verify_terminal(expected_digest)
    if not hash_cache.verify(zstd, zstd_digest):
        raise RuntimeError("zstd decoder changed during the prefix audit")
    for record in softlinks:
        alias = prefix / record["path"]
        try:
            target_text = os.readlink(alias)
            target = alias.resolve(strict=True)
            target.relative_to(prefix)
            target_info = target.stat()
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"package-owned symlink changed: {record['path']}"
            ) from error
        target_kind = (
            "file"
            if stat.S_ISREG(target_info.st_mode)
            else "directory" if stat.S_ISDIR(target_info.st_mode) else "special"
        )
        if (
            target_text != record["link_target"]
            or target_kind != record["target_kind"]
            or (
                target_kind == "file"
                and hash_cache.digest(target) != record["target_sha256"]
            )
        ):
            raise RuntimeError(
                f"package-owned symlink changed: {record['path']}"
            )
    return {
        "schema_version": 4,
        "environment_prefix": str(prefix),
        "python_version": python_version,
        "package_cache": {
            "path": str(package_cache),
            "archive_count": len(archive_digests),
        },
        "archive_snapshot": {
            "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
            "hash_and_parse_same_private_snapshot": True,
            "descriptor_retained_until_terminal_verification": True,
            "pathname_fingerprint_rechecked": True,
        },
        "archive_decoder": {
            "path": str(zstd),
            "sha256": zstd_digest,
            "version": zstd_version,
            "timeout_seconds": ZSTD_TIMEOUT_SECONDS,
        },
        "lock": {
            "path": str(lock.resolve()),
            "package_count": len(locked),
            "sha256": lock_digest,
        },
        "metadata_package_count": len(metadata_files),
        "owned_path_count": len(owned),
        "hashed_installed_path_count": sum(
            "__pycache__" not in pathlib.PurePosixPath(relative).parts
            and record.get("path_type") != "softlink"
            and isinstance(record.get("sha256_in_prefix"), str)
            for relative, record in owned.items()
        ),
        "archive_bound_hardlink_count": archive_bound_hardlinks,
        "relocated_hardlink_count": relocated_hardlinks,
        "relocated_source_bytecode_count": len(relocated_source_bytecode),
        "relocated_source_bytecode": relocated_source_bytecode,
        "generated_source_bytecode_count": len(generated_source_bytecode),
        "generated_source_bytecode": generated_source_bytecode,
        "declared_generated_bytecode_path_count": unhashed_bytecode,
        "generated_entry_points": generated_entry_points,
        "softlink_count": len(softlinks),
        "softlinks": softlinks,
        "unowned_allowed_path_count": len(actual_paths - set(owned)),
        "package_records": sorted(package_records, key=lambda record: record["url"]),
        "pass": True,
    }


def audit(
    prefix: pathlib.Path, lock: pathlib.Path, package_cache: pathlib.Path
) -> dict[str, Any]:
    with contextlib.ExitStack() as archive_stack:
        return _audit(prefix, lock, package_cache, archive_stack)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=pathlib.Path, required=True)
    parser.add_argument("--lock", type=pathlib.Path, required=True)
    parser.add_argument("--package-cache", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        atomic_write_json(
            args.output.resolve(), audit(args.prefix, args.lock, args.package_cache)
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
