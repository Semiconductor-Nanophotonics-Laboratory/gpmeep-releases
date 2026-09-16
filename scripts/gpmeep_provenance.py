#!/usr/bin/env python3
"""Fail-closed provenance helpers shared by gpmeep build/test tooling."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import pathlib
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Iterable


TRUSTED_GIT_CANDIDATES = (
    pathlib.Path("/usr/bin/git"),
    pathlib.Path("/bin/git"),
    pathlib.Path(sys.executable).resolve().parent / "git",
)


def _load_qualification_contract_from_source() -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().with_name(
        "gpmeep_qualification_contract.py"
    )
    module = types.ModuleType("gpmeep_qualification_contract_for_provenance")
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def trusted_git() -> str:
    """Return the fixed system Git, never a PATH-selected executable."""
    for candidate in dict.fromkeys(TRUSTED_GIT_CANDIDATES):
        try:
            info = os.stat(candidate, follow_symlinks=True)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode) and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(
        "trusted Git is unavailable at fixed system/Python-prefix paths"
    )


def trusted_git_environment() -> dict[str, str]:
    """Use no caller-controlled loader, Git, shell, or Python environment."""
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _terminate_command_group(proc: subprocess.Popen[bytes]) -> None:
    group_was_signalled = False
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        group_was_signalled = True
    except (ProcessLookupError, PermissionError):
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass
    # Reaping the group leader is not proof that a same-group helper has
    # exited.  After the grace period, finish the owned group even when the
    # leader ended first.
    if group_was_signalled:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def bounded_command(
    command: list[str],
    *,
    cwd: pathlib.Path,
    environment: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
    stdout_limit: int = 64 * 1024 * 1024,
    stderr_limit: int = 1024 * 1024,
    stdout_fd: int | None = None,
) -> dict[str, Any]:
    """Collect a small external command with hard time and memory bounds."""
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    sizes = {"stdout": 0, "stderr": 0}
    output_limit: str | None = None
    timed_out = False
    error: str | None = None
    exit_code: int | None = None
    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            # Keep stdout behind our pipe even when the caller supplied a
            # destination descriptor.  That is the only way to enforce the
            # same byte cap before streaming data into a potentially large
            # evidence file.
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        return {
            "command": command,
            "exit_code": None,
            "stdout": b"",
            "stderr": b"",
            "timeout": False,
            "output_limit": None,
            "error": str(exc),
        }
    assert proc.stdout is not None and proc.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
    deadline = time.monotonic() + timeout_seconds
    terminated = False
    drain_deadline: float | None = None
    limits = {"stdout": stdout_limit, "stderr": stderr_limit}
    try:
        while selector.get_map() or proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 and not terminated:
                timed_out = True
                terminated = True
                _terminate_command_group(proc)
                drain_deadline = time.monotonic() + 0.25
            if (
                terminated
                and drain_deadline is not None
                and time.monotonic() >= drain_deadline
            ):
                for key in list(selector.get_map().values()):
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                continue
            if selector.get_map():
                events = selector.select(
                    0.1 if terminated else max(0.0, min(0.1, remaining))
                )
            else:
                time.sleep(0.01 if terminated else max(0.0, min(0.01, remaining)))
                events = []
            for key, _ in events:
                name = key.data
                try:
                    block = os.read(key.fileobj.fileno(), 64 * 1024)
                except OSError:
                    block = b""
                if not block:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                accepted = min(len(block), max(0, limits[name] - sizes[name]))
                if accepted:
                    if name == "stdout" and stdout_fd is not None:
                        view = memoryview(block)[:accepted]
                        written = 0
                        try:
                            while written < accepted:
                                count = os.write(stdout_fd, view[written:])
                                if count <= 0:
                                    raise OSError(
                                        errno.EIO,
                                        "short write while streaming command stdout",
                                    )
                                written += count
                        except OSError as exc:
                            sizes[name] += written
                            error = f"cannot stream command stdout: {exc}"
                            if not terminated:
                                terminated = True
                                _terminate_command_group(proc)
                                drain_deadline = time.monotonic() + 0.25
                        else:
                            sizes[name] += accepted
                    else:
                        chunks[name].append(block[:accepted])
                        sizes[name] += accepted
                if accepted != len(block) and output_limit is None:
                    output_limit = name
                    terminated = True
                    _terminate_command_group(proc)
                    drain_deadline = time.monotonic() + 0.25
        exit_code = proc.poll()
    finally:
        selector.close()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        if proc.poll() is None:
            _terminate_command_group(proc)
        try:
            exit_code = proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            error = "child process could not be reaped after termination"
    return {
        "command": command,
        "exit_code": exit_code,
        "stdout": b"".join(chunks["stdout"]),
        "stderr": b"".join(chunks["stderr"]),
        "timeout": timed_out,
        "output_limit": output_limit,
        "error": error,
    }


class ProvenanceError(RuntimeError):
    """Raised when evidence cannot be bound to its recorded build inputs."""


class StatHashCache:
    """Stat-validated digest cache scoped to one explicit verification epoch."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[tuple[int, ...], str]] = {}

    @staticmethod
    def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    @classmethod
    def _path_matches_fingerprint(
        cls, lexical: pathlib.Path, fingerprint: tuple[int, ...]
    ) -> bool:
        try:
            current = lexical.lstat()
            resolved = lexical.resolve(strict=True)
        except OSError:
            return False
        return (
            stat.S_ISREG(current.st_mode)
            and resolved == lexical
            and cls._fingerprint(current) == fingerprint
        )

    def _open_verified(
        self, path: pathlib.Path
    ) -> tuple[pathlib.Path, int, tuple[int, ...]]:
        lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
        try:
            lexical_info = lexical.lstat()
            resolved = lexical.resolve(strict=True)
        except OSError as error:
            raise RuntimeError(f"cannot resolve provenance file: {lexical}") from error
        if not stat.S_ISREG(lexical_info.st_mode) or resolved != lexical:
            raise RuntimeError(
                f"provenance path is not a canonical regular file: {lexical}"
            )
        initial_fingerprint = self._fingerprint(lexical_info)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lexical, flags)
        except OSError as error:
            raise RuntimeError(
                f"provenance path changed before opening: {lexical}"
            ) from error
        opened = os.fstat(descriptor)
        fingerprint = self._fingerprint(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or fingerprint != initial_fingerprint
            or not self._path_matches_fingerprint(lexical, fingerprint)
        ):
            os.close(descriptor)
            raise RuntimeError(
                f"provenance path changed before hashing: {lexical}"
            )
        return lexical, descriptor, fingerprint

    def _validate_open_file(
        self,
        lexical: pathlib.Path,
        descriptor: int,
        fingerprint: tuple[int, ...],
        context: str,
    ) -> None:
        if (
            self._fingerprint(os.fstat(descriptor)) != fingerprint
            or not self._path_matches_fingerprint(lexical, fingerprint)
        ):
            raise RuntimeError(f"provenance file changed {context}: {lexical}")

    def digest(self, path: pathlib.Path) -> str:
        lexical, descriptor, fingerprint = self._open_verified(path)
        try:
            cached = self._records.get(str(lexical))
            if cached is not None and cached[0] == fingerprint:
                self._validate_open_file(
                    lexical, descriptor, fingerprint, "during cache lookup"
                )
                return cached[1]
            # Hash through the already-open descriptor so a concurrent
            # pathname rename/swap cannot substitute a different inode.
            digest = sha256_file(pathlib.Path(f"/proc/self/fd/{descriptor}"))
            self._validate_open_file(
                lexical, descriptor, fingerprint, "while hashing"
            )
            self._records[str(lexical)] = (fingerprint, digest)
            return digest
        finally:
            os.close(descriptor)

    def read_bytes(self, path: pathlib.Path) -> bytes:
        """Read a small control file through a topology-validated descriptor."""
        lexical, descriptor, fingerprint = self._open_verified(path)
        try:
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
                digest.update(block)
            self._validate_open_file(
                lexical, descriptor, fingerprint, "while reading"
            )
            self._records[str(lexical)] = (fingerprint, digest.hexdigest())
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def verify(self, path: pathlib.Path, expected: str) -> bool:
        return self.digest(path) == expected


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def command_probe(command: list[str], cwd: pathlib.Path) -> dict[str, Any]:
    result = bounded_command(
        command,
        cwd=cwd,
        timeout_seconds=30,
        stdout_limit=1024 * 1024,
        stderr_limit=1024 * 1024,
    )
    if (
        result["error"] is not None
        or result["timeout"]
        or result["output_limit"] is not None
    ):
        return {
            "command": command,
            "available": False,
            "error": result["error"] or "command exceeded time/output limits",
        }
    return {
        "command": command,
        "available": True,
        "exit_code": result["exit_code"],
        "stdout": result["stdout"].decode("utf-8", errors="replace").strip(),
        "stderr": result["stderr"].decode("utf-8", errors="replace").strip(),
    }


def git_output(repo: pathlib.Path, *arguments: str) -> str:
    result = bounded_command(
        [trusted_git(), *arguments],
        cwd=repo,
        environment=trusted_git_environment(),
        timeout_seconds=30,
    )
    if (
        result["exit_code"] != 0
        or result["error"] is not None
        or result["timeout"]
        or result["output_limit"] is not None
    ):
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: "
            + result["stderr"].decode("utf-8", errors="replace").strip()
        )
    return result["stdout"].decode("utf-8", errors="strict")


def _source_relative_paths(repo: pathlib.Path) -> list[str]:
    result = bounded_command(
        [trusted_git(), "ls-files", "-c", "-o", "--exclude-standard", "-z"],
        cwd=repo,
        environment=trusted_git_environment(),
        timeout_seconds=30,
    )
    if (
        result["exit_code"] != 0
        or result["error"] is not None
        or result["timeout"]
        or result["output_limit"] is not None
    ):
        raise RuntimeError(
            "git ls-files failed: "
            + result["stderr"].decode("utf-8", errors="replace").strip()
        )
    return sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in result["stdout"].split(b"\0")
        if item
    )


def source_snapshot(
    repo: pathlib.Path, *, hash_cache: StatHashCache | None = None
) -> dict[str, Any]:
    """Hash and enumerate every tracked/untracked non-ignored source file."""
    cache = hash_cache if hash_cache is not None else StatHashCache()
    relative_paths = _source_relative_paths(repo)
    digest = hashlib.sha256()
    missing: list[str] = []
    file_records: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = pathlib.Path(os.path.abspath(os.fspath(repo / relative)))
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        try:
            info = path.lstat()
        except FileNotFoundError:
            info = None
        if info is not None:
            if not stat.S_ISREG(info.st_mode) or path.resolve(strict=True) != path:
                raise RuntimeError(
                    "source path is not a canonical regular file: " + relative
                )
            initial_fingerprint = StatHashCache._fingerprint(info)
            file_digest = cache.digest(path)
            try:
                final_info = path.lstat()
            except OSError as error:
                raise RuntimeError(
                    "source path changed while hashing: " + relative
                ) from error
            if (
                StatHashCache._fingerprint(final_info) != initial_fingerprint
                or not cache._path_matches_fingerprint(
                    path, initial_fingerprint
                )
            ):
                raise RuntimeError(
                    "source path changed while hashing: " + relative
                )
            mode_octal = f"{stat.S_IMODE(info.st_mode):04o}"
            digest.update(b"FILE\0")
            digest.update(mode_octal.encode("ascii"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(file_digest))
            file_records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode_octal": mode_octal,
                    "size_bytes": info.st_size,
                    "sha256": file_digest,
                }
            )
        else:
            digest.update(b"MISSING")
            missing.append(relative)
            file_records.append({"path": relative, "missing": True})
        digest.update(b"\0")
    if _source_relative_paths(repo) != relative_paths:
        raise RuntimeError("source inventory changed while hashing")
    return {
        "algorithm": (
            "sha256(path NUL FILE NUL mode NUL sha256(file-content)-bytes NUL), "
            "tracked+untracked-nonignored"
        ),
        "file_count": len(relative_paths),
        "missing_paths": missing,
        "source_manifest_schema_version": 2,
        "source_manifest_sha256": canonical_sha256(file_records),
        "files": file_records,
        "sha256": digest.hexdigest(),
    }


def display_path(path: pathlib.Path, repo: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def file_record(
    path: pathlib.Path,
    repo: pathlib.Path,
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    lexical = pathlib.Path(os.path.abspath(os.fspath(path)))
    try:
        info = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(
            f"required provenance artifact is absent: {lexical}"
        ) from error
    if not stat.S_ISREG(info.st_mode) or resolved != lexical:
        raise FileNotFoundError(f"required provenance artifact is absent: {path}")
    cache = hash_cache if hash_cache is not None else StatHashCache()
    initial_fingerprint = StatHashCache._fingerprint(info)
    digest = cache.digest(lexical)
    if not cache._path_matches_fingerprint(lexical, initial_fingerprint):
        raise RuntimeError(f"provenance artifact changed while hashing: {lexical}")
    return {
        "path": display_path(lexical, repo),
        "size_bytes": info.st_size,
        "sha256": digest,
    }


def _tree_topology(
    root: pathlib.Path, suffixes: tuple[str, ...]
) -> dict[str, Any]:
    """Capture names, kinds, modes, links, and inode state without hashing."""
    try:
        root_info = root.lstat()
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(
            f"required provenance directory is absent: {root}"
        ) from error
    if not stat.S_ISDIR(root_info.st_mode) or resolved_root != root:
        raise FileNotFoundError(f"required provenance directory is absent: {root}")
    entries: list[dict[str, Any]] = []
    try:
        paths = sorted(root.rglob("*"))
    except OSError as error:
        raise RuntimeError(
            f"provenance directory changed during traversal: {root}"
        ) from error
    for path in paths:
        if path.name.endswith(suffixes):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            info = path.lstat()
        except OSError as error:
            raise RuntimeError(
                f"provenance entry changed during traversal: {path}"
            ) from error
        fingerprint = StatHashCache._fingerprint(info)
        if stat.S_ISLNK(info.st_mode):
            try:
                target_text = os.readlink(path)
            except OSError as error:
                raise RuntimeError(
                    f"provenance symlink changed during traversal: {path}"
                ) from error
            if os.path.isabs(target_text):
                raise RuntimeError(
                    f"provenance directory contains an absolute symlink: {path}"
                )
            try:
                target = path.resolve(strict=True)
                target.relative_to(root)
                target_info = target.stat()
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    f"provenance directory symlink is broken or escapes its root: {path}"
                ) from error
            if stat.S_ISREG(target_info.st_mode):
                target_kind = "file"
            elif stat.S_ISDIR(target_info.st_mode):
                target_kind = "directory"
            else:
                raise RuntimeError(
                    f"provenance directory symlink targets a special file: {path}"
                )
            entries.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "fingerprint": fingerprint,
                    "link_target": target_text,
                    "target_kind": target_kind,
                }
            )
        elif stat.S_ISREG(info.st_mode):
            if path.resolve(strict=True) != path:
                raise RuntimeError(
                    f"provenance file has a symlinked ancestor: {path}"
                )
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "fingerprint": fingerprint,
                    "mode_octal": f"{stat.S_IMODE(info.st_mode):04o}",
                    "size_bytes": info.st_size,
                }
            )
        elif stat.S_ISDIR(info.st_mode):
            if path.resolve(strict=True) != path:
                raise RuntimeError(
                    f"provenance directory has a symlinked ancestor: {path}"
                )
            entries.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "fingerprint": fingerprint,
                    "mode_octal": f"{stat.S_IMODE(info.st_mode):04o}",
                }
            )
        else:
            raise RuntimeError(
                f"provenance directory contains a special file: {path}"
            )
    return {
        "root_fingerprint": StatHashCache._fingerprint(root_info),
        "root_mode_octal": f"{stat.S_IMODE(root_info.st_mode):04o}",
        "entries": entries,
    }


def tree_manifest(
    root: pathlib.Path, repo: pathlib.Path,
    *,
    excluded_suffixes: Iterable[str] = (),
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    lexical_root = pathlib.Path(os.path.abspath(os.fspath(root)))
    suffixes = tuple(excluded_suffixes)
    cache = hash_cache if hash_cache is not None else StatHashCache()
    before = _tree_topology(lexical_root, suffixes)
    records: list[dict[str, Any]] = []
    symlinks: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []
    for entry in before["entries"]:
        public = {
            key: value
            for key, value in entry.items()
            if key != "fingerprint"
        }
        kind = entry["kind"]
        if kind == "file":
            path = lexical_root / entry["path"]
            public["sha256"] = cache.digest(path)
            if not cache._path_matches_fingerprint(
                path, entry["fingerprint"]
            ):
                raise RuntimeError(
                    f"provenance file changed while hashing: {path}"
                )
            records.append(public)
        elif kind == "directory":
            public.pop("kind")
            directories.append(public)
        else:
            public.pop("kind")
            symlinks.append(public)
    after = _tree_topology(lexical_root, suffixes)
    if after != before:
        raise RuntimeError(
            f"provenance directory topology changed while hashing: {lexical_root}"
        )
    if not records:
        raise RuntimeError(
            f"provenance directory contains no files: {lexical_root}"
        )
    payload = {
        "directories": directories,
        "files": records,
        "root_mode_octal": before["root_mode_octal"],
        "symlinks": symlinks,
    }
    manifest = {
        "root": display_path(lexical_root, repo),
        "file_count": len(records),
        "directory_count": len(directories),
        "symlink_count": len(symlinks),
        "tree_manifest_schema_version": 2,
        "root_mode_octal": payload["root_mode_octal"],
        "sha256": canonical_sha256(payload),
        "files": records,
        "directories": directories,
        "symlinks": symlinks,
    }
    if suffixes:
        manifest["excluded_suffixes"] = list(suffixes)
    return manifest


def immutable_directory_record(
    root: pathlib.Path,
    repo: pathlib.Path,
    *,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    """Record a recursively non-writable, symlink-free runtime directory."""
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"immutable provenance path is not a real directory: {root}")
    resolved = root.resolve()
    root_info = resolved.lstat()
    root_mode = stat.S_IMODE(root_info.st_mode)
    if root_mode & 0o222:
        raise RuntimeError(
            f"immutable provenance directory has write bits: {resolved}"
        )
    entries: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*")):
        info = path.lstat()
        relative = path.relative_to(resolved).as_posix()
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(
                f"immutable provenance directory contains a symlink: {path}"
            )
        if mode & 0o222:
            raise RuntimeError(
                f"immutable provenance entry has write bits: {path}"
            )
        if stat.S_ISDIR(info.st_mode):
            entries.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "mode_octal": f"{mode:04o}",
                }
            )
        elif stat.S_ISREG(info.st_mode):
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode_octal": f"{mode:04o}",
                    "size_bytes": info.st_size,
                    "sha256": (
                        hash_cache.digest(path)
                        if hash_cache is not None
                        else sha256_file(path)
                    ),
                }
            )
        else:
            raise RuntimeError(
                f"immutable provenance directory contains a special entry: {path}"
            )
    return {
        "root": display_path(resolved, repo),
        "mode_octal": f"{root_mode:04o}",
        "entry_count": len(entries),
        "sha256": canonical_sha256(entries),
        "entries": entries,
    }


def _resolve_recorded_path(value: str, repo: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else repo / path


def verify_build_receipt(
    receipt_path: pathlib.Path,
    repo: pathlib.Path,
    *,
    verify_source: bool = True,
    hash_cache: StatHashCache | None = None,
) -> dict[str, Any]:
    """Verify the receipt IDs and every recorded file/tree against disk."""
    # All recorded relative paths are repository-relative.  Canonicalize the
    # caller inputs at the trust-boundary so a perfectly valid receipt cannot
    # spuriously fail merely because a public consumer supplied ``Path('.')``
    # instead of an already-resolved repository root.  Resolving the receipt
    # relative to that root also makes CLI and in-process callers agree.
    try:
        repo = repo.resolve(strict=True)
        if not repo.is_dir():
            raise NotADirectoryError(repo)
        receipt_path = (
            receipt_path
            if receipt_path.is_absolute()
            else repo / receipt_path
        ).resolve(strict=True)
    except OSError as exc:
        raise ProvenanceError(
            f"cannot canonicalize build receipt inputs: {exc}"
        ) from exc
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read build receipt {receipt_path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ProvenanceError("unsupported build receipt schema")
    if value.get("state") != "complete" or value.get("source_unchanged") is not True:
        raise ProvenanceError("build receipt is not complete and source-stable")
    if value.get("source_start") != value.get("source_end"):
        raise ProvenanceError("receipt start/end source snapshots differ")

    unsigned = dict(value)
    recorded_receipt_id = unsigned.pop("receipt_id", None)
    if recorded_receipt_id != canonical_sha256(unsigned):
        raise ProvenanceError("build receipt ID does not match its payload")
    expected_input_id = canonical_sha256(
        {
            "schema_version": value["schema_version"],
            "build_kind": value.get("build_kind"),
            "source_start": value.get("source_start"),
            "configuration": value.get("configuration"),
        }
    )
    if value.get("build_input_id") != expected_input_id:
        raise ProvenanceError("build input ID does not match its payload")
    artifact_payload = {
        "configuration_files": value.get("configuration_files"),
        "toolchain": value.get("toolchain"),
        "artifacts": value.get("artifacts"),
        "manifests": value.get("manifests"),
    }
    if "immutable_directories" in value:
        artifact_payload["immutable_directories"] = value.get(
            "immutable_directories"
        )
    if value.get("artifact_set_id") != canonical_sha256(artifact_payload):
        raise ProvenanceError("artifact-set ID does not match its payload")

    problems: list[str] = []
    for group_name in ("configuration_files", "artifacts"):
        group = value.get(group_name)
        if not isinstance(group, dict):
            problems.append(f"{group_name} is not an object")
            continue
        for name, record in group.items():
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                problems.append(f"invalid {group_name}.{name} record")
                continue
            path = _resolve_recorded_path(record["path"], repo)
            try:
                current = file_record(path, repo, hash_cache=hash_cache)
            except FileNotFoundError as exc:
                problems.append(str(exc))
                continue
            if current != record:
                problems.append(f"{group_name}.{name} changed: {record['path']}")

    toolchain = value.get("toolchain")
    if not isinstance(toolchain, dict) or not toolchain:
        problems.append("toolchain is not a nonempty object")
    else:
        for name, record in toolchain.items():
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                problems.append(f"invalid toolchain.{name} record")
                continue
            path = _resolve_recorded_path(record["path"], repo)
            try:
                current = file_record(path, repo, hash_cache=hash_cache)
            except FileNotFoundError as exc:
                problems.append(str(exc))
                continue
            recorded_file = {
                key: record.get(key) for key in ("path", "size_bytes", "sha256")
            }
            if current != recorded_file:
                problems.append(f"toolchain.{name} changed: {record['path']}")

    manifests = value.get("manifests")
    if not isinstance(manifests, dict):
        problems.append("manifests is not an object")
    else:
        for name, record in manifests.items():
            if not isinstance(record, dict) or not isinstance(record.get("root"), str):
                problems.append(f"invalid manifests.{name} record")
                continue
            excluded_suffixes = record.get("excluded_suffixes", [])
            if (
                not isinstance(excluded_suffixes, list)
                or any(
                    not isinstance(suffix, str) or not suffix
                    for suffix in excluded_suffixes
                )
                or len(excluded_suffixes) != len(set(excluded_suffixes))
            ):
                problems.append(
                    f"invalid manifests.{name}.excluded_suffixes record"
                )
                continue
            root = _resolve_recorded_path(record["root"], repo)
            try:
                current = tree_manifest(
                    root,
                    repo,
                    excluded_suffixes=excluded_suffixes,
                    hash_cache=hash_cache,
                )
            except (FileNotFoundError, RuntimeError) as exc:
                problems.append(str(exc))
                continue
            if current != record:
                problems.append(f"manifests.{name} changed: {record['root']}")

    immutable_directories = value.get("immutable_directories")
    if immutable_directories is not None:
        if not isinstance(immutable_directories, dict) or not immutable_directories:
            problems.append("immutable_directories is not a nonempty object")
        else:
            for name, record in immutable_directories.items():
                if not isinstance(record, dict) or not isinstance(
                    record.get("root"), str
                ):
                    problems.append(
                        f"invalid immutable_directories.{name} record"
                    )
                    continue
                root = _resolve_recorded_path(record["root"], repo)
                try:
                    current = immutable_directory_record(
                        root, repo, hash_cache=hash_cache
                    )
                except (FileNotFoundError, OSError, RuntimeError) as exc:
                    problems.append(str(exc))
                    continue
                if current != record:
                    problems.append(
                        f"immutable_directories.{name} changed: {record['root']}"
                    )

    configuration = value.get("configuration")
    builder = configuration.get("builder") if isinstance(configuration, dict) else None
    if not isinstance(builder, dict) or not isinstance(builder.get("path"), str):
        problems.append("configuration.builder is invalid")
    else:
        builder_path = _resolve_recorded_path(builder["path"], repo)
        try:
            current_builder = file_record(
                builder_path, repo, hash_cache=hash_cache
            )
        except FileNotFoundError as exc:
            problems.append(str(exc))
        else:
            if current_builder != builder:
                problems.append(f"configuration.builder changed: {builder['path']}")
    lockfiles = configuration.get("lockfiles") if isinstance(configuration, dict) else None
    if not isinstance(lockfiles, dict):
        problems.append("configuration.lockfiles is not an object")
    else:
        for name, record in lockfiles.items():
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                problems.append(f"invalid lockfile record {name}")
                continue
            path = _resolve_recorded_path(record["path"], repo)
            try:
                current = file_record(path, repo, hash_cache=hash_cache)
            except FileNotFoundError as exc:
                problems.append(str(exc))
                continue
            if current != record:
                problems.append(f"lockfile changed: {record['path']}")

    if verify_source:
        recorded_head = value.get("git_head")
        recorded_status = value.get("git_status_porcelain")
        if not isinstance(recorded_head, str) or not isinstance(
            recorded_status, list
        ) or any(not isinstance(line, str) for line in recorded_status):
            problems.append("receipt Git source state is invalid")
        else:
            try:
                current_head = git_output(repo, "rev-parse", "HEAD").strip()
                current_status = git_output(
                    repo,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ).splitlines()
            except RuntimeError as error:
                problems.append(str(error))
            else:
                if current_head != recorded_head:
                    problems.append("repository Git HEAD differs from the build receipt")
                if current_status != recorded_status:
                    problems.append(
                        "repository Git status differs from the build receipt"
                    )
        try:
            current_source = source_snapshot(repo, hash_cache=hash_cache)
        except RuntimeError as error:
            problems.append(str(error))
        else:
            if current_source != value.get("source_end"):
                problems.append(
                    "repository source snapshot differs from the build receipt"
                )
    try:
        _load_qualification_contract_from_source().validate_v2_receipt(value, repo)
    except Exception as exc:
        problems.append(f"qualification-v2 contract failed: {exc}")
    if verify_source and isinstance(value.get("git_head"), str) and isinstance(
        value.get("git_status_porcelain"), list
    ):
        try:
            final_head = git_output(repo, "rev-parse", "HEAD").strip()
            final_status = git_output(
                repo,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ).splitlines()
        except RuntimeError as error:
            problems.append(str(error))
        else:
            if final_head != value["git_head"]:
                problems.append(
                    "repository Git HEAD changed during receipt verification"
                )
            if final_status != value["git_status_porcelain"]:
                problems.append(
                    "repository Git status changed during receipt verification"
                )
        try:
            final_source = source_snapshot(repo, hash_cache=hash_cache)
        except RuntimeError as error:
            problems.append(str(error))
        else:
            if final_source != value.get("source_end"):
                problems.append(
                    "repository source snapshot changed during receipt verification"
                )
        try:
            terminal_head = git_output(repo, "rev-parse", "HEAD").strip()
            terminal_status = git_output(
                repo,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ).splitlines()
        except RuntimeError as error:
            problems.append(str(error))
        else:
            if terminal_head != value["git_head"]:
                problems.append(
                    "repository Git HEAD changed during terminal source verification"
                )
            if terminal_status != value["git_status_porcelain"]:
                problems.append(
                    "repository Git status changed during terminal source verification"
                )
    if problems:
        raise ProvenanceError("; ".join(problems))
    return value
