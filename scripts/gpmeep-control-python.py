#!/usr/bin/env python3
"""Execute checked-in build-control Python from a cache-disabled runtime."""

from __future__ import annotations

import os
import pathlib
import stat
import sys


MAX_CONTROL_SCRIPT_BYTES = 16 * 1024 * 1024
ALLOWED_CONTROL_SCRIPTS = frozenset(
    {
        "audit-conda-prefix.py",
        "capture-build-environment.py",
        "fresh-environment-attestation.py",
        "gpmeep_qualification_contract.py",
        "normalize-conda-generated-bytecode.py",
        "normalize-conda-relocated-bytecode.py",
        "verify-protected-bash.py",
        "write-build-receipt.py",
    }
)


def fail(message: str) -> "NoReturn":
    print(f"gpmeep control-python error: {message}", file=sys.stderr)
    raise SystemExit(1)


def metadata(info: os.stat_result) -> tuple[int, ...]:
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


def read_exact(descriptor: int, size: int, path: pathlib.Path) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        block = os.read(descriptor, min(1024 * 1024, remaining))
        if not block:
            fail(f"control source was truncated while reading: {path}")
        chunks.append(block)
        remaining -= len(block)
    if os.read(descriptor, 1):
        fail(f"control source grew while reading: {path}")
    return b"".join(chunks)


def assert_stable_open_source(
    descriptor: int,
    path: pathlib.Path,
    initial: os.stat_result,
    payload: bytes,
    phase: str,
) -> None:
    current_fd = os.fstat(descriptor)
    try:
        current_path = os.lstat(path)
    except OSError as exc:
        fail(f"control source path vanished during {phase}: {path}: {exc}")
    if (
        metadata(current_fd) != metadata(initial)
        or metadata(current_path) != metadata(initial)
        or stat.S_ISLNK(current_path.st_mode)
    ):
        fail(f"control source identity changed during {phase}: {path}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    replay = read_exact(descriptor, initial.st_size, path)
    if replay != payload:
        fail(f"control source bytes changed during {phase}: {path}")


def snapshot_closed_source(
    path: pathlib.Path,
) -> tuple[os.stat_result, bytes]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        fail(f"cannot open control source {path}: {exc}")
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or not (
            0 < initial.st_size <= MAX_CONTROL_SCRIPT_BYTES
        ):
            fail(f"control source is not a bounded regular file: {path}")
        payload = read_exact(descriptor, initial.st_size, path)
        assert_stable_open_source(
            descriptor, path, initial, payload, "initial snapshot"
        )
        return initial, payload
    finally:
        os.close(descriptor)


def assert_unchanged_closed_source(
    path: pathlib.Path,
    initial: os.stat_result,
    payload: bytes,
    phase: str,
) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        fail(f"control source path vanished during {phase}: {path}: {exc}")
    try:
        assert_stable_open_source(descriptor, path, initial, payload, phase)
    finally:
        os.close(descriptor)


def main() -> None:
    if (
        sys.flags.ignore_environment
        or sys.flags.isolated
        or not sys.flags.no_site
        or not sys.flags.safe_path
        or not sys.dont_write_bytecode
        or sys.pycache_prefix != "/dev/null"
    ):
        fail(
            "requires non--E/-I Python with -S -P -B and "
            "PYTHONPYCACHEPREFIX=/dev/null"
        )
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        if name in os.environ:
            fail(f"unsafe inherited Python environment is present: {name}")
    if len(sys.argv) < 2:
        fail("expected a checked-in control script path")
    runner = pathlib.Path(__file__).absolute()
    runner_initial, runner_payload = snapshot_closed_source(runner)
    target = pathlib.Path(sys.argv[1]).absolute()
    if target.parent != runner.parent or target.name not in ALLOWED_CONTROL_SCRIPTS:
        fail(f"control script is not allowlisted: {target}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        fail(f"cannot open control script {target}: {exc}")
    initial = os.fstat(descriptor)
    try:
        try:
            initial_path = os.lstat(target)
        except OSError as exc:
            fail(f"cannot inspect control script path {target}: {exc}")
        if (
            metadata(initial_path) != metadata(initial)
            or stat.S_ISLNK(initial_path.st_mode)
        ):
            fail(f"control script path does not name the opened file: {target}")
        if not stat.S_ISREG(initial.st_mode) or not (
            0 < initial.st_size <= MAX_CONTROL_SCRIPT_BYTES
        ):
            fail(f"control script is not a bounded regular file: {target}")
        payload = read_exact(descriptor, initial.st_size, target)
        assert_stable_open_source(
            descriptor, target, initial, payload, "read"
        )
        code = compile(payload, str(target), "exec")
        assert_stable_open_source(
            descriptor, target, initial, payload, "compile"
        )
        sys.argv = [str(target), *sys.argv[2:]]
        namespace = {
            "__builtins__": __builtins__,
            "__cached__": None,
            "__file__": str(target),
            "__loader__": None,
            "__name__": "__main__",
            "__package__": None,
            "__spec__": None,
        }
        try:
            exec(code, namespace)
        except BaseException:
            assert_unchanged_closed_source(
                runner,
                runner_initial,
                runner_payload,
                "target execution",
            )
            assert_stable_open_source(
                descriptor, target, initial, payload, "execution"
            )
            raise
        assert_unchanged_closed_source(
            runner,
            runner_initial,
            runner_payload,
            "target execution",
        )
        assert_stable_open_source(
            descriptor, target, initial, payload, "execution"
        )
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    main()
