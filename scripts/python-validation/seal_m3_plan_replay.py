#!/usr/bin/env python3
"""Seal a deterministic final-commit replay of every M3 release plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import secrets
import stat
import subprocess
import sys
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m3_execution_plan  # noqa: E402
import m3_feature_plan  # noqa: E402
import m3_source_equivalence  # noqa: E402
import m3_specialized_plan  # noqa: E402


SCHEMA = "gpmeep-m3-final-plan-replay-v1"
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
MAXIMUM_RECORD_BYTES = 64 * 1024**2


class SealError(RuntimeError):
    """Raised when a final plan replay cannot be proven or published."""


def _canonical_directory(path: pathlib.Path, label: str) -> pathlib.Path:
    lexical = pathlib.Path(os.path.abspath(path))
    try:
        resolved = lexical.resolve(strict=True)
        metadata = lexical.lstat()
    except OSError as exc:
        raise SealError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not lexical.is_dir() or resolved != lexical:
        raise SealError(f"{label} is not a canonical non-symlink directory")
    return lexical


def _git(repo: pathlib.Path, *arguments: str) -> str:
    try:
        process = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=120,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise SealError(f"git {' '.join(arguments)} failed: {exc}") from exc
    if process.returncode != 0:
        raise SealError(
            f"git {' '.join(arguments)} failed: {process.stderr[-4096:]}"
        )
    return process.stdout.strip()


def _stable_file_value(
    path: pathlib.Path, label: str
) -> tuple[dict[str, Any], bytes]:
    path = pathlib.Path(os.path.abspath(path))
    try:
        before = path.lstat()
    except OSError as exc:
        raise SealError(f"{label} is unavailable: {exc}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size > MAXIMUM_RECORD_BYTES
    ):
        raise SealError(f"{label} is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SealError(f"{label} could not be opened safely: {exc}") from exc
    chunks = []
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            opened = os.fstat(stream.fileno())
            total = 0
            while chunk := stream.read(
                min(1024 * 1024, MAXIMUM_RECORD_BYTES - total + 1)
            ):
                total += len(chunk)
                if total > MAXIMUM_RECORD_BYTES:
                    raise SealError(f"{label} grew beyond its size bound")
                chunks.append(chunk)
                digest.update(chunk)
        after = path.lstat()
    except OSError as exc:
        raise SealError(f"{label} could not be hashed: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(opened) or identity(before) != identity(after):
        raise SealError(f"{label} changed while it was hashed")
    payload = b"".join(chunks)
    return (
        {
            "path": str(path),
            "size_bytes": before.st_size,
            "sha256": digest.hexdigest(),
        },
        payload,
    )


def _stable_file_record(path: pathlib.Path, label: str) -> dict[str, Any]:
    return _stable_file_value(path, label)[0]


def _code_records() -> dict[str, dict[str, Any]]:
    paths = {
        "sealer": pathlib.Path(__file__),
        "execution_plan": pathlib.Path(m3_execution_plan.__file__),
        "feature_plan": pathlib.Path(m3_feature_plan.__file__),
        "specialized_plan": pathlib.Path(m3_specialized_plan.__file__),
        "source_equivalence": pathlib.Path(m3_source_equivalence.__file__),
    }
    return {
        name: _stable_file_record(path, f"M3 final plan {name}")
        for name, path in paths.items()
    }


def derive(repo: pathlib.Path, plan_root: pathlib.Path) -> dict[str, Any]:
    repo = _canonical_directory(repo, "repository")
    plan_root = _canonical_directory(plan_root, "plan root")
    before_commit = _git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    before_status = _git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if COMMIT.fullmatch(before_commit) is None or before_status:
        raise SealError("final M3 repository is not a clean fixed commit")
    code = _code_records()
    plan = m3_feature_plan.load_release_plan(plan_root, repo)
    after_code = _code_records()
    after_commit = _git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    after_status = _git(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if (
        after_commit != before_commit
        or after_status != before_status
        or after_code != code
    ):
        raise SealError("final M3 source changed during plan replay")
    if plan.get("counts", {}).get("features") != 42:
        raise SealError("final M3 feature inventory differs")
    return {
        "schema": SCHEMA,
        "outcome": "PASS",
        "repository": str(repo),
        "plan_root": str(plan_root),
        "git_commit": before_commit,
        "git_status_porcelain": before_status,
        "source_equivalence_policy": {
            "schema": m3_source_equivalence.SCHEMA,
            "excluded_paths": sorted(m3_source_equivalence.EXCLUDED_PATHS),
        },
        "code": code,
        "plan": plan,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SealError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: pathlib.Path) -> dict[str, Any]:
    record, payload = _stable_file_value(path, "M3 final plan replay")
    if record["size_bytes"] > MAXIMUM_RECORD_BYTES:
        raise SealError("M3 final plan replay is too large")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SealError(f"invalid JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SealError(f"M3 final plan replay is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SealError("M3 final plan replay is not a JSON object")
    return value


def verify(path: pathlib.Path) -> dict[str, Any]:
    retained = load(path)
    if retained.get("schema") != SCHEMA or retained.get("outcome") != "PASS":
        raise SealError("M3 final plan replay is not a PASS")
    try:
        derived = derive(
            pathlib.Path(retained["repository"]),
            pathlib.Path(retained["plan_root"]),
        )
    except (KeyError, TypeError) as exc:
        raise SealError("M3 final plan replay paths differ") from exc
    if retained != derived:
        raise SealError("M3 final plan replay was not exactly re-derived")
    return retained


def _publish(path: pathlib.Path, value: dict[str, Any]) -> None:
    lexical = pathlib.Path(os.path.abspath(path))
    parent = _canonical_directory(lexical.parent, "output parent")
    if lexical.parent != parent or os.path.lexists(lexical):
        raise SealError("M3 final plan replay output must be a fresh path")
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = parent / f".{lexical.name}.pending.{secrets.token_hex(16)}"
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, lexical, follow_symlinks=False)
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise SealError(f"M3 final plan replay publication failed: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    create = subparsers.add_parser("create", allow_abbrev=False)
    create.add_argument("--repo", required=True, type=pathlib.Path)
    create.add_argument("--plan-root", required=True, type=pathlib.Path)
    create.add_argument("--output", required=True, type=pathlib.Path)
    replay = subparsers.add_parser("verify", allow_abbrev=False)
    replay.add_argument("--input", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.operation == "create":
        repo = _canonical_directory(args.repo, "repository")
        plan_root = _canonical_directory(args.plan_root, "plan root")
        output = pathlib.Path(os.path.abspath(args.output))
        if (
            output == repo
            or repo in output.parents
            or output == plan_root
            or plan_root in output.parents
        ):
            raise SealError("M3 final plan replay output overlaps an input root")
        _publish(output, derive(repo, plan_root))
        value = verify(output)
    else:
        value = verify(args.input)
    digest = _stable_file_record(
        args.output if args.operation == "create" else args.input,
        "M3 final plan replay",
    )["sha256"]
    print(
        "GPMEEP_M3_PLAN_REPLAY="
        + json.dumps(
            {
                "schema": SCHEMA,
                "outcome": value["outcome"],
                "git_commit": value["git_commit"],
                "sha256": digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SealError, ValueError) as error:
        print(f"M3 final plan replay error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
