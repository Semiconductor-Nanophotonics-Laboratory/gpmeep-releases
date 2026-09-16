#!/usr/bin/env python3
"""Create and verify a single-use exact-prefix qualification attestation."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import pathlib
import secrets
import stat
import sys
import time
import types


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
file_record = _PROVENANCE.file_record
sha256_file = _PROVENANCE.sha256_file
tree_manifest = _PROVENANCE.tree_manifest

SCHEMA_VERSION = 1
ENVIRONMENT_MANIFEST_EXCLUDED_SUFFIXES = (".pyc",)


def _explicit_packages(lock: pathlib.Path) -> list[str]:
    packages = sorted(
        line.strip()
        for line in lock.read_text(encoding="utf-8").splitlines()
        if "://" in line
    )
    if not packages or len(packages) != len(set(packages)):
        raise RuntimeError("exact environment lock is empty or duplicated")
    if any(
        len(line.rpartition("#")[2]) != 64
        for line in packages
    ):
        raise RuntimeError("exact environment lock lacks SHA-256 package pins")
    return packages


def create(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    prefix = args.prefix.resolve()
    lock = args.lock.resolve()
    output_directory = args.output_directory.resolve()
    if not prefix.is_dir() or not lock.is_file():
        raise RuntimeError("fresh prefix or exact lock is absent")
    output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_directory.chmod(0o700)
    nonce = secrets.token_hex(32)
    output = output_directory / f"cuda-mpi-{nonce}.json"
    value = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fresh-exact-lock-environment",
        "nonce": nonce,
        "created_at_unix_ns": time.time_ns(),
        "creator_pid": os.getpid(),
        "repo": str(repo),
        "environment_prefix": str(prefix),
        "lock": file_record(lock, repo),
        "explicit_package_count": len(_explicit_packages(lock)),
        "environment_manifest": tree_manifest(
            prefix,
            repo,
            excluded_suffixes=ENVIRONMENT_MANIFEST_EXCLUDED_SUFFIXES,
        ),
    }
    atomic_write_json(output, value)
    output.chmod(0o600)
    print(nonce)
    print(output)
    return 0


def verify(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    prefix = args.prefix.resolve()
    lock = args.lock.resolve()
    attestation = args.attestation.resolve()
    attestation_directory = args.attestation_directory.resolve()
    if (
        attestation.parent != attestation_directory
        or not attestation.name.startswith("cuda-mpi-")
        or attestation.suffix != ".json"
    ):
        raise RuntimeError("fresh-environment attestation path is invalid")
    info = attestation.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.getuid()
    ):
        raise RuntimeError("fresh-environment attestation permissions are invalid")
    try:
        value = json.loads(attestation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("fresh-environment attestation is unreadable") from error
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "kind",
        "nonce",
        "created_at_unix_ns",
        "creator_pid",
        "repo",
        "environment_prefix",
        "lock",
        "explicit_package_count",
        "environment_manifest",
    }:
        raise RuntimeError("fresh-environment attestation schema is invalid")
    nonce = value.get("nonce")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != "fresh-exact-lock-environment"
        or not isinstance(nonce, str)
        or len(nonce) != 64
        or not hmac.compare_digest(nonce, args.nonce)
        or value.get("repo") != str(repo)
        or value.get("environment_prefix") != str(prefix)
        or value.get("lock") != file_record(lock, repo)
        or value.get("explicit_package_count") != len(_explicit_packages(lock))
    ):
        raise RuntimeError("fresh-environment attestation binding is invalid")
    created = value.get("created_at_unix_ns")
    age_seconds = (time.time_ns() - created) / 1.0e9 if isinstance(created, int) else -1
    if age_seconds < 0 or age_seconds > args.maximum_age_seconds:
        raise RuntimeError("fresh-environment attestation is stale")
    current_manifest = tree_manifest(
        prefix,
        repo,
        excluded_suffixes=ENVIRONMENT_MANIFEST_EXCLUDED_SUFFIXES,
    )
    if current_manifest != value.get("environment_manifest"):
        raise RuntimeError("fresh exact-lock environment changed after recreation")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="action", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--repo", type=pathlib.Path, required=True)
    create_parser.add_argument("--prefix", type=pathlib.Path, required=True)
    create_parser.add_argument("--lock", type=pathlib.Path, required=True)
    create_parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    create_parser.set_defaults(function=create)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--repo", type=pathlib.Path, required=True)
    verify_parser.add_argument("--prefix", type=pathlib.Path, required=True)
    verify_parser.add_argument("--lock", type=pathlib.Path, required=True)
    verify_parser.add_argument("--attestation", type=pathlib.Path, required=True)
    verify_parser.add_argument(
        "--attestation-directory", type=pathlib.Path, required=True
    )
    verify_parser.add_argument("--nonce", required=True)
    verify_parser.add_argument("--maximum-age-seconds", type=float, default=900.0)
    verify_parser.set_defaults(function=verify)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return args.function(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
