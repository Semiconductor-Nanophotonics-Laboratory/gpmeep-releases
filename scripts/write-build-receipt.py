#!/usr/bin/env python3
"""Create an atomic source-to-binary receipt for an isolated gpmeep build."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import sys
import types
from typing import Any


def _load_provenance_from_source() -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().with_name("gpmeep_provenance.py")
    module = types.ModuleType("gpmeep_provenance")
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _load_qualification_contract_from_source() -> types.ModuleType:
    path = pathlib.Path(__file__).resolve().with_name(
        "gpmeep_qualification_contract.py"
    )
    module = types.ModuleType("gpmeep_qualification_contract_for_receipt")
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


_PROVENANCE = _load_provenance_from_source()
_QUALIFICATION = _load_qualification_contract_from_source()
atomic_write_json = _PROVENANCE.atomic_write_json
canonical_sha256 = _PROVENANCE.canonical_sha256
command_probe = _PROVENANCE.command_probe
file_record = _PROVENANCE.file_record
git_output = _PROVENANCE.git_output
source_snapshot = _PROVENANCE.source_snapshot
tree_manifest = _PROVENANCE.tree_manifest


SCHEMA_VERSION = 1
PENDING_NAME = "build-provenance.pending.json"
RECEIPT_NAME = "build-provenance.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_named_path(value: str) -> tuple[str, pathlib.Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, pathlib.Path(raw_path).resolve()


def unique_named(values: list[str]) -> dict[str, pathlib.Path]:
    result: dict[str, pathlib.Path] = {}
    for value in values:
        name, path = parse_named_path(value)
        if name in result:
            raise RuntimeError(f"duplicate provenance name: {name}")
        result[name] = path
    return result


def named_values(values: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or not name or not item:
            raise RuntimeError("expected NAME=VALUE")
        result.setdefault(name, []).append(item)
    for name, items in result.items():
        if len(items) != len(set(items)):
            raise RuntimeError(f"duplicate provenance value for {name}")
    return result


def begin(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    pending = build_dir / PENDING_NAME
    receipt = build_dir / RECEIPT_NAME
    pending.unlink(missing_ok=True)
    receipt.unlink(missing_ok=True)
    lockfiles = {
        name: file_record(path, repo)
        for name, path in unique_named(args.lockfile).items()
    }
    snapshot = source_snapshot(repo)
    value = {
        "schema_version": SCHEMA_VERSION,
        "state": "building",
        "build_kind": args.build_kind,
        "started_at_utc": utc_now(),
        "repo": str(repo),
        "build_dir": str(build_dir),
        "source_start": snapshot,
        "git_head": git_output(repo, "rev-parse", "HEAD").strip(),
        "git_status_porcelain": git_output(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines(),
        "configuration": {
            "builder": file_record(args.builder.resolve(), repo),
            "qualification_contract": args.qualification_contract,
            "configure_argv": args.configure_arg,
            "environment": {
                name: os.environ.get(name) for name in args.record_env
            },
            "lockfiles": lockfiles,
        },
    }
    value["build_input_id"] = canonical_sha256(
        {
            "schema_version": value["schema_version"],
            "build_kind": value["build_kind"],
            "source_start": value["source_start"],
            "configuration": value["configuration"],
        }
    )
    atomic_write_json(pending, value)
    return 0


def finalize(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    build_dir = args.build_dir.resolve()
    pending = build_dir / PENDING_NAME
    receipt = build_dir / RECEIPT_NAME
    if not pending.is_file():
        raise RuntimeError(f"build receipt pending state is absent: {pending}")
    value = json.loads(pending.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION or value.get("state") != "building":
        raise RuntimeError("invalid build receipt pending state")
    # Run all external version probes before the final source/artifact/tree
    # hashing epoch.  Nothing executable is invoked after those final hashes,
    # removing the former probe-to-receipt mutation window.
    tools: dict[str, Any] = {}
    for name in args.tool:
        executable = shutil.which(name)
        if executable is None:
            raise RuntimeError(f"required build tool is absent from PATH: {name}")
        executable_path = pathlib.Path(executable).resolve()
        tools[name] = {
            **file_record(executable_path, repo),
            "invoked_path": executable,
            "version": command_probe([executable, "--version"], repo),
        }
    source_end = source_snapshot(repo)
    if source_end != value.get("source_start"):
        receipt.unlink(missing_ok=True)
        raise RuntimeError("source tree changed while the build was running")

    artifacts = {
        name: file_record(path, repo)
        for name, path in unique_named(args.artifact).items()
    }
    manifest_paths = unique_named(args.manifest)
    manifest_exclusions = named_values(args.manifest_exclude_suffix)
    unknown_exclusions = sorted(set(manifest_exclusions) - set(manifest_paths))
    if unknown_exclusions:
        raise RuntimeError(
            "manifest exclusions refer to unknown manifests: "
            + ", ".join(unknown_exclusions)
        )
    manifests = {
        name: tree_manifest(
            path, repo,
            excluded_suffixes=manifest_exclusions.get(name, ()),
        )
        for name, path in manifest_paths.items()
    }
    configuration_files = {
        name: file_record(path, repo)
        for name, path in unique_named(args.configuration_file).items()
    }
    immutable_directories = {
        name: _PROVENANCE.immutable_directory_record(path, repo)
        for name, path in unique_named(args.immutable_directory).items()
    }
    value.update(
        {
            "state": "complete",
            "completed_at_utc": utc_now(),
            "source_end": source_end,
            "source_unchanged": True,
            "configuration_files": configuration_files,
            "toolchain": tools,
            "artifacts": artifacts,
            "manifests": manifests,
        }
    )
    if immutable_directories:
        value["immutable_directories"] = immutable_directories
    # The v2 name is a semantic contract, not a caller-selected label.  Refuse
    # to mint a receipt until the central contract has bound the complete log
    # set and the tested source/prefix/binaries to these final receipt groups.
    _QUALIFICATION.validate_v2_receipt(value, repo)
    artifact_payload = {
        "configuration_files": configuration_files,
        "toolchain": tools,
        "artifacts": artifacts,
        "manifests": manifests,
    }
    if immutable_directories:
        artifact_payload["immutable_directories"] = immutable_directories
    value["artifact_set_id"] = canonical_sha256(artifact_payload)
    unsigned = dict(value)
    value["receipt_id"] = canonical_sha256(unsigned)
    atomic_write_json(receipt, value)
    pending.unlink()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="action", required=True)
    start = subparsers.add_parser("begin")
    start.add_argument("--repo", type=pathlib.Path, required=True)
    start.add_argument("--build-dir", type=pathlib.Path, required=True)
    start.add_argument("--build-kind", required=True)
    start.add_argument("--builder", type=pathlib.Path, required=True)
    start.add_argument("--qualification-contract", required=True)
    start.add_argument("--configure-arg", action="append", default=[])
    start.add_argument("--lockfile", action="append", default=[])
    start.add_argument("--record-env", action="append", default=[])
    start.set_defaults(function=begin)

    finish = subparsers.add_parser("finalize")
    finish.add_argument("--repo", type=pathlib.Path, required=True)
    finish.add_argument("--build-dir", type=pathlib.Path, required=True)
    finish.add_argument("--configuration-file", action="append", default=[])
    finish.add_argument("--artifact", action="append", default=[])
    finish.add_argument("--manifest", action="append", default=[])
    finish.add_argument("--manifest-exclude-suffix", action="append", default=[])
    finish.add_argument("--immutable-directory", action="append", default=[])
    finish.add_argument("--tool", action="append", default=[])
    finish.set_defaults(function=finalize)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.function(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
