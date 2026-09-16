#!/usr/bin/env python3
"""Verify a gpmeep build receipt and emit its stable identity."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpmeep_provenance import ProvenanceError, verify_build_receipt  # noqa: E402


def verify_conda_prefix_against_archives(
    receipt: dict[str, object], repo: pathlib.Path
) -> None:
    if receipt.get("build_kind") not in {
        "cpu-mpi-python-fp64",
        "cuda-mpi-python-fp32",
    }:
        return
    configuration_files = receipt.get("configuration_files")
    if not isinstance(configuration_files, dict):
        raise RuntimeError("build receipt conda-prefix audit record is absent")
    record = configuration_files.get("conda_prefix_content_audit")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise RuntimeError("build receipt conda-prefix audit record is absent")
    audit_path = pathlib.Path(record["path"])
    if not audit_path.is_absolute():
        audit_path = repo / audit_path
    try:
        retained = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("build receipt conda-prefix audit is unreadable") from error
    relocated_bytecode = (
        retained.get("relocated_source_bytecode")
        if isinstance(retained, dict)
        else None
    )
    generated_bytecode = (
        retained.get("generated_source_bytecode")
        if isinstance(retained, dict)
        else None
    )
    build_kind = receipt.get("build_kind")
    expected_schema = 4 if build_kind == "cuda-mpi-python-fp32" else 3
    if (
        not isinstance(retained, dict)
        or retained.get("schema_version") != expected_schema
        or retained.get("pass") is not True
        or not isinstance(retained.get("environment_prefix"), str)
        or not isinstance(retained.get("lock"), dict)
        or not isinstance(retained["lock"].get("path"), str)
        or not isinstance(retained.get("package_cache"), dict)
        or not isinstance(retained["package_cache"].get("path"), str)
        or retained.get("archive_snapshot")
        != {
            "open_flags": ["O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"],
            "hash_and_parse_same_private_snapshot": True,
            "descriptor_retained_until_terminal_verification": True,
            "pathname_fingerprint_rechecked": True,
        }
        or (
            expected_schema == 4
            and (
                not isinstance(relocated_bytecode, list)
                or not relocated_bytecode
                or retained.get("relocated_source_bytecode_count")
                != len(relocated_bytecode)
                or any(
                    not isinstance(item, dict)
                    or item.get("current_state") != "relocated-source-compiled"
                    or item.get("current_sha256") != item.get("derived_sha256")
                    for item in relocated_bytecode
                )
                or not isinstance(generated_bytecode, list)
                or not generated_bytecode
                or retained.get("generated_source_bytecode_count")
                != len(generated_bytecode)
                or any(
                    not isinstance(item, dict)
                    or item.get("current_state") != "source-compiled"
                    or item.get("current_sha256") != item.get("derived_sha256")
                    for item in generated_bytecode
                )
            )
        )
    ):
        raise RuntimeError("build receipt conda-prefix audit is invalid")
    with tempfile.TemporaryDirectory(
        prefix="gpmeep-receipt-prefix-audit-", dir="/tmp"
    ) as temporary:
        output = pathlib.Path(temporary) / "audit.json"
        command = [
            "/usr/bin/python3",
            "-S",
            "-P",
            "-B",
            str(repo / "scripts" / "audit-conda-prefix.py"),
            "--prefix",
            retained["environment_prefix"],
            "--lock",
            retained["lock"]["path"],
            "--package-cache",
            retained["package_cache"]["path"],
            "--output",
            str(output),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=repo,
                env={
                    "HOME": "/tmp",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": "/dev/null",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(
                "cannot reproduce the build receipt conda-prefix audit"
            ) from error
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                "current exact-archive conda-prefix audit failed"
                + (f": {diagnostic}" if diagnostic else "")
            )
        try:
            reproduced = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("reproduced conda-prefix audit is invalid") from error
    if reproduced != retained:
        raise RuntimeError(
            "build receipt conda-prefix audit differs from the current exact archives"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--receipt", required=True, type=pathlib.Path)
    parser.add_argument("--expected-build-kind")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        value = verify_build_receipt(args.receipt, args.repo, verify_source=True)
        if (
            args.expected_build_kind is not None
            and value.get("build_kind") != args.expected_build_kind
        ):
            raise ProvenanceError(
                "build kind mismatch: expected "
                f"{args.expected_build_kind!r}, observed {value.get('build_kind')!r}"
            )
        verify_conda_prefix_against_archives(value, args.repo.resolve())
    except (OSError, ProvenanceError, RuntimeError, ValueError) as exc:
        print(f"gpmeep build receipt verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": "gpmeep-build-receipt-verification-v1",
                "build_kind": value["build_kind"],
                "build_input_id": value["build_input_id"],
                "artifact_set_id": value["artifact_set_id"],
                "receipt_id": value["receipt_id"],
                "source_sha256": value["source_end"]["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
