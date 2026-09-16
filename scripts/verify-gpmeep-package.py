#!/usr/bin/env python3
"""Verify installed gpmeep package ownership and runtime provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
from typing import Any

_SCRIPT_DIRECTORY = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))
from gpmeep_release_manifest import (  # noqa: E402
    DISTRIBUTION_VERSION,
    RELEASE_MANIFEST_FIELDS,
    RELEASE_MANIFEST_SCHEMA_VERSION,
    SEALED_M3_ANCHOR,
    ReleaseManifestError,
    validate_cuda_architecture_audit,
    validate_glibc_compatibility_audit,
    validate_release_manifest,
)

SCHEMA_VERSION = 1
CONDA_BUILD_NUMBER = 0
CONDA_BUILD_STRING = "cuda124_mpi_openmpi_py311_0"
CONDA_SUBDIR = "linux-64"


class VerificationError(RuntimeError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise VerificationError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant {value!r}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: pathlib.Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _one_conda_record(prefix: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    matches = sorted(
        (prefix / "conda-meta").glob(f"gpmeep-{DISTRIBUTION_VERSION}-*.json")
    )
    if len(matches) != 1:
        raise VerificationError(
            f"expected one gpmeep {DISTRIBUTION_VERSION} conda record, "
            f"found {len(matches)}"
        )
    if list((prefix / "conda-meta").glob("pymeep-*.json")):
        raise VerificationError("gpmeep cannot share an environment with pymeep")
    try:
        value = json.loads(
            matches[0].read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise VerificationError("gpmeep conda record is unreadable") from error
    if value.get("name") != "gpmeep" or value.get("version") != DISTRIBUTION_VERSION:
        raise VerificationError("gpmeep conda record identity is invalid")
    if (
        type(value.get("build_number")) is not int
        or value["build_number"] != CONDA_BUILD_NUMBER
        or value.get("build") != CONDA_BUILD_STRING
        or value.get("subdir") != CONDA_SUBDIR
    ):
        raise VerificationError("gpmeep conda build identity is invalid")
    return matches[0], value


def _owned_paths(record: dict[str, Any]) -> list[str]:
    values = record.get("files")
    if not isinstance(values, list) or not values:
        paths_data = record.get("paths_data")
        if isinstance(paths_data, dict) and isinstance(paths_data.get("paths"), list):
            values = [item.get("_path") for item in paths_data["paths"]]
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value for value in values)
    ):
        raise VerificationError("gpmeep conda record has no owned-file list")
    if len(values) != len(set(values)):
        raise VerificationError("gpmeep conda record repeats an owned path")
    return sorted(values)


def _validate_package_identity(
    record: dict[str, Any], expected_sha256: str | None
) -> str:
    actual = record.get("sha256")
    if not isinstance(actual, str) or re.fullmatch(r"[0-9a-f]{64}", actual) is None:
        raise VerificationError("gpmeep conda record has no valid package SHA-256")
    if expected_sha256 is not None and actual != expected_sha256:
        raise VerificationError("gpmeep conda record package SHA-256 differs")
    return actual


def _owned_path_metadata(
    record: dict[str, Any], paths: list[str]
) -> dict[str, dict[str, Any]]:
    paths_data = record.get("paths_data")
    if not isinstance(paths_data, dict) or not isinstance(
        paths_data.get("paths"), list
    ):
        raise VerificationError("gpmeep conda record has no paths_data metadata")
    metadata: dict[str, dict[str, Any]] = {}
    for item in paths_data["paths"]:
        if not isinstance(item, dict):
            raise VerificationError("gpmeep paths_data contains a non-object entry")
        relative = item.get("_path")
        if not isinstance(relative, str) or not relative or relative in metadata:
            raise VerificationError("gpmeep paths_data has an invalid or repeated path")
        metadata[relative] = item
    if set(metadata) != set(paths):
        raise VerificationError("gpmeep files and paths_data inventories differ")
    return metadata


def _audit_owned_paths(
    prefix: pathlib.Path,
    paths: list[str],
    metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    regular = 0
    symlinks = 0
    verified_hashes = 0
    verified_sizes = 0
    prefix_rewritten_sizes = 0
    for relative in paths:
        candidate = pathlib.PurePosixPath(relative)
        if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
            raise VerificationError(f"unsafe package-owned path: {relative}")
        path = prefix.joinpath(*candidate.parts)
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(prefix)
            except ValueError as error:
                raise VerificationError(
                    f"package symlink escapes the environment: {relative}"
                ) from error
            symlinks += 1
        elif path.is_file():
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(prefix)
            except ValueError as error:
                raise VerificationError(
                    f"package regular file escapes the environment: {relative}"
                ) from error
            regular += 1
        else:
            raise VerificationError(f"package-owned file is absent: {relative}")
        if metadata is not None:
            item = metadata[relative]
            path_type = item.get("path_type")
            if path_type == "softlink" and not path.is_symlink():
                raise VerificationError(f"expected a package symlink: {relative}")
            if path_type == "hardlink" and path.is_symlink():
                raise VerificationError(f"unexpected package symlink: {relative}")
            if path_type not in {"hardlink", "softlink"}:
                raise VerificationError(
                    f"unsupported package path type {path_type!r}: {relative}"
                )
            source_hash = item.get("sha256")
            expected_hash = item.get("sha256_in_prefix") or source_hash
            if (
                not isinstance(expected_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            ):
                raise VerificationError(f"package path has no valid hash: {relative}")
            if _sha256_file(resolved) != expected_hash:
                raise VerificationError(f"package-owned file hash differs: {relative}")
            expected_size = item.get("size_in_bytes")
            if not isinstance(expected_size, int) or expected_size < 0:
                raise VerificationError(f"package path has no valid size: {relative}")
            if source_hash is not None and source_hash != expected_hash:
                # Conda records the pre-rewrite size even though replacing its
                # fixed prefix placeholder can change a text file's length.
                # sha256_in_prefix remains authoritative for installed bytes.
                prefix_rewritten_sizes += 1
            else:
                if resolved.stat().st_size != expected_size:
                    raise VerificationError(
                        f"package-owned file size differs: {relative}"
                    )
                verified_sizes += 1
            verified_hashes += 1
    return {
        "owned_paths": len(paths),
        "regular_files": regular,
        "symlinks": symlinks,
        "content_hashes_verified": verified_hashes,
        "content_sizes_verified": verified_sizes,
        "prefix_rewritten_sizes": prefix_rewritten_sizes,
    }


def _mapped_libmeep(prefix: pathlib.Path) -> pathlib.Path:
    candidates: set[pathlib.Path] = set()
    for line in (
        pathlib.Path("/proc/self/maps")
        .read_text(encoding="utf-8", errors="replace")
        .splitlines()
    ):
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith("/"):
            path = pathlib.Path(columns[5])
            if path.name.startswith("libmeep.so") and path.is_file():
                candidates.add(path.resolve(strict=True))
    if len(candidates) != 1:
        raise VerificationError(f"expected one mapped libmeep, found {len(candidates)}")
    path = next(iter(candidates))
    try:
        path.relative_to(prefix)
    except ValueError as error:
        raise VerificationError(
            f"mapped libmeep escaped gpmeep prefix: {path}"
        ) from error
    return path


def _runtime(prefix: pathlib.Path) -> dict[str, Any]:
    if pathlib.Path(sys.prefix).resolve() != prefix:
        raise VerificationError(
            f"verifier must run with {prefix}/bin/python, got {sys.executable}"
        )
    os.environ.setdefault("MEEP_GPU_BACKEND", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import meep as mp
    import meep._meep as native

    if not mp.gpu.compiled or not mp.with_mpi() or not mp.is_single_precision():
        raise VerificationError("installed runtime is not CUDA+MPI FP32 gpmeep")
    module = pathlib.Path(mp.__file__).resolve(strict=True)
    extension = pathlib.Path(native.__file__).resolve(strict=True)
    for name, path in (("module", module), ("extension", extension)):
        try:
            path.relative_to(prefix)
        except ValueError as error:
            raise VerificationError(f"{name} escaped gpmeep prefix: {path}") from error
    return {
        "python": _file_record(pathlib.Path(sys.executable)),
        "module": _file_record(module),
        "extension": _file_record(extension),
        "libmeep": _file_record(_mapped_libmeep(prefix)),
        "meep_version": mp.__version__,
        "cuda_compiled": bool(mp.gpu.compiled),
        "mpi_enabled": bool(mp.with_mpi()),
        "single_precision": bool(mp.is_single_precision()),
    }


def _manifest(prefix: pathlib.Path, expected_commit: str | None) -> dict[str, Any]:
    lexical_path = prefix / "share/gpmeep/release.json"
    try:
        path = lexical_path.resolve(strict=True)
        path.relative_to(prefix)
        if lexical_path.is_symlink() or not lexical_path.is_file():
            raise OSError("release manifest is not regular")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise VerificationError("gpmeep release manifest is unreadable") from error
    try:
        validate_release_manifest(value, expected_source_commit=expected_commit)
        validate_cuda_architecture_audit(
            prefix,
            value["cuda_architecture_audit_sha256"],
            value["cuda_real_architectures"],
            value["cuda_virtual_architectures"],
        )
        validate_glibc_compatibility_audit(
            prefix, value["glibc_compatibility_audit_sha256"]
        )
    except ReleaseManifestError as error:
        raise VerificationError(str(error)) from error
    value["file"] = _file_record(path)
    return value


def _write_new(path: pathlib.Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(_canonical_json(value) + "\n")
    except FileExistsError as error:
        raise VerificationError(f"refusing to overwrite report: {path}") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify installed gpmeep provenance")
    parser.add_argument("--prefix", type=pathlib.Path, default=pathlib.Path(sys.prefix))
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-package-sha256")
    parser.add_argument("--report", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    prefix = args.prefix.resolve(strict=True)
    if (
        args.expected_source_commit is not None
        and re.fullmatch(r"[0-9a-f]{40}", args.expected_source_commit) is None
    ):
        raise VerificationError("--expected-source-commit must be 40 lowercase hex")
    if (
        args.expected_package_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", args.expected_package_sha256) is None
    ):
        raise VerificationError("--expected-package-sha256 must be 64 lowercase hex")
    record_path, conda_record = _one_conda_record(prefix)
    package_sha256 = _validate_package_identity(
        conda_record, args.expected_package_sha256
    )
    paths = _owned_paths(conda_record)
    metadata = _owned_path_metadata(conda_record, paths)
    audit = _audit_owned_paths(prefix, paths, metadata)
    required_paths = {
        "bin/gpmeep-self-check",
        "bin/gpmeep-verify-provenance",
        "share/gpmeep/cuda-architecture-inventory.json",
        "share/gpmeep/glibc-compatibility.json",
        "share/gpmeep/libexec/check-cuda-architecture-inventory.py",
        "share/gpmeep/libexec/gpmeep_release_manifest.py",
        "share/gpmeep/release.json",
    }
    missing = sorted(required_paths - set(paths))
    if missing:
        raise VerificationError(f"gpmeep package omits required paths: {missing}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "prefix": str(prefix),
        "conda_record": _file_record(record_path),
        "package_url": conda_record.get("url"),
        "package_sha256": package_sha256,
        "ownership": audit,
        "manifest": _manifest(prefix, args.expected_source_commit),
        "runtime": _runtime(prefix),
    }
    report["report_id"] = hashlib.sha256(
        _canonical_json(report).encode("utf-8")
    ).hexdigest()
    if args.report is not None:
        _write_new(args.report, report)
    print(
        "gpmeep provenance PASS: "
        f"source={report['manifest']['source_commit']}, report_id={report['report_id']}"
    )
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"gpmeep provenance FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
