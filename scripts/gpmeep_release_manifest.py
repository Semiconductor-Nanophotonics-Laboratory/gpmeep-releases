#!/usr/bin/env python3
"""Shared validation for the installed gpmeep release manifest.

The build records the requested CUDA architecture policy and the native/PTX
inventories that were actually compiled.  AUTO remains a sealed CUDA 12.4
release policy, while an explicit LIST is validated structurally instead of
being rejected merely because it differs from AUTO.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
import stat
from typing import Any


DISTRIBUTION_VERSION = "1.0.3"
RELEASE_MANIFEST_SCHEMA_VERSION = 3
GLIBC_MINIMUM = "2.17"
GLIBC_AUDIT_SCHEMA = "gpmeep-glibc-compatibility-v1"
GLIBC_AUDIT_PATH = pathlib.Path("share/gpmeep/glibc-compatibility.json")
CUDA_AUDIT_SCHEMA = "gpmeep-cuda-architecture-inventory-v1"
CUDA_AUDIT_PATH = pathlib.Path("share/gpmeep/cuda-architecture-inventory.json")
AUTO_REAL_ARCHITECTURES = (60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90)
AUTO_VIRTUAL_ARCHITECTURES = (90,)
RUNTIME_VALIDATED_NATIVE_ARCHITECTURES = (86,)
ARCHITECTURE_KEYS = {
    "cuda_architectures",
    "cuda_real_architectures",
    "cuda_virtual_architectures",
    "runtime_validated_architectures",
}
SEALED_M3_ANCHOR = {
    "schema": "gpmeep-sealed-m3-anchor-v1",
    "qualification_eligible": True,
    "source_commit": "adc67c3466774bbdb1875d996e4e0e50aeb7a845",
    "report_sha256": "50d2d8539000c76fb554318e2aa641577342e0f069ea5078d6f5ce596c5c36dc",
    "complete_sha256": (
        "7d3d522c4dbcce60e9c264acfa925c5f0ac0de18ff0540e8e9368835e45a4d2f"
    ),
    "audit_sha256": "4b8a257b2bcf9349f2bb9ccdd0dec172bb2d7542d078082a637394a479d75639",
    "means_seconds": {
        "cpu8": {"fdtd": 163.82354227802716, "end_to_end": 178.19756312202662},
        "gpu1": {"fdtd": 16.680280592525378, "end_to_end": 41.04173981299391},
        "gpu2": {"fdtd": 10.951151260989718, "end_to_end": 30.025476827519014},
    },
}
FIXED_RELEASE_MANIFEST_FIELDS = {
    "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
    "distribution": "gpmeep",
    "distribution_version": DISTRIBUTION_VERSION,
    "upstream_source_version": "1.35.0-beta",
    "python_import": "meep",
    "precision": "fp32",
    "cuda_fast_math": False,
    "cuda_toolkit": "12.4",
    "glibc_minimum": GLIBC_MINIMUM,
    "mpi": "openmpi",
    "scheme": False,
    "package_policy": "dedicated-environment-only",
    "performance_anchor": SEALED_M3_ANCHOR,
}


class ReleaseManifestError(ValueError):
    """Raised when a release manifest is not self-consistent."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReleaseManifestError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ReleaseManifestError(f"non-finite JSON constant {value!r}")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_regular_within(
    root: pathlib.Path, path: pathlib.Path, label: str
) -> pathlib.Path:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ReleaseManifestError(f"{label} escapes its gpmeep prefix") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ReleaseManifestError(f"{label} is not a regular file")
    try:
        resolved_metadata = resolved.stat()
    except OSError as error:
        raise ReleaseManifestError(f"{label} is unavailable") from error
    if not stat.S_ISREG(resolved_metadata.st_mode):
        raise ReleaseManifestError(f"{label} is not a regular file")
    return resolved


def typed_json_equal(actual: Any, expected: Any) -> bool:
    """Compare decoded JSON without Python's bool/int type coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            typed_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            typed_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _architecture_list(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not all(type(item) is int and 60 <= item <= 999 for item in value)
        or len(value) != len(set(value))
    ):
        raise ReleaseManifestError(f"release manifest {label} is invalid")
    return value


def _append_unique(values: list[int], item: int) -> None:
    if item not in values:
        values.append(item)


def architectures_for_specification(specification: str) -> tuple[list[int], list[int]]:
    """Return real/virtual architecture lists for an explicit build policy."""

    if specification == "AUTO":
        return list(AUTO_REAL_ARCHITECTURES), list(AUTO_VIRTUAL_ARCHITECTURES)
    if (
        not isinstance(specification, str)
        or re.fullmatch(
            r"[0-9]+(?:-(?:real|virtual))?(?:[,;][0-9]+(?:-(?:real|virtual))?)*",
            specification,
        )
        is None
    ):
        raise ReleaseManifestError("release manifest cuda_architectures is invalid")
    real: list[int] = []
    virtual: list[int] = []
    for entry in re.split(r"[,;]", specification):
        raw, separator, kind = entry.partition("-")
        if len(raw) > 1 and raw.startswith("0"):
            raise ReleaseManifestError("CUDA architecture has a leading zero")
        architecture = int(raw)
        if not 60 <= architecture <= 999:
            raise ReleaseManifestError("CUDA architecture is outside the supported range")
        if not separator:
            _append_unique(real, architecture)
            _append_unique(virtual, architecture)
        elif kind == "real":
            _append_unique(real, architecture)
        elif kind == "virtual":
            _append_unique(virtual, architecture)
        else:  # The regular expression makes this branch defensive only.
            raise ReleaseManifestError("CUDA architecture kind is invalid")
    if not real:
        raise ReleaseManifestError(
            "CUDA architecture LIST must include at least one native -real target"
        )
    return real, virtual


def runtime_validated_architectures(real: list[int]) -> list[str]:
    return [
        f"sm{architecture}"
        for architecture in RUNTIME_VALIDATED_NATIVE_ARCHITECTURES
        if architecture in real
    ]


def default_release_manifest_fields() -> dict[str, Any]:
    real, virtual = architectures_for_specification("AUTO")
    return copy.deepcopy(
        {
            **FIXED_RELEASE_MANIFEST_FIELDS,
            "cuda_architectures": "AUTO",
            "cuda_real_architectures": real,
            "cuda_virtual_architectures": virtual,
            "runtime_validated_architectures": runtime_validated_architectures(real),
            # Test fixtures replace these with exact build-produced digests.
            "cuda_architecture_audit_sha256": "b" * 64,
            # Test fixtures replace this with the exact build-produced digest.
            "glibc_compatibility_audit_sha256": "a" * 64,
        }
    )


RELEASE_MANIFEST_FIELDS = default_release_manifest_fields()


def validate_release_manifest(
    value: Any, *, expected_source_commit: str | None = None
) -> dict[str, Any]:
    expected_keys = {
        *FIXED_RELEASE_MANIFEST_FIELDS,
        *ARCHITECTURE_KEYS,
        "cuda_architecture_audit_sha256",
        "glibc_compatibility_audit_sha256",
        "source_commit",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ReleaseManifestError("release manifest schema is invalid")
    for key, expected in FIXED_RELEASE_MANIFEST_FIELDS.items():
        if not typed_json_equal(value.get(key), expected):
            raise ReleaseManifestError(f"release manifest field {key!r} is invalid")

    specification = value.get("cuda_architectures")
    if not isinstance(specification, str):
        raise ReleaseManifestError("release manifest cuda_architectures is invalid")
    expected_real, expected_virtual = architectures_for_specification(specification)
    actual_real = _architecture_list(
        value.get("cuda_real_architectures"), "cuda_real_architectures"
    )
    actual_virtual = _architecture_list(
        value.get("cuda_virtual_architectures"), "cuda_virtual_architectures"
    )
    if actual_real != expected_real or actual_virtual != expected_virtual:
        raise ReleaseManifestError(
            "release manifest CUDA inventories differ from cuda_architectures"
        )
    expected_validated = runtime_validated_architectures(actual_real)
    if not typed_json_equal(
        value.get("runtime_validated_architectures"), expected_validated
    ):
        raise ReleaseManifestError(
            "release manifest runtime-validated architecture claim is invalid"
        )
    cuda_audit_sha256 = value.get("cuda_architecture_audit_sha256")
    if not isinstance(cuda_audit_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", cuda_audit_sha256
    ) is None:
        raise ReleaseManifestError("release manifest CUDA audit digest is invalid")
    audit_sha256 = value.get("glibc_compatibility_audit_sha256")
    if not isinstance(audit_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", audit_sha256
    ) is None:
        raise ReleaseManifestError("release manifest GLIBC audit digest is invalid")
    commit = value.get("source_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ReleaseManifestError("release manifest source_commit is invalid")
    if expected_source_commit is not None and commit != expected_source_commit:
        raise ReleaseManifestError("release manifest source commit differs from expected")
    return value


def validate_cuda_architecture_audit(
    prefix: pathlib.Path,
    expected_sha256: str,
    expected_real: list[int],
    expected_virtual: list[int],
) -> dict[str, Any]:
    """Validate the build-produced SASS/PTX inventory and current libmeep bytes."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ReleaseManifestError("CUDA architecture audit digest is invalid")
    prefix = prefix.resolve(strict=True)
    path = _resolved_regular_within(
        prefix,
        prefix / CUDA_AUDIT_PATH,
        "CUDA architecture audit",
    )
    if _sha256_file(path) != expected_sha256:
        raise ReleaseManifestError("CUDA architecture audit digest differs")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseManifestError("CUDA architecture audit is unreadable") from error
    expected_keys = {
        "schema",
        "state",
        "cuda_binary",
        "cuobjdump_version",
        "real_architectures",
        "virtual_architectures",
        "elf_record_count",
        "ptx_record_count",
        "elf_listing_sha256",
        "ptx_listing_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ReleaseManifestError("CUDA architecture audit schema is invalid")
    if value.get("schema") != CUDA_AUDIT_SCHEMA or value.get("state") != "PASS":
        raise ReleaseManifestError("CUDA architecture audit policy is invalid")
    actual_real = _architecture_list(
        value.get("real_architectures"), "CUDA audit real_architectures"
    )
    actual_virtual = _architecture_list(
        value.get("virtual_architectures"), "CUDA audit virtual_architectures"
    )
    if actual_real != sorted(expected_real) or actual_virtual != sorted(expected_virtual):
        raise ReleaseManifestError("CUDA architecture audit inventories differ")
    elf_count = value.get("elf_record_count")
    ptx_count = value.get("ptx_record_count")
    if (
        type(elf_count) is not int
        or elf_count < 1
        or type(ptx_count) is not int
        or ptx_count < 0
        or (bool(actual_virtual) != bool(ptx_count))
    ):
        raise ReleaseManifestError("CUDA architecture audit record counts are invalid")
    if not isinstance(value.get("cuobjdump_version"), str) or not value[
        "cuobjdump_version"
    ].strip():
        raise ReleaseManifestError("CUDA architecture audit tool identity is invalid")
    for label in ("elf_listing_sha256", "ptx_listing_sha256"):
        digest = value.get(label)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ReleaseManifestError(f"CUDA architecture audit {label} is invalid")
    record = value.get("cuda_binary")
    if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
        raise ReleaseManifestError("CUDA architecture binary record is invalid")
    relative = record.get("path")
    candidate = pathlib.PurePosixPath(relative) if isinstance(relative, str) else None
    if (
        candidate is None
        or not relative
        or "\\" in relative
        or candidate.is_absolute()
        or ".." in candidate.parts
        or re.fullmatch(r"lib/libmeep[.]so(?:[.][0-9]+)+", relative) is None
    ):
        raise ReleaseManifestError("CUDA architecture binary path is unsafe")
    binary = _resolved_regular_within(
        prefix,
        prefix.joinpath(*candidate.parts),
        "CUDA architecture binary",
    )
    metadata = binary.stat()
    size = record.get("size_bytes")
    digest = record.get("sha256")
    if (
        type(size) is not int
        or size < 1
        or metadata.st_size != size
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or _sha256_file(binary) != digest
    ):
        raise ReleaseManifestError("CUDA architecture binary identity differs")
    return value


def validate_glibc_compatibility_audit(
    prefix: pathlib.Path, expected_sha256: str
) -> dict[str, Any]:
    """Validate the build-produced GLIBC report and its current ELF bytes."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ReleaseManifestError("GLIBC audit digest is invalid")
    prefix = prefix.resolve(strict=True)
    path = _resolved_regular_within(
        prefix,
        prefix / GLIBC_AUDIT_PATH,
        "GLIBC compatibility audit",
    )
    if _sha256_file(path) != expected_sha256:
        raise ReleaseManifestError("GLIBC compatibility audit digest differs")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseManifestError("GLIBC compatibility audit is unreadable") from error
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "state",
        "maximum_allowed",
        "overall_maximum_required",
        "elf_count",
        "elf_files",
    }:
        raise ReleaseManifestError("GLIBC compatibility audit schema is invalid")
    if (
        value.get("schema") != GLIBC_AUDIT_SCHEMA
        or value.get("state") != "PASS"
        or value.get("maximum_allowed") != GLIBC_MINIMUM
    ):
        raise ReleaseManifestError("GLIBC compatibility audit policy is invalid")
    records = value.get("elf_files")
    if (
        not isinstance(records, list)
        or not records
        or type(value.get("elf_count")) is not int
        or value["elf_count"] != len(records)
    ):
        raise ReleaseManifestError("GLIBC compatibility ELF inventory is invalid")
    seen: set[str] = set()
    maxima: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
            "required_versions",
            "maximum_required",
        }:
            raise ReleaseManifestError("GLIBC compatibility ELF record is invalid")
        relative = record.get("path")
        candidate = pathlib.PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            candidate is None
            or not relative
            or "\\" in relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in seen
        ):
            raise ReleaseManifestError("GLIBC compatibility ELF path is unsafe")
        seen.add(relative)
        elf = _resolved_regular_within(
            prefix,
            prefix.joinpath(*candidate.parts),
            f"GLIBC compatibility ELF {relative}",
        )
        elf_metadata = elf.stat()
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            type(size) is not int
            or size < 0
            or elf_metadata.st_size != size
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or _sha256_file(elf) != digest
        ):
            raise ReleaseManifestError(
                f"GLIBC compatibility ELF identity differs: {relative}"
            )
        versions = record.get("required_versions")
        maximum = record.get("maximum_required")
        if (
            not isinstance(versions, list)
            or not versions
            or not all(isinstance(item, str) for item in versions)
            or versions != sorted(set(versions), key=_glibc_version_key)
            or maximum != versions[-1]
            or _glibc_version_key(maximum) > _glibc_version_key(GLIBC_MINIMUM)
        ):
            raise ReleaseManifestError(
                f"GLIBC compatibility versions are invalid: {relative}"
            )
        maxima.append(maximum)
    overall = value.get("overall_maximum_required")
    if overall != max(maxima, key=_glibc_version_key):
        raise ReleaseManifestError("GLIBC compatibility overall maximum differs")
    return value


def _glibc_version_key(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or re.fullmatch(
        r"[0-9]+(?:\.[0-9]+)+", value
    ) is None:
        raise ReleaseManifestError(f"invalid GLIBC version {value!r}")
    return tuple(int(part) for part in value.split("."))
