#!/usr/bin/env python3
"""Seal and aggregate disjoint gpmeep Python-validation batches.

This is deliberately a coverage/evidence aggregator, not a scheduler.  A sealed
plan fixes the build/source/manifest/native-artifact identity and the exact
``(case_id, backend)`` pairs owned by each batch.  Aggregation succeeds only
when fresh runner evidence covers that plan exactly and every assigned run
passes its backend contract.

The ``covered_by`` dependency graph and MPI/external-gate pass semantics are
intentionally not inferred here.  Their identifiers remain explicit unresolved
inputs in both the plan and aggregate evidence so this prototype cannot be
mistaken for a complete release qualification gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for import_root in (SCRIPTS_DIR, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpmeep_provenance import (  # noqa: E402
    ProvenanceError,
    verify_build_receipt,
)
import archive_validation as archive_contract  # noqa: E402


PLAN_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA_VERSION = 1
RUNNER_SCHEMA_VERSION = archive_contract.RUNNER_REPORT_SCHEMA_VERSION
PLAN_KIND = "gpmeep-validation-aggregate-plan"
AGGREGATE_KIND = "gpmeep-validation-aggregate-report"
MAX_CONTROL_BYTES = 16 * 1024 * 1024
MAX_FUTURE_CLOCK_SKEW = dt.timedelta(minutes=5)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
BATCH_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REPORT_KEYS = set(archive_contract.RUNNER_REPORT_KEYS)
COMPLETE_KEYS = set(archive_contract.RUNNER_COMPLETE_KEYS)
RESULT_KEYS = set(archive_contract.RESULT_RECORD_KEYS)
CONFIGURATION_KEYS = set(archive_contract.RUNNER_CONFIGURATION_KEYS)
PERFORMANCE_KEYS = set(archive_contract.PERFORMANCE_EVIDENCE_KEYS)
SOURCE_IDENTITY_KEYS = {
    "algorithm",
    "file_count",
    "sha256",
    "source_manifest_schema_version",
    "source_manifest_sha256",
}
NATIVE_RUNNER_KEYS = {
    "python_extension": "extension",
    "libmeep": "libmeep",
}


class AggregationError(RuntimeError):
    """Evidence or plan failed a closed-world aggregation check."""


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"value is not canonical finite JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_read(path: pathlib.Path, *, limit: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AggregationError(f"cannot open regular evidence file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AggregationError(f"evidence path is not a regular file: {path}")
        if limit is not None and before.st_size > limit:
            raise AggregationError(
                f"evidence file exceeds {limit} bytes: {path}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise AggregationError(f"evidence file was truncated while read: {path}")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise AggregationError(f"evidence file grew while read: {path}")
        after = os.fstat(descriptor)
        try:
            named = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise AggregationError(f"evidence path changed while read: {path}: {exc}") from exc
        fingerprint = lambda value: (  # noqa: E731
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if fingerprint(before) != fingerprint(after) or fingerprint(before) != fingerprint(named):
            raise AggregationError(f"evidence file changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_snapshot(path: pathlib.Path) -> dict[str, Any]:
    data = _stable_read(path)
    return {"size_bytes": len(data), "sha256": _sha256_bytes(data)}


def _load_json(path: pathlib.Path, label: str) -> tuple[dict[str, Any], bytes]:
    data = _stable_read(path, limit=MAX_CONTROL_BYTES)
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AggregationError(f"{label} is not strict finite JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"{label} must be a JSON object")
    return value, data


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AggregationError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AggregationError(f"{label} must be an array")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise AggregationError(f"{label} must be a nonempty string")
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AggregationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_size(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AggregationError(f"{label} must be a nonnegative integer")
    return value


def _require_finite_seconds(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise AggregationError(f"{label} must be finite and nonnegative")
    try:
        result = float(value)
    except OverflowError as exc:
        raise AggregationError(f"{label} must be finite and nonnegative") from exc
    if not math.isfinite(result):
        raise AggregationError(f"{label} must be finite and nonnegative")
    return result


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any, label: str) -> dt.datetime:
    text = _require_text(value, label)
    try:
        parsed = dt.datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise AggregationError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise AggregationError(f"{label} must identify a UTC instant")
    return parsed


def _display_path(path: pathlib.Path, repo: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_recorded_path(value: Any, repo: pathlib.Path, label: str) -> pathlib.Path:
    text = _require_text(value, label)
    candidate = pathlib.Path(text)
    return candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()


def _atomic_write_json_noclobber(path: pathlib.Path, value: Any) -> dict[str, Any]:
    encoded = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        view = memoryview(encoded)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise AggregationError(f"short write while publishing {path}")
            view = view[count:]
        os.fsync(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise AggregationError(f"publication destination already exists: {path}") from exc
        except OSError as exc:
            if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise AggregationError(f"publication destination already exists: {path}") from exc
            raise AggregationError(f"cannot publish {path} without replacement: {exc}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {"size_bytes": len(encoded), "sha256": _sha256_bytes(encoded)}
    finally:
        os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _source_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    source = _require_dict(receipt.get("source_end"), "receipt source_end")
    if set(SOURCE_IDENTITY_KEYS) - set(source):
        raise AggregationError("receipt source snapshot lacks sealed identity fields")
    identity = {
        "algorithm": _require_text(source.get("algorithm"), "source algorithm"),
        "file_count": _require_size(source.get("file_count"), "source file_count"),
        "sha256": _require_sha(source.get("sha256"), "source SHA-256"),
        "source_manifest_schema_version": source.get("source_manifest_schema_version"),
        "source_manifest_sha256": _require_sha(
            source.get("source_manifest_sha256"), "source-manifest SHA-256"
        ),
    }
    if (
        not isinstance(identity["source_manifest_schema_version"], int)
        or isinstance(identity["source_manifest_schema_version"], bool)
        or identity["source_manifest_schema_version"] < 1
    ):
        raise AggregationError("source-manifest schema version is invalid")
    return identity


def _compact_source_matches(value: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("available") is True
        and value.get("algorithm") == expected["algorithm"]
        and value.get("file_count") == expected["file_count"]
        and value.get("sha256") == expected["sha256"]
    )


def _verified_receipt(
    receipt_path: pathlib.Path, repo: pathlib.Path
) -> tuple[dict[str, Any], bytes]:
    parsed, raw = _load_json(receipt_path, "build receipt")
    try:
        verified = verify_build_receipt(receipt_path, repo, verify_source=True)
    except (OSError, RuntimeError, ProvenanceError) as exc:
        raise AggregationError(f"build receipt verification failed: {exc}") from exc
    if verified != parsed:
        raise AggregationError("verified build receipt differs from its stable bytes")
    if parsed.get("state") != "complete" or parsed.get("source_unchanged") is not True:
        raise AggregationError("build receipt is not complete and source-stable")
    if parsed.get("source_start") != parsed.get("source_end"):
        raise AggregationError("build receipt source start/end differ")
    _require_sha(parsed.get("receipt_id"), "build receipt ID")
    return parsed, raw


def _normalise_string_set(values: Iterable[str], label: str) -> list[str]:
    result = [_require_text(value, label) for value in values]
    if len(result) != len(set(result)):
        raise AggregationError(f"{label} values must be unique")
    return sorted(result)


def _normalise_batches(
    batch_assignments: Mapping[str, Iterable[tuple[str, str]]]
) -> list[dict[str, Any]]:
    if not isinstance(batch_assignments, Mapping) or not batch_assignments:
        raise AggregationError("at least one batch assignment is required")
    batches: list[dict[str, Any]] = []
    globally_seen: set[tuple[str, str]] = set()
    case_owner: dict[str, str] = {}
    for batch_id, raw_assignments in batch_assignments.items():
        if not isinstance(batch_id, str) or BATCH_ID_RE.fullmatch(batch_id) is None:
            raise AggregationError(f"invalid batch ID: {batch_id!r}")
        assignments: list[dict[str, str]] = []
        locally_seen: set[tuple[str, str]] = set()
        for raw in raw_assignments:
            if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                raise AggregationError(f"batch {batch_id!r} assignment must be (case, backend)")
            case_id = _require_text(raw[0], f"batch {batch_id} case ID")
            backend = _require_text(raw[1], f"batch {batch_id} backend")
            if backend not in {"cpu", "cuda"}:
                raise AggregationError(
                    f"batch {batch_id!r} uses an unsupported runner backend: {backend!r}"
                )
            pair = (case_id, backend)
            if pair in locally_seen:
                raise AggregationError(f"duplicate assignment in batch {batch_id!r}: {pair!r}")
            if pair in globally_seen:
                raise AggregationError(f"assignment appears in multiple batches: {pair!r}")
            previous_owner = case_owner.setdefault(case_id, batch_id)
            if previous_owner != batch_id:
                raise AggregationError(
                    f"case appears in multiple batches: {case_id!r} "
                    f"({previous_owner!r}, {batch_id!r})"
                )
            locally_seen.add(pair)
            globally_seen.add(pair)
            assignments.append({"case_id": case_id, "backend": backend})
        if not assignments:
            raise AggregationError(f"batch {batch_id!r} has no assignments")
        batches.append(
            {
                "batch_id": batch_id,
                "assignments": sorted(
                    assignments, key=lambda item: (item["case_id"], item["backend"])
                ),
            }
        )
    return sorted(batches, key=lambda item: item["batch_id"])


def _manifest_plan_requirements(
    manifest: dict[str, Any], repo: pathlib.Path
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Materialize IDs/scopes without evaluating covered_by pass semantics."""
    try:
        cases = archive_contract.runner_contract.materialize_cases(manifest, repo)
    except archive_contract.runner_contract.ManifestError as exc:
        raise AggregationError(f"validation manifest is invalid: {exc}") from exc
    by_id = {case["id"]: case for case in cases}
    covered = sorted(
        case["id"]
        for case in cases
        if case.get("disposition") == "covered_by_test"
    )
    mpi = sorted(
        case["id"]
        for case in cases
        if case.get("disposition") == "mpi_only"
        or case.get("compute_scope") == "mpi_fdtd_cuda"
    )
    return by_id, covered, mpi


def _bind_manifest_assignment_policy(
    batches: list[dict[str, Any]], catalog: Mapping[str, dict[str, Any]]
) -> None:
    for batch in batches:
        case_backends: dict[str, set[str]] = {}
        for assignment in batch["assignments"]:
            case_backends.setdefault(assignment["case_id"], set()).add(
                assignment["backend"]
            )
        for assignment in batch["assignments"]:
            case_id = assignment["case_id"]
            backend = assignment["backend"]
            case = catalog.get(case_id)
            if case is None:
                raise AggregationError(
                    f"batch assignment names no materialized manifest case: {case_id!r}"
                )
            if case.get("disposition") != "run":
                raise AggregationError(
                    f"batch assignment is not a runnable manifest case: {case_id!r}"
                )
            compute_scope = case.get("compute_scope")
            if compute_scope not in {"fdtd_cuda", "host_only"}:
                raise AggregationError(
                    f"runnable case has unsupported compute scope: {case_id!r}"
                )
            gpu_contract = _require_text(
                case.get("gpu_contract"), f"manifest GPU contract for {case_id}"
            )
            allowed_backends = ["cpu", "cuda"]
            if backend not in allowed_backends:
                raise AggregationError(
                    f"backend {backend!r} is not allowed for manifest case {case_id!r}"
                )
            fdtd_scope = compute_scope == "fdtd_cuda"
            assignment.update(
                {
                    "compute_scope": compute_scope,
                    "gpu_contract": gpu_contract,
                    "allowed_backends": allowed_backends,
                    "counts_toward_gpu_coverage": fdtd_scope and backend == "cuda",
                    "counts_toward_gpu_performance": (
                        fdtd_scope
                        and case_backends[case_id] == {"cpu", "cuda"}
                    ),
                }
            )


def _verify_plan_manifest_binding(
    plan: dict[str, Any], manifest: dict[str, Any], repo: pathlib.Path
) -> None:
    """Re-derive every manifest-owned plan field at aggregation time.

    A canonical plan ID proves only that the plan is internally self-consistent.
    It does not authorize a caller to relabel host-only work as CUDA coverage or
    to omit manifest-derived unresolved gates and then recompute that ID.
    """
    catalog, derived_covered_by, derived_mpi = _manifest_plan_requirements(
        manifest, repo
    )
    assignments = {
        batch["batch_id"]: [
            (assignment["case_id"], assignment["backend"])
            for assignment in batch["assignments"]
        ]
        for batch in plan["batches"]
    }
    rebound_batches = _normalise_batches(assignments)
    _bind_manifest_assignment_policy(rebound_batches, catalog)
    if rebound_batches != plan["batches"]:
        raise AggregationError(
            "sealed plan assignment policy differs from the current manifest"
        )
    expected_unresolved = {
        "covered_by": derived_covered_by,
        "mpi_gate_semantics": derived_mpi,
    }
    if plan["unresolved_inputs"] != expected_unresolved:
        raise AggregationError(
            "sealed plan unresolved inputs differ from the current manifest"
        )


def build_plan(
    *,
    repo: pathlib.Path,
    receipt_path: pathlib.Path,
    manifest_path: pathlib.Path,
    native_artifact_names: Iterable[str],
    batch_assignments: Mapping[str, Iterable[tuple[str, str]]],
    external_gate_ids: Iterable[str] = (),
    unresolved_covered_by: Iterable[str] = (),
    unresolved_mpi_gate_semantics: Iterable[str] = (),
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and seal an in-memory aggregation plan."""
    repo = repo.resolve()
    if not repo.is_dir():
        raise AggregationError(f"repository directory is unavailable: {repo}")
    receipt_path = receipt_path.resolve()
    manifest_path = manifest_path.resolve()
    receipt, receipt_raw = _verified_receipt(receipt_path, repo)
    # A manifest must itself be strict finite JSON even though aggregation only
    # consumes its byte identity.
    manifest, manifest_raw = _load_json(manifest_path, "validation manifest")
    catalog, derived_covered_by, derived_mpi = _manifest_plan_requirements(
        manifest, repo
    )
    batches = _normalise_batches(batch_assignments)
    _bind_manifest_assignment_policy(batches, catalog)

    provided_covered_by = _normalise_string_set(
        unresolved_covered_by, "unresolved covered_by input"
    )
    provided_mpi = _normalise_string_set(
        unresolved_mpi_gate_semantics,
        "unresolved MPI-gate semantics input",
    )
    if provided_covered_by and provided_covered_by != derived_covered_by:
        raise AggregationError(
            "provided covered_by unresolved IDs differ from the materialized manifest"
        )
    if provided_mpi and provided_mpi != derived_mpi:
        raise AggregationError(
            "provided MPI unresolved IDs differ from the materialized manifest"
        )

    names = _normalise_string_set(native_artifact_names, "native artifact name")
    if not names:
        raise AggregationError("at least one native artifact must be sealed")
    unsupported = sorted(set(names) - set(NATIVE_RUNNER_KEYS))
    if unsupported:
        raise AggregationError(
            "native artifacts are not observable in runner runtime identity: "
            + ", ".join(unsupported)
        )
    receipt_artifacts = _require_dict(receipt.get("artifacts"), "receipt artifacts")
    artifacts: list[dict[str, Any]] = []
    for name in names:
        record = _require_dict(receipt_artifacts.get(name), f"receipt artifact {name}")
        path = _resolve_recorded_path(record.get("path"), repo, f"artifact {name} path")
        expected = {
            "size_bytes": _require_size(record.get("size_bytes"), f"artifact {name} size"),
            "sha256": _require_sha(record.get("sha256"), f"artifact {name} SHA-256"),
        }
        if _file_snapshot(path) != expected:
            raise AggregationError(f"native artifact differs from receipt: {name}")
        artifacts.append(
            {
                "name": name,
                "path": _display_path(path, repo),
                **expected,
            }
        )

    timestamp = created_at_utc or _utc_now()
    _parse_utc(timestamp, "plan created_at_utc")
    unsigned = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "created_at_utc": timestamp,
        "identity": {
            "build_receipt": {
                "path": _display_path(receipt_path, repo),
                "receipt_id": receipt["receipt_id"],
                "size_bytes": len(receipt_raw),
                "sha256": _sha256_bytes(receipt_raw),
            },
            "source_snapshot": _source_identity(receipt),
            "manifest": {
                "path": _display_path(manifest_path, repo),
                "size_bytes": len(manifest_raw),
                "sha256": _sha256_bytes(manifest_raw),
            },
            "native_artifacts": artifacts,
        },
        "batches": batches,
        "external_gate_ids": _normalise_string_set(
            external_gate_ids, "external gate ID"
        ),
        "unresolved_inputs": {
            "covered_by": derived_covered_by,
            "mpi_gate_semantics": derived_mpi,
        },
    }
    return {**unsigned, "plan_id": canonical_sha256(unsigned)}


def seal_plan(*, output_path: pathlib.Path, **arguments: Any) -> dict[str, Any]:
    plan = build_plan(**arguments)
    _atomic_write_json_noclobber(output_path.resolve(), plan)
    return plan


def _validate_plan(plan: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "created_at_utc",
        "identity",
        "batches",
        "external_gate_ids",
        "unresolved_inputs",
        "plan_id",
    }
    if set(plan) != expected_keys:
        raise AggregationError("aggregate plan has unexpected or missing fields")
    if (
        type(plan.get("schema_version")) is not int
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("kind") != PLAN_KIND
    ):
        raise AggregationError("aggregate plan schema/kind is unsupported")
    unsigned = dict(plan)
    recorded_id = unsigned.pop("plan_id")
    _require_sha(recorded_id, "plan ID")
    if recorded_id != canonical_sha256(unsigned):
        raise AggregationError("plan ID does not match the plan payload")
    _parse_utc(plan.get("created_at_utc"), "plan created_at_utc")

    identity = _require_dict(plan.get("identity"), "plan identity")
    if set(identity) != {"build_receipt", "source_snapshot", "manifest", "native_artifacts"}:
        raise AggregationError("plan identity fields differ")
    for name in ("build_receipt", "manifest"):
        record = _require_dict(identity.get(name), f"plan {name}")
        expected = {"path", "size_bytes", "sha256"}
        if name == "build_receipt":
            expected.add("receipt_id")
            _require_sha(record.get("receipt_id"), "plan receipt ID")
        if set(record) != expected:
            raise AggregationError(f"plan {name} fields differ")
        _require_text(record.get("path"), f"plan {name} path")
        _require_size(record.get("size_bytes"), f"plan {name} size")
        _require_sha(record.get("sha256"), f"plan {name} SHA-256")
    source = _require_dict(identity.get("source_snapshot"), "plan source snapshot")
    if set(source) != SOURCE_IDENTITY_KEYS:
        raise AggregationError("plan source-snapshot fields differ")
    _source_identity({"source_end": source})

    artifacts = _require_list(identity.get("native_artifacts"), "plan native artifacts")
    artifact_names: list[str] = []
    for index, value in enumerate(artifacts):
        record = _require_dict(value, f"plan native artifact {index}")
        if set(record) != {"name", "path", "size_bytes", "sha256"}:
            raise AggregationError("plan native-artifact fields differ")
        name = _require_text(record.get("name"), "plan native-artifact name")
        if name not in NATIVE_RUNNER_KEYS:
            raise AggregationError(f"plan names an unobservable native artifact: {name}")
        artifact_names.append(name)
        _require_text(record.get("path"), f"plan artifact {name} path")
        _require_size(record.get("size_bytes"), f"plan artifact {name} size")
        _require_sha(record.get("sha256"), f"plan artifact {name} SHA-256")
    if not artifact_names or artifact_names != sorted(set(artifact_names)):
        raise AggregationError("plan native artifacts are empty, duplicated, or unsorted")

    batches = _require_list(plan.get("batches"), "plan batches")
    reconstructed: dict[str, list[tuple[str, str]]] = {}
    for index, value in enumerate(batches):
        batch = _require_dict(value, f"plan batch {index}")
        if set(batch) != {"batch_id", "assignments"}:
            raise AggregationError("plan batch fields differ")
        batch_id = _require_text(batch.get("batch_id"), "plan batch ID")
        if BATCH_ID_RE.fullmatch(batch_id) is None or batch_id in reconstructed:
            raise AggregationError(f"plan batch ID is invalid or duplicated: {batch_id!r}")
        assignments = _require_list(batch.get("assignments"), f"plan batch {batch_id} assignments")
        reconstructed[batch_id] = []
        for assignment_value in assignments:
            assignment = _require_dict(assignment_value, "plan assignment")
            if set(assignment) != {
                "case_id",
                "backend",
                "compute_scope",
                "gpu_contract",
                "allowed_backends",
                "counts_toward_gpu_coverage",
                "counts_toward_gpu_performance",
            }:
                raise AggregationError("plan assignment fields differ")
            case_id = _require_text(assignment.get("case_id"), "plan case ID")
            backend = _require_text(assignment.get("backend"), "plan backend")
            compute_scope = assignment.get("compute_scope")
            _require_text(assignment.get("gpu_contract"), "plan GPU contract")
            allowed_backends = assignment.get("allowed_backends")
            if (
                compute_scope not in {"fdtd_cuda", "host_only"}
                or allowed_backends != ["cpu", "cuda"]
                or backend not in allowed_backends
                or not isinstance(
                    assignment.get("counts_toward_gpu_coverage"), bool
                )
                or not isinstance(
                    assignment.get("counts_toward_gpu_performance"), bool
                )
                or assignment["counts_toward_gpu_coverage"]
                != (compute_scope == "fdtd_cuda" and backend == "cuda")
                or assignment["counts_toward_gpu_performance"]
                and compute_scope != "fdtd_cuda"
            ):
                raise AggregationError("plan assignment compute-scope policy differs")
            reconstructed[batch_id].append((case_id, backend))
    stripped_batches = [
        {
            "batch_id": batch["batch_id"],
            "assignments": [
                {
                    "case_id": assignment["case_id"],
                    "backend": assignment["backend"],
                }
                for assignment in batch["assignments"]
            ],
        }
        for batch in batches
    ]
    if stripped_batches != _normalise_batches(reconstructed):
        raise AggregationError("plan batches are not canonical and globally disjoint")
    for batch in batches:
        backends_by_case: dict[str, set[str]] = {}
        for assignment in batch["assignments"]:
            backends_by_case.setdefault(assignment["case_id"], set()).add(
                assignment["backend"]
            )
        for assignment in batch["assignments"]:
            expected_performance = (
                assignment["compute_scope"] == "fdtd_cuda"
                and backends_by_case[assignment["case_id"]] == {"cpu", "cuda"}
            )
            if assignment["counts_toward_gpu_performance"] != expected_performance:
                raise AggregationError(
                    "plan assignment GPU-performance policy differs"
                )

    gates = _require_list(plan.get("external_gate_ids"), "plan external gates")
    if gates != _normalise_string_set(gates, "external gate ID"):
        raise AggregationError("plan external gate IDs are not canonical")
    unresolved = _require_dict(plan.get("unresolved_inputs"), "plan unresolved inputs")
    if set(unresolved) != {"covered_by", "mpi_gate_semantics"}:
        raise AggregationError("plan unresolved-input fields differ")
    for key in ("covered_by", "mpi_gate_semantics"):
        values = _require_list(unresolved.get(key), f"plan unresolved {key}")
        if values != _normalise_string_set(values, f"unresolved {key} input"):
            raise AggregationError(f"plan unresolved {key} inputs are not canonical")


def _verify_current_identity(plan: dict[str, Any], repo: pathlib.Path) -> dict[str, Any]:
    identity = plan["identity"]
    receipt_record = identity["build_receipt"]
    receipt_path = _resolve_recorded_path(receipt_record["path"], repo, "plan receipt path")
    receipt, receipt_raw = _verified_receipt(receipt_path, repo)
    current_receipt = {"size_bytes": len(receipt_raw), "sha256": _sha256_bytes(receipt_raw)}
    if current_receipt != {
        "size_bytes": receipt_record["size_bytes"],
        "sha256": receipt_record["sha256"],
    } or receipt.get("receipt_id") != receipt_record["receipt_id"]:
        raise AggregationError("current build receipt differs from the sealed plan")
    if _source_identity(receipt) != identity["source_snapshot"]:
        raise AggregationError("current source snapshot differs from the sealed plan")

    manifest_record = identity["manifest"]
    manifest_path = _resolve_recorded_path(manifest_record["path"], repo, "plan manifest path")
    _load_json(manifest_path, "current validation manifest")
    if _file_snapshot(manifest_path) != {
        "size_bytes": manifest_record["size_bytes"],
        "sha256": manifest_record["sha256"],
    }:
        raise AggregationError("current validation manifest differs from the sealed plan")

    receipt_artifacts = _require_dict(receipt.get("artifacts"), "current receipt artifacts")
    for artifact in identity["native_artifacts"]:
        name = artifact["name"]
        current_record = _require_dict(
            receipt_artifacts.get(name), f"current artifact {name}"
        )
        path = _resolve_recorded_path(artifact["path"], repo, f"plan artifact {name} path")
        expected = {
            "size_bytes": artifact["size_bytes"],
            "sha256": artifact["sha256"],
        }
        if (
            current_record.get("size_bytes") != expected["size_bytes"]
            or current_record.get("sha256") != expected["sha256"]
            or _resolve_recorded_path(
                current_record.get("path"),
                repo,
                f"receipt artifact {name} path",
            )
            != path
            or _file_snapshot(path) != expected
        ):
            raise AggregationError(
                f"current native artifact differs from the sealed plan: {name}"
            )
    return receipt


def _safe_member(root: pathlib.Path, value: Any, label: str) -> pathlib.Path:
    text = _require_text(value, label)
    member = pathlib.PurePosixPath(text)
    if member.is_absolute() or member.as_posix() != text or any(
        part in ("", ".", "..") for part in member.parts
    ):
        raise AggregationError(f"{label} must be a normalized relative POSIX path")
    resolved_root = root.resolve()
    resolved = (resolved_root / pathlib.Path(*member.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise AggregationError(f"{label} escapes its batch directory") from exc
    return resolved


def _verify_evidence_record(
    batch_root: pathlib.Path,
    record_value: Any,
    label: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    record = _require_dict(record_value, label)
    if (
        set(record) != {"available", "path", "size_bytes", "sha256"}
        or record.get("available") is not True
    ):
        raise AggregationError(f"{label} is not a complete available file record")
    path = _safe_member(batch_root, record.get("path"), f"{label} path")
    expected = {
        "size_bytes": _require_size(record.get("size_bytes"), f"{label} size"),
        "sha256": _require_sha(record.get("sha256"), f"{label} SHA-256"),
    }
    if _file_snapshot(path) != expected:
        raise AggregationError(f"{label} differs from its reported identity")
    return path, expected


def _verify_report_identity(
    report: dict[str, Any],
    plan: dict[str, Any],
    receipt: dict[str, Any],
    repo: pathlib.Path,
) -> None:
    provenance = _require_dict(report.get("provenance"), "runner provenance")
    expected = plan["identity"]
    if provenance.get("manifest_sha256") != expected["manifest"]["sha256"]:
        raise AggregationError("runner manifest identity differs from the plan")
    if not _compact_source_matches(
        provenance.get("source_snapshot"), expected["source_snapshot"]
    ):
        raise AggregationError("runner source identity differs from the plan")
    repository = _require_text(provenance.get("repository"), "runner repository")
    if pathlib.Path(repository).resolve() != repo.resolve():
        raise AggregationError("runner repository differs from the aggregate repository")

    window = _require_dict(
        provenance.get("validation_window"), "runner validation window"
    )
    if window.get("unchanged") is not True or window.get("problems") not in ([], None):
        raise AggregationError("runner validation window is integrity-invalid")
    for endpoint_name in ("start", "end"):
        endpoint = _require_dict(
            window.get(endpoint_name), f"runner {endpoint_name} snapshot"
        )
        if endpoint.get("available") is not True or endpoint.get("problems") not in (
            [],
            None,
        ):
            raise AggregationError(f"runner {endpoint_name} snapshot is unavailable")
        if not _compact_source_matches(
            endpoint.get("source_snapshot"), expected["source_snapshot"]
        ):
            raise AggregationError(f"runner {endpoint_name} source differs from the plan")
        receipt_evidence = _require_dict(
            endpoint.get("build_receipt"), f"runner {endpoint_name} receipt"
        )
        if (
            receipt_evidence.get("available") is not True
            or receipt_evidence.get("sha256") != expected["build_receipt"]["sha256"]
            or receipt_evidence.get("receipt_id") != expected["build_receipt"]["receipt_id"]
            or receipt_evidence.get("build_input_id") != receipt.get("build_input_id")
            or receipt_evidence.get("artifact_set_id") != receipt.get("artifact_set_id")
        ):
            raise AggregationError(f"runner {endpoint_name} receipt differs from the plan")
        runtime = _require_dict(
            endpoint.get("runtime_contract"), f"runner {endpoint_name} runtime contract"
        )
        if runtime.get("receipt_id") != expected["build_receipt"]["receipt_id"]:
            raise AggregationError(f"runner {endpoint_name} runtime receipt ID differs")
        for artifact in expected["native_artifacts"]:
            runner_key = NATIVE_RUNNER_KEYS[artifact["name"]]
            observed = _require_dict(
                runtime.get(runner_key),
                f"runner {endpoint_name} native artifact {artifact['name']}",
            )
            observed_path = _resolve_recorded_path(
                observed.get("path"), repo, f"runner artifact {artifact['name']} path"
            )
            expected_path = _resolve_recorded_path(
                artifact["path"], repo, f"plan artifact {artifact['name']} path"
            )
            if (
                observed_path != expected_path
                or observed.get("size_bytes") != artifact["size_bytes"]
                or observed.get("sha256") != artifact["sha256"]
            ):
                raise AggregationError(
                    f"runner native artifact differs from the plan: {artifact['name']}"
                )


def _assignment_set(batch: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (assignment["case_id"], assignment["backend"])
        for assignment in batch["assignments"]
    }


def _verify_runner_batch(
    *,
    batch: dict[str, Any],
    root: pathlib.Path,
    plan: dict[str, Any],
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    receipt_source_records: Mapping[str, Any],
    repo: pathlib.Path,
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise AggregationError(f"batch directory is unavailable or a symlink: {root}")
    report_path = root / "report.json"
    markdown_path = root / "report.md"
    complete_path = root / "COMPLETE"
    report, report_raw = _load_json(report_path, f"batch {batch['batch_id']} report")
    complete, complete_raw = _load_json(complete_path, f"batch {batch['batch_id']} COMPLETE")
    if (
        set(report) != REPORT_KEYS
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != RUNNER_SCHEMA_VERSION
    ):
        raise AggregationError("runner report schema or top-level fields differ")
    if (
        set(complete) != COMPLETE_KEYS
        or type(complete.get("schema_version")) is not int
        or complete.get("schema_version") != RUNNER_SCHEMA_VERSION
    ):
        raise AggregationError("runner COMPLETE schema or fields differ")
    try:
        archive_contract._validate_runner_control(report, complete)
        archive_contract._collect_report_evidence(report, root.resolve())
        # This authoritative replay validates only the runner's selected
        # runnable cases: commands, isolated environment, CUDA/CPU statistics,
        # backend counters, unittest evidence, and CPU/CUDA comparisons.  It
        # deliberately does not turn covered_by dispositions into aggregate
        # coverage; those graph semantics remain explicit unresolved inputs.
        archive_contract._replay_report_semantics(
            report,
            manifest,
            root.resolve(),
            receipt,
            receipt_source_records,
            bind_environment_to_evidence_root=True,
        )
    except archive_contract.ArchiveError as exc:
        raise AggregationError(
            f"batch {batch['batch_id']} is not structurally valid runner evidence: {exc}"
        ) from exc
    if report.get("exit_code") != 0 or complete.get("exit_code") != 0:
        raise AggregationError(f"batch {batch['batch_id']} runner did not pass")
    if type(report.get("exit_code")) is not int or type(complete.get("exit_code")) is not int:
        raise AggregationError("runner/COMPLETE exit codes must be integers")
    report_snapshot = {"size_bytes": len(report_raw), "sha256": _sha256_bytes(report_raw)}
    markdown_snapshot = _file_snapshot(markdown_path)
    _require_sha(complete.get("report_sha256"), "runner COMPLETE report SHA-256")
    _require_sha(complete.get("markdown_sha256"), "runner COMPLETE Markdown SHA-256")
    _require_size(complete.get("report_size_bytes"), "runner COMPLETE report size")
    _require_size(complete.get("markdown_size_bytes"), "runner COMPLETE Markdown size")
    if (
        complete.get("report_sha256") != report_snapshot["sha256"]
        or complete.get("report_size_bytes") != report_snapshot["size_bytes"]
        or complete.get("markdown_sha256") != markdown_snapshot["sha256"]
        or complete.get("markdown_size_bytes") != markdown_snapshot["size_bytes"]
    ):
        raise AggregationError(f"batch {batch['batch_id']} COMPLETE does not bind runner output")

    plan_time = _parse_utc(plan.get("created_at_utc"), "plan created_at_utc")
    generated = _parse_utc(report.get("generated_at_utc"), "runner generated_at_utc")
    completed = _parse_utc(complete.get("completed_at_utc"), "runner completed_at_utc")
    latest_allowed = dt.datetime.now(dt.timezone.utc) + MAX_FUTURE_CLOCK_SKEW
    if generated < plan_time or completed < generated or completed > latest_allowed:
        raise AggregationError(f"batch {batch['batch_id']} is stale or has inverted timestamps")

    configuration = _require_dict(report.get("configuration"), "runner configuration")
    if set(configuration) != CONFIGURATION_KEYS:
        raise AggregationError("runner configuration fields differ")
    performance = _require_dict(
        configuration.get("performance_evidence"), "runner performance evidence"
    )
    if set(performance) != PERFORMANCE_KEYS:
        raise AggregationError("runner performance-evidence fields differ")
    configured_backends = configuration.get("backends")
    if (
        not isinstance(configured_backends, list)
        or not configured_backends
        or not all(isinstance(value, str) and value for value in configured_backends)
        or len(configured_backends) != len(set(configured_backends))
    ):
        raise AggregationError("runner configured backends are invalid")
    expected_backends = sorted({backend for _, backend in _assignment_set(batch)})
    if configured_backends != expected_backends:
        raise AggregationError("runner configured backends differ from the batch plan")
    _verify_report_identity(report, plan, receipt, repo)
    window = report["provenance"]["validation_window"]
    validation_start = _parse_utc(
        window["start"].get("captured_at_utc"),
        "runner validation-start captured_at_utc",
    )
    validation_end = _parse_utc(
        window["end"].get("captured_at_utc"),
        "runner validation-end captured_at_utc",
    )
    provenance_time = _parse_utc(
        report["provenance"].get("captured_at_utc"),
        "runner provenance captured_at_utc",
    )
    if not (
        plan_time
        <= validation_start
        <= provenance_time
        <= validation_end
        <= generated
        <= completed
        <= latest_allowed
    ):
        raise AggregationError("runner validation/report timestamp order differs")

    results = _require_list(report.get("results"), "runner results")
    expected = _assignment_set(batch)
    assignment_policy = {
        (assignment["case_id"], assignment["backend"]): assignment
        for assignment in batch["assignments"]
    }
    actual: set[tuple[str, str]] = set()
    seen_result_ids: set[str] = set()
    seen_evidence_paths: set[pathlib.Path] = set()
    timings: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    selected_count = 0
    for index, result_value in enumerate(results):
        result = _require_dict(result_value, f"runner result {index}")
        allowed_result_keys = set(RESULT_KEYS)
        if "missing_dependencies" in result:
            allowed_result_keys.add("missing_dependencies")
        if set(result) != allowed_result_keys:
            raise AggregationError("runner result fields differ")
        case_id = _require_text(result.get("id"), f"runner result {index} ID")
        if case_id in seen_result_ids:
            raise AggregationError(f"duplicate case ID in runner report: {case_id}")
        seen_result_ids.add(case_id)
        outcome = _require_text(result.get("outcome"), f"runner result {case_id} outcome")
        outcomes[outcome] += 1
        selected = result.get("selected")
        if not isinstance(selected, bool):
            raise AggregationError(f"runner result {case_id} selected flag is invalid")
        runs = _require_dict(result.get("runs"), f"runner result {case_id} runs")
        if not selected and runs:
            raise AggregationError(f"unselected runner case has executions: {case_id}")
        if selected:
            selected_count += 1
            if case_id not in {pair[0] for pair in expected}:
                raise AggregationError(f"batch contains an extra selected case: {case_id}")
        for backend, run_value in runs.items():
            _require_text(backend, f"runner result {case_id} backend")
            pair = (case_id, backend)
            if pair in actual:
                raise AggregationError(f"duplicate case/backend run in batch: {pair!r}")
            actual.add(pair)
            if backend not in configured_backends:
                raise AggregationError(f"runner executed an unconfigured backend: {backend}")
            run = _require_dict(run_value, f"runner run {case_id}/{backend}")
            if set(run) != archive_contract.RUN_RECORD_KEYS:
                raise AggregationError(
                    f"runner run fields differ from runner schema: {case_id}/{backend}"
                )
            if run.get("backend") != backend:
                raise AggregationError(f"runner backend identity differs: {case_id}/{backend}")
            duration = _require_finite_seconds(
                run.get("duration_seconds"), f"runner duration {case_id}/{backend}"
            )
            run_started = _parse_utc(
                run.get("started_at_utc"),
                f"runner start timestamp {case_id}/{backend}",
            )
            if not validation_start <= run_started <= validation_end:
                raise AggregationError(
                    f"runner start lies outside validation window: {case_id}/{backend}"
                )
            run_outcome = _require_text(
                run.get("outcome"), f"runner outcome {case_id}/{backend}"
            )
            if (
                run_outcome not in {"PASS", "PASS_WITH_DECLARED_SKIPS"}
                or run.get("exit_code") != 0
                or run.get("timeout") is not False
                or run.get("output_limit") is not None
                or run.get("backend_contract_ok") is not True
                or run.get("backend_contract_problems") not in ([], None)
            ):
                raise AggregationError(
                    f"assigned runner execution did not pass: {case_id}/{backend}"
                )
            evidence_files = _require_dict(
                run.get("evidence_files"), f"runner evidence {case_id}/{backend}"
            )
            if set(evidence_files) != {"stdout", "stderr", "statistics"}:
                raise AggregationError(
                    f"runner evidence file set differs: {case_id}/{backend}"
                )
            direct_keys = {
                "stdout": "stdout_log",
                "stderr": "stderr_log",
                "statistics": "statistics_file",
            }
            for evidence_name, direct_key in direct_keys.items():
                evidence_path, _ = _verify_evidence_record(
                    root,
                    evidence_files[evidence_name],
                    f"runner {case_id}/{backend} {evidence_name}",
                )
                if run.get(direct_key) != evidence_files[evidence_name].get("path"):
                    raise AggregationError(
                        f"runner {case_id}/{backend} direct evidence path differs"
                    )
                if evidence_path in seen_evidence_paths:
                    raise AggregationError(f"runner evidence path is duplicated: {evidence_path}")
                seen_evidence_paths.add(evidence_path)
                if evidence_name == "statistics":
                    statistics, _ = _load_json(
                        evidence_path,
                        f"runner statistics {case_id}/{backend}",
                    )
                    if statistics != run.get("statistics"):
                        raise AggregationError(
                            f"runner statistics payload differs: {case_id}/{backend}"
                        )
            environment_contract = _require_dict(
                run.get("environment_contract"),
                f"runner environment contract {case_id}/{backend}",
            )
            if set(environment_contract) != archive_contract.RUN_ENVIRONMENT_CONTRACT_KEYS:
                raise AggregationError(
                    f"runner environment-contract fields differ: {case_id}/{backend}"
                )
            environment = _require_dict(
                environment_contract.get("environment"),
                f"runner exact environment {case_id}/{backend}",
            )
            if (
                not all(
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in environment.items()
                )
                or environment_contract.get("keys") != sorted(environment)
                or environment_contract.get("sha256")
                != archive_contract.runner_contract.environment_sha256(environment)
                or environment_contract.get("MEEP_GPU_BACKEND") != backend
                or environment_contract.get("strict_cuda") != (backend == "cuda")
            ):
                raise AggregationError(
                    f"runner exact environment identity differs: {case_id}/{backend}"
                )
            timings.append(
                {
                    "case_id": case_id,
                    "backend": backend,
                    "duration_seconds": duration,
                    "compute_scope": assignment_policy[pair]["compute_scope"],
                    "counts_toward_gpu_coverage": assignment_policy[pair][
                        "counts_toward_gpu_coverage"
                    ],
                    "counts_toward_gpu_performance": assignment_policy[pair][
                        "counts_toward_gpu_performance"
                    ],
                }
            )

        if selected and runs:
            comparison = _require_dict(
                result.get("comparison"), f"runner comparison {case_id}"
            )
            if set(runs) == {"cpu", "cuda"}:
                if comparison.get("outcome") != "PASS" or outcome != "PASS":
                    raise AggregationError(
                        f"runner CPU/CUDA comparison did not pass: {case_id}"
                    )
            elif (
                comparison.get("outcome") != "SINGLE_BACKEND_ONLY"
                or outcome != "SINGLE_BACKEND_ONLY"
            ):
                raise AggregationError(
                    f"runner single-backend result semantics differ: {case_id}"
                )

    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AggregationError(
            f"batch {batch['batch_id']} assignment coverage differs "
            f"(missing={missing}, extra={extra})"
        )
    inventory = _require_dict(report.get("inventory"), "runner inventory")
    if (
        type(inventory.get("total")) is not int
        or type(inventory.get("selected")) is not int
        or inventory.get("total") != len(results)
        or inventory.get("selected") != selected_count
    ):
        raise AggregationError("runner inventory counts differ from results")
    summary = _require_dict(report.get("summary"), "runner summary")
    if not all(
        isinstance(key, str)
        and key
        and type(value) is int
        and value >= 0
        for key, value in summary.items()
    ):
        raise AggregationError("runner summary value types are invalid")
    if summary != dict(sorted(outcomes.items())):
        raise AggregationError("runner summary differs from result outcomes")
    return {
        "batch_id": batch["batch_id"],
        "source_directory": str(root.resolve()),
        "generated_at_utc": report["generated_at_utc"],
        "completed_at_utc": complete["completed_at_utc"],
        "report": report_snapshot,
        "complete": {"size_bytes": len(complete_raw), "sha256": _sha256_bytes(complete_raw)},
        "markdown": markdown_snapshot,
        "assignments": batch["assignments"],
        "timings": sorted(timings, key=lambda item: (item["case_id"], item["backend"])),
    }


def build_aggregate(
    *,
    repo: pathlib.Path,
    plan_path: pathlib.Path,
    batch_directories: Mapping[str, pathlib.Path],
) -> dict[str, Any]:
    """Verify exact fresh batch coverage and return aggregate evidence."""
    repo = repo.resolve()
    plan_path = plan_path.resolve()
    plan, plan_raw = _load_json(plan_path, "aggregate plan")
    _validate_plan(plan)
    receipt = _verify_current_identity(plan, repo)
    manifest_path = _resolve_recorded_path(
        plan["identity"]["manifest"]["path"], repo, "plan manifest path"
    )
    manifest, _ = _load_json(manifest_path, "sealed validation manifest")
    _verify_plan_manifest_binding(plan, manifest, repo)
    try:
        receipt_source_records = archive_contract._receipt_source_records(receipt)
    except archive_contract.ArchiveError as exc:
        raise AggregationError(f"receipt source manifest is invalid: {exc}") from exc
    expected_batches = {batch["batch_id"]: batch for batch in plan["batches"]}
    if not isinstance(batch_directories, Mapping):
        raise AggregationError("batch directories must be a mapping")
    if set(batch_directories) != set(expected_batches):
        missing = sorted(set(expected_batches) - set(batch_directories))
        extra = sorted(set(batch_directories) - set(expected_batches))
        raise AggregationError(
            f"batch set differs from the plan (missing={missing}, extra={extra})"
        )
    resolved_roots = [pathlib.Path(value).resolve() for value in batch_directories.values()]
    if len(resolved_roots) != len(set(resolved_roots)):
        raise AggregationError("multiple batch IDs reference the same runner directory")
    for index, left in enumerate(resolved_roots):
        for right in resolved_roots[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise AggregationError("runner batch directories must not be nested")

    evidence = []
    observed: set[tuple[str, str]] = set()
    for batch_id in sorted(expected_batches):
        record = _verify_runner_batch(
            batch=expected_batches[batch_id],
            root=pathlib.Path(batch_directories[batch_id]),
            plan=plan,
            receipt=receipt,
            manifest=manifest,
            receipt_source_records=receipt_source_records,
            repo=repo,
        )
        pairs = {
            (assignment["case_id"], assignment["backend"])
            for assignment in record["assignments"]
        }
        overlap = observed.intersection(pairs)
        if overlap:
            raise AggregationError(f"batch reports overlap assignments: {sorted(overlap)}")
        observed.update(pairs)
        evidence.append(record)
    expected_all = {
        (assignment["case_id"], assignment["backend"])
        for batch in plan["batches"]
        for assignment in batch["assignments"]
    }
    if observed != expected_all:
        raise AggregationError("aggregate assignment coverage is incomplete")
    all_assignments = [
        assignment
        for batch in plan["batches"]
        for assignment in batch["assignments"]
    ]

    # Close the verification window over mutable inputs.  The second pass must
    # reproduce byte-identical evidence records, then the authoritative build,
    # source, manifest, and native artifacts are verified once more.
    repeated_evidence = []
    for batch_id in sorted(expected_batches):
        repeated_evidence.append(
            _verify_runner_batch(
                batch=expected_batches[batch_id],
                root=pathlib.Path(batch_directories[batch_id]),
                plan=plan,
                receipt=receipt,
                manifest=manifest,
                receipt_source_records=receipt_source_records,
                repo=repo,
            )
        )
    if repeated_evidence != evidence:
        raise AggregationError("runner batch evidence changed during aggregation")
    repeated_receipt = _verify_current_identity(plan, repo)
    if repeated_receipt != receipt:
        raise AggregationError("build/source identity changed during aggregation")

    unresolved = plan["unresolved_inputs"]
    has_unresolved = bool(
        unresolved["covered_by"]
        or unresolved["mpi_gate_semantics"]
        or plan["external_gate_ids"]
    )
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "kind": AGGREGATE_KIND,
        "generated_at_utc": _utc_now(),
        "status": (
            "ASSIGNED_RUNS_PASS_WITH_UNRESOLVED_INPUTS"
            if has_unresolved
            else "ASSIGNED_RUNS_PASS"
        ),
        "plan": {
            "path": str(plan_path),
            "plan_id": plan["plan_id"],
            "size_bytes": len(plan_raw),
            "sha256": _sha256_bytes(plan_raw),
        },
        "identity": plan["identity"],
        "coverage": {
            "expected_assignment_count": len(expected_all),
            "observed_assignment_count": len(observed),
            "batch_count": len(evidence),
            "all_assigned_runs_pass": True,
            "gpu_coverage_assignment_count": sum(
                int(assignment["counts_toward_gpu_coverage"])
                for assignment in all_assignments
            ),
            "gpu_performance_assignment_count": sum(
                int(assignment["counts_toward_gpu_performance"])
                for assignment in all_assignments
            ),
            "host_only_assignment_count": sum(
                int(assignment["compute_scope"] == "host_only")
                for assignment in all_assignments
            ),
            "host_only_excluded_from_gpu_coverage_and_performance": True,
        },
        "batches": evidence,
        "external_gate_ids": plan["external_gate_ids"],
        "external_gate_evaluation": "IDENTIFIERS_ONLY_NOT_EVALUATED",
        "verification_passes": 2,
        "unresolved_inputs": unresolved,
        "unimplemented_semantics": [
            "covered_by dependency-graph resolution",
            "MPI/external-gate pass/fail evaluation",
        ],
    }


def _stable_recheck_before_complete(
    *,
    aggregate: dict[str, Any],
    repo: pathlib.Path,
    plan_path: pathlib.Path,
    batch_directories: Mapping[str, pathlib.Path],
) -> None:
    """Reread every bound mutable input immediately before COMPLETE."""
    plan, plan_raw = _load_json(plan_path.resolve(), "aggregate plan recheck")
    _validate_plan(plan)
    if (
        plan.get("plan_id") != aggregate["plan"]["plan_id"]
        or len(plan_raw) != aggregate["plan"]["size_bytes"]
        or _sha256_bytes(plan_raw) != aggregate["plan"]["sha256"]
    ):
        raise AggregationError("aggregate plan changed before COMPLETE")
    expected_batches = {record["batch_id"]: record for record in aggregate["batches"]}
    if set(batch_directories) != set(expected_batches):
        raise AggregationError("runner batch set changed before COMPLETE")
    for batch_id in sorted(expected_batches):
        root = pathlib.Path(batch_directories[batch_id])
        expected = expected_batches[batch_id]
        report, report_raw = _load_json(root / "report.json", "runner report recheck")
        complete, complete_raw = _load_json(root / "COMPLETE", "runner COMPLETE recheck")
        if (
            {"size_bytes": len(report_raw), "sha256": _sha256_bytes(report_raw)}
            != expected["report"]
            or {
                "size_bytes": len(complete_raw),
                "sha256": _sha256_bytes(complete_raw),
            }
            != expected["complete"]
            or _file_snapshot(root / "report.md") != expected["markdown"]
        ):
            raise AggregationError(
                f"runner control evidence changed before COMPLETE: {batch_id}"
            )
        try:
            archive_contract._validate_runner_control(report, complete)
            archive_contract._collect_report_evidence(report, root.resolve())
        except archive_contract.ArchiveError as exc:
            raise AggregationError(
                f"runner raw evidence changed before COMPLETE: {batch_id}: {exc}"
            ) from exc
    _verify_current_identity(plan, repo.resolve())


def aggregate_to_directory(
    *,
    output_dir: pathlib.Path,
    **arguments: Any,
) -> dict[str, Any]:
    aggregate = build_aggregate(**arguments)
    output_dir = output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise AggregationError(f"aggregate output directory already exists: {output_dir}") from exc
    report_snapshot = _atomic_write_json_noclobber(output_dir / "report.json", aggregate)
    _stable_recheck_before_complete(
        aggregate=aggregate,
        repo=pathlib.Path(arguments["repo"]),
        plan_path=pathlib.Path(arguments["plan_path"]),
        batch_directories=arguments["batch_directories"],
    )
    complete = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "kind": "gpmeep-validation-aggregate-complete",
        "completed_at_utc": _utc_now(),
        "status": aggregate["status"],
        "plan_id": aggregate["plan"]["plan_id"],
        "plan_sha256": aggregate["plan"]["sha256"],
        "report_sha256": report_snapshot["sha256"],
        "report_size_bytes": report_snapshot["size_bytes"],
    }
    _atomic_write_json_noclobber(output_dir / "COMPLETE", complete)
    return aggregate


def _parse_assignment(value: str) -> tuple[str, str, str]:
    try:
        batch_id, remainder = value.split("=", 1)
        case_id, backend = remainder.rsplit("@", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("assignment must be BATCH=CASE_ID@BACKEND") from exc
    if not batch_id or not case_id or not backend:
        raise argparse.ArgumentTypeError("assignment must be BATCH=CASE_ID@BACKEND")
    return batch_id, case_id, backend


def _parse_batch_directory(value: str) -> tuple[str, pathlib.Path]:
    try:
        batch_id, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch directory must be BATCH=PATH") from exc
    if not batch_id or not path:
        raise argparse.ArgumentTypeError("batch directory must be BATCH=PATH")
    return batch_id, pathlib.Path(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-plan", allow_abbrev=False)
    seal.add_argument("--repo", type=pathlib.Path, required=True)
    seal.add_argument("--receipt", type=pathlib.Path, required=True)
    seal.add_argument("--manifest", type=pathlib.Path, required=True)
    seal.add_argument(
        "--native-artifact",
        action="append",
        default=[],
        choices=sorted(NATIVE_RUNNER_KEYS),
        help="receipt artifact observable in runner evidence; repeatable",
    )
    seal.add_argument("--assign", action="append", type=_parse_assignment, required=True)
    seal.add_argument("--external-gate-id", action="append", default=[])
    seal.add_argument("--unresolved-covered-by", action="append", default=[])
    seal.add_argument("--unresolved-mpi-gate", action="append", default=[])
    seal.add_argument("--output", type=pathlib.Path, required=True)

    aggregate = subparsers.add_parser("aggregate", allow_abbrev=False)
    aggregate.add_argument("--repo", type=pathlib.Path, required=True)
    aggregate.add_argument("--plan", type=pathlib.Path, required=True)
    aggregate.add_argument(
        "--batch", action="append", type=_parse_batch_directory, required=True
    )
    aggregate.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        if args.command == "seal-plan":
            assignments: dict[str, list[tuple[str, str]]] = {}
            for batch_id, case_id, backend in args.assign:
                assignments.setdefault(batch_id, []).append((case_id, backend))
            plan = seal_plan(
                output_path=args.output,
                repo=args.repo,
                receipt_path=args.receipt,
                manifest_path=args.manifest,
                native_artifact_names=args.native_artifact or sorted(NATIVE_RUNNER_KEYS),
                batch_assignments=assignments,
                external_gate_ids=args.external_gate_id,
                unresolved_covered_by=args.unresolved_covered_by,
                unresolved_mpi_gate_semantics=args.unresolved_mpi_gate,
            )
            print(f"sealed plan: {args.output.resolve()}")
            print(f"plan ID: {plan['plan_id']}")
            return 0

        directories: dict[str, pathlib.Path] = {}
        for batch_id, path in args.batch:
            if batch_id in directories:
                raise AggregationError(f"duplicate --batch ID: {batch_id}")
            directories[batch_id] = path
        aggregate = aggregate_to_directory(
            output_dir=args.output,
            repo=args.repo,
            plan_path=args.plan,
            batch_directories=directories,
        )
        print(f"aggregate report: {(args.output / 'report.json').resolve()}")
        print(f"status: {aggregate['status']}")
        return 0
    except AggregationError as exc:
        print(f"aggregation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
