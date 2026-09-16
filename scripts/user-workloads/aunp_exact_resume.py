#!/usr/bin/env python3
"""Fail-closed stage state for the exact AuNP workload.

The archived AuNP engine writes useful completion markers, but a marker is not
an adapter checkpoint.  This module makes the durable boundary explicit: an
exclusive, hash-chained journal event containing a complete file manifest.
It deliberately has no Meep dependency so recovery logic can be mutation
tested without allocating an FDTD grid.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import stat
from typing import Any, Iterable

from common import StableFile, WorkloadError, atomic_write_json, file_record


JOURNAL_EVENT_SCHEMA = "gpmeep-aunp-exact-stage-event-v3"
CHECKPOINT_SCHEMA = "gpmeep-aunp-exact-stage-checkpoint-v3"
STAGE_ORDER = (
    "TM_Ex-reference",
    "TM_Ex-structure",
    "postprocessing",
)
BASE_REQUIRED_PATHS = (
    "simulation_config.json",
    "run_environment.json",
)
REFERENCE_REQUIRED_BASENAMES = (
    "reference_complete.json",
    "reference_data.npz",
    "reference_dense_probes.npz",
    "reference_dense_line_metadata.npz",
    "reference_monitor1.h5",
    "reference_monitor2.h5",
    "reference_bottom_flux.h5",
    "ref_line_vertical.h5",
    "ref_line_gap.h5",
    "ref_line_top.h5",
)
STRUCTURE_REQUIRED_BASENAMES = (
    "structure_complete.json",
    "spectra.npz",
    "field_metadata.npz",
    "raw_complex_fields.h5",
    "structure_dense_probes.npz",
    "structure_dense_line_metadata.npz",
    "material_masks.npz",
    "str_scat_m1.h5",
    "str_scat_m2.h5",
    "str_bottom_flux.h5",
    "str_line_vertical.h5",
    "str_line_gap.h5",
    "str_line_top.h5",
)
POSTPROCESS_REQUIRED_ROOT_BASENAMES = (
    "postprocess_complete.json",
    "output_manifest.json",
)
EVENT_TYPES = {
    "initialized",
    "attempt-started",
    "base-sealed",
    "quarantine-sealed",
    "stage-sealed",
    "publication-sealed",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: Any, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value:
        raise WorkloadError(f"{label} path is not a nonempty string")
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise WorkloadError(f"{label} path is unsafe: {value!r}")
    if "\\" in value:
        raise WorkloadError(f"{label} path is non-portable: {value!r}")
    return relative


def inventory_files(root: pathlib.Path) -> list[str]:
    """Return every regular file below *root*, rejecting links and specials."""

    root = pathlib.Path(os.path.abspath(os.fspath(root)))
    if not root.exists():
        return []
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise WorkloadError(f"AuNP inventory root is not a directory: {root}")
    result: list[str] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = pathlib.Path(directory)
        for name in list(directory_names):
            candidate = current / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise WorkloadError(f"AuNP output contains a directory symlink: {candidate}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise WorkloadError(f"AuNP output contains a special entry: {candidate}")
        for name in file_names:
            candidate = current / name
            metadata = candidate.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise WorkloadError(f"AuNP output contains a non-regular file: {candidate}")
            result.append(candidate.relative_to(root).as_posix())
    return sorted(result)


def make_manifest(
    root: pathlib.Path, relative_paths: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    root = pathlib.Path(os.path.abspath(os.fspath(root)))
    paths = inventory_files(root) if relative_paths is None else sorted(set(relative_paths))
    records = []
    for raw in paths:
        relative = _safe_relative(raw, "AuNP manifest")
        records.append(file_record(root.joinpath(*relative.parts), root))
    return records


def manifest_paths(manifest: Any, label: str) -> set[str]:
    if not isinstance(manifest, list):
        raise WorkloadError(f"{label} manifest is not a list")
    result: set[str] = set()
    for record in manifest:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise WorkloadError(f"{label} manifest record schema differs")
        relative = _safe_relative(record["path"], label).as_posix()
        size = record["size_bytes"]
        digest = record["sha256"]
        if (
            relative in result
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise WorkloadError(f"{label} manifest record is invalid or duplicated")
        result.add(relative)
    return result


def replay_manifest(
    root: pathlib.Path,
    manifest: Any,
    label: str,
    *,
    exact_inventory: bool = False,
) -> set[str]:
    root = pathlib.Path(os.path.abspath(os.fspath(root)))
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise WorkloadError(f"{label} root is unavailable: {root}: {exc}") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise WorkloadError(f"{label} root is not a non-symlink directory: {root}")
    paths = manifest_paths(manifest, label)
    records = {record["path"]: record for record in manifest}
    for relative in sorted(paths):
        path = root.joinpath(*pathlib.PurePosixPath(relative).parts)
        with StableFile(path, f"{label} {relative}", records[relative]):
            pass
    if exact_inventory:
        actual = set(inventory_files(root))
        if actual != paths:
            raise WorkloadError(
                f"{label} inventory differs: missing={sorted(paths-actual)}, "
                f"extra={sorted(actual-paths)}"
            )
    return paths


def stage_relative_paths(stage: str) -> tuple[str, ...]:
    if stage not in STAGE_ORDER:
        raise WorkloadError(f"unknown AuNP exact stage: {stage}")
    if stage == "postprocessing":
        return POSTPROCESS_REQUIRED_ROOT_BASENAMES
    directory, kind = stage.rsplit("-", 1)
    names = (
        REFERENCE_REQUIRED_BASENAMES
        if kind == "reference"
        else STRUCTURE_REQUIRED_BASENAMES
    )
    return tuple(f"{directory}/{name}" for name in names)


def seal_fdtd_stage(
    simulation_root: pathlib.Path,
    stage: str,
    completed_stage_manifests: Iterable[Any],
) -> list[dict[str, Any]]:
    if stage == "postprocessing" or stage not in STAGE_ORDER:
        raise WorkloadError(f"not an AuNP FDTD stage: {stage}")
    required = set(stage_relative_paths(stage))
    directory = stage.split("-", 1)[0]
    prior: set[str] = set()
    for index, manifest in enumerate(completed_stage_manifests):
        prior.update(manifest_paths(manifest, f"prior stage {index}"))
    expected_directory = {
        path for path in prior | required if pathlib.PurePosixPath(path).parts[0] == directory
    }
    actual_directory = {
        f"{directory}/{path}"
        for path in inventory_files(simulation_root / directory)
    }
    if actual_directory != expected_directory:
        raise WorkloadError(
            f"{stage} file ownership differs: "
            f"missing={sorted(expected_directory-actual_directory)}, "
            f"extra={sorted(actual_directory-expected_directory)}"
        )
    return make_manifest(simulation_root, required)


def seal_postprocessing(simulation_root: pathlib.Path) -> list[dict[str, Any]]:
    paths = set(inventory_files(simulation_root))
    required = set(stage_relative_paths("postprocessing"))
    if not required.issubset(paths):
        raise WorkloadError(
            "AuNP postprocessing root files are missing: " + ", ".join(sorted(required - paths))
        )
    return make_manifest(simulation_root, paths)


def quarantine_unsealed_files(
    simulation_root: pathlib.Path,
    allowed_paths: Iterable[str],
    quarantine_parent: pathlib.Path,
    attempt_index: int,
    reason: str,
) -> dict[str, Any] | None:
    """Move only files outside the last verified namespace into quarantine."""

    if type(attempt_index) is not int or attempt_index < 0 or not reason:
        raise WorkloadError("invalid AuNP quarantine identity")
    simulation_root = pathlib.Path(os.path.abspath(os.fspath(simulation_root)))
    allowed = {
        _safe_relative(path, "AuNP allowed namespace").as_posix()
        for path in allowed_paths
    }
    actual = set(inventory_files(simulation_root))
    extras = sorted(actual - allowed)
    if not extras:
        if simulation_root.exists():
            for directory, _, _ in os.walk(simulation_root, topdown=False):
                path = pathlib.Path(directory)
                if path != simulation_root:
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        return None
    quarantine_parent = pathlib.Path(os.path.abspath(os.fspath(quarantine_parent)))
    if quarantine_parent.exists() and quarantine_parent.is_symlink():
        raise WorkloadError(
            f"AuNP quarantine parent must not be a symlink: {quarantine_parent}"
        )
    quarantine_parent.mkdir(parents=True, exist_ok=True)
    destination = quarantine_parent / f"attempt-{attempt_index:03d}"
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise WorkloadError(f"AuNP quarantine already exists: {destination}") from exc
    modified_directories: set[pathlib.Path] = {destination, quarantine_parent}
    for relative in extras:
        source = simulation_root.joinpath(*pathlib.PurePosixPath(relative).parts)
        target = destination.joinpath(*pathlib.PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise WorkloadError(f"AuNP quarantine target already exists: {target}")
        os.replace(source, target)
        modified_directories.update((source.parent, target.parent))
    for directory, _, _ in os.walk(simulation_root, topdown=False):
        path = pathlib.Path(directory)
        if path != simulation_root:
            try:
                path.rmdir()
            except OSError:
                pass
    for directory in sorted(modified_directories):
        if not directory.exists():
            continue
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return {
        "attempt_index": attempt_index,
        "reason": reason,
        "root": destination.relative_to(quarantine_parent.parent).as_posix(),
        "manifest": make_manifest(destination),
    }


def _event_file(events_directory: pathlib.Path, sequence: int) -> pathlib.Path:
    return events_directory / f"{sequence:08d}.json"


def _write_exclusive(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise WorkloadError("AuNP journal write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_events(events_directory: pathlib.Path) -> list[dict[str, Any]]:
    if not events_directory.exists():
        return []
    if events_directory.is_symlink() or not events_directory.is_dir():
        raise WorkloadError("AuNP stage journal is not a directory")
    names = sorted(path.name for path in events_directory.iterdir())
    expected = [f"{index:08d}.json" for index in range(len(names))]
    if names != expected:
        raise WorkloadError("AuNP stage journal sequence is not contiguous and exact")
    events: list[dict[str, Any]] = []
    previous = None
    for index, name in enumerate(names):
        path = events_directory / name
        with StableFile(path, f"AuNP journal event {index}") as stable:
            with stable.file_object() as handle:
                payload = handle.read()
        try:
            event = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkloadError(f"AuNP journal event {index} is invalid JSON") from exc
        if payload != canonical_json_bytes(event):
            raise WorkloadError(f"AuNP journal event {index} is not canonical")
        if not isinstance(event, dict) or set(event) != {
            "schema",
            "sequence",
            "previous_event_sha256",
            "contract_sha256",
            "event_type",
            "payload",
        }:
            raise WorkloadError(f"AuNP journal event {index} schema differs")
        if (
            event["schema"] != JOURNAL_EVENT_SCHEMA
            or event["sequence"] != index
            or event["previous_event_sha256"] != previous
            or event["event_type"] not in EVENT_TYPES
            or not isinstance(event["payload"], dict)
            or not isinstance(event["contract_sha256"], str)
            or len(event["contract_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in event["contract_sha256"]
            )
        ):
            raise WorkloadError(f"AuNP journal event {index} identity differs")
        digest = sha256_bytes(payload)
        events.append({"event": event, "file_sha256": digest, "size_bytes": len(payload)})
        previous = digest
    return events


def append_event(
    events_directory: pathlib.Path,
    contract_sha256: str,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if event_type not in EVENT_TYPES or not isinstance(payload, dict):
        raise WorkloadError("invalid AuNP stage journal event")
    events = load_events(events_directory)
    if events and events[0]["event"]["contract_sha256"] != contract_sha256:
        raise WorkloadError("AuNP stage journal contract differs")
    if any(item["event"]["contract_sha256"] != contract_sha256 for item in events):
        raise WorkloadError("AuNP stage journal contains mixed contracts")
    sequence = len(events)
    event = {
        "schema": JOURNAL_EVENT_SCHEMA,
        "sequence": sequence,
        "previous_event_sha256": events[-1]["file_sha256"] if events else None,
        "contract_sha256": contract_sha256,
        "event_type": event_type,
        "payload": payload,
    }
    encoded = canonical_json_bytes(event)
    _write_exclusive(_event_file(events_directory, sequence), encoded)
    return {
        "event": event,
        "file_sha256": sha256_bytes(encoded),
        "size_bytes": len(encoded),
    }


def derive_checkpoint(contract: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    contract_sha256 = sha256_bytes(canonical_json_bytes(contract))
    if not events:
        raise WorkloadError("AuNP stage journal is empty")
    initialized = events[0]["event"]
    if (
        initialized["event_type"] != "initialized"
        or initialized["contract_sha256"] != contract_sha256
        or initialized["payload"] != {"contract": contract}
    ):
        raise WorkloadError("AuNP stage journal initialization differs")
    attempts: list[dict[str, Any]] = []
    base_manifest = None
    completed_stages: list[dict[str, Any]] = []
    quarantines: list[dict[str, Any]] = []
    publication = None
    for item in events[1:]:
        event = item["event"]
        if event["contract_sha256"] != contract_sha256:
            raise WorkloadError("AuNP stage journal contract changed")
        kind = event["event_type"]
        payload = event["payload"]
        if publication is not None:
            raise WorkloadError("AuNP journal continues after terminal publication")
        if kind == "attempt-started":
            if set(payload) != {"attempt_index", "started_utc"} or payload[
                "attempt_index"
            ] != len(attempts) or not isinstance(payload["started_utc"], str) or not payload[
                "started_utc"
            ]:
                raise WorkloadError("AuNP attempt journal order differs")
            attempts.append(payload)
        elif kind == "base-sealed":
            if (
                not attempts
                or base_manifest is not None
                or set(payload) != {"manifest"}
            ):
                raise WorkloadError("AuNP base manifest was sealed more than once")
            paths = manifest_paths(payload["manifest"], "AuNP base")
            if paths != set(BASE_REQUIRED_PATHS):
                raise WorkloadError(
                    "AuNP TM-only base manifest differs: "
                    f"missing={sorted(set(BASE_REQUIRED_PATHS)-paths)}, "
                    f"extra={sorted(paths-set(BASE_REQUIRED_PATHS))}"
                )
            base_manifest = payload["manifest"]
        elif kind == "quarantine-sealed":
            if set(payload) != {"attempt_index", "reason", "root", "manifest"}:
                raise WorkloadError("AuNP quarantine event schema differs")
            if (
                type(payload["attempt_index"]) is not int
                or payload["attempt_index"] < 0
                or payload["attempt_index"] >= len(attempts)
                or not isinstance(payload["reason"], str)
                or not payload["reason"]
            ):
                raise WorkloadError("AuNP quarantine references a future attempt")
            _safe_relative(payload["root"], "AuNP quarantine")
            manifest_paths(payload["manifest"], "AuNP quarantine")
            quarantines.append(payload)
        elif kind == "stage-sealed":
            if set(payload) != {
                "stage",
                "attempt_index",
                "manifest",
                "rank_records",
                "wall_seconds",
            }:
                raise WorkloadError("AuNP stage event schema differs")
            if base_manifest is None:
                raise WorkloadError("AuNP exact stage was sealed before its base")
            if len(completed_stages) >= len(STAGE_ORDER):
                raise WorkloadError("AuNP journal contains too many sealed stages")
            expected_stage = STAGE_ORDER[len(completed_stages)]
            if (
                payload["stage"] != expected_stage
                or type(payload["attempt_index"]) is not int
                or payload["attempt_index"] < 0
                or payload["attempt_index"] >= len(attempts)
                or isinstance(payload["wall_seconds"], bool)
                or not isinstance(payload["wall_seconds"], (int, float))
                or not math.isfinite(float(payload["wall_seconds"]))
                or payload["wall_seconds"] < 0
                or not isinstance(payload["rank_records"], list)
            ):
                raise WorkloadError("AuNP sealed stage order or timing differs")
            manifest_paths(payload["manifest"], f"AuNP {expected_stage}")
            if expected_stage == "postprocessing":
                if payload["rank_records"]:
                    raise WorkloadError("AuNP postprocessing unexpectedly has rank records")
            elif not payload["rank_records"]:
                raise WorkloadError("AuNP FDTD stage has no rank records")
            completed_stages.append(payload)
        elif kind == "publication-sealed":
            if (
                publication is not None
                or len(completed_stages) != len(STAGE_ORDER)
                or set(payload) != {"attempt_index", "summary", "complete"}
                or type(payload["attempt_index"]) is not int
                or payload["attempt_index"] < 0
                or payload["attempt_index"] >= len(attempts)
            ):
                raise WorkloadError("AuNP publication event is out of order")
            manifest_paths(
                [payload["summary"], payload["complete"]],
                "AuNP terminal publication",
            )
            publication = payload
        elif kind != "initialized":
            raise WorkloadError(f"unsupported AuNP journal event type: {kind}")
        else:
            raise WorkloadError("AuNP journal was initialized more than once")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "contract": contract,
        "journal_event_count": len(events),
        "journal_terminal_sha256": events[-1]["file_sha256"],
        "attempts": attempts,
        "base_manifest": base_manifest,
        "completed_stages": completed_stages,
        "quarantines": quarantines,
        "publication": publication,
    }


def publish_checkpoint(
    checkpoint_path: pathlib.Path,
    contract: dict[str, Any],
    events_directory: pathlib.Path,
) -> dict[str, Any]:
    checkpoint = derive_checkpoint(contract, load_events(events_directory))
    atomic_write_json(checkpoint_path, checkpoint)
    return checkpoint


def replay_checkpoint(
    checkpoint_path: pathlib.Path,
    contract: dict[str, Any],
    events_directory: pathlib.Path,
    *,
    permit_trailing_durable_events: bool = True,
) -> dict[str, Any]:
    events = load_events(events_directory)
    derived = derive_checkpoint(contract, events)
    if not checkpoint_path.exists():
        if not permit_trailing_durable_events:
            raise WorkloadError("AuNP stage checkpoint is missing")
        atomic_write_json(checkpoint_path, derived)
        return derived
    with StableFile(checkpoint_path, "AuNP stage checkpoint") as stable:
        try:
            with stable.file_object() as handle:
                stored = json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkloadError("AuNP stage checkpoint is invalid JSON") from exc
    if stored == derived:
        return derived
    count = stored.get("journal_event_count") if isinstance(stored, dict) else None
    if (
        permit_trailing_durable_events
        and type(count) is int
        and 0 < count < len(events)
        and stored == derive_checkpoint(contract, events[:count])
    ):
        atomic_write_json(checkpoint_path, derived)
        return derived
    raise WorkloadError("AuNP stage checkpoint disagrees with its durable journal")


def replay_completed_output(
    simulation_root: pathlib.Path, checkpoint: dict[str, Any]
) -> set[str]:
    allowed: set[str] = set()
    base = checkpoint.get("base_manifest")
    if base is not None:
        allowed.update(replay_manifest(simulation_root, base, "AuNP base"))
    stages = checkpoint.get("completed_stages")
    if not isinstance(stages, list):
        raise WorkloadError("AuNP checkpoint stage inventory is invalid")
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or stage.get("stage") != STAGE_ORDER[index]:
            raise WorkloadError("AuNP checkpoint stage order differs")
        allowed.update(
            replay_manifest(
                simulation_root,
                stage.get("manifest"),
                f"AuNP sealed {stage['stage']}",
            )
        )
    if stages and stages[-1]["stage"] == "postprocessing":
        replay_manifest(
            simulation_root,
            stages[-1]["manifest"],
            "AuNP final output",
            exact_inventory=True,
        )
    return allowed
