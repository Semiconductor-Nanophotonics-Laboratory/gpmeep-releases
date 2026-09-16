#!/usr/bin/env python3
"""Compare retained CPU/GPU results from the two user-supplied workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import stat
import sys
import tempfile
from itertools import product
from typing import Any, Iterator

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    AUNP_OUTPUT_POLARIZATIONS,
    AUNP_PHYSICAL_OVERRIDES,
    AUNP_EXPECTED_DEFAULT_CONFIG,
    MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME,
    TERS_SHA256,
    TERS_PERFORMANCE_ADAPTATION,
    TERS_PERFORMANCE_MEASUREMENT_MODE,
    WorkloadError,
    StableFile,
    aunp_qualification_profile,
    atomic_write_json,
    atomic_write_text,
    extract_verified_aunp,
    file_record,
    regular_file,
    sha256_file,
    validate_dft_coordinate_metadata,
    validate_phase_contract,
    validate_runtime_against_receipt,
)
import aunp_exact_resume as exact_resume  # noqa: E402


PAIR_TOLERANCES = {
    "fp64-fp32": {
        "field_nrmse": 5e-3,
        "field_peak_linf": 2e-2,
        "field_energy_relative": 1e-2,
        "field_roi_percentile_relative": 3e-2,
        "field_peak_relative": 5e-2,
        "phase_rms": 1e-2,
        "spectral_max_abs": 5e-4,
        "spectral_rmse": 1e-4,
        "coefficient_nrmse": 1e-3,
        "resonance_nm": 0.5,
        "resonance_fwhm_nm": 1.0,
        "resonance_q": 0.25,
        "resonance_scalar_relative": 2e-2,
        "resonance_phase_rad": 2e-2,
    },
    "same-fp32": {
        "field_nrmse": 2.5e-3,
        "field_peak_linf": 1e-2,
        "field_energy_relative": 5e-3,
        "field_roi_percentile_relative": 1.5e-2,
        "field_peak_relative": 2.5e-2,
        "phase_rms": 1e-2,
        "spectral_max_abs": 2e-4,
        "spectral_rmse": 5e-5,
        "coefficient_nrmse": 5e-4,
        "resonance_nm": 0.25,
        "resonance_fwhm_nm": 0.5,
        "resonance_q": 0.12,
        "resonance_scalar_relative": 1e-2,
        "resonance_phase_rad": 1e-2,
    },
    "expected-fp64": {
        "spectral_max_abs": 1e-4,
        "spectral_rmse": 5e-5,
        "coefficient_nrmse": 5e-4,
        "resonance_nm": 0.5,
        "resonance_fwhm_nm": 1.0,
        "resonance_q": 0.25,
        "resonance_scalar_relative": 2e-2,
        "resonance_phase_rad": 2e-2,
    },
}

TERS_SCHEMA = "gpmeep-ters-workload-v2"
AUNP_SCHEMA = "gpmeep-aunp-r4000-workload-v4"
CHUNK_VALUES = 1024 * 1024

COMPARISON_CLASSES = {
    "cpu-fp64-fp32": {
        "precision": (False, True),
        "reference_backend": "cpu",
        "candidate_backend": "cpu",
        "tolerance": "fp64-fp32",
        "coordinate_weight_policy": "symmetric-relative",
        "coordinate_weight_rtol": 5.0e-7,
    },
    "cuda-fp64-fp32": {
        "precision": (False, True),
        "reference_backend": "cpu",
        "candidate_backend": "cuda",
        "tolerance": "fp64-fp32",
        "coordinate_weight_policy": "symmetric-relative",
        "coordinate_weight_rtol": 5.0e-7,
    },
    "cuda-same-fp32": {
        "precision": (True, True),
        "reference_backend": "cpu",
        "candidate_backend": "cuda",
        "tolerance": "same-fp32",
        "coordinate_weight_policy": "symmetric-relative",
        "coordinate_weight_rtol": 5.0e-7,
    },
    "repeat-cpu-fp64": {
        "precision": (False, False),
        "reference_backend": "cpu",
        "candidate_backend": "cpu",
        "tolerance": "same-fp32",
        "coordinate_weight_policy": "byte-exact",
        "coordinate_weight_rtol": 0.0,
    },
    "repeat-cpu-fp32": {
        "precision": (True, True),
        "reference_backend": "cpu",
        "candidate_backend": "cpu",
        "tolerance": "same-fp32",
        "coordinate_weight_policy": "byte-exact",
        "coordinate_weight_rtol": 0.0,
    },
    "repeat-cuda-fp32": {
        "precision": (True, True),
        "reference_backend": "cuda",
        "candidate_backend": "cuda",
        "tolerance": "same-fp32",
        "coordinate_weight_policy": "byte-exact",
        "coordinate_weight_rtol": 0.0,
    },
}

AUNP_KEYS = {
    "wavelength_nm",
    "frequency",
    "incident_flux",
    "reflected_flux_monitor1",
    "reflected_flux_monitor2",
    "reflectance_monitor1",
    "reflectance_monitor2",
    "bottom_flux",
    "reference_bottom_flux",
    "bottom_residual_fraction",
    "absorptance_opaque",
    "absorptance_to_bottom",
    "mode_coefficients_monitor1",
    "mode_coefficients_monitor2",
    "reflection_coefficient_monitor1",
    "reflection_coefficient_monitor2",
}

SPECTRAL_POWER_KEYS = {
    "reflectance_monitor1",
    "reflectance_monitor2",
    "absorptance_opaque",
    "absorptance_to_bottom",
    "bottom_residual_fraction",
}
SPECTRAL_COMPLEX_KEYS = {
    "mode_coefficients_monitor1",
    "mode_coefficients_monitor2",
    "reflection_coefficient_monitor1",
    "reflection_coefficient_monitor2",
}


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise WorkloadError(f"non-finite {label}")
    return result


def _record_path(root: pathlib.Path, record: dict[str, Any]) -> pathlib.Path:
    raw = record.get("path")
    if not isinstance(raw, str):
        raise WorkloadError("artifact record path is missing")
    relative = pathlib.PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise WorkloadError(f"unsafe artifact record path: {raw!r}")
    lexical_path = root.joinpath(*relative.parts)
    # Reject symlinks in every component below the sealed lane root.  Checking
    # only the final file is insufficient because an attacker could replace a
    # parent directory after the record was written.
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise WorkloadError(f"recorded artifact is unavailable: {current}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise WorkloadError(f"recorded artifact traverses a symlink: {current}")
    path = regular_file(lexical_path, "recorded artifact")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkloadError(f"recorded artifact escapes lane root: {path}") from exc
    if path.stat().st_size != int(record.get("size_bytes", -1)):
        raise WorkloadError(f"recorded artifact size mismatch: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise WorkloadError(f"recorded artifact digest mismatch: {path}")
    return path


def _stable_record(root: pathlib.Path, record: dict[str, Any], label: str) -> StableFile:
    path = _record_path(root, record)
    return StableFile(path, label, expected=record)


def _stable_artifact(value: Any, label: str) -> StableFile:
    if isinstance(value, pathlib.Path):
        return StableFile(value, label)
    if not isinstance(value, dict) or not isinstance(value.get("root"), pathlib.Path):
        raise WorkloadError(f"invalid stable artifact descriptor for {label}")
    return _stable_record(value["root"], value["record"], label)


def _external_record_path(record: dict[str, Any], label: str) -> pathlib.Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not pathlib.Path(raw).is_absolute():
        raise WorkloadError(f"{label} must use an absolute path")
    path = regular_file(pathlib.Path(raw), label)
    if path.stat().st_size != int(record.get("size_bytes", -1)):
        raise WorkloadError(f"{label} size mismatch: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise WorkloadError(f"{label} digest mismatch: {path}")
    return path


def validate_lane_provenance(lane: dict[str, Any]) -> dict[str, Any]:
    summary = lane["summary"]
    repo = SCRIPT_DIR.parents[1]
    receipt_record = summary.get("build_receipt")
    if not isinstance(receipt_record, dict):
        raise WorkloadError("lane has no build receipt binding")
    receipt_path = _external_record_path(receipt_record, "build receipt")
    scripts_dir = repo / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from gpmeep_provenance import ProvenanceError, verify_build_receipt

        receipt = verify_build_receipt(receipt_path, repo)
    except (ImportError, ProvenanceError) as exc:
        raise WorkloadError(f"could not verify lane build receipt: {exc}") from exc
    for key in ("receipt_id", "build_input_id", "artifact_set_id", "source_start"):
        if receipt_record.get(key) != receipt.get(key):
            raise WorkloadError(f"lane build receipt {key} binding differs")

    adapter = _record_path(repo, summary.get("adapter", {}))
    common = _record_path(repo, summary.get("adapter_common", {}))
    expected_adapter = {
        TERS_SCHEMA: SCRIPT_DIR / "run_ters_workload.py",
        AUNP_SCHEMA: SCRIPT_DIR / "run_aunp_workload.py",
    }[summary["schema"]]
    if adapter != expected_adapter or common != SCRIPT_DIR / "common.py":
        raise WorkloadError("lane was produced by an unexpected workload adapter")
    exact_stage_state = None
    if summary["schema"] == AUNP_SCHEMA:
        exact_stage_state = _record_path(
            repo, summary.get("exact_stage_state_code", {})
        )
        if exact_stage_state != SCRIPT_DIR / "aunp_exact_resume.py":
            raise WorkloadError(
                "AuNP lane was produced by unexpected exact-stage recovery code"
            )

    device_identifiers: dict[int, str] = {}
    claimed_cores: set[tuple[str, int, int]] = set()
    for rank_record in summary["rank_records"]:
        rank = int(rank_record["rank"])
        if rank_record.get("build_receipt_id") != receipt["receipt_id"]:
            raise WorkloadError(f"rank {rank} has a different build receipt ID")
        runtime = rank_record.get("runtime_artifacts")
        if not isinstance(runtime, dict):
            raise WorkloadError(f"rank {rank} has no runtime artifact evidence")
        for name, record in runtime.items():
            _external_record_path(record, f"rank {rank} runtime {name}")
        validate_runtime_against_receipt(runtime, receipt, repo)
        physical_cores = rank_record.get("physical_cores")
        if not isinstance(physical_cores, list) or len(physical_cores) != 1:
            raise WorkloadError(
                f"rank {rank} must be pinned to exactly one physical CPU core"
            )
        core_record = physical_cores[0]
        if not isinstance(core_record, dict) or set(core_record) != {
            "hostname",
            "package_id",
            "core_id",
        }:
            raise WorkloadError(f"rank {rank} has invalid CPU affinity evidence")
        core = (
            str(core_record["hostname"]),
            int(core_record["package_id"]),
            int(core_record["core_id"]),
        )
        if core in claimed_cores:
            raise WorkloadError("two workload ranks share one physical CPU core")
        claimed_cores.add(core)

        identifiers = {record.get("device_identifier") for record in rank_record["records"]}
        ordinals = {record.get("device_ordinal") for record in rank_record["records"]}
        if summary["expected_backend"] == "cuda":
            if len(identifiers) != 1 or len(ordinals) != 1:
                raise WorkloadError(f"rank {rank} changed GPU during its workload")
            identifier = next(iter(identifiers))
            ordinal = next(iter(ordinals))
            normalized = (
                identifier.lower().removeprefix("gpu-").replace("-", "")
                if isinstance(identifier, str)
                else ""
            )
            if len(normalized) != 32 or any(character not in "0123456789abcdef" for character in normalized):
                raise WorkloadError(f"rank {rank} has an invalid physical GPU UUID")
            if type(ordinal) is not int or ordinal < 0:
                raise WorkloadError(f"rank {rank} has an invalid GPU ordinal")
            device_identifiers[rank] = normalized
        elif identifiers != {None} or ordinals != {-1}:
            raise WorkloadError(f"CPU rank {rank} retained a selected GPU")
    if len(set(device_identifiers.values())) != len(device_identifiers):
        raise WorkloadError("multiple CUDA ranks were mapped to the same physical GPU")
    result = {
        "receipt_id": receipt["receipt_id"],
        "build_input_id": receipt.get("build_input_id"),
        "artifact_set_id": receipt.get("artifact_set_id"),
        "source_start": receipt["source_start"],
        "adapter_sha256": sha256_file(adapter),
        "common_sha256": sha256_file(common),
        "gpu_devices": device_identifiers,
        "physical_cores": [list(core) for core in sorted(claimed_cores)],
    }
    if exact_stage_state is not None:
        result["exact_stage_state_sha256"] = sha256_file(exact_stage_state)
    return result


def load_lane(
    root: pathlib.Path, schema: str, expected_single_precision: bool | None = None
) -> dict[str, Any]:
    root = root.resolve()
    complete_path = root / "COMPLETE"
    with StableFile(complete_path, "workload COMPLETE marker") as stable_complete:
        with stable_complete.file_object() as handle:
            complete = json.load(handle)
    expected_complete_schema = {
        TERS_SCHEMA: "gpmeep-ters-workload-complete-v2",
        AUNP_SCHEMA: "gpmeep-aunp-r4000-workload-complete-v4",
    }[schema]
    if complete.get("schema") != expected_complete_schema:
        raise WorkloadError(f"wrong workload COMPLETE schema in {root}")
    summary_record = complete.get("summary", {})
    summary_path = _record_path(root, summary_record)
    if summary_path != (root / "summary.json").resolve():
        raise WorkloadError("COMPLETE does not bind the canonical summary.json")
    with _stable_record(root, summary_record, "workload summary") as stable_summary:
        with stable_summary.file_object() as handle:
            summary = json.load(handle)
    if summary.get("schema") != schema:
        raise WorkloadError(f"wrong workload summary schema in {root}")
    single_precision = summary.get("single_precision")
    if not isinstance(single_precision, bool):
        raise WorkloadError(f"lane precision is not explicitly recorded: {root}")
    if (
        expected_single_precision is not None
        and single_precision is not expected_single_precision
    ):
        precision = "FP32" if expected_single_precision else "FP64"
        raise WorkloadError(f"comparison requires an explicit {precision} lane: {root}")
    mpi_size = int(summary.get("mpi_size", 0))
    ranks = summary.get("rank_records")
    if mpi_size <= 0 or not isinstance(ranks, list) or len(ranks) != mpi_size:
        raise WorkloadError(f"invalid rank inventory in {root}")
    if sorted(int(item.get("rank", -1)) for item in ranks) != list(range(mpi_size)):
        raise WorkloadError(f"non-canonical rank inventory in {root}")
    expected_backend = summary.get("expected_backend")
    if expected_backend not in ("cpu", "cuda"):
        raise WorkloadError(f"lane has no exact expected backend: {root}")
    expected_runs = {TERS_SCHEMA: 2, AUNP_SCHEMA: 2}[schema]
    for rank_record in ranks:
        records = rank_record.get("records")
        if not isinstance(records, list) or len(records) != expected_runs:
            raise WorkloadError(
                f"rank {rank_record.get('rank')} has an invalid run inventory: {root}"
            )
        for run_index, record in enumerate(records):
            if not isinstance(record, dict) or record.get("run_index") != run_index:
                raise WorkloadError(f"non-canonical run record in {root}")
            if (
                record.get("active_backend") != expected_backend
                or record.get("requested_backend") != expected_backend
            ):
                raise WorkloadError(f"run backend/request does not match lane in {root}")
            if (
                type(record.get("timestep_delta")) is not int
                or record["timestep_delta"] <= 0
                or finite_float(record.get("wall_seconds"), "run wall time") <= 0
                or not math.isfinite(float(record.get("meep_time")))
            ):
                raise WorkloadError(f"run did not retain positive finite work in {root}")
        timing_name = (
            "exact_end_to_end_seconds" if schema == TERS_SCHEMA else "process_wall_seconds"
        )
        if finite_float(rank_record.get(timing_name), timing_name) <= 0:
            raise WorkloadError(f"rank has no positive {timing_name} in {root}")
    validate_phase_contract(ranks, expected_backend, summary.get("phase_contract"))
    if schema == TERS_SCHEMA:
        source_path = _external_record_path(summary.get("source", {}), "TERS source")
        if (
            sha256_file(source_path) != TERS_SHA256
            or summary.get("source", {}).get("sha256") != TERS_SHA256
            or summary.get("source_sha256") != TERS_SHA256
        ):
            raise WorkloadError("TERS source record is not byte-exact")
        source_name = pathlib.PurePosixPath(summary.get("source", {}).get("path", "")).name
        expected_names = {
            source_name.removesuffix(".py") + "_backup.py",
            "withtip_geo.png",
            "withtip_dft.csv",
            "withtip_dft.png",
            "withtip_dft.npy",
            "wotip_geo.png",
            "wotip_dft.csv",
            "wotip_dft.png",
            "wotip_dft.npy",
        }
        for rank_record in ranks:
            records = rank_record.get("original_outputs")
            if not isinstance(records, list):
                raise WorkloadError("TERS rank original output evidence is absent")
            paths = [_record_path(root, record) for record in records]
            if {path.name for path in paths} != expected_names or len(
                {path.parent for path in paths}
            ) != 1:
                raise WorkloadError("TERS rank original output inventory is not exact")
            result_relative = pathlib.PurePosixPath(
                str(rank_record.get("original_result_root", ""))
            )
            expected_prefix = (
                "work",
                f"rank-{int(rank_record['rank']):05d}",
                "results",
            )
            if (
                result_relative.is_absolute()
                or ".." in result_relative.parts
                or "." in result_relative.parts
                or len(result_relative.parts) != 4
                or result_relative.parts[:3] != expected_prefix
                or not result_relative.parts[3].startswith(f"{source_name}_")
            ):
                raise WorkloadError("TERS rank result directory is not canonical")
            result_directory = root.joinpath(*result_relative.parts)
            current = root
            for part in result_relative.parts:
                current = current / part
                if stat.S_ISLNK(current.lstat().st_mode):
                    raise WorkloadError("TERS rank result directory traverses a symlink")
            if not result_directory.is_dir() or {path.parent for path in paths} != {
                result_directory.resolve()
            }:
                raise WorkloadError("TERS rank outputs do not bind their result directory")
            backup = next(path for path in paths if path.name.endswith("_backup.py"))
            if sha256_file(backup) != summary.get("source_sha256"):
                raise WorkloadError("TERS rank backup is not the byte-exact source")
    return {"root": root, "summary": summary}


def _validate_precision_pair(
    reference: dict[str, Any], candidate: dict[str, Any], comparison_class: str
) -> None:
    contract = COMPARISON_CLASSES.get(comparison_class)
    if contract is None:
        raise WorkloadError(f"unknown comparison class: {comparison_class}")
    reference_fp32 = reference["summary"]["single_precision"]
    candidate_fp32 = candidate["summary"]["single_precision"]
    expected = contract["precision"]
    if (reference_fp32, candidate_fp32) != expected:
        actual = (
            "FP32" if reference_fp32 else "FP64",
            "FP32" if candidate_fp32 else "FP64",
        )
        wanted = tuple("FP32" if value else "FP64" for value in expected)
        raise WorkloadError(
            f"{comparison_class} requires {wanted[0]} reference and "
            f"{wanted[1]} candidate, got {actual[0]} and {actual[1]}"
        )


def _validate_backend_roles(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    label: str,
    comparison_class: str,
) -> None:
    contract = COMPARISON_CLASSES.get(comparison_class)
    if contract is None:
        raise WorkloadError(f"unknown comparison class: {comparison_class}")
    if reference["summary"].get("expected_backend") != contract["reference_backend"]:
        raise WorkloadError(
            f"comparison class {comparison_class} requires a "
            f"{contract['reference_backend']} reference"
        )
    if candidate["summary"].get("expected_backend") != contract["candidate_backend"]:
        raise WorkloadError(
            f"comparison candidate {label} must be a {contract['candidate_backend']} lane"
        )


def _comparison_tolerance(comparison_class: str) -> dict[str, float]:
    contract = COMPARISON_CLASSES.get(comparison_class)
    if contract is None:
        raise WorkloadError(f"unknown comparison class: {comparison_class}")
    return PAIR_TOLERANCES[contract["tolerance"]]


def parse_candidate(value: str) -> tuple[str, str, pathlib.Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not parts[0] or parts[1] not in COMPARISON_CLASSES:
        raise argparse.ArgumentTypeError(
            "candidate must be LABEL:COMPARISON_CLASS:PATH with a declared class"
        )
    return parts[0], parts[1], pathlib.Path(parts[2])


def _chunks(array: np.ndarray) -> Iterator[np.ndarray]:
    flat = array.reshape(-1)
    for start in range(0, flat.size, CHUNK_VALUES):
        yield np.asarray(flat[start : start + CHUNK_VALUES])


def _safe_relative_path(value: Any, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str):
        raise WorkloadError(f"{label} path is missing")
    relative = pathlib.PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in value
    ):
        raise WorkloadError(f"unsafe {label} path: {value!r}")
    return relative


def load_aunp_output_manifest(lane: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Replay every retained AuNP file record and HDF5 descriptor.

    This intentionally hashes the large raw field files.  A release comparison
    must not trust the producer's JSON merely because spectra.npz is intact.
    """

    try:
        import h5py
    except ImportError as exc:
        raise WorkloadError("h5py is required to replay AuNP output evidence") from exc

    lane_root = lane["root"]
    summary = lane["summary"]
    simulation_relative = _safe_relative_path(
        summary.get("simulation_root"), "simulation root"
    )
    simulation_root = lane_root.joinpath(*simulation_relative.parts)
    if not simulation_root.is_dir() or simulation_root.is_symlink():
        raise WorkloadError(f"invalid AuNP simulation root: {simulation_root}")
    manifest_record = summary.get("output_manifest", {})
    manifest_path = _record_path(lane_root, manifest_record)
    expected_manifest_path = simulation_root / "output_manifest.json"
    if manifest_path != expected_manifest_path:
        raise WorkloadError("AuNP summary does not bind the canonical output manifest")
    with _stable_record(
        lane_root, manifest_record, "AuNP output manifest"
    ) as stable_manifest:
        with stable_manifest.file_object() as handle:
            manifest = json.load(handle)
    records = manifest.get("files")
    if not isinstance(records, list):
        raise WorkloadError("AuNP output manifest has no file inventory")
    if int(manifest.get("file_count_excluding_manifest", -1)) != len(records):
        raise WorkloadError("AuNP output manifest file count is inconsistent")

    by_path: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for record in records:
        if not isinstance(record, dict):
            raise WorkloadError("AuNP output manifest contains a non-object record")
        relative = _safe_relative_path(record.get("path"), "AuNP output")
        name = relative.as_posix()
        if name == "output_manifest.json" or name in by_path:
            raise WorkloadError(f"duplicate or recursive AuNP manifest record: {name}")
        path = _record_path(simulation_root, record)
        total_bytes += path.stat().st_size
        descriptor = record.get("hdf5_datasets")
        if path.suffix == ".h5":
            if not isinstance(descriptor, dict):
                raise WorkloadError(f"HDF5 descriptor is missing for {name}")
            with _stable_record(
                simulation_root, record, f"AuNP HDF5 {name}"
            ) as stable, stable.file_object() as stream, h5py.File(
                stream, "r"
            ) as handle:
                actual_descriptor = {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in handle.items()
                    if isinstance(value, h5py.Dataset)
                }
            if actual_descriptor != descriptor:
                raise WorkloadError(f"HDF5 descriptor changed for {name}")
        elif descriptor is not None:
            raise WorkloadError(f"non-HDF5 record carries an HDF5 descriptor: {name}")
        by_path[name] = {
            "path": path,
            "root": simulation_root,
            "record": record,
        }

    if total_bytes != int(manifest.get("total_bytes_excluding_manifest", -1)):
        raise WorkloadError("AuNP output manifest byte total is inconsistent")

    actual_paths: set[str] = set()
    for path in simulation_root.rglob("*"):
        if path.is_symlink():
            raise WorkloadError(f"AuNP output contains a symlink: {path}")
        if path.is_file() and path != manifest_path:
            actual_paths.add(path.relative_to(simulation_root).as_posix())
    if actual_paths != set(by_path):
        raise WorkloadError(
            "AuNP output inventory differs from its manifest: missing="
            f"{sorted(set(by_path) - actual_paths)}, extra="
            f"{sorted(actual_paths - set(by_path))}"
        )
    return by_path


def validate_aunp_simulation_config(
    lane: dict[str, Any], artifacts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Bind the engine manifest to the exact package defaults and lane policy."""

    summary = lane["summary"]
    if summary.get("default_config") != AUNP_EXPECTED_DEFAULT_CONFIG:
        raise WorkloadError("AuNP default physical configuration is not exact")
    effective = summary.get("effective_config")
    if not isinstance(effective, dict):
        raise WorkloadError("AuNP effective configuration is missing")
    differing = {
        key
        for key in set(effective) | set(AUNP_EXPECTED_DEFAULT_CONFIG)
        if effective.get(key) != AUNP_EXPECTED_DEFAULT_CONFIG.get(key)
    }
    if differing != {"output_dir", "required_mpi_ranks", *AUNP_PHYSICAL_OVERRIDES}:
        raise WorkloadError(
            f"AuNP effective configuration changed forbidden keys: {sorted(differing)}"
        )
    if any(effective.get(name) != value for name, value in AUNP_PHYSICAL_OVERRIDES.items()):
        raise WorkloadError("AuNP effective DFT stop-condition overrides differ")
    if effective["required_mpi_ranks"] != summary["mpi_size"]:
        raise WorkloadError("AuNP effective configuration has the wrong MPI size")
    if (
        summary.get("qualification_profile") != aunp_qualification_profile()
        or summary.get("physical_overrides") != AUNP_PHYSICAL_OVERRIDES
    ):
        raise WorkloadError("AuNP user-selected qualification profile differs")
    overrides = summary.get("execution_policy_overrides", {})
    if overrides.get("required_mpi_ranks") != summary["mpi_size"] or overrides.get(
        "output_dir"
    ) != effective["output_dir"]:
        raise WorkloadError("AuNP execution-policy evidence is inconsistent")
    artifact = artifacts.get("simulation_config.json")
    if artifact is None:
        raise WorkloadError("AuNP output omits simulation_config.json")
    summary_record = summary.get("simulation_config", {})
    summary_path = _record_path(lane["root"], summary_record)
    if summary_path != artifact["path"]:
        raise WorkloadError("AuNP summary does not bind simulation_config.json")
    if any(
        summary_record.get(key) != artifact["record"].get(key)
        for key in ("sha256", "size_bytes")
    ):
        raise WorkloadError("AuNP simulation-config records are inconsistent")
    with _stable_artifact(artifact, "AuNP simulation configuration") as stable:
        with stable.file_object() as handle:
            document = json.load(handle)
    if set(document) != {
        "schema_version",
        "config",
        "mpi_processes",
        "meep_version",
        "script_sha256",
        "material_fingerprint",
        "estimated_raw_dft_storage",
    }:
        raise WorkloadError("AuNP simulation-config schema is not exact")
    if document["config"] != effective or document["mpi_processes"] != summary["mpi_size"]:
        raise WorkloadError("AuNP engine manifest differs from its effective configuration")
    if document["script_sha256"] != summary.get("engine", {}).get("sha256"):
        raise WorkloadError("AuNP engine manifest has the wrong script identity")
    normalized_config = dict(effective)
    normalized_config.pop("output_dir")
    normalized_config.pop("required_mpi_ranks")
    return {
        "schema_version": document["schema_version"],
        "config": normalized_config,
        "meep_version": document["meep_version"],
        "script_sha256": document["script_sha256"],
        "material_fingerprint": document["material_fingerprint"],
        "estimated_raw_dft_storage": document["estimated_raw_dft_storage"],
    }


def validate_aunp_exact_stage_resume(lane: dict[str, Any]) -> dict[str, Any]:
    """Replay the exact AuNP stage journal and its terminal publication."""

    root = lane["root"]
    summary = lane["summary"]
    retained = summary.get("exact_stage_resume")
    if not isinstance(retained, dict) or set(retained) != {
        "schema",
        "contract",
        "attempts",
        "completed_stages",
        "quarantines",
        "journal_event_count_before_publication",
        "journal_terminal_sha256_before_publication",
        "cumulative_sealed_stage_wall_seconds",
        "current_attempt_index",
    }:
        raise WorkloadError("AuNP exact-stage summary schema differs")
    state_root = root / "exact-stage-state"
    if retained["contract"].get("qualification_profile") != aunp_qualification_profile():
        raise WorkloadError("AuNP exact-stage qualification profile differs")
    if (
        retained["contract"].get("default_config") != summary.get("default_config")
        or retained["contract"].get("effective_config")
        != summary.get("effective_config")
    ):
        raise WorkloadError("AuNP exact-stage physical configuration differs")
    events_directory = state_root / "events"
    checkpoint = exact_resume.replay_checkpoint(
        state_root / "CHECKPOINT.json",
        retained["contract"],
        events_directory,
        permit_trailing_durable_events=False,
    )
    base_paths = exact_resume.manifest_paths(
        checkpoint.get("base_manifest"), "AuNP TM-only base"
    )
    if base_paths != set(exact_resume.BASE_REQUIRED_PATHS):
        raise WorkloadError("AuNP TM-only base manifest is not exact")
    if checkpoint["schema"] != retained["schema"]:
        raise WorkloadError("AuNP exact-stage checkpoint schema differs")
    publication = checkpoint.get("publication")
    if not isinstance(publication, dict):
        raise WorkloadError("AuNP exact-stage journal has no terminal publication")
    current_attempt = retained["current_attempt_index"]
    if (
        type(current_attempt) is not int
        or current_attempt < 0
        or current_attempt >= len(checkpoint["attempts"])
    ):
        raise WorkloadError("AuNP current exact-stage attempt is invalid")
    actual_publication = {
        "attempt_index": current_attempt,
        "summary": file_record(root / "summary.json", root),
        "complete": file_record(root / "COMPLETE", root),
    }
    if publication != actual_publication:
        raise WorkloadError("AuNP exact-stage terminal publication differs")
    events = exact_resume.load_events(events_directory)
    if (
        len(events) < 2
        or retained["journal_event_count_before_publication"] != len(events) - 1
        or retained["journal_terminal_sha256_before_publication"]
        != events[-2]["file_sha256"]
    ):
        raise WorkloadError("AuNP pre-publication journal boundary differs")
    if (
        retained["attempts"] != checkpoint["attempts"]
        or retained["quarantines"] != checkpoint["quarantines"]
    ):
        raise WorkloadError("AuNP exact-stage attempt or quarantine history differs")
    expected_stages = [
        {
            "stage": item["stage"],
            "attempt_index": item["attempt_index"],
            "wall_seconds": item["wall_seconds"],
            "manifest": item["manifest"],
        }
        for item in checkpoint["completed_stages"]
    ]
    if (
        retained["completed_stages"] != expected_stages
        or [item["stage"] for item in expected_stages]
        != list(exact_resume.STAGE_ORDER)
    ):
        raise WorkloadError("AuNP exact-stage inventory differs")
    cumulative = sum(float(item["wall_seconds"]) for item in expected_stages)
    published_cumulative = retained["cumulative_sealed_stage_wall_seconds"]
    if (
        isinstance(published_cumulative, bool)
        or not isinstance(published_cumulative, (int, float))
        or not math.isfinite(float(published_cumulative))
    ):
        raise WorkloadError("AuNP cumulative exact-stage timing is invalid")
    if not math.isclose(
        float(published_cumulative),
        cumulative,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise WorkloadError("AuNP cumulative exact-stage timing differs")
    ranks = summary["rank_records"]
    for stage_index, stage in enumerate(checkpoint["completed_stages"][:-1]):
        expected_rank_records = [
            rank_record["records"][stage_index] for rank_record in ranks
        ]
        if stage["rank_records"] != expected_rank_records:
            raise WorkloadError(
                f"AuNP {stage['stage']} rank-record checkpoint differs"
            )
    if checkpoint["completed_stages"][-1]["rank_records"]:
        raise WorkloadError("AuNP postprocessing carries unexpected rank records")
    simulation_relative = _safe_relative_path(
        summary.get("simulation_root"), "simulation root"
    )
    simulation_root = root.joinpath(*simulation_relative.parts)
    exact_resume.replay_completed_output(simulation_root, checkpoint)
    if "output_manifest.json" not in exact_resume.manifest_paths(
        checkpoint["completed_stages"][-1]["manifest"],
        "AuNP final stage",
    ):
        raise WorkloadError("AuNP final stage does not bind output_manifest.json")
    return {
        "attempt_count": len(checkpoint["attempts"]),
        "quarantine_count": len(checkpoint["quarantines"]),
        "journal_event_count": len(events),
        "journal_terminal_sha256": events[-1]["file_sha256"],
    }


def _stream_metrics(
    reference_chunks: Iterator[np.ndarray],
    candidate_chunks: Iterator[np.ndarray],
) -> dict[str, float]:
    reference_square = 0.0
    candidate_square = 0.0
    difference_square = 0.0
    reference_maximum = 0.0
    candidate_maximum = 0.0
    difference_maximum = 0.0
    count = 0
    for reference, candidate in zip(reference_chunks, candidate_chunks):
        if reference.shape != candidate.shape:
            raise WorkloadError("scientific output chunk shape differs")
        ref_magnitude = np.abs(reference).astype(np.float64, copy=False)
        cand_magnitude = np.abs(candidate).astype(np.float64, copy=False)
        difference = np.abs(candidate - reference).astype(np.float64, copy=False)
        if not (
            np.all(np.isfinite(ref_magnitude))
            and np.all(np.isfinite(cand_magnitude))
            and np.all(np.isfinite(difference))
        ):
            raise WorkloadError("scientific output contains a non-finite value")
        reference_square += float(np.dot(ref_magnitude.reshape(-1), ref_magnitude.reshape(-1)))
        candidate_square += float(np.dot(cand_magnitude.reshape(-1), cand_magnitude.reshape(-1)))
        difference_square += float(np.dot(difference.reshape(-1), difference.reshape(-1)))
        reference_maximum = max(
            reference_maximum, float(np.max(ref_magnitude, initial=0.0))
        )
        candidate_maximum = max(
            candidate_maximum, float(np.max(cand_magnitude, initial=0.0))
        )
        difference_maximum = max(
            difference_maximum, float(np.max(difference, initial=0.0))
        )
        count += reference.size
    if count == 0:
        raise WorkloadError("scientific output array is empty")
    reference_l2 = math.sqrt(reference_square)
    return {
        "count": count,
        "reference_l2": reference_l2,
        "candidate_l2": math.sqrt(candidate_square),
        "reference_max_abs": reference_maximum,
        "candidate_max_abs": candidate_maximum,
        "nrmse": math.sqrt(difference_square) / reference_l2
        if reference_l2
        else (0.0 if difference_square == 0 else math.inf),
        "peak_normalized_linf": difference_maximum / reference_maximum
        if reference_maximum
        else (0.0 if difference_maximum == 0 else math.inf),
    }


def _numeric_outcome(metrics: dict[str, float], tolerance: dict[str, float]) -> str:
    return (
        "PASS"
        if metrics["nrmse"] <= tolerance["field_nrmse"]
        and metrics["peak_normalized_linf"] <= tolerance["field_peak_linf"]
        else "FAIL"
    )


def _hdf5_selections(shape: tuple[int, ...]) -> Iterator[Any]:
    if not shape:
        yield ()
        return
    block = [1] * len(shape)
    budget = CHUNK_VALUES
    for axis in range(len(shape) - 1, -1, -1):
        block[axis] = min(shape[axis], max(1, budget))
        budget = max(1, budget // max(1, block[axis]))
    starts = [range(0, extent, width) for extent, width in zip(shape, block)]
    for indices in product(*starts):
        yield tuple(
            slice(start, min(start + width, extent))
            for start, width, extent in zip(indices, block, shape)
        )


def _hdf5_chunks(dataset: Any) -> Iterator[np.ndarray]:
    if dataset.shape == ():
        yield np.asarray(dataset[()]).reshape(1)
        return
    # Use a layout-independent traversal so two semantically identical files
    # still align if HDF5 selected different physical chunking/compression.
    for selection in _hdf5_selections(dataset.shape):
        yield np.asarray(dataset[selection]).reshape(-1)


def _complex_hdf5_chunks(handle: Any, base: str) -> Iterator[np.ndarray]:
    real_dataset = handle[f"{base}.r"]
    imaginary_dataset = handle.get(f"{base}.i")
    for selection in _hdf5_selections(real_dataset.shape):
        real_part = np.asarray(real_dataset[selection])
        if imaginary_dataset is None:
            yield real_part.astype(np.complex128).reshape(-1)
        else:
            yield (real_part + 1j * np.asarray(imaginary_dataset[selection])).reshape(-1)


_AUNP_NPZ_FLOAT64_FILES = {
    "dense_line_heatmaps/center_vertical_derived.npz",
    "dense_line_heatmaps/gap_midline_derived.npz",
    "dense_line_heatmaps/particle_top_air_derived.npz",
    "field_metrics.npz",
    "reference_dense_line_metadata.npz",
    "structure_dense_line_metadata.npz",
}
_AUNP_NPZ_COMPLEX128_KEYS = {
    "incident_physical_Dx",
    "incident_physical_Dy",
    "incident_physical_Dz",
    "incident_physical_Ex",
    "incident_physical_Ey",
    "incident_physical_Ez",
    "incident_physical_Hx",
    "incident_physical_Hy",
    "incident_physical_Hz",
    "mode_coefficients_monitor1",
    "mode_coefficients_monitor2",
    "physical_Dx",
    "physical_Dy",
    "physical_Dz",
    "physical_Ex",
    "physical_Ey",
    "physical_Ez",
    "physical_Hx",
    "physical_Hy",
    "physical_Hz",
    "reflection_coefficient_monitor1",
    "reflection_coefficient_monitor2",
}
_AUNP_NPZ_MIXED_FILES = {
    "dense_probe_derived.npz",
    "field_metadata.npz",
    "material_masks.npz",
    "reference_data.npz",
    "reference_dense_probes.npz",
    "spectra.npz",
    "structure_dense_probes.npz",
}
_AUNP_HDF5_ENGINE_PRECISION_FILES = {
    "raw_complex_fields.h5",
    "reference_bottom_flux.h5",
    "reference_monitor1.h5",
    "reference_monitor2.h5",
    "str_bottom_flux.h5",
    "str_scat_m1.h5",
    "str_scat_m2.h5",
}
_AUNP_HDF5_FLOAT64_FILES = {
    "ref_line_gap.h5",
    "ref_line_top.h5",
    "ref_line_vertical.h5",
    "str_line_gap.h5",
    "str_line_top.h5",
    "str_line_vertical.h5",
}


def _artifact_relative_name(artifact: Any) -> str | None:
    if not isinstance(artifact, dict):
        return None
    path = artifact.get("path")
    root = artifact.get("root")
    if not isinstance(path, pathlib.Path) or not isinstance(root, pathlib.Path):
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise WorkloadError("scientific artifact escapes its manifest root") from error


def _comparison_precision_pair(comparison_class: str | None) -> tuple[bool, bool] | None:
    if comparison_class is None:
        return None
    contract = COMPARISON_CLASSES.get(comparison_class)
    if contract is None:
        raise WorkloadError(f"unknown comparison class: {comparison_class}")
    precision = contract.get("precision")
    if (
        not isinstance(precision, tuple)
        or len(precision) != 2
        or any(not isinstance(value, bool) for value in precision)
    ):
        raise WorkloadError(f"comparison class has invalid precision policy: {comparison_class}")
    return precision


def _aunp_npz_expected_dtype(relative: str, key: str) -> np.dtype[Any]:
    parts = pathlib.PurePosixPath(relative).parts
    if len(parts) < 2 or parts[0] not in set(AUNP_OUTPUT_POLARIZATIONS):
        raise WorkloadError(f"unknown AuNP NPZ artifact path: {relative}")
    name = "/".join(parts[1:])
    if name in _AUNP_NPZ_FLOAT64_FILES:
        return np.dtype(np.float64)
    if name not in _AUNP_NPZ_MIXED_FILES:
        raise WorkloadError(f"unknown AuNP NPZ dtype policy: {relative}:{key}")
    if name == "material_masks.npz":
        expected = {
            "gap_mask": np.dtype(np.bool_),
            "material_label": np.dtype(np.uint8),
            "particle_mask": np.dtype(np.bool_),
            "substrate_mask": np.dtype(np.bool_),
        }.get(key)
        if expected is None:
            raise WorkloadError(f"unknown AuNP mask dtype policy: {relative}:{key}")
        return expected
    if key == "probe_name":
        return np.dtype("<U22")
    if key.endswith("_dataset_prefix"):
        return np.dtype("<U2")
    if key in _AUNP_NPZ_COMPLEX128_KEYS:
        return np.dtype(np.complex128)
    return np.dtype(np.float64)


def _aunp_hdf5_expected_dtype(relative: str, single_precision: bool) -> np.dtype[Any]:
    parts = pathlib.PurePosixPath(relative).parts
    if len(parts) != 2 or parts[0] not in set(AUNP_OUTPUT_POLARIZATIONS):
        raise WorkloadError(f"unknown AuNP HDF5 artifact path: {relative}")
    name = parts[1]
    if name in _AUNP_HDF5_ENGINE_PRECISION_FILES:
        return np.dtype(np.float32 if single_precision else np.float64)
    if name in _AUNP_HDF5_FLOAT64_FILES:
        return np.dtype(np.float64)
    raise WorkloadError(f"unknown AuNP HDF5 dtype policy: {relative}")


def _require_dtype_pair(
    reference_dtype: np.dtype[Any],
    candidate_dtype: np.dtype[Any],
    comparison_class: str | None,
    label: str,
) -> tuple[str, str]:
    reference_text = reference_dtype.str
    candidate_text = candidate_dtype.str
    precision = _comparison_precision_pair(comparison_class)
    if precision is None or precision[0] == precision[1]:
        if reference_text != candidate_text:
            raise WorkloadError(f"scientific dtype differs: {label}")
    else:
        allowed = {
            (np.dtype(np.float64).str, np.dtype(np.float64).str),
            (np.dtype(np.float64).str, np.dtype(np.float32).str),
            (np.dtype(np.complex128).str, np.dtype(np.complex128).str),
            (np.dtype(np.complex128).str, np.dtype(np.complex64).str),
        }
        if reference_dtype.kind in "iubSU":
            accepted = reference_text == candidate_text
        else:
            accepted = (reference_text, candidate_text) in allowed
        if not accepted:
            raise WorkloadError(f"scientific cross-precision dtype is not allowed: {label}")
    return reference_text, candidate_text


def _require_aunp_expected_dtype(
    artifact: Any,
    key: str,
    observed: np.dtype[Any],
    single_precision: bool,
    kind: str,
) -> str:
    relative = _artifact_relative_name(artifact)
    if relative is None:
        return observed.str
    expected = (
        _aunp_npz_expected_dtype(relative, key)
        if kind == "npz"
        else _aunp_hdf5_expected_dtype(relative, single_precision)
    )
    if observed.str != expected.str:
        raise WorkloadError(
            f"AuNP scientific dtype violates the fixed policy: "
            f"{relative}:{key}: expected {expected.str}, observed {observed.str}"
        )
    return expected.str


def compare_hdf5_file(
    reference: Any,
    candidate: Any,
    tolerance: dict[str, float],
    comparison_class: str | None = None,
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise WorkloadError("h5py is required for AuNP field comparison") from exc

    results: dict[str, Any] = {}
    failed = False
    precision = _comparison_precision_pair(comparison_class)
    reference_single = precision[0] if precision is not None else False
    candidate_single = precision[1] if precision is not None else False
    reference_path = reference if isinstance(reference, pathlib.Path) else reference["path"]
    candidate_path = candidate if isinstance(candidate, pathlib.Path) else candidate["path"]
    with _stable_artifact(reference, "reference HDF5") as ref_stable, _stable_artifact(
        candidate, "candidate HDF5"
    ) as cand_stable, ref_stable.file_object() as ref_stream, cand_stable.file_object() as cand_stream, h5py.File(
        ref_stream, "r"
    ) as ref_handle, h5py.File(cand_stream, "r") as cand_handle:
        if set(ref_handle) != set(cand_handle):
            raise WorkloadError(f"HDF5 dataset inventory differs: {reference_path.name}")
        names = set(ref_handle)
        dangling_imaginary = sorted(
            name for name in names if name.endswith(".i") and f"{name[:-2]}.r" not in names
        )
        if dangling_imaginary:
            raise WorkloadError(
                f"HDF5 imaginary datasets have no real partner: {dangling_imaginary}"
            )
        complex_bases = sorted(name[:-2] for name in names if name.endswith(".r"))
        ordinary_names = sorted(
            name for name in names if not name.endswith(".r") and not name.endswith(".i")
        )
        for base in complex_bases:
            real_name = f"{base}.r"
            ref_real = ref_handle[real_name]
            cand_real = cand_handle[real_name]
            ref_imaginary = ref_handle.get(f"{base}.i")
            cand_imaginary = cand_handle.get(f"{base}.i")
            if (ref_imaginary is None) != (cand_imaginary is None):
                raise WorkloadError(f"HDF5 complex pairing differs: {reference_path.name}:{base}")
            datasets = [ref_real, cand_real]
            if ref_imaginary is not None:
                datasets.extend((ref_imaginary, cand_imaginary))
            if any(
                not isinstance(dataset, h5py.Dataset)
                or dataset.shape != ref_real.shape
                or dataset.dtype.kind not in "fc"
                for dataset in datasets
            ):
                raise WorkloadError(f"invalid HDF5 complex dataset: {reference_path.name}:{base}")
            dtype_evidence: dict[str, dict[str, str]] = {}
            dataset_pairs = [("real", ref_real, cand_real)]
            if ref_imaginary is not None:
                dataset_pairs.append(("imaginary", ref_imaginary, cand_imaginary))
            for suffix, ref_dataset, cand_dataset in dataset_pairs:
                assert ref_dataset is not None and cand_dataset is not None
                _require_aunp_expected_dtype(
                    reference, f"{base}.{suffix[0]}", ref_dataset.dtype,
                    reference_single, "hdf5"
                )
                _require_aunp_expected_dtype(
                    candidate, f"{base}.{suffix[0]}", cand_dataset.dtype,
                    candidate_single, "hdf5"
                )
                ref_dtype, cand_dtype = _require_dtype_pair(
                    ref_dataset.dtype,
                    cand_dataset.dtype,
                    comparison_class,
                    f"{reference_path.name}:{base}.{suffix[0]}",
                )
                dtype_evidence[suffix] = {
                    "reference": ref_dtype,
                    "candidate": cand_dtype,
                }
            if ref_imaginary is not None and (
                ref_real.dtype.str != ref_imaginary.dtype.str
                or cand_real.dtype.str != cand_imaginary.dtype.str
            ):
                raise WorkloadError(
                    f"HDF5 complex real/imag dtype differs: {reference_path.name}:{base}"
                )
            metrics = _stream_metrics(
                _complex_hdf5_chunks(ref_handle, base),
                _complex_hdf5_chunks(cand_handle, base),
            )
            result = {
                **metrics,
                "dtypes": dtype_evidence,
                "outcome": _numeric_outcome(metrics, tolerance),
            }
            results[base] = result
            failed |= result["outcome"] != "PASS"

        for name in ordinary_names:
            ref_dataset = ref_handle[name]
            cand_dataset = cand_handle[name]
            if not isinstance(ref_dataset, h5py.Dataset) or not isinstance(cand_dataset, h5py.Dataset):
                raise WorkloadError(f"nested HDF5 objects are unsupported: {reference_path}:{name}")
            if ref_dataset.shape != cand_dataset.shape or ref_dataset.dtype.kind != cand_dataset.dtype.kind:
                raise WorkloadError(f"HDF5 dataset shape/type differs: {reference_path.name}:{name}")
            _require_aunp_expected_dtype(
                reference, name, ref_dataset.dtype, reference_single, "hdf5"
            )
            _require_aunp_expected_dtype(
                candidate, name, cand_dataset.dtype, candidate_single, "hdf5"
            )
            ref_dtype, cand_dtype = _require_dtype_pair(
                ref_dataset.dtype,
                cand_dataset.dtype,
                comparison_class,
                f"{reference_path.name}:{name}",
            )
            if ref_dataset.dtype.kind not in "fciub":
                exact = all(
                    np.array_equal(left, right)
                    for left, right in zip(_hdf5_chunks(ref_dataset), _hdf5_chunks(cand_dataset))
                )
                result = {"exact": exact, "outcome": "PASS" if exact else "FAIL"}
            elif ref_dataset.dtype.kind in "iub":
                exact = all(
                    np.array_equal(left, right)
                    for left, right in zip(_hdf5_chunks(ref_dataset), _hdf5_chunks(cand_dataset))
                )
                result = {"exact": exact, "outcome": "PASS" if exact else "FAIL"}
            else:
                metrics = _stream_metrics(
                    _hdf5_chunks(ref_dataset), _hdf5_chunks(cand_dataset)
                )
                result = {**metrics, "outcome": _numeric_outcome(metrics, tolerance)}
            result["reference_dtype"] = ref_dtype
            result["candidate_dtype"] = cand_dtype
            results[name] = result
            failed |= result["outcome"] != "PASS"
    return {"datasets": results, "outcome": "FAIL" if failed else "PASS"}


def _npz_exact_key(path: pathlib.Path, key: str, value: np.ndarray) -> bool:
    exact_tokens = (
        "wavelength",
        "frequency",
        "coordinate",
        "integration_weight",
        "probe_name",
        "probe_x",
        "probe_meep_y",
        "dataset_prefix",
        "_sign",
    )
    return (
        value.dtype.kind not in "fc"
        or path.name in {"field_metadata.npz", "material_masks.npz"}
        or any(token in key for token in exact_tokens)
    )


def compare_npz_file(
    reference: Any,
    candidate: Any,
    tolerance: dict[str, float],
    comparison_class: str | None = None,
) -> dict[str, Any]:
    reference_path = reference if isinstance(reference, pathlib.Path) else reference["path"]
    results: dict[str, Any] = {}
    failed = False
    precision = _comparison_precision_pair(comparison_class)
    reference_single = precision[0] if precision is not None else False
    candidate_single = precision[1] if precision is not None else False
    with _stable_artifact(reference, "reference NPZ") as ref_stable, _stable_artifact(
        candidate, "candidate NPZ"
    ) as cand_stable, ref_stable.file_object() as ref_stream, cand_stable.file_object() as cand_stream, np.load(
        ref_stream, allow_pickle=False
    ) as ref_archive, np.load(cand_stream, allow_pickle=False) as cand_archive:
        if set(ref_archive.files) != set(cand_archive.files):
            raise WorkloadError(f"NPZ array inventory differs: {reference_path.name}")
        for name in sorted(ref_archive.files):
            ref = np.asarray(ref_archive[name])
            cand = np.asarray(cand_archive[name])
            if ref.shape != cand.shape or ref.dtype.kind != cand.dtype.kind:
                raise WorkloadError(f"NPZ array shape/type differs: {reference_path.name}:{name}")
            _require_aunp_expected_dtype(
                reference, name, ref.dtype, reference_single, "npz"
            )
            _require_aunp_expected_dtype(
                candidate, name, cand.dtype, candidate_single, "npz"
            )
            ref_dtype, cand_dtype = _require_dtype_pair(
                ref.dtype,
                cand.dtype,
                comparison_class,
                f"{reference_path.name}:{name}",
            )
            if _npz_exact_key(reference_path, name, ref):
                exact = bool(np.array_equal(ref, cand))
                result = {"exact": exact, "outcome": "PASS" if exact else "FAIL"}
            else:
                metrics = _stream_metrics(_chunks(ref), _chunks(cand))
                result = {**metrics, "outcome": _numeric_outcome(metrics, tolerance)}
            result["reference_dtype"] = ref_dtype
            result["candidate_dtype"] = cand_dtype
            results[name] = result
            failed |= result["outcome"] != "PASS"
    return {"arrays": results, "outcome": "FAIL" if failed else "PASS"}


def compare_aunp_scientific_outputs(
    reference_manifest: dict[str, dict[str, Any]],
    candidate_manifest: dict[str, dict[str, Any]],
    tolerance: dict[str, float],
    comparison_class: str,
) -> dict[str, Any]:
    reference_paths = {
        name for name in reference_manifest if pathlib.PurePosixPath(name).suffix in {".h5", ".npz"}
    }
    candidate_paths = {
        name for name in candidate_manifest if pathlib.PurePosixPath(name).suffix in {".h5", ".npz"}
    }
    if reference_paths != candidate_paths:
        raise WorkloadError(
            "AuNP scientific file inventory differs: missing="
            f"{sorted(reference_paths - candidate_paths)}, extra="
            f"{sorted(candidate_paths - reference_paths)}"
        )
    files: dict[str, Any] = {}
    failed = False
    for name in sorted(reference_paths):
        reference = reference_manifest[name]
        candidate = candidate_manifest[name]
        if reference["path"].suffix == ".h5":
            result = compare_hdf5_file(
                reference, candidate, tolerance, comparison_class
            )
        else:
            result = compare_npz_file(
                reference, candidate, tolerance, comparison_class
            )
        files[name] = result
        failed |= result["outcome"] != "PASS"
    return {"files": files, "outcome": "FAIL" if failed else "PASS"}


def _array_stats(array: np.ndarray) -> dict[str, float]:
    square_sum = 0.0
    maximum = 0.0
    count = 0
    for chunk in _chunks(array):
        magnitude = np.abs(chunk).astype(np.float64, copy=False)
        if not np.all(np.isfinite(magnitude)):
            raise WorkloadError("field array contains a non-finite value")
        square_sum += float(np.dot(magnitude, magnitude))
        maximum = max(maximum, float(np.max(magnitude, initial=0.0)))
        count += magnitude.size
    if count == 0 or square_sum <= 0 or maximum <= 0:
        raise WorkloadError("field array has no nonzero finite signal")
    return {"l2": math.sqrt(square_sum), "max_abs": maximum, "energy": square_sum}


def _relative(actual: float, reference: float) -> float:
    if reference == 0:
        if actual == 0:
            return 0.0
        raise WorkloadError("relative comparison has a zero reference")
    return abs(actual - reference) / abs(reference)


def _orient_ters(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if array.shape == (y.size, x.size):
        return array
    if array.shape == (x.size, y.size):
        return array.T
    raise WorkloadError(
        f"TERS field shape {array.shape} is inconsistent with coordinates "
        f"({x.size}, {y.size})"
    )


def compare_ters_field(
    reference: np.ndarray,
    candidate: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.dtype.kind != "c" or candidate.dtype.kind != "c":
        raise WorkloadError("TERS complex field shape/type mismatch")
    ref_stats = _array_stats(reference)
    cand_stats = _array_stats(candidate)
    diff_square = 0.0
    diff_maximum = 0.0
    phase_square = 0.0
    phase_count = 0
    phase_floor = ref_stats["max_abs"] * 1e-3
    for ref_chunk, cand_chunk in zip(_chunks(reference), _chunks(candidate)):
        difference = cand_chunk - ref_chunk
        magnitude = np.abs(difference).astype(np.float64, copy=False)
        if not np.all(np.isfinite(magnitude)):
            raise WorkloadError("TERS comparison produced a non-finite difference")
        diff_square += float(np.dot(magnitude, magnitude))
        diff_maximum = max(diff_maximum, float(np.max(magnitude, initial=0.0)))
        mask = np.abs(ref_chunk) >= phase_floor
        if np.any(mask):
            phase = np.angle(cand_chunk[mask] / ref_chunk[mask]).astype(
                np.float64, copy=False
            )
            phase_square += float(np.dot(phase, phase))
            phase_count += phase.size
    oriented_reference = _orient_ters(reference, x, y)
    oriented_candidate = _orient_ters(candidate, x, y)
    x_mask = np.abs(x) <= 0.0100000001
    y_mask = (y >= 0.0384999999) & (y <= 0.0415000001)
    if not np.any(x_mask) or not np.any(y_mask):
        raise WorkloadError("TERS coordinates do not contain the physical 3 nm gap ROI")
    ref_roi = np.abs(oriented_reference[np.ix_(y_mask, x_mask)]) ** 2
    cand_roi = np.abs(oriented_candidate[np.ix_(y_mask, x_mask)]) ** 2
    roi = {}
    roi_failures = []
    for name, percentile in (("mean", None), ("p95", 95), ("p99", 99)):
        ref_value = float(np.mean(ref_roi)) if percentile is None else float(np.percentile(ref_roi, percentile))
        cand_value = float(np.mean(cand_roi)) if percentile is None else float(np.percentile(cand_roi, percentile))
        relative = _relative(cand_value, ref_value)
        roi[name] = {"reference": ref_value, "candidate": cand_value, "relative_error": relative}
        if relative > tolerance["field_roi_percentile_relative"]:
            roi_failures.append(name)
    metrics = {
        "nrmse": math.sqrt(diff_square) / ref_stats["l2"],
        "peak_normalized_linf": diff_maximum / ref_stats["max_abs"],
        "energy_relative_error": _relative(cand_stats["energy"], ref_stats["energy"]),
        "peak_relative_error": _relative(cand_stats["max_abs"] ** 2, ref_stats["max_abs"] ** 2),
        "masked_phase_rms_rad": math.sqrt(phase_square / phase_count) if phase_count else math.inf,
        "phase_sample_count": phase_count,
        "roi": roi,
        "reference": ref_stats,
        "candidate": cand_stats,
    }
    failures = [
        name
        for name, limit_key in (
            ("nrmse", "field_nrmse"),
            ("peak_normalized_linf", "field_peak_linf"),
            ("energy_relative_error", "field_energy_relative"),
            ("peak_relative_error", "field_peak_relative"),
            ("masked_phase_rms_rad", "phase_rms"),
        )
        if metrics[name] > tolerance[limit_key]
    ] + [f"roi.{name}" for name in roi_failures]
    metrics["failures"] = failures
    metrics["outcome"] = "PASS" if not failures else "FAIL"
    return metrics


def _coordinate_evidence(values: dict[str, np.ndarray]) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for name, value in values.items():
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
        axes[name] = {
            "count": int(array.size),
            "dtype": str(array.dtype),
            "sha256": digest.hexdigest(),
        }
    return axes


def _array_bit_exact(reference: np.ndarray, candidate: np.ndarray) -> bool:
    if reference.shape != candidate.shape or reference.dtype.str != candidate.dtype.str:
        return False
    reference_bytes = np.ascontiguousarray(reference).view(np.uint8).reshape(-1)
    candidate_bytes = np.ascontiguousarray(candidate).view(np.uint8).reshape(-1)
    return all(
        np.array_equal(
            reference_bytes[start : start + CHUNK_VALUES],
            candidate_bytes[start : start + CHUNK_VALUES],
        )
        for start in range(0, reference_bytes.size, CHUNK_VALUES)
    )


def _compare_ters_coordinates(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    comparison_class: str,
) -> dict[str, Any]:
    """Compare spatial axes exactly and cubature weights by an explicit policy.

    Meep exposes the spatial grid in double precision for both builds, so the
    three axes are execution invariants and must remain byte-identical.  The
    cubature weights are also returned as doubles, but their construction can
    include FP32 arithmetic and MPI reductions.  Non-repeat comparison classes
    therefore use a small symmetric relative bound, while repeat classes keep
    the stronger byte-exact reproducibility gate.  No absolute tolerance is
    allowed, so a zero weight cannot silently become nonzero.
    """

    contract = COMPARISON_CLASSES.get(comparison_class)
    if contract is None:
        raise WorkloadError(f"unknown comparison class: {comparison_class}")
    for label, values in (("reference", reference), ("candidate", candidate)):
        try:
            validate_dft_coordinate_metadata(values, np.isfinite)
        except WorkloadError as error:
            raise WorkloadError(f"TERS {label} coordinates are invalid: {error}") from error

    spatial_axes: dict[str, Any] = {}
    for name in ("x", "y", "z"):
        reference_axis = np.asarray(reference[name])
        candidate_axis = np.asarray(candidate[name])
        if (
            reference_axis.dtype.str != np.dtype(np.float64).str
            or candidate_axis.dtype.str != np.dtype(np.float64).str
        ):
            raise WorkloadError(
                f"TERS spatial coordinate {name} is not native float64"
            )
        exact = _array_bit_exact(reference_axis, candidate_axis)
        reference_evidence = _coordinate_evidence({name: reference_axis})[name]
        candidate_evidence = _coordinate_evidence({name: candidate_axis})[name]
        spatial_axes[name] = {
            "reference": reference_evidence,
            "candidate": candidate_evidence,
            "byte_exact": exact,
            "outcome": "PASS" if exact else "FAIL",
        }
        if not exact:
            raise WorkloadError(
                f"TERS spatial coordinate {name} is not byte-exact for candidate"
            )

    reference_weights = np.asarray(reference["w"])
    candidate_weights = np.asarray(candidate["w"])
    if reference_weights.shape != candidate_weights.shape:
        raise WorkloadError("TERS cubature-weight shape differs for candidate")
    if (
        reference_weights.dtype.str != np.dtype(np.float64).str
        or candidate_weights.dtype.str != np.dtype(np.float64).str
    ):
        raise WorkloadError(
            "TERS cubature weights are not native float64"
        )

    policy = contract.get("coordinate_weight_policy")
    rtol = contract.get("coordinate_weight_rtol")
    if (
        policy not in {"byte-exact", "symmetric-relative"}
        or isinstance(rtol, bool)
        or not isinstance(rtol, (int, float))
        or not math.isfinite(float(rtol))
        or float(rtol) < 0.0
        or (policy == "byte-exact" and float(rtol) != 0.0)
        or (policy == "symmetric-relative" and float(rtol) <= 0.0)
    ):
        raise WorkloadError(
            f"comparison class {comparison_class} has an invalid coordinate-weight policy"
        )
    rtol = float(rtol)

    reference_flat = reference_weights.reshape(-1)
    candidate_flat = candidate_weights.reshape(-1)
    different_values = 0
    maximum_absolute_difference = 0.0
    maximum_symmetric_relative_difference = 0.0
    within_tolerance = True
    reference_has_positive = False
    candidate_has_positive = False
    for start in range(0, reference_flat.size, CHUNK_VALUES):
        reference_chunk = reference_flat[start : start + CHUNK_VALUES]
        candidate_chunk = candidate_flat[start : start + CHUNK_VALUES]
        if not np.all(np.isfinite(reference_chunk)) or not np.all(
            np.isfinite(candidate_chunk)
        ):
            raise WorkloadError("TERS cubature weights contain non-finite values")
        if np.any(np.signbit(reference_chunk)) or np.any(np.signbit(candidate_chunk)):
            raise WorkloadError("TERS cubature weights are not canonically nonnegative")
        reference_has_positive |= bool(np.any(reference_chunk > 0.0))
        candidate_has_positive |= bool(np.any(candidate_chunk > 0.0))

        difference = np.abs(candidate_chunk - reference_chunk)
        scale = np.maximum(np.abs(reference_chunk), np.abs(candidate_chunk))
        relative = np.divide(
            difference,
            scale,
            out=np.zeros_like(difference, dtype=np.float64),
            where=scale != 0.0,
        )
        different_values += int(np.count_nonzero(difference))
        maximum_absolute_difference = max(
            maximum_absolute_difference,
            float(np.max(difference, initial=0.0)),
        )
        maximum_symmetric_relative_difference = max(
            maximum_symmetric_relative_difference,
            float(np.max(relative, initial=0.0)),
        )
        if policy == "symmetric-relative" and np.any(difference > rtol * scale):
            within_tolerance = False

    if not reference_has_positive or not candidate_has_positive:
        raise WorkloadError("TERS cubature weights have no positive values")
    if policy == "byte-exact":
        within_tolerance = _array_bit_exact(reference_weights, candidate_weights)
        if not within_tolerance:
            raise WorkloadError("TERS cubature weights are not byte-exact for repeat")
    elif not within_tolerance:
        raise WorkloadError(
            "TERS cubature weights exceed the symmetric relative tolerance"
        )

    weights = {
        "shape": list(reference_weights.shape),
        "reference_dtype": reference_weights.dtype.str,
        "candidate_dtype": candidate_weights.dtype.str,
        "policy": policy,
        "different_values": different_values,
        "maximum_absolute_difference": maximum_absolute_difference,
        "maximum_symmetric_relative_difference": (
            maximum_symmetric_relative_difference
        ),
        "relative_tolerance": rtol,
        "absolute_tolerance": 0.0,
        "outcome": "PASS",
    }
    return {
        "spatial_axes": spatial_axes,
        "cubature_weights": weights,
        "outcome": "PASS",
    }


def _ters_files(
    lane: dict[str, Any], *, allow_performance_window: bool = False
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    summary = lane["summary"]
    root = lane["root"]
    paths = {}
    for record in summary.get("arrays", []):
        path = _record_path(root, record)
        paths[path.name] = {"path": path, "root": root, "record": record}
    if set(paths) != {"withtip_dft.npy", "wotip_dft.npy"}:
        raise WorkloadError("TERS array inventory is not exact")
    original_outputs = summary.get("original_outputs")
    if not isinstance(original_outputs, list):
        raise WorkloadError("TERS exact original output inventory is absent")
    if original_outputs != summary["rank_records"][0].get("original_outputs"):
        raise WorkloadError("TERS retained outputs are not rank-zero original outputs")
    output_paths: dict[str, pathlib.Path] = {}
    for record in original_outputs:
        with _stable_record(root, record, "TERS exact original output") as stable:
            if stable.path.name in output_paths:
                raise WorkloadError("TERS exact original output inventory has duplicates")
            output_paths[stable.path.name] = stable.path
    source_name = pathlib.PurePosixPath(summary.get("source", {}).get("path", "")).name
    if not source_name.endswith(".py"):
        raise WorkloadError("TERS source name is absent from summary")
    expected_output_names = {
        source_name.removesuffix(".py") + "_backup.py",
        "withtip_geo.png",
        "withtip_dft.csv",
        "withtip_dft.png",
        "withtip_dft.npy",
        "wotip_geo.png",
        "wotip_dft.csv",
        "wotip_dft.png",
        "wotip_dft.npy",
    }
    if set(output_paths) != expected_output_names or len(
        {path.parent for path in output_paths.values()}
    ) != 1:
        raise WorkloadError("TERS exact original output inventory is not canonical")
    actual_output_names = {
        path.name
        for path in next(iter(output_paths.values())).parent.iterdir()
        if path.is_file()
    }
    if actual_output_names != expected_output_names:
        raise WorkloadError("TERS exact original output directory changed after publication")
    if summary.get("presentation_suppressed") is not False:
        raise WorkloadError("TERS presentation output was suppressed")
    if allow_performance_window:
        window = summary.get("performance_window_meep_time")
        if (
            summary.get("measurement_mode") != TERS_PERFORMANCE_MEASUREMENT_MODE
            or not isinstance(window, (int, float))
            or isinstance(window, bool)
            or not math.isfinite(float(window))
            or not MINIMUM_TERS_PERFORMANCE_WINDOW_MEEP_TIME <= float(window) <= 1.0
            or summary.get("physics_overrides") != [TERS_PERFORMANCE_ADAPTATION]
        ):
            raise WorkloadError("TERS performance adaptation contract differs")
    elif (
        summary.get("measurement_mode") not in (None, "exact")
        or summary.get("performance_window_meep_time") is not None
        or summary.get("physics_overrides") != []
    ):
        raise WorkloadError("TERS exact execution was modified")
    coordinate_records = summary.get("coordinates")
    if not isinstance(coordinate_records, list) or len(coordinate_records) != 2:
        raise WorkloadError("TERS must retain exactly two coordinate sets")
    loaded_coordinates = []
    for run_index, coordinate_record in enumerate(coordinate_records):
        if coordinate_record.get("run_index") != run_index:
            raise WorkloadError("TERS coordinate run inventory is not canonical")
        with _stable_record(
            root, coordinate_record, f"TERS run {run_index} coordinates"
        ) as stable:
            with stable.file_object() as handle, np.load(
                handle, allow_pickle=False
            ) as coordinates:
                if set(coordinates.files) != {"x", "y", "z", "w"}:
                    raise WorkloadError("TERS coordinate inventory is not exact")
                values = {
                    name: np.asarray(coordinates[name]).copy()
                    for name in ("x", "y", "z", "w")
                }
        validate_dft_coordinate_metadata(values, np.isfinite)
        if any(
            np.asarray(values[name]).dtype.str != np.dtype(np.float64).str
            for name in ("x", "y", "z", "w")
        ):
            raise WorkloadError("TERS DFT coordinates are not native float64")
        evidence = _coordinate_evidence(values)
        if coordinate_record.get("axes") != evidence:
            raise WorkloadError("TERS coordinate evidence differs from retained arrays")
        for rank_record in summary["rank_records"]:
            if rank_record["records"][run_index].get("dft_coordinates") != evidence:
                raise WorkloadError("TERS rank coordinate evidence is inconsistent")
        loaded_coordinates.append(values)
    for name in ("x", "y", "z", "w"):
        if not _array_bit_exact(
            loaded_coordinates[0][name], loaded_coordinates[1][name]
        ):
            raise WorkloadError("the two TERS DFT monitors use different coordinates")
    return paths, loaded_coordinates[0]


def _ters_gap_enhancement(
    paths: dict[str, dict[str, Any]], x: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    with _stable_artifact(paths["withtip_dft.npy"], "TERS with-tip field") as with_stable, _stable_artifact(
        paths["wotip_dft.npy"], "TERS no-tip field"
    ) as without_stable:
        with_tip = _orient_ters(
            np.load(with_stable.proc_path, mmap_mode="r", allow_pickle=False), x, y
        )
        without_tip = _orient_ters(
            np.load(without_stable.proc_path, mmap_mode="r", allow_pickle=False), x, y
        )
        x_mask = np.abs(x) <= 0.0100000001
        y_mask = (y >= 0.0384999999) & (y <= 0.0415000001)
        with_intensity = np.abs(with_tip[np.ix_(y_mask, x_mask)]).astype(np.float64) ** 2
        without_intensity = np.abs(without_tip[np.ix_(y_mask, x_mask)]).astype(np.float64) ** 2
        if not np.all(np.isfinite(with_intensity)) or not np.all(np.isfinite(without_intensity)):
            raise WorkloadError("TERS gap enhancement contains non-finite source fields")
        floor = float(np.max(without_intensity, initial=0.0)) * 1e-12
        valid = without_intensity > floor
        if not np.any(valid):
            raise WorkloadError("TERS no-tip field has no usable signal in the gap ROI")
        enhancement = with_intensity[valid] / without_intensity[valid]
    return {
        "sample_count": int(enhancement.size),
        "mean": finite_float(np.mean(enhancement), "TERS enhancement mean"),
        "p95": finite_float(np.percentile(enhancement, 95), "TERS enhancement p95"),
        "p99": finite_float(np.percentile(enhancement, 99), "TERS enhancement p99"),
        "peak": finite_float(np.max(enhancement), "TERS enhancement peak"),
    }


def _compare_ters_enhancement(
    reference: dict[str, float],
    candidate: dict[str, float],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    failures = []
    for name in ("mean", "p95", "p99", "peak"):
        relative_error = _relative(candidate[name], reference[name])
        limit = (
            tolerance["field_peak_relative"]
            if name == "peak"
            else tolerance["field_roi_percentile_relative"]
        )
        metrics[name] = {
            "reference": reference[name],
            "candidate": candidate[name],
            "relative_error": relative_error,
            "limit": limit,
        }
        if relative_error > limit:
            failures.append(name)
    return {
        "metrics": metrics,
        "failures": failures,
        "outcome": "PASS" if not failures else "FAIL",
    }


def compare_ters(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_lane(args.reference, TERS_SCHEMA)
    reference_provenance = validate_lane_provenance(reference)
    reference_paths, reference_coordinates = _ters_files(reference)
    reference_x = reference_coordinates["x"]
    reference_y = reference_coordinates["y"]
    reference_enhancement = _ters_gap_enhancement(
        reference_paths, reference_x, reference_y
    )
    comparisons = []
    failed = False
    for label, comparison_class, path in args.candidate:
        candidate = load_lane(path, TERS_SCHEMA)
        _validate_backend_roles(reference, candidate, label, comparison_class)
        candidate_provenance = validate_lane_provenance(candidate)
        if candidate_provenance["source_start"] != reference_provenance["source_start"]:
            raise WorkloadError(f"TERS source/build snapshot differs for candidate {label}")
        _validate_precision_pair(reference, candidate, comparison_class)
        candidate_paths, candidate_coordinates = _ters_files(candidate)
        coordinate_comparison = _compare_ters_coordinates(
            reference_coordinates, candidate_coordinates, comparison_class
        )
        candidate_x = candidate_coordinates["x"]
        candidate_y = candidate_coordinates["y"]
        fields = {}
        for name in sorted(reference_paths):
            with _stable_artifact(reference_paths[name], f"reference TERS {name}") as ref_stable, _stable_artifact(
                candidate_paths[name], f"candidate TERS {name}"
            ) as cand_stable:
                ref_array = np.load(ref_stable.proc_path, mmap_mode="r", allow_pickle=False)
                cand_array = np.load(cand_stable.proc_path, mmap_mode="r", allow_pickle=False)
                fields[name] = compare_ters_field(
                    ref_array,
                    cand_array,
                    reference_x,
                    reference_y,
                    _comparison_tolerance(comparison_class),
                )
        enhancement = _compare_ters_enhancement(
            reference_enhancement,
            _ters_gap_enhancement(candidate_paths, candidate_x, candidate_y),
            _comparison_tolerance(comparison_class),
        )
        outcome = (
            "PASS"
            if all(item["outcome"] == "PASS" for item in fields.values())
            and enhancement["outcome"] == "PASS"
            else "FAIL"
        )
        failed |= outcome != "PASS"
        comparisons.append(
            {
                "label": label,
                "comparison_class": comparison_class,
                "candidate": str(path.resolve()),
                "provenance": candidate_provenance,
                "tolerance": _comparison_tolerance(comparison_class),
                "coordinates": coordinate_comparison,
                "fields": fields,
                "gap_enhancement": enhancement,
                "outcome": outcome,
            }
        )
    return {
        "schema": "gpmeep-ters-comparison-v1",
        "reference": str(args.reference.resolve()),
        "reference_provenance": reference_provenance,
        "comparisons": comparisons,
        "outcome": "FAIL" if failed else "PASS",
    }


def _load_spectra(artifact: Any) -> dict[str, np.ndarray]:
    path = artifact if isinstance(artifact, pathlib.Path) else artifact["path"]
    with _stable_artifact(artifact, "AuNP spectra") as stable, stable.file_object() as handle, np.load(
        handle, allow_pickle=False
    ) as archive:
        if set(archive.files) != AUNP_KEYS:
            raise WorkloadError(f"AuNP spectra inventory mismatch: {path}")
        result = {name: np.asarray(archive[name]) for name in archive.files}
    for name, value in result.items():
        if value.size == 0 or value.dtype.kind not in "fci" or not np.all(np.isfinite(value)):
            raise WorkloadError(f"invalid AuNP spectra array {name}: {path}")
    if result["wavelength_nm"].shape != (301,):
        raise WorkloadError(f"AuNP wavelength grid is not the fixed 301-vector: {path}")
    for name, value in result.items():
        expected_shape = (301, 2) if name.startswith("mode_coefficients") else (301,)
        if value.shape != expected_shape:
            raise WorkloadError(f"AuNP spectra shape mismatch for {name}: {value.shape}")
    residuals = {
        "absorptance_opaque": np.max(
            np.abs(result["absorptance_opaque"] - (1 - result["reflectance_monitor1"]))
        ),
        "absorptance_to_bottom": np.max(
            np.abs(
                result["absorptance_to_bottom"]
                - (1 - result["reflectance_monitor1"] - result["bottom_residual_fraction"])
            )
        ),
        "reflectance_monitor1": np.max(
            np.abs(
                result["reflectance_monitor1"]
                - result["reflected_flux_monitor1"] / result["incident_flux"]
            )
        ),
    }
    if any(float(value) > 5e-7 for value in residuals.values()):
        raise WorkloadError(f"AuNP spectra derived-value invariant failed: {path}")
    return result


def _aunp_spectra_files(lane: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for record in lane["summary"].get("spectra", []):
        path = _record_path(lane["root"], record)
        polarization = path.parent.name
        result[polarization] = {
            "path": path,
            "root": lane["root"],
            "record": record,
        }
    if set(result) != set(AUNP_OUTPUT_POLARIZATIONS):
        raise WorkloadError("AuNP retained spectra inventory is not exact")
    return result


def fit_tm_resonance(spectra: dict[str, np.ndarray]) -> dict[str, float]:
    try:
        from scipy.optimize import curve_fit
    except ImportError as exc:
        raise WorkloadError("SciPy is required for the fixed AuNP resonance oracle") from exc

    wavelength = spectra["wavelength_nm"].astype(np.float64)
    absorptance = spectra["absorptance_opaque"].astype(np.float64)
    mask = (wavelength >= 700) & (wavelength <= 950)
    x = wavelength[mask]
    y = absorptance[mask]

    def model(lam, lambda0, fwhm, peak, a0, slope):
        return a0 + slope * (lam - lambda0) + (peak - a0) / (
            1 + 4 * ((lam - lambda0) / fwhm) ** 2
        )

    peak_index = int(np.argmax(y))
    parameters, covariance = curve_fit(
        model,
        x,
        y,
        p0=(float(x[peak_index]), 60.0, float(y[peak_index]), float(np.min(y)), 0.0),
        bounds=((700.0, 1.0, -1.0, -1.0, -0.1), (950.0, 300.0, 2.0, 1.0, 0.1)),
        maxfev=100000,
    )
    residual = y - model(x, *parameters)
    phase = np.unwrap(np.angle(spectra["reflection_coefficient_monitor1"][mask]))
    return {
        "lambda0_nm": finite_float(parameters[0], "lambda0"),
        "lambda0_err_nm": finite_float(
            math.sqrt(covariance[0, 0]), "lambda0 uncertainty"
        ),
        "fwhm_nm": finite_float(parameters[1], "FWHM"),
        "fwhm_err_nm": finite_float(
            math.sqrt(covariance[1, 1]), "FWHM uncertainty"
        ),
        "q_factor": finite_float(parameters[0] / parameters[1], "Q factor"),
        "peak_absorptance": finite_float(parameters[2], "peak absorptance"),
        "background_a0": finite_float(parameters[3], "background a0"),
        "background_slope": finite_float(parameters[4], "background slope"),
        "residual_rms": finite_float(math.sqrt(float(np.mean(residual**2))), "fit RMS"),
        "phase_swing_rad": finite_float(float(np.ptp(phase)), "phase swing"),
        "fit_window_nm": [700.0, 950.0],
    }


def compare_resonance_fit(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    if reference.get("fit_window_nm") != [700.0, 950.0] or candidate.get(
        "fit_window_nm"
    ) != [700.0, 950.0]:
        raise WorkloadError("AuNP resonance fit window is not exact")
    metrics = {}
    failures = []
    for name, limit_kind in (
        ("lambda0_nm", "resonance_nm"),
        ("fwhm_nm", "resonance_fwhm_nm"),
        ("q_factor", "resonance_q"),
        ("phase_swing_rad", "resonance_phase_rad"),
    ):
        difference = abs(finite_float(candidate[name], name) - finite_float(reference[name], name))
        metrics[name] = {"absolute_difference": difference, "limit": tolerance[limit_kind]}
        if difference > tolerance[limit_kind]:
            failures.append(name)
    for name in (
        "lambda0_err_nm",
        "fwhm_err_nm",
        "peak_absorptance",
        "background_a0",
        "background_slope",
        "residual_rms",
    ):
        relative = _relative(
            finite_float(candidate[name], name), finite_float(reference[name], name)
        )
        metrics[name] = {
            "relative_error": relative,
            "limit": tolerance["resonance_scalar_relative"],
        }
        if relative > tolerance["resonance_scalar_relative"]:
            failures.append(name)
    return {
        "metrics": metrics,
        "failures": failures,
        "outcome": "PASS" if not failures else "FAIL",
    }


def compare_spectra_pair(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    tolerance: dict[str, float],
) -> dict[str, Any]:
    metrics = {}
    failures = []
    for name in sorted(AUNP_KEYS):
        ref = reference[name]
        cand = candidate[name]
        if ref.shape != cand.shape:
            raise WorkloadError(f"AuNP spectra shape differs for {name}")
        difference = np.abs(cand - ref).astype(np.float64)
        if name in {"wavelength_nm", "frequency"}:
            exact = bool(np.array_equal(ref, cand))
            metrics[name] = {"exact": exact}
            if not exact:
                failures.append(name)
            continue
        rmse = math.sqrt(float(np.mean(difference**2)))
        maximum = float(np.max(difference))
        ref_l2 = math.sqrt(float(np.sum(np.abs(ref).astype(np.float64) ** 2)))
        nrmse = math.sqrt(float(np.sum(difference**2))) / ref_l2 if ref_l2 else math.inf
        metrics[name] = {"max_abs": maximum, "rmse": rmse, "nrmse": nrmse}
        if name in SPECTRAL_POWER_KEYS:
            if maximum > tolerance["spectral_max_abs"] or rmse > tolerance["spectral_rmse"]:
                failures.append(name)
        elif name in SPECTRAL_COMPLEX_KEYS:
            if nrmse > tolerance["coefficient_nrmse"]:
                failures.append(name)
        elif nrmse > 5e-3:
            failures.append(name)
    ref_fit = fit_tm_resonance(reference)
    cand_fit = fit_tm_resonance(candidate)
    resonance = compare_resonance_fit(ref_fit, cand_fit, tolerance)
    failures.extend(f"resonance.{name}" for name in resonance["failures"])
    return {
        "arrays": metrics,
        "reference_resonance": ref_fit,
        "candidate_resonance": cand_fit,
        "resonance": resonance,
        "failures": failures,
        "outcome": "PASS" if not failures else "FAIL",
    }


def compare_aunp(args: argparse.Namespace) -> dict[str, Any]:
    reference_lane = load_lane(args.reference, AUNP_SCHEMA)
    reference_provenance = validate_lane_provenance(reference_lane)
    reference_stage_resume = validate_aunp_exact_stage_resume(reference_lane)
    reference_paths = _aunp_spectra_files(reference_lane)
    reference_spectra = {name: _load_spectra(path) for name, path in reference_paths.items()}
    reference_manifest = load_aunp_output_manifest(reference_lane)
    reference_config = validate_aunp_simulation_config(
        reference_lane, reference_manifest
    )
    comparisons = []
    failed = False
    expected_comparison_class = (
        "fp64-fp32"
        if reference_lane["summary"]["single_precision"]
        else "expected-fp64"
    )
    expected_tolerance = PAIR_TOLERANCES[expected_comparison_class]
    with tempfile.TemporaryDirectory(prefix="gpmeep-aunp-expected-") as temporary:
        expected_root = extract_verified_aunp(args.expected_archive, pathlib.Path(temporary))
        expected = {
            name: _load_spectra(expected_root / "expected" / f"{name}_spectra.npz")
            for name in AUNP_OUTPUT_POLARIZATIONS
        }
        expected_results = {
            name: compare_spectra_pair(
                expected[name], reference_spectra[name], expected_tolerance
            )
            for name in expected
        }
        packaged_fit = json.loads(
            (expected_root / "expected" / "A_resonance_fit.json").read_text(
                encoding="utf-8"
            )
        )
        expected_fit_keys = set(fit_tm_resonance(expected["TM_Ex"]))
        if set(packaged_fit) != expected_fit_keys:
            raise WorkloadError("packaged AuNP resonance-fit schema is not exact")
        packaged_fit_result = compare_resonance_fit(
            packaged_fit,
            fit_tm_resonance(reference_spectra["TM_Ex"]),
            expected_tolerance,
        )
        expected_outcome = (
            "PASS"
            if all(item["outcome"] == "PASS" for item in expected_results.values())
            and packaged_fit_result["outcome"] == "PASS"
            else "FAIL"
        )
        failed |= expected_outcome != "PASS"
    comparisons.append(
        {
            "label": "packaged-upstream-expected",
            "comparison_class": expected_comparison_class,
            "polarizations": expected_results,
            "packaged_resonance_fit": packaged_fit_result,
            "outcome": expected_outcome,
        }
    )
    for label, comparison_class, path in args.candidate:
        lane = load_lane(path, AUNP_SCHEMA)
        _validate_backend_roles(reference_lane, lane, label, comparison_class)
        candidate_provenance = validate_lane_provenance(lane)
        candidate_stage_resume = validate_aunp_exact_stage_resume(lane)
        if candidate_provenance["source_start"] != reference_provenance["source_start"]:
            raise WorkloadError(f"AuNP source/build snapshot differs for candidate {label}")
        _validate_precision_pair(reference_lane, lane, comparison_class)
        paths = _aunp_spectra_files(lane)
        results = {
            name: compare_spectra_pair(
                reference_spectra[name],
                _load_spectra(paths[name]),
                _comparison_tolerance(comparison_class),
            )
            for name in reference_spectra
        }
        scientific_outputs = compare_aunp_scientific_outputs(
            reference_manifest,
            candidate_manifest := load_aunp_output_manifest(lane),
            _comparison_tolerance(comparison_class),
            comparison_class,
        )
        if validate_aunp_simulation_config(lane, candidate_manifest) != reference_config:
            raise WorkloadError(
                f"AuNP normalized physical configuration differs for candidate {label}"
            )
        outcome = (
            "PASS"
            if all(item["outcome"] == "PASS" for item in results.values())
            and scientific_outputs["outcome"] == "PASS"
            else "FAIL"
        )
        failed |= outcome != "PASS"
        comparisons.append(
            {
                "label": label,
                "comparison_class": comparison_class,
                "candidate": str(path.resolve()),
                "provenance": candidate_provenance,
                "exact_stage_resume": candidate_stage_resume,
                "polarizations": results,
                "scientific_outputs": scientific_outputs,
                "outcome": outcome,
            }
        )
    return {
        "schema": "gpmeep-aunp-r4000-comparison-v2",
        "reference": str(args.reference.resolve()),
        "reference_provenance": reference_provenance,
        "reference_exact_stage_resume": reference_stage_resume,
        "comparisons": comparisons,
        "outcome": "FAIL" if failed else "PASS",
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# gpmeep user-workload comparison",
        "",
        f"Overall outcome: **{report['outcome']}**",
        "",
        "| Comparison | Class | Outcome |",
        "| --- | --- | --- |",
    ]
    for item in report["comparisons"]:
        lines.append(
            f"| `{item['label']}` | `{item['comparison_class']}` | "
            f"**{item['outcome']}** |"
        )
    lines.extend(
        [
            "",
            "Raw full-vector metrics and fixed tolerances are retained in `report.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    subparsers = parser.add_subparsers(dest="workload", required=True)
    for name in ("ters", "aunp"):
        child = subparsers.add_parser(name, allow_abbrev=False)
        child.add_argument("--reference", required=True, type=pathlib.Path)
        child.add_argument(
            "--candidate", required=True, action="append", type=parse_candidate
        )
        if name == "aunp":
            child.add_argument("--expected-archive", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise WorkloadError(f"comparison output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = compare_ters(args) if args.workload == "ters" else compare_aunp(args)
    atomic_write_json(output / "report.json", report)
    atomic_write_text(output / "report.md", markdown_report(report))
    atomic_write_json(
        output / "COMPLETE",
        {
            "schema": "gpmeep-user-workload-comparison-complete-v2",
            "report": file_record(output / "report.json", output),
            "markdown": file_record(output / "report.md", output),
            "outcome": report["outcome"],
        },
    )
    return 0 if report["outcome"] == "PASS" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WorkloadError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"user workload comparison error: {error}", file=sys.stderr)
        raise SystemExit(2)
