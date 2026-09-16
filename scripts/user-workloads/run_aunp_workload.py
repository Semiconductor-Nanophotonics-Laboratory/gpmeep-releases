#!/usr/bin/env python3
"""Run the exact TM-only AuNP r4000 qualification under gpmeep.

The archive remains byte-exact.  The user-selected qualification profile runs
only physical Ex/TM, changes the minimum DFT window to 20 um/c, and requires
all DFT monitors to decay below 5e-8.  MPI/output/recovery policy is isolated
per lane and every physical override is retained in the sealed contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import importlib.util
import json
import os
import pathlib
import socket
import sys
import time
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    AUNP_ARCHIVE_SHA256,
    AUNP_DFT_DECAY_TOLERANCE,
    AUNP_DFT_MINIMUM_RUN_TIME,
    AUNP_ENGINE_POLARIZATIONS,
    AUNP_EXPECTED_DEFAULT_CONFIG,
    AUNP_MEMBER_SHA256,
    AUNP_OUTPUT_POLARIZATIONS,
    AUNP_PHYSICAL_OVERRIDES,
    WorkloadError,
    absolute_file_record,
    aunp_qualification_profile,
    atomic_write_json,
    extract_verified_aunp,
    file_record,
    json_safe,
    phase_call_totals,
    phase_counter_view,
    physical_core_affinity,
    statistics_delta,
    validate_phase_contract,
    verify_file,
    verify_build_receipt_collective,
)
import aunp_exact_resume as exact_resume  # noqa: E402


SCHEMA = "gpmeep-aunp-r4000-workload-v4"
PHASE_CONTRACT = [
    {
        "stage": stage,
        "required_per_rank": ["curl", "update_eh"],
        "required_aggregate": ["source", "boundary", "dft"]
        + (["polarization"] if structure else []),
        "forbidden": [] if structure else ["polarization"],
    }
    for stage, structure in (("TM_Ex-reference", False), ("TM_Ex-structure", True))
]
STAGE_STATE_DIRECTORY = "exact-stage-state"
STAGE_CHECKPOINT_NAME = "CHECKPOINT.json"
STAGE_EVENT_DIRECTORY = "events"
STAGE_LOCK_NAME = "ENGINE.lock"
ALLOWED_OUTPUT_ROOT_ENTRIES = {
    "input",
    "outputs",
    STAGE_STATE_DIRECTORY,
    "attempt-quarantine",
    "summary.json",
    "COMPLETE",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--build-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--expected-backend", required=True, choices=("cpu", "cuda"))
    return parser.parse_args(argv)


def _load_engine(path: pathlib.Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("gpmeep_user_aunp_engine", path)
    if spec is None or spec.loader is None:
        raise WorkloadError(f"could not load AuNP engine: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_backend_record(record: dict[str, Any], expected_backend: str) -> None:
    if record["active_backend"] != expected_backend:
        raise WorkloadError(
            f"AuNP selected {record['active_backend']!r}, expected {expected_backend!r}"
        )
    if record["timestep_delta"] <= 0 or record["wall_seconds"] <= 0:
        raise WorkloadError("AuNP did not perform positive timed FDTD work")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _verified_extracted_package(input_root: pathlib.Path) -> pathlib.Path:
    actual = set(exact_resume.inventory_files(input_root))
    expected = set(AUNP_MEMBER_SHA256)
    if actual != expected:
        raise WorkloadError(
            "AuNP extracted input inventory differs: "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    for relative, digest in AUNP_MEMBER_SHA256.items():
        verify_file(
            input_root.joinpath(*pathlib.PurePosixPath(relative).parts),
            digest,
            f"AuNP extracted member {relative}",
        )
    return input_root / "aunp_r4000_repro"


def _prepare_package(
    archive: pathlib.Path, output: pathlib.Path
) -> tuple[pathlib.Path, bool]:
    """Prepare or replay the archive without adopting a partial extraction."""

    if output.is_symlink():
        raise WorkloadError(f"AuNP output root must not be a symlink: {output}")
    output.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in output.iterdir()} - ALLOWED_OUTPUT_ROOT_ENTRIES
    if unexpected:
        raise WorkloadError(
            f"AuNP output root contains unexpected entries: {sorted(unexpected)}"
        )
    directory_entries = {
        "input",
        "outputs",
        STAGE_STATE_DIRECTORY,
        "attempt-quarantine",
    }
    for path in output.iterdir():
        if path.is_symlink():
            raise WorkloadError(f"AuNP output root contains a symlink: {path}")
        if path.name in directory_entries and not path.is_dir():
            raise WorkloadError(f"AuNP output entry must be a directory: {path}")
        if path.name not in directory_entries and not path.is_file():
            raise WorkloadError(f"AuNP output entry must be a regular file: {path}")
    input_root = output / "input"
    events = output / STAGE_STATE_DIRECTORY / STAGE_EVENT_DIRECTORY
    if input_root.exists():
        try:
            return _verified_extracted_package(input_root), True
        except WorkloadError:
            if events.exists() and any(events.iterdir()):
                raise
            quarantine_parent = output / "attempt-quarantine"
            quarantine_parent.mkdir(parents=True, exist_ok=True)
            quarantine = None
            for index in range(1000):
                candidate = quarantine_parent / f"bootstrap-input-{index:03d}"
                if not candidate.exists() and not candidate.is_symlink():
                    quarantine = candidate
                    break
            if quarantine is None:
                raise WorkloadError("AuNP bootstrap input quarantine slots are exhausted")
            input_root.replace(quarantine)
    package_root = extract_verified_aunp(archive, input_root)
    _verified_extracted_package(input_root)
    return package_root, False


def _stage_contract(
    archive: pathlib.Path,
    output: pathlib.Path,
    engine_path: pathlib.Path,
    build_receipt: dict[str, Any],
    runtime_artifacts: dict[str, Any],
    default_config: dict[str, Any],
    effective_config: dict[str, Any],
    expected_backend: str,
    mpi_size: int,
) -> dict[str, Any]:
    repo = SCRIPT_DIR.parents[1]
    return {
        "schema": "gpmeep-aunp-exact-stage-contract-v3",
        "archive": absolute_file_record(archive, "AuNP archive"),
        "archive_member_sha256": dict(sorted(AUNP_MEMBER_SHA256.items())),
        "engine": file_record(engine_path, output),
        "adapter": file_record(pathlib.Path(__file__), repo),
        "adapter_common": file_record(SCRIPT_DIR / "common.py", repo),
        "stage_state_code": file_record(
            SCRIPT_DIR / "aunp_exact_resume.py", repo
        ),
        "build_receipt": build_receipt,
        "build_receipt_id": build_receipt["receipt_id"],
        "runtime_artifacts": runtime_artifacts,
        "default_config": default_config,
        "effective_config": effective_config,
        "qualification_profile": aunp_qualification_profile(),
        "expected_backend": expected_backend,
        "mpi_size": mpi_size,
        "stage_order": list(exact_resume.STAGE_ORDER),
        "reference_required_basenames": list(
            exact_resume.REFERENCE_REQUIRED_BASENAMES
        ),
        "structure_required_basenames": list(
            exact_resume.STRUCTURE_REQUIRED_BASENAMES
        ),
        "postprocessing_required_root_basenames": list(
            exact_resume.POSTPROCESS_REQUIRED_ROOT_BASENAMES
        ),
    }


def _postprocess_tm_only(engine: Any, cfg: Any, root: pathlib.Path) -> pathlib.Path:
    """Run the package's scientific postprocessing for the selected TM lane."""

    for polarization in AUNP_ENGINE_POLARIZATIONS:
        stage_directory = root / engine.polarization_directory_name(polarization)
        required_files = (
            stage_directory / "reference_complete.json",
            stage_directory / "structure_complete.json",
            stage_directory / "reference_data.npz",
            stage_directory / "reference_dense_probes.npz",
            stage_directory / "reference_dense_line_metadata.npz",
            stage_directory / "spectra.npz",
            stage_directory / "field_metadata.npz",
            stage_directory / "raw_complex_fields.h5",
            stage_directory / "structure_dense_probes.npz",
            stage_directory / "structure_dense_line_metadata.npz",
            stage_directory / "material_masks.npz",
            stage_directory / "str_scat_m1.h5",
            stage_directory / "str_scat_m2.h5",
            stage_directory / "str_bottom_flux.h5",
            *(
                stage_directory / f"{prefix}_line_{line_name}.h5"
                for prefix in ("ref", "str")
                for line_name in ("vertical", "gap", "top")
            ),
        )
        missing = [path for path in required_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "TM-only simulation output is incomplete: "
                + ", ".join(str(path) for path in missing)
            )
        engine.validate_dft_h5_components(
            stage_directory / "raw_complex_fields.h5",
            polarization,
            len(engine.field_wavelengths_nm(cfg)),
        )
        dense_frequency_count = len(engine.dense_analysis_wavelengths_nm(cfg))
        for prefix in ("ref", "str"):
            for line_name in ("vertical", "gap", "top"):
                engine.validate_dft_h5_components(
                    stage_directory / f"{prefix}_line_{line_name}.h5",
                    polarization,
                    dense_frequency_count,
                )
        engine.plot_field_maps_and_metrics(cfg, root, polarization)
        engine.plot_dense_line_heatmaps(root, polarization)
        engine.plot_dense_probe_spectra(root, polarization)
    engine.write_json_master(
        root / "postprocess_complete.json",
        {
            "completed": True,
            "completed_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "qualification_profile": aunp_qualification_profile(),
        },
    )
    engine.write_output_manifest(root)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive = verify_file(args.archive, AUNP_ARCHIVE_SHA256, "AuNP archive")
    output = args.output.resolve()

    import meep as mp
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    repo = SCRIPT_DIR.parents[1]
    build_receipt, runtime_artifacts = verify_build_receipt_collective(
        comm, rank, args.build_receipt, repo, mp
    )
    preparation_error = None
    if rank == 0:
        try:
            package_root, _ = _prepare_package(archive, output)
        except Exception as exc:
            preparation_error = f"{type(exc).__name__}: {exc}"
            package_root = None
    else:
        package_root = None
    preparation_error = comm.bcast(preparation_error, root=0)
    if preparation_error is not None:
        raise WorkloadError(preparation_error)
    comm.Barrier()
    package_root = output / "input" / "aunp_r4000_repro"
    engine_path = package_root / "fdtd" / "meep" / "aunp_periodic_fdtd.py"
    sys.path.insert(0, str(engine_path.parent))
    sys.path.insert(0, str(package_root / "fdtd"))
    engine = _load_engine(engine_path)

    default_config = engine.serializable_config(engine.AppConfig())
    if default_config != AUNP_EXPECTED_DEFAULT_CONFIG:
        raise WorkloadError("AuNP package defaults differ from the sealed physical contract")
    cfg = dataclasses.replace(
        engine.AppConfig(),
        run_mode="simulate",
        output_dir=output / "outputs",
        required_mpi_ranks=size,
        dft_minimum_run_time=AUNP_DFT_MINIMUM_RUN_TIME,
        dft_decay_tolerance=AUNP_DFT_DECAY_TOLERANCE,
    )
    effective_config = engine.serializable_config(cfg)
    changed_config = {
        key
        for key in set(default_config) | set(effective_config)
        if default_config.get(key) != effective_config.get(key)
    }
    expected_changed_config = {
        "output_dir",
        "required_mpi_ranks",
        *AUNP_PHYSICAL_OVERRIDES,
    }
    if changed_config != expected_changed_config:
        raise WorkloadError(
            f"AuNP adapter changed forbidden configuration keys: {sorted(changed_config)}"
        )
    cfg.validate()
    engine.validate_runtime_resources(cfg)

    stage_contract = None
    if rank == 0:
        stage_contract = _stage_contract(
            archive,
            output,
            engine_path,
            build_receipt,
            runtime_artifacts,
            default_config,
            effective_config,
            args.expected_backend,
            size,
        )
    stage_contract = comm.bcast(stage_contract, root=0)
    contract_sha256 = exact_resume.sha256_bytes(
        exact_resume.canonical_json_bytes(stage_contract)
    )
    state_root = output / STAGE_STATE_DIRECTORY
    events_directory = state_root / STAGE_EVENT_DIRECTORY
    stage_checkpoint_path = state_root / STAGE_CHECKPOINT_NAME
    simulation_root_path = engine.simulation_root(cfg)
    state_result = None
    if rank == 0:
        try:
            state_root.mkdir(parents=True, exist_ok=True)
            if state_root.is_symlink() or not state_root.is_dir():
                raise WorkloadError(
                    f"AuNP stage-state root is not a non-symlink directory: {state_root}"
                )
            unexpected_state = {
                path.name for path in state_root.iterdir()
            } - {STAGE_EVENT_DIRECTORY, STAGE_CHECKPOINT_NAME, STAGE_LOCK_NAME}
            if unexpected_state:
                raise WorkloadError(
                    "AuNP stage-state directory contains unexpected entries: "
                    f"{sorted(unexpected_state)}"
                )
            existing_events = exact_resume.load_events(events_directory)
            if not existing_events:
                if stage_checkpoint_path.exists():
                    raise WorkloadError(
                        "AuNP stage checkpoint exists without its durable journal"
                    )
                exact_resume.append_event(
                    events_directory,
                    contract_sha256,
                    "initialized",
                    {"contract": stage_contract},
                )
                checkpoint = exact_resume.publish_checkpoint(
                    stage_checkpoint_path, stage_contract, events_directory
                )
            else:
                checkpoint = exact_resume.replay_checkpoint(
                    stage_checkpoint_path, stage_contract, events_directory
                )

            allowed_paths = exact_resume.replay_completed_output(
                simulation_root_path, checkpoint
            )
            if checkpoint["publication"] is not None:
                publication = checkpoint["publication"]
                exact_resume.replay_manifest(
                    output,
                    [publication["summary"], publication["complete"]],
                    "AuNP terminal publication",
                )
                state_result = {
                    "checkpoint": checkpoint,
                    "attempt_index": None,
                    "already_complete": True,
                }
            else:
                attempt_index = len(checkpoint["attempts"])
                exact_resume.append_event(
                    events_directory,
                    contract_sha256,
                    "attempt-started",
                    {"attempt_index": attempt_index, "started_utc": _utc_now()},
                )
                checkpoint = exact_resume.publish_checkpoint(
                    stage_checkpoint_path, stage_contract, events_directory
                )

                quarantine_parent = output / "attempt-quarantine"
                sealed_quarantine_roots = {
                    item["root"] for item in checkpoint["quarantines"]
                }
                if quarantine_parent.exists():
                    for path in sorted(quarantine_parent.iterdir()):
                        if not path.is_dir() or path.is_symlink():
                            raise WorkloadError(
                                f"invalid AuNP quarantine entry: {path}"
                            )
                        relative = path.relative_to(output).as_posix()
                        if relative in sealed_quarantine_roots:
                            exact_resume.replay_manifest(
                                path,
                                next(
                                    item["manifest"]
                                    for item in checkpoint["quarantines"]
                                    if item["root"] == relative
                                ),
                                f"AuNP retained quarantine {relative}",
                                exact_inventory=True,
                            )
                            continue
                        recovered_attempt_index = attempt_index
                        if path.name.startswith("attempt-") and path.name[8:].isdigit():
                            recovered_attempt_index = int(path.name[8:])
                            if recovered_attempt_index >= len(checkpoint["attempts"]):
                                raise WorkloadError(
                                    "AuNP recovered quarantine references a future attempt"
                                )
                        recovered = {
                            "attempt_index": recovered_attempt_index,
                            "reason": "recovered durable quarantine after interruption",
                            "root": relative,
                            "manifest": exact_resume.make_manifest(path),
                        }
                        exact_resume.append_event(
                            events_directory,
                            contract_sha256,
                            "quarantine-sealed",
                            recovered,
                        )
                        checkpoint = exact_resume.publish_checkpoint(
                            stage_checkpoint_path, stage_contract, events_directory
                        )

                quarantined = exact_resume.quarantine_unsealed_files(
                    simulation_root_path,
                    allowed_paths,
                    quarantine_parent,
                    attempt_index,
                    "unsealed files after the last receipt-bound exact stage",
                )
                if quarantined is not None:
                    exact_resume.append_event(
                        events_directory,
                        contract_sha256,
                        "quarantine-sealed",
                        quarantined,
                    )
                    checkpoint = exact_resume.publish_checkpoint(
                        stage_checkpoint_path, stage_contract, events_directory
                    )
                state_result = {
                    "checkpoint": checkpoint,
                    "attempt_index": attempt_index,
                    "already_complete": False,
                }
        except Exception as exc:
            state_result = {"error": f"{type(exc).__name__}: {exc}"}
    state_result = comm.bcast(state_result, root=0)
    if "error" in state_result:
        raise WorkloadError(f"AuNP exact-stage recovery failed: {state_result['error']}")
    checkpoint = state_result["checkpoint"]
    if state_result["already_complete"]:
        comm.Barrier()
        return 0
    attempt_index = state_result["attempt_index"]

    records: list[dict[str, Any]] = []
    original_run = mp.Simulation.run
    original_wait_for_stage_memory = engine.wait_for_stage_memory

    def audited_run(simulation: Any, *step_functions: Any, **kwargs: Any) -> Any:
        before_timestep = int(simulation.fields.t) if simulation.fields is not None else 0
        before_stats_full = json_safe(mp.gpu.statistics())
        comm.Barrier()
        started = time.perf_counter()
        result = original_run(simulation, *step_functions, **kwargs)
        comm.Barrier()
        wall_seconds = time.perf_counter() - started
        record = None
        validation_error = None
        try:
            after_stats_full = json_safe(mp.gpu.statistics())
            delta = statistics_delta(before_stats_full, after_stats_full)
            record = {
                "rank": rank,
                "run_index": len(records),
                "timestep": int(simulation.timestep()),
                "timestep_delta": int(simulation.timestep()) - before_timestep,
                "meep_time": float(simulation.meep_time()),
                "wall_seconds": wall_seconds,
                "active_backend": str(mp.gpu.active_backend),
                "requested_backend": str(
                    getattr(mp.gpu, "requested_backend", mp.gpu.active_backend)
                ),
                "device_ordinal": int(mp.gpu.selected_device),
                "device_identifier": str(mp.gpu.selected_device_identifier)
                if args.expected_backend == "cuda"
                else None,
                "phase_calls": phase_call_totals(delta),
                "phase_counters": phase_counter_view(delta),
                "statistics_before": before_stats_full,
                "statistics_after": after_stats_full,
                "statistics_delta": delta,
            }
            _validate_backend_record(record, args.expected_backend)
        except Exception as exc:
            validation_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        validation_errors = comm.allgather(validation_error)
        if any(error is not None for error in validation_errors):
            raise WorkloadError(
                "collective AuNP run audit failed: "
                + "; ".join(error for error in validation_errors if error is not None)
            )
        assert record is not None
        records.append(record)
        return result

    def require_stage_memory(
        _cfg: Any, stage_name: str, required_gib: float
    ) -> None:
        state = None
        if rank == 0:
            try:
                state = {"snapshot": engine.memory_admission_snapshot()}
            except Exception as exc:
                state = {"error": f"{type(exc).__name__}: {exc}"}
        state = comm.bcast(state, root=0)
        if "error" in state:
            raise WorkloadError(f"memory admission snapshot failed: {state['error']}")
        snapshot = state["snapshot"]
        available = float(snapshot["effective_available_gib"])
        if available < required_gib:
            raise WorkloadError(
                f"insufficient memory for {stage_name}: {available:.3f} GiB "
                f"available, {required_gib:.3f} GiB required"
            )
        if rank == 0:
            print(
                f"Memory gate passed for {stage_name}: {available:.1f}/"
                f"{required_gib:.1f} GiB.",
                flush=True,
            )

    def _restore_rank_record(stage_index: int) -> dict[str, Any]:
        sealed = checkpoint["completed_stages"][stage_index]
        rank_records = sealed["rank_records"]
        if len(rank_records) != size:
            raise WorkloadError("AuNP sealed stage rank inventory differs")
        record = json.loads(json.dumps(rank_records[rank], allow_nan=False))
        if (
            record.get("rank") != rank
            or record.get("run_index") != stage_index
            or record.get("requested_backend") != args.expected_backend
        ):
            raise WorkloadError("AuNP sealed stage rank record identity differs")
        _validate_backend_record(record, args.expected_backend)
        return record

    def _run_or_restore_fdtd_stage(
        stage: str,
        original_stage: Any,
        stage_cfg: Any,
        polarization: Any,
        stage_directory: pathlib.Path,
    ) -> Any:
        nonlocal checkpoint
        stage_index = exact_resume.STAGE_ORDER.index(stage)
        completed_count = len(checkpoint["completed_stages"])
        if stage_index < completed_count:
            replay_error = None
            if rank == 0:
                try:
                    exact_resume.replay_manifest(
                        simulation_root_path,
                        checkpoint["completed_stages"][stage_index]["manifest"],
                        f"AuNP resume {stage}",
                    )
                except Exception as exc:
                    replay_error = f"{type(exc).__name__}: {exc}"
            replay_error = comm.bcast(replay_error, root=0)
            if replay_error is not None:
                raise WorkloadError(f"AuNP sealed stage replay failed: {replay_error}")
            records.append(_restore_rank_record(stage_index))
            return None
        if stage_index != completed_count or len(records) != stage_index:
            raise WorkloadError("AuNP exact stage execution order differs")

        before_count = len(records)
        stage_started = time.perf_counter()
        result = original_stage(stage_cfg, polarization, stage_directory)
        stage_process_wall = time.perf_counter() - stage_started
        if len(records) != before_count + 1:
            raise WorkloadError(f"AuNP {stage} did not produce one audited FDTD record")
        gathered_stage_payloads = comm.gather(
            {"record": records[-1], "process_wall_seconds": stage_process_wall},
            root=0,
        )
        publication = None
        if rank == 0:
            try:
                gathered_stage = [item["record"] for item in gathered_stage_payloads]
                manifest = exact_resume.seal_fdtd_stage(
                    simulation_root_path,
                    stage,
                    [
                        item["manifest"]
                        for item in checkpoint["completed_stages"]
                    ],
                )
                validate_phase_contract(
                    [
                        {"rank": candidate_rank, "records": [record]}
                        for candidate_rank, record in enumerate(gathered_stage)
                    ],
                    args.expected_backend,
                    [PHASE_CONTRACT[stage_index]],
                )
                exact_resume.append_event(
                    events_directory,
                    contract_sha256,
                    "stage-sealed",
                    {
                        "stage": stage,
                        "attempt_index": attempt_index,
                        "manifest": manifest,
                        "rank_records": gathered_stage,
                        "wall_seconds": max(
                            *(
                                float(item["process_wall_seconds"])
                                for item in gathered_stage_payloads
                            ),
                            *(float(record["wall_seconds"]) for record in gathered_stage),
                        ),
                    },
                )
                publication = {
                    "checkpoint": exact_resume.publish_checkpoint(
                        stage_checkpoint_path, stage_contract, events_directory
                    )
                }
            except Exception as exc:
                publication = {"error": f"{type(exc).__name__}: {exc}"}
        publication = comm.bcast(publication, root=0)
        if "error" in publication:
            raise WorkloadError(
                f"AuNP exact stage publication failed: {publication['error']}"
            )
        checkpoint = publication["checkpoint"]
        comm.Barrier()
        return result

    original_reference_stage = engine.run_reference_stage
    original_structure_stage = engine.run_structure_stage
    original_previews = engine.render_simulation_stage_previews
    original_postprocess = engine.postprocess_simulation

    def resumable_reference_stage(
        stage_cfg: Any, polarization: Any, stage_directory: pathlib.Path
    ) -> Any:
        directory = engine.polarization_directory_name(polarization)
        if directory not in AUNP_OUTPUT_POLARIZATIONS:
            return None
        stage = f"{directory}-reference"
        return _run_or_restore_fdtd_stage(
            stage,
            original_reference_stage,
            stage_cfg,
            polarization,
            stage_directory,
        )

    def resumable_structure_stage(
        stage_cfg: Any, polarization: Any, stage_directory: pathlib.Path
    ) -> Any:
        directory = engine.polarization_directory_name(polarization)
        if directory not in AUNP_OUTPUT_POLARIZATIONS:
            return None
        stage = f"{directory}-structure"
        return _run_or_restore_fdtd_stage(
            stage,
            original_structure_stage,
            stage_cfg,
            polarization,
            stage_directory,
        )

    def resumable_previews(stage_cfg: Any, root: pathlib.Path) -> Any:
        nonlocal checkpoint
        if checkpoint["base_manifest"] is not None:
            paths = exact_resume.manifest_paths(
                checkpoint["base_manifest"], "AuNP resumed TM-only base"
            )
            if paths != set(exact_resume.BASE_REQUIRED_PATHS):
                raise WorkloadError("AuNP resumed TM-only base manifest differs")
            return []
        if checkpoint["completed_stages"]:
            raise WorkloadError("AuNP exact stages exist without a sealed base")
        # The immutable archive's preview helper constructs both physical-Ey/TE
        # and physical-Ex/TM simulations.  The qualification contract is
        # literally TM-only, so even unstepped TE construction is forbidden.
        # Preview images are not scientific outputs and are omitted entirely.
        result: list[pathlib.Path] = []
        publication = None
        if rank == 0:
            try:
                expected_base = set(exact_resume.BASE_REQUIRED_PATHS)
                actual = set(exact_resume.inventory_files(root))
                if actual != expected_base:
                    raise WorkloadError(
                        "AuNP preflight file ownership differs: "
                        f"missing={sorted(expected_base-actual)}, "
                        f"extra={sorted(actual-expected_base)}"
                    )
                exact_resume.append_event(
                    events_directory,
                    contract_sha256,
                    "base-sealed",
                    {"manifest": exact_resume.make_manifest(root, expected_base)},
                )
                publication = {
                    "checkpoint": exact_resume.publish_checkpoint(
                        stage_checkpoint_path, stage_contract, events_directory
                    )
                }
            except Exception as exc:
                publication = {"error": f"{type(exc).__name__}: {exc}"}
        publication = comm.bcast(publication, root=0)
        if "error" in publication:
            raise WorkloadError(
                f"AuNP base-stage publication failed: {publication['error']}"
            )
        checkpoint = publication["checkpoint"]
        comm.Barrier()
        return result

    def resumable_postprocess(stage_cfg: Any, root: pathlib.Path | None = None) -> Any:
        nonlocal checkpoint
        root = engine.simulation_root(stage_cfg) if root is None else root
        stage_index = exact_resume.STAGE_ORDER.index("postprocessing")
        if len(checkpoint["completed_stages"]) > stage_index:
            exact_resume.replay_manifest(
                simulation_root_path,
                checkpoint["completed_stages"][stage_index]["manifest"],
                "AuNP resumed postprocessing",
                exact_inventory=True,
            )
            return root
        if len(checkpoint["completed_stages"]) != stage_index:
            raise WorkloadError("AuNP postprocessing began before two sealed TM FDTD stages")
        stage_started = time.perf_counter()
        result = _postprocess_tm_only(engine, stage_cfg, root)
        manifest = exact_resume.seal_postprocessing(simulation_root_path)
        exact_resume.append_event(
            events_directory,
            contract_sha256,
            "stage-sealed",
            {
                "stage": "postprocessing",
                "attempt_index": attempt_index,
                "manifest": manifest,
                "rank_records": [],
                "wall_seconds": time.perf_counter() - stage_started,
            },
        )
        checkpoint = exact_resume.publish_checkpoint(
            stage_checkpoint_path, stage_contract, events_directory
        )
        return result

    def acquire_adapter_lock(_cfg: Any):
        lock_handle = None
        lock_state = None
        if rank == 0:
            try:
                lock_path = state_root / STAGE_LOCK_NAME
                flags = (
                    os.O_RDWR
                    | os.O_CREAT
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(lock_path, flags, 0o600)
                lock_handle = os.fdopen(descriptor, "r+", encoding="utf-8")
                fcntl.flock(
                    lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
                lock_handle.seek(0)
                lock_handle.truncate()
                lock_handle.write(
                    json.dumps(
                        {
                            "hostname": socket.gethostname(),
                            "pid": os.getpid(),
                            "attempt_index": attempt_index,
                            "acquired_utc": _utc_now(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                lock_handle.flush()
                os.fsync(lock_handle.fileno())
                lock_state = {"acquired": True}
            except Exception as exc:
                if lock_handle is not None:
                    lock_handle.close()
                    lock_handle = None
                lock_state = {"error": f"{type(exc).__name__}: {exc}"}
        lock_state = comm.bcast(lock_state, root=0)
        if "error" in lock_state:
            raise WorkloadError(f"AuNP adapter lock failed: {lock_state['error']}")
        return lock_handle

    def release_adapter_lock(lock_handle: Any) -> None:
        if rank == 0 and lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    previous_cwd = pathlib.Path.cwd()
    started = time.perf_counter()
    mp.Simulation.run = audited_run
    engine.wait_for_stage_memory = require_stage_memory
    engine.render_simulation_stage_previews = resumable_previews
    engine.run_reference_stage = resumable_reference_stage
    engine.run_structure_stage = resumable_structure_stage
    engine.postprocess_simulation = resumable_postprocess
    original_acquire_lock = engine.acquire_global_run_lock
    original_release_lock = engine.release_global_run_lock
    engine.acquire_global_run_lock = acquire_adapter_lock
    engine.release_global_run_lock = release_adapter_lock
    mp.gpu.reset_statistics()
    try:
        os.chdir(output)
        simulation_root = engine.run_simulations(cfg)
        checkpoint = comm.bcast(checkpoint if rank == 0 else None, root=0)
    finally:
        engine.release_global_run_lock = original_release_lock
        engine.acquire_global_run_lock = original_acquire_lock
        engine.postprocess_simulation = original_postprocess
        engine.run_structure_stage = original_structure_stage
        engine.run_reference_stage = original_reference_stage
        engine.render_simulation_stage_previews = original_previews
        engine.wait_for_stage_memory = original_wait_for_stage_memory
        mp.Simulation.run = original_run
        os.chdir(previous_cwd)
    comm.Barrier()
    process_wall = time.perf_counter() - started

    if len(records) != len(PHASE_CONTRACT):
        raise WorkloadError(
            "AuNP expected two audited or restored TM FDTD records, "
            f"got {len(records)}"
        )
    local = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "records": records,
        "runtime_artifacts": runtime_artifacts,
        "physical_cores": physical_core_affinity(),
        "build_receipt_id": build_receipt["receipt_id"],
        "process_wall_seconds": process_wall,
        "final_statistics": json_safe(mp.gpu.statistics()),
    }
    gathered = comm.gather(local, root=0)
    comm.Barrier()

    publication_error = None
    if rank == 0:
        try:
            if [
                item["stage"] for item in checkpoint["completed_stages"]
            ] != list(exact_resume.STAGE_ORDER):
                raise WorkloadError("AuNP exact-stage checkpoint is incomplete")
            exact_resume.replay_manifest(
                simulation_root,
                checkpoint["completed_stages"][-1]["manifest"],
                "AuNP final exact-stage output",
                exact_inventory=True,
            )
            validate_phase_contract(gathered, args.expected_backend, PHASE_CONTRACT)
            spectra_files = [
                simulation_root / polarization / "spectra.npz"
                for polarization in AUNP_OUTPUT_POLARIZATIONS
            ]
            required = [
                *spectra_files,
                simulation_root / "simulation_config.json",
                simulation_root / "postprocess_complete.json",
                simulation_root / "output_manifest.json",
            ]
            if any(not path.is_file() for path in required):
                raise WorkloadError("AuNP production output is incomplete")
            simulation_config_path = simulation_root / "simulation_config.json"
            simulation_config = json.loads(
                simulation_config_path.read_text(encoding="utf-8")
            )
            if simulation_config.get("config") != effective_config:
                raise WorkloadError("AuNP output manifest changed the effective configuration")
            if simulation_config.get("mpi_processes") != size:
                raise WorkloadError("AuNP output manifest recorded the wrong MPI size")
            if simulation_config.get("script_sha256") != file_record(
                engine_path, output
            )["sha256"]:
                raise WorkloadError("AuNP output manifest recorded the wrong engine")
            summary = {
                "schema": SCHEMA,
                "archive": file_record(archive, archive.parent),
                "archive_sha256": AUNP_ARCHIVE_SHA256,
                "engine": file_record(engine_path, output),
                "expected_backend": args.expected_backend,
                "single_precision": bool(mp.is_single_precision()),
                "meep_version": str(getattr(mp, "__version__", "unknown")),
                "meep_module": str(pathlib.Path(mp.__file__).resolve()),
                "build_receipt": build_receipt,
                "adapter": file_record(pathlib.Path(__file__), repo),
                "adapter_common": file_record(SCRIPT_DIR / "common.py", repo),
                "exact_stage_state_code": file_record(
                    SCRIPT_DIR / "aunp_exact_resume.py", repo
                ),
                "mpi_size": size,
                "rank_records": gathered,
                "phase_contract": PHASE_CONTRACT,
                "simulation_root": simulation_root.relative_to(output).as_posix(),
                "spectra": [file_record(path, output) for path in spectra_files],
                "output_manifest": file_record(
                    simulation_root / "output_manifest.json", output
                ),
                "simulation_config": file_record(simulation_config_path, output),
                "default_config": default_config,
                "effective_config": effective_config,
                "qualification_profile": aunp_qualification_profile(),
                "physical_overrides": dict(AUNP_PHYSICAL_OVERRIDES),
                "execution_policy_overrides": {
                    "required_mpi_ranks": size,
                    "output_dir": str(cfg.output_dir),
                    "memory_wait_policy": "single-snapshot-fail-closed",
                },
                "timing_semantics": {
                    "fdtd_only": "sum of rank-synchronized Simulation.run wall intervals",
                    "adapted_end_to_end": (
                        "maximum rank interval around the exact engine run with only the "
                        "declared output/rank/memory-admission policy adaptations"
                    ),
                    "adapter_process": (
                        "controller-observed wrapper/provenance/publication interval"
                    ),
                },
                "exact_stage_resume": {
                    "schema": checkpoint["schema"],
                    "contract": stage_contract,
                    "attempts": checkpoint["attempts"],
                    "completed_stages": [
                        {
                            "stage": item["stage"],
                            "attempt_index": item["attempt_index"],
                            "wall_seconds": item["wall_seconds"],
                            "manifest": item["manifest"],
                        }
                        for item in checkpoint["completed_stages"]
                    ],
                    "quarantines": checkpoint["quarantines"],
                    "journal_event_count_before_publication": checkpoint[
                        "journal_event_count"
                    ],
                    "journal_terminal_sha256_before_publication": checkpoint[
                        "journal_terminal_sha256"
                    ],
                    "cumulative_sealed_stage_wall_seconds": sum(
                        float(item["wall_seconds"])
                        for item in checkpoint["completed_stages"]
                    ),
                    "current_attempt_index": attempt_index,
                },
            }
            atomic_write_json(output / "summary.json", summary)
            atomic_write_json(
                output / "COMPLETE",
                {
                    "schema": "gpmeep-aunp-r4000-workload-complete-v4",
                    "summary": file_record(output / "summary.json", output),
                },
            )
            exact_resume.append_event(
                events_directory,
                contract_sha256,
                "publication-sealed",
                {
                    "attempt_index": attempt_index,
                    "summary": file_record(output / "summary.json", output),
                    "complete": file_record(output / "COMPLETE", output),
                },
            )
            checkpoint = exact_resume.publish_checkpoint(
                stage_checkpoint_path, stage_contract, events_directory
            )
        except Exception as exc:
            publication_error = f"{type(exc).__name__}: {exc}"
    publication_error = comm.bcast(publication_error, root=0)
    if publication_error is not None:
        raise WorkloadError(f"AuNP evidence publication failed: {publication_error}")
    comm.Barrier()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkloadError as error:
        print(f"AuNP workload error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
