#!/usr/bin/env python3
"""Run a fixed-work, non-scientific TM-only AuNP r4000 performance sample.

The byte-exact archive and TM production monitor inventory are retained.  The
user-selected 20 um/c and 5e-8 exact stop profile is recorded, while the
decay-based stop itself is replaced by an exact fixed Meep-time interval for
timing.  This adapter never publishes spectra or exact-result claims.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
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
    AUNP_EXPECTED_DEFAULT_CONFIG,
    AUNP_MEMBER_SHA256,
    AUNP_PHYSICAL_OVERRIDES,
    StableFile,
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
    validate_package_provenance_attestation,
    verify_file,
    verify_build_receipt_collective,
    verify_package_attestation_collective,
)


SCHEMA = "gpmeep-aunp-r4000-performance-workload-v2"
COMPLETE_SCHEMA = "gpmeep-aunp-r4000-performance-workload-complete-v2"
MEASUREMENT_MODE = "fixed-meep-time-performance-window"
PERFORMANCE_ADAPTATION = (
    "each production decay stop is replaced by a fixed 0.25 Meep-time interval"
)
FIXED_WINDOW_MEEP_TIME = 0.25
MINIMUM_TIMESTEPS_PER_PHASE = 2000
PHASE_SPECS = (
    ("TM_Ex-reference", "physical_ex", False, "analysis_reference"),
    ("TM_Ex-structure", "physical_ex", True, "analysis_structure"),
)
PHASE_CONTRACT = [
    {
        "stage": stage,
        "required_per_rank": ["curl", "update_eh"],
        "required_aggregate": ["source", "boundary", "dft"]
        + (["polarization"] if structure else []),
        "forbidden": [] if structure else ["polarization"],
    }
    for stage, _polarization, structure, _monitor_kind in PHASE_SPECS
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    provenance = parser.add_mutually_exclusive_group(required=True)
    provenance.add_argument("--build-receipt", type=pathlib.Path)
    provenance.add_argument("--package-provenance", type=pathlib.Path)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-package-sha256")
    parser.add_argument("--expected-backend", required=True, choices=("cpu", "cuda"))
    return parser.parse_args(argv)


def _load_engine(path: pathlib.Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "gpmeep_user_aunp_performance_engine", path
    )
    if spec is None or spec.loader is None:
        raise WorkloadError(f"could not load AuNP engine: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_backend_record(record: dict[str, Any], expected_backend: str) -> None:
    if (
        record.get("active_backend") != expected_backend
        or record.get("requested_backend") != expected_backend
    ):
        raise WorkloadError("AuNP performance phase selected the wrong backend")
    if (
        type(record.get("timestep_delta")) is not int
        or record["timestep_delta"] < MINIMUM_TIMESTEPS_PER_PHASE
        or isinstance(record.get("meep_time"), bool)
        or not isinstance(record["meep_time"], (int, float))
        or not math.isfinite(float(record["meep_time"]))
        or float(record["meep_time"]) < FIXED_WINDOW_MEEP_TIME
        or isinstance(record.get("wall_seconds"), bool)
        or not isinstance(record["wall_seconds"], (int, float))
        or not math.isfinite(float(record["wall_seconds"]))
        or record["wall_seconds"] <= 0
    ):
        raise WorkloadError("AuNP performance phase violates its fixed-work contract")


def _finite_positive(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise WorkloadError(f"AuNP performance {label} is not positive and finite")
    return float(value)


def _summary_from_output(output: pathlib.Path) -> dict[str, Any]:
    with StableFile(output / "COMPLETE", "AuNP performance COMPLETE") as stable:
        with stable.file_object() as handle:
            complete = json.load(handle)
    if complete.get("schema") != COMPLETE_SCHEMA:
        raise WorkloadError("AuNP performance COMPLETE schema differs")
    summary_record = complete.get("summary")
    if not isinstance(summary_record, dict):
        raise WorkloadError("AuNP performance COMPLETE has no summary record")
    summary_path = output / str(summary_record.get("path", ""))
    if summary_path.resolve() != (output / "summary.json").resolve():
        raise WorkloadError("AuNP performance COMPLETE binds the wrong summary")
    with StableFile(
        summary_path, "AuNP performance summary", expected=summary_record
    ) as stable:
        with stable.file_object() as handle:
            summary = json.load(handle)
    return summary


def _record_path(
    root: pathlib.Path, record: Any, label: str
) -> pathlib.Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise WorkloadError(f"AuNP performance {label} record schema differs")
    raw = record["path"]
    if not isinstance(raw, str) or not raw:
        raise WorkloadError(f"AuNP performance {label} path is invalid")
    relative = pathlib.PurePosixPath(raw)
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise WorkloadError(f"AuNP performance {label} path is unsafe")
    path = root.joinpath(*relative.parts)
    with StableFile(path, f"AuNP performance {label}", expected=record):
        pass
    return path.resolve()


def _output_file_inventory(output: pathlib.Path) -> set[str]:
    result = set()
    for directory, directory_names, file_names in os.walk(output, followlinks=False):
        current = pathlib.Path(directory)
        if any((current / name).is_symlink() for name in directory_names):
            raise WorkloadError("AuNP performance output contains a directory symlink")
        for name in file_names:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise WorkloadError("AuNP performance output contains a special file")
            result.add(path.relative_to(output).as_posix())
    return result


def validate_performance_output(
    output: pathlib.Path,
    expected_backend: str | None = None,
    expected_mpi_size: int | None = None,
) -> dict[str, Any]:
    """Replay a terminal performance sample without trusting reported timing."""

    output = output.resolve()
    summary = _summary_from_output(output)
    if summary.get("schema") != SCHEMA:
        raise WorkloadError("AuNP performance summary schema differs")
    if (
        summary.get("single_precision") is not True
        or summary.get("measurement_mode") != MEASUREMENT_MODE
        or summary.get("performance_adaptation") != PERFORMANCE_ADAPTATION
        or summary.get("fixed_window_meep_time") != FIXED_WINDOW_MEEP_TIME
        or summary.get("minimum_timesteps_per_phase")
        != MINIMUM_TIMESTEPS_PER_PHASE
        or summary.get("physical_overrides") != AUNP_PHYSICAL_OVERRIDES
        or summary.get("qualification_profile") != aunp_qualification_profile()
        or summary.get("scientific_output_claims") != []
    ):
        raise WorkloadError("AuNP performance adaptation contract differs")
    backend = summary.get("expected_backend")
    mpi_size = summary.get("mpi_size")
    if backend not in ("cpu", "cuda") or type(mpi_size) is not int or mpi_size <= 0:
        raise WorkloadError("AuNP performance topology is invalid")
    if expected_backend is not None and backend != expected_backend:
        raise WorkloadError("AuNP performance backend differs from its task")
    if expected_mpi_size is not None and mpi_size != expected_mpi_size:
        raise WorkloadError("AuNP performance MPI size differs from its task")
    if summary.get("archive_sha256") != AUNP_ARCHIVE_SHA256:
        raise WorkloadError("AuNP performance archive identity differs")
    archive_record = summary.get("archive")
    if (
        not isinstance(archive_record, dict)
        or not pathlib.Path(str(archive_record.get("path", ""))).is_absolute()
    ):
        raise WorkloadError("AuNP performance archive record is not absolute")
    with StableFile(
        pathlib.Path(archive_record["path"]),
        "AuNP performance archive",
        expected=archive_record,
    ) as stable:
        if stable.sha256 != AUNP_ARCHIVE_SHA256:
            raise WorkloadError("AuNP performance archive digest differs")
    repo = SCRIPT_DIR.parents[1]
    if _record_path(repo, summary.get("adapter"), "adapter") != pathlib.Path(
        __file__
    ).resolve():
        raise WorkloadError("AuNP performance adapter path differs")
    if _record_path(
        repo, summary.get("adapter_common"), "common helper"
    ) != (SCRIPT_DIR / "common.py").resolve():
        raise WorkloadError("AuNP performance common-helper path differs")
    engine_path = _record_path(output, summary.get("engine"), "engine")
    expected_engine = (
        output
        / "input"
        / "aunp_r4000_repro"
        / "fdtd"
        / "meep"
        / "aunp_periodic_fdtd.py"
    ).resolve()
    if engine_path != expected_engine or summary["engine"]["sha256"] != (
        AUNP_MEMBER_SHA256[
            "aunp_r4000_repro/fdtd/meep/aunp_periodic_fdtd.py"
        ]
    ):
        raise WorkloadError("AuNP performance engine identity differs")
    if summary.get("default_config") != AUNP_EXPECTED_DEFAULT_CONFIG:
        raise WorkloadError("AuNP performance default physical configuration differs")
    effective = summary.get("effective_config")
    if not isinstance(effective, dict):
        raise WorkloadError("AuNP performance effective configuration is absent")
    changed = {
        key
        for key in set(effective) | set(AUNP_EXPECTED_DEFAULT_CONFIG)
        if effective.get(key) != AUNP_EXPECTED_DEFAULT_CONFIG.get(key)
    }
    if changed != {"output_dir", "required_mpi_ranks", *AUNP_PHYSICAL_OVERRIDES}:
        raise WorkloadError("AuNP performance physical configuration was changed")
    if any(effective.get(name) != value for name, value in AUNP_PHYSICAL_OVERRIDES.items()):
        raise WorkloadError("AuNP performance DFT stop-condition overrides differ")
    if effective["required_mpi_ranks"] != mpi_size:
        raise WorkloadError("AuNP performance effective MPI size differs")
    ranks = summary.get("rank_records")
    if (
        not isinstance(ranks, list)
        or len(ranks) != mpi_size
        or sorted(int(item.get("rank", -1)) for item in ranks)
        != list(range(mpi_size))
    ):
        raise WorkloadError("AuNP performance rank inventory differs")
    provenance_kind = summary.get("runtime_provenance_kind", "build-receipt-v1")
    provenance = summary.get("build_receipt")
    if provenance_kind not in {"build-receipt-v1", "installed-package-v1"}:
        raise WorkloadError("AuNP performance runtime provenance kind differs")
    if not isinstance(provenance, dict) or not isinstance(
        provenance.get("receipt_id"), str
    ):
        raise WorkloadError("AuNP performance runtime provenance is absent")
    package_report = None
    if provenance_kind == "installed-package-v1":
        if set(provenance) != {
            "path",
            "size_bytes",
            "sha256",
            "receipt_id",
            "package_sha256",
            "source_commit",
            "prefix",
        }:
            raise WorkloadError("AuNP installed-package provenance record differs")
        package_report = validate_package_provenance_attestation(
            pathlib.Path(provenance["path"]),
            provenance["source_commit"],
            provenance["package_sha256"],
        )
        with StableFile(
            pathlib.Path(provenance["path"]),
            "AuNP installed-package provenance",
            expected={
                key: provenance[key]
                for key in ("path", "size_bytes", "sha256")
            },
        ):
            pass
        if (
            package_report["report_id"] != provenance["receipt_id"]
            or package_report["package_sha256"] != provenance["package_sha256"]
            or package_report["manifest"]["source_commit"]
            != provenance["source_commit"]
            or package_report["prefix"] != provenance["prefix"]
        ):
            raise WorkloadError("AuNP installed-package provenance ID differs")
    for rank_record in ranks:
        rank = int(rank_record["rank"])
        if (
            rank_record.get("runtime_provenance_kind", "build-receipt-v1")
            != provenance_kind
            or rank_record.get("build_receipt_id") != provenance["receipt_id"]
        ):
            raise WorkloadError("AuNP performance rank provenance differs")
        if package_report is not None:
            artifacts = rank_record.get("runtime_artifacts")
            if not isinstance(artifacts, dict) or any(
                artifacts.get(name) != package_report["runtime"][name]
                for name in ("python", "extension", "libmeep")
            ):
                raise WorkloadError(
                    "AuNP performance rank runtime escaped the installed package"
                )
        records = rank_record.get("records")
        if not isinstance(records, list) or len(records) != len(PHASE_SPECS):
            raise WorkloadError("AuNP performance phase inventory differs")
        for index, record in enumerate(records):
            if (
                record.get("rank") != rank
                or record.get("run_index") != index
                or record.get("stage") != PHASE_SPECS[index][0]
            ):
                raise WorkloadError("AuNP performance phase order differs")
            _validate_backend_record(record, backend)
            _finite_positive(
                record.get("phase_process_seconds_before_reset"),
                "phase process wall time",
            )
    if summary.get("phase_contract") != PHASE_CONTRACT:
        raise WorkloadError("AuNP performance phase contract differs")
    validate_phase_contract(ranks, backend, PHASE_CONTRACT)
    for index in range(len(PHASE_SPECS)):
        if len({rank["records"][index]["timestep_delta"] for rank in ranks}) != 1:
            raise WorkloadError("AuNP performance rank timestep counts differ")
        meep_times = [float(rank["records"][index]["meep_time"]) for rank in ranks]
        if max(meep_times) - min(meep_times) > 1e-12:
            raise WorkloadError("AuNP performance rank Meep times differ")
    phase_wall_seconds = [
        max(float(rank["records"][index]["wall_seconds"]) for rank in ranks)
        for index in range(len(PHASE_SPECS))
    ]
    fdtd_wall_seconds = sum(phase_wall_seconds)
    process_wall_seconds = max(
        _finite_positive(rank.get("process_wall_seconds"), "rank process wall time")
        for rank in ranks
    )
    published_fdtd = _finite_positive(
        summary.get("fdtd_wall_seconds"), "FDTD wall time"
    )
    published_process = _finite_positive(
        summary.get("workload_end_to_end_seconds"), "end-to-end wall time"
    )
    if summary.get("phase_wall_seconds") != phase_wall_seconds or not math.isclose(
        published_fdtd,
        fdtd_wall_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise WorkloadError("AuNP performance FDTD timing was not re-derived")
    if not math.isclose(
        published_process,
        process_wall_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise WorkloadError("AuNP performance process timing was not re-derived")
    if any(
        forbidden in summary
        for forbidden in ("spectra", "output_manifest", "resonance_fit")
    ):
        raise WorkloadError("AuNP performance sample makes a scientific output claim")
    expected_files = {"summary.json", "COMPLETE"} | {
        f"input/{relative}" for relative in AUNP_MEMBER_SHA256
    }
    actual_files = _output_file_inventory(output)
    if actual_files != expected_files:
        raise WorkloadError(
            "AuNP performance output inventory differs: "
            f"missing={sorted(expected_files-actual_files)}, "
            f"extra={sorted(actual_files-expected_files)}"
        )
    for relative, digest in AUNP_MEMBER_SHA256.items():
        verify_file(
            output / "input" / relative,
            digest,
            f"AuNP performance input {relative}",
        )
    return summary


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
    if args.package_provenance is not None:
        if args.expected_source_commit is None or args.expected_package_sha256 is None:
            raise WorkloadError(
                "installed package provenance requires expected source and package identities"
            )
        build_receipt, runtime_artifacts = verify_package_attestation_collective(
            comm,
            rank,
            args.package_provenance,
            args.expected_source_commit,
            args.expected_package_sha256,
            mp,
        )
        runtime_provenance_kind = "installed-package-v1"
    else:
        if args.expected_source_commit is not None or args.expected_package_sha256 is not None:
            raise WorkloadError(
                "expected package identities require --package-provenance"
            )
        assert args.build_receipt is not None
        build_receipt, runtime_artifacts = verify_build_receipt_collective(
            comm, rank, args.build_receipt, repo, mp
        )
        runtime_provenance_kind = "build-receipt-v1"
    preparation = None
    if rank == 0:
        try:
            if output.is_symlink() or (
                output.exists() and (not output.is_dir() or any(output.iterdir()))
            ):
                raise WorkloadError(
                    f"AuNP performance output is not a fresh directory: {output}"
                )
            output.mkdir(parents=True, exist_ok=True)
            package_root = extract_verified_aunp(archive, output / "input")
            preparation = {"package_root": str(package_root)}
        except Exception as exc:
            preparation = {"error": f"{type(exc).__name__}: {exc}"}
    preparation = comm.bcast(preparation, root=0)
    if "error" in preparation:
        raise WorkloadError(f"AuNP performance preparation failed: {preparation['error']}")
    comm.Barrier()

    package_root = pathlib.Path(preparation["package_root"])
    engine_path = package_root / "fdtd" / "meep" / "aunp_periodic_fdtd.py"
    sys.path.insert(0, str(engine_path.parent))
    sys.path.insert(0, str(package_root / "fdtd"))
    engine = _load_engine(engine_path)
    default_config = engine.serializable_config(engine.AppConfig())
    if default_config != AUNP_EXPECTED_DEFAULT_CONFIG:
        raise WorkloadError("AuNP package defaults differ from the sealed contract")
    cfg = dataclasses.replace(
        engine.AppConfig(),
        run_mode="simulate",
        output_dir=output / "ephemeral-engine-output",
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
    if changed_config != {
        "output_dir",
        "required_mpi_ranks",
        *AUNP_PHYSICAL_OVERRIDES,
    }:
        raise WorkloadError("AuNP performance adapter changed physical configuration")
    cfg.validate()
    engine.validate_runtime_resources(cfg)

    def memory_snapshot(stage: str, required_gib: float) -> dict[str, Any]:
        state = None
        if rank == 0:
            try:
                snapshot = engine.memory_admission_snapshot()
                available = float(snapshot["effective_available_gib"])
                if available < required_gib:
                    raise WorkloadError(
                        f"insufficient memory for {stage}: {available:.3f} GiB "
                        f"available, {required_gib:.3f} GiB required"
                    )
                state = {"snapshot": json_safe(snapshot)}
            except Exception as exc:
                state = {"error": f"{type(exc).__name__}: {exc}"}
        state = comm.bcast(state, root=0)
        if "error" in state:
            raise WorkloadError(f"AuNP performance memory gate failed: {state['error']}")
        return state["snapshot"]

    records: list[dict[str, Any]] = []
    reference_flux: dict[str, dict[str, Any]] = {}
    process_started = time.perf_counter()
    previous_cwd = pathlib.Path.cwd()
    mp.gpu.reset_statistics()
    try:
        os.chdir(output)
        for run_index, (stage, polarization, structure, monitor_kind) in enumerate(
            PHASE_SPECS
        ):
            required_gib = (
                float(cfg.structure_memory_gib)
                if structure
                else float(cfg.reference_memory_gib)
            )
            admission = memory_snapshot(stage, required_gib)
            phase_started = time.perf_counter()
            simulation, monitors = engine.build_simulation(
                cfg,
                polarization,
                include_structure=structure,
                field_monitor_kind=monitor_kind,
            )
            if structure:
                saved = reference_flux.pop(polarization, None)
                if not isinstance(saved, dict) or set(saved) != {"monitor1", "monitor2"}:
                    raise WorkloadError("AuNP performance reference flux is absent")
                simulation.load_minus_flux_data(monitors.monitor1, saved["monitor1"])
                simulation.load_minus_flux_data(monitors.monitor2, saved["monitor2"])
            before_timestep = (
                int(simulation.fields.t) if simulation.fields is not None else 0
            )
            before_stats = json_safe(mp.gpu.statistics())
            comm.Barrier()
            fdtd_started = time.perf_counter()
            simulation.run(until=FIXED_WINDOW_MEEP_TIME)
            comm.Barrier()
            wall_seconds = time.perf_counter() - fdtd_started
            after_stats = json_safe(mp.gpu.statistics())
            delta = statistics_delta(before_stats, after_stats)
            record = {
                "rank": rank,
                "run_index": run_index,
                "stage": stage,
                "timestep": int(simulation.timestep()),
                "timestep_delta": int(simulation.timestep()) - before_timestep,
                "meep_time": float(simulation.meep_time()),
                "wall_seconds": wall_seconds,
                "phase_process_seconds_before_reset": time.perf_counter()
                - phase_started,
                "active_backend": str(mp.gpu.active_backend),
                "requested_backend": str(
                    getattr(mp.gpu, "requested_backend", mp.gpu.active_backend)
                ),
                "device_ordinal": int(mp.gpu.selected_device),
                "device_identifier": (
                    str(mp.gpu.selected_device_identifier)
                    if args.expected_backend == "cuda"
                    else None
                ),
                "memory_admission": admission,
                "phase_calls": phase_call_totals(delta),
                "phase_counters": phase_counter_view(delta),
                "statistics_before": before_stats,
                "statistics_after": after_stats,
                "statistics_delta": delta,
            }
            _validate_backend_record(record, args.expected_backend)
            if not structure:
                reference_flux[polarization] = {
                    "monitor1": simulation.get_flux_data(monitors.monitor1),
                    "monitor2": simulation.get_flux_data(monitors.monitor2),
                }
            simulation.reset_meep()
            comm.Barrier()
            records.append(record)
    finally:
        os.chdir(previous_cwd)
    comm.Barrier()
    process_wall_seconds = time.perf_counter() - process_started

    local = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "records": records,
        "runtime_artifacts": runtime_artifacts,
        "runtime_provenance_kind": runtime_provenance_kind,
        "physical_cores": physical_core_affinity(),
        "build_receipt_id": build_receipt["receipt_id"],
        "process_wall_seconds": process_wall_seconds,
        "final_statistics": json_safe(mp.gpu.statistics()),
    }
    gathered = comm.gather(local, root=0)
    comm.Barrier()
    publication_error = None
    if rank == 0:
        try:
            validate_phase_contract(gathered, args.expected_backend, PHASE_CONTRACT)
            phase_wall_seconds = [
                max(
                    float(rank_record["records"][index]["wall_seconds"])
                    for rank_record in gathered
                )
                for index in range(len(PHASE_SPECS))
            ]
            summary = {
                "schema": SCHEMA,
                "measurement_mode": MEASUREMENT_MODE,
                "performance_adaptation": PERFORMANCE_ADAPTATION,
                "fixed_window_meep_time": FIXED_WINDOW_MEEP_TIME,
                "minimum_timesteps_per_phase": MINIMUM_TIMESTEPS_PER_PHASE,
                "archive": absolute_file_record(archive, "AuNP archive"),
                "archive_sha256": AUNP_ARCHIVE_SHA256,
                "engine": file_record(engine_path, output),
                "adapter": file_record(pathlib.Path(__file__), repo),
                "adapter_common": file_record(SCRIPT_DIR / "common.py", repo),
                "expected_backend": args.expected_backend,
                "single_precision": bool(mp.is_single_precision()),
                "meep_version": str(getattr(mp, "__version__", "unknown")),
                "meep_module": str(pathlib.Path(mp.__file__).resolve()),
                "build_receipt": build_receipt,
                "runtime_provenance_kind": runtime_provenance_kind,
                "mpi_size": size,
                "rank_records": gathered,
                "phase_contract": PHASE_CONTRACT,
                "phase_wall_seconds": phase_wall_seconds,
                "fdtd_wall_seconds": sum(phase_wall_seconds),
                "workload_end_to_end_seconds": max(
                    float(rank_record["process_wall_seconds"])
                    for rank_record in gathered
                ),
                "default_config": default_config,
                "effective_config": effective_config,
                "qualification_profile": aunp_qualification_profile(),
                "physical_overrides": dict(AUNP_PHYSICAL_OVERRIDES),
                "execution_policy_overrides": {
                    "required_mpi_ranks": size,
                    "output_dir": str(cfg.output_dir),
                },
                "scientific_output_claims": [],
                "timing_semantics": {
                    "primary": "sum of two TM rank-synchronized Simulation.run maxima",
                    "excluded": [
                        "archive verification and extraction",
                        "simulation construction",
                        "reference flux-data capture and structure loading",
                        "reset_meep",
                        "provenance and summary publication",
                    ],
                },
            }
            atomic_write_json(output / "summary.json", summary)
            atomic_write_json(
                output / "COMPLETE",
                {
                    "schema": COMPLETE_SCHEMA,
                    "summary": file_record(output / "summary.json", output),
                },
            )
            validate_performance_output(
                output,
                expected_backend=args.expected_backend,
                expected_mpi_size=size,
            )
        except Exception as exc:
            publication_error = f"{type(exc).__name__}: {exc}"
    publication_error = comm.bcast(publication_error, root=0)
    if publication_error is not None:
        raise WorkloadError(
            f"AuNP performance evidence publication failed: {publication_error}"
        )
    comm.Barrier()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkloadError as error:
        print(f"AuNP performance workload error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
